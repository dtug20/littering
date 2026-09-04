"""
Máy trạng thái phát hiện vật tĩnh, có cổng xác thực semantic trước cảnh báo.

Khác biệt so với v1: object không còn đến từ YOLO (đa lớp), mà đến từ
BlobTracker (MOG static blob đã qua shape_filters). Vì blob CHỈ tồn tại khi
vật đã đứng yên (MOG không sinh blob cho vật đang di chuyển cùng người), nên
không cần state "CARRIED" tường minh - việc "đang được mang theo" tự động
không bao giờ sinh ra track (đúng bản chất test case 4).

Các trạng thái:
  CONFIRMING        -> blob mới xuất hiện, đang chờ đủ số frame liên tiếp
                        (chống nhiễu MOG chớp nhoáng)
  STATIC_WITH_OWNER -> đã xác nhận đứng yên, chủ nhân (được suy ra từ
                        PersonHistoryBuffer tại thời điểm xác nhận) còn ở gần
  STATIC_NO_OWNER   -> chủ nhân đã rời xa ngưỡng khoảng cách -> đang đếm giờ
  AWAITING_TARGET   -> đã đủ thời gian bỏ quên, chờ model xác nhận đúng loại
  ABANDONED         -> đã vượt ngưỡng thời gian -> đã phát sự kiện cảnh báo
  PENDING_CLAIM     -> blob biến mất VÀ có người đang chồng lấn vị trí cũ
                        -> nghi vấn đang được nhặt lại, chờ xác nhận
  CLAIMED           -> xác nhận đã bị nhặt đi (không tái xuất hiện tại chỗ cũ)
  BACKGROUND        -> vật có sẵn từ trước (baseline) -> không theo dõi cảnh báo
  IGNORED           -> model semantic xác nhận không phải loại mục tiêu

Ánh xạ 5 test case:
  1. Đặt xuống rồi rời khỏi khu vực     -> STATIC_NO_OWNER vượt dwell -> ABANDONED (Cảnh báo)
  2. Đặt xuống rồi quay lại lấy         -> PENDING_CLAIM -> CLAIMED (Không cảnh báo)
  3. Đứng cạnh đồ vật sau khi đặt xuống -> luôn STATIC_WITH_OWNER (Không cảnh báo)
  4. Đi ngang qua với đồ vật trên tay   -> KHÔNG BAO GIỜ sinh blob (vật luôn di
                                            chuyển cùng người) -> không có track
  5. Đồ vật cố định từ trước            -> BACKGROUND ngay khi tạo (baseline)
"""
import time
import uuid
import logging
from dataclasses import dataclass, field


logger = logging.getLogger("aod.rules")


@dataclass
class ObjectTrack:
    object_id: str
    camera_id: str
    bbox: tuple
    state: str = "CONFIRMING"
    owner_track_id: str = None
    related_vehicle_id: str = None
    related_vehicle_class: str = None
    person_dwell_seconds: float = 0.0
    vehicle_dwell_seconds: float = 0.0
    first_seen_at: float = field(default_factory=time.time)
    owner_left_at: float = None
    owner_absent_since: float = None
    last_seen_at: float = field(default_factory=time.time)
    pending_claim_at: float = None
    state_before_pending_claim: str = None
    label: str = "unknown_object"
    label_confidence: float = 0.0
    event_id: str = None
    alerted: bool = False

    def touch(self, bbox):
        self.bbox = bbox
        self.last_seen_at = time.time()


class ObjectStateTracker:
    def __init__(self, rules_cfg: dict):
        self.rules = rules_cfg
        self.tracks: dict[str, ObjectTrack] = {}

    # ------------------------------------------------------------------
    def _nearby(self, obj_bbox, person_boxes, iou_fn) -> bool:
        return any(iou_fn(obj_bbox, pbox) > 0.02 for pbox in person_boxes)

    # ------------------------------------------------------------------
    def update(self, camera_id, blob_results, person_tracks, roi_polygons,
               is_in_baseline, geometry_mod, owner_lookup_fn,
               defer_abandoned_event=False):
        """
        blob_results: list[dict] {object_id, bbox, is_new, consecutive_hits}
                      (đầu ra của BlobTracker, đã qua shape_filters)
        person_tracks: dict[person_track_id] -> bbox (frame hiện tại)
        owner_lookup_fn: callable(location_xy) -> person_id | None, dùng
                      PersonHistoryBuffer để suy ra chủ tại thời điểm xác nhận
        Trả về list events.
        """
        events = []
        now = time.time()
        seen_ids = set()
        person_boxes = list(person_tracks.values())

        for blob in blob_results:
            object_id = blob["object_id"]
            bbox = blob["bbox"]
            if not geometry_mod.box_in_any_polygon(bbox, roi_polygons):
                continue
            seen_ids.add(object_id)

            track = self.tracks.get(object_id)
            if track is None:
                if is_in_baseline:
                    track = ObjectTrack(object_id=object_id, camera_id=camera_id,
                                         bbox=bbox, state="BACKGROUND")
                    self.tracks[object_id] = track
                    continue
                track = ObjectTrack(object_id=object_id, camera_id=camera_id, bbox=bbox)
                self.tracks[object_id] = track

            track.touch(bbox)

            # Detector-first tracks carry a semantic label and confidence.
            # Preserve them for both live bbox payloads and lifecycle events.
            if blob.get("label"):
                track.label = str(blob["label"])
                track.label_confidence = float(blob.get("label_confidence", 0.0))

            # Association hints originate at the person-departure frame. Store
            # them immediately because subsequent static-blob frames may no
            # longer carry the original actor/vehicle metadata.
            owner_hint = blob.get("owner_track_id_hint")
            if track.owner_track_id is None and owner_hint is not None:
                track.owner_track_id = owner_hint
            vehicle_hint = blob.get("related_vehicle_id_hint")
            if track.related_vehicle_id is None and vehicle_hint is not None:
                track.related_vehicle_id = vehicle_hint
                track.related_vehicle_class = blob.get("related_vehicle_class_hint")
            track.person_dwell_seconds = max(
                track.person_dwell_seconds,
                float(blob.get("person_dwell_seconds", 0.0)),
            )
            track.vehicle_dwell_seconds = max(
                track.vehicle_dwell_seconds,
                float(blob.get("vehicle_dwell_seconds", 0.0)),
            )

            if track.state in ("BACKGROUND", "IGNORED"):
                continue

            if track.state == "PENDING_CLAIM":
                # blob tái xuất hiện tại (gần) vị trí cũ -> chỉ là occlusion
                # tạm thời (người cúi xuống xem rồi để lại), KHÔNG phải nhặt đi
                track.state = track.state_before_pending_claim or "STATIC_NO_OWNER"
                track.pending_claim_at = None
                track.state_before_pending_claim = None

            if track.state == "CONFIRMING":
                if blob["consecutive_hits"] >= self.rules.get("static_confirm_frames", 5):
                    owner_id = track.owner_track_id or owner_lookup_fn(
                        geometry_mod.centroid(bbox)
                    )
                    track.owner_track_id = owner_id
                    track.state = (
                        "STATIC_WITH_OWNER" if owner_id is not None
                        else "STATIC_NO_OWNER"
                    )
                else:
                    continue

            if track.state in (
                "STATIC_WITH_OWNER", "STATIC_NO_OWNER", "AWAITING_TARGET", "ABANDONED"
            ):
                # Một alert đã phát chỉ kết thúc khi vật biến mất/được nhặt.
                # Người khác đi ngang/đứng gần không được reset để bắn lặp.
                if track.state == "ABANDONED":
                    continue

                owner_bbox = person_tracks.get(track.owner_track_id)
                owner_nearby = False
                
                # Áp dụng Hysteresis (biên độ trễ) để chống nháy trạng thái
                base_threshold = self.rules["owner_distance_threshold_px"]
                threshold = base_threshold * 1.5 if track.state == "STATIC_WITH_OWNER" else base_threshold

                if owner_bbox is not None:
                    d = geometry_mod.euclidean_distance(
                        geometry_mod.centroid(bbox), geometry_mod.centroid(owner_bbox)
                    )
                    if d <= threshold:
                        owner_nearby = True
                        track.owner_last_seen_near_at = now

                actually_nearby = owner_nearby
                if actually_nearby:
                    track.owner_absent_since = None
                elif track.owner_absent_since is None:
                    track.owner_absent_since = now

                # Grace period: Nếu owner bị mất dấu tạm thời (VD: tracker rớt 1-2 frame) 
                # hoặc vừa mới bước ra khỏi vùng, cho thêm 2 giây ân hạn trước khi đổi state
                grace_period = self.rules.get("owner_lost_grace_seconds", 2.0)
                if not owner_nearby and hasattr(track, 'owner_last_seen_near_at'):
                    if now - track.owner_last_seen_near_at < grace_period:
                        owner_nearby = True

                if owner_nearby:
                    if track.state != "STATIC_WITH_OWNER":
                        logger.debug(
                            "[AOD] object=%s owner=%s is nearby",
                            object_id, track.owner_track_id,
                        )
                    track.state = "STATIC_WITH_OWNER"
                    if actually_nearby:
                        track.owner_left_at = None
                else:
                    if track.state == "STATIC_WITH_OWNER":
                        track.state = "STATIC_NO_OWNER"
                        track.owner_left_at = track.owner_absent_since or now
                        logger.debug(
                            "[AOD] object=%s owner=%s left the object",
                            object_id, track.owner_track_id,
                        )
                    elif track.owner_left_at is None:
                        track.owner_left_at = track.owner_absent_since or now

                    dwell = now - track.owner_left_at
                    if dwell >= self.rules["abandonment_dwell_seconds"] and not track.alerted:
                        if defer_abandoned_event:
                            track.state = "AWAITING_TARGET"
                            # Event nội bộ: rule engine chỉ chuyển thành cảnh báo
                            # sau khi model semantic xác nhận đúng loại mục tiêu.
                            events.append(self._build_event("target_verification_requested", track))
                        else:
                            event = self.confirm_abandoned(object_id)
                            if event is not None:
                                events.append(event)

        # --- xử lý các track không xuất hiện trong blob_results frame này ---
        for object_id, track in list(self.tracks.items()):
            if object_id in seen_ids or track.camera_id != camera_id:
                continue
            if track.state in ("BACKGROUND", "CONFIRMING", "IGNORED"):
                # blob nhiễu biến mất trước khi kịp xác nhận -> âm thầm bỏ qua
                if (
                    track.state in ("CONFIRMING", "IGNORED")
                    and now - track.last_seen_at > self.rules["max_occlusion_gap_seconds"]
                ):
                    del self.tracks[object_id]
                continue

            overlapped_by_person = self._nearby(track.bbox, person_boxes, geometry_mod.iou)

            if track.state != "PENDING_CLAIM" and overlapped_by_person:
                track.state_before_pending_claim = track.state
                track.state = "PENDING_CLAIM"
                track.pending_claim_at = now
                continue

            if track.state == "PENDING_CLAIM":
                gap = now - track.pending_claim_at
                if gap >= self.rules.get("pickup_confirm_gap_seconds", 3.0):
                    track.state = "CLAIMED"
                    # Lifecycle event chỉ hợp lệ nếu object_abandoned đã từng
                    # được publish; vật được nhặt trước ngưỡng phải im lặng.
                    if track.alerted:
                        events.append(self._build_event("object_claimed", track))
                    del self.tracks[object_id]
                continue

            # occlusion thường (đám đông che khuất, không chạm vào vật)
            gap = now - track.last_seen_at
            if gap > self.rules["max_occlusion_gap_seconds"]:
                if track.alerted:
                    events.append(self._build_event("object_removed", track))
                del self.tracks[object_id]

        return events

    def confirm_abandoned(self, object_id: str, label="object", confidence=0.0) -> dict | None:
        """Commit a public alert after semantic verification succeeds."""
        track = self.tracks.get(object_id)
        if track is None or track.alerted:
            return None
        track.alerted = True
        track.state = "ABANDONED"
        track.label = label
        track.label_confidence = float(confidence)
        track.event_id = str(uuid.uuid4())
        return self._build_event("object_abandoned", track)

    def reject_target(self, object_id: str, label="non_target", confidence=0.0):
        """Keep a rejected static region silent until it disappears."""
        track = self.tracks.get(object_id)
        if track is None or track.alerted:
            return
        track.state = "IGNORED"
        track.label = label
        track.label_confidence = float(confidence)

    def clear_camera(self, camera_id: str):
        """Clear object state when a camera ROI is replaced/removed."""
        self.tracks = {
            object_id: track
            for object_id, track in self.tracks.items()
            if track.camera_id != camera_id
        }

    def _build_event(self, event_type, track: ObjectTrack) -> dict:
        incident_type = (
            "illegal_dumping"
            if track.owner_track_id is not None and track.related_vehicle_id is not None
            else "trash_accumulation"
        )
        return {
            "event_type": event_type,
            # Keep the existing event_type contract for downstream systems,
            # while exposing the business-level outcome explicitly.
            "incident_type": incident_type,
            "camera_id": track.camera_id,
            "object_id": track.object_id,
            "owner_track_id": track.owner_track_id,
            "related_vehicle_id": track.related_vehicle_id,
            "related_vehicle_class": track.related_vehicle_class,
            "person_dwell_seconds": track.person_dwell_seconds,
            "vehicle_dwell_seconds": track.vehicle_dwell_seconds,
            "bbox": track.bbox,
            "label": track.label,
            "label_confidence": track.label_confidence,
            "first_seen_static_at": track.first_seen_at,
            "triggered_at": time.time(),
            "event_id": track.event_id or str(uuid.uuid4()),
        }
