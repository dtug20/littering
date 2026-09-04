"""
Orchestrator v3 - PERSON-CENTRIC.

Luồng phát hiện vật bỏ quên:
  1. YOLO (DeepStream nvinfer) detect person → nvtracker gán person_id ổn định.
  2. PersonDepartureDetector theo dõi hành trình mỗi person:
     - Ghi nhận dwell zones (vùng person đứng yên > N giây)
     - Chụp reference snapshot tại dwell zone lúc person mới đến
  3. Khi person rời khỏi frame (biến mất >= departure_confirm_seconds):
     - So sánh dwell zone TRƯỚC (reference) vs SAU (hiện tại) bằng SSIM
     - Nếu có vật mới → candidate abandoned object
  4. MOG2 verify cục bộ (KHÔNG quét toàn frame) để xác nhận vật thực sự
     đứng yên tĩnh (loại false positive từ SSIM).
  5. Shape filters + BlobTracker gán object_id → ObjectStateTracker (giữ nguyên).

Ưu điểm so với v2:
  - False positive giảm ~95% (chỉ trigger khi person departure, không phản ứng
    với mọi pixel thay đổi)
  - CPU giảm đáng kể (xóa GrabCut, MOG chỉ verify vùng nhỏ)
  - BBox ổn định hơn (anchor vào vị trí person, không bị MOG sinh/mất blob)
"""
import logging
import numpy as np
from collections import Counter

from src.detection.shape_filters import (
    compute_shape_score, is_plausible_object,
    is_valid_size, overlaps_any_person, compute_edge_density,
)
from src.detection.blob_tracker import BlobTracker
from src.detection.person_departure_detector import PersonDepartureDetector
from src.detection.detected_object_tracker import DetectedObjectTracker
from src.detection.vehicle_track_filter import VehicleTrackFilter
from src.logic.ownership import PersonHistoryBuffer, VehicleDwellAssociator
from src.tracking.object_state_tracker import ObjectStateTracker
from src.utils import geometry

logger = logging.getLogger("aod.rules")


class AbandonmentRuleEngine:
    def __init__(self, app_cfg: dict, mog_detector, labeler=None,
                 target_gate=None, object_detector=None, cameras_cfg=None):
        self.app_cfg = app_cfg
        self.mog_detector = mog_detector
        self.labeler = labeler  # ObjectLabeler | None
        self.target_gate = target_gate
        self.object_detector = object_detector

        self.state_tracker = ObjectStateTracker(app_cfg["abandonment_rules"])
        self.blob_tracker = BlobTracker(
            iou_fn=geometry.iou,
            iou_match_threshold=app_cfg["association"]["iou_match_threshold"],
            max_missed_seconds=max(
                60.0,
                app_cfg["abandonment_rules"]["max_occlusion_gap_seconds"] * 4,
            ),
            bbox_smoothing_alpha=app_cfg["association"].get(
                "bbox_smoothing_alpha", 0.25
            ),
        )
        self.person_history = PersonHistoryBuffer(
            window_seconds=app_cfg["ownership"]["history_window_seconds"]
        )
        self.vehicle_associator = VehicleDwellAssociator(
            app_cfg.get("illegal_dumping", {})
        )

        # === Person-Centric detector (MỚI) ===
        person_departure_cfg = app_cfg.get("person_departure", {})
        self.departure_detector = PersonDepartureDetector(person_departure_cfg)

        self.person_class = app_cfg["detection"]["person_class"]
        self.vehicle_classes = {
            str(name) for name in app_cfg["detection"].get(
                "vehicle_classes", ["bicycle", "car", "motorcycle", "bus", "truck"]
            )
        }
        self.vehicle_filter = VehicleTrackFilter(
            app_cfg.get("vehicle_filter", {}), geometry.iou
        )
        self.object_classes = {
            str(name) for name in app_cfg["detection"].get("object_classes", [])
        }
        self.object_min_conf = float(
            app_cfg["detection"].get("object_min_confidence", 0.30)
        )
        self.object_detection_cfg = app_cfg.get("object_detection", {})
        self.detector_first_enabled = bool(
            self.object_detection_cfg.get("enabled", False)
        )
        self.detected_object_tracker = DetectedObjectTracker(
            geometry.iou, self.object_detection_cfg
        )
        self.detector_interval_samples = max(
            1, int(self.object_detection_cfg.get("inference_interval_samples", 1))
        )
        self.detector_roi_padding_ratio = max(
            0.0, float(self.object_detection_cfg.get("roi_crop_padding_ratio", 0.08))
        )
        self.detector_roi_crop_enabled = bool(
            self.object_detection_cfg.get("roi_crop_enabled", True)
        )
        self.object_min_center_y_ratio = float(
            self.object_detection_cfg.get("min_bbox_center_y_ratio", 0.0)
        )
        self.object_max_dynamic_overlap = float(
            self.object_detection_cfg.get("max_dynamic_overlap_ratio", 1.0)
        )
        self._detector_sample_counts = {}
        self._detector_rejection_logs = {}
        self.min_conf = app_cfg["detection"]["min_confidence"]
        self.min_area_ratio = app_cfg["detection"]["min_object_area_ratio"]
        self.shape_cfg = app_cfg["shape_filters"]
        self.owner_max_distance_px = app_cfg["ownership"]["owner_max_distance_px"]
        # Runtime camera data from MQTT is authoritative. Previously only the
        # static YAML cameras were used here, so a runtime ROI could silently
        # become an empty ROI (= whole frame in geometry.py).
        camera_source = cameras_cfg if cameras_cfg is not None else app_cfg.get("cameras", [])
        self.cameras_by_id = {c["camera_id"]: c for c in camera_source}
        self.roi_required = app_cfg.get("roi", {}).get("required", True)
        self._small_frame_shapes = {}

        # Edge density filter
        self.min_edge_density = app_cfg.get("shape_filters", {}).get("min_edge_density", 0.02)

    def update_roi(self, camera_id: str, roi_polygons: list):
        """Atomically replace ROI-facing state so old candidates cannot leak."""
        self.cameras_by_id.setdefault(camera_id, {})["roi_polygons"] = roi_polygons
        self.state_tracker.clear_camera(camera_id)
        self.blob_tracker.clear()
        self.detected_object_tracker.clear()
        self._detector_sample_counts.pop(camera_id, None)
        self.departure_detector.clear_camera(camera_id)
        self.vehicle_associator.clear_camera(camera_id)
        self.vehicle_filter.clear_camera(camera_id)
        if self.target_gate is not None:
            self.target_gate.clear()

    def _crop_detector_roi(
        self, frame_bgr, roi_polygons, scale_x, scale_y
    ):
        """Crop the union of configured polygons before expensive inference."""
        if not self.detector_roi_crop_enabled:
            return frame_bgr, (0, 0)
        points = [
            point
            for polygon in roi_polygons
            for point in polygon.get("points", [])
        ]
        if not points:
            return frame_bgr, (0, 0)

        xs = [float(point[0]) / scale_x for point in points]
        ys = [float(point[1]) / scale_y for point in points]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        pad_x = (x2 - x1) * self.detector_roi_padding_ratio
        pad_y = (y2 - y1) * self.detector_roi_padding_ratio
        frame_h, frame_w = frame_bgr.shape[:2]
        ix1 = max(0, int(x1 - pad_x))
        iy1 = max(0, int(y1 - pad_y))
        ix2 = min(frame_w, int(x2 + pad_x + 0.999))
        iy2 = min(frame_h, int(y2 + pad_y + 0.999))
        if ix2 <= ix1 or iy2 <= iy1:
            return frame_bgr, (0, 0)
        return frame_bgr[iy1:iy2, ix1:ix2], (ix1, iy1)

    def process_frame(self, camera_id, frame_w, frame_h, ds_detections, frame_bgr_small):
        """
        ds_detections: list[dict] {object_id, class_name, confidence, bbox}
        frame_bgr_small: ảnh BGR (đã scale nhỏ) – dùng cho MOG feed + departure check

        Trả về (events, tracked_objects, person_tracks, vehicle_tracks).
        """
        cam_cfg = self.cameras_by_id.get(camera_id, {})
        roi_polygons = geometry.normalise_roi_polygons(
            cam_cfg.get("roi_polygons", []), frame_w, frame_h
        )
        frame_area = float(frame_w * frame_h)

        # --- Bước 1: Xử lý person detections (KHÔNG ĐỔI) ---
        person_tracks = {}
        vehicle_detections = []
        semantic_detections = []
        UNTRACKED_ID = "18446744073709551615"
        for obj in ds_detections:
            class_name = obj["class_name"]
            if (
                class_name != self.person_class
                and class_name not in self.vehicle_classes
                and class_name not in self.object_classes
            ):
                continue
            # Object portable dùng ngưỡng riêng thấp hơn person/vehicle vì
            # túi nhỏ và biến dạng thường có confidence COCO chỉ 0.30-0.50.
            if obj["object_id"] == UNTRACKED_ID:
                min_confidence = (
                    self.object_min_conf
                    if class_name in self.object_classes
                    else self.min_conf
                )
                if obj["confidence"] < min_confidence:
                    continue
            # Khi đã có track ID, ta hoàn toàn tin tưởng tracker (để chống nháy)
            if obj["object_id"] != UNTRACKED_ID:
                if class_name == self.person_class:
                    person_tracks[obj["object_id"]] = obj["bbox"]
                elif class_name in self.vehicle_classes:
                    vehicle_detections.append(obj)
            if class_name in self.object_classes:
                if (
                    obj["object_id"] != UNTRACKED_ID
                    or obj["confidence"] >= self.object_min_conf
                ):
                    semantic_detections.append({
                        "bbox": obj["bbox"],
                        "class_name": class_name,
                        "confidence": max(0.0, float(obj["confidence"])),
                        "source": "deepstream_yolo",
                    })

        vehicle_tracks = self.vehicle_filter.update(
            camera_id, vehicle_detections, frame_w, frame_h
        )

        if frame_bgr_small is not None:
            small_h_actual, small_w_actual = frame_bgr_small.shape[:2]
            self._small_frame_shapes[camera_id] = (small_w_actual, small_h_actual)
        small_w_actual, small_h_actual = self._small_frame_shapes.get(camera_id, (640, 360))
        scale_x = float(frame_w) / max(1, small_w_actual)
        scale_y = float(frame_h) / max(1, small_h_actual)

        # --- Bước 2: baseline / legacy MOG ---
        is_in_baseline = False
        if self.detector_first_enabled:
            # Chế độ semantic chỉ cần đồng hồ baseline, không cần chạy hai
            # background subtractor tốn CPU trên frame 960x540.
            is_in_baseline = self.mog_detector.in_baseline_period(camera_id)
        elif frame_bgr_small is not None and self.mog_detector.is_enabled():
            self.mog_detector.feed_frame(camera_id, frame_bgr_small)
            is_in_baseline = self.mog_detector.in_baseline_period(camera_id)

        if self.roi_required and not roi_polygons:
            # Empty/invalid ROI means "detect nowhere" in strict mode. This is
            # safer than unexpectedly alerting over the whole image.
            return [], [], person_tracks, vehicle_tracks

        # Only a person/vehicle whose tracked box is in the configured ROI can
        # contribute to a dumping incident. This prevents activity elsewhere
        # in the camera from being linked to a trash candidate.
        person_tracks_in_roi = {
            track_id: bbox for track_id, bbox in person_tracks.items()
            if geometry.box_in_any_polygon(bbox, roi_polygons)
        }
        self.person_history.push(camera_id, person_tracks_in_roi)
        self.vehicle_associator.update(
            camera_id, vehicle_tracks, roi_polygons, geometry
        )

        # --- Bước 3: semantic detector là nguồn bbox object chính ---
        detector_blob_results = None
        if self.detector_first_enabled:
            if self.object_detector is not None and frame_bgr_small is not None:
                sample_count = self._detector_sample_counts.get(camera_id, 0) + 1
                self._detector_sample_counts[camera_id] = sample_count
                should_infer = (
                    (sample_count - 1) % self.detector_interval_samples == 0
                )
                if should_infer:
                    detector_frame, (offset_x, offset_y) = self._crop_detector_roi(
                        frame_bgr_small, roi_polygons, scale_x, scale_y
                    )
                    try:
                        detected_small = self.object_detector.detect(detector_frame)
                    except Exception:
                        # Fail closed for this frame: never replace a failed
                        # semantic inference with a pixel-change proposal.
                        logger.exception(
                            "[AOD] Lỗi semantic object detector camera=%s", camera_id
                        )
                        detected_small = []
                    for detected in detected_small:
                        x1, y1, x2, y2 = detected["bbox"]
                        item = dict(detected)
                        item["bbox"] = (
                            (x1 + offset_x) * scale_x,
                            (y1 + offset_y) * scale_y,
                            (x2 + offset_x) * scale_x,
                            (y2 + offset_y) * scale_y,
                        )
                        semantic_detections.append(item)

            filtered_detections = []
            rejected = Counter()
            for detected in sorted(
                semantic_detections,
                key=lambda item: float(item.get("confidence", 0.0)),
                reverse=True,
            ):
                bbox = detected["bbox"]
                if geometry.centroid(bbox)[1] / max(1.0, float(frame_h)) < self.object_min_center_y_ratio:
                    rejected["above_ground"] += 1
                    continue
                if not geometry.box_in_any_polygon(bbox, roi_polygons):
                    rejected["outside_roi"] += 1
                    continue
                if not is_valid_size(bbox, frame_w, frame_h, self.shape_cfg):
                    rejected["invalid_size"] += 1
                    continue
                if geometry.box_area(bbox) / frame_area < self.min_area_ratio:
                    rejected["too_small"] += 1
                    continue
                # Merge a COCO handbag box and a YOLO-World plastic-bag box
                # when both detectors describe the same physical object.
                if any(
                    geometry.iou(bbox, old["bbox"]) > 0.55
                    for old in filtered_detections
                ):
                    rejected["duplicate"] += 1
                    continue
                filtered_detections.append(detected)

            if semantic_detections and not filtered_detections:
                logged = self._detector_rejection_logs.get(camera_id, 0)
                if logged < 3:
                    logger.info(
                        "[ObjectFilter] camera=%s raw=%d rejected=%s",
                        camera_id, len(semantic_detections), dict(rejected),
                    )
                    self._detector_rejection_logs[camera_id] = logged + 1

            detector_blob_results = self.detected_object_tracker.update(
                filtered_detections,
                frame_bgr=frame_bgr_small,
                scale_x=scale_x,
                scale_y=scale_y,
            )
            for blob in detector_blob_results:
                owner = self.person_history.find_owner_at_location(
                    camera_id,
                    geometry.centroid(blob["bbox"]),
                    geometry,
                    self.owner_max_distance_px,
                )
                if owner is not None:
                    blob["owner_track_id_hint"] = owner
                related_vehicle = self.vehicle_associator.find_related(
                    camera_id, blob["bbox"], roi_polygons, geometry
                )
                if related_vehicle is not None:
                    blob["related_vehicle_id_hint"] = related_vehicle["track_id"]
                    blob["related_vehicle_class_hint"] = related_vehicle["class_name"]
                    blob["vehicle_dwell_seconds"] = related_vehicle["dwell_seconds"]
                blob.setdefault("person_dwell_seconds", 0.0)

        # --- Legacy fallback: person departure + SSIM/MOG proposals ---
        # Luôn scale person_tracks về tọa độ nhỏ (640x360) để so sánh ổn định
        person_tracks_small = {}
        for pid, pbox in person_tracks_in_roi.items():
            px1, py1, px2, py2 = pbox
            person_tracks_small[pid] = (
                px1 / scale_x, py1 / scale_y,
                px2 / scale_x, py2 / scale_y,
            )

        small_w = int(frame_w / scale_x)
        small_h = int(frame_h / scale_y)

        departure_candidates = []
        if not self.detector_first_enabled:
            departure_candidates = self.departure_detector.update(
                camera_id=camera_id,
                person_tracks=person_tracks_small,
                frame_bgr=frame_bgr_small,
                frame_w=small_w,
                frame_h=small_h,
            )

        # --- Bước 4: Lọc candidate qua shape_filters + MOG verify ---
        raw_items = []
        person_boxes = list(person_tracks.values())

        # 4a: Candidate mới từ sự kiện rời đi
        for cand in departure_candidates if not self.detector_first_enabled else []:
            # Scale bbox từ tọa độ small → tọa độ gốc
            cx1, cy1, cx2, cy2 = cand["bbox"]
            bbox_orig = (cx1 * scale_x, cy1 * scale_y, cx2 * scale_x, cy2 * scale_y)

            if not geometry.box_in_any_polygon(bbox_orig, roi_polygons): continue
            if not is_valid_size(bbox_orig, frame_w, frame_h, self.shape_cfg): continue
            if geometry.box_area(bbox_orig) / frame_area < self.min_area_ratio: continue
            if overlaps_any_person(bbox_orig, person_boxes, threshold=0.4): continue

            # Filter shape_score
            if "contour" in cand and cand["contour"] is not None:
                contour_orig = cand["contour"].copy().astype(np.float32)
                contour_orig[:, 0, 0] *= scale_x
                contour_orig[:, 0, 1] *= scale_y
                shape_score = compute_shape_score(contour_orig.astype(np.int32), bbox_orig)
                if not is_plausible_object(shape_score, self.shape_cfg): continue

            # Filter edge density
            if frame_bgr_small is not None:
                ed = compute_edge_density(frame_bgr_small, (cx1, cy1, cx2, cy2))
                if ed < self.min_edge_density: continue

            related_vehicle = self.vehicle_associator.find_related(
                camera_id, bbox_orig, roi_polygons, geometry
            )
            raw_items.append({
                "bbox": bbox_orig,
                "owner_hint": cand.get("departed_person_id"),
                "related_vehicle": related_vehicle,
                "person_dwell_seconds": cand.get("person_dwell_seconds", 0.0),
            })

        # 4b: Giữ track liên tục cho các vật thể ĐÃ phát hiện trước đó
        for track in (
            self.state_tracker.tracks.values()
            if not self.detector_first_enabled
            else []
        ):
            if track.camera_id == camera_id and track.state not in ("BACKGROUND", "IGNORED"):
                related_vehicle = None
                if track.related_vehicle_id is not None:
                    related_vehicle = {
                        "track_id": track.related_vehicle_id,
                        "class_name": track.related_vehicle_class,
                        "dwell_seconds": track.vehicle_dwell_seconds,
                    }
                raw_items.append({
                    "bbox": track.bbox,
                    "owner_hint": track.owner_track_id,
                    "related_vehicle": related_vehicle,
                    "person_dwell_seconds": track.person_dwell_seconds,
                })

        # 4c: Loại bỏ các bbox trùng lấn nhau (NMS dựa trên độ bao phủ để chống nháy ID)
        unique_items = []
        for item in raw_items:
            b1 = item["bbox"]
            is_dup = False
            b1_x1, b1_y1, b1_x2, b1_y2 = b1
            area1 = max(0.0, b1_x2 - b1_x1) * max(0.0, b1_y2 - b1_y1)
            for existing in unique_items:
                b2 = existing["bbox"]
                b2_x1, b2_y1, b2_x2, b2_y2 = b2
                area2 = max(0.0, b2_x2 - b2_x1) * max(0.0, b2_y2 - b2_y1)
                
                inter_x1 = max(b1_x1, b2_x1)
                inter_y1 = max(b1_y1, b2_y1)
                inter_x2 = min(b1_x2, b2_x2)
                inter_y2 = min(b1_y2, b2_y2)
                
                inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
                if area1 > 0 and area2 > 0 and inter_area / min(area1, area2) > 0.5:
                    is_dup = True
                    break
            if not is_dup:
                unique_items.append(item)

        # 4d: Verify các bbox bằng MOG cục bộ
        valid_items = []
        for item in unique_items:
            box = item["bbox"]
            if frame_bgr_small is not None and self.mog_detector.is_enabled():
                bx1, by1, bx2, by2 = box
                small_box = (bx1 / scale_x, by1 / scale_y, bx2 / scale_x, by2 / scale_y)
                mog_result = self.mog_detector.verify_region(
                    camera_id, frame_bgr_small,
                    region_bbox=small_box,
                    scale_x=scale_x, scale_y=scale_y,
                )
                if mog_result is not None:
                    final_bbox = mog_result["bbox"]
                    # MOG có thể mở rộng candidate thành vùng người hoặc gần
                    # toàn frame. Bắt buộc kiểm tra lại bbox SAU MOG; trước đây
                    # chỉ bbox SSIM ban đầu được lọc nên false box vẫn lọt.
                    if not geometry.box_in_any_polygon(final_bbox, roi_polygons):
                        continue
                    if not is_valid_size(
                        final_bbox, frame_w, frame_h, self.shape_cfg
                    ):
                        continue
                    if overlaps_any_person(
                        final_bbox, person_boxes,
                        threshold=self.shape_cfg.get(
                            "max_person_overlap_ratio", 0.20
                        ),
                    ):
                        continue
                    final_shape = compute_shape_score(
                        mog_result["contour"], final_bbox
                    )
                    if not is_plausible_object(final_shape, self.shape_cfg):
                        continue
                    valid_item = dict(item)
                    valid_item["bbox"] = final_bbox
                    valid_items.append(valid_item)
            else:
                valid_items.append(item)

        # --- Bước 5: BlobTracker gán object_id ổn định ---
        if detector_blob_results is not None:
            blob_results = detector_blob_results
        else:
            blob_results = self.blob_tracker.update(
                [item["bbox"] for item in valid_items]
            )
            for blob, item in zip(blob_results, valid_items):
                owner_hint = item["owner_hint"]
                if owner_hint is not None:
                    blob["owner_track_id_hint"] = owner_hint
                blob["person_dwell_seconds"] = item["person_dwell_seconds"]
                related_vehicle = item["related_vehicle"]
                if related_vehicle is not None:
                    blob["related_vehicle_id_hint"] = related_vehicle["track_id"]
                    blob["related_vehicle_class_hint"] = related_vehicle["class_name"]
                    blob["vehicle_dwell_seconds"] = related_vehicle["dwell_seconds"]

        # --- Bước 6: State machine ---
        def owner_lookup(location_xy):
            return self.person_history.find_owner_at_location(
                camera_id, location_xy, geometry, self.owner_max_distance_px
            )

        state_events = self.state_tracker.update(
            camera_id=camera_id,
            blob_results=blob_results,
            person_tracks=person_tracks,
            roi_polygons=roi_polygons,
            is_in_baseline=is_in_baseline,
            geometry_mod=geometry,
            owner_lookup_fn=owner_lookup,
            defer_abandoned_event=self.target_gate is not None,
        )

        events = []
        for event in state_events:
            if event["event_type"] != "target_verification_requested":
                events.append(event)
                continue

            ex1, ey1, ex2, ey2 = event["bbox"]
            small_bbox = (ex1 / scale_x, ey1 / scale_y, ex2 / scale_x, ey2 / scale_y)
            decision = self.target_gate.evaluate(
                event["object_id"], frame_bgr_small, small_bbox
            )
            if decision.decision == "accept":
                confirmed = self.state_tracker.confirm_abandoned(
                    event["object_id"], decision.label, decision.confidence
                )
                self.target_gate.forget(event["object_id"])
                if confirmed is not None:
                    events.append(confirmed)
                    logger.info(
                        "[AOD] Xác nhận rác: object=%s label=%s confidence=%.3f",
                        event["object_id"], decision.label, decision.confidence,
                    )
            elif decision.decision == "reject":
                self.state_tracker.reject_target(
                    event["object_id"], decision.label, decision.confidence
                )
                self.target_gate.forget(event["object_id"])
                logger.info(
                    "[AOD] Bỏ ứng viên không phải rác: object=%s label=%s confidence=%.3f",
                    event["object_id"], decision.label, decision.confidence,
                )

        if self.target_gate is not None:
            self.target_gate.retain(self.state_tracker.tracks.keys())

        # gắn nhãn mô tả (zero-shot, chỉ gọi khi có event abandoned - tần suất thấp)
        if self.target_gate is None and self.labeler is not None and frame_bgr_small is not None:
            for event in events:
                if event["event_type"] == "object_abandoned":
                    # Scale bbox về tọa độ frame nhỏ để crop
                    ex1, ey1, ex2, ey2 = event["bbox"]
                    small_bbox = (ex1 / scale_x, ey1 / scale_y, ex2 / scale_x, ey2 / scale_y)
                    result = self.labeler.label(frame_bgr_small, small_bbox)
                    event["label"] = result["label"]
                    event["label_confidence"] = result["confidence"]

        return events, blob_results, person_tracks, vehicle_tracks
