"""
PersonDepartureDetector – Trung tâm của kiến trúc Person-Centric.

Theo dõi hành trình từng person qua các frame, và khi person rời khỏi
khung hình (hoặc tracker drop ID), thực hiện:
  1. Xác định "dwell zones" – vùng person đứng yên lâu nhất
  2. So sánh ảnh crop tại dwell zone TRƯỚC (reference) vs SAU (hiện tại)
  3. Nếu có vật mới xuất hiện → trả về candidate abandoned object

Thiết kế chống false positive:
  - Person phải biến mất >= departure_confirm_seconds (mặc định 3s) mới
    trigger kiểm tra → tránh occlude tạm thời bởi đám đông.
  - Chỉ kiểm tra dwell zones (person đứng yên > min_dwell_seconds) →
    loại trường hợp "đi ngang qua" không đặt gì.
  - Reference snapshot chụp sớm (lúc person mới đến vùng dwell) nên ít
    bị ảnh hưởng bởi thay đổi ánh sáng dần dần.
"""
import time
import logging
from dataclasses import dataclass, field
from collections import defaultdict

import cv2
import numpy as np

from src.detection.region_comparator import compare_regions
from src.utils.geometry import centroid, euclidean_distance

logger = logging.getLogger("aod.person_departure")


@dataclass
class _DwellZone:
    """Vùng person đứng yên đủ lâu – nơi có khả năng đặt đồ."""
    center: tuple  # (cx, cy)
    bbox: tuple  # (x1, y1, x2, y2) vùng expanded quanh person
    entered_at: float  # thời điểm bắt đầu đứng yên
    reference_snapshot: np.ndarray = None  # ảnh crop BGR lúc đầu (trước khi đặt đồ)
    last_person_bbox: tuple = None  # bbox cuối cùng của person tại zone này


@dataclass
class _TrackedPerson:
    """Thông tin theo dõi cho mỗi person_id."""
    person_id: str
    camera_id: str
    first_seen_at: float = field(default_factory=time.time)
    last_seen_at: float = field(default_factory=time.time)
    last_bbox: tuple = None
    # Lịch sử vị trí gần đây (dùng để tính dwell)
    position_history: list = field(default_factory=list)  # [(timestamp, bbox)]
    # Các vùng đứng yên đủ lâu
    dwell_zones: list = field(default_factory=list)  # [_DwellZone]
    # Đã xử lý departure chưa
    departure_processed: bool = False
    # Ứng viên đứng yên đang hình thành. Reference phải được chụp ngay khi
    # người bắt đầu dừng, không chờ hết min_dwell_seconds (lúc đó rác có thể
    # đã được đặt xuống và nằm sẵn trong ảnh "trước").
    stationary_anchor: tuple = None
    stationary_since: float = None
    stationary_zone_bbox: tuple = None
    stationary_reference: np.ndarray = None


class PersonDepartureDetector:
    """Phát hiện khi person rời đi và kiểm tra vật bỏ lại."""

    def __init__(self, cfg: dict):
        self.departure_confirm_seconds = cfg.get("departure_confirm_seconds", 3.0)
        self.region_padding_px = cfg.get("region_padding_px", 50)
        self.ssim_change_threshold = cfg.get("ssim_change_threshold", 0.85)
        self.min_dwell_seconds = cfg.get("min_dwell_seconds", 2.0)
        self.max_reference_snapshots = cfg.get("max_reference_snapshots", 3)
        # Khoảng cách tối đa (px) giữa 2 vị trí liên tiếp để coi person đứng yên
        self.stationary_distance_px = cfg.get("stationary_distance_px", 40.0)
        # Khoảng cách tối đa để match dwell zone cũ
        self.dwell_zone_match_distance = cfg.get("dwell_zone_match_distance", 60.0)
        self.min_contour_area_px = cfg.get("min_contour_area_px", 300)

        self._persons: dict[str, _TrackedPerson] = {}  # person_id -> _TrackedPerson
        # Lưu "person đã mất tích" (chưa qua departure_confirm_seconds)
        self._missing_since: dict[str, float] = {}  # person_id -> timestamp mất tích

    def clear_camera(self, camera_id: str):
        """Bỏ trajectory/reference cũ khi ROI của camera thay đổi."""
        removed_ids = [
            person_id
            for person_id, person in self._persons.items()
            if person.camera_id == camera_id
        ]
        for person_id in removed_ids:
            self._persons.pop(person_id, None)
            self._missing_since.pop(person_id, None)

    def update(
        self,
        camera_id: str,
        person_tracks: dict,
        frame_bgr: np.ndarray | None,
        frame_w: int,
        frame_h: int,
    ) -> list[dict]:
        """Cập nhật mỗi frame. Trả về list abandoned candidates khi có person departure.

        Parameters
        ----------
        person_tracks : dict[person_id] -> bbox (x1, y1, x2, y2)
        frame_bgr : ảnh BGR (có thể None nếu không pull được frame)

        Returns
        -------
        list[dict] mỗi dict có: {bbox, contour, ssim_score, departed_person_id,
                                  dwell_zone_bbox}
        """
        now = time.time()
        seen_ids = set(person_tracks.keys())
        candidates = []

        # ----- Bước 1: Cập nhật person đang hiện diện -----
        for pid, bbox in person_tracks.items():
            person = self._persons.get(pid)
            if person is None:
                person = _TrackedPerson(
                    person_id=pid, camera_id=camera_id, last_bbox=bbox
                )
                self._persons[pid] = person

            person.last_seen_at = now
            person.last_bbox = bbox

            # Lưu lịch sử vị trí (giới hạn 300 entries ~ 60s ở 5fps)
            person.position_history.append((now, bbox))
            if len(person.position_history) > 300:
                person.position_history = person.position_history[-300:]

            # Person đang hiện diện → xóa khỏi missing list
            self._missing_since.pop(pid, None)
            person.departure_processed = False

            # ----- Cập nhật dwell zones -----
            self._update_dwell_zones(person, bbox, frame_bgr, frame_w, frame_h, now)

        # ----- Bước 2: Xử lý person mất tích -----
        for pid, person in list(self._persons.items()):
            if pid in seen_ids:
                continue
            if person.camera_id != camera_id:
                continue

            # Person không xuất hiện trong frame này
            if pid not in self._missing_since:
                self._missing_since[pid] = now

            missing_duration = now - self._missing_since[pid]

            if missing_duration >= self.departure_confirm_seconds and not person.departure_processed:
                # Chờ đúng frame có ảnh CPU để kiểm tra. Trước đây cờ processed
                # bị set ngay cả khi frame_bgr=None (do throttle 1/5 frame), làm
                # mất vĩnh viễn phần lớn departure event.
                if frame_bgr is None:
                    continue

                # Person đã rời đi đủ lâu → kiểm tra vật bỏ lại
                person.departure_processed = True
                logger.debug(
                    f"[Departure] Person {pid} đã rời khỏi camera {camera_id} "
                    f"sau {missing_duration:.1f}s. Kiểm tra {len(person.dwell_zones)} dwell zones."
                )

                if frame_bgr is not None:
                    zone_candidates = self._check_dwell_zones_for_abandoned(
                        person, frame_bgr, frame_w, frame_h
                    )
                    for c in zone_candidates:
                        c["departed_person_id"] = pid
                    candidates.extend(zone_candidates)

            # Xóa person quá cũ (đã xử lý và mất tích > 60s)
            if missing_duration > 60.0:
                del self._persons[pid]
                self._missing_since.pop(pid, None)

        return candidates

    def _update_dwell_zones(
        self,
        person: _TrackedPerson,
        bbox: tuple,
        frame_bgr: np.ndarray | None,
        frame_w: int,
        frame_h: int,
        now: float,
    ):
        """Cập nhật dwell zones của person dựa trên vị trí hiện tại."""
        cx, cy = centroid(bbox)

        # Tìm dwell zone gần nhất đã có
        matched_zone = None
        for zone in person.dwell_zones:
            d = euclidean_distance((cx, cy), zone.center)
            if d <= self.dwell_zone_match_distance:
                matched_zone = zone
                break

        if matched_zone is not None:
            # Person vẫn đứng gần zone cũ → cập nhật
            matched_zone.last_person_bbox = bbox
            # Cập nhật center (moving average)
            old_cx, old_cy = matched_zone.center
            matched_zone.center = (old_cx * 0.9 + cx * 0.1, old_cy * 0.9 + cy * 0.1)
            person.stationary_anchor = matched_zone.center
            person.stationary_since = matched_zone.entered_at
            return

        current_center = (cx, cy)
        anchor = person.stationary_anchor
        moved_from_anchor = (
            anchor is None
            or euclidean_distance(current_center, anchor) > self.stationary_distance_px
        )
        if moved_from_anchor:
            expanded_bbox = self._expand_bbox(bbox, frame_w, frame_h)
            person.stationary_anchor = current_center
            person.stationary_since = now
            person.stationary_zone_bbox = expanded_bbox
            person.stationary_reference = (
                self._crop_region(frame_bgr, expanded_bbox)
                if frame_bgr is not None
                else None
            )
            return

        # Giữ ảnh sớm nhất có thể nếu frame đầu của stationary run bị throttle.
        if person.stationary_reference is None and frame_bgr is not None:
            person.stationary_reference = self._crop_region(
                frame_bgr, person.stationary_zone_bbox
            )

        if now - person.stationary_since < self.min_dwell_seconds:
            return

        new_zone = _DwellZone(
            center=person.stationary_anchor,
            bbox=person.stationary_zone_bbox,
            entered_at=person.stationary_since,
            reference_snapshot=person.stationary_reference,
            last_person_bbox=bbox,
        )
        person.dwell_zones.append(new_zone)

        if len(person.dwell_zones) > self.max_reference_snapshots:
            person.dwell_zones = person.dwell_zones[-self.max_reference_snapshots:]

        logger.debug(
            f"[Dwell] Person {person.person_id} tạo dwell zone mới tại "
            f"({cx:.0f}, {cy:.0f}), tổng {len(person.dwell_zones)} zones."
        )

    def _is_person_stationary(self, person: _TrackedPerson, now: float) -> bool:
        """Kiểm tra person đứng yên >= min_dwell_seconds."""
        if len(person.position_history) < 2:
            return False

        # Duyệt ngược lịch sử, tìm khoảng thời gian đứng yên liên tục
        latest_bbox = person.position_history[-1][1]
        latest_center = centroid(latest_bbox)

        stationary_since = now
        for ts, bbox in reversed(person.position_history[:-1]):
            c = centroid(bbox)
            d = euclidean_distance(latest_center, c)
            if d > self.stationary_distance_px:
                break
            stationary_since = ts

        duration = now - stationary_since
        return duration >= self.min_dwell_seconds

    def _check_dwell_zones_for_abandoned(
        self,
        person: _TrackedPerson,
        frame_bgr: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> list[dict]:
        """Kiểm tra từng dwell zone xem có vật mới bỏ lại không."""
        candidates = []

        for zone in person.dwell_zones:
            if zone.reference_snapshot is None:
                continue

            # Chỉ kiểm tra zone mà person đứng đủ lâu (> min_dwell_seconds)
            # (vì reference snapshot đã được tạo khi stationary, nên zone nào
            # tồn tại đều đã đủ điều kiện)

            # Crop ảnh hiện tại tại vùng dwell zone
            # Dùng vùng BÊN DƯỚI person (phần chân → nơi đặt đồ) thay vì toàn bbox
            check_bbox = self._get_ground_region(zone, frame_w, frame_h)
            current_crop = self._crop_region(frame_bgr, check_bbox)

            if current_crop is None or current_crop.size == 0:
                continue

            # Crop reference tương ứng (cùng vùng ground)
            ref_bbox_local = self._map_to_local_coords(check_bbox, zone.bbox)
            if ref_bbox_local is None:
                continue

            rx1, ry1, rx2, ry2 = ref_bbox_local
            h_ref, w_ref = zone.reference_snapshot.shape[:2]
            rx1 = max(0, min(rx1, w_ref))
            ry1 = max(0, min(ry1, h_ref))
            rx2 = max(0, min(rx2, w_ref))
            ry2 = max(0, min(ry2, h_ref))

            ref_crop = zone.reference_snapshot[ry1:ry2, rx1:rx2]
            if ref_crop.size == 0:
                continue

            # So sánh reference vs current
            diff_results = compare_regions(
                ref_crop_bgr=ref_crop,
                cur_crop_bgr=current_crop,
                ssim_threshold=self.ssim_change_threshold,
                min_contour_area_px=self.min_contour_area_px,
            )

            for result in diff_results:
                # Chuyển bbox từ tọa độ crop → tọa độ frame gốc
                lx1, ly1, lx2, ly2 = result["bbox"]
                gx1 = check_bbox[0] + lx1
                gy1 = check_bbox[1] + ly1
                gx2 = check_bbox[0] + lx2
                gy2 = check_bbox[1] + ly2

                candidates.append({
                    "bbox": (float(gx1), float(gy1), float(gx2), float(gy2)),
                    "contour": result["contour"],
                    "ssim_score": result["ssim_score"],
                    "dwell_zone_bbox": zone.bbox,
                    # Duration is frozen at the person's last tracked frame,
                    # not at the later departure-confirmation frame.
                    "person_dwell_seconds": max(
                        0.0, person.last_seen_at - zone.entered_at
                    ),
                })

        return candidates

    def _expand_bbox(self, bbox: tuple, frame_w: int, frame_h: int) -> tuple:
        """Mở rộng bbox thêm padding xung quanh."""
        x1, y1, x2, y2 = bbox
        pad = self.region_padding_px
        return (
            max(0, int(x1 - pad)),
            max(0, int(y1 - pad)),
            min(frame_w, int(x2 + pad)),
            min(frame_h, int(y2 + pad)),
        )

    def _get_ground_region(self, zone: _DwellZone, frame_w: int, frame_h: int) -> tuple:
        """Lấy vùng 'mặt đất' quanh person – nơi đồ vật thường được đặt.

        Lấy nửa dưới + hai bên của zone bbox (không lấy phần đầu/thân trên
        của person vì đồ vật thường đặt ở dưới chân).
        """
        x1, y1, x2, y2 = zone.bbox
        h = y2 - y1
        # Lấy 60% dưới (bỏ 40% trên – phần đầu/vai)
        ground_y1 = int(y1 + h * 0.4)
        return (
            max(0, x1),
            max(0, ground_y1),
            min(frame_w, x2),
            min(frame_h, y2),
        )

    @staticmethod
    def _crop_region(frame_bgr: np.ndarray, bbox: tuple) -> np.ndarray | None:
        """Crop vùng từ frame."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame_bgr[y1:y2, x1:x2].copy()

    @staticmethod
    def _map_to_local_coords(global_bbox: tuple, container_bbox: tuple) -> tuple | None:
        """Chuyển tọa độ global → local (trong container)."""
        gx1, gy1, gx2, gy2 = [int(v) for v in global_bbox]
        cx1, cy1, cx2, cy2 = [int(v) for v in container_bbox]

        lx1 = gx1 - cx1
        ly1 = gy1 - cy1
        lx2 = gx2 - cx1
        ly2 = gy2 - cy1

        if lx2 <= lx1 or ly2 <= ly1:
            return None
        return (lx1, ly1, lx2, ly2)
