"""
Bộ đệm lịch sử vị trí người (person track), dùng để xác định "ai vừa đặt vật
xuống" tại thời điểm vật được BlobTracker xác nhận là ứng viên hợp lệ - vì
lúc xác nhận (sau vài frame đứng yên liên tiếp), người có thể đã di chuyển ra
xa vị trí đặt đồ vài bước rồi. Cần nhìn lại lịch sử vài giây gần nhất.
"""
import time
from collections import deque
from dataclasses import dataclass


class PersonHistoryBuffer:
    def __init__(self, window_seconds: float = 5.0):
        self.window_seconds = window_seconds
        self._history = {}  # camera_id -> deque[(timestamp, {person_id: bbox})]

    def push(self, camera_id, person_tracks: dict):
        now = time.time()
        buf = self._history.setdefault(camera_id, deque())
        buf.append((now, dict(person_tracks)))
        cutoff = now - self.window_seconds
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    def find_owner_at_location(self, camera_id, location_xy, geometry_mod,
                                max_distance_px=200.0):
        """
        Duyệt lịch sử từ gần nhất -> xa nhất (ưu tiên frame gần hiện tại, vì
        đó là lúc vật vừa tách ra khỏi người), trả về person_id gần
        location_xy nhất trong ngưỡng max_distance_px.
        """
        buf = self._history.get(camera_id)
        if not buf:
            return None
        best_id, best_d = None, float("inf")
        for _, persons in reversed(buf):
            for pid, pbox in persons.items():
                d = geometry_mod.euclidean_distance(location_xy, geometry_mod.centroid(pbox))
                if d < best_d:
                    best_d = d
                    best_id = pid
        if best_d <= max_distance_px:
            return best_id
        return None


@dataclass
class _VehicleDwellRecord:
    track_id: str
    class_name: str
    bbox: tuple
    last_inside_bbox: tuple | None
    roi_index: int | None
    first_inside_at: float | None
    last_seen_at: float
    dwell_seconds: float = 0.0
    is_inside: bool = False


class VehicleDwellAssociator:
    """Track vehicle dwell in an ROI and associate it with a trash location.

    A vehicle is eligible only when it was tracked inside the same polygon for
    ``min_dwell_seconds``. Short detector/tracker gaps are tolerated, while a
    vehicle that merely crosses the polygon is never eligible.
    """

    def __init__(self, cfg: dict):
        self.min_dwell_seconds = float(cfg.get("min_vehicle_dwell_seconds", 2.0))
        self.history_window_seconds = float(
            cfg.get("vehicle_history_window_seconds", 15.0)
        )
        self.max_distance_px = float(cfg.get("vehicle_max_distance_px", 400.0))
        self.max_tracking_gap_seconds = float(
            cfg.get("max_vehicle_tracking_gap_seconds", 1.5)
        )
        self._records: dict[tuple[str, str], _VehicleDwellRecord] = {}

    @staticmethod
    def _anchor(bbox: tuple) -> tuple:
        x1, _y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, y2)

    @classmethod
    def _roi_index(cls, bbox: tuple, roi_polygons: list, geometry_mod) -> int | None:
        anchor = cls._anchor(bbox)
        for index, polygon in enumerate(roi_polygons):
            if geometry_mod.point_in_polygon(anchor, polygon["points"]):
                return index
        return None

    def update(self, camera_id: str, vehicle_tracks: dict, roi_polygons: list,
               geometry_mod, now: float | None = None):
        now = time.time() if now is None else float(now)
        seen_keys = set()

        for track_id, vehicle in vehicle_tracks.items():
            key = (camera_id, str(track_id))
            seen_keys.add(key)
            bbox = vehicle["bbox"]
            roi_index = self._roi_index(bbox, roi_polygons, geometry_mod)
            record = self._records.get(key)
            if record is None:
                record = _VehicleDwellRecord(
                    track_id=str(track_id),
                    class_name=str(vehicle["class_name"]),
                    bbox=bbox,
                    last_inside_bbox=bbox if roi_index is not None else None,
                    roi_index=roi_index,
                    first_inside_at=now if roi_index is not None else None,
                    last_seen_at=now,
                    is_inside=roi_index is not None,
                )
                self._records[key] = record
                continue

            # Entering a different polygon starts a new dwell interval.
            if roi_index is not None:
                if not record.is_inside or record.roi_index != roi_index:
                    record.first_inside_at = now
                    record.dwell_seconds = 0.0
                record.is_inside = True
                record.roi_index = roi_index
                record.last_inside_bbox = bbox
                if record.first_inside_at is not None:
                    record.dwell_seconds = max(
                        record.dwell_seconds, now - record.first_inside_at
                    )
            else:
                self._freeze_dwell(record)
                record.is_inside = False
                record.first_inside_at = None

            record.class_name = str(vehicle["class_name"])
            record.bbox = bbox
            record.last_seen_at = now

        # Freeze dwell after a real tracking gap. Keep the record for delayed
        # person-departure/SSIM association.
        for key, record in list(self._records.items()):
            if key[0] != camera_id or key in seen_keys:
                continue
            if record.is_inside and now - record.last_seen_at > self.max_tracking_gap_seconds:
                self._freeze_dwell(record, end_at=record.last_seen_at)
                record.is_inside = False
                record.first_inside_at = None
            if now - record.last_seen_at > self.history_window_seconds:
                del self._records[key]

    @staticmethod
    def _freeze_dwell(record: _VehicleDwellRecord, end_at: float | None = None):
        if record.is_inside and record.first_inside_at is not None:
            end_at = record.last_seen_at if end_at is None else end_at
            record.dwell_seconds = max(
                record.dwell_seconds, max(0.0, end_at - record.first_inside_at)
            )

    def find_related(self, camera_id: str, trash_bbox: tuple, roi_polygons: list,
                     geometry_mod, now: float | None = None) -> dict | None:
        now = time.time() if now is None else float(now)
        trash_roi = self._roi_index(trash_bbox, roi_polygons, geometry_mod)
        if trash_roi is None:
            return None

        trash_location = self._anchor(trash_bbox)
        candidates = []
        for (record_camera, _), record in self._records.items():
            if record_camera != camera_id or record.roi_index != trash_roi:
                continue
            if now - record.last_seen_at > self.history_window_seconds:
                continue
            dwell = record.dwell_seconds
            if record.is_inside and record.first_inside_at is not None:
                dwell = max(dwell, record.last_seen_at - record.first_inside_at)
            if dwell < self.min_dwell_seconds:
                continue
            distance = geometry_mod.euclidean_distance(
                trash_location, self._anchor(record.last_inside_bbox or record.bbox)
            )
            if distance <= self.max_distance_px:
                candidates.append((distance, -dwell, record, dwell))

        if not candidates:
            return None
        _distance, _neg_dwell, record, dwell = min(candidates, key=lambda item: item[:2])
        return {
            "track_id": record.track_id,
            "class_name": record.class_name,
            "dwell_seconds": float(dwell),
        }

    def clear_camera(self, camera_id: str):
        self._records = {
            key: record for key, record in self._records.items() if key[0] != camera_id
        }
