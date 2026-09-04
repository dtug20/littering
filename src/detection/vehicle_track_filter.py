"""Confidence and temporal filter for DeepStream vehicle tracks."""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class _VehicleTrack:
    class_name: str
    bbox: tuple
    detector_hits: int
    last_detector_at: float
    last_seen_at: float
    confidence: float
    confirmed: bool = False


class VehicleTrackFilter:
    """Reject one-frame/low-quality vehicle boxes before MQTT publication.

    NvDCF predictions may have a negative detector confidence. They are only
    accepted after the same track has first received repeated PGIE evidence.
    """

    def __init__(self, cfg: dict, iou_fn):
        self.enabled = bool(cfg.get("enabled", True))
        self.default_confidence = float(cfg.get("min_detector_confidence", 0.45))
        self.class_confidence = {
            str(key).lower(): float(value)
            for key, value in cfg.get("class_min_confidence", {}).items()
        }
        self.class_min_aspect = {
            str(key).lower(): float(value)
            for key, value in cfg.get("class_min_aspect_ratio", {}).items()
        }
        self.min_detector_hits = max(1, int(cfg.get("min_detector_hits", 2)))
        self.confirmation_window = float(cfg.get("confirmation_window_seconds", 1.5))
        self.max_detector_gap = float(cfg.get("max_detector_gap_seconds", 1.5))
        self.min_area_ratio = float(cfg.get("min_area_ratio", 0.0008))
        self.max_area_ratio = float(cfg.get("max_area_ratio", 0.55))
        self.min_aspect_ratio = float(cfg.get("min_aspect_ratio", 0.40))
        self.max_aspect_ratio = float(cfg.get("max_aspect_ratio", 4.50))
        self.iou_threshold = float(cfg.get("confirmation_iou_threshold", 0.20))
        self._iou = iou_fn
        self._tracks: dict[tuple[str, str], _VehicleTrack] = {}

    def _plausible(self, detection: dict, frame_w: int, frame_h: int) -> bool:
        x1, y1, x2, y2 = detection["bbox"]
        width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
        if width <= 0 or height <= 0:
            return False
        area_ratio = width * height / max(1.0, float(frame_w * frame_h))
        aspect = width / height
        class_name = str(detection["class_name"]).lower()
        min_aspect = self.class_min_aspect.get(class_name, self.min_aspect_ratio)
        return (
            self.min_area_ratio <= area_ratio <= self.max_area_ratio
            and min_aspect <= aspect <= self.max_aspect_ratio
        )

    def update(self, camera_id: str, detections: list[dict], frame_w: int,
               frame_h: int, now: float | None = None) -> dict:
        now = time.time() if now is None else float(now)
        accepted = {}
        if not self.enabled:
            return {
                str(item["object_id"]): {
                    "bbox": item["bbox"], "class_name": item["class_name"]
                }
                for item in detections
            }

        for item in detections:
            object_id = str(item.get("object_id", ""))
            if not object_id or not self._plausible(item, frame_w, frame_h):
                continue
            key = (str(camera_id), object_id)
            class_name = str(item["class_name"]).lower()
            confidence = float(item.get("confidence", -1.0))
            threshold = self.class_confidence.get(class_name, self.default_confidence)
            detector_hit = confidence >= threshold
            record = self._tracks.get(key)

            if record is None:
                if not detector_hit:
                    continue
                record = _VehicleTrack(
                    class_name=class_name,
                    bbox=tuple(item["bbox"]),
                    detector_hits=1,
                    last_detector_at=now,
                    last_seen_at=now,
                    confidence=confidence,
                )
                self._tracks[key] = record
            else:
                if detector_hit:
                    continuous = (
                        class_name == record.class_name
                        and now - record.last_detector_at <= self.confirmation_window
                        and self._iou(tuple(item["bbox"]), record.bbox) >= self.iou_threshold
                    )
                    record.detector_hits = record.detector_hits + 1 if continuous else 1
                    record.last_detector_at = now
                    record.confidence = confidence
                record.class_name = class_name
                record.bbox = tuple(item["bbox"])
                record.last_seen_at = now

            if record.detector_hits >= self.min_detector_hits:
                record.confirmed = True
            if record.confirmed and now - record.last_detector_at <= self.max_detector_gap:
                accepted[object_id] = {
                    "bbox": record.bbox,
                    "class_name": record.class_name,
                    "confidence": record.confidence,
                }

        stale = [
            key for key, record in self._tracks.items()
            if key[0] == str(camera_id) and now - record.last_detector_at > self.max_detector_gap
        ]
        for key in stale:
            del self._tracks[key]
        return accepted

    def clear_camera(self, camera_id: str):
        self._tracks = {
            key: value for key, value in self._tracks.items()
            if key[0] != str(camera_id)
        }
