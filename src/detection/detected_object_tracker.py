"""Small IoU tracker for semantic detections not handled by NvDCF."""
from __future__ import annotations

import time
import uuid

import cv2
import numpy as np


class _Track:
    def __init__(self, detection, now):
        self.object_id = str(uuid.uuid4())
        self.bbox = tuple(detection["bbox"])
        self.label = str(detection.get("class_name", "object"))
        self.confidence = float(detection.get("confidence", 0.0))
        self.source = str(detection.get("source", "detector"))
        self.last_seen_at = now
        self.consecutive_hits = 1
        self.semantic_hits = 1
        self.template = None
        self.visual_score = 0.0
        self.target_margin = float(detection.get("target_margin", 0.0))


class DetectedObjectTracker:
    """Keep detector IDs stable and count only spatially-stable observations."""

    def __init__(self, iou_fn, cfg: dict):
        self._iou = iou_fn
        self.iou_threshold = float(cfg.get("iou_match_threshold", 0.25))
        self.max_center_shift_px = float(cfg.get("stationary_center_shift_px", 35.0))
        self.max_missed_seconds = float(cfg.get("max_missed_seconds", 1.0))
        self.unconfirmed_max_missed_seconds = float(
            cfg.get("unconfirmed_max_missed_seconds", self.max_missed_seconds)
        )
        self.visual_enabled = bool(cfg.get("visual_tracking_enabled", True))
        self.visual_match_threshold = float(
            cfg.get("visual_match_threshold", 0.58)
        )
        self.strong_visual_match_threshold = float(
            cfg.get("strong_visual_match_threshold", self.visual_match_threshold)
        )
        self.visual_search_padding_ratio = float(
            cfg.get("visual_search_padding_ratio", 0.75)
        )
        self.min_semantic_hits_for_visual = max(
            1, int(cfg.get("min_semantic_hits_for_visual", 2))
        )
        self.min_semantic_hits_for_static = max(
            2, int(cfg.get("min_semantic_hits_for_static", 2))
        )
        self.min_semantic_hits_for_publish = max(
            2, int(cfg.get("min_semantic_hits_for_publish", 2))
        )
        self.strong_publish_margin = float(
            cfg.get("strong_publish_margin", 1.0)
        )
        # A semantically clear one-frame hit may be the only time a bag is
        # unobscured. Let that track use the cheap local visual tracker and a
        # longer grace period without relaxing the detector for weak boxes.
        self.strong_visual_margin = float(
            cfg.get("strong_visual_margin", self.strong_publish_margin)
        )
        self.strong_max_missed_seconds = max(
            self.max_missed_seconds,
            float(cfg.get("strong_max_missed_seconds", self.max_missed_seconds)),
        )
        self.track_nms_iou_threshold = float(
            cfg.get("track_nms_iou_threshold", 0.40)
        )
        self.track_containment_threshold = float(
            cfg.get("track_containment_threshold", 0.75)
        )
        self.bbox_alpha = min(
            1.0, max(0.05, float(cfg.get("bbox_smoothing_alpha", 0.35)))
        )
        self.visual_bbox_alpha = min(
            1.0,
            max(0.05, float(cfg.get("visual_bbox_smoothing_alpha", 1.0))),
        )
        self._tracks: dict[str, _Track] = {}

    def _is_strong(self, track):
        return track.target_margin >= self.strong_visual_margin

    def _max_missed_for(self, track):
        if self._is_strong(track):
            return self.strong_max_missed_seconds
        if track.semantic_hits >= self.min_semantic_hits_for_publish:
            return self.max_missed_seconds
        return self.unconfirmed_max_missed_seconds

    @staticmethod
    def _center(box):
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    @staticmethod
    def _small_box(box, scale_x, scale_y):
        return (
            box[0] / scale_x, box[1] / scale_y,
            box[2] / scale_x, box[3] / scale_y,
        )

    @staticmethod
    def _gray(frame_bgr):
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def _capture_template(self, track, gray, scale_x, scale_y):
        x1, y1, x2, y2 = self._small_box(track.bbox, scale_x, scale_y)
        frame_h, frame_w = gray.shape[:2]
        x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
        x2, y2 = min(frame_w, int(round(x2))), min(frame_h, int(round(y2)))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return
        template = gray[y1:y2, x1:x2].copy()
        # Uniform patches are unsafe for correlation tracking.
        if float(template.std()) >= 5.0:
            track.template = template

    def _visual_update(self, track, gray, scale_x, scale_y, now):
        template = track.template
        visual_eligible = (
            track.semantic_hits >= self.min_semantic_hits_for_visual
            or self._is_strong(track)
        )
        if template is None or not visual_eligible:
            return False

        x1, y1, x2, y2 = self._small_box(track.bbox, scale_x, scale_y)
        box_w, box_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        pad_x = max(16, int(round(box_w * self.visual_search_padding_ratio)))
        pad_y = max(16, int(round(box_h * self.visual_search_padding_ratio)))
        frame_h, frame_w = gray.shape[:2]
        sx1 = max(0, int(round(x1)) - pad_x)
        sy1 = max(0, int(round(y1)) - pad_y)
        sx2 = min(frame_w, int(round(x2)) + pad_x)
        sy2 = min(frame_h, int(round(y2)) + pad_y)
        search = gray[sy1:sy2, sx1:sx2]
        template_h, template_w = template.shape[:2]
        if search.shape[0] < template_h or search.shape[1] < template_w:
            return False

        scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, best_score, _, best_location = cv2.minMaxLoc(scores)
        match_threshold = (
            self.strong_visual_match_threshold
            if self._is_strong(track)
            else self.visual_match_threshold
        )
        if not np.isfinite(best_score) or best_score < match_threshold:
            return False

        nx1 = sx1 + best_location[0]
        ny1 = sy1 + best_location[1]
        new_bbox = (
            nx1 * scale_x,
            ny1 * scale_y,
            (nx1 + template_w) * scale_x,
            (ny1 + template_h) * scale_y,
        )
        old_center = self._center(track.bbox)
        new_center = self._center(new_bbox)
        shift = (
            (old_center[0] - new_center[0]) ** 2
            + (old_center[1] - new_center[1]) ** 2
        ) ** 0.5
        # A single semantic false-positive must never become an abandoned
        # object merely because template matching can rediscover itself.
        # Visual tracking is allowed to keep the box visible immediately,
        # while static confirmation still requires repeated detector hits.
        if track.semantic_hits >= self.min_semantic_hits_for_static:
            track.consecutive_hits = (
                track.consecutive_hits + 1
                if shift <= self.max_center_shift_px
                else 1
            )
        else:
            track.consecutive_hits = 1
        alpha = self.visual_bbox_alpha
        track.bbox = tuple(
            old * (1.0 - alpha) + new * alpha
            for old, new in zip(track.bbox, new_bbox)
        )
        track.last_seen_at = now
        track.visual_score = float(best_score)
        return True

    def update(
        self,
        detections: list[dict],
        frame_bgr=None,
        scale_x=1.0,
        scale_y=1.0,
        now=None,
    ) -> list[dict]:
        now = time.time() if now is None else float(now)
        used_ids = set()
        gray = self._gray(frame_bgr) if frame_bgr is not None else None

        for detection in sorted(
            detections,
            key=lambda item: float(item.get("confidence", 0.0)),
            reverse=True,
        ):
            bbox = tuple(detection["bbox"])
            best_id, best_iou = None, 0.0
            for object_id, track in self._tracks.items():
                if object_id in used_ids:
                    continue
                score = self._iou(bbox, track.bbox)
                if score > best_iou:
                    best_id, best_iou = object_id, score

            if best_id is None or best_iou < self.iou_threshold:
                track = _Track(detection, now)
                self._tracks[track.object_id] = track
                used_ids.add(track.object_id)
                if gray is not None:
                    self._capture_template(track, gray, scale_x, scale_y)
                continue

            track = self._tracks[best_id]
            old_center = self._center(track.bbox)
            new_center = self._center(bbox)
            shift = (
                (old_center[0] - new_center[0]) ** 2
                + (old_center[1] - new_center[1]) ** 2
            ) ** 0.5
            track.consecutive_hits = (
                track.consecutive_hits + 1
                if shift <= self.max_center_shift_px
                else 1
            )
            track.semantic_hits += 1
            alpha = self.bbox_alpha
            track.bbox = tuple(
                old * (1.0 - alpha) + new * alpha
                for old, new in zip(track.bbox, bbox)
            )
            new_source = str(detection.get("source", track.source))
            # The open-vocabulary prompt is more specific than a COCO
            # fallback class. Do not downgrade "plastic garbage bag" to the
            # generic "handbag" on frames where YOLO-World is not sampled.
            if track.source != "yolo_world" or new_source == "yolo_world":
                track.label = str(detection.get("class_name", track.label))
                track.source = new_source
            track.confidence = max(
                float(detection.get("confidence", 0.0)), track.confidence * 0.90
            )
            track.target_margin = max(
                float(detection.get("target_margin", 0.0)),
                track.target_margin * 0.90,
            )
            track.last_seen_at = now
            used_ids.add(best_id)
            if gray is not None:
                self._capture_template(track, gray, scale_x, scale_y)

        # On frames where expensive YOLO inference is skipped or misses an
        # intermittent bag, local template matching keeps the box alive on CPU.
        if self.visual_enabled and gray is not None:
            for object_id, track in self._tracks.items():
                if object_id not in used_ids:
                    self._visual_update(
                        track, gray, float(scale_x), float(scale_y), now
                    )

        stale_ids = [
            object_id
            for object_id, track in self._tracks.items()
            if now - track.last_seen_at > self._max_missed_for(track)
        ]
        for object_id in stale_ids:
            del self._tracks[object_id]
        return self.snapshot(now)

    def snapshot(self, now=None) -> list[dict]:
        now = time.time() if now is None else float(now)
        live_tracks = [
            track for track in self._tracks.values()
            if now - track.last_seen_at <= self._max_missed_for(track)
        ]
        # Different prompts/tiles can describe the same physical bag. Keep the
        # strongest temporal track when boxes overlap or one contains another.
        selected = []
        for track in sorted(
            live_tracks,
            key=lambda item: (item.semantic_hits, item.confidence),
            reverse=True,
        ):
            area = max(1.0, (track.bbox[2] - track.bbox[0]) * (track.bbox[3] - track.bbox[1]))
            duplicate = False
            for kept in selected:
                kept_area = max(
                    1.0,
                    (kept.bbox[2] - kept.bbox[0]) * (kept.bbox[3] - kept.bbox[1]),
                )
                ax1, ay1, ax2, ay2 = track.bbox
                bx1, by1, bx2, by2 = kept.bbox
                intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
                    0.0, min(ay2, by2) - max(ay1, by1)
                )
                containment = intersection / min(area, kept_area)
                if (
                    self._iou(track.bbox, kept.bbox) >= self.track_nms_iou_threshold
                    or containment >= self.track_containment_threshold
                ):
                    duplicate = True
                    break
            if not duplicate:
                selected.append(track)

        return [
            {
                "object_id": track.object_id,
                "bbox": track.bbox,
                "is_new": track.consecutive_hits == 1,
                "consecutive_hits": track.consecutive_hits,
                "label": track.label,
                "label_confidence": track.confidence,
                "source": track.source,
                "semantic_hits": track.semantic_hits,
                "visual_score": track.visual_score,
                "target_margin": track.target_margin,
                "publishable": (
                    track.semantic_hits >= self.min_semantic_hits_for_publish
                    or track.target_margin >= self.strong_publish_margin
                ),
            }
            for track in selected
        ]

    def clear(self):
        self._tracks.clear()
