"""Unit tests for semantic bbox detection and temporal tracking."""
import unittest

import numpy as np

from src.detection.detected_object_tracker import DetectedObjectTracker
from src.detection.open_vocab_object_detector import OnnxYoloWorldObjectDetector
from src.utils import geometry


class _Input:
    name = "images"
    shape = [1, 3, 100, 100]


class _Meta:
    custom_metadata_map = {"names": "{0: 'plastic garbage bag', 1: 'person'}"}


class _Session:
    def __init__(self, anchors):
        self.anchors = anchors

    def get_inputs(self):
        return [_Input()]

    def get_modelmeta(self):
        return _Meta()

    def run(self, *_args):
        # Ultralytics detect output: [batch, 4 + classes, anchors].
        return [np.asarray(self.anchors, dtype=np.float32).T[None]]


class OpenVocabularyDetectorTests(unittest.TestCase):
    def _detector(self, anchors):
        return OnnxYoloWorldObjectDetector(
            {
                "target_labels": ["plastic garbage bag"],
                "min_confidence": 0.30,
                "min_target_margin": 0.05,
                "nms_iou_threshold": 0.45,
            },
            session=_Session(anchors),
        )

    def test_decodes_two_distinct_bag_boxes_and_suppresses_duplicate(self):
        # cx, cy, w, h, bag_score, person_score
        detector = self._detector([
            [25, 30, 20, 20, 0.90, 0.10],
            [26, 30, 20, 20, 0.80, 0.10],  # duplicate of first
            [70, 65, 18, 24, 0.85, 0.05],
        ])
        detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0]["class_name"], "plastic garbage bag")
        self.assertAlmostEqual(detections[0]["target_margin"], 0.80, places=5)
        self.assertEqual(tuple(round(v) for v in detections[0]["bbox"]), (15, 20, 35, 40))
        self.assertEqual(tuple(round(v) for v in detections[1]["bbox"]), (61, 53, 79, 77))

    def test_hard_negative_person_must_not_become_a_bag(self):
        detector = self._detector([
            [50, 50, 30, 60, 0.70, 0.72],
        ])
        self.assertEqual(
            detector.detect(np.zeros((100, 100, 3), dtype=np.uint8)),
            [],
        )


class DetectedObjectTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = DetectedObjectTracker(
            geometry.iou,
            {
                "iou_match_threshold": 0.20,
                "stationary_center_shift_px": 5,
                "max_missed_seconds": 1.0,
                "bbox_smoothing_alpha": 0.5,
            },
        )

    @staticmethod
    def _bag(bbox, source="yolo_world", label="plastic garbage bag"):
        return {
            "bbox": bbox,
            "class_name": label,
            "confidence": 0.80,
            "source": source,
        }

    def test_stable_detection_keeps_id_and_increments_static_hits(self):
        first = self.tracker.update([self._bag((10, 10, 30, 30))], now=0)[0]
        second = self.tracker.update([self._bag((12, 10, 32, 30))], now=0.2)[0]
        self.assertEqual(first["object_id"], second["object_id"])
        self.assertEqual(second["consecutive_hits"], 2)

    def test_moving_detection_resets_static_hits(self):
        self.tracker.update([self._bag((10, 10, 40, 40))], now=0)
        moved = self.tracker.update([self._bag((20, 10, 50, 40))], now=0.2)[0]
        self.assertEqual(moved["consecutive_hits"], 1)

    def test_world_label_is_not_downgraded_by_coco_fallback(self):
        first = self.tracker.update([self._bag((10, 10, 30, 30))], now=0)[0]
        second = self.tracker.update([
            self._bag(
                (10, 10, 30, 30),
                source="deepstream_yolo",
                label="handbag",
            )
        ], now=0.2)[0]
        self.assertEqual(first["object_id"], second["object_id"])
        self.assertEqual(second["label"], "plastic garbage bag")

    def test_short_detector_gap_is_cached_but_stale_track_expires(self):
        first = self.tracker.update([self._bag((10, 10, 30, 30))], now=0)[0]
        cached = self.tracker.update([], now=0.5)
        self.assertEqual(cached[0]["object_id"], first["object_id"])
        self.assertEqual(self.tracker.update([], now=1.1), [])

    def test_cpu_template_tracking_keeps_confirmed_box_between_inferences(self):
        texture = np.random.default_rng(7).integers(
            0, 255, size=(20, 20), dtype=np.uint8
        )
        first_frame = np.zeros((80, 100, 3), dtype=np.uint8)
        first_frame[20:40, 20:40] = texture[:, :, None]
        detection = self._bag((20, 20, 40, 40))

        first = self.tracker.update(
            [detection], frame_bgr=first_frame, now=0
        )[0]
        self.tracker.update(
            [detection], frame_bgr=first_frame, now=0.2
        )

        moved_frame = np.zeros_like(first_frame)
        moved_frame[22:42, 23:43] = texture[:, :, None]
        tracked = self.tracker.update(
            [], frame_bgr=moved_frame, now=0.4
        )[0]

        self.assertEqual(first["object_id"], tracked["object_id"])
        self.assertGreaterEqual(tracked["visual_score"], 0.99)
        self.assertEqual(tuple(round(v) for v in tracked["bbox"]), (23, 22, 43, 42))

    def test_first_hit_can_be_visually_tracked_but_cannot_confirm_static(self):
        tracker = DetectedObjectTracker(
            geometry.iou,
            {
                "min_semantic_hits_for_visual": 1,
                "min_semantic_hits_for_static": 2,
                "visual_match_threshold": 0.65,
            },
        )
        texture = np.random.default_rng(11).integers(
            0, 255, size=(20, 20), dtype=np.uint8
        )
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        frame[20:40, 20:40] = texture[:, :, None]

        first = tracker.update(
            [self._bag((20, 20, 40, 40))], frame_bgr=frame, now=0
        )[0]
        tracked = tracker.update([], frame_bgr=frame, now=0.2)[0]

        self.assertEqual(first["object_id"], tracked["object_id"])
        self.assertEqual(tracked["semantic_hits"], 1)
        self.assertEqual(tracked["consecutive_hits"], 1)

    def test_strong_first_hit_can_use_visual_tracker_without_relaxing_weak_hits(self):
        tracker = DetectedObjectTracker(
            geometry.iou,
            {
                "min_semantic_hits_for_visual": 3,
                "strong_visual_margin": 0.10,
                "strong_visual_match_threshold": 0.58,
            },
        )
        texture = np.random.default_rng(19).integers(
            0, 255, size=(20, 20), dtype=np.uint8
        )
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        frame[20:40, 20:40] = texture[:, :, None]

        strong = self._bag((20, 20, 40, 40))
        strong["target_margin"] = 0.11
        tracker.update([strong], frame_bgr=frame, now=0)
        tracked = tracker.update([], frame_bgr=frame, now=0.2)[0]
        self.assertGreaterEqual(tracked["visual_score"], 0.99)

        tracker.clear()
        weak = self._bag((20, 20, 40, 40))
        weak["target_margin"] = 0.09
        tracker.update([weak], frame_bgr=frame, now=0)
        cached = tracker.update([], frame_bgr=frame, now=0.2)[0]
        self.assertEqual(cached["visual_score"], 0.0)

    def test_strong_track_uses_longer_miss_grace(self):
        tracker = DetectedObjectTracker(
            geometry.iou,
            {
                "max_missed_seconds": 6.0,
                "unconfirmed_max_missed_seconds": 1.0,
                "strong_max_missed_seconds": 15.0,
                "strong_visual_margin": 0.10,
                "min_semantic_hits_for_publish": 2,
            },
        )
        strong = self._bag((10, 10, 30, 30))
        strong["target_margin"] = 0.11
        first = tracker.update([strong], now=0)[0]
        cached = tracker.update([], now=10.0)[0]
        self.assertEqual(cached["object_id"], first["object_id"])
        self.assertEqual(tracker.update([], now=15.1), [])

    def test_track_is_publishable_only_after_three_semantic_hits(self):
        tracker = DetectedObjectTracker(
            geometry.iou,
            {"min_semantic_hits_for_publish": 3},
        )
        first = tracker.update([self._bag((10, 10, 30, 30))], now=0)[0]
        second = tracker.update([self._bag((10, 10, 30, 30))], now=.2)[0]
        third = tracker.update([self._bag((10, 10, 30, 30))], now=.4)[0]
        self.assertFalse(first["publishable"])
        self.assertFalse(second["publishable"])
        self.assertTrue(third["publishable"])

    def test_one_hit_requires_strong_target_margin_to_publish(self):
        tracker = DetectedObjectTracker(
            geometry.iou,
            {
                "min_semantic_hits_for_publish": 2,
                "strong_publish_margin": 0.10,
            },
        )
        weak = self._bag((10, 10, 30, 30))
        weak["target_margin"] = 0.09
        self.assertFalse(tracker.update([weak], now=0)[0]["publishable"])

        tracker.clear()
        strong = self._bag((10, 10, 30, 30))
        strong["target_margin"] = 0.11
        self.assertTrue(tracker.update([strong], now=.1)[0]["publishable"])

    def test_nested_prompt_boxes_are_merged_into_one_track_output(self):
        tracker = DetectedObjectTracker(
            geometry.iou,
            {
                "track_nms_iou_threshold": 0.35,
                "track_containment_threshold": 0.70,
            },
        )
        tracks = tracker.update([
            self._bag((10, 10, 40, 40), label="pile of garbage bags"),
            self._bag((12, 12, 38, 38), label="black plastic garbage bag"),
        ], now=0)
        self.assertEqual(len(tracks), 1)

    def test_unconfirmed_track_expires_faster_than_confirmed_track(self):
        tracker = DetectedObjectTracker(
            geometry.iou,
            {
                "max_missed_seconds": 6.0,
                "unconfirmed_max_missed_seconds": 1.0,
                "min_semantic_hits_for_publish": 3,
            },
        )
        tracker.update([self._bag((10, 10, 30, 30))], now=0)
        self.assertEqual(tracker.update([], now=1.1), [])

    def test_dynamic_overlap_fraction_is_asymmetric(self):
        candidate = (10, 10, 30, 30)
        covering_vehicle = (0, 0, 100, 100)
        self.assertEqual(
            geometry.intersection_over_box(candidate, covering_vehicle), 1.0
        )
        self.assertAlmostEqual(
            geometry.intersection_over_box(covering_vehicle, candidate), 0.04
        )

if __name__ == "__main__":
    unittest.main()
