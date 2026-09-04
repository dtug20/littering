"""Safety tests for trash-only alerting; no DeepStream/GPU is required."""
import unittest
from unittest.mock import patch

import numpy as np

from src.detection.person_departure_detector import PersonDepartureDetector
from src.detection.blob_tracker import BlobTracker
from src.detection.shape_filters import is_valid_size, overlaps_any_person
from src.detection.target_verifier import (
    OnnxYoloWorldTargetVerifier,
    TargetPrediction,
    TemporalTargetGate,
    UltralyticsYoloTargetVerifier,
    UnavailableTargetVerifier,
)
from src.logic.ownership import VehicleDwellAssociator
from src.tracking.object_state_tracker import ObjectStateTracker
from src.utils import geometry


class _SequenceVerifier:
    def __init__(self, predictions):
        self.predictions = iter(predictions)

    def verify(self, frame_bgr, bbox):
        return next(self.predictions)


class _ArrayLike:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _FakeBoxes:
    def __init__(self, classes, confidences):
        self.cls = _ArrayLike(classes)
        self.conf = _ArrayLike(confidences)


class _FakeResult:
    def __init__(self, names, classes, confidences):
        self.names = names
        self.boxes = _FakeBoxes(classes, confidences)
        self.probs = None


class _FakeModel:
    def __init__(self, result):
        self.result = result
        self.names = result.names

    def predict(self, **kwargs):
        return [self.result]


class _FakeOnnxInput:
    name = "images"
    shape = [1, 3, 384, 384]


class _FakeOnnxMetadata:
    custom_metadata_map = {"names": "{0: 'garbage bag', 1: 'backpack'}"}


class _FakeOnnxSession:
    def __init__(self, target_score, negative_score):
        self.target_score = target_score
        self.negative_score = negative_score

    def get_inputs(self):
        return [_FakeOnnxInput()]

    def get_modelmeta(self):
        return _FakeOnnxMetadata()

    def run(self, *_args):
        output = np.zeros((1, 6, 2), dtype=np.float32)
        output[0, 4, 0] = self.target_score
        output[0, 5, 1] = self.negative_score
        return [output]


class TargetVerifierTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.bbox = (20, 20, 60, 70)

    def test_temporal_consensus_accepts_only_after_three_target_frames(self):
        verifier = _SequenceVerifier([
            TargetPrediction(True, "garbage_bag", 0.80),
            TargetPrediction(False, "cardboard_box", 0.90),
            TargetPrediction(True, "garbage_bag", 0.85),
            TargetPrediction(True, "garbage_bag", 0.90),
        ])
        gate = TemporalTargetGate(verifier, {
            "confirmations": 3,
            "max_samples": 5,
            "min_positive_ratio": 0.60,
            "sample_interval_seconds": 0,
        })

        self.assertEqual(gate.evaluate("obj", self.frame, self.bbox).decision, "pending")
        self.assertEqual(gate.evaluate("obj", self.frame, self.bbox).decision, "pending")
        self.assertEqual(gate.evaluate("obj", self.frame, self.bbox).decision, "pending")
        accepted = gate.evaluate("obj", self.frame, self.bbox)
        self.assertEqual(accepted.decision, "accept")
        self.assertEqual(accepted.label, "garbage_bag")
        self.assertAlmostEqual(accepted.confidence, 0.85)

    def test_non_target_is_rejected_when_consensus_is_impossible(self):
        verifier = _SequenceVerifier([
            TargetPrediction(False, "backpack", 0.91),
            TargetPrediction(False, "suitcase", 0.88),
            TargetPrediction(True, "garbage_bag", 0.75),
            TargetPrediction(False, "cardboard_box", 0.93),
        ])
        gate = TemporalTargetGate(verifier, {
            "confirmations": 3,
            "max_samples": 5,
            "min_positive_ratio": 0.60,
            "sample_interval_seconds": 0,
        })
        self.assertEqual(gate.evaluate("obj", self.frame, self.bbox).decision, "pending")
        self.assertEqual(gate.evaluate("obj", self.frame, self.bbox).decision, "pending")
        self.assertEqual(gate.evaluate("obj", self.frame, self.bbox).decision, "pending")
        self.assertEqual(gate.evaluate("obj", self.frame, self.bbox).decision, "reject")

    def test_unavailable_model_is_fail_closed(self):
        gate = TemporalTargetGate(
            UnavailableTargetVerifier("missing_model"),
            {"confirmations": 1, "max_samples": 1, "sample_interval_seconds": 0},
        )
        decision = gate.evaluate("obj", self.frame, self.bbox)
        self.assertEqual(decision.decision, "pending")

    def test_custom_yolo_detector_accepts_only_configured_target_class(self):
        config = {
            "target_labels": ["garbage_bag", "loose_waste"],
            "min_confidence": 0.70,
            "crop_padding_ratio": 0,
        }
        positive_model = _FakeModel(
            _FakeResult(
                {0: "garbage_bag", 1: "cardboard_box", 2: "loose_waste"},
                [1, 0], [0.95, 0.82],
            )
        )
        verifier = UltralyticsYoloTargetVerifier(config, model=positive_model)
        result = verifier.verify(self.frame, self.bbox)
        self.assertTrue(result.is_target)
        self.assertEqual(result.label, "garbage_bag")

        negative_model = _FakeModel(
            _FakeResult(
                {0: "garbage_bag", 1: "cardboard_box", 2: "loose_waste"},
                [1], [0.96],
            )
        )
        verifier = UltralyticsYoloTargetVerifier(config, model=negative_model)
        result = verifier.verify(self.frame, self.bbox)
        self.assertFalse(result.is_target)
        self.assertEqual(result.label, "cardboard_box")

    def test_onnx_world_requires_target_to_beat_hard_negative(self):
        config = {
            "target_labels": ["garbage bag"],
            "min_confidence": 0.35,
            "min_target_margin": 0.05,
            "crop_padding_ratio": 0,
        }
        positive = OnnxYoloWorldTargetVerifier(
            config, session=_FakeOnnxSession(0.80, 0.20)
        ).verify(self.frame, self.bbox)
        self.assertTrue(positive.is_target)
        self.assertEqual(positive.label, "garbage bag")

        ambiguous = OnnxYoloWorldTargetVerifier(
            config, session=_FakeOnnxSession(0.80, 0.78)
        ).verify(self.frame, self.bbox)
        self.assertFalse(ambiguous.is_target)
        self.assertEqual(ambiguous.label, "garbage bag")


class ObjectProposalRegressionTests(unittest.TestCase):
    def test_rejects_full_height_bbox_even_when_area_is_small(self):
        cfg = {
            "min_width_px": 1, "min_height_px": 1,
            "min_area_ratio": 0, "max_area_ratio": 1,
            "max_width_ratio": 0.5, "max_height_ratio": 0.6,
        }
        self.assertFalse(is_valid_size((10, 0, 20, 100), 100, 100, cfg))

    def test_rejects_candidate_inside_person(self):
        candidate = (40, 40, 60, 80)
        persons = [(20, 10, 80, 95)]
        self.assertTrue(overlaps_any_person(candidate, persons, threshold=0.2))

    def test_blob_bbox_is_smoothed_without_changing_id(self):
        tracker = BlobTracker(
            geometry.iou, iou_match_threshold=0.1, bbox_smoothing_alpha=0.25
        )
        first = tracker.update([(0, 0, 100, 100)])[0]
        second = tracker.update([(20, 20, 120, 120)])[0]
        self.assertEqual(first["object_id"], second["object_id"])
        self.assertEqual(second["bbox"], (5.0, 5.0, 105.0, 105.0))


class VehicleDwellTests(unittest.TestCase):
    ROI = [{"name": "dumping", "points": [[0, 0], [500, 0], [500, 500], [0, 500]]}]

    @staticmethod
    def _vehicle(bbox=(100, 100, 200, 200)):
        return {"v1": {"bbox": bbox, "class_name": "car"}}

    def test_pass_through_vehicle_is_not_associated(self):
        tracker = VehicleDwellAssociator({
            "min_vehicle_dwell_seconds": 2.0,
            "vehicle_history_window_seconds": 10.0,
            "vehicle_max_distance_px": 300.0,
        })
        tracker.update("cam", self._vehicle(), self.ROI, geometry, now=0.0)
        tracker.update("cam", self._vehicle((200, 100, 300, 200)), self.ROI, geometry, now=1.0)
        self.assertIsNone(
            tracker.find_related("cam", (210, 180, 250, 230), self.ROI, geometry, now=1.0)
        )

    def test_stopped_vehicle_in_same_roi_is_associated(self):
        tracker = VehicleDwellAssociator({
            "min_vehicle_dwell_seconds": 2.0,
            "vehicle_history_window_seconds": 10.0,
            "vehicle_max_distance_px": 300.0,
        })
        tracker.update("cam", self._vehicle(), self.ROI, geometry, now=0.0)
        tracker.update("cam", self._vehicle(), self.ROI, geometry, now=2.2)
        related = tracker.find_related(
            "cam", (150, 200, 190, 240), self.ROI, geometry, now=2.2
        )
        self.assertEqual(related["track_id"], "v1")
        self.assertEqual(related["class_name"], "car")
        self.assertGreaterEqual(related["dwell_seconds"], 2.0)

    def test_vehicle_remains_associable_just_after_leaving_roi(self):
        tracker = VehicleDwellAssociator({
            "min_vehicle_dwell_seconds": 2.0,
            "vehicle_history_window_seconds": 10.0,
            "vehicle_max_distance_px": 300.0,
        })
        tracker.update("cam", self._vehicle(), self.ROI, geometry, now=0.0)
        tracker.update("cam", self._vehicle(), self.ROI, geometry, now=2.2)
        tracker.update(
            "cam", self._vehicle((600, 100, 700, 200)), self.ROI, geometry, now=3.0
        )
        related = tracker.find_related(
            "cam", (150, 200, 190, 240), self.ROI, geometry, now=4.0
        )
        self.assertEqual(related["track_id"], "v1")


class StateAndRoiSafetyTests(unittest.TestCase):
    RULES = {
        "static_confirm_frames": 1,
        "owner_distance_threshold_px": 100,
        "owner_lost_grace_seconds": 0,
        "abandonment_dwell_seconds": 0,
        "max_occlusion_gap_seconds": 5,
        "pickup_confirm_gap_seconds": 1,
    }

    @staticmethod
    def _blob(object_id="obj", bbox=(10, 10, 30, 30)):
        return [{
            "object_id": object_id,
            "bbox": bbox,
            "is_new": True,
            "consecutive_hits": 1,
            "owner_track_id_hint": "owner",
        }]

    def test_public_alert_is_deferred_until_semantic_confirmation(self):
        tracker = ObjectStateTracker(dict(self.RULES))
        internal = tracker.update(
            "cam", self._blob(), {}, [], False, geometry, lambda _: None,
            defer_abandoned_event=True,
        )
        self.assertEqual([event["event_type"] for event in internal], [
            "target_verification_requested"
        ])
        self.assertFalse(tracker.tracks["obj"].alerted)

        public = tracker.confirm_abandoned("obj", "garbage_bag", 0.87)
        self.assertEqual(public["event_type"], "object_abandoned")
        self.assertEqual(public["label"], "garbage_bag")
        self.assertAlmostEqual(public["label_confidence"], 0.87)
        self.assertIsNone(tracker.confirm_abandoned("obj", "garbage_bag", 0.90))

    def test_alert_links_person_and_dwelled_vehicle(self):
        tracker = ObjectStateTracker(dict(self.RULES))
        blob = self._blob()
        blob[0].update({
            "related_vehicle_id_hint": "vehicle-7",
            "related_vehicle_class_hint": "truck",
            "person_dwell_seconds": 3.1,
            "vehicle_dwell_seconds": 4.2,
        })
        tracker.update(
            "cam", blob, {}, [], False, geometry, lambda _: None,
            defer_abandoned_event=True,
        )
        event = tracker.confirm_abandoned("obj", "garbage bag", 0.88)
        self.assertEqual(event["incident_type"], "illegal_dumping")
        self.assertEqual(event["owner_track_id"], "owner")
        self.assertEqual(event["related_vehicle_id"], "vehicle-7")
        self.assertEqual(event["related_vehicle_class"], "truck")

    def test_rejected_object_never_becomes_abandoned(self):
        tracker = ObjectStateTracker(dict(self.RULES))
        tracker.update(
            "cam", self._blob(), {}, [], False, geometry, lambda _: None,
            defer_abandoned_event=True,
        )
        tracker.reject_target("obj", "suitcase", 0.95)
        later = tracker.update(
            "cam", self._blob(), {}, [], False, geometry, lambda _: None,
            defer_abandoned_event=True,
        )
        self.assertEqual(later, [])
        self.assertEqual(tracker.tracks["obj"].state, "IGNORED")

    def test_reappearing_blob_restores_state_before_pending_claim(self):
        rules = dict(self.RULES)
        rules["abandonment_dwell_seconds"] = 100
        tracker = ObjectStateTracker(rules)
        far_owner = {"owner": (200, 200, 220, 220)}

        tracker.update(
            "cam", self._blob(), far_owner, [], False, geometry, lambda _: None,
        )
        self.assertEqual(tracker.tracks["obj"].state, "STATIC_NO_OWNER")

        # A person briefly hides the object, then the same blob reappears.
        tracker.update(
            "cam", [], {"passerby": (10, 10, 30, 30)}, [], False,
            geometry, lambda _: None,
        )
        self.assertEqual(tracker.tracks["obj"].state, "PENDING_CLAIM")

        tracker.update(
            "cam", self._blob(), far_owner, [], False, geometry, lambda _: None,
        )
        self.assertEqual(tracker.tracks["obj"].state, "STATIC_NO_OWNER")

    def test_outside_polygon_is_not_tracked_and_boundary_is_inside(self):
        polygon = [{"name": "zone", "points": [[0, 0], [100, 0], [100, 100], [0, 100]]}]
        tracker = ObjectStateTracker(dict(self.RULES))
        tracker.update(
            "cam", self._blob(bbox=(120, 120, 140, 140)), {}, polygon,
            False, geometry, lambda _: None,
        )
        self.assertEqual(tracker.tracks, {})
        self.assertTrue(geometry.point_in_polygon((100, 50), polygon[0]["points"]))

    def test_normalised_mqtt_polygon_is_scaled(self):
        polygons = geometry.normalise_roi_polygons(
            [{"points": [{"x": 0.25, "y": 0.25}, {"x": 0.75, "y": 0.25},
                         {"x": 0.75, "y": 0.75}, {"x": 0.25, "y": 0.75}]}],
            200, 100,
        )
        self.assertEqual(polygons[0]["points"][0], [50.0, 25.0])
        self.assertTrue(geometry.box_in_any_polygon((90, 40, 110, 60), polygons))

    def test_roi_uses_bbox_ground_anchor_to_avoid_edge_flicker(self):
        polygon = [{"name": "ground", "points": [[0, 50], [100, 50], [100, 100], [0, 100]]}]
        self.assertTrue(geometry.box_in_any_polygon((40, 10, 60, 60), polygon))
        self.assertFalse(geometry.box_in_any_polygon((40, 10, 60, 49), polygon))


class DepartureSafetyTests(unittest.TestCase):
    def test_reference_is_captured_at_start_of_stationary_run(self):
        detector = PersonDepartureDetector({
            "departure_confirm_seconds": 0.5,
            "region_padding_px": 5,
            "min_dwell_seconds": 1.0,
            "stationary_distance_px": 10,
        })
        early_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        late_frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        bbox = (40, 20, 60, 80)

        with patch("src.detection.person_departure_detector.time.time", return_value=0.0):
            detector.update("cam", {"p1": bbox}, early_frame, 100, 100)
        with patch("src.detection.person_departure_detector.time.time", return_value=1.1):
            detector.update("cam", {"p1": bbox}, late_frame, 100, 100)

        zone = detector._persons["p1"].dwell_zones[0]
        self.assertEqual(float(zone.reference_snapshot.mean()), 0.0)

    def test_departure_waits_for_an_available_cpu_frame(self):
        detector = PersonDepartureDetector({
            "departure_confirm_seconds": 0.5,
            "region_padding_px": 5,
            "min_dwell_seconds": 0.1,
            "stationary_distance_px": 10,
        })
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = (40, 20, 60, 80)
        with patch("src.detection.person_departure_detector.time.time", return_value=0.0):
            detector.update("cam", {"p1": bbox}, frame, 100, 100)
        with patch("src.detection.person_departure_detector.time.time", return_value=0.2):
            detector.update("cam", {"p1": bbox}, frame, 100, 100)
        with patch("src.detection.person_departure_detector.time.time", return_value=1.0):
            detector.update("cam", {}, None, 100, 100)
        with patch("src.detection.person_departure_detector.time.time", return_value=1.6):
            self.assertEqual(detector.update("cam", {}, None, 100, 100), [])
        self.assertFalse(detector._persons["p1"].departure_processed)

        detector._check_dwell_zones_for_abandoned = lambda *args: [{"bbox": (1, 1, 2, 2)}]
        with patch("src.detection.person_departure_detector.time.time", return_value=1.7):
            candidates = detector.update("cam", {}, frame, 100, 100)
        self.assertEqual(candidates[0]["departed_person_id"], "p1")
        self.assertTrue(detector._persons["p1"].departure_processed)


if __name__ == "__main__":
    unittest.main()
