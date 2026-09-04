"""Tests for the zone camera contract, MQTT payload and vehicle filtering."""
from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.detection.vehicle_track_filter import VehicleTrackFilter
from src.mqtt.camera_contract import camera_pipeline_signature, normalize_web_cameras
from src.mqtt.mqtt_client import MqttAodClient
from src.utils.geometry import iou


ROOT = Path(__file__).resolve().parents[1]


class _FakeMqtt:
    def __init__(self):
        self.subscriptions = []
        self.published = []

    def subscribe(self, topic, qos=0):
        self.subscriptions.append((topic, qos))

    def publish(self, topic, payload, qos=0):
        self.published.append((topic, payload, qos))


class WebCameraContractTests(unittest.TestCase):
    CAMERA_ID = "c88f28d5-1111-2222-3333-444455556666"

    @staticmethod
    def _raw_camera():
        return {
            "id": WebCameraContractTests.CAMERA_ID,
            "code": "CAM_KCN_GATE_01",
            "name": "Camera Cổng Chính KCN",
            "status": "ONLINE",
            "ai_modules": ["LITTERING_DETECTION"],
            "restream_urls": {
                "LITTERING_DETECTION": "rtsp://camera.example/live"
            },
        }

    def test_normalizes_uuid_code_name_and_module_restream(self):
        result = normalize_web_cameras(
            [self._raw_camera()], "LITTERING_DETECTION"
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["camera_id"], self.CAMERA_ID)
        self.assertEqual(result[0]["camera_code"], "CAM_KCN_GATE_01")
        self.assertEqual(result[0]["camera_name"], "Camera Cổng Chính KCN")
        self.assertEqual(result[0]["uri"], "rtsp://camera.example/live")

    def test_filters_camera_for_another_ai_module(self):
        raw = self._raw_camera()
        raw["ai_modules"] = ["PLATE"]
        self.assertEqual(
            normalize_web_cameras([raw], "LITTERING_DETECTION"), []
        )

    def test_pipeline_signature_ignores_unrelated_web_fields(self):
        first = normalize_web_cameras(
            [self._raw_camera()], "LITTERING_DETECTION"
        )
        changed = self._raw_camera()
        changed["last_seen_at"] = "volatile-value"
        second = normalize_web_cameras([changed], "LITTERING_DETECTION")
        self.assertEqual(
            camera_pipeline_signature(first), camera_pipeline_signature(second)
        )

    def test_requests_zone_camera_and_builds_exact_littering_event(self):
        cfg = yaml.safe_load((ROOT / "configs/mqtt_config.yaml").read_text())
        mqtt_client = MqttAodClient(cfg)
        mqtt_client.set_camera_registry([self._raw_camera()])
        fake = _FakeMqtt()
        mqtt_client.client = fake

        mqtt_client._on_connect(fake, None, None, 0)
        self.assertIn(("smart_vms/cameras/zone/42", 0), fake.subscriptions)
        request = next(
            item for item in fake.published
            if item[0] == "smart_vms/cameras/getdevice/zone/42"
        )
        self.assertEqual(json.loads(request[1]), {"zone_id": 42})

        fake.published.clear()
        timestamp = datetime(2026, 9, 3, 11, 0, tzinfo=timezone.utc).timestamp()
        sent = mqtt_client.publish_event(
            self.CAMERA_ID,
            {
                "event_type": "object_abandoned",
                "event_id": "event-1",
                "triggered_at": timestamp,
                "bbox": [0.1, 0.2, 0.3, 0.4],
            },
            snapshot_path="http://minio:9000/vms/snapshots/littering_01.jpg",
        )
        self.assertTrue(sent)
        topic, raw_payload, qos = fake.published[-1]
        self.assertEqual(topic, "smart_vms/ai_events/LITTERING_DETECTION")
        self.assertEqual(qos, 1)
        self.assertEqual(json.loads(raw_payload), {
            "ai_modules": "LITTERING_DETECTION",
            "event_type": "LITTERING",
            "camera_id": self.CAMERA_ID,
            "camera_code": "CAM_KCN_GATE_01",
            "camera_name": "Camera Cổng Chính KCN",
            "event_time": "2026-09-03T11:00:00Z",
            "snapshot_url": "http://minio:9000/vms/snapshots/littering_01.jpg",
        })

    def test_does_not_publish_removed_as_new_littering_event(self):
        cfg = yaml.safe_load((ROOT / "configs/mqtt_config.yaml").read_text())
        mqtt_client = MqttAodClient(cfg)
        fake = _FakeMqtt()
        mqtt_client.client = fake
        self.assertFalse(mqtt_client.publish_event(
            self.CAMERA_ID,
            {"event_type": "object_removed", "bbox": [0, 0, 1, 1]},
        ))
        self.assertEqual(fake.published, [])

    def test_bbox_uses_camera_code_topic_metadata_and_real_confidence(self):
        cfg = yaml.safe_load((ROOT / "configs/mqtt_config.yaml").read_text())
        mqtt_client = MqttAodClient(cfg)
        mqtt_client.set_camera_registry([self._raw_camera()])
        fake = _FakeMqtt()
        mqtt_client.client = fake

        mqtt_client.publish_bbox(self.CAMERA_ID, [{
            "object_id": "car-7",
            "class_name": "car",
            "confidence": 0.7346,
            "bbox": [0.1, 0.2, 0.3, 0.4],
            "state": "TRACKED_VEHICLE",
        }], frame_timestamp=123.5)

        topic, raw_payload, qos = fake.published[-1]
        payload = json.loads(raw_payload)
        self.assertEqual(topic, "smart_vms/ai/bbox/CAM_KCN_GATE_01")
        self.assertEqual(qos, 0)
        self.assertEqual(payload["camera_id"], self.CAMERA_ID)
        self.assertEqual(payload["camera_code"], "CAM_KCN_GATE_01")
        self.assertEqual(payload["camera_name"], "Camera Cổng Chính KCN")
        self.assertEqual(payload["ai_modules"], "LITTERING_DETECTION")
        self.assertEqual(payload["detections"][0]["confidence"], 0.735)


class VehicleTrackFilterTests(unittest.TestCase):
    def setUp(self):
        self.filter = VehicleTrackFilter({
            "min_detector_confidence": 0.45,
            "class_min_confidence": {"car": 0.55},
            "min_detector_hits": 2,
            "confirmation_window_seconds": 1.0,
            "max_detector_gap_seconds": 1.5,
            "min_area_ratio": 0.001,
            "class_min_aspect_ratio": {"car": 0.75},
        }, iou)

    @staticmethod
    def _car(confidence=0.80, bbox=(100, 100, 300, 220)):
        return {
            "object_id": "car-1", "class_name": "car",
            "confidence": confidence, "bbox": bbox,
        }

    def test_requires_two_detector_hits_then_keeps_nvdcf_prediction(self):
        self.assertEqual(
            self.filter.update("cam", [self._car()], 1920, 1080, now=0), {}
        )
        confirmed = self.filter.update(
            "cam", [self._car(bbox=(104, 100, 304, 220))], 1920, 1080, now=.2
        )
        self.assertIn("car-1", confirmed)
        predicted = self.filter.update(
            "cam", [self._car(-.1, (108, 100, 308, 220))], 1920, 1080, now=.4
        )
        self.assertIn("car-1", predicted)
        self.assertEqual(
            self.filter.update("cam", [], 1920, 1080, now=2.0), {}
        )

    def test_rejects_low_confidence_or_person_shaped_car(self):
        self.assertEqual(
            self.filter.update("cam", [self._car(.40)], 1920, 1080, now=0), {}
        )
        narrow = self._car(.90, (100, 100, 140, 300))
        self.assertEqual(
            self.filter.update("cam", [narrow], 1920, 1080, now=.2), {}
        )


if __name__ == "__main__":
    unittest.main()
