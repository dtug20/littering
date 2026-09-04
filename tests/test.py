#!/usr/bin/env python3
"""OpenCV + YOLO11: lấy CAM_THANG_MAY qua MQTT, track người trong polygon.

Cài thư viện:
    pip install ultralytics opencv-python paho-mqtt

Chạy:
    python test.py

Có thể ghi đè cấu hình bằng biến môi trường MQTT_HOST, MQTT_PORT,
MQTT_USERNAME, MQTT_PASSWORD, COMPANY_ID, YOLO_MODEL và SHOW_WINDOW.
"""
from __future__ import annotations

import json
import os
import signal
import threading
import time
from typing import Any

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from ultralytics import YOLO


AI_MODULE = "ABANDONED_DETECTION"

MQTT_HOST = os.getenv("MQTT_HOST", "192.168.1.196")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "atin")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "team1@123#")
COMPANY_ID = os.getenv("COMPANY_ID", "1")

CAMERA_TOPIC = f"smart_vms/cameras/company/{COMPANY_ID}"
ZONE_TOPIC = f"smart_vms/cameras/{CAMERA_CODE}/abandoned_object"
BBOX_TOPIC = f"smart_vms/ai/bbox/{CAMERA_CODE}"

MODEL_PATH = os.getenv("YOLO_MODEL", "yolo11n.pt")
CONFIDENCE = float(os.getenv("YOLO_CONF", "0.35"))
SHOW_WINDOW = os.getenv("SHOW_WINDOW", "1").lower() not in {"0", "false", "no"}


def module_enabled(raw: Any) -> bool:
    """Hỗ trợ ai_modules dạng list, chuỗi CSV hoặc chuỗi JSON."""
    if isinstance(raw, list):
        modules = raw
    elif isinstance(raw, str):
        try:
            decoded = json.loads(raw)
            modules = decoded if isinstance(decoded, list) else raw.split(",")
        except json.JSONDecodeError:
            modules = raw.split(",")
    else:
        return False
    return AI_MODULE in {str(item).strip().upper() for item in modules}


def point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon.astype(np.float32), point, False) >= 0


class App:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stream_url = ""
        # Polygon chuẩn hóa 0..1. Nếu chưa nhận zone thì không detect trong vùng nào.
        self.polygons: list[list[tuple[float, float]]] = []
        self.running = True
        self.mqtt_connected = False

        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        except AttributeError:  # paho-mqtt < 2
            self.client = mqtt.Client()
        if MQTT_USERNAME:
            self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)

    def on_connect(self, client, userdata, flags, rc) -> None:
        self.mqtt_connected = rc == 0
        if rc != 0:
            print(f"[MQTT] Kết nối lỗi rc={rc}")
            return
        client.subscribe([(CAMERA_TOPIC, 0), (ZONE_TOPIC, 0)])
        print(f"[MQTT] Đã subscribe {CAMERA_TOPIC} và {ZONE_TOPIC}")

    def on_disconnect(self, client, userdata, rc) -> None:
        self.mqtt_connected = False
        print(f"[MQTT] Mất kết nối rc={rc}")

    def on_message(self, client, userdata, msg) -> None:
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            if msg.topic == CAMERA_TOPIC:
                self.update_camera(payload)
            elif msg.topic == ZONE_TOPIC:
                self.update_zones(payload)
        except Exception as exc:
            print(f"[MQTT] Payload không hợp lệ ({msg.topic}): {exc}")

    def update_camera(self, payload: dict[str, Any]) -> None:
        for camera in payload.get("cameras", []):
            if str(camera.get("code", "")).strip() != CAMERA_CODE:
                continue
            if not module_enabled(camera.get("ai_modules", [])):
                continue

            restream = camera.get("restream_urls") or {}
            url = str(restream.get(AI_MODULE, "")).strip() if isinstance(restream, dict) else ""
            if not url:
                for key in ("link", "rtsp", "rtsp_url", "record_url"):
                    url = str(camera.get(key) or "").strip()
                    if url:
                        break
            if url:
                with self.lock:
                    changed = url != self.stream_url
                    self.stream_url = url
                if changed:
                    print(f"[CAMERA] Đã nhận stream {CAMERA_CODE}: {url}")
            return

    def update_zones(self, payload: dict[str, Any]) -> None:
        polygons: list[list[tuple[float, float]]] = []
        for zone in payload.get("zones", []):
            if not zone.get("is_active", True) or not module_enabled(zone.get("ai_modules", [])):
                continue
            points = zone.get("points", [])
            polygon = [
                (float(p["x"]), float(p["y"]))
                for p in points
                if isinstance(p, dict) and "x" in p and "y" in p
            ]
            if len(polygon) >= 3:
                polygons.append(polygon)
        with self.lock:
            self.polygons = polygons
        print(f"[ZONE] Đã cập nhật {len(polygons)} polygon")

    def publish_boxes(self, detections: list[dict[str, Any]]) -> None:
        if not self.mqtt_connected:
            return
        payload = {
            "camera_code": CAMERA_CODE,
            "ai_modules": AI_MODULE,
            "timestamp": time.time(),
            "detections": detections,
        }
        self.client.publish(BBOX_TOPIC, json.dumps(payload), qos=0)

    def start_mqtt(self) -> None:
        self.client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=60)
        self.client.loop_start()

    def stop(self) -> None:
        self.running = False

    def run(self) -> None:
        self.start_mqtt()
        print(f"[YOLO] Đang load {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        cap: cv2.VideoCapture | None = None
        opened_url = ""

        try:
            while self.running:
                with self.lock:
                    stream_url = self.stream_url
                    polygons_norm = [p.copy() for p in self.polygons]

                if not stream_url:
                    time.sleep(0.2)
                    continue
                if cap is None or stream_url != opened_url:
                    if cap is not None:
                        cap.release()
                    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    opened_url = stream_url
                    if not cap.isOpened():
                        print("[CAMERA] Không mở được stream, thử lại sau 2 giây")
                        cap.release()
                        cap = None
                        time.sleep(2)
                        continue

                ok, frame = cap.read()
                if not ok:
                    print("[CAMERA] Mất frame, đang kết nối lại...")
                    cap.release()
                    cap = None
                    time.sleep(1)
                    continue

                height, width = frame.shape[:2]
                polygons_px = [
                    np.array([(x * width, y * height) for x, y in poly], dtype=np.int32)
                    for poly in polygons_norm
                ]
                for polygon in polygons_px:
                    cv2.polylines(frame, [polygon], True, (0, 255, 255), 2)

                detections: list[dict[str, Any]] = []
                # COCO class 0 = person; persist=True giữ track ID giữa các frame.
                result = model.track(frame, persist=True, classes=[0], conf=CONFIDENCE, verbose=False)[0]
                boxes = result.boxes
                if boxes is not None:
                    for index, xyxy in enumerate(boxes.xyxy.cpu().numpy()):
                        x1, y1, x2, y2 = map(float, xyxy)
                        # Dùng điểm giữa đáy bbox (vị trí chân) để xét người trong vùng.
                        foot = ((x1 + x2) / 2.0, y2)
                        if not polygons_px or not any(point_in_polygon(foot, p) for p in polygons_px):
                            continue
                        confidence = float(boxes.conf[index].item())
                        track_id = int(boxes.id[index].item()) if boxes.id is not None else index
                        detections.append({
                            "id": str(track_id),
                            "cls": "PERSON",
                            "confidence": round(confidence, 3),
                            "bbox": [
                                round(x1 / width, 4), round(y1 / height, 4),
                                round(x2 / width, 4), round(y2 / height, 4),
                            ],
                            "label": f"person #{track_id}",
                            "color": "#0000FF",
                        })
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(frame, f"person #{track_id} {confidence:.2f}",
                                    (int(x1), max(20, int(y1) - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Publish cả mảng rỗng để liveview xóa box cũ.
                self.publish_boxes(detections)

                if SHOW_WINDOW:
                    cv2.imshow(f"{CAMERA_CODE} - {AI_MODULE}", frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            self.client.loop_stop()
            self.client.disconnect()


if __name__ == "__main__":
    app = App()
    signal.signal(signal.SIGINT, lambda *_: app.stop())
    signal.signal(signal.SIGTERM, lambda *_: app.stop())
    app.run()
