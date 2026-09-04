"""Gọi MQTT 1 lần — list camera zone (config trong file)."""
from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

# Host may lack pip/paho; reuse a backend vendored package when this file is
# copied to a shallower directory such as /app inside the AI container.
for _parent in Path(__file__).resolve().parents:
    _vendor = _parent / "backend" / ".python_vendor"
    if _vendor.is_dir():
        sys.path.insert(0, str(_vendor))
        break

MQTT = SimpleNamespace(
    broker="192.168.1.200",
    port=18648,
    username="",
    password="",
    qos=1,
    zone_id=42,
    camera_topic="smart_vms/cameras/zone/{zone_id}",
    camera_getdevice_topic="smart_vms/cameras/getdevice/zone/{zone_id}",
)


def fetch_zone_cameras(timeout: float = 5.0):
    try:
        import paho.mqtt.client as mqtt
    except ModuleNotFoundError:
        raise SystemExit(
            "Thiếu paho-mqtt. Cài: pip install paho-mqtt "
            "hoặc chạy trong container vms_dev_backend."
        ) from None

    cfg = MQTT
    zone_id = int(cfg.zone_id)
    sub = cfg.camera_topic.format(zone_id=zone_id)
    get = cfg.camera_getdevice_topic.format(zone_id=zone_id)
    got = {}

    def on_message(client, userdata, msg):
        got["topic"] = msg.topic
        got["payload"] = json.loads(msg.payload.decode())
        client.disconnect()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    if str(cfg.username or "").strip():
        client.username_pw_set(cfg.username, cfg.password)
    client.on_message = on_message
    client.connect(cfg.broker, int(cfg.port), 60)
    client.subscribe(sub, qos=int(cfg.qos))
    client.publish(get, json.dumps({"zone_id": zone_id}), qos=int(cfg.qos))
    client.loop_start()
    deadline = time.time() + timeout
    while time.time() < deadline and "payload" not in got:
        time.sleep(0.1)
    client.loop_stop()
    if "payload" not in got:
        raise TimeoutError(f"no message on {sub}")
    return got


class TestMqttZoneCameras(unittest.TestCase):
    def test_zone_cameras(self):
        cfg = MQTT
        zone_id = int(cfg.zone_id)
        msg = fetch_zone_cameras()
        payload = msg["payload"]
        print(f"\nbroker={cfg.broker}:{cfg.port}")
        print(f"subscribe={cfg.camera_topic.format(zone_id=zone_id)}")
        print(f"publish={cfg.camera_getdevice_topic.format(zone_id=zone_id)}")
        print(f"topic={msg.get('topic')}")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        self.assertIn("cameras", payload)


if __name__ == "__main__":
    unittest.main()
