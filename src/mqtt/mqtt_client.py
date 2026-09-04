"""
MQTT client tổng quát cho AOD engine.

- Kết nối theo broker host/port cấu hình trong mqtt_config.yaml (sửa YAML,
  không cần build lại code / restart pipeline DeepStream).
- publish_bbox(camera_id, objects) / publish_event(camera_id, event) render
  payload theo template cấu hình sẵn (payload_templates trong YAML) - đổi tên
  field / cấu trúc JSON chỉ cần sửa YAML.
- subscribe_cmd: web gửi lệnh xuống (get_camera, update_roi, ack_event, ...),
  engine dispatch tới handler tương ứng và trả kết quả qua publish_cmd_response.
"""
import json
import logging
import time
import copy
import threading
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

from src.mqtt.camera_contract import module_enabled

logger = logging.getLogger("aod.mqtt")


def _render_template(node, context: dict):
    """Đệ quy .format(**context) cho mọi string leaf trong dict/list template."""
    if isinstance(node, dict):
        return {k: _render_template(v, context) for k, v in node.items()}
    if isinstance(node, list):
        return [_render_template(v, context) for v in node]
    if isinstance(node, str):
        try:
            return node.format(**context)
        except (KeyError, IndexError):
            return node
    return node


class MqttAodClient:
    def __init__(self, mqtt_cfg: dict, cmd_handlers: dict = None, on_camera_config_received=None):
        self.cfg = mqtt_cfg
        self.cmd_handlers = cmd_handlers or {}
        self.on_camera_config_received = on_camera_config_received
        self._latest_zone_payloads = {}
        self._handler_lock = threading.RLock()
        self._camera_by_id = {}
        self._camera_id_by_code = {}
        
        # Hỗ trợ cả config cũ ('broker') và cấu trúc mới ('mqtt')
        if "mqtt" in mqtt_cfg:
            broker = mqtt_cfg["mqtt"]
            self._host = broker.get("broker", "127.0.0.1")
            self._port = broker.get("port", 1883)
            self._company_id = broker.get("company_id", 1)
            self._zone_id = int(broker.get("zone_id", 0))
            self._ai_module = str(
                broker.get("ai_module") or broker.get("module_ai") or "LITTERING_DETECTION"
            ).upper()
            
            topics = {
                "publish_bbox": broker.get("bbox_topic", "smart_vms/ai/bbox/{camera_code}"),
                "publish_event": broker.get("events_topic", "smart_vms/ai_events/{ai_module}"),
                "subscribe_cmd": broker.get("zones_topic", "smart_vms/cameras/{camera_code}/ai_zones"),
                "publish_camera_list": broker.get("camera_topic", "smart_vms/cameras/zone/{zone_id}"),
                "request_camera_list": broker.get("camera_request_topic", "smart_vms/cameras/getdevice/zone/{zone_id}"),
                "publish_cmd_response": "smart_vms/cameras/{camera_code}/cmd_response"
            }
            if "topics" not in self.cfg:
                self.cfg["topics"] = topics
            else:
                self.cfg["topics"].update(topics)
        else:
            broker = mqtt_cfg["broker"]
            self._host = broker["host"]
            self._port = broker.get("port", 1883)
            self._company_id = 1
            self._zone_id = int(broker.get("zone_id", 0))
            self._ai_module = str(broker.get("ai_module", "LITTERING_DETECTION")).upper()

        self._event_source_types = {
            str(value) for value in self.cfg.get("event_source_types", ["object_abandoned"])
        }

        try:
            self.client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=broker.get("client_id", "aod_engine"),
            )
        except AttributeError:  # paho-mqtt < 2
            self.client = mqtt.Client(
                client_id=broker.get("client_id", "aod_engine"), clean_session=True
            )
        if broker.get("username"):
            self.client.username_pw_set(broker.get("username"), broker.get("password", ""))
        if broker.get("tls_enable"):
            self.client.tls_set()

        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

        self._reconnect_min = broker.get("reconnect_min_delay", 1)
        self._reconnect_max = broker.get("reconnect_max_delay", 30)
        self.client.reconnect_delay_set(min_delay=self._reconnect_min,
                                         max_delay=self._reconnect_max)

        self._keepalive = broker.get("keepalive", 60)

    def _format_topic(self, topic_tmpl: str, camera_id: str) -> str:
        camera = self._camera_by_id.get(str(camera_id), {})
        camera_code = str(camera.get("camera_code") or camera.get("code") or camera_id)
        return topic_tmpl.format(
            camera_id=camera_id,
            camera_code=camera_code,
            company_id=self._company_id,
            zone_id=self._zone_id,
            ai_module=self._ai_module,
        )

    def _camera_context(self, camera_id: str) -> dict:
        camera = self._camera_by_id.get(str(camera_id), {})
        code = str(camera.get("camera_code") or camera.get("code") or camera_id)
        name = str(camera.get("camera_name") or camera.get("name") or code)
        return {
            "camera_id": str(camera.get("camera_id") or camera.get("id") or camera_id),
            "camera_code": code,
            "camera_name": name,
            "ai_module": self._ai_module,
        }

    def set_camera_registry(self, cameras: list):
        """Index raw web camera metadata by UUID and by camera code."""
        by_id, by_code = {}, {}
        for raw in cameras or []:
            if not isinstance(raw, dict):
                continue
            code = str(raw.get("camera_code") or raw.get("code") or "").strip()
            camera_id = str(raw.get("camera_id") or raw.get("id") or code).strip()
            if not camera_id or not code:
                continue
            camera = dict(raw)
            camera.update({
                "camera_id": camera_id,
                "camera_code": code,
                "camera_name": str(raw.get("camera_name") or raw.get("name") or code),
            })
            by_id[camera_id] = camera
            by_code[code] = camera_id
        with self._handler_lock:
            self._camera_by_id = by_id
            self._camera_id_by_code = by_code
            self._latest_zone_payloads = {
                by_code.get(key, key): value
                for key, value in self._latest_zone_payloads.items()
            }

    # ------------------------------------------------------------------
    def connect(self):
        logger.info(f"[MQTT] Connecting to {self._host}:{self._port} ...")
        self.client.connect(self._host, self._port, self._keepalive)
        self.client.loop_start()

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            logger.info("[MQTT] Connected.")
            cmd_topic_tmpl = self.cfg["topics"]["subscribe_cmd"]
            # subscribe wildcard cho mọi camera
            wildcard_topic = self._format_topic(cmd_topic_tmpl, "+")
            self.client.subscribe(wildcard_topic)
            logger.info(f"[MQTT] Subscribed: {wildcard_topic}")
            
            # subscribe vào topic nhận cấu hình camera từ web
            camera_topic = self._format_topic(self.cfg["topics"]["publish_camera_list"], "")
            self.client.subscribe(camera_topic)
            logger.info(f"[MQTT] Subscribed to camera config: {camera_topic}")
            request_tmpl = self.cfg["topics"].get("request_camera_list")
            if request_tmpl:
                request_topic = self._format_topic(request_tmpl, "")
                qos = self.cfg.get("qos", {}).get("request_camera_list", 1)
                self.client.publish(
                    request_topic, json.dumps({"zone_id": self._zone_id}), qos=qos
                )
                logger.info(
                    "[MQTT] Requested camera list: %s (zone_id=%s)",
                    request_topic, self._zone_id,
                )
        else:
            logger.error(f"[MQTT] Connect failed, rc={rc}")

    def _on_disconnect(
        self, client, userdata, disconnect_flags=None, reason_code=None, properties=None
    ):
        # VERSION1 passes rc as the third argument; VERSION2 passes flags then rc.
        rc = disconnect_flags if reason_code is None else reason_code
        logger.warning(f"[MQTT] Disconnected (rc={rc}), auto-reconnect sẽ tự xử lý.")

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except json.JSONDecodeError:
            logger.error(f"[MQTT] Payload không phải JSON hợp lệ: {msg.payload}")
            return

        expected_camera_topic = self._format_topic(self.cfg["topics"]["publish_camera_list"], "")
        if msg.topic == expected_camera_topic:
            if self.on_camera_config_received:
                # Payload có thể là mảng trực tiếp [...] hoặc object {"cameras": [...]}
                cameras = payload if isinstance(payload, list) else payload.get("cameras", [])
                self.set_camera_registry(cameras)
                self.on_camera_config_received(cameras)
            return

        camera_code = self._extract_camera_id(msg.topic, self.cfg["topics"]["subscribe_cmd"])
        camera_id = self._camera_id_by_code.get(camera_code, camera_code)
        camera = self._camera_by_id.get(camera_id)
        if camera is not None and not module_enabled(
            camera.get("ai_modules"), self._ai_module
        ):
            return
        cmd = payload.get("cmd")
        # Topic zones của web gửi trực tiếp {zones:[...]} thay vì command.
        # Chuẩn hóa về handler update_roi để cả hai hợp đồng đều hoạt động.
        if cmd is None and (
            "zones" in payload or "roi_polygons" in payload or "ai_zones" in payload
        ):
            cmd = "update_roi"
        if cmd == "update_roi":
            with self._handler_lock:
                self._latest_zone_payloads[camera_id] = copy.deepcopy(payload)
        with self._handler_lock:
            handler = self.cmd_handlers.get(cmd)
        request_id = payload.get("request_id")

        if handler is None:
            if cmd == "update_roi":
                logger.debug(
                    "[MQTT] Đã giữ ROI mới nhất cho camera=%s; chờ pipeline sẵn sàng",
                    camera_id,
                )
                return
            logger.warning(f"[MQTT] Không có handler cho cmd='{cmd}'")
            self.publish_cmd_response(camera_id, request_id, status="unknown_cmd", data={})
            return

        try:
            data = handler(camera_id, payload)
            self.publish_cmd_response(camera_id, request_id, status="ok", data=data)
        except Exception as e:
            logger.exception(f"[MQTT] Lỗi khi xử lý cmd='{cmd}'")
            self.publish_cmd_response(camera_id, request_id, status="error", data={"error": str(e)})

    def set_cmd_handlers(self, cmd_handlers: dict):
        """Install handlers and replay the latest retained ROI per camera."""
        with self._handler_lock:
            self.cmd_handlers = cmd_handlers or {}
            pending = copy.deepcopy(self._latest_zone_payloads)
            update_handler = self.cmd_handlers.get("update_roi")
        if update_handler is None:
            return
        for camera_id, payload in pending.items():
            try:
                update_handler(camera_id, payload)
                logger.info("[MQTT] Đã áp lại retained ROI cho camera=%s", camera_id)
            except Exception:
                # Payload camera khác pipeline hiện tại là bình thường; giữ nó
                # để áp lại nếu camera đó được kích hoạt ở lần cấu hình sau.
                logger.debug(
                    "[MQTT] Chưa thể áp retained ROI cho camera=%s", camera_id,
                    exc_info=True,
                )

    @staticmethod
    def _extract_camera_id(topic: str, topic_template: str) -> str:
        placeholder = "{camera_code}" if "{camera_code}" in topic_template else "{camera_id}"
        prefix, _, suffix = topic_template.partition(placeholder)
        if topic.startswith(prefix) and topic.endswith(suffix):
            return topic[len(prefix): len(topic) - len(suffix)]
        return "unknown"

    # ------------------------------------------------------------------
    def publish_bbox(self, camera_id: str, objects: list, frame_timestamp=None):
        """objects: list[dict] {object_id, class_name, bbox=(x1,y1,x2,y2), state}"""
        template = self.cfg["payload_templates"]["bbox"]
        object_template = template["detections"][0]

        rendered_objects = []
        for obj in objects:
            c_name = str(obj["class_name"]).upper()
            state = obj.get("state", "")
            
            if c_name == "PERSON":
                color = "#0000FF"
                label = f"person #{obj['object_id']}"
            elif state == "TRACKED_VEHICLE":
                color = "#00BFFF"
                label = f"{str(obj['class_name']).lower()} #{obj['object_id']}"
            else:
                if state == "ABANDONED":
                    color = "#FF0000"
                    label = obj.get("label") or "trash"
                elif state == "BACKGROUND":
                    color = "#808080"
                    label = obj.get("label") or "detected object"
                elif state == "STATIC_NO_OWNER":
                    color = "#FFA500"
                    label = obj.get("label") or "unowned obj"
                elif state == "STATIC_WITH_OWNER":
                    color = "#00FF00"
                    label = obj.get("label") or "owned obj"
                else:
                    color = "#808080"
                    label = obj.get("label") or "confirming obj"

            ctx = {
                "object_id": obj["object_id"],
                "class_name": c_name,
                "x1": obj["bbox"][0], "y1": obj["bbox"][1],
                "x2": obj["bbox"][2], "y2": obj["bbox"][3],
                "state": state,
                "label": label,
                "color": color,
                "confidence": float(obj.get("confidence", 0.99)),
            }
            rendered = _render_template(copy.deepcopy(object_template), ctx)
            rendered["bbox"] = [
                round(float(obj["bbox"][0]), 4), round(float(obj["bbox"][1]), 4),
                round(float(obj["bbox"][2]), 4), round(float(obj["bbox"][3]), 4)
            ]
            rendered["confidence"] = round(float(ctx["confidence"]), 3)
            rendered_objects.append(rendered)

        ctx = self._camera_context(camera_id)
        ctx["frame_timestamp"] = float(frame_timestamp) if frame_timestamp else time.time()
        payload = _render_template(copy.deepcopy(template), ctx)
        payload["detections"] = rendered_objects
        payload["timestamp"] = ctx["frame_timestamp"] # Ép kiểu float thay vì string do _render_template trả về

        topic = self._format_topic(self.cfg["topics"]["publish_bbox"], camera_id)
        qos = self.cfg.get("qos", {}).get("publish_bbox", 0)
        
        payload_str = json.dumps(payload)
        logger.debug(f"[MQTT] Publishing to {topic}: {payload_str}")
        self.client.publish(topic, payload_str, qos=qos)

    def publish_event(self, camera_id: str, event: dict, snapshot_path: str = ""):
        if str(event.get("event_type", "")) not in self._event_source_types:
            return False
        template = self.cfg["payload_templates"]["event"]
        camera_ctx = self._camera_context(camera_id)
        triggered_at = float(event.get("triggered_at", time.time()))
        event_time = datetime.fromtimestamp(triggered_at, timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )
        snapshot_url = str(event.get("snapshot_url") or snapshot_path or "")
        ctx = {
            "event_id": event.get("event_id", ""),
            **camera_ctx,
            "event_type": "LITTERING",
            "event_time": event_time,
            "snapshot_url": snapshot_url,
            "incident_type": event.get("incident_type", ""),
            "object_id": event.get("object_id", ""),
            "owner_track_id": event.get("owner_track_id") or "",
            "related_vehicle_id": event.get("related_vehicle_id") or "",
            "related_vehicle_class": event.get("related_vehicle_class") or "",
            "person_dwell_seconds": event.get("person_dwell_seconds", 0.0),
            "vehicle_dwell_seconds": event.get("vehicle_dwell_seconds", 0.0),
            "x1": event["bbox"][0], "y1": event["bbox"][1],
            "x2": event["bbox"][2], "y2": event["bbox"][3],
            "label": event.get("label", "unknown_object"),
            "label_confidence": event.get("label_confidence", 0.0),
            "first_seen_static_at": event.get("first_seen_static_at") or "",
            "triggered_at": event.get("triggered_at", time.time()),
            "snapshot_path": snapshot_path,
        }
        payload = _render_template(copy.deepcopy(template), ctx)
        topic = self._format_topic(self.cfg["topics"]["publish_event"], camera_id)
        qos = self.cfg.get("qos", {}).get("publish_event", 1)
        self.client.publish(topic, json.dumps(payload), qos=qos)
        logger.debug(f"[MQTT] Published event '{event.get('event_type')}' -> {topic}")
        return True

    def publish_camera_list(self, cameras: list):
        # AI engine giờ chỉ nhận cấu hình camera từ web, không publish.
        pass

    def publish_cmd_response(self, camera_id, request_id, status, data):
        template = self.cfg["payload_templates"]["cmd_response"]
        ctx = {
            "request_id": request_id or "",
            "camera_id": camera_id,
            "status": status,
            "data": json.dumps(data) if not isinstance(data, str) else data,
        }
        payload = _render_template(copy.deepcopy(template), ctx)
        topic = self._format_topic(self.cfg["topics"]["publish_cmd_response"], camera_id)
        qos = self.cfg.get("qos", {}).get("publish_cmd_response", 1)
        self.client.publish(topic, json.dumps(payload), qos=qos)
