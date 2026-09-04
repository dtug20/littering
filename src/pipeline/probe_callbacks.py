"""
Callback được gọi từ probe của DeepStreamAodPipeline mỗi khi có metadata
frame mới. Đây là nơi nối detection -> rule engine -> MQTT publish.

v3: Tối ưu tần suất pull frame từ GPU (5-7 fps thay vì 3fps), vì kiến trúc
Person-Centric cần frame liên tục hơn để theo dõi trajectory & capture
reference snapshot chính xác.
"""
import logging
import numpy as np
import cv2
try:
    import pyds
except ImportError:  # pragma: no cover - DeepStream runtime only
    pyds = None

from src.utils import geometry

logger = logging.getLogger("aod.probe")


class FrameHandler:
    def __init__(self, rule_engines: dict, mqtt_client, frame_size_by_camera: dict):
        """
        rule_engines: dict[camera_id] -> AbandonmentRuleEngine
        frame_size_by_camera: dict[camera_id] -> (width, height)
        """
        self.rule_engines = rule_engines
        self.mqtt_client = mqtt_client
        self.frame_size_by_camera = frame_size_by_camera

    @staticmethod
    def _should_publish_detected_object(engine, camera_id, detected, dynamic_boxes):
        """Keep live semantic boxes visible without relaxing alert rules."""
        if not detected.get("publishable", True):
            return False
        if any(
            geometry.intersection_over_box(detected["bbox"], dynamic_bbox)
            > engine.object_max_dynamic_overlap
            for dynamic_bbox in dynamic_boxes
        ):
            # Track candidate qua giai đoạn người đặt đồ nhưng không vẽ
            # nhãn rác lên người/xe. Box chỉ xuất hiện khi vùng đã thoáng.
            return False
        track = engine.state_tracker.tracks.get(detected["object_id"])
        if (
            track is None
            or track.camera_id != camera_id
            or track.state == "IGNORED"
        ):
            return False
        # Liveview cần thấy túi/rác đã được detector semantic xác nhận trong
        # polygon ngay cả khi state machine còn đang đếm dwell/chờ target gate.
        # Event object_abandoned vẫn chỉ publish ở ObjectStateTracker.
        return True

    def __call__(self, camera_id, frame_num, detections, source_id, gst_buffer, batch_id):
        engine = self.rule_engines.get(camera_id)
        if engine is None:
            return

        try:
            # Throttling: Pull frame từ GPU 5-7 fps
            # Với pipeline ~30fps, lấy 1/5 frame → ~6fps
            frame_bgr_small = None
            if frame_num % 5 == 0:
                frame_bgr_small = self._extract_frame_bgr_small(gst_buffer, batch_id)
        except Exception as e:
            import logging
            logging.getLogger("aod.probe").error(f"[AOD] Lỗi trích xuất BGR small: {e}")
            frame_bgr_small = None

        # Lấy kích thước ảnh gốc của camera (mặc định 1920x1080)
        w, h = self.frame_size_by_camera.get(camera_id, (1920, 1080))

        # Probe nằm sau nhánh detector đã scale; metadata bbox cũng dùng kích
        # thước caps của nhánh này nên phải scale về frame gốc để rule engine
        # tính diện tích và khoảng cách theo cùng một hệ tọa độ.
        if frame_bgr_small is not None:
            processing_h, processing_w = frame_bgr_small.shape[:2]
        else:
            processing_w, processing_h = 960, 540
        scale_x = w / float(processing_w)
        scale_y = h / float(processing_h)

        relevant_detections = []
        accepted_classes = (
            set(engine.vehicle_classes)
            | {engine.person_class}
            | set(engine.object_classes)
        )
        for d in detections:
            if d["class_name"] in accepted_classes:
                d_copy = d.copy()
                bx1, by1, bx2, by2 = d["bbox"]
                d_copy["bbox"] = (bx1 * scale_x, by1 * scale_y, bx2 * scale_x, by2 * scale_y)
                relevant_detections.append(d_copy)

        events, tracked_objects, person_tracks, vehicle_tracks = engine.process_frame(
            camera_id, w, h, relevant_detections, frame_bgr_small
        )

        # publish realtime bbox (người + vật thể đang theo dõi)
        bbox_payload_objects = []

        # Chỉ publish track còn được semantic detector quan sát/cached. Không
        # publish toàn bộ state history vì đó là nguyên nhân box ảo tồn tại.
        dynamic_boxes = list(person_tracks.values()) + [
            item["bbox"] for item in vehicle_tracks.values()
        ]
        for detected in tracked_objects:
            if not self._should_publish_detected_object(
                engine, camera_id, detected, dynamic_boxes
            ):
                continue
            track = engine.state_tracker.tracks[detected["object_id"]]
            x1, y1, x2, y2 = detected["bbox"]
            bbox_payload_objects.append({
                "object_id": track.object_id,
                "class_name": "object",
                "bbox": [x1/w, y1/h, x2/w, y2/h],
                "state": track.state,
                "label": detected.get("label") or track.label,
                "confidence": detected.get("label_confidence", track.label_confidence),
            })
        for pid, pbox in person_tracks.items():
            x1, y1, x2, y2 = pbox
            bbox_payload_objects.append({
                "object_id": pid, "class_name": "person", "bbox": [x1/w, y1/h, x2/w, y2/h], "state": "",
            })
        for vehicle_id, vehicle in vehicle_tracks.items():
            x1, y1, x2, y2 = vehicle["bbox"]
            bbox_payload_objects.append({
                "object_id": vehicle_id,
                "class_name": vehicle["class_name"],
                "bbox": [x1/w, y1/h, x2/w, y2/h],
                "state": "TRACKED_VEHICLE",
                "confidence": vehicle.get("confidence", 0.0),
            })

        if self.mqtt_client:
            self.mqtt_client.publish_bbox(camera_id, bbox_payload_objects)
            for event in events:
                ex1, ey1, ex2, ey2 = event["bbox"]
                event["bbox"] = [ex1/w, ey1/h, ex2/w, ey2/h]
                self.mqtt_client.publish_event(camera_id, event)
                logger.info(
                    f"[AOD] camera={camera_id} event={event['event_type']} "
                    f"object_id={event['object_id']} owner={event.get('owner_track_id')} "
                    f"vehicle={event.get('related_vehicle_id')} "
                    f"incident={event.get('incident_type')} label={event.get('label', '')}"
                )

    @staticmethod
    def _extract_frame_bgr_small(gst_buffer, batch_id):
        if pyds is None:
            logger.error("[AOD] pyds chưa được cài trong runtime hiện tại")
            return None
        try:
            n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), batch_id)
            frame_rgba = np.array(n_frame, copy=True, order="C")
            return cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
        except Exception:
            logger.exception("[AOD] Không lấy được frame từ NvBufSurface")
            return None
