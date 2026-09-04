"""
Entry point: khởi tạo config, MQTT client, rule engine theo từng camera,
DeepStream pipeline, và chạy vòng lặp chính.

Chạy:
    python -m src.main --app-config configs/app_config.yaml --mqtt-config configs/mqtt_config.yaml
"""
import argparse
import logging
import queue
import threading
import time
import cv2

# Giới hạn số luồng của OpenCV (ví dụ MOG2) để không dùng hết CPU (mặc định bằng số core vật lý có thể lên tới 100+ cores)
cv2.setNumThreads(2)

from src.utils.config_loader import ConfigLoader
from src.detection.mog_static_detector import MogStaticObjectDetector
from src.detection.optional_labeling import build_labeler
from src.detection.target_verifier import build_target_gate, build_target_verifier
from src.detection.open_vocab_object_detector import build_object_detector
from src.logic.abandonment_rules import AbandonmentRuleEngine
from src.mqtt.mqtt_client import MqttAodClient
from src.mqtt.camera_contract import camera_pipeline_signature, normalize_web_cameras
from src.mqtt.cmd_handlers import build_cmd_handlers
from src.pipeline.deepstream_pipeline import DeepStreamAodPipeline
from src.pipeline.probe_callbacks import FrameHandler


def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main():
    parser = argparse.ArgumentParser(description="AOD Engine - DeepStream + MOG + MQTT")
    parser.add_argument("--app-config", default="configs/app_config.yaml")
    parser.add_argument("--mqtt-config", default="configs/mqtt_config.yaml")
    args = parser.parse_args()

    app_cfg_loader = ConfigLoader(args.app_config, watch=True)
    mqtt_cfg_loader = ConfigLoader(args.mqtt_config, watch=False)

    setup_logging(app_cfg_loader.get("system.log_level", "INFO"))
    logger = logging.getLogger("aod.main")

    app_cfg = app_cfg_loader.data
    mqtt_cfg = mqtt_cfg_loader.data
    broker_cfg = mqtt_cfg.get("mqtt", mqtt_cfg.get("broker", {}))
    ai_module = str(
        broker_cfg.get("ai_module")
        or broker_cfg.get("module_ai")
        or "LITTERING_DETECTION"
    ).upper()
    labeler = build_labeler(app_cfg.get("labeling", {}))
    target_cfg = app_cfg.get("target_filter", {})
    target_verifier = (
        build_target_verifier(target_cfg) if target_cfg.get("enabled", False) else None
    )
    object_detector = build_object_detector(app_cfg.get("object_detection", {}))

    camera_config_queue = queue.Queue()
    camera_config_lock = threading.Lock()
    last_camera_signature = None

    def on_camera_config_received(cameras):
        nonlocal last_camera_signature
        normalized = normalize_web_cameras(cameras, ai_module)
        signature = camera_pipeline_signature(normalized)
        with camera_config_lock:
            if signature == last_camera_signature:
                logger.debug("[AOD] Bỏ qua camera config trùng lặp từ web")
                return
            last_camera_signature = signature
        camera_config_queue.put(normalized)

    mqtt_client = MqttAodClient(mqtt_cfg, None, on_camera_config_received)
    mqtt_client.connect()

    logger.info("[AOD] Khởi động AI Engine. Đang chờ cấu hình camera từ MQTT...")
    
    current_pipeline = None
    pipeline_thread = None
    stop_event = threading.Event()

    def run_pipeline_thread(cameras):
        nonlocal current_pipeline
        # Khởi tạo rule engine với danh sách camera mới
        mog_detector = MogStaticObjectDetector(app_cfg["mog"], cameras)
        rule_engines = {}
        frame_size_by_camera = {}
        for cam in cameras:
            target_gate = build_target_gate(target_cfg, target_verifier)
            rule_engines[cam["camera_id"]] = AbandonmentRuleEngine(
                app_cfg,
                mog_detector,
                labeler,
                target_gate=target_gate,
                object_detector=object_detector,
                cameras_cfg=cameras,
            )
            frame_size_by_camera[cam["camera_id"]] = (1920, 1080)
            
        cmd_handlers = build_cmd_handlers(app_cfg_loader, rule_engines)
        mqtt_client.set_cmd_handlers(cmd_handlers)

        frame_handler = FrameHandler(rule_engines, mqtt_client, frame_size_by_camera)

        current_pipeline = DeepStreamAodPipeline(
            cameras=cameras,
            pgie_config_path=app_cfg["detection"]["pgie_config_path"],
            tracker_config_path=app_cfg["detection"]["tracker_config_path"],
            on_batch_meta_cb=frame_handler,
        )

        logger.info(f"[AOD] Bắt đầu chạy pipeline với {len(cameras)} camera...")
        try:
            current_pipeline.run()
        except Exception as e:
            logger.exception(f"[Pipeline] Lỗi khi chạy pipeline: {e}")
        logger.info("[AOD] Pipeline đã dừng.")

    try:
        while not stop_event.is_set():
            try:
                new_cameras = camera_config_queue.get(timeout=1.0)
                
                # normalize_web_cameras đã lọc ONLINE + đúng AI module và
                # giữ camera_id UUID tách biệt với camera_code dùng cho MQTT.
                active_cameras = list(new_cameras)

                logger.info(f"[AOD] Nhận được cấu hình từ web, kích hoạt {len(active_cameras)} camera (đã lọc).")
                if len(active_cameras) > 0:
                    logger.info(
                        "[AOD] Camera active: %s",
                        ", ".join(str(camera["camera_id"]) for camera in active_cameras),
                    )
                
                # Dừng pipeline cũ nếu đang chạy
                if current_pipeline is not None and current_pipeline.loop is not None:
                    logger.info("[AOD] Yêu cầu dừng pipeline cũ...")
                    if current_pipeline.loop.is_running():
                        current_pipeline.loop.quit()
                        
                if pipeline_thread and pipeline_thread.is_alive():
                    pipeline_thread.join(timeout=15)
                    
                if not active_cameras:
                    logger.info("[AOD] Cấu hình không có camera nào active, chờ tiếp...")
                    continue
                    
                pipeline_thread = threading.Thread(target=run_pipeline_thread, args=(active_cameras,))
                pipeline_thread.daemon = True
                pipeline_thread.start()
                
            except queue.Empty:
                pass

    except KeyboardInterrupt:
        logger.info("[AOD] Received KeyboardInterrupt, shutting down...")
        stop_event.set()
        if current_pipeline and current_pipeline.loop and current_pipeline.loop.is_running():
            current_pipeline.loop.quit()
        if pipeline_thread and pipeline_thread.is_alive():
            pipeline_thread.join(timeout=5)
    finally:
        mqtt_client.disconnect()
        app_cfg_loader.stop()


if __name__ == "__main__":
    main()
