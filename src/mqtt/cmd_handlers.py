"""Handlers cho các lệnh (cmd) mà web gửi xuống qua MQTT topic subscribe_cmd."""
import logging

logger = logging.getLogger("aod.cmd")


def build_cmd_handlers(app_config_loader, rule_engines: dict):
    """
    rule_engines: dict[camera_id] -> AbandonmentRuleEngine
    Trả về dict[cmd_name] -> callable(camera_id, payload) -> data(dict)
    """

    def handle_get_camera(camera_id, payload):
        cameras = app_config_loader.get("cameras", [])
        if camera_id and camera_id != "unknown":
            cam = next((c for c in cameras if c["camera_id"] == camera_id), None)
            return {"camera": cam}
        return {"cameras": cameras}

    def handle_update_roi(camera_id, payload):
        """payload: {cmd:'update_roi', camera_id, roi_polygons:[{name, points}]}"""
        engine = rule_engines.get(camera_id)
        if engine is None:
            raise ValueError(f"Camera '{camera_id}' không tồn tại")
        raw_zones = payload.get("roi_polygons")
        if raw_zones is None:
            raw_zones = payload.get("zones")
        if raw_zones is None:
            raw_zones = payload.get("ai_zones", [])
        new_roi = []
        for index, zone in enumerate(raw_zones or []):
            if not isinstance(zone, dict) or not zone.get("is_active", True):
                continue
            modules = str(zone.get("ai_modules", "")).upper()
            if modules and not any(
                name in modules for name in (
                    "LITTERING_DETECTION", "ABANDONED_OBJECT", "ABANDONED_DETECTION"
                )
            ):
                continue
            points = zone.get("points", [])
            if len(points) < 3:
                continue
            new_roi.append({
                "name": zone.get("name", zone.get("id", f"roi_{index}")),
                "points": points,
            })
        engine.update_roi(camera_id, new_roi)
        logger.info(f"[CMD] Cập nhật ROI cho {camera_id}: {len(new_roi)} vùng")
        return {"updated": True, "roi_polygons": new_roi}

    def handle_ack_event(camera_id, payload):
        """Web xác nhận đã xử lý event -> dùng để log / clear cảnh báo trên UI."""
        event_id = payload.get("event_id")
        logger.info(f"[CMD] ACK event_id={event_id} cho camera={camera_id}")
        return {"acked": event_id}

    return {
        "get_camera": handle_get_camera,
        "update_roi": handle_update_roi,
        "ack_event": handle_ack_event,
    }
