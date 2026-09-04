"""Normalize the zone-camera contract received from the VMS MQTT broker."""
from __future__ import annotations

import json
from typing import Any


def module_enabled(raw: Any, module_name: str) -> bool:
    """Accept module lists, JSON strings and comma-separated legacy values."""
    if isinstance(raw, (list, tuple, set)):
        values = raw
    elif isinstance(raw, str):
        try:
            decoded = json.loads(raw)
            values = decoded if isinstance(decoded, list) else raw.split(",")
        except json.JSONDecodeError:
            values = raw.split(",")
    else:
        return False
    expected = str(module_name).strip().upper()
    return expected in {str(value).strip().upper() for value in values}


def _stream_url(camera: dict, module_name: str) -> str:
    restreams = camera.get("restream_urls") or {}
    if isinstance(restreams, dict):
        expected = str(module_name).strip().upper()
        for key, value in restreams.items():
            if str(key).strip().upper() == expected and str(value or "").strip():
                return str(value).strip()
    for key in ("uri", "rtsp_url", "stream_url", "url", "link", "rtsp", "record_url"):
        value = str(camera.get(key) or "").strip()
        if value:
            return value
    return ""


def normalize_web_cameras(cameras: list, module_name: str) -> list[dict]:
    """Preserve VMS UUID metadata while producing the pipeline camera schema."""
    normalized = []
    for raw in cameras or []:
        if not isinstance(raw, dict) or not module_enabled(raw.get("ai_modules"), module_name):
            continue
        status = str(raw.get("status", "ONLINE")).strip().upper()
        if status not in {"", "ONLINE", "ACTIVE"} or not raw.get("enable", True):
            continue
        camera_code = str(
            raw.get("camera_code") or raw.get("code") or raw.get("camera_id") or raw.get("id") or ""
        ).strip()
        camera_id = str(raw.get("camera_id") or raw.get("id") or camera_code).strip()
        uri = _stream_url(raw, module_name)
        if not camera_id or not camera_code or not uri:
            continue
        camera_name = str(raw.get("camera_name") or raw.get("name") or camera_code).strip()
        zones = raw.get("roi_polygons")
        if zones is None:
            zones = raw.get("ai_zones")
        if zones is None:
            zones = raw.get("zones", [])
        camera = dict(raw)
        camera.update({
            "camera_id": camera_id,
            "camera_code": camera_code,
            "camera_name": camera_name,
            "uri": uri,
            "roi_polygons": zones or [],
            "enable": True,
        })
        normalized.append(camera)
    return normalized


def camera_pipeline_signature(cameras: list) -> tuple:
    """Stable signature of fields that actually require a pipeline rebuild."""
    rows = []
    for camera in cameras or []:
        rows.append((
            str(camera.get("camera_id", "")),
            str(camera.get("camera_code", "")),
            str(camera.get("uri", "")),
            json.dumps(camera.get("roi_polygons", []), sort_keys=True, ensure_ascii=False),
            int(camera.get("baseline_learning_seconds", 60)),
        ))
    return tuple(sorted(rows))
