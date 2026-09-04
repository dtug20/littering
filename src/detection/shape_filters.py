"""
Bộ lọc "objectness" theo hình học - CLASS-AGNOSTIC, KHÔNG CẦN TRAIN – v3.

Phiên bản v3 bổ sung:
  - Min/max absolute size filter (loại blob quá nhỏ / quá lớn)
  - Person overlap filter (loại blob trùng bbox person đang có)
  - Edge density check (vật thật có cạnh rõ, nhiễu ánh sáng thì mờ)

Giữ nguyên các filter v2: aspect_ratio, solidity, extent.
"""
import cv2
import numpy as np


def compute_shape_score(contour, bbox) -> dict:
    x1, y1, x2, y2 = bbox
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull) if len(hull) >= 3 else area
    solidity = (area / hull_area) if hull_area > 0 else 0.0
    aspect_ratio = w / h
    extent = area / (w * h)
    return {"solidity": solidity, "aspect_ratio": aspect_ratio, "extent": extent, "area": area}


def compute_edge_density(frame_bgr: np.ndarray, bbox: tuple) -> float:
    """Tính mật độ cạnh (edge density) trong vùng bbox.

    Vật thể thật (balo, vali, hộp...) có cạnh rõ ràng. Nhiễu ánh sáng,
    bóng đổ thường mờ, ít cạnh.

    Returns: tỷ lệ pixel cạnh / tổng pixel trong bbox (0.0 - 1.0)
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    crop = frame_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    total_pixels = edges.size
    edge_pixels = cv2.countNonZero(edges)

    return edge_pixels / max(1, total_pixels)


def is_plausible_object(shape_score: dict, cfg: dict) -> bool:
    ar = shape_score["aspect_ratio"]
    min_ar = cfg.get("min_aspect_ratio", 0.15)
    max_ar = cfg.get("max_aspect_ratio", 6.0)
    if ar < min_ar or ar > max_ar:
        return False  # quá dẹt/quá cao -> khả năng cột, bóng đổ dài, vệt sáng

    min_solidity = cfg.get("min_solidity", 0.35)
    if shape_score["solidity"] < min_solidity:
        return False  # hình dạng rời rạc -> khả năng nhiễu (lá cây, ánh sáng, occlusion vỡ)

    min_extent = cfg.get("min_extent", 0.25)
    if shape_score["extent"] < min_extent:
        return False  # vùng quá thưa so với bbox bao quanh

    return True


def is_valid_size(bbox: tuple, frame_w: int, frame_h: int, cfg: dict) -> bool:
    """Kiểm tra kích thước tuyệt đối (px) và tương đối (% frame)."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    area = w * h
    frame_area = frame_w * frame_h

    # Kích thước tối thiểu tuyệt đối (px)
    min_w = cfg.get("min_width_px", 20)
    min_h = cfg.get("min_height_px", 20)
    if w < min_w or h < min_h:
        return False

    # Kích thước tối đa (% frame) – loại blob quá lớn (toàn frame, vùng lớn)
    max_area_ratio = cfg.get("max_area_ratio", 0.25)  # tối đa 25% frame
    if area / max(1, frame_area) > max_area_ratio:
        return False

    # Chặn bbox dạng dải rộng/cao do MOG bắt nhầm người, rung camera hoặc
    # thay đổi ánh sáng. Kiểm tra area đơn thuần không loại được một bbox
    # hẹp nhưng cao hết khung hình.
    max_width_ratio = cfg.get("max_width_ratio", 0.50)
    max_height_ratio = cfg.get("max_height_ratio", 0.60)
    if w / max(1, frame_w) > max_width_ratio:
        return False
    if h / max(1, frame_h) > max_height_ratio:
        return False

    # Kích thước tối thiểu (% frame)
    min_area_ratio = cfg.get("min_area_ratio", 0.001)  # tối thiểu 0.1% frame
    if area / max(1, frame_area) < min_area_ratio:
        return False

    return True


def overlaps_any_person(bbox: tuple, person_bboxes: list, threshold: float = 0.5) -> bool:
    """Kiểm tra bbox trùng lấn > threshold với bất kỳ person bbox nào.

    Nếu trùng nhiều (Intersection / Area(bbox) > threshold) → đó là 
    một phần cơ thể NGƯỜI (ví dụ: tay, chân) chứ không phải vật thể bỏ quên.
    """
    x1, y1, x2, y2 = bbox
    area_obj = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area_obj <= 0:
        return False

    for pbox in person_bboxes:
        px1, py1, px2, py2 = pbox
        inter_x1 = max(x1, px1)
        inter_y1 = max(y1, py1)
        inter_x2 = min(x2, px2)
        inter_y2 = min(y2, py2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        if inter_area / area_obj > threshold:
            return True
    return False
