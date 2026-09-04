"""Các hàm hình học dùng chung: IoU, khoảng cách tâm, point-in-polygon."""
import math


def iou(box_a, box_b) -> float:
    """box = (x1, y1, x2, y2)"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def euclidean_distance(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def box_area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_over_box(box, other) -> float:
    """Fraction of ``box`` covered by ``other`` (asymmetric overlap)."""
    x1, y1, x2, y2 = box
    ox1, oy1, ox2, oy2 = other
    intersection = max(0.0, min(x2, ox2) - max(x1, ox1)) * max(
        0.0, min(y2, oy2) - max(y1, oy1)
    )
    return intersection / max(1.0, box_area(box))


def point_in_polygon(point, polygon) -> bool:
    """Ray casting algorithm. polygon = [[x,y], ...]"""
    x, y = point
    n = len(polygon)
    if n < 3:
        return False

    # Xem điểm trên cạnh là nằm trong ROI. Ray casting thuần có thể trả kết
    # quả khác nhau ở đúng biên, gây bbox nhấp nháy khi vật nằm sát polygon.
    epsilon = 1e-6
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) > epsilon:
            continue
        if (
            min(x1, x2) - epsilon <= x <= max(x1, x2) + epsilon
            and min(y1, y2) - epsilon <= y <= max(y1, y2) + epsilon
        ):
            return True

    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def box_in_any_polygon(box, polygons) -> bool:
    """box được coi là trong ROI nếu tâm bbox nằm trong bất kỳ polygon nào."""
    if not polygons:
        return True  # không cấu hình ROI -> áp dụng toàn khung hình
    c = centroid(box)
    return any(point_in_polygon(c, poly["points"]) for poly in polygons)


def normalise_roi_polygons(polygons, frame_w: int, frame_h: int) -> list[dict]:
    """Chuẩn hóa ROI từ MQTT/YAML về ``[{name, points:[[x,y], ...]}]``.

    Hỗ trợ point dạng ``[x,y]`` hoặc ``{"x": x, "y": y}``. Nếu toàn bộ
    tọa độ của một polygon nằm trong [0, 1], chúng được hiểu là normalized.
    Polygon lỗi hoặc có ít hơn ba điểm bị loại an toàn.
    """
    result = []
    for index, polygon in enumerate(polygons or []):
        raw_points = polygon.get("points", []) if isinstance(polygon, dict) else polygon
        points = []
        for point in raw_points or []:
            if isinstance(point, dict) and "x" in point and "y" in point:
                px, py = point["x"], point["y"]
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                px, py = point[0], point[1]
            else:
                continue
            try:
                points.append([float(px), float(py)])
            except (TypeError, ValueError):
                continue
        if len(points) < 3:
            continue

        is_normalised = all(0.0 <= px <= 1.0 and 0.0 <= py <= 1.0 for px, py in points)
        if is_normalised:
            points = [[px * frame_w, py * frame_h] for px, py in points]

        name = polygon.get("name", f"roi_{index}") if isinstance(polygon, dict) else f"roi_{index}"
        result.append({"name": name, "points": points})
    return result
