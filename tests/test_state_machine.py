"""
Test giả lập (không cần DeepStream/GStreamer/OpenCV-MOG) để kiểm chứng
ObjectStateTracker v2 (class-agnostic, dựa trên blob) xử lý đúng 5 test case.

Mô phỏng trực tiếp đầu ra của BlobTracker (thay vì chạy MOG thật trên video),
vì mục tiêu là kiểm chứng LOGIC state machine + ownership, không phải kiểm
chứng chất lượng background subtraction (việc đó cần video thật).

Chạy: cd aod_deepstream && python3 tests/test_state_machine.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tracking.object_state_tracker import ObjectStateTracker
from src.logic.ownership import PersonHistoryBuffer
from src.utils import geometry

RULES = {
    "static_confirm_frames": 2,
    "owner_distance_threshold_px": 150,
    "abandonment_dwell_seconds": 2,          # rút ngắn để test nhanh
    "max_occlusion_gap_seconds": 5,
    "pickup_confirm_gap_seconds": 1.0,
}


def run_case(name, steps, expect_event_types):
    """
    steps: list[dict] mỗi dict mô tả 1 frame quan sát:
      - "blobs": list[(x1,y1,x2,y2)] các blob MOG (đã qua shape_filters) frame này
      - "persons": dict[person_id] -> bbox frame này
      - "baseline": bool (mặc định False)
      - "sleep": số giây chờ TRƯỚC khi quan sát frame này
    """
    tracker = ObjectStateTracker(dict(RULES))
    history = PersonHistoryBuffer(window_seconds=5.0)
    fired = []

    # gán object_id ổn định thủ công theo vị trí xấp xỉ (giả lập BlobTracker
    # đơn giản hoá cho test, không cần import cv2/MOG thật)
    known_blob_id = {}

    def blob_key(bbox):
        return (round(bbox[0] / 20), round(bbox[1] / 20))

    for step in steps:
        if "sleep" in step:
            time.sleep(step["sleep"])

        persons = step.get("persons", {})
        history.push("cam_01", persons)

        blob_results = []
        for bbox in step.get("blobs", []):
            key = blob_key(bbox)
            is_new = key not in known_blob_id
            if is_new:
                known_blob_id[key] = f"blob_{len(known_blob_id)}"
            oid = known_blob_id[key]
            blob_results.append({
                "object_id": oid, "bbox": bbox,
                "is_new": is_new, "consecutive_hits": 99,  # đã qua ngưỡng confirm ngay để test nhanh
            })

        def owner_lookup(loc):
            return history.find_owner_at_location("cam_01", loc, geometry, 200.0)

        events = tracker.update(
            camera_id="cam_01",
            blob_results=blob_results,
            person_tracks=persons,
            roi_polygons=[],
            is_in_baseline=step.get("baseline", False),
            geometry_mod=geometry,
            owner_lookup_fn=owner_lookup,
        )
        fired.extend(e["event_type"] for e in events)

    ok = fired == expect_event_types
    print(f"[{'PASS' if ok else 'FAIL'}] {name} -> fired={fired} expected={expect_event_types}")
    return ok


def main():
    results = []

    # 1. Đặt đồ xuống rồi rời khỏi khu vực -> Cảnh báo
    # (MOG chỉ sinh blob SAU KHI vật đứng yên -> ở đây blob "obj" chỉ xuất
    #  hiện từ bước 2 trở đi, mô phỏng đúng đặc tính class-agnostic của MOG)
    results.append(run_case(
        "1. Dat do xuong roi roi khoi khu vuc -> Canh bao",
        steps=[
            {"blobs": [], "persons": {"p1": (95, 95, 135, 135)}},  # đang cầm, chưa đặt -> chưa có blob
            {"blobs": [(100, 100, 130, 130)], "persons": {"p1": (200, 200, 240, 240)}},  # vừa đặt xuống, blob xuất hiện
            {"sleep": 0.3, "blobs": [(100, 100, 130, 130)], "persons": {"p1": (500, 500, 540, 540)}},
            {"sleep": 2.2, "blobs": [(100, 100, 130, 130)], "persons": {"p1": (500, 500, 540, 540)}},
        ],
        expect_event_types=["object_abandoned"],
    ))

    # 2. Đặt đồ xuống rồi quay lại lấy -> Không cảnh báo
    results.append(run_case(
        "2. Dat do xuong roi quay lai lay -> Khong canh bao",
        steps=[
            {"blobs": [], "persons": {"p2": (195, 195, 235, 235)}},
            {"blobs": [(200, 200, 230, 230)], "persons": {"p2": (600, 600, 640, 640)}},  # đặt xuống rời đi
            {"sleep": 0.3, "blobs": [(200, 200, 230, 230)], "persons": {"p2": (600, 600, 640, 640)}},
            # quay lại: đứng chồng lên vị trí vật -> blob biến mất (MOG không
            # còn thấy static vì có người che/chạm) -> PENDING_CLAIM
            {"blobs": [], "persons": {"p2": (195, 195, 235, 235)}},
            {"sleep": 1.2, "blobs": [], "persons": {"p2": (600, 600, 640, 640)}},  # vật không tái xuất hiện -> đã bị mang đi
        ],
        expect_event_types=[],
    ))

    # 3. Đứng cạnh đồ vật sau khi đặt xuống -> Không cảnh báo
    results.append(run_case(
        "3. Dung canh do vat sau khi dat xuong -> Khong canh bao",
        steps=[
            {"blobs": [], "persons": {"p3": (295, 295, 335, 335)}},
            {"blobs": [(300, 300, 330, 330)], "persons": {"p3": (335, 300, 375, 330)}},  # đặt xuống, đứng ngay cạnh
            {"sleep": 2.5, "blobs": [(300, 300, 330, 330)], "persons": {"p3": (335, 300, 375, 330)}},
        ],
        expect_event_types=[],
    ))

    # 4. Đi ngang qua với đồ vật trên tay -> Không cảnh báo
    # (vật luôn di chuyển cùng người -> MOG KHÔNG BAO GIỜ sinh static blob
    #  cho nó -> không có track nào được tạo => class-agnostic tự đáp ứng)
    results.append(run_case(
        "4. Di ngang qua voi do vat tren tay -> Khong canh bao",
        steps=[
            {"blobs": [], "persons": {"p4": (395, 395, 435, 435)}},
            {"blobs": [], "persons": {"p4": (415, 395, 455, 435)}},
            {"sleep": 0.3, "blobs": [], "persons": {"p4": (445, 395, 485, 435)}},
        ],
        expect_event_types=[],
    ))

    # 5. Đồ vật cố định từ trước trong khu vực -> Không cảnh báo
    results.append(run_case(
        "5. Do vat co dinh tu truoc -> Khong canh bao",
        steps=[
            {"blobs": [(500, 500, 530, 530)], "persons": {}, "baseline": True},
            {"sleep": 0.3, "blobs": [(500, 500, 530, 530)], "persons": {}, "baseline": False},
            {"sleep": 2.5, "blobs": [(500, 500, 530, 530)], "persons": {}, "baseline": False},
        ],
        expect_event_types=[],
    ))

    print()
    total = len(results)
    passed = sum(results)
    print(f"=== KET QUA: {passed}/{total} test case PASS ===")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
