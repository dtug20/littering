"""
Tracker class-agnostic đơn giản (IoU) để gán object_id ổn định cho các static
blob sinh ra từ MOG - vì các blob này KHÔNG đi qua nvtracker của DeepStream
(nvtracker giờ chỉ track kết quả nvinfer, mà nvinfer chỉ detect person).

Vì đối tượng đã đứng yên khi được MOG báo, việc track ở đây đơn giản và ổn
định hơn nhiều so với track vật đang chuyển động - chỉ cần match IoU giữa
frame liên tiếp là đủ, không cần Kalman/ReID phức tạp.
"""
import time
import uuid


class _TrackedBlob:
    def __init__(self, blob_id, bbox):
        self.blob_id = blob_id
        self.bbox = bbox
        self.last_seen_at = time.time()
        self.consecutive_hits = 1


class BlobTracker:
    def __init__(self, iou_fn, iou_match_threshold=0.2,
                 max_missed_seconds=60.0, bbox_smoothing_alpha=0.25):
        """
        iou_fn: hàm iou(box_a, box_b) tái sử dụng từ utils.geometry
        max_missed_seconds: thời gian giữ ID trước khi xoá hẳn track - nên đặt
          LỚN HƠN max_occlusion_gap_seconds ở tầng state machine, vì việc
          quyết định "huỷ theo dõi" hay "bắn sự kiện removed" thuộc trách
          nhiệm của ObjectStateTracker, BlobTracker chỉ có nhiệm vụ giữ ID.
        """
        self._iou = iou_fn
        self.iou_match_threshold = iou_match_threshold
        self.max_missed_seconds = max_missed_seconds
        self.bbox_smoothing_alpha = min(
            1.0, max(0.05, float(bbox_smoothing_alpha))
        )
        self._tracks: dict[str, _TrackedBlob] = {}

    def update(self, boxes: list) -> list:
        """
        boxes: list[(x1,y1,x2,y2)] các blob hình học hợp lệ của frame hiện tại
               (đã qua shape_filters).
        Trả về list[dict] {object_id, bbox, is_new, consecutive_hits}
        """
        now = time.time()
        used_track_ids = set()
        results = []

        for box in boxes:
            best_id, best_score = None, 0.0
            for tid, tb in self._tracks.items():
                if tid in used_track_ids:
                    continue
                score = self._iou(box, tb.bbox)
                if score > best_score:
                    best_score = score
                    best_id = tid

            if best_id is not None and best_score >= self.iou_match_threshold:
                tb = self._tracks[best_id]
                alpha = self.bbox_smoothing_alpha
                tb.bbox = tuple(
                    old_value * (1.0 - alpha) + new_value * alpha
                    for old_value, new_value in zip(tb.bbox, box)
                )
                tb.last_seen_at = now
                tb.consecutive_hits += 1
                used_track_ids.add(best_id)
                results.append({
                    "object_id": best_id, "bbox": tb.bbox,
                    "is_new": False, "consecutive_hits": tb.consecutive_hits,
                })
            else:
                new_id = str(uuid.uuid4())
                self._tracks[new_id] = _TrackedBlob(new_id, box)
                used_track_ids.add(new_id)
                results.append({
                    "object_id": new_id, "bbox": box,
                    "is_new": True, "consecutive_hits": 1,
                })

        stale = [tid for tid, tb in self._tracks.items()
                 if now - tb.last_seen_at > self.max_missed_seconds]
        for tid in stale:
            del self._tracks[tid]

        return results

    def get_last_bbox(self, object_id):
        tb = self._tracks.get(object_id)
        return tb.bbox if tb else None

    def forget(self, object_id):
        self._tracks.pop(object_id, None)

    def clear(self):
        self._tracks.clear()
