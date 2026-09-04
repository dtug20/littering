"""
Gắn nhãn mô tả đồ vật (CHỈ để hiển thị UI / log, KHÔNG ảnh hưởng logic phát
hiện/cảnh báo) - dùng cách tiếp cận zero-shot / open-vocabulary thay vì train
riêng một classifier đa lớp cho từng loại đồ vật.

Vì hàm này chỉ được gọi MỘT LẦN mỗi khi có sự kiện `object_abandoned` (tần
suất rất thấp - vài lần/giờ, không phải mỗi frame), ta có thể chấp nhận dùng
model nặng hơn nhiều so với pipeline realtime mà không ảnh hưởng hiệu năng:

  - CLIP zero-shot classification: so khớp ảnh crop với một danh sách prompt
    dạng text ("a photo of a backpack", "a photo of a suitcase", ...) - danh
    sách này sửa được trong config, không cần train lại khi muốn thêm loại
    đồ vật mới.
  - Open-vocabulary detector (YOLO-World, Grounding DINO): nhận text prompt
    tuỳ ý làm class, không cần train riêng.
  - Gọi API VLM (vision-language model) bên ngoài, trả về mô tả tự nhiên.

Module này định nghĩa interface pluggable; mặc định là NoOpLabeler để không
bắt buộc cài thêm dependency nặng nếu người dùng chưa cần tính năng này.
"""
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger("aod.labeling")


class ObjectLabeler(ABC):
    @abstractmethod
    def label(self, frame_bgr, bbox) -> dict:
        """Trả về {'label': str, 'confidence': float}"""
        raise NotImplementedError


class NoOpLabeler(ObjectLabeler):
    """Mặc định: không gắn nhãn, chỉ trả 'unknown_object'."""

    def label(self, frame_bgr, bbox) -> dict:
        return {"label": "unknown_object", "confidence": 0.0}


class ClipZeroShotLabeler(ObjectLabeler):
    """
    Ví dụ khung sườn dùng CLIP zero-shot - so khớp ảnh crop với danh sách
    prompt cấu hình được (KHÔNG cần train, chỉ cần sửa candidate_labels
    trong app_config.yaml để thêm loại đồ vật mới).

    Cài đặt thực tế cần: pip install open_clip_torch torch --break-system-packages
    Đây là khung sườn - implement _load_model theo hạ tầng GPU thực tế.
    """

    def __init__(self, candidate_labels: list, device: str = "cuda"):
        self.candidate_labels = candidate_labels
        self.device = device
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    def _lazy_load(self):
        if self._model is not None:
            return
        import open_clip  # import trễ để không bắt buộc cài nếu không dùng labeler này
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self._model.to(self.device).eval()
        self._tokenizer = open_clip.get_tokenizer("ViT-B-32")
        logger.info("[Labeler] Đã load CLIP zero-shot model")

    def label(self, frame_bgr, bbox) -> dict:
        try:
            import torch
            import cv2
            from PIL import Image

            self._lazy_load()
            x1, y1, x2, y2 = [int(v) for v in bbox]
            crop = frame_bgr[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                return {"label": "unknown_object", "confidence": 0.0}

            image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            image_input = self._preprocess(image).unsqueeze(0).to(self.device)
            text_input = self._tokenizer(self.candidate_labels).to(self.device)

            with torch.no_grad():
                image_features = self._model.encode_image(image_input)
                text_features = self._model.encode_text(text_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            best_idx = similarity.argmax().item()
            return {
                "label": self.candidate_labels[best_idx],
                "confidence": float(similarity[0, best_idx]),
            }
        except Exception:
            logger.exception("[Labeler] Lỗi khi gắn nhãn zero-shot")
            return {"label": "unknown_object", "confidence": 0.0}


def build_labeler(cfg: dict) -> ObjectLabeler:
    backend = cfg.get("backend", "none")
    if backend == "clip_zero_shot":
        return ClipZeroShotLabeler(
            candidate_labels=cfg.get("candidate_labels", [
                "a backpack", "a suitcase", "a box", "a plastic bag",
                "a helmet", "a bottle", "a bicycle", "an umbrella",
            ]),
            device=cfg.get("device", "cuda"),
        )
    return NoOpLabeler()
