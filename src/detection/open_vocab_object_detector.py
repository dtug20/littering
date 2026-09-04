"""Full-frame open-vocabulary detector used as the source of object boxes.

Unlike MOG/SSIM, this module returns boxes produced by a semantic detector.
The exported YOLO-World model contains both target prompts and hard-negative
prompts; an anchor is accepted only when a target prompt beats every negative
prompt by the configured margin.
"""
from __future__ import annotations

import ast
import logging
import os

import cv2
import numpy as np


logger = logging.getLogger("aod.object_detector")


def _normalise_label(label: str) -> str:
    return " ".join(str(label).lower().replace("_", " ").replace("-", " ").split())


class OnnxYoloWorldObjectDetector:
    """Decode an Ultralytics YOLO-World detect ONNX into image-space boxes."""

    def __init__(self, cfg: dict, session=None):
        self.model_path = str(cfg.get("model_path", "")).strip()
        self.target_labels = {
            _normalise_label(label)
            for label in cfg.get("target_labels", [])
            if str(label).strip()
        }
        if not self.target_labels:
            raise ValueError("object_detection.target_labels không được rỗng")

        self.min_confidence = float(cfg.get("min_confidence", 0.30))
        self.min_target_margin = float(cfg.get("min_target_margin", 0.03))
        self.nms_iou_threshold = float(cfg.get("nms_iou_threshold", 0.45))
        self.max_detections = max(1, int(cfg.get("max_detections", 30)))
        self.tile_enabled = bool(cfg.get("tile_enabled", True))
        self.tile_overlap_ratio = min(
            0.50, max(0.10, float(cfg.get("tile_overlap_ratio", 0.20)))
        )
        self.tile_aspect_ratio_threshold = max(
            1.20, float(cfg.get("tile_aspect_ratio_threshold", 1.60))
        )

        if session is None:
            if not self.model_path or not os.path.isfile(self.model_path):
                raise FileNotFoundError(
                    f"Không tìm thấy model object detector: {self.model_path}"
                )
            import onnxruntime as ort

            requested = cfg.get(
                "providers", ["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            available = set(ort.get_available_providers())
            providers = [provider for provider in requested if provider in available]
            if not providers:
                providers = ["CPUExecutionProvider"]
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = max(
                1, int(cfg.get("cpu_intra_op_threads", 2))
            )
            session_options.inter_op_num_threads = max(
                1, int(cfg.get("cpu_inter_op_threads", 1))
            )
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers,
            )
            logger.info("[ObjectDetector] ONNX providers: %s", session.get_providers())
        self.session = session

        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("YOLO-World ONNX phải có đúng một input image")
        self.input_name = inputs[0].name
        input_shape = inputs[0].shape
        self.input_h = int(input_shape[2])
        self.input_w = int(input_shape[3])

        metadata = self.session.get_modelmeta().custom_metadata_map or {}
        try:
            raw_names = ast.literal_eval(metadata.get("names", "{}"))
        except (SyntaxError, ValueError) as exc:
            raise ValueError("YOLO-World ONNX thiếu metadata names hợp lệ") from exc
        if isinstance(raw_names, list):
            raw_names = dict(enumerate(raw_names))
        self.names = {int(index): str(name) for index, name in raw_names.items()}
        normalised_names = {
            _normalise_label(name): class_id for class_id, name in self.names.items()
        }
        missing = sorted(self.target_labels - set(normalised_names))
        if missing:
            raise ValueError(
                "YOLO-World ONNX không chứa target prompt: " + ", ".join(missing)
            )
        self.target_class_ids = {
            normalised_names[label] for label in self.target_labels
        }
        self.negative_class_ids = set(self.names) - self.target_class_ids

    def _letterbox(self, image: np.ndarray):
        image_h, image_w = image.shape[:2]
        ratio = min(self.input_w / image_w, self.input_h / image_h)
        resized_w = max(1, int(round(image_w * ratio)))
        resized_h = max(1, int(round(image_h * ratio)))
        resized = cv2.resize(
            image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
        )
        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        left = (self.input_w - resized_w) // 2
        top = (self.input_h - resized_h) // 2
        canvas[top:top + resized_h, left:left + resized_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(
            (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        )
        return tensor, ratio, left, top

    def _detect_single(self, frame_bgr: np.ndarray) -> list[dict]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        tensor, ratio, pad_x, pad_y = self._letterbox(frame_bgr)
        prediction = np.asarray(
            self.session.run(None, {self.input_name: tensor})[0]
        ).squeeze(0)
        if prediction.ndim != 2:
            raise ValueError("YOLO-World trả tensor không đúng số chiều")
        if prediction.shape[0] == 4 + len(self.names):
            prediction = prediction.T
        if prediction.shape[1] != 4 + len(self.names):
            raise ValueError("YOLO-World trả tensor không đúng số class")

        scores = prediction[:, 4:]
        target_ids = np.asarray(sorted(self.target_class_ids), dtype=np.int64)
        target_scores = scores[:, target_ids]
        best_target_pos = target_scores.argmax(axis=1)
        best_target_scores = target_scores[
            np.arange(target_scores.shape[0]), best_target_pos
        ]
        best_target_ids = target_ids[best_target_pos]

        if self.negative_class_ids:
            negative_ids = np.asarray(sorted(self.negative_class_ids), dtype=np.int64)
            best_negative_scores = scores[:, negative_ids].max(axis=1)
        else:
            best_negative_scores = np.zeros_like(best_target_scores)

        keep = np.flatnonzero(
            (best_target_scores >= self.min_confidence)
            & (best_target_scores - best_negative_scores >= self.min_target_margin)
        )
        if keep.size == 0:
            return []

        frame_h, frame_w = frame_bgr.shape[:2]
        boxes_xywh = []
        confidences = []
        negative_confidences = []
        target_margins = []
        labels = []
        boxes_xyxy = []
        for anchor_index in keep:
            cx, cy, width, height = prediction[anchor_index, :4].astype(float)
            x1 = max(0.0, min(float(frame_w), (cx - width / 2.0 - pad_x) / ratio))
            y1 = max(0.0, min(float(frame_h), (cy - height / 2.0 - pad_y) / ratio))
            x2 = max(0.0, min(float(frame_w), (cx + width / 2.0 - pad_x) / ratio))
            y2 = max(0.0, min(float(frame_h), (cy + height / 2.0 - pad_y) / ratio))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes_xyxy.append((x1, y1, x2, y2))
            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(float(best_target_scores[anchor_index]))
            negative_confidence = float(best_negative_scores[anchor_index])
            negative_confidences.append(negative_confidence)
            target_margins.append(
                float(best_target_scores[anchor_index]) - negative_confidence
            )
            labels.append(self.names[int(best_target_ids[anchor_index])])

        if not boxes_xywh:
            return []
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
            confidences,
            self.min_confidence,
            self.nms_iou_threshold,
        )
        selected = np.asarray(indices).reshape(-1).tolist() if len(indices) else []
        selected.sort(key=lambda index: confidences[index], reverse=True)
        return [
            {
                "bbox": boxes_xyxy[index],
                "class_name": labels[index],
                "confidence": confidences[index],
                "negative_confidence": negative_confidences[index],
                "target_margin": target_margins[index],
                "source": "yolo_world",
            }
            for index in selected[:self.max_detections]
        ]

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """Detect on overlapping square tiles, then apply global NMS.

        CCTV bags are often only a few dozen source pixels. Feeding the whole
        16:9 image to a 384x384 model makes them too small; square tiling keeps
        roughly twice as much detail without lowering the confidence threshold.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        frame_h, frame_w = frame_bgr.shape[:2]
        if (
            not self.tile_enabled
            or max(frame_w, frame_h)
            <= min(frame_w, frame_h) * self.tile_aspect_ratio_threshold
        ):
            return self._detect_single(frame_bgr)

        horizontal = frame_w >= frame_h
        long_side = frame_w if horizontal else frame_h
        tile_size = min(frame_w, frame_h)
        stride = max(1, int(round(tile_size * (1.0 - self.tile_overlap_ratio))))
        starts = list(range(0, max(1, long_side - tile_size + 1), stride))
        final_start = max(0, long_side - tile_size)
        if not starts or starts[-1] != final_start:
            starts.append(final_start)

        detections = []
        for start in starts:
            if horizontal:
                tile = frame_bgr[:, start:start + tile_size]
                offset_x, offset_y = start, 0
            else:
                tile = frame_bgr[start:start + tile_size, :]
                offset_x, offset_y = 0, start
            for detection in self._detect_single(tile):
                x1, y1, x2, y2 = detection["bbox"]
                item = dict(detection)
                item["bbox"] = (
                    x1 + offset_x, y1 + offset_y,
                    x2 + offset_x, y2 + offset_y,
                )
                detections.append(item)

        if not detections:
            return []
        boxes_xywh = []
        confidences = []
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(float(detection["confidence"]))
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh,
            confidences,
            self.min_confidence,
            self.nms_iou_threshold,
        )
        selected = np.asarray(indices).reshape(-1).tolist() if len(indices) else []
        selected.sort(key=lambda index: confidences[index], reverse=True)
        return [detections[index] for index in selected[:self.max_detections]]


def build_object_detector(cfg: dict):
    if not cfg.get("enabled", False):
        return None
    backend = str(cfg.get("backend", "onnx_yolo_world")).lower()
    if backend != "onnx_yolo_world":
        raise ValueError(f"object_detection.backend không hỗ trợ: {backend}")
    return OnnxYoloWorldObjectDetector(cfg)
