"""Semantic verification for trash-only abandoned-object alerts.

The motion/SSIM pipeline can establish that a new static region exists, but
it cannot establish what that region is.  This module is the mandatory final
gate between an abandoned-object candidate and a public alert.

Production deployments should use a custom YOLO detection/classification
model trained with the exact target classes and camera domain.  The verifier
is deliberately fail-closed: an unavailable model never turns an unknown
object into a trash alert.
"""
from __future__ import annotations

import logging
import os
import time
import ast
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np


logger = logging.getLogger("aod.target_verifier")


@dataclass(frozen=True)
class TargetPrediction:
    """One semantic observation.

    ``is_target`` is ``None`` when inference was unavailable.  Unavailable is
    intentionally different from a negative prediction so a temporary model
    error does not permanently discard a real trash item.
    """

    is_target: bool | None
    label: str = "unknown"
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class GateDecision:
    decision: str  # pending | accept | reject
    label: str = "unknown"
    confidence: float = 0.0
    reason: str = ""


class UnavailableTargetVerifier:
    """Fail-closed verifier used when the configured model is not available."""

    def __init__(self, reason: str):
        self.reason = reason

    def verify(self, frame_bgr: np.ndarray, bbox: tuple) -> TargetPrediction:
        return TargetPrediction(None, reason=self.reason)


def _normalise_label(label: str) -> str:
    return " ".join(str(label).lower().replace("_", " ").replace("-", " ").split())


class UltralyticsYoloTargetVerifier:
    """Verify a candidate crop with a custom Ultralytics model.

    Both YOLO detection/segmentation models (``result.boxes``) and YOLO
    classification models (``result.probs``) are supported.  For a detector,
    at least one target-class box above ``min_confidence`` must be present in
    the expanded candidate crop.  For a classifier, the best target class
    must also beat the best non-target class by ``min_target_margin``.
    """

    def __init__(self, cfg: dict, model=None):
        self.model_path = str(cfg.get("model_path", "")).strip()
        self.target_labels = {
            _normalise_label(label) for label in cfg.get("target_labels", []) if str(label).strip()
        }
        if not self.target_labels:
            raise ValueError("target_filter.target_labels phải có ít nhất một nhãn rác")

        self.min_confidence = float(cfg.get("min_confidence", 0.70))
        self.min_target_margin = float(cfg.get("min_target_margin", 0.10))
        self.crop_padding_ratio = float(cfg.get("crop_padding_ratio", 0.15))
        self.imgsz = int(cfg.get("imgsz", 384))
        self.device = cfg.get("device", 0)

        if model is None:
            if not self.model_path:
                raise ValueError("target_filter.model_path chưa được cấu hình")
            if not os.path.isfile(self.model_path):
                raise FileNotFoundError(
                    f"Không tìm thấy model xác thực rác: {self.model_path}"
                )
            from ultralytics import YOLO

            model = YOLO(self.model_path)
        self.model = model

        configured_names = getattr(self.model, "names", None)
        if configured_names:
            available = {
                _normalise_label(name)
                for name in (
                    configured_names.values()
                    if isinstance(configured_names, dict)
                    else configured_names
                )
            }
            missing = sorted(self.target_labels - available)
            if missing:
                raise ValueError(
                    "Model không chứa target label đã cấu hình: " + ", ".join(missing)
                )

    @staticmethod
    def _names_for_result(result) -> dict:
        names = getattr(result, "names", {}) or {}
        if isinstance(names, dict):
            return names
        return {index: name for index, name in enumerate(names)}

    def _crop(self, frame_bgr: np.ndarray, bbox: tuple) -> np.ndarray | None:
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        x1, y1, x2, y2 = [float(v) for v in bbox]
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        pad_x = box_w * self.crop_padding_ratio
        pad_y = box_h * self.crop_padding_ratio
        frame_h, frame_w = frame_bgr.shape[:2]
        ix1 = max(0, int(x1 - pad_x))
        iy1 = max(0, int(y1 - pad_y))
        ix2 = min(frame_w, int(x2 + pad_x + 0.999))
        iy2 = min(frame_h, int(y2 + pad_y + 0.999))
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        return frame_bgr[iy1:iy2, ix1:ix2]

    def verify(self, frame_bgr: np.ndarray, bbox: tuple) -> TargetPrediction:
        crop = self._crop(frame_bgr, bbox)
        if crop is None:
            return TargetPrediction(None, reason="empty_candidate_crop")

        try:
            results = self.model.predict(
                source=crop,
                imgsz=self.imgsz,
                device=self.device,
                conf=self.min_confidence,
                verbose=False,
            )
        except Exception as exc:
            logger.exception("[TargetVerifier] Lỗi inference model rác")
            return TargetPrediction(None, reason=f"inference_error:{type(exc).__name__}")

        if not results:
            return TargetPrediction(False, reason="no_prediction")
        result = results[0]
        names = self._names_for_result(result)

        # YOLO classification result.
        probs = getattr(result, "probs", None)
        prob_data = getattr(probs, "data", None) if probs is not None else None
        if prob_data is not None:
            values = prob_data.detach().cpu().numpy().astype(float).reshape(-1)
            target_candidates = []
            non_target_candidates = []
            for class_id, confidence in enumerate(values):
                label = str(names.get(class_id, class_id))
                item = (float(confidence), label)
                if _normalise_label(label) in self.target_labels:
                    target_candidates.append(item)
                else:
                    non_target_candidates.append(item)
            best_target = max(target_candidates, default=(0.0, "unknown"))
            best_other = max(non_target_candidates, default=(0.0, "unknown"))
            accepted = (
                best_target[0] >= self.min_confidence
                and best_target[0] - best_other[0] >= self.min_target_margin
            )
            return TargetPrediction(
                accepted,
                label=best_target[1] if accepted else best_other[1],
                confidence=best_target[0] if accepted else best_other[0],
                reason="classification_target" if accepted else "classification_non_target",
            )

        # YOLO detection/segmentation result.
        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "cls", None) is None:
            return TargetPrediction(False, reason="no_target_detection")

        class_ids = boxes.cls.detach().cpu().numpy().astype(int).reshape(-1)
        confidences = boxes.conf.detach().cpu().numpy().astype(float).reshape(-1)
        best_target = (0.0, "unknown")
        best_other = (0.0, "unknown")
        for class_id, confidence in zip(class_ids, confidences):
            label = str(names.get(int(class_id), class_id))
            item = (float(confidence), label)
            if _normalise_label(label) in self.target_labels:
                best_target = max(best_target, item)
            else:
                best_other = max(best_other, item)

        if best_target[0] >= self.min_confidence:
            return TargetPrediction(True, best_target[1], best_target[0], "target_detected")
        return TargetPrediction(False, best_other[1], best_other[0], "no_target_detection")


class OnnxYoloWorldTargetVerifier:
    """Run a prompt-frozen YOLO-World ONNX model without PyTorch.

    Text embeddings are baked into the ONNX during export, so runtime only
    needs ONNX Runtime. Positive and hard-negative prompts compete in the
    same output; a target must beat the strongest negative by
    ``min_target_margin``.
    """

    def __init__(self, cfg: dict, session=None):
        self.model_path = str(cfg.get("model_path", "")).strip()
        self.target_labels = {
            _normalise_label(label) for label in cfg.get("target_labels", []) if str(label).strip()
        }
        if not self.target_labels:
            raise ValueError("target_filter.target_labels phải có ít nhất một prompt rác")
        self.min_confidence = float(cfg.get("min_confidence", 0.35))
        self.min_target_margin = float(cfg.get("min_target_margin", 0.05))
        self.crop_padding_ratio = float(cfg.get("crop_padding_ratio", 0.15))

        if session is None:
            if not self.model_path:
                raise ValueError("target_filter.model_path chưa được cấu hình")
            if not os.path.isfile(self.model_path):
                raise FileNotFoundError(f"Không tìm thấy YOLO-World ONNX: {self.model_path}")
            import onnxruntime as ort

            requested = cfg.get(
                "providers", ["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            available = set(ort.get_available_providers())
            providers = [provider for provider in requested if provider in available]
            if not providers:
                providers = ["CPUExecutionProvider"]
            session = ort.InferenceSession(self.model_path, providers=providers)
            logger.info("[TargetVerifier] ONNX providers: %s", session.get_providers())
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
            raise ValueError("ONNX thiếu metadata names hợp lệ") from exc
        if isinstance(raw_names, list):
            raw_names = dict(enumerate(raw_names))
        self.names = {int(index): str(name) for index, name in raw_names.items()}
        available_labels = {_normalise_label(name) for name in self.names.values()}
        missing = sorted(self.target_labels - available_labels)
        if missing:
            raise ValueError("ONNX không chứa target prompt: " + ", ".join(missing))

    def _crop(self, frame_bgr: np.ndarray, bbox: tuple) -> np.ndarray | None:
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        x1, y1, x2, y2 = [float(v) for v in bbox]
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        pad_x = width * self.crop_padding_ratio
        pad_y = height * self.crop_padding_ratio
        frame_h, frame_w = frame_bgr.shape[:2]
        ix1 = max(0, int(x1 - pad_x))
        iy1 = max(0, int(y1 - pad_y))
        ix2 = min(frame_w, int(x2 + pad_x + 0.999))
        iy2 = min(frame_h, int(y2 + pad_y + 0.999))
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        return frame_bgr[iy1:iy2, ix1:ix2]

    def _letterbox(self, image: np.ndarray) -> np.ndarray:
        image_h, image_w = image.shape[:2]
        ratio = min(self.input_w / image_w, self.input_h / image_h)
        resized_w = max(1, int(round(image_w * ratio)))
        resized_h = max(1, int(round(image_h * ratio)))
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        left = (self.input_w - resized_w) // 2
        top = (self.input_h - resized_h) // 2
        canvas[top:top + resized_h, left:left + resized_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None])

    def verify(self, frame_bgr: np.ndarray, bbox: tuple) -> TargetPrediction:
        crop = self._crop(frame_bgr, bbox)
        if crop is None:
            return TargetPrediction(None, reason="empty_candidate_crop")
        try:
            output = self.session.run(None, {self.input_name: self._letterbox(crop)})[0]
        except Exception as exc:
            logger.exception("[TargetVerifier] Lỗi inference YOLO-World ONNX")
            return TargetPrediction(None, reason=f"inference_error:{type(exc).__name__}")

        prediction = np.asarray(output).squeeze(0)
        if prediction.ndim != 2:
            return TargetPrediction(None, reason="unexpected_onnx_output")
        # Ultralytics detect export is [4 + classes, anchors].
        if prediction.shape[0] == 4 + len(self.names):
            prediction = prediction.T
        if prediction.shape[1] != 4 + len(self.names):
            return TargetPrediction(None, reason="unexpected_onnx_output")

        scores = prediction[:, 4:]
        if scores.size == 0:
            return TargetPrediction(False, reason="no_prediction")
        best_score_by_class = scores.max(axis=0)
        best_target = (0.0, "unknown")
        best_other = (0.0, "unknown")
        for class_id, confidence in enumerate(best_score_by_class):
            label = self.names.get(class_id, str(class_id))
            item = (float(confidence), label)
            if _normalise_label(label) in self.target_labels:
                best_target = max(best_target, item)
            else:
                best_other = max(best_other, item)

        accepted = (
            best_target[0] >= self.min_confidence
            and best_target[0] - best_other[0] >= self.min_target_margin
        )
        if accepted:
            return TargetPrediction(True, best_target[1], best_target[0], "open_vocab_target")
        return TargetPrediction(
            False,
            best_other[1] if best_other[0] >= best_target[0] else best_target[1],
            max(best_other[0], best_target[0]),
            "open_vocab_non_target",
        )


class TemporalTargetGate:
    """Require semantic agreement over several sampled frames."""

    def __init__(self, verifier, cfg: dict):
        self.verifier = verifier
        self.confirmations = max(1, int(cfg.get("confirmations", 3)))
        self.max_samples = max(self.confirmations, int(cfg.get("max_samples", 5)))
        self.min_positive_ratio = float(cfg.get("min_positive_ratio", 0.60))
        self.sample_interval_seconds = max(0.0, float(cfg.get("sample_interval_seconds", 0.15)))
        self._samples: dict[str, list[TargetPrediction]] = {}
        self._last_sample_at: dict[str, float] = {}

    def evaluate(self, object_id: str, frame_bgr: np.ndarray | None, bbox: tuple) -> GateDecision:
        if frame_bgr is None:
            return GateDecision("pending", reason="frame_unavailable")

        now = time.time()
        last_sample = self._last_sample_at.get(object_id)
        if last_sample is not None and now - last_sample < self.sample_interval_seconds:
            return GateDecision("pending", reason="waiting_next_sample")
        self._last_sample_at[object_id] = now

        prediction = self.verifier.verify(frame_bgr, bbox)
        if prediction.is_target is None:
            # Fail closed, but keep the track pending so a transient inference
            # problem can recover on a later sampled frame.
            return GateDecision("pending", reason=prediction.reason or "verifier_unavailable")

        samples = self._samples.setdefault(object_id, [])
        samples.append(prediction)
        positives = [sample for sample in samples if sample.is_target]
        positive_ratio = len(positives) / len(samples)

        if len(positives) >= self.confirmations and positive_ratio >= self.min_positive_ratio:
            label_counts = Counter(sample.label for sample in positives)
            label = label_counts.most_common(1)[0][0]
            matching_conf = [sample.confidence for sample in positives if sample.label == label]
            confidence = sum(matching_conf) / max(1, len(matching_conf))
            return GateDecision("accept", label, confidence, "temporal_consensus")

        samples_left = self.max_samples - len(samples)
        cannot_reach_confirmations = len(positives) + samples_left < self.confirmations
        if len(samples) >= self.max_samples or cannot_reach_confirmations:
            best = max(samples, key=lambda sample: sample.confidence)
            return GateDecision("reject", best.label, best.confidence, "insufficient_target_consensus")

        return GateDecision("pending", reason="collecting_semantic_samples")

    def forget(self, object_id: str):
        self._samples.pop(object_id, None)
        self._last_sample_at.pop(object_id, None)

    def clear(self):
        self._samples.clear()
        self._last_sample_at.clear()

    def retain(self, active_object_ids):
        active = set(active_object_ids)
        for object_id in list(self._samples):
            if object_id not in active:
                self.forget(object_id)


def build_target_verifier(cfg: dict):
    """Build one model instance that can be shared by camera-specific gates.

    A missing/broken model becomes an unavailable verifier unless
    ``fail_on_model_error`` is true.  In both cases an unknown object is never
    accepted as trash.
    """

    backend = str(cfg.get("backend", "ultralytics_yolo")).lower()
    try:
        if backend == "ultralytics_yolo":
            verifier = UltralyticsYoloTargetVerifier(cfg)
        elif backend == "onnx_yolo_world":
            verifier = OnnxYoloWorldTargetVerifier(cfg)
        else:
            raise ValueError(f"target_filter.backend không hỗ trợ: {backend}")
        logger.info("[TargetVerifier] Đã load model rác: %s", cfg.get("model_path"))
    except Exception as exc:
        if cfg.get("fail_on_model_error", False):
            raise
        logger.error(
            "[TargetVerifier] Model rác chưa sẵn sàng; hệ thống chạy fail-closed "
            "và KHÔNG phát cảnh báo object: %s",
            exc,
        )
        verifier = UnavailableTargetVerifier(str(exc))
    return verifier


def build_target_gate(cfg: dict, verifier=None) -> TemporalTargetGate | None:
    if not cfg.get("enabled", False):
        return None
    return TemporalTargetGate(verifier or build_target_verifier(cfg), cfg)
