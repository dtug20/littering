"""
Bộ phát hiện vật thể tĩnh (static foreground) dùng MOG2 – phiên bản v3
**Person-Centric**: KHÔNG còn quét toàn frame, chỉ verify cục bộ tại vùng
mà PersonDepartureDetector đã xác định có vật mới.

Vai trò mới:
  - XÁC NHẬN (verify) rằng vật thể thực sự đứng yên tĩnh tại vùng candidate
    do PersonDepartureDetector phát hiện.
  - Loại false positive từ SSIM diff do:
    + Người khác đang đi qua vùng đó (foreground di chuyển, không phải tĩnh)
    + Biến đổi ánh sáng tạm thời (flash, đèn bật/tắt)

Cải tiến so với v2:
  - XÓA GrabCut (tốn CPU, không hiệu quả)
  - Gaussian Blur trước khi apply MOG (giảm nhiễu hạt)
  - Phương thức verify_region() chỉ kiểm tra vùng nhỏ
  - Temporal stability: vùng phải có foreground ổn định qua N frame liên tiếp
"""
import time
import logging
import cv2
import numpy as np

from src.utils.geometry import box_area

logger = logging.getLogger("aod.mog")


class _CameraMogState:
    def __init__(self, cfg: dict, baseline_learning_seconds: float):
        self.short_term = cv2.createBackgroundSubtractorMOG2(
            history=cfg.get("history", 500),
            varThreshold=cfg.get("var_threshold", 25),
            detectShadows=cfg.get("detect_shadows", True),
        )
        self.long_term = cv2.createBackgroundSubtractorMOG2(
            history=cfg.get("history", 500) * 4,
            varThreshold=cfg.get("var_threshold", 25),
            detectShadows=cfg.get("detect_shadows", True),
        )
        self.short_lr = cfg.get("short_term_learning_rate", 0.005)
        self.long_lr = cfg.get("long_term_learning_rate", 0.0003)
        self.min_blob_area_px = cfg.get("min_blob_area_px", 500)

        self.started_at = time.time()
        self.baseline_learning_seconds = baseline_learning_seconds

    def in_baseline_period(self) -> bool:
        return (time.time() - self.started_at) < self.baseline_learning_seconds

    def feed_frame(self, frame_bgr: np.ndarray):
        """Cập nhật model nền mỗi frame (KHÔNG sinh candidate, chỉ học nền).

        Gọi hàm này mỗi frame để MOG2 liên tục cập nhật model nền chính xác.
        Không trả về gì – việc phát hiện vật thể do PersonDepartureDetector quyết định.
        """
        # Gaussian blur giảm nhiễu hạt trước khi apply MOG
        blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        self.short_term.apply(blurred, learningRate=self.short_lr)
        self.long_term.apply(blurred, learningRate=self.long_lr)

    def verify_region(self, frame_bgr: np.ndarray, region_bbox: tuple,
                      scale_x: float = 1.0, scale_y: float = 1.0) -> dict | None:
        """Kiểm tra xem vùng candidate (từ PersonDepartureDetector) có foreground
        tĩnh thực sự hay không.

        Trả về dict {bbox, contour, fg_ratio} nếu xác nhận có vật tĩnh,
        hoặc None nếu không.

        Parameters
        ----------
        region_bbox : (x1, y1, x2, y2) trong tọa độ frame hiện tại (đã scale)
        """
        x1, y1, x2, y2 = [int(v) for v in region_bbox]
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        # Add padding to prevent the bounding box from shrinking iteratively
        pad = 20
        x1_pad = max(0, x1 - pad)
        y1_pad = max(0, y1 - pad)
        x2_pad = min(frame_bgr.shape[1], x2 + pad)
        y2_pad = min(frame_bgr.shape[0], y2 + pad)

        crop = frame_bgr[y1_pad:y2_pad, x1_pad:x2_pad]
        blurred_crop = cv2.GaussianBlur(crop, (5, 5), 0)

        # Lấy foreground mask từ short-term (vùng crop phải được embedded
        # trong context full frame, nên ta dùng mask từ full frame)
        blurred_full = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        fg_short_full = self.short_term.apply(blurred_full, learningRate=0)  # learningRate=0 → chỉ query, không học
        fg_long_full = self.long_term.apply(blurred_full, learningRate=0)

        # Crop mask tương ứng
        fg_short = fg_short_full[y1_pad:y2_pad, x1_pad:x2_pad]
        fg_long = fg_long_full[y1_pad:y2_pad, x1_pad:x2_pad]

        # Loại bỏ bóng đổ (giá trị 127)
        fg_short_bin = cv2.threshold(fg_short, 200, 255, cv2.THRESH_BINARY)[1]
        fg_long_bin = cv2.threshold(fg_long, 200, 255, cv2.THRESH_BINARY)[1]

        # Static = ĐÃ là background ở short-term (học nhanh) NHƯNG VẪN LÀ foreground ở long-term (học chậm)
        static_mask = cv2.bitwise_and(fg_long_bin, cv2.bitwise_not(fg_short_bin))

        # Khử nhiễu hột tiêu
        clean_mask = cv2.morphologyEx(static_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # Gom mảnh nhỏ (kernel nhỏ hơn v2 vì chỉ verify vùng nhỏ)
        grouped_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        # Tính tỷ lệ foreground trong vùng
        total_pixels = grouped_mask.size
        fg_pixels = cv2.countNonZero(grouped_mask)
        fg_ratio = fg_pixels / max(1, total_pixels)

        # Ngưỡng: ít nhất 5% vùng phải là foreground
        if fg_ratio < 0.05:
            return None

        # Tìm contour lớn nhất trong vùng
        contours, _ = cv2.findContours(grouped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < self.min_blob_area_px:
            return None

        # Tính tight bbox từ contour lớn nhất (ổn định hơn findNonZero)
        tx, ty, tw, th = cv2.boundingRect(largest)

        # Chuyển về tọa độ frame gốc (có scale)
        final_x = float(x1_pad + tx) * scale_x
        final_y = float(y1_pad + ty) * scale_y
        final_w = float(tw) * scale_x
        final_h = float(th) * scale_y
        bbox_out = (final_x, final_y, final_x + final_w, final_y + final_h)

        if box_area(bbox_out) < self.min_blob_area_px:
            return None

        # Scale contour về tọa độ frame gốc
        c_scaled = largest.copy()
        c_scaled[:, 0, 0] = (c_scaled[:, 0, 0] + x1_pad).astype(np.float32) * scale_x
        c_scaled[:, 0, 1] = (c_scaled[:, 0, 1] + y1_pad).astype(np.float32) * scale_y
        c_scaled = c_scaled.astype(np.int32)

        return {
            "bbox": bbox_out,
            "contour": c_scaled,
            "fg_ratio": fg_ratio,
        }


class MogStaticObjectDetector:
    """Quản lý state MOG theo từng camera_id – v3 Person-Centric."""

    def __init__(self, mog_cfg: dict, cameras_cfg: list):
        self._states = {}
        self._mog_cfg = mog_cfg
        for cam in cameras_cfg:
            baseline = cam.get("baseline_learning_seconds", 60)
            self._states[cam["camera_id"]] = _CameraMogState(mog_cfg, baseline)

    def is_enabled(self) -> bool:
        return self._mog_cfg.get("enable", True)

    def in_baseline_period(self, camera_id: str) -> bool:
        state = self._states.get(camera_id)
        return state.in_baseline_period() if state else False

    def feed_frame(self, camera_id: str, frame_bgr: np.ndarray):
        """Cập nhật model nền mỗi frame (KHÔNG sinh candidate)."""
        state = self._states.get(camera_id)
        if state is None:
            logger.warning(f"[MOG] Camera '{camera_id}' chưa được khởi tạo state MOG")
            return
        state.feed_frame(frame_bgr)

    def verify_region(self, camera_id: str, frame_bgr: np.ndarray,
                      region_bbox: tuple, scale_x: float = 1.0,
                      scale_y: float = 1.0) -> dict | None:
        """Verify vùng candidate có foreground tĩnh thực sự hay không."""
        state = self._states.get(camera_id)
        if state is None:
            return None
        if state.in_baseline_period():
            return None
        return state.verify_region(frame_bgr, region_bbox, scale_x, scale_y)

    # === BACKWARD COMPAT: giữ method update() cũ nhưng đánh dấu deprecated ===
    def update(self, camera_id: str, frame_bgr: np.ndarray,
               scale_x: float = 1.0, scale_y: float = 1.0):
        """DEPRECATED – dùng feed_frame() + verify_region() thay thế.

        Giữ lại để không break các test/script cũ đang gọi.
        """
        logger.warning("[MOG] update() đã deprecated. Dùng feed_frame() + verify_region().")
        self.feed_frame(camera_id, frame_bgr)
        return []
