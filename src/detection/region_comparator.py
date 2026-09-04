"""
RegionComparator – so sánh ảnh crop TRƯỚC / SAU khi person rời đi
để phát hiện vật thể mới xuất hiện tại vùng đó.

Dùng kết hợp:
  - Structural Similarity Index (SSIM): nhạy với cấu trúc, bất biến ánh sáng
    nhẹ  →  phát hiện vật mới mà không bị ảnh hưởng bởi biến đổi ánh sáng
    đều (illuminate change).
  - Contour-based diff: tìm contour mới trong vùng diff → trả về bbox ôm sát
    vật mới xuất hiện.

Không dùng cho toàn frame – chỉ gọi trên crop nhỏ (vùng person từng đứng).
"""
import cv2
import numpy as np
import logging

logger = logging.getLogger("aod.region_cmp")


def _ssim_map(img_a_gray: np.ndarray, img_b_gray: np.ndarray):
    """Tính SSIM map pixel-wise (simplified, không cần skimage)."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    k_size = (11, 11)
    sigma = 1.5

    mu_a = cv2.GaussianBlur(img_a_gray.astype(np.float64), k_size, sigma)
    mu_b = cv2.GaussianBlur(img_b_gray.astype(np.float64), k_size, sigma)

    mu_a_sq = mu_a ** 2
    mu_b_sq = mu_b ** 2
    mu_ab = mu_a * mu_b

    sigma_a_sq = cv2.GaussianBlur(img_a_gray.astype(np.float64) ** 2, k_size, sigma) - mu_a_sq
    sigma_b_sq = cv2.GaussianBlur(img_b_gray.astype(np.float64) ** 2, k_size, sigma) - mu_b_sq
    sigma_ab = cv2.GaussianBlur(
        img_a_gray.astype(np.float64) * img_b_gray.astype(np.float64), k_size, sigma
    ) - mu_ab

    numerator = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
    denominator = (mu_a_sq + mu_b_sq + C1) * (sigma_a_sq + sigma_b_sq + C2)

    ssim_map = numerator / (denominator + 1e-12)
    return ssim_map


def compare_regions(
    ref_crop_bgr: np.ndarray,
    cur_crop_bgr: np.ndarray,
    ssim_threshold: float = 0.85,
    min_contour_area_px: int = 300,
    morph_kernel_size: int = 7,
) -> list[dict]:
    """So sánh 2 crop (reference vs current) cùng kích thước.

    Trả về list[dict] {bbox=(x1,y1,x2,y2), contour, ssim_score} cho các vùng
    có vật mới xuất hiện.

    Parameters
    ----------
    ref_crop_bgr : ảnh BGR reference (trước khi person đặt đồ / lúc chưa có vật)
    cur_crop_bgr : ảnh BGR hiện tại (sau khi person rời đi)
    ssim_threshold : ngưỡng SSIM, vùng nào < threshold coi là có thay đổi
    min_contour_area_px : diện tích contour tối thiểu (lọc nhiễu nhỏ)
    morph_kernel_size : kernel morphology để gom mảnh vỡ diff
    """
    if ref_crop_bgr is None or cur_crop_bgr is None:
        return []

    # resize nếu kích thước khác nhau (do frame resolution thay đổi)
    if ref_crop_bgr.shape[:2] != cur_crop_bgr.shape[:2]:
        cur_crop_bgr = cv2.resize(cur_crop_bgr, (ref_crop_bgr.shape[1], ref_crop_bgr.shape[0]))

    gray_ref = cv2.cvtColor(ref_crop_bgr, cv2.COLOR_BGR2GRAY)
    gray_cur = cv2.cvtColor(cur_crop_bgr, cv2.COLOR_BGR2GRAY)

    # Gaussian blur nhẹ để giảm nhiễu hạt trước khi so sánh
    gray_ref = cv2.GaussianBlur(gray_ref, (5, 5), 0)
    gray_cur = cv2.GaussianBlur(gray_cur, (5, 5), 0)

    # Tính SSIM map
    ssim_map = _ssim_map(gray_ref, gray_cur)
    mean_ssim = float(np.mean(ssim_map))

    # Nếu toàn vùng giống nhau → không có thay đổi
    if mean_ssim >= ssim_threshold:
        return []

    # Tạo diff mask: vùng nào SSIM thấp (thay đổi nhiều) → foreground
    diff_mask = (ssim_map < ssim_threshold).astype(np.uint8) * 255

    # Morphology: gom mảnh vỡ, loại hạt nhiễu
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_contour_area_px:
            continue

        x, y, w, h = cv2.boundingRect(c)
        results.append({
            "bbox": (x, y, x + w, y + h),
            "contour": c,
            "ssim_score": mean_ssim,
            "diff_area": area,
        })

    return results
