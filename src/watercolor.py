"""
Module Hiệu Ứng Tranh Màu Nước (Watercolor Art Effect)
Thuần NumPy - Tuyệt đối không dùng OpenCV (Chương 5).

Quy trình thuật toán mô phỏng chất liệu màu nước chân thực:
1. Lọc song phương 2 lượt (Two-Pass Bilateral Filter) đa tỷ lệ:
   Làm phẳng các hạt nhiễu và chuyển các chi tiết vụn thành từng mảng màu loang tự nhiên.
2. Hiệu ứng đọng sắc tố viền loang (Pigment Pooling / "Coffee-Ring" Effect):
   Tại ranh giới của các vệt nước khô đi, các hạt sắc tố màu tự nhiên bị kéo dồn ra mép
   tạo nên viền sẫm màu đặc trưng của tranh màu nước vẽ tay.
3. Tăng cường độ rực rỡ sắc tố màu nước (Vibrant Pure Pigments) và nâng sáng nền giấy:
   Màu nước vẽ trên giấy Arches/Canson trắng sáng có độ trong suốt và bắt sáng cao.
4. Nét phác chì lót mềm mại (Soft Graphite / Sepia Underdrawing):
   Các đường nét cọ hoặc bút chì phác thảo mềm mại hòa trộn tự nhiên vào lớp màu ướt.
5. Vi cấu trúc vân giấy sần màu nước (Cold-Press Watercolor Paper Tooth).
"""

import numpy as np

from src.color_adjust import adjust_contrast, adjust_saturation
from src.edges import compute_gradients
from src.filters import bilateral_filter, gaussian_blur
from src.grayscale import to_grayscale
from src.io_handler import to_float32, to_uint8


def watercolor_effect(
    image: np.ndarray,
    bilat_d: int = 7,
    bilat_sigma_s: float = 7.0,
    bilat_sigma_r: float = 0.12,
    edge_threshold: float = 0.10,
    edge_strength: float = 0.5,
    saturation_boost: float = 1.35,
    as_float: bool = True
) -> np.ndarray:
    """
    Tạo hiệu ứng tranh màu nước chất lượng cao (Watercolor Effect).

    Args:
        image: Ma trận ảnh (float32 hoặc uint8).
        bilat_d: Bán kính cửa sổ lọc song phương (số lẻ).
        bilat_sigma_s: Độ lệch không gian cho lọc song phương.
        bilat_sigma_r: Độ lệch màu sắc cho lọc song phương.
        edge_threshold: Ngưỡng phát hiện nét vẽ phác mềm.
        edge_strength: Độ đậm của nét cọ viền hòa trộn (0.0 đến 1.0).
        saturation_boost: Hệ số tăng bão hòa sắc màu nước (mặc định 1.35).
        as_float: Nếu True trả về float32 [0.0, 1.0], False trả về uint8 [0, 255].

    Returns:
        np.ndarray chứa ảnh kết quả tranh màu nước sống động.
    """
    img = to_float32(image)
    orig_h, orig_w = img.shape[:2]

    # 1. Làm phẳng mảng màu bằng Lọc song phương 2 lượt (Two-Pass Bilateral Filter)
    MAX_BILAT_DIM = 650
    if max(orig_h, orig_w) > MAX_BILAT_DIM:
        from src.resizer import resize_bilinear, resize_max_dimension
        small_img, _ = resize_max_dimension(img, max_dim=MAX_BILAT_DIM)
        b1 = bilateral_filter(small_img, d=bilat_d, sigma_s=bilat_sigma_s, sigma_r=bilat_sigma_r)
        b2 = bilateral_filter(b1, d=bilat_d, sigma_s=bilat_sigma_s, sigma_r=bilat_sigma_r)
        smoothed = resize_bilinear(b2, (orig_h, orig_w))
    else:
        b1 = bilateral_filter(img, d=bilat_d, sigma_s=bilat_sigma_s, sigma_r=bilat_sigma_r)
        smoothed = bilateral_filter(b1, d=bilat_d, sigma_s=bilat_sigma_s, sigma_r=bilat_sigma_r)

    # 2. Hiệu ứng đọng sắc tố viền loang (Pigment Pooling / Coffee-Ring Effect)
    gray_smooth = to_grayscale(smoothed)
    _, _, mag_smooth = compute_gradients(gray_smooth)
    # Vùng biên của mảng màu sẽ đọng sắc tố sẫm hơn 25-35%
    pooling_factor = 1.0 - 0.35 * np.clip((mag_smooth - 0.03) / 0.12, 0.0, 1.0)
    if smoothed.ndim == 3:
        wash = smoothed * pooling_factor[:, :, np.newaxis]
    else:
        wash = smoothed * pooling_factor

    # 3. Tăng cường sắc tố màu nước và độ sáng trong suốt của giấy vẽ
    if wash.ndim == 3 and wash.shape[2] >= 3:
        wash = adjust_saturation(wash, factor=saturation_boost)
        wash = adjust_contrast(wash, factor=1.08)
        # Nâng nhẹ dải midtone để màu trong suốt, lộ ánh sáng giấy vẽ
        wash = np.power(np.clip(wash, 0.0, 1.0), 0.90)

    # 4. Bóc tách nét phác chì lót mềm mại (Soft Underdrawing)
    gray_orig = to_grayscale(img)
    gray_clean = gaussian_blur(gray_orig, size=3, sigma=1.0)
    _, _, mag_orig = compute_gradients(gray_clean)

    mag_max = float(np.max(mag_orig))
    if mag_max > 1e-6:
        norm_mag = mag_orig / mag_max
    else:
        norm_mag = mag_orig

    # Phân ngưỡng mềm mại cho nét cọ chì lót
    soft_edge = np.clip((norm_mag - edge_threshold) / max(1.0 - edge_threshold, 1e-5), 0.0, 1.0)
    soft_edge = gaussian_blur(soft_edge, size=3, sigma=0.8)

    # Hòa trộn nét phác chì lót vào lớp màu nước
    if wash.ndim == 3:
        edge_multiplier = 1.0 - (edge_strength * 0.65 * soft_edge[:, :, np.newaxis])
        watercolor = wash * edge_multiplier
    else:
        edge_multiplier = 1.0 - (edge_strength * 0.65 * soft_edge)
        watercolor = wash * edge_multiplier

    # 5. Vi cấu trúc vân giấy sần màu nước (Cold-Press Watercolor Paper Grain)
    y_grid, x_grid = np.mgrid[0:orig_h, 0:orig_w]
    paper_grain = 1.0 - 0.015 * np.sin(x_grid * 1.5 + y_grid * 1.2) - 0.012 * np.cos(x_grid * 0.9 - y_grid * 1.8)

    if watercolor.ndim == 3:
        watercolor = np.clip(watercolor * paper_grain[:, :, np.newaxis], 0.0, 1.0)
    else:
        watercolor = np.clip(watercolor * paper_grain, 0.0, 1.0)

    return watercolor.astype(np.float32) if as_float else to_uint8(watercolor)
