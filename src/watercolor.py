"""
Module Hiệu Ứng Tranh Màu Nước (Watercolor Art Effect)
Thuần NumPy - Tuyệt đối không dùng OpenCV.

Quy trình thuật toán:
1. Làm mịn mảng màu và khử chi tiết vụn bằng Bilateral Filter bảo toàn biên.
2. Trích xuất đường viền nét vẽ mềm (Soft Edge Mask) bằng Sobel và làm mờ Gaussian nhẹ.
3. Tăng cường độ rực rỡ sắc tố màu (Color Saturation Boost).
4. Hòa trộn lớp màu nước với nét viền để tạo cảm giác loang màu tự nhiên trên giấy vẽ.
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
    saturation_boost: float = 1.25,
    as_float: bool = True
) -> np.ndarray:
    """
    Tạo hiệu ứng tranh màu nước (Watercolor Effect).

    Args:
        image: Ma trận ảnh (float32 hoặc uint8).
        bilat_d: Bán kính cửa sổ lọc song phương.
        bilat_sigma_s: Độ lệch không gian cho lọc song phương.
        bilat_sigma_r: Độ lệch màu sắc cho lọc song phương.
        edge_threshold: Ngưỡng phát hiện nét vẽ mềm.
        edge_strength: Độ đậm của nét viền hòa trộn (0.0 đến 1.0).
        saturation_boost: Hệ số tăng bão hòa sắc màu (1.0 là giữ nguyên).
        as_float: Nếu True trả về float32 [0.0, 1.0], False trả về uint8 [0, 255].

    Returns:
        np.ndarray chứa ảnh kết quả tranh màu nước.
    """
    img = to_float32(image)

    # 1. Làm mịn bảo toàn biên bằng Bilateral Filter đa tỷ lệ (Multi-scale Fast Bilateral)
    orig_h, orig_w = img.shape[:2]
    MAX_BILAT_DIM = 650
    if max(orig_h, orig_w) > MAX_BILAT_DIM:
        from src.resizer import resize_bilinear, resize_max_dimension
        small_img, _ = resize_max_dimension(img, max_dim=MAX_BILAT_DIM)
        smoothed_small = bilateral_filter(
            small_img,
            d=bilat_d,
            sigma_s=bilat_sigma_s,
            sigma_r=bilat_sigma_r
        )
        smoothed = resize_bilinear(smoothed_small, (orig_h, orig_w))
    else:
        smoothed = bilateral_filter(
            img,
            d=bilat_d,
            sigma_s=bilat_sigma_s,
            sigma_r=bilat_sigma_r
        )

    # 2. Tạo nét viền mềm (Soft Edge)
    gray = to_grayscale(img)
    _, _, mag = compute_gradients(gray)
    
    # Chuẩn hóa ma trận gradient về [0.0, 1.0]
    mag_max = float(np.max(mag))
    if mag_max > 1e-6:
        mag = mag / mag_max

    # Phân ngưỡng mềm: điểm vượt ngưỡng sẽ tạo bóng nét cọ
    soft_edge = np.clip((mag - edge_threshold) / max(1.0 - edge_threshold, 1e-5), 0.0, 1.0)
    
    # Làm mờ nét cọ một chút để tạo độ loang mềm mại
    soft_edge = gaussian_blur(soft_edge, size=3, sigma=1.0)

    # 3. Tăng bão hòa sắc tố màu và tương phản
    if smoothed.ndim == 3 and smoothed.shape[2] == 3:
        color_boosted = adjust_saturation(smoothed, factor=saturation_boost)
        color_boosted = adjust_contrast(color_boosted, factor=1.1)
    else:
        color_boosted = smoothed

    # 4. Hòa trộn màu với nét viền loang (làm tối vùng biên)
    if color_boosted.ndim == 3:
        edge_factor = 1.0 - (edge_strength * soft_edge[:, :, np.newaxis])
    else:
        edge_factor = 1.0 - (edge_strength * soft_edge)

    watercolor = color_boosted * edge_factor
    watercolor = np.clip(watercolor, 0.0, 1.0)

    return watercolor.astype(np.float32) if as_float else to_uint8(watercolor)
