"""
Module Tinh Chỉnh Màu Sắc & Ánh Sáng (Color & Tone Adjustment)
Thuần NumPy - Tuyệt đối không dùng OpenCV.
Hỗ trợ điều chỉnh độ sáng (brightness), độ tương phản (contrast), và độ bão hòa màu (saturation).
"""

import numpy as np

from src.io_handler import to_float32


def adjust_brightness(image: np.ndarray, factor: float = 0.0) -> np.ndarray:
    """
    Điều chỉnh độ sáng của ảnh.
    Công thức vector hóa: I' = clip(I + factor, 0.0, 1.0)
    
    Args:
        image: Ma trận ảnh (float32 [0.0, 1.0] hoặc uint8 [0, 255]).
        factor: Độ bù sáng (-0.5 đến +0.5). Giá trị > 0 làm ảnh sáng hơn, < 0 làm tối hơn.
    
    Returns:
        np.ndarray float32 trong dải [0.0, 1.0].
    """
    img = to_float32(image)
    adjusted = img + float(factor)
    return np.clip(adjusted, 0.0, 1.0)


def adjust_contrast(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    Điều chỉnh độ tương phản quanh mức trung bình 0.5.
    Công thức vector hóa: I' = clip((I - 0.5) * factor + 0.5, 0.0, 1.0)
    
    Args:
        image: Ma trận ảnh.
        factor: Hệ số tương phản (0.2 đến 2.5). 1.0 là giữ nguyên.
    
    Returns:
        np.ndarray float32 trong dải [0.0, 1.0].
    """
    img = to_float32(image)
    adjusted = (img - 0.5) * float(factor) + 0.5
    return np.clip(adjusted, 0.0, 1.0)


def adjust_saturation(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    Điều chỉnh độ bão hòa màu sắc (Saturation) dựa trên mô hình phối màu tuyến tính:
    I' = clip(Y + factor * (I - Y), 0.0, 1.0) trong đó Y là độ chói (luminance).
    
    Nếu ảnh đơn sắc (grayscale 2D), trả về ảnh gốc (vì không có thông tin bão hòa màu).
    
    Args:
        image: Ma trận ảnh màu RGB hoặc grayscale.
        factor: Hệ số bão hòa (0.0: ảnh đen trắng, 1.0: giữ nguyên, > 1.0: màu rực rỡ).
    
    Returns:
        np.ndarray float32 trong dải [0.0, 1.0].
    """
    img = to_float32(image)
    
    # Nếu ảnh xám (2D hoặc kênh đơn), không có sắc màu để điều chỉnh
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 1:
        return img
        
    # Tính độ chói chuẩn ITU-R BT.601: Y = 0.299R + 0.587G + 0.114B
    y = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    y = y[:, :, np.newaxis]  # Shape (H, W, 1) để broadcast qua 3 kênh R, G, B
    
    adjusted = y + float(factor) * (img - y)
    return np.clip(adjusted, 0.0, 1.0)


def apply_tone_adjustments(
    image: np.ndarray,
    brightness: float = 0.0,
    contrast: float = 1.0,
    saturation: float = 1.0
) -> np.ndarray:
    """
    Áp dụng chuỗi tinh chỉnh độ sáng, độ tương phản và bão hòa màu liên hoàn.
    
    Args:
        image: Ma trận ảnh đầu vào.
        brightness: Độ bù sáng (-0.5 đến 0.5).
        contrast: Hệ số tương phản (0.2 đến 2.5).
        saturation: Hệ số bão hòa (0.0 đến 2.5).
        
    Returns:
        np.ndarray float32 trong dải [0.0, 1.0].
    """
    result = to_float32(image)
    if brightness != 0.0:
        result = adjust_brightness(result, factor=brightness)
    if contrast != 1.0:
        result = adjust_contrast(result, factor=contrast)
    if saturation != 1.0:
        result = adjust_saturation(result, factor=saturation)
    return result
