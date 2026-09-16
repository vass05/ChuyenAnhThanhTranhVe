"""
Module: grayscale.py
Mô tả: Bộ chuyển đổi ảnh sang mức xám (Grayscale Engine) chuẩn ITU-R BT.601.
DoD: Vector hóa hoàn toàn bằng NumPy, không dùng vòng lặp pixel, đầu ra ma trận 2D (H, W).
"""

import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Chuyển đổi ảnh màu (RGB hoặc RGBA) sang ảnh mức xám (Grayscale) 2D.
    
    Công thức ITU-R BT.601:
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        
    Vector hóa ma trận:
        Không duyệt từng pixel, tính toán song song trên toàn bộ tensor ảnh.
        
    Tham số:
        image: np.ndarray kích thước (H, W), (H, W, 1), (H, W, 3) hoặc (H, W, 4).
        
    Trả về:
        np.ndarray: Ma trận 2D kích thước chính xác (H, W).
    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    # Trường hợp ảnh đã là 2D (H, W)
    if image.ndim == 2:
        return image.copy()

    # Trường hợp ảnh có 1 kênh (H, W, 1)
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0].copy()

    # Trường hợp ảnh RGB hoặc RGBA
    if image.ndim == 3 and image.shape[2] >= 3:
        r = image[..., 0]
        g = image[..., 1]
        b = image[..., 2]

        # Phép toán vector hóa trên toàn mảng 2D
        gray = 0.299 * r + 0.587 * g + 0.114 * b

        # Bảo toàn kiểu dữ liệu đầu vào
        if np.issubdtype(image.dtype, np.integer):
            gray = np.clip(np.round(gray), 0, 255).astype(image.dtype)
        else:
            gray = gray.astype(np.float32)

        # Đảm bảo đầu ra chính xác 2D (H, W)
        assert gray.ndim == 2, f"Đầu ra phải là ma trận 2D, nhận được ndim={gray.ndim}"
        return gray

    raise ValueError(f"Kích thước ma trận ảnh không được hỗ trợ: shape={image.shape}")
