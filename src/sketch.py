"""
Module: sketch.py
Mô tả: Pipeline tạo hiệu ứng tranh phác thảo chì (Pencil Sketch) từ số 0 thuần NumPy.
DoD:
- Chuyển đổi mức xám I_gray.
- Nghịch đảo I_inv = 255 - I_gray.
- Gaussian Blur trên I_inv tạo bóng mờ I_blur.
- Hòa trộn Color Dodge Blending:
    I_sketch = min(255, (I_gray * 256) / (255 - I_blur + 1))
"""

import numpy as np

from .filters import gaussian_blur
from .grayscale import to_grayscale


def color_dodge(gray: np.ndarray, blur: np.ndarray) -> np.ndarray:
    """
    Hòa trộn Color Dodge Blending theo công thức chuẩn:
        I_sketch = min(255, (I_gray * 256) / (255 - I_blur + 1))
        
    Tham số:
        gray: Ảnh xám thang đo [0.0, 255.0].
        blur: Ảnh mờ nghịch đảo thang đo [0.0, 255.0].
        
    Trả về:
        np.ndarray: Mảng float32 trong thang [0.0, 255.0].
    """
    g = gray.astype(np.float32)
    b = blur.astype(np.float32)
    # Tránh chia cho 0 với +1.0 ở mẫu số
    dodge = (g * 256.0) / (255.0 - b + 1.0)
    return np.clip(dodge, 0.0, 255.0)


def pencil_sketch(
    image: np.ndarray,
    blur_size: int = 21,
    blur_sigma: float = 10.0,
    as_float: bool = True
) -> np.ndarray:
    """
    Pipeline hoàn chỉnh chuyển ảnh bất kỳ sang tranh phác thảo chì (Pencil Sketch).
    
    Tham số:
        image: Ảnh đầu vào 2D (H, W) hoặc 3D (H, W, C), dải [0, 1] hoặc [0, 255].
        blur_size: Kích thước kernel làm mờ (mặc định 21).
        blur_sigma: Bán kính làm mờ Gaussian (mặc định 10.0).
        as_float: Nếu True, trả về float32 [0.0, 1.0]. Nếu False, trả về uint8 [0, 255].
        
    Trả về:
        np.ndarray: Ma trận 2D phác thảo nét chì chân thực.
    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    # 1. Chuyển đổi sang ảnh mức xám 2D
    gray = to_grayscale(image)
    if gray.max() <= 1.0:
        gray_255 = gray.astype(np.float32) * 255.0
    else:
        gray_255 = gray.astype(np.float32)

    # 2. Nghịch đảo mức xám: I_inv = 255 - I_gray
    inv_255 = 255.0 - gray_255

    # 3. Gaussian Blur trên I_inv tạo bóng mờ I_blur
    blur_255 = gaussian_blur(inv_255, size=blur_size, sigma=blur_sigma)

    # 4. Color Dodge Blending
    sketch_255 = color_dodge(gray_255, blur_255)

    if as_float:
        return (sketch_255 / 255.0).astype(np.float32)
    return np.clip(np.round(sketch_255), 0, 255).astype(np.uint8)
