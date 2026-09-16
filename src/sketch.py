"""
Module: sketch.py
Mô tả: Pipeline tạo hiệu ứng tranh phác thảo chì (Pencil Sketch & Color Pencil) từ số 0 thuần NumPy.
DoD:
- Chuyển đổi mức xám I_gray.
- Nghịch đảo I_inv = 255 - I_gray.
- Gaussian Blur trên I_inv tạo bóng mờ I_blur.
- Hòa trộn Color Dodge Blending:
    I_sketch = min(255, (I_gray * 256) / (255 - I_blur + 1))
- Tăng cường chiều sâu than chì tự nhiên (Graphite Depth Shading).
- Tranh chì màu (Color Pencil Sketch) đa kênh rực rỡ.
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
        np.ndarray: Ma trận 2D phác thảo nét chì chân thực với chiều sâu than chì.
    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    # 1. Chuyển đổi sang ảnh mức xám 2D
    gray = to_grayscale(image)
    if gray.max() <= 1.0:
        gray_255 = gray.astype(np.float32) * 255.0
        gray_norm = gray.astype(np.float32)
    else:
        gray_255 = gray.astype(np.float32)
        gray_norm = gray.astype(np.float32) / 255.0

    # 2. Nghịch đảo mức xám: I_inv = 255 - I_gray
    inv_255 = 255.0 - gray_255

    # 3. Gaussian Blur trên I_inv tạo bóng mờ I_blur
    blur_255 = gaussian_blur(inv_255, size=blur_size, sigma=blur_sigma)

    # 4. Color Dodge Blending
    sketch_255 = color_dodge(gray_255, blur_255)
    sketch_norm = sketch_255 / 255.0

    # 5. Tăng cường chiều sâu than chì (Graphite Depth Shading)
    # Giữ nền giấy trắng sạch sẽ nhưng tạo độ chuyển sắc đậm đà cho tóc, mắt, nếp gấp áo
    depth_shading = 0.70 + 0.30 * np.power(gray_norm, 0.8)
    final_sketch = np.clip(sketch_norm * depth_shading, 0.0, 1.0)

    if as_float:
        return final_sketch.astype(np.float32)
    return np.clip(np.round(final_sketch * 255.0), 0, 255).astype(np.uint8)


def color_pencil_sketch(
    image: np.ndarray,
    blur_size: int = 21,
    blur_sigma: float = 10.0,
    as_float: bool = True
) -> np.ndarray:
    """
    Pipeline tạo tranh phác thảo chì màu nghệ thuật (Color Pencil Sketch).

    Phương pháp:
    - Tính ma trận nét vẽ phác thảo chì đơn sắc (I_sketch) từ độ sáng.
    - Hòa trộn nét chì với ma trận màu gốc theo công thức Multiply Shading:
        I_color_sketch = I_rgb * I_sketch
      giúp các nét chì phủ bóng tự nhiên lên từng khối màu gốc.
    """
    from .color_adjust import adjust_saturation
    from .io_handler import to_float32, to_uint8

    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    # Nếu ảnh xám, trả về tranh chì thường
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        return pencil_sketch(image, blur_size=blur_size, blur_sigma=blur_sigma, as_float=as_float)

    img_float = to_float32(image)
    sketch_bw = pencil_sketch(img_float, blur_size=blur_size, blur_sigma=blur_sigma, as_float=True)

    # Tăng nhẹ độ rực rỡ sắc tố cho hiệu ứng chì màu Prismacolor
    vibrant_img = adjust_saturation(img_float, factor=1.20)

    # Shading đa kênh: Nhân từng kênh màu RGB với ma trận bóng chì
    color_sketch = vibrant_img * sketch_bw[:, :, np.newaxis]
    color_sketch = np.clip(color_sketch, 0.0, 1.0)

    if as_float:
        return color_sketch.astype(np.float32)
    return to_uint8(color_sketch)
