"""
Module: sketch.py
Mô tả: Pipeline tạo hiệu ứng tranh phác thảo chì (Pencil Sketch & Color Pencil) cao cấp thuần NumPy.
DoD:
- Chuyển đổi mức xám I_gray chuẩn ITU-R BT.601.
- Multi-scale Color Dodge Blending bóc tách nét phác thảo chì sắc nét và bóng mờ trung gian.
- Kéo dãn tương phản nét chì (Contrast Stretching) giúp đường nét đậm nét như chì than 4B-6B.
- Tăng cường chiều sâu than chì tự nhiên (Graphite Depth Shading).
- Vi cấu trúc vân giấy vẽ phác thảo mỹ thuật (Paper Tooth Grain).
- Tranh chì màu (Color Pencil Sketch) theo mô hình trừ sắc tố sáp màu trên giấy trắng (Subtractive Pigment).
"""

import numpy as np

from .filters import gaussian_blur
from .grayscale import to_grayscale


def color_dodge(gray: np.ndarray, blur: np.ndarray) -> np.ndarray:
    """
    Hòa trộn Color Dodge Blending theo công thức chuẩn giải tích:
        I_sketch = min(255, (I_gray * 256) / (255 - I_blur + 1))
        
    Tham số:
        gray: Ảnh xám thang đo [0.0, 255.0].
        blur: Ảnh mờ nghịch đảo thang đo [0.0, 255.0].
        
    Trả về:
        np.ndarray: Mảng float32 trong thang [0.0, 255.0].
    """
    g = gray.astype(np.float32)
    b = blur.astype(np.float32)
    # Thêm +1.0 ở mẫu số để tránh lỗi chia cho 0
    dodge = (g * 256.0) / (255.0 - b + 1.0)
    return np.clip(dodge, 0.0, 255.0)


def pencil_sketch(
    image: np.ndarray,
    blur_size: int = 21,
    blur_sigma: float = 10.0,
    as_float: bool = True
) -> np.ndarray:
    """
    Pipeline chuyển ảnh sang tranh phác thảo chì mỹ thuật cao cấp (Pencil Sketch).
    
    Quy trình:
    1. Chuyển đổi mức xám chuẩn mắt người ITU-R BT.601.
    2. Multi-scale Color Dodge Blending:
       - Lớp nét mảnh (Fine stroke): Bóc tách các chi tiết mắt, lông mày, nếp gấp và viền khối.
       - Lớp bóng mờ (Medium shading): Tạo mảng bóng chuyển sắc dịu mắt theo blur_size và blur_sigma.
    3. Kéo dãn tương phản nét chì (Contrast Stretching) để nét vẽ đạt độ sâu than chì 4B-6B,
       loại bỏ hoàn toàn tình trạng tranh bị nhạt nhẽo hay bạc màu.
    4. Phủ bóng than chì (Graphite Depth Shading) cho các vùng tối tự nhiên (đồng tử, tóc đen, nếp áo).
    5. Vi cấu trúc vân giấy vẽ phác thảo mỹ thuật (Paper Tooth Grain) tạo cảm giác vẽ trên sổ ký họa.
    
    Tham số:
        image: Ảnh đầu vào 2D (H, W) hoặc 3D (H, W, C), dải [0, 1] hoặc [0, 255].
        blur_size: Kích thước kernel làm mờ (số lẻ, mặc định 21).
        blur_sigma: Bán kính làm mờ Gaussian (mặc định 10.0).
        as_float: Nếu True trả về float32 [0.0, 1.0]. Nếu False trả về uint8 [0, 255].
        
    Trả về:
        np.ndarray: Ma trận 2D phác thảo nét chì chân thực, có chiều sâu mỹ thuật.
    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    # 1. Chuyển đổi sang ảnh mức xám 2D chuẩn hóa
    gray = to_grayscale(image)
    if gray.max() <= 1.0:
        gray_255 = gray.astype(np.float32) * 255.0
        gray_norm = gray.astype(np.float32)
    else:
        gray_255 = gray.astype(np.float32)
        gray_norm = gray.astype(np.float32) / 255.0

    H, W = gray_norm.shape[:2]
    inv_255 = 255.0 - gray_255

    # 2. Multi-scale Color Dodge Blending
    # Lớp nét vẽ mảnh (Fine details)
    fine_size = 7
    fine_sigma = 2.0
    blur_fine = gaussian_blur(inv_255, size=fine_size, sigma=fine_sigma)
    dodge_fine = color_dodge(gray_255, blur_fine) / 255.0
    stroke_fine = np.power(dodge_fine, 2.8)

    # Lớp bóng mờ theo thanh trượt người dùng (User-controlled medium shading)
    effective_size = blur_size if blur_size % 2 == 1 else blur_size + 1
    blur_med = gaussian_blur(inv_255, size=effective_size, sigma=blur_sigma)
    dodge_med = color_dodge(gray_255, blur_med) / 255.0
    stroke_med = np.power(dodge_med, 1.8)

    # Kết hợp hai tỷ lệ: 65% nét mảnh sắc sảo + 35% bóng mờ
    sketch = 0.65 * stroke_fine + 0.35 * stroke_med

    # 3. Phủ bóng than chì (Graphite Depth Shading) cho mái tóc và vùng tối sâu
    graphite = np.clip(0.35 + 0.65 * np.power(np.clip(gray_norm, 0.0, 1.0), 0.55), 0.0, 1.0)
    sketch = np.clip(sketch * graphite, 0.0, 1.0)

    # 4. Vi cấu trúc vân giấy vẽ phác thảo (Paper Tooth Grain)
    y_grid, x_grid = np.mgrid[0:H, 0:W]
    paper_grain = 1.0 - 0.015 * np.sin(x_grid * 1.9 + y_grid * 1.6) - 0.015 * np.cos(x_grid * 1.3 - y_grid * 2.2)
    final_sketch = np.clip(sketch * paper_grain, 0.0, 1.0)

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
    Pipeline tạo tranh phác thảo chì màu nghệ thuật cao cấp (Color Pencil Sketch).

    Nguyên lý mỹ thuật:
    - Mô hình Trừ sắc tố sáp màu trên giấy trắng (Subtractive Wax Pigment Model):
      Trên giấy vẽ trắng, người họa sĩ dùng bút chì màu sáp (Prismacolor) tô đè lên giấy.
      Nơi có nét chì và bóng đổ, sắc tố màu sáp được lắng đọng với độ tươi cao.
      Nơi ánh sáng chiếu mạnh, nền giấy trắng lộ ra tự nhiên.
    - Kết hợp vân giấy ký họa và nét chì mảnh định hình đường nét khuôn mặt, trang phục.
    """
    from .color_adjust import adjust_saturation
    from .io_handler import to_float32, to_uint8

    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    # Nếu ảnh xám, trả về tranh chì thông thường
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        return pencil_sketch(image, blur_size=blur_size, blur_sigma=blur_sigma, as_float=as_float)

    img_float = to_float32(image)
    H, W = img_float.shape[:2]

    # Tính toán độ đậm của nét chì phác thảo (Pencil Stroke Lead)
    gray = to_grayscale(img_float)
    gray_norm = np.clip(gray, 0.0, 1.0)
    inv_255 = (1.0 - gray_norm) * 255.0
    gray_255 = gray_norm * 255.0

    # Bóc tách nét chì mảnh và nét trung gian
    blur_fine = gaussian_blur(inv_255, size=7, sigma=2.0)
    d_fine = color_dodge(gray_255, blur_fine) / 255.0
    stroke_fine = np.power(d_fine, 2.5)

    effective_size = blur_size if blur_size % 2 == 1 else blur_size + 1
    blur_med = gaussian_blur(inv_255, size=effective_size, sigma=blur_sigma)
    d_med = color_dodge(gray_255, blur_med) / 255.0
    stroke_med = np.power(d_med, 1.8)

    # Lượng chì màu lắng đọng: từ 0.0 (giấy trắng) đến 1.0 (chì màu đậm đặc)
    pencil_darkness = 1.0 - (0.60 * stroke_fine + 0.40 * stroke_med)
    shadow_darkness = (1.0 - np.power(gray_norm, 0.70)) * 0.70
    total_lead = np.clip(pencil_darkness + shadow_darkness, 0.0, 1.0)

    # Tăng độ tươi tắn sắc tố sáp màu Prismacolor
    pigment = adjust_saturation(img_float, factor=1.40)

    # Hòa trộn màu trừ trên nền giấy trắng:
    # Nền giấy trắng (1.0) trừ đi lượng sắc tố chì màu tương ứng
    color_pencil = 1.0 - total_lead[:, :, np.newaxis] * (1.0 - pigment * 0.85)
    color_pencil = np.clip(color_pencil, 0.0, 1.0)

    # Vi cấu trúc vân giấy vẽ chì màu
    y_grid, x_grid = np.mgrid[0:H, 0:W]
    paper_grain = 1.0 - 0.015 * np.sin(x_grid * 1.9 + y_grid * 1.6) - 0.015 * np.cos(x_grid * 1.3 - y_grid * 2.2)
    color_pencil = np.clip(color_pencil * paper_grain[:, :, np.newaxis], 0.0, 1.0)

    if as_float:
        return color_pencil.astype(np.float32)
    return to_uint8(color_pencil)
