"""
Module: cartoon.py
Mô tả: Pipeline tạo hiệu ứng tranh hoạt hình (Cartoonify / Stylization) thuần NumPy.
DoD:
- 2 lượt Bilateral Filter làm phẳng texture bảo toàn biên.
- Lượng tử hóa màu sắc (Color Quantization) thành N khoảng.
- Lấy mask biên Sobel đảo màu (nền trắng nét đen).
- Phủ đè nét biên (nhân ma trận) lên ảnh lượng tử hóa.
"""

import numpy as np

from .edges import compute_gradients, create_edge_mask
from .filters import bilateral_filter, gaussian_blur
from .grayscale import to_grayscale


def quantize_colors(image: np.ndarray, num_levels: int = 8) -> np.ndarray:
    """
    Lượng tử hóa màu sắc (Color Quantization) chia dải cường độ thành N khoảng cố định.
    Tạo hiệu ứng phân mảng màu phẳng (cel-shading) của truyện tranh/hoạt hình.
    
    Tham số:
        image: Mảng ảnh kiểu float32 dải [0.0, 1.0] hoặc uint8 [0, 255].
        num_levels: Số mức cường độ màu trên mỗi kênh (mặc định 8).
        
    Trả về:
        np.ndarray: Ảnh đã được làm phẳng mức màu.
    """
    if num_levels < 2:
        raise ValueError("num_levels phải >= 2")

    is_uint8 = image.dtype == np.uint8
    img_float = image.astype(np.float32, copy=False)

    if img_float.max() <= 1.0:
        # Thang [0.0, 1.0]
        step = num_levels - 1
        quantized = np.round(img_float * step) / step
        res = np.clip(quantized, 0.0, 1.0)
    else:
        # Thang [0, 255]
        step = 255.0 / (num_levels - 1)
        quantized = np.round(img_float / step) * step
        res = np.clip(quantized, 0.0, 255.0)

    if is_uint8:
        return np.round(res).astype(np.uint8)
    return res.astype(np.float32)


def cartoonify(
    image: np.ndarray,
    num_levels: int = 8,
    d: int = 7,
    sigma_s: float = 7.0,
    sigma_r: float = 0.1,
    edge_threshold: float = 0.15,
    as_float: bool = True
) -> np.ndarray:
    """
    Pipeline hoàn chỉnh tạo tranh hoạt hình (Cartoonify) từ ảnh bất kỳ.
    
    Tham số:
        image: Mảng ảnh đầu vào 2D (H, W) hoặc 3D (H, W, C).
        num_levels: Số mức lượng tử hóa màu (mặc định 8).
        d: Đường kính cửa sổ Bilateral (mặc định 7).
        sigma_s: Độ lệch chuẩn không gian Bilateral (mặc định 7.0).
        sigma_r: Độ lệch chuẩn màu sắc Bilateral (mặc định 0.1 cho ảnh [0, 1]).
        edge_threshold: Ngưỡng trích xuất nét vẽ viền Sobel (mặc định 0.15).
        as_float: Trả về float32 [0, 1] hay uint8 [0, 255].
        
    Trả về:
        np.ndarray: Ảnh hoạt hình với mảng màu cel-shaded và viền nét vẽ đen sắc sảo.
    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    # Chuẩn hóa về float32 [0.0, 1.0]
    if image.max() > 1.0:
        img_norm = image.astype(np.float32) / 255.0
    else:
        img_norm = image.astype(np.float32)

    # Task 8.3: Trích xuất nét vẽ biên từ ảnh gốc bằng Sobel
    gray = to_grayscale(img_norm)
    # Khử nhiễu vi mô trước khi tính đạo hàm biên để nét vẽ mượt mà
    gray_blurred = gaussian_blur(gray, size=3, sigma=1.0)
    _, _, magnitude = compute_gradients(gray_blurred)
    # Mask nét vẽ: 0.0 là nét đen, 1.0 là nền trắng
    edge_mask = create_edge_mask(magnitude, threshold=edge_threshold, invert=True)

    # Task 8.1: Giảm chi tiết texture bằng 2 lượt Bilateral Filter
    smooth = bilateral_filter(img_norm, d=d, sigma_s=sigma_s, sigma_r=sigma_r)
    smooth = bilateral_filter(smooth, d=d, sigma_s=sigma_s, sigma_r=sigma_r)

    # Task 8.2: Lượng tử hóa màu sắc (Color Quantization)
    quantized = quantize_colors(smooth, num_levels=num_levels)

    # Task 8.3: Phủ đè nét biên đen lên ảnh màu hoạt hình
    if quantized.ndim == 3:
        cartoon = quantized * edge_mask[..., np.newaxis]
    else:
        cartoon = quantized * edge_mask

    cartoon = np.clip(cartoon, 0.0, 1.0)

    if as_float:
        return cartoon.astype(np.float32)
    return np.clip(np.round(cartoon * 255.0), 0, 255).astype(np.uint8)
