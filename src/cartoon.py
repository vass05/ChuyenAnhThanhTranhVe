"""
Module: cartoon.py
Mô tả: Pipeline tạo hiệu ứng tranh hoạt hình (Cartoonify / Stylization) cao cấp thuần NumPy.
DoD:
- Multi-scale Bilateral Filter làm phẳng texture bảo toàn biên siêu mượt và tối ưu tốc độ.
- Lượng tử hóa màu sắc cel-shading bảo toàn sắc thái màu (Preserve Hue), chống loang màu da.
- Trích xuất nét mực đen mịn màng khử răng cưa (Anti-aliased Ink Outline), lọc sạch nhiễu nền.
- Phủ đè nét viền lên mảng màu hoạt hình rực rỡ.
"""

import numpy as np

from .color_adjust import adjust_saturation
from .edges import compute_gradients
from .filters import bilateral_filter, gaussian_blur
from .grayscale import to_grayscale
from .resizer import resize_bilinear, resize_max_dimension


def quantize_colors(image: np.ndarray, num_levels: int = 8, preserve_hue: bool = True) -> np.ndarray:
    """
    Lượng tử hóa màu sắc (Color Quantization) tạo mảng màu cel-shading hoạt hình.
    
    Khi preserve_hue = True và ảnh là RGB 3 kênh:
        - Lượng tử hóa trên kênh độ chói (Luminance Y) theo chuẩn ITU-R BT.601.
        - Bảo toàn 100% tỷ lệ màu gốc (Hue & Saturation), loại bỏ hoàn toàn hiện tượng
          da người hoặc chuyển sắc bị vỡ thành các mảng màu xanh/nâu loang lổ.
    
    Khi ảnh là 1D/2D đơn sắc hoặc preserve_hue = False:
        - Lượng tử hóa đều trực tiếp trên từng kênh theo N mức cố định.

    Tham số:
        image: Mảng ảnh kiểu float32 dải [0.0, 1.0] hoặc uint8 [0, 255].
        num_levels: Số mức cường độ màu (mặc định 8).
        preserve_hue: Có bảo toàn sắc thái màu hay không.
        
    Trả về:
        np.ndarray: Ảnh đã được làm phẳng mức màu nghệ thuật.
    """
    if num_levels < 2:
        raise ValueError("num_levels phải >= 2")

    is_uint8 = image.dtype == np.uint8
    img_float = image.astype(np.float32, copy=False)

    if is_uint8:
        img_norm = img_float / 255.0
    else:
        img_norm = img_float if img_float.max() <= 1.0 else img_float / 255.0

    step = num_levels - 1

    # Trường hợp ảnh màu RGB và bật bảo toàn sắc thái
    if img_norm.ndim == 3 and img_norm.shape[2] >= 3 and preserve_hue:
        # Độ chói Y = 0.299R + 0.587G + 0.114B
        y = 0.299 * img_norm[:, :, 0] + 0.587 * img_norm[:, :, 1] + 0.114 * img_norm[:, :, 2]
        y_quant = np.round(y * step) / step
        y_quant = np.clip(y_quant, 0.0, 1.0)

        eps = 1e-3
        ratio = (y_quant + eps) / (y + eps)
        ratio = np.clip(ratio, 0.2, 1.8)

        # Áp dụng tỷ lệ cel-shading: giữ nguyên sắc thái màu gốc
        cel = img_norm * ratio[:, :, np.newaxis]
        # Hòa trộn mượt mà: 85% phẳng cel-shading + 15% chuyển sắc tự nhiên
        res = 0.85 * cel + 0.15 * img_norm
        res = np.clip(res, 0.0, 1.0)
    else:
        # Lượng tử hóa trực tiếp trên từng kênh (dành cho ảnh xám 1D/2D)
        quantized = np.round(img_norm * step) / step
        res = np.clip(quantized, 0.0, 1.0)

    if is_uint8:
        return np.round(res * 255.0).astype(np.uint8)
    return res.astype(np.float32)


def cartoonify(
    image: np.ndarray,
    num_levels: int = 8,
    d: int = 7,
    sigma_s: float = 7.0,
    sigma_r: float = 0.1,
    edge_threshold: float = 0.12,
    as_float: bool = True
) -> np.ndarray:
    """
    Pipeline hoạt hình hóa chất lượng cao (High-Fidelity Cartoonify) thuần NumPy:
    - Multi-scale Bilateral Filter làm phẳng siêu tốc và không bị treo giật.
    - Cel-shading Quantization bảo toàn màu sắc da và vật thể.
    - Anti-aliased Ink Outline: Nét mực truyện tranh khử răng cưa, sạch nhiễu.

    Tham số:
        image: Mảng ảnh đầu vào 2D (H, W) hoặc 3D (H, W, C).
        num_levels: Số mức lượng tử hóa màu (mặc định 8).
        d: Đường kính cửa sổ Bilateral (mặc định 7).
        sigma_s: Độ lệch chuẩn không gian Bilateral (mặc định 7.0).
        sigma_r: Độ lệch chuẩn màu sắc Bilateral (mặc định 0.1 cho ảnh [0, 1]).
        edge_threshold: Ngưỡng trích xuất nét vẽ viền Sobel (mặc định 0.12).
        as_float: Trả về float32 [0, 1] hay uint8 [0, 255].
        
    Trả về:
        np.ndarray: Ảnh hoạt hình mượt mà, màu sắc rực rỡ và viền đen sắc sảo.
    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    # Chuẩn hóa về float32 [0.0, 1.0]
    if image.max() > 1.0:
        img_norm = image.astype(np.float32) / 255.0
    else:
        img_norm = image.astype(np.float32)

    orig_h, orig_w = img_norm.shape[:2]

    # Task 1: Làm mịn đa tỷ lệ bằng 2 lượt Bilateral Filter (Multi-scale Fast Bilateral)
    # Nếu ảnh có kích thước lớn (> 650px), downscale để lọc phẳng mảng màu siêu tốc rồi upscale lại
    MAX_BILAT_DIM = 650
    if max(orig_h, orig_w) > MAX_BILAT_DIM:
        small_img, _ = resize_max_dimension(img_norm, max_dim=MAX_BILAT_DIM)
        smooth_small = bilateral_filter(small_img, d=d, sigma_s=sigma_s, sigma_r=sigma_r)
        smooth_small = bilateral_filter(smooth_small, d=d, sigma_s=sigma_s, sigma_r=sigma_r)
        smooth = resize_bilinear(smooth_small, (orig_h, orig_w))
    else:
        smooth = bilateral_filter(img_norm, d=d, sigma_s=sigma_s, sigma_r=sigma_r)
        smooth = bilateral_filter(smooth, d=d, sigma_s=sigma_s, sigma_r=sigma_r)

    # Task 2: Tăng cường sắc thái màu anime nhẹ và lượng tử hóa cel-shading
    if smooth.ndim == 3 and smooth.shape[2] >= 3:
        smooth_vibrant = adjust_saturation(smooth, factor=1.15)
        quantized = quantize_colors(smooth_vibrant, num_levels=num_levels, preserve_hue=True)
    else:
        quantized = quantize_colors(smooth, num_levels=num_levels, preserve_hue=False)

    # Task 3: Trích xuất nét vẽ biên khử răng cưa và sạch nhiễu
    gray = to_grayscale(img_norm)
    # Điều chỉnh độ mịn theo độ phân giải ảnh để khử nhiễu vi mô trên nền/tường/sàn
    blur_k = 5 if max(orig_h, orig_w) >= 800 else 3
    blur_sig = 1.2 if max(orig_h, orig_w) >= 800 else 0.8
    gray_blurred = gaussian_blur(gray, size=blur_k, sigma=blur_sig)
    _, _, mag = compute_gradients(gray_blurred)

    # Tạo nét mực chuyển tiếp mượt mà nhưng đậm và rõ ràng (Bold Anti-aliased Ink Outline)
    t_low = edge_threshold * 0.75
    t_high = edge_threshold * 1.25
    edge_factor = np.clip((mag - t_low) / max(t_high - t_low, 1e-4), 0.0, 1.0)
    # Tăng độ đậm nét viền bằng hàm lũy thừa để đường viền rõ nét như truyện tranh
    edge_mask = np.power(1.0 - edge_factor, 1.5)

    # Task 4: Hòa trộn nét viền lên mảng màu hoạt hình
    if quantized.ndim == 3:
        cartoon = quantized * edge_mask[:, :, np.newaxis]
    else:
        cartoon = quantized * edge_mask

    cartoon = np.clip(cartoon, 0.0, 1.0)

    if as_float:
        return cartoon.astype(np.float32)
    return np.clip(np.round(cartoon * 255.0), 0, 255).astype(np.uint8)
