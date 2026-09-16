"""
Module: cartoon.py
Mô tả: Pipeline tạo hiệu ứng tranh hoạt hình phong cách Anime cao cấp thuần NumPy (Chương 5).
DoD:
- Multi-scale Bilateral Filter 2 lượt làm phẳng texture bảo toàn biên mượt mà.
- Cel-shading Quantization bảo toàn 100% sắc thái màu da và nâng sáng bóng đổ (chống sẫm đen).
- Anime Bloom & Dreamy Glow: Trích xuất highlight và tán xạ ánh sáng mộng mơ.
- Intelligent Contour Ink Lines: Nét viền mực G-pen bút sắt sắc sảo, khử nhiễu nền gạch/sàn.
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
        - Lượng tử hóa trên kênh độ chói (Luminance Y) kết hợp nâng sáng bóng đổ.
        - Bảo toàn 100% tỷ lệ màu gốc (Hue & Saturation), loại bỏ hoàn toàn hiện tượng
          da người bị vỡ thành các mảng màu đen hoặc xanh/nâu loang lổ.
    
    Khi ảnh là 1D/2D đơn sắc hoặc preserve_hue = False:
        - Lượng tử hóa đều trực tiếp trên từng kênh theo N mức cố định.
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
        
        # Nâng nhẹ bóng đổ để da mặt và hốc mắt không bị biến thành hố đen
        y_lifted = np.power(np.clip(y, 0.0, 1.0), 0.85)
        y_quant = np.round(y_lifted * step) / step
        y_quant = np.clip(y_quant, 0.10, 1.0)

        eps = 1e-3
        ratio = (y_quant + eps) / (y + eps)
        # Giới hạn tỷ lệ không làm cháy sáng hay sẫm đen đột ngột
        ratio = np.clip(ratio, 0.45, 1.45)

        # Áp dụng tỷ lệ cel-shading: giữ nguyên sắc thái màu gốc
        cel = img_norm * ratio[:, :, np.newaxis]
        # Hòa trộn cân bằng 50% phẳng cel-shading + 50% chuyển sắc mượt mà
        res = 0.50 * cel + 0.50 * img_norm
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
    sigma_r: float = 0.12,
    edge_threshold: float = 0.08,
    as_float: bool = True
) -> np.ndarray:
    """
    Pipeline hoạt hình hóa chất lượng cao chuẩn phong cách Anime thuần NumPy (Chương 5):
    - Multi-scale Bilateral Filter 2 lượt làm phẳng siêu tốc và mịn màng.
    - Cel-shading Quantization bảo toàn màu sắc da và nâng sáng vùng tối.
    - Anime Highlight Bloom: Ánh sáng mộng mơ tán xạ highlight.
    - Intelligent Ink Outline: Nét viền mực G-pen thanh thoát, sạch sẽ, không nhiễu hạt.
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
    MAX_BILAT_DIM = 650
    if max(orig_h, orig_w) > MAX_BILAT_DIM:
        small_img, _ = resize_max_dimension(img_norm, max_dim=MAX_BILAT_DIM)
        smooth_small = bilateral_filter(small_img, d=d, sigma_s=sigma_s, sigma_r=sigma_r)
        smooth_small = bilateral_filter(smooth_small, d=d, sigma_s=sigma_s, sigma_r=sigma_r)
        smooth = resize_bilinear(smooth_small, (orig_h, orig_w))
    else:
        smooth = bilateral_filter(img_norm, d=d, sigma_s=sigma_s, sigma_r=sigma_r)
        smooth = bilateral_filter(smooth, d=d, sigma_s=sigma_s, sigma_r=sigma_r)

    # Task 2: Nâng tông ấm áp Anime và lượng tử hóa cel-shading
    if smooth.ndim == 3 and smooth.shape[2] >= 3:
        # Tăng độ rực rỡ sắc tố tự nhiên cho trang phục và cảnh vật
        smooth_vibrant = adjust_saturation(smooth, factor=1.22)
        # Nâng nhẹ sắc ấm (Warm tone): R + 3%, B - 2%
        smooth_vibrant[:, :, 0] = np.clip(smooth_vibrant[:, :, 0] * 1.03, 0.0, 1.0)
        smooth_vibrant[:, :, 2] = np.clip(smooth_vibrant[:, :, 2] * 0.98, 0.0, 1.0)
        quantized = quantize_colors(smooth_vibrant, num_levels=num_levels, preserve_hue=True)
    else:
        quantized = quantize_colors(smooth, num_levels=num_levels, preserve_hue=False)

    # Task 3: Hiệu ứng Phát sáng Mộng mơ Hoạt hình (Anime Bloom / Highlight Glow)
    if quantized.ndim == 3 and quantized.shape[2] >= 3:
        lum = 0.299 * quantized[:, :, 0] + 0.587 * quantized[:, :, 1] + 0.114 * quantized[:, :, 2]
        hl_mask = np.clip((lum - 0.50) / 0.50, 0.0, 1.0)[:, :, np.newaxis]
        hl_layer = quantized * hl_mask
        glow_blur = gaussian_blur(hl_layer, size=15, sigma=5.0)
        quantized = np.clip(quantized + 0.16 * glow_blur, 0.0, 1.0)

    # Task 4: Trích xuất nét vẽ biên thông minh, sạch nhiễu nền và khử răng cưa
    gray = to_grayscale(img_norm)
    blur_k = 7 if max(orig_h, orig_w) >= 600 else 5
    blur_sig = 1.8 if max(orig_h, orig_w) >= 600 else 1.4
    gray_blurred = gaussian_blur(gray, size=blur_k, sigma=blur_sig)
    _, _, mag = compute_gradients(gray_blurred)

    # Tạo nét mực viền mềm mại, sắc nét, loại bỏ nhiễu nền gạch/sàn
    t_low = edge_threshold * 0.75
    t_high = edge_threshold * 1.35
    edge_factor = np.clip((mag - t_low) / max(t_high - t_low, 1e-4), 0.0, 1.0)
    edge_factor = np.power(edge_factor, 1.3)

    # Task 5: Hòa trộn nét mực than chì sâu sắc (Charcoal Ink Line) lên mảng màu hoạt hình
    if quantized.ndim == 3:
        ink_color = np.array([0.05, 0.05, 0.05], dtype=np.float32)
        edge_3d = edge_factor[:, :, np.newaxis]
        cartoon = quantized * (1.0 - edge_3d) + ink_color * edge_3d
    else:
        cartoon = quantized * (1.0 - edge_factor)

    cartoon = np.clip(cartoon, 0.0, 1.0)

    if as_float:
        return cartoon.astype(np.float32)
    return np.clip(np.round(cartoon * 255.0), 0, 255).astype(np.uint8)
