"""
Module: edges.py
Mô tả: Trích xuất biên độ & hướng gradient (Sobel Edge Detection) thuần NumPy.
DoD:
- Sử dụng ma trận Sobel Kx, Ky.
- Tính đạo hàm qua tích chập convolve2d.
- Tính ma trận độ lớn gradient G = sqrt(Gx^2 + Gy^2).
- Phân ngưỡng tạo mask nét vẽ đen trắng (hỗ trợ cả nhị phân và khử răng cưa mượt mà).
"""

import numpy as np

from .convolution import convolve2d
from .grayscale import to_grayscale

# Khởi tạo hai ma trận lọc Sobel 3x3 chuẩn giải tích
SOBEL_KX = np.array([
    [-1.0, 0.0, 1.0],
    [-2.0, 0.0, 2.0],
    [-1.0, 0.0, 1.0]
], dtype=np.float32)

SOBEL_KY = np.array([
    [-1.0, -2.0, -1.0],
    [ 0.0,  0.0,  0.0],
    [ 1.0,  2.0,  1.0]
], dtype=np.float32)


def get_sobel_kernels() -> tuple[np.ndarray, np.ndarray]:
    """Trả về cặp ma trận kernel Sobel (Kx, Ky)."""
    return SOBEL_KX.copy(), SOBEL_KY.copy()


def compute_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tính đạo hàm riêng bậc nhất Gx, Gy và độ lớn gradient G.
    
    Tham số:
        image: Mảng ảnh đầu vào 2D (H, W) hoặc 3D RGB (H, W, 3), dải [0.0, 1.0].
        
    Trả về:
        Tuple (Gx, Gy, magnitude):
            - Gx: Đạo hàm theo phương ngang.
            - Gy: Đạo hàm theo phương dọc.
            - magnitude: Ma trận độ lớn gradient G = sqrt(Gx^2 + Gy^2), chuẩn hóa về [0.0, 1.0].
    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    # Chuyển về mức xám 2D nếu là ảnh màu
    gray = to_grayscale(image)
    if gray.max() > 1.0:
        gray = gray.astype(np.float32) / 255.0
    else:
        gray = gray.astype(np.float32)

    kx, ky = get_sobel_kernels()

    # Tích chập tính đạo hàm bậc nhất theo 2 chiều
    gx = convolve2d(gray, kx, mode="reflect")
    gy = convolve2d(gray, ky, mode="reflect")

    # Độ lớn gradient: G = sqrt(Gx^2 + Gy^2)
    magnitude = np.hypot(gx, gy)

    # Chuẩn hóa độ lớn về dải [0.0, 1.0]
    max_val = float(magnitude.max())
    if max_val > 0.0:
        norm_mag = magnitude / max_val
    else:
        norm_mag = magnitude

    return gx, gy, norm_mag.astype(np.float32)


def create_edge_mask(
    magnitude: np.ndarray,
    threshold: float = 0.15,
    invert: bool = True,
    smooth: bool = False
) -> np.ndarray:
    """
    Phân ngưỡng ma trận độ lớn gradient để tạo mask nét vẽ đen trắng.
    
    Tham số:
        magnitude: Ma trận độ lớn gradient (H, W) trong [0.0, 1.0].
        threshold: Ngưỡng phân tách biên (mặc định 0.15).
        invert:
            - True: Nền trắng, nét vẽ mực đen -> phục vụ hoạt hình, phác thảo & in ấn.
            - False: Nền đen, nét vẽ sáng trắng.
        smooth:
            - False: Phân ngưỡng nhị phân cứng {0.0, 1.0}.
            - True: Phân ngưỡng mượt khử răng cưa (Anti-aliased Line Art), nét vẽ thanh thoát.
            
    Trả về:
        np.ndarray: Ma trận mask 2D kiểu float32.
    """
    if not smooth:
        is_edge = magnitude >= threshold
        if invert:
            return np.where(is_edge, 0.0, 1.0).astype(np.float32)
        return np.where(is_edge, 1.0, 0.0).astype(np.float32)

    # Chế độ khử răng cưa mượt mà (Anti-aliased Smoothstep)
    t_low = threshold * 0.70
    t_high = threshold * 1.30
    edge_str = np.clip((magnitude - t_low) / max(t_high - t_low, 1e-4), 0.0, 1.0)
    edge_str = np.power(edge_str, 1.2)

    if invert:
        # Nền trắng (1.0), nét mực đen đậm (0.0 đến 1.0)
        mask = 1.0 - edge_str * 0.95
    else:
        # Nền đen (0.0), nét vẽ sáng trắng (0.0 đến 1.0)
        mask = edge_str * 0.95

    return mask.astype(np.float32)
