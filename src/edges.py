"""
Module: edges.py
Mô tả: Trích xuất biên độ & hướng gradient (Sobel Edge Detection) thuần NumPy.
DoD:
- Sử dụng ma trận Sobel Kx, Ky.
- Tính đạo hàm qua tích chập convolve2d.
- Tính ma trận độ lớn gradient G = sqrt(Gx^2 + Gy^2).
- Phân ngưỡng tạo mask nét vẽ đen trắng (nền trắng nét đen).
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
    invert: bool = True
) -> np.ndarray:
    """
    Phân ngưỡng ma trận độ lớn gradient để tạo mask nét vẽ đen trắng.
    
    Tham số:
        magnitude: Ma trận độ lớn gradient (H, W) trong [0.0, 1.0].
        threshold: Ngưỡng phân tách biên (mặc định 0.15).
        invert:
            - True: Nền trắng (1.0), nét vẽ đen (0.0) -> phục vụ hoạt hình & phác thảo.
            - False: Nền đen (0.0), nét vẽ trắng (1.0).
            
    Trả về:
        np.ndarray: Ma trận mask nhị phân 2D kiểu float32 {0.0, 1.0}.
    """
    is_edge = magnitude >= threshold

    if invert:
        # Nền trắng (1.0), nét đen (0.0)
        mask = np.where(is_edge, 0.0, 1.0).astype(np.float32)
    else:
        # Nền đen (0.0), nét trắng (1.0)
        mask = np.where(is_edge, 1.0, 0.0).astype(np.float32)

    return mask
