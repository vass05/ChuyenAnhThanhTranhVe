"""
Module: filters.py
Mô tả: Bộ lọc làm mịn ảnh Gauss (Gaussian Blurring Engine).
DoD:
- Sinh kernel Gauss đối xứng giải tích bằng NumPy vector hóa, tổng kernel = 1.0.
- Tích hợp với động cơ tích chập convolve2d (hỗ trợ cả separable 1D tăng tốc).
- Khử nhiễu, làm mịn ảnh 2D và 3D đa kênh thuần túy, không dùng OpenCV.
"""

import numpy as np

from .convolution import convolve1d_axis, convolve2d


def get_gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
    """
    Tự động sinh ma trận kernel Gauss 1D đối xứng theo giải tích.
    
    Công thức:
        g(x) = exp(-x^2 / (2 * sigma^2))
    
    Tham số:
        size: Kích thước kernel (phải là số lẻ, ví dụ: 3, 5, 7).
        sigma: Độ lệch chuẩn của phân phối Gauss (sigma > 0).
        
    Trả về:
        np.ndarray: Vector 1D kích thước (size,) đã chuẩn hóa tổng bằng 1.0.
    """
    if size % 2 == 0 or size < 1:
        raise ValueError(f"Kích thước size phải là số nguyên dương lẻ, nhận được {size}")
    if sigma <= 0.0:
        raise ValueError(f"Độ lệch chuẩn sigma phải > 0, nhận được {sigma}")

    radius = size // 2
    # Vector hóa tọa độ đối xứng qua tâm [-radius, ..., radius]
    x = np.arange(-radius, radius + 1, dtype=np.float32)

    # Tính giá trị Gauss giải tích
    kernel_1d = np.exp(-(x ** 2) / (2.0 * (sigma ** 2)))

    # Chuẩn hóa để tổng trọng số bằng 1.0 (bảo toàn độ sáng)
    kernel_sum = np.sum(kernel_1d)
    if kernel_sum > 0:
        kernel_1d /= kernel_sum

    return kernel_1d.astype(np.float32)


def get_gaussian_kernel_2d(size: int, sigma: float) -> np.ndarray:
    """
    Tự động sinh ma trận kernel Gauss 2D đối xứng theo giải tích.
    
    Công thức 2D:
        G(x, y) = exp(-(x^2 + y^2) / (2 * sigma^2))
        
    Tham số:
        size: Kích thước ma trận vuông (phải là số lẻ, ví dụ: 3, 5, 7).
        sigma: Độ lệch chuẩn của phân phối Gauss.
        
    Trả về:
        np.ndarray: Ma trận 2D kích thước (size, size) đối xứng hoàn hảo, tổng = 1.0.
    """
    k1d = get_gaussian_kernel_1d(size, sigma)
    # Nhân outer product của hai kernel 1D đối xứng: G(x, y) = g(x) * g(y)
    kernel_2d = np.outer(k1d, k1d)
    # Đảm bảo tổng = 1.0 tuyệt đối
    kernel_2d /= np.sum(kernel_2d)
    return kernel_2d.astype(np.float32)


def gaussian_blur(
    image: np.ndarray,
    size: int = 5,
    sigma: float = 1.0,
    mode: str = "reflect",
    use_separable: bool = True
) -> np.ndarray:
    """
    Bộ lọc làm mịn Gauss (Gaussian Blurring).
    
    Tham số:
        image: Mảng ảnh đầu vào 2D (H, W) hoặc 3D (H, W, C), giá trị [0.0, 1.0] hoặc [0, 255].
        size: Kích thước cửa sổ lọc (số nguyên lẻ, mặc định 5).
        sigma: Bán kính mờ (độ lệch chuẩn Gauss, mặc định 1.0).
        mode: Kiểu đệm biên ('reflect', 'edge', 'constant').
        use_separable: Nếu True, dùng tích chập tách biệt 1D (nhanh hơn gấp nhiều lần).
                       Nếu False, dùng tích chập 2D trực tiếp.
                       
    Trả về:
        np.ndarray: Ảnh đã được làm mịn chống nhiễu, cùng kiểu dữ liệu và cấu trúc với ảnh gốc.
    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    is_uint8 = image.dtype == np.uint8
    img_float = image.astype(np.float32, copy=False)

    if use_separable:
        # Tích chập tách biệt: trục ngang rồi trục dọc
        k1d = get_gaussian_kernel_1d(size, sigma)
        blurred_x = convolve1d_axis(img_float, k1d, axis=1, mode=mode)
        blurred = convolve1d_axis(blurred_x, k1d, axis=0, mode=mode)
    else:
        # Tích chập ma trận 2D trực tiếp
        k2d = get_gaussian_kernel_2d(size, sigma)
        blurred = convolve2d(img_float, k2d, mode=mode)

    if is_uint8:
        return np.clip(np.round(blurred), 0, 255).astype(np.uint8)
    return blurred.astype(np.float32)
