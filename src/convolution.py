"""
Module: convolution.py
Mô tả: Động cơ tích chập 2D ma trận thuần túy (2D Convolution Engine).
DoD:
- Sử dụng np.pad cho padding_2d.
- Tích chập vector hóa thông qua NumPy array slicing trên kernel, không lặp từng pixel.
- Tối ưu bộ nhớ O(H * W), an toàn tuyệt đối với ảnh độ phân giải cao.
- Không sử dụng bất kỳ hàm nào từ OpenCV (cv2).
"""

from typing import Any

import numpy as np


def padding_2d(
    image: np.ndarray,
    pad_width: int | tuple[int, int] | tuple[tuple[int, int], tuple[int, int]],
    mode: Any = "reflect"
) -> np.ndarray:
    """
    Thực hiện đệm biên ma trận ảnh 2D hoặc 3D sử dụng np.pad.
    
    Tham số:
        image: Mảng ảnh 2D (H, W) hoặc 3D (H, W, C).
        pad_width: Độ rộng đệm:
            - Số nguyên p: đệm đều 4 phía (trên, dưới, trái, phải).
            - Tuple (pad_h, pad_w): đệm đối xứng theo chiều cao và rộng.
            - Tuple ((pad_top, pad_bottom), (pad_left, pad_right)).
        mode: Kiểu đệm của numpy.pad ('reflect', 'edge', 'constant', 'symmetric').
        
    Trả về:
        np.ndarray: Ma trận ảnh sau khi đệm biên.
    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    # Chuẩn hóa cấu trúc pad_width cho 2 chiều không gian (H, W)
    if isinstance(pad_width, int):
        pad_spec_2d = ((pad_width, pad_width), (pad_width, pad_width))
    elif isinstance(pad_width, tuple):
        if len(pad_width) == 2 and isinstance(pad_width[0], int):
            pad_h, pad_w = pad_width
            pad_spec_2d = ((pad_h, pad_h), (pad_w, pad_w))
        elif len(pad_width) == 2 and isinstance(pad_width[0], tuple):
            pad_spec_2d = pad_width
        else:
            raise ValueError(f"pad_width không hợp lệ: {pad_width}")
    else:
        raise TypeError(f"Kiểu dữ liệu pad_width không được hỗ trợ: {type(pad_width)}")

    if image.ndim == 2:
        return np.pad(image, pad_spec_2d, mode=mode)
    elif image.ndim == 3:
        # Giữ nguyên kênh màu (không đệm kênh thứ 3)
        full_pad = pad_spec_2d + ((0, 0),)
        return np.pad(image, full_pad, mode=mode)
    else:
        raise ValueError(f"Chỉ hỗ trợ ma trận 2D hoặc 3D, nhận được ndim={image.ndim}")


def convolve2d(
    image: np.ndarray,
    kernel: np.ndarray,
    mode: Any = "reflect"
) -> np.ndarray:
    """
    Toán tử tích chập 2D ma trận thuần (Pure NumPy 2D Convolution).
    
    Phương pháp Vector hóa tối ưu:
        - Chuẩn tích chập toán học: Lật ma trận kernel 180 độ (flipud + fliplr).
        - Duyệt vòng lặp duy nhất theo kích thước kernel (kh x kw bước, thường chỉ 3x3=9 hoặc 5x5=25).
        - Tuyệt đối KHÔNG dùng vòng lặp duyệt từng pixel (H x W).
        - Sử dụng lát cắt mảng (array slicing) song song của NumPy:
            output += kernel[i, j] * padded[i:i+H, j:j+W]
        - Tiết kiệm RAM: Độ phức tạp không gian O(H * W), không gây tràn bộ nhớ như sliding_window_view/im2col.
        
    Tham số:
        image: Ma trận ảnh 2D (H, W) hoặc 3D (H, W, C), kiểu float hoặc uint8.
        kernel: Ma trận hạt nhân 2D (kh, kw), kích thước lẻ (3x3, 5x5,...).
        mode: Kiểu đệm biên ('reflect', 'edge', 'constant').
        
    Trả về:
        np.ndarray: Ma trận kết quả tích chập cùng kích thước không gian với ảnh gốc.
    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)
    if not isinstance(kernel, np.ndarray):
        kernel = np.asarray(kernel)

    if kernel.ndim != 2:
        raise ValueError(f"Kernel phải là ma trận 2D, nhận được kernel ndim={kernel.ndim}")

    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError(f"Kích thước kernel phải là số lẻ, nhận được ({kh}, {kw})")

    # Đảm bảo kiểu tính toán float32
    img_float = image.astype(np.float32, copy=False)
    kernel_float = kernel.astype(np.float32, copy=False)

    # Chuẩn toán học: Tích chập yêu cầu lật ma trận kernel 180 độ
    kernel_flipped = np.flip(kernel_float)

    pad_h = kh // 2
    pad_w = kw // 2

    H, W = img_float.shape[:2]

    # Đệm biên ảnh để giữ nguyên kích thước (same convolution)
    padded = padding_2d(img_float, (pad_h, pad_w), mode=mode)

    # Khởi tạo ma trận kết quả với đúng số chiều (2D hoặc 3D)
    output = np.zeros_like(img_float, dtype=np.float32)

    # Vector hóa lát cắt (Vectorized slice accumulation)
    # Lặp qua các phần tử của kernel, thực hiện phép nhân ma trận 2D/3D đồng thời
    for i in range(kh):
        for j in range(kw):
            weight = kernel_flipped[i, j]
            if weight != 0.0:
                if img_float.ndim == 2:
                    output += weight * padded[i : i + H, j : j + W]
                else:
                    output += weight * padded[i : i + H, j : j + W, :]

    return output


def convolve1d_axis(
    image: np.ndarray,
    kernel_1d: np.ndarray,
    axis: int,
    mode: Any = "reflect"
) -> np.ndarray:
    """
    Tích chập 1D dọc theo một trục xác định (axis=0 cho hàng, axis=1 cho cột).
    Tối ưu hóa cho các bộ lọc khả tách (Separable Filters) như Gaussian Blur.
    """
    if kernel_1d.ndim != 1:
        raise ValueError("kernel_1d phải là mảng 1 chiều.")

    k_len = len(kernel_1d)
    if k_len % 2 == 0:
        raise ValueError("Độ dài kernel phải là số lẻ.")

    pad = k_len // 2
    k_flipped = kernel_1d[::-1].astype(np.float32)
    img_float = image.astype(np.float32, copy=False)
    H, W = img_float.shape[:2]

    if axis == 0:
        # Tích chập theo chiều dọc (chiều cao H)
        padded = padding_2d(img_float, ((pad, pad), (0, 0)), mode=mode)
        output = np.zeros_like(img_float, dtype=np.float32)
        for i in range(k_len):
            w = k_flipped[i]
            if w != 0.0:
                if img_float.ndim == 2:
                    output += w * padded[i : i + H, :]
                else:
                    output += w * padded[i : i + H, :, :]
        return output

    elif axis == 1:
        # Tích chập theo chiều ngang (chiều rộng W)
        padded = padding_2d(img_float, ((0, 0), (pad, pad)), mode=mode)
        output = np.zeros_like(img_float, dtype=np.float32)
        for j in range(k_len):
            w = k_flipped[j]
            if w != 0.0:
                if img_float.ndim == 2:
                    output += w * padded[:, j : j + W]
                else:
                    output += w * padded[:, j : j + W, :]
        return output

    else:
        raise ValueError(f"Chỉ hỗ trợ axis=0 hoặc axis=1, nhận được axis={axis}")
