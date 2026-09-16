"""
Module: resizer.py
Mô tả: Thu phóng ma trận ảnh nhanh phục vụ xem trước tương tác (Interactive Preview) thuần NumPy.
DoD: Vector hóa hoàn toàn với phép nội suy song tuyến tính (Bilinear Interpolation), không dùng OpenCV.
"""

import numpy as np


def resize_bilinear(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Thu phóng ma trận ảnh 2D hoặc 3D bằng phương pháp nội suy song tuyến tính (Bilinear Interpolation).
    
    Phương pháp Vector hóa ma trận thuần (Pure NumPy Vectorization):
        - Tạo lưới tọa độ tương ứng trên ảnh gốc.
        - Lấy 4 điểm lân cận (top-left, top-right, bottom-left, bottom-right).
        - Tính trọng số khoảng cách và nội suy song song trên toàn bộ tensor ảnh.
        
    Tham số:
        image: Mảng ảnh 2D (H, W) hoặc 3D (H, W, C), float hoặc uint8.
        target_size: Tuple (target_height, target_width).
        
    Trả về:
        np.ndarray: Ảnh sau khi thu phóng với kích thước target_size.
    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    target_h, target_w = target_size
    orig_h, orig_w = image.shape[:2]

    if (orig_h, orig_w) == (target_h, target_w):
        return image.copy()

    is_uint8 = image.dtype == np.uint8
    img_float = image.astype(np.float32, copy=False)

    # Tạo tọa độ liên tục trên ảnh gốc
    y_coords = np.linspace(0.0, orig_h - 1, target_h, dtype=np.float32)
    x_coords = np.linspace(0.0, orig_w - 1, target_w, dtype=np.float32)

    # Chỉ số 4 góc xung quanh
    y0 = np.floor(y_coords).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, orig_h - 1)
    x0 = np.floor(x_coords).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, orig_w - 1)

    # Trọng số nội suy theo phương dọc (wy) và phương ngang (wx)
    wy = (y_coords - y0).reshape(-1, 1)  # (target_h, 1)
    wx = (x_coords - x0).reshape(1, -1)  # (1, target_w)

    if img_float.ndim == 3:
        # Mở rộng chiều cho ảnh màu đa kênh (target_h, target_w, 1)
        wy = wy[..., np.newaxis]
        wx = wx[..., np.newaxis]

    # Trích xuất 4 góc ma trận bằng chỉ mục mảng NumPy
    y0_grid, x0_grid = np.meshgrid(y0, x0, indexing="ij")
    y1_grid, x1_grid = np.meshgrid(y1, x1, indexing="ij")

    # 4 góc lân cận
    top_left = img_float[y0_grid, x0_grid]
    top_right = img_float[y0_grid, x1_grid]
    bottom_left = img_float[y1_grid, x0_grid]
    bottom_right = img_float[y1_grid, x1_grid]

    # Nội suy 2 chiều:
    top = top_left * (1.0 - wx) + top_right * wx
    bottom = bottom_left * (1.0 - wx) + bottom_right * wx

    resized = top * (1.0 - wy) + bottom * wy

    if is_uint8:
        return np.clip(np.round(resized), 0, 255).astype(np.uint8)
    return resized.astype(np.float32)


def resize_max_dimension(image: np.ndarray, max_dim: int = 400) -> tuple[np.ndarray, bool]:
    """
    Thu nhỏ ảnh theo tỷ lệ cạnh lớn nhất để tăng tốc hiển thị tương tác (Fast Preview).
    
    Tham số:
        image: Mảng ảnh gốc.
        max_dim: Kích thước tối đa của chiều dài nhất (mặc định 400px).
        
    Trả về:
        Tuple (resized_image, was_resized):
            - resized_image: Mảng ảnh sau thu nhỏ (hoặc ảnh gốc nếu kích thước đã nhỏ hơn max_dim).
            - was_resized: True nếu ảnh được thu nhỏ, False nếu giữ nguyên.
    """
    h, w = image.shape[:2]
    max_current = max(h, w)

    if max_current <= max_dim:
        return image, False

    scale = max_dim / float(max_current)
    target_h = max(1, round(h * scale))
    target_w = max(1, round(w * scale))

    resized = resize_bilinear(image, (target_h, target_w))
    return resized, True
