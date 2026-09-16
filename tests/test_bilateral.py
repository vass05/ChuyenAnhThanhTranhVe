"""Unit test cho module filters.py (Bilateral Filter)"""

import numpy as np

from src.filters import bilateral_filter, gaussian_blur


def test_bilateral_flat_image():
    """Kiểm tra ảnh phẳng giá trị không đổi qua bộ lọc Bilateral."""
    flat = np.full((15, 15), 0.6, dtype=np.float32)
    filtered = bilateral_filter(flat, d=5, sigma_s=5.0, sigma_r=0.1)
    np.testing.assert_allclose(filtered, 0.6, atol=1e-5)


def test_bilateral_edge_preservation():
    """Kiểm tra tính năng bảo toàn biên: Bilateral giữ nguyên độ dốc sắc nét hơn Gaussian Blur."""
    # Tạo ảnh bậc thang: nửa trái 0.0, nửa phải 1.0
    step = np.zeros((20, 20), dtype=np.float32)
    step[:, 10:] = 1.0

    # Gaussian blur làm nhòe cạnh biên
    gauss = gaussian_blur(step, size=7, sigma=2.0)

    # Bilateral với sigma_r nhỏ (0.05) ngăn không cho trộn các pixel có độ chênh lệch màu lớn
    bilat = bilateral_filter(step, d=7, sigma_s=2.0, sigma_r=0.05)

    # Tại cột 9 (ngay sát vách), Gaussian bị trộn màu nên giá trị tăng cao
    # Trong khi Bilateral bảo toàn vách biên nên giá trị tại cột 9 vẫn gần sát 0.0
    assert bilat[10, 9] < 0.05
    assert gauss[10, 9] > 0.20
    assert bilat[10, 9] < gauss[10, 9]


def test_bilateral_3d_color():
    """Kiểm tra xử lý ảnh màu 3D RGB."""
    img_rgb = np.random.rand(25, 25, 3).astype(np.float32)
    filtered_rgb = bilateral_filter(img_rgb, d=5, sigma_s=3.0, sigma_r=0.1)
    assert filtered_rgb.shape == (25, 25, 3)
    assert filtered_rgb.dtype == np.float32
    assert filtered_rgb.min() >= 0.0
    assert filtered_rgb.max() <= 1.0


def test_bilateral_uint8():
    """Kiểm tra xử lý ảnh kiểu uint8 [0, 255]."""
    img_u8 = np.random.randint(0, 256, size=(20, 20), dtype=np.uint8)
    res_u8 = bilateral_filter(img_u8, d=5, sigma_s=3.0, sigma_r=25.0)
    assert res_u8.dtype == np.uint8
    assert 0 <= res_u8.min() and res_u8.max() <= 255
