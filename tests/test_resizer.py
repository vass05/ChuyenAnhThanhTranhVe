"""Unit test cho module resizer.py"""

import numpy as np

from src.resizer import resize_bilinear, resize_max_dimension


def test_resize_bilinear_shape_and_range():
    """Kiểm tra kích thước đầu ra và bảo toàn dải giá trị khi nội suy song tuyến tính."""
    # Test ma trận 2D
    img_2d = np.random.rand(60, 80).astype(np.float32)
    resized_2d = resize_bilinear(img_2d, (30, 40))
    assert resized_2d.shape == (30, 40)
    assert resized_2d.min() >= 0.0 - 1e-5
    assert resized_2d.max() <= 1.0 + 1e-5

    # Test ma trận 3D RGB
    img_3d = np.random.rand(50, 70, 3).astype(np.float32)
    resized_3d = resize_bilinear(img_3d, (100, 140))
    assert resized_3d.shape == (100, 140, 3)


def test_resize_max_dimension():
    """Kiểm tra thu nhỏ ảnh theo cạnh lớn nhất."""
    # Ảnh kích thước lớn 800x600 -> thu nhỏ về max 400px (400x300)
    large_img = np.ones((600, 800, 3), dtype=np.float32)
    res, was_resized = resize_max_dimension(large_img, max_dim=400)
    assert was_resized is True
    assert res.shape == (300, 400, 3)

    # Ảnh nhỏ hơn max_dim -> giữ nguyên
    small_img = np.ones((150, 200, 3), dtype=np.float32)
    res_small, was_small_resized = resize_max_dimension(small_img, max_dim=400)
    assert was_small_resized is False
    assert res_small.shape == (150, 200, 3)
