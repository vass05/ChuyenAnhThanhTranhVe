"""Unit test cho module grayscale.py"""

import numpy as np

from src.grayscale import to_grayscale


def test_grayscale_primary_colors():
    """Kiểm tra hệ số chuyển đổi chuẩn ITU-R BT.601 với các màu cơ bản."""
    # Ảnh 1x1 các màu chuẩn
    red = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    green = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)
    blue = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
    white = np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32)
    black = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)

    assert np.isclose(to_grayscale(red)[0, 0], 0.299, atol=1e-5)
    assert np.isclose(to_grayscale(green)[0, 0], 0.587, atol=1e-5)
    assert np.isclose(to_grayscale(blue)[0, 0], 0.114, atol=1e-5)
    assert np.isclose(to_grayscale(white)[0, 0], 1.0, atol=1e-5)
    assert np.isclose(to_grayscale(black)[0, 0], 0.0, atol=1e-5)


def test_grayscale_output_shape_2d():
    """Đảm bảo đầu ra luôn là ma trận 2D chuẩn xác kích thước (H, W)."""
    H, W = 120, 160

    # Test với RGB (H, W, 3)
    img_rgb = np.random.rand(H, W, 3).astype(np.float32)
    gray_rgb = to_grayscale(img_rgb)
    assert gray_rgb.ndim == 2
    assert gray_rgb.shape == (H, W)

    # Test với RGBA (H, W, 4)
    img_rgba = np.random.rand(H, W, 4).astype(np.float32)
    gray_rgba = to_grayscale(img_rgba)
    assert gray_rgba.ndim == 2
    assert gray_rgba.shape == (H, W)

    # Test với ảnh 1 kênh (H, W, 1)
    img_1ch = np.random.rand(H, W, 1).astype(np.float32)
    gray_1ch = to_grayscale(img_1ch)
    assert gray_1ch.ndim == 2
    assert gray_1ch.shape == (H, W)

    # Test với ảnh đã là 2D (H, W)
    img_2d = np.random.rand(H, W).astype(np.float32)
    gray_2d = to_grayscale(img_2d)
    assert gray_2d.ndim == 2
    assert gray_2d.shape == (H, W)


def test_grayscale_uint8_preservation():
    """Kiểm tra xử lý ảnh uint8 giữ đúng kiểu và dải [0, 255]."""
    img_uint8 = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    res = to_grayscale(img_uint8)
    assert res.dtype == np.uint8
    assert res.shape == (2, 2)
    assert res[0, 0] == int(np.round(0.299 * 255))
    assert res[0, 1] == int(np.round(0.587 * 255))
    assert res[1, 0] == int(np.round(0.114 * 255))
    assert res[1, 1] == 255
