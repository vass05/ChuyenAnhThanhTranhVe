"""Unit test cho module convolution.py"""

import numpy as np

from src.convolution import convolve1d_axis, convolve2d, padding_2d


def test_padding_2d_shapes_and_values():
    """Kiểm tra kích thước và giá trị đệm biên 2D và 3D."""
    # Test ma trận 2D
    img_2d = np.array([[1, 2], [3, 4]], dtype=np.float32)
    padded = padding_2d(img_2d, pad_width=1, mode="constant")
    assert padded.shape == (4, 4)
    assert padded[0, 0] == 0.0  # constant=0
    assert padded[1, 1] == 1.0

    # Test đệm reflect
    padded_reflect = padding_2d(img_2d, pad_width=1, mode="reflect")
    assert padded_reflect.shape == (4, 4)
    assert padded_reflect[0, 1] == 3.0  # Phản xạ qua biên trên

    # Test ma trận 3D (H, W, C): chỉ đệm 2 chiều không gian, không đệm kênh
    img_3d = np.ones((10, 20, 3), dtype=np.float32)
    padded_3d = padding_2d(img_3d, pad_width=(2, 3), mode="reflect")
    assert padded_3d.shape == (14, 26, 3)


def test_convolve2d_identity():
    """Kiểm tra tích chập với Identity Kernel (Delta function). Kết quả phải giữ nguyên ảnh gốc."""
    H, W = 25, 30
    image = np.random.randn(H, W).astype(np.float32)

    # Identity kernel 3x3
    identity_kernel = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0]
    ], dtype=np.float32)

    output = convolve2d(image, identity_kernel, mode="reflect")
    assert output.shape == (H, W)
    np.testing.assert_allclose(output, image, rtol=1e-5, atol=1e-5)


def test_convolve2d_box_blur():
    """Kiểm tra tích chập với bộ lọc trung bình Box Filter (3x3)."""
    # Tạo ảnh phẳng giá trị 5.0
    image = np.full((15, 15), 5.0, dtype=np.float32)
    box_kernel = np.ones((3, 3), dtype=np.float32) / 9.0

    output = convolve2d(image, box_kernel, mode="reflect")
    assert output.shape == (15, 15)
    np.testing.assert_allclose(output, 5.0, rtol=1e-5, atol=1e-5)


def test_convolve2d_3d_rgb():
    """Kiểm tra tích chập áp dụng trên ảnh màu 3D (H, W, 3)."""
    H, W = 30, 40
    rgb = np.random.rand(H, W, 3).astype(np.float32)
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0

    output = convolve2d(rgb, kernel, mode="reflect")
    assert output.shape == (H, W, 3)
    assert output.dtype == np.float32


def test_convolve1d_axis_separable_equivalence():
    """Kiểm tra tính tương đương toán học giữa 2D convolution và 1D separable convolution."""
    H, W = 35, 45
    image = np.random.randn(H, W).astype(np.float32)

    k1d = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    k2d = np.outer(k1d, k1d)

    out_2d = convolve2d(image, k2d, mode="reflect")

    out_sep = convolve1d_axis(image, k1d, axis=1, mode="reflect")
    out_sep = convolve1d_axis(out_sep, k1d, axis=0, mode="reflect")

    np.testing.assert_allclose(out_2d, out_sep, rtol=1e-5, atol=1e-5)
