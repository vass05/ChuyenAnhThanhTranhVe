"""Unit test cho module sketch.py (Pencil Sketch Pipeline)"""

import numpy as np

from src.sketch import color_dodge, pencil_sketch


def test_color_dodge_values():
    """Kiểm tra tính toán công thức Color Dodge Blending."""
    # Khi ảnh xám là màu đen (0.0), kết quả phải là đen (0.0)
    zero_gray = np.zeros((5, 5), dtype=np.float32)
    blur = np.full((5, 5), 128.0, dtype=np.float32)
    res_zero = color_dodge(zero_gray, blur)
    np.testing.assert_allclose(res_zero, 0.0)

    # Khi ảnh xám là màu trắng (255.0), kết quả bị kẹp ở 255.0
    white_gray = np.full((5, 5), 255.0, dtype=np.float32)
    res_white = color_dodge(white_gray, blur)
    np.testing.assert_allclose(res_white, 255.0)


def test_pencil_sketch_shapes_and_types():
    """Kiểm tra kích thước đầu ra 2D và kiểu dữ liệu chuẩn của tranh phác thảo chì."""
    H, W = 40, 50
    rgb = np.random.rand(H, W, 3).astype(np.float32)

    # Chế độ float32 [0.0, 1.0]
    sketch_f32 = pencil_sketch(rgb, blur_size=7, blur_sigma=3.0, as_float=True)
    assert sketch_f32.ndim == 2
    assert sketch_f32.shape == (H, W)
    assert sketch_f32.dtype == np.float32
    assert sketch_f32.min() >= 0.0
    assert sketch_f32.max() <= 1.0

    # Chế độ uint8 [0, 255]
    sketch_u8 = pencil_sketch(rgb, blur_size=7, blur_sigma=3.0, as_float=False)
    assert sketch_u8.ndim == 2
    assert sketch_u8.shape == (H, W)
    assert sketch_u8.dtype == np.uint8
    assert 0 <= sketch_u8.min() and sketch_u8.max() <= 255


def test_color_pencil_sketch():
    from src.sketch import color_pencil_sketch
    H, W = 30, 30
    rgb = np.random.rand(H, W, 3).astype(np.float32)
    res_float = color_pencil_sketch(rgb, blur_size=7, blur_sigma=3.0, as_float=True)
    assert res_float.shape == (H, W, 3)
    assert res_float.dtype == np.float32
    assert res_float.min() >= 0.0 and res_float.max() <= 1.0

    res_uint8 = color_pencil_sketch(rgb, blur_size=7, blur_sigma=3.0, as_float=False)
    assert res_uint8.shape == (H, W, 3)
    assert res_uint8.dtype == np.uint8
