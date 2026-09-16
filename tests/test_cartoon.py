"""Unit test cho module cartoon.py (Cartoonify Pipeline)"""

import numpy as np

from src.cartoon import cartoonify, quantize_colors


def test_quantize_colors_levels():
    """Kiểm tra lượng tử hóa màu: số lượng mức màu rời rạc không vượt quá num_levels."""
    ramp = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    quant = quantize_colors(ramp, num_levels=4)

    unique_vals = np.unique(quant)
    assert len(unique_vals) <= 4
    np.testing.assert_allclose(unique_vals, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], atol=1e-5)


def test_cartoonify_pipeline():
    """Kiểm tra pipeline cartoonify bảo toàn số chiều và áp dụng nét viền đen."""
    H, W = 30, 40
    # Tạo ảnh có 2 mảng màu lớn (nửa đỏ, nửa xanh)
    img_rgb = np.zeros((H, W, 3), dtype=np.float32)
    img_rgb[:, :20] = [0.8, 0.2, 0.2]   # Đỏ
    img_rgb[:, 20:] = [0.2, 0.3, 0.9]   # Xanh

    cartoon = cartoonify(
        img_rgb,
        num_levels=6,
        d=5,
        sigma_s=3.0,
        sigma_r=0.1,
        edge_threshold=0.10,
        as_float=True
    )

    assert cartoon.shape == (H, W, 3)
    assert cartoon.dtype == np.float32
    assert cartoon.min() >= 0.0
    assert cartoon.max() <= 1.0

    # Tại đường biên giữa (cột 19-20), nét vẽ biên màu đen nên cường độ sáng giảm mạnh
    edge_pixel = cartoon[15, 19]
    assert np.mean(edge_pixel) < 0.1  # Nét viền đen
