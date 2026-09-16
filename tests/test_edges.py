"""Unit test cho module edges.py (Sobel Edge Detection)"""

import numpy as np

from src.edges import compute_gradients, create_edge_mask, get_sobel_kernels


def test_sobel_kernels_properties():
    """Kiểm tra đặc tính của kernel Sobel: kích thước 3x3, tổng bằng 0, tính phản đối xứng."""
    kx, ky = get_sobel_kernels()

    assert kx.shape == (3, 3)
    assert ky.shape == (3, 3)

    # Tổng trọng số Sobel phải bằng 0 (khử thành phần tần số thấp / nền phẳng)
    assert np.isclose(np.sum(kx), 0.0)
    assert np.isclose(np.sum(ky), 0.0)

    # Ky là chuyển vị hoặc lật của Kx
    np.testing.assert_allclose(ky, kx.T)


def test_compute_gradients_step_edges():
    """Kiểm tra phát hiện biên đứng bằng Kx và biên ngang bằng Ky."""
    # Tạo ảnh có biên đứng ở giữa (nửa trái 0.0, nửa phải 1.0)
    vert_edge = np.zeros((20, 20), dtype=np.float32)
    vert_edge[:, 10:] = 1.0

    gx, gy, mag = compute_gradients(vert_edge)

    # Biên đứng phải có |Gx| lớn tại cột 9 và 10, Gy xấp xỉ 0
    assert np.max(np.abs(gx)) > 1.0
    assert np.max(np.abs(gy)) < 1e-4
    assert mag.shape == (20, 20)
    assert mag.min() >= 0.0
    assert mag.max() <= 1.0

    # Tạo ảnh có biên ngang (nửa trên 0.0, nửa dưới 1.0)
    horiz_edge = np.zeros((20, 20), dtype=np.float32)
    horiz_edge[10:, :] = 1.0

    gx_h, gy_h, _mag_h = compute_gradients(horiz_edge)
    assert np.max(np.abs(gy_h)) > 1.0
    assert np.max(np.abs(gx_h)) < 1e-4


def test_create_edge_mask():
    """Kiểm tra phân ngưỡng nhị phân và đảo màu mask."""
    mag = np.array([[0.05, 0.20], [0.80, 0.01]], dtype=np.float32)

    # Nền trắng (1.0), nét đen (0.0)
    mask_inv = create_edge_mask(mag, threshold=0.15, invert=True)
    assert mask_inv[0, 0] == 1.0  # 0.05 < 0.15 -> nền trắng
    assert mask_inv[0, 1] == 0.0  # 0.20 >= 0.15 -> nét đen
    assert mask_inv[1, 0] == 0.0  # 0.80 >= 0.15 -> nét đen
    assert mask_inv[1, 1] == 1.0  # 0.01 < 0.15 -> nền trắng

    # Nền đen (0.0), nét trắng (1.0)
    mask_norm = create_edge_mask(mag, threshold=0.15, invert=False)
    np.testing.assert_allclose(mask_norm, 1.0 - mask_inv)
