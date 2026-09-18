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


def test_continuous_line_art_basic_and_shapes():
    """Kiểm tra pipeline continuous_line_art: bảo toàn kích thước, kiểu dữ liệu và dải giá trị."""
    from src.edges import continuous_line_art

    # Kiểm tra với ảnh RGB 3 kênh
    img_rgb = np.ones((50, 60, 3), dtype=np.float32)
    # Vẽ một hình vuông đen ở giữa
    img_rgb[15:35, 20:40] = 0.1

    line_art_float = continuous_line_art(img_rgb, threshold=0.10, boldness=1.0, as_float=True)
    assert line_art_float.shape == (50, 60)
    assert line_art_float.dtype == np.float32
    assert line_art_float.min() >= 0.0
    assert line_art_float.max() <= 1.0

    # Nền trắng (xấp xỉ 1.0), nét viền đen (nhỏ hơn 0.5)
    assert line_art_float[0, 0] > 0.95
    # Tại chu vi hình vuông có nét viền mực đậm
    assert np.min(line_art_float[14:36, 19:41]) < 0.3

    # Kiểm tra với đầu ra uint8
    line_art_uint8 = continuous_line_art(img_rgb, as_float=False)
    assert line_art_uint8.shape == (50, 60)
    assert line_art_uint8.dtype == np.uint8
    assert line_art_uint8.max() <= 255


def test_continuous_line_art_color_edges():
    """Kiểm tra khả năng bóc tách biên độ màu sắc (Color Gradient) trên kênh RGB."""
    from src.edges import continuous_line_art

    # Tạo ảnh có 2 mảng màu có cùng độ sáng nhưng khác biệt kênh màu (Đỏ vs Xanh)
    img_color = np.zeros((40, 40, 3), dtype=np.float32)
    img_color[:, :20] = [0.8, 0.1, 0.1]
    img_color[:, 20:] = [0.1, 0.8, 0.1]

    art = continuous_line_art(img_color, threshold=0.08, boldness=1.0, invert=True)
    # Đường ranh giới giữa cột 19 và 20 phải có nét vẽ mực đen đậm
    boundary_pixels = art[10:30, 18:22]
    assert np.min(boundary_pixels) < 0.4


def test_continuous_line_art_thickness():
    """Kiểm tra tham số độ dày nét vẽ thickness làm tăng số lượng pixel nét vẽ."""
    from src.edges import continuous_line_art

    img = np.ones((50, 50), dtype=np.float32)
    img[20:30, 20:30] = 0.0

    art_thin = continuous_line_art(img, threshold=0.10, thickness=1, invert=True)
    art_thick = continuous_line_art(img, threshold=0.10, thickness=2, invert=True)

    dark_thin = np.sum(art_thin < 0.5)
    dark_thick = np.sum(art_thick < 0.5)

    assert dark_thick > dark_thin

