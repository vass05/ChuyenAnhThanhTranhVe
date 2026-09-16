"""Unit test cho module filters.py"""

import numpy as np

from src.filters import gaussian_blur, get_gaussian_kernel_2d


def test_gaussian_kernel_symmetry_and_normalization():
    """Kiểm tra tính đối xứng giải tích và chuẩn hóa tổng = 1.0 của kernel Gauss."""
    for size in [3, 5, 7]:
        for sigma in [0.5, 1.0, 2.0]:
            k2d = get_gaussian_kernel_2d(size=size, sigma=sigma)

            assert k2d.shape == (size, size)
            # Tổng các trọng số phải bằng 1.0
            assert np.isclose(np.sum(k2d), 1.0, atol=1e-6)

            # Tính đối xứng qua đường chéo chính (K = K^T)
            np.testing.assert_allclose(k2d, k2d.T, atol=1e-7)

            # Tính đối xứng qua trục ngang và trục dọc
            np.testing.assert_allclose(k2d, np.flipud(k2d), atol=1e-7)
            np.testing.assert_allclose(k2d, np.fliplr(k2d), atol=1e-7)

            # Điểm cực đại phải nằm ở tâm
            center = size // 2
            assert k2d[center, center] == np.max(k2d)


def test_gaussian_blur_noise_reduction():
    """Kiểm tra hiệu ứng làm mịn: giảm độ lệch chuẩn của nhiễu cao tần."""
    # Tạo ảnh phẳng có thêm nhiễu Gauss ngẫu nhiên
    np.random.seed(42)
    clean_img = np.full((100, 100), 0.5, dtype=np.float32)
    noise = np.random.normal(0, 0.1, size=(100, 100)).astype(np.float32)
    noisy_img = clean_img + noise

    # Làm mịn Gauss
    blurred = gaussian_blur(noisy_img, size=5, sigma=1.5)

    # Độ lệch chuẩn của ảnh sau khi làm mịn phải nhỏ hơn đáng kể so với ảnh nhiễu ban đầu
    std_before = float(np.std(noisy_img))
    std_after = float(np.std(blurred))
    assert std_after < std_before * 0.6  # Giảm ít nhất 40% biên độ nhiễu


def test_gaussian_blur_separable_vs_direct():
    """Kiểm tra tính tương đương giữa làm mờ tích chập trực tiếp và tích chập tách biệt 1D."""
    img = np.random.rand(64, 64).astype(np.float32)
    out_direct = gaussian_blur(img, size=5, sigma=1.2, use_separable=False)
    out_separable = gaussian_blur(img, size=5, sigma=1.2, use_separable=True)

    np.testing.assert_allclose(out_direct, out_separable, atol=1e-5)


def test_gaussian_blur_3d_color():
    """Kiểm tra làm mờ trên ảnh màu 3D RGB."""
    img_rgb = np.random.rand(50, 50, 3).astype(np.float32)
    blurred_rgb = gaussian_blur(img_rgb, size=5, sigma=1.0)
    assert blurred_rgb.shape == (50, 50, 3)
    assert blurred_rgb.dtype == np.float32
