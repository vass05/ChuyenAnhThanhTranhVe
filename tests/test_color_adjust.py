"""
Unit tests cho module tinh chỉnh màu sắc src/color_adjust.py
"""

import numpy as np

from src.color_adjust import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    apply_tone_adjustments,
)


def test_adjust_brightness():
    img = np.array([[0.2, 0.5], [0.8, 1.0]], dtype=np.float32)
    brighter = adjust_brightness(img, factor=0.2)
    assert np.allclose(brighter[0, 0], 0.4)
    assert np.allclose(brighter[1, 1], 1.0)  # Clipped at 1.0
    
    darker = adjust_brightness(img, factor=-0.3)
    assert np.allclose(darker[0, 0], 0.0)  # Clipped at 0.0
    assert np.allclose(darker[0, 1], 0.2)


def test_adjust_contrast():
    img = np.array([[0.0, 0.5], [0.8, 1.0]], dtype=np.float32)
    # Neutral contrast factor = 1.0
    unchanged = adjust_contrast(img, factor=1.0)
    assert np.allclose(img, unchanged)
    
    # High contrast factor = 2.0
    # (0.8 - 0.5) * 2.0 + 0.5 = 1.1 -> clipped to 1.0
    high_c = adjust_contrast(img, factor=2.0)
    assert high_c.min() >= 0.0 and high_c.max() <= 1.0
    assert high_c[0, 1] == 0.5  # Midpoint stays midpoint


def test_adjust_saturation():
    # Create an RGB image: red pixel and neutral gray pixel
    img = np.zeros((2, 2, 3), dtype=np.float32)
    img[0, 0] = [1.0, 0.0, 0.0]  # Pure Red
    img[0, 1] = [0.5, 0.5, 0.5]  # Neutral Gray
    
    # 0 saturation -> becomes grayscale
    desat = adjust_saturation(img, factor=0.0)
    # Red pixel luminance Y = 0.299*1 = 0.299
    assert np.allclose(desat[0, 0, 0], 0.299, atol=1e-3)
    assert np.allclose(desat[0, 0, 1], 0.299, atol=1e-3)
    assert np.allclose(desat[0, 0, 2], 0.299, atol=1e-3)
    
    # Gray pixel stays gray
    assert np.allclose(desat[0, 1], [0.5, 0.5, 0.5])
    
    # Grayscale image (2D) should not fail
    gray_img = np.full((3, 3), 0.5, dtype=np.float32)
    res_gray = adjust_saturation(gray_img, factor=1.5)
    assert np.allclose(res_gray, gray_img)


def test_apply_tone_adjustments():
    img = np.full((4, 4, 3), 0.5, dtype=np.float32)
    out = apply_tone_adjustments(img, brightness=0.1, contrast=1.2, saturation=1.5)
    assert out.shape == (4, 4, 3)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
