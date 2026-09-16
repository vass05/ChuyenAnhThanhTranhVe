"""
Unit tests cho module tranh màu nước src/watercolor.py
"""

import numpy as np

from src.watercolor import watercolor_effect


def test_watercolor_effect_shape_and_range():
    # Test on random RGB image
    np.random.seed(42)
    img_rgb = np.random.rand(16, 16, 3).astype(np.float32)
    
    out_float = watercolor_effect(img_rgb, bilat_d=3, as_float=True)
    assert out_float.shape == (16, 16, 3)
    assert out_float.dtype == np.float32
    assert out_float.min() >= 0.0 and out_float.max() <= 1.0

    out_uint8 = watercolor_effect(img_rgb, bilat_d=3, as_float=False)
    assert out_uint8.shape == (16, 16, 3)
    assert out_uint8.dtype == np.uint8


def test_watercolor_effect_grayscale():
    img_gray = np.random.rand(16, 16).astype(np.float32)
    out = watercolor_effect(img_gray, bilat_d=3, as_float=True)
    assert out.shape == (16, 16)
    assert out.dtype == np.float32
