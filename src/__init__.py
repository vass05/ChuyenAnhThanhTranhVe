"""
Thư viện Thuật toán Xử lý Ảnh Cốt lõi (Core Image Processing Engine)
Sprint 1: I/O, Grayscale, 2D Convolution, Gaussian Filter
Sprint 2: Sobel Edge Detection, Bilateral Filter, Pencil Sketch, Cartoonify
Thuần NumPy - Tuyệt đối không dùng OpenCV.
"""

from .cartoon import cartoonify, quantize_colors
from .convolution import convolve1d_axis, convolve2d, padding_2d
from .edges import compute_gradients, create_edge_mask, get_sobel_kernels
from .filters import (
    bilateral_filter,
    gaussian_blur,
    get_gaussian_kernel_1d,
    get_gaussian_kernel_2d,
)
from .grayscale import to_grayscale
from .io_handler import (
    load_dicom,
    load_image,
    save_image,
    to_float32,
    to_uint8,
)
from .sketch import color_dodge, pencil_sketch

__all__ = [
    "bilateral_filter",
    "cartoonify",
    "color_dodge",
    "compute_gradients",
    "convolve1d_axis",
    "convolve2d",
    "create_edge_mask",
    "gaussian_blur",
    "get_gaussian_kernel_1d",
    "get_gaussian_kernel_2d",
    "get_sobel_kernels",
    "load_dicom",
    "load_image",
    "padding_2d",
    "pencil_sketch",
    "quantize_colors",
    "save_image",
    "to_float32",
    "to_grayscale",
    "to_uint8",
]
