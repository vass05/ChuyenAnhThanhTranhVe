"""
Module: io_handler.py
Mô tả: Nạp và chuẩn hóa dữ liệu ảnh đa nguồn (PIL cho .png/.jpg/.bmp, pydicom cho .dcm).
DoD: Thuần NumPy và PIL/pydicom, tuyệt đối không dùng OpenCV.
"""

from pathlib import Path

import numpy as np
from PIL import Image

try:
    import pydicom
except ImportError:
    pydicom = None


def to_float32(image: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa mảng ảnh sang float32 trong dải [0.0, 1.0].
    
    Hỗ trợ:
    - Ảnh uint8 thông thường (0..255): chia cho 255.0.
    - Ảnh y tế DICOM (12/16-bit nguyên, Hounsfield units, v.v.): Min-Max scaling về [0.0, 1.0].
    - Mảng float sẵn có: kẹp giá trị về [0.0, 1.0] và ép kiểu float32.
    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    # Nếu là kiểu số nguyên thông thường uint8 (0 -> 255)
    if image.dtype == np.uint8:
        return (image.astype(np.float32)) / 255.0

    # Nếu là kiểu số nguyên lớn hơn (uint16, int16, int32 từ DICOM)
    if np.issubdtype(image.dtype, np.integer):
        img_float = image.astype(np.float32)
        min_val = float(img_float.min())
        max_val = float(img_float.max())
        if max_val > min_val:
            return (img_float - min_val) / (max_val - min_val)
        return np.zeros_like(img_float, dtype=np.float32)

    # Nếu là kiểu float
    img_float = image.astype(np.float32)
    # Nếu giá trị đang ở thang đo 0..255
    if img_float.max() > 1.0 and img_float.min() >= 0.0 and img_float.max() <= 255.0:
        return np.clip(img_float / 255.0, 0.0, 1.0)
    elif img_float.max() > 1.0 or img_float.min() < 0.0:
        min_val = float(img_float.min())
        max_val = float(img_float.max())
        if max_val > min_val:
            return (img_float - min_val) / (max_val - min_val)
        return np.zeros_like(img_float, dtype=np.float32)

    return np.clip(img_float, 0.0, 1.0)


def to_uint8(image: np.ndarray, is_normalized: bool | None = None) -> np.ndarray:
    """
    Chuyển đổi mảng ảnh về định dạng uint8 chuẩn trong dải [0, 255].
    Kẹp các giá trị ngoài phạm vi để tránh tràn số (underflow/overflow).
    
    Tham số:
        image: Ma trận ảnh (float hoặc int).
        is_normalized:
            - True: Ép coi ảnh ở dải [0.0, 1.0], nhân 255.
            - False: Coi ảnh ở dải [0.0, 255.0], chỉ kẹp [0, 255].
            - None: Tự động nhận diện (nếu max <= 1.5 thì coi là [0.0, 1.0]).
    """
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)

    if image.dtype == np.uint8:
        return image.copy()

    if np.issubdtype(image.dtype, np.floating):
        if is_normalized is True or (is_normalized is None and image.max() <= 1.5):
            clipped = np.clip(image, 0.0, 1.0)
            return (clipped * 255.0 + 0.5).astype(np.uint8)
        else:
            clipped = np.clip(image, 0.0, 255.0)
            return (clipped + 0.5).astype(np.uint8)

    # Các kiểu nguyên khác (int16, uint16, int32)
    clipped = np.clip(image, 0, 255)
    return clipped.astype(np.uint8)


def load_image(file_path: str | Path, as_float: bool = True) -> np.ndarray:
    """
    Nạp ảnh thông thường (.png, .jpg, .jpeg, .bmp, .webp) sử dụng PIL.
    
    Tham số:
        file_path: Đường dẫn tới file ảnh.
        as_float: Nếu True, chuẩn hóa về np.float32 dải [0.0, 1.0].
                  Nếu False, trả về np.uint8 [0, 255].
                  
    Trả về:
        np.ndarray: Ma trận ảnh 2D (grayscale) hoặc 3D (RGB).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {file_path}")

    with Image.open(path) as img:
        # Nếu là ảnh RGBA, chuyển sang RGB nếu không cần alpha
        if img.mode == "RGBA":
            # Tạo background trắng và dán RGBA lên
            rgb_img = Image.new("RGB", img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])
            arr = np.array(rgb_img)
        elif img.mode in ("RGB", "L"):
            arr = np.array(img)
        else:
            # Chuyển các mode khác (P, CMYK, etc.) về RGB
            arr = np.array(img.convert("RGB"))

    if as_float:
        return to_float32(arr)
    return arr


def save_image(file_path: str | Path, image: np.ndarray) -> None:
    """
    Lưu ma trận ảnh ra file (.png, .jpg, .bmp) sử dụng PIL.
    
    Tham số:
        file_path: Đường dẫn đích.
        image: Mảng NumPy (2D grayscale hoặc 3D RGB, kiểu uint8 hoặc float).
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    uint8_img = to_uint8(image)

    # Kiểm tra số chiều
    if uint8_img.ndim == 2:
        pil_img = Image.fromarray(uint8_img, mode="L")
    elif uint8_img.ndim == 3 and uint8_img.shape[2] == 3:
        pil_img = Image.fromarray(uint8_img, mode="RGB")
    elif uint8_img.ndim == 3 and uint8_img.shape[2] == 4:
        pil_img = Image.fromarray(uint8_img, mode="RGBA")
    else:
        raise ValueError(f"Cấu trúc ma trận ảnh không hợp lệ: shape={uint8_img.shape}")

    pil_img.save(path)


def load_dicom(file_path: str | Path, as_float: bool = True) -> np.ndarray:
    """
    Nạp file ảnh y tế DICOM (.dcm) và trích xuất ma trận pixel 2D.
    
    Tham số:
        file_path: Đường dẫn tới file .dcm
        as_float: Nếu True, chuẩn hóa về np.float32 trong [0.0, 1.0].
        
    Trả về:
        np.ndarray: Ma trận 2D của ảnh CT / X-Quang.
    """
    if pydicom is None:
        raise ImportError("Thư viện 'pydicom' chưa được cài đặt. Hãy chạy: pip install pydicom")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file DICOM: {file_path}")

    dcm = pydicom.dcmread(str(path))
    pixel_array = dcm.pixel_array.astype(np.float32)

    # Áp dụng Rescale Slope và Rescale Intercept nếu có trong metadata chuẩn DICOM
    slope = getattr(dcm, "RescaleSlope", 1.0)
    intercept = getattr(dcm, "RescaleIntercept", 0.0)
    if slope != 1.0 or intercept != 0.0:
        pixel_array = pixel_array * float(slope) + float(intercept)

    # Nếu dữ liệu có nhiều slice (3D), lấy slice đầu tiên hoặc nén về 2D
    if pixel_array.ndim == 3:
        pixel_array = pixel_array[0]

    if as_float:
        return to_float32(pixel_array)
    return to_uint8(pixel_array)
