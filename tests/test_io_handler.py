"""Unit test cho module io_handler.py"""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from src.io_handler import load_dicom, load_image, save_image, to_float32, to_uint8


def test_to_float32_uint8():
    """Kiểm tra chuyển đổi từ uint8 [0, 255] sang float32 [0.0, 1.0]."""
    arr = np.array([[0, 127, 255], [51, 102, 204]], dtype=np.uint8)
    res = to_float32(arr)
    assert res.dtype == np.float32
    assert res.shape == arr.shape
    assert res.min() == 0.0
    assert res.max() == 1.0
    assert np.isclose(res[0, 1], 127.0 / 255.0)


def test_to_float32_dicom_16bit():
    """Kiểm tra chuyển đổi từ số nguyên 16-bit của DICOM sang float32 [0.0, 1.0]."""
    arr = np.array([[0, 2000], [1000, 4000]], dtype=np.uint16)
    res = to_float32(arr)
    assert res.dtype == np.float32
    assert res.min() == 0.0
    assert res.max() == 1.0
    assert np.isclose(res[1, 1], 1.0)
    assert np.isclose(res[1, 0], 0.25)


def test_to_uint8_clipping():
    """Kiểm tra chuyển đổi float32 sang uint8 và kẹp ngoài biên an toàn."""
    arr = np.array([[-0.5, 0.0, 0.5], [1.0, 1.5, 0.2]], dtype=np.float32)
    res = to_uint8(arr)
    assert res.dtype == np.uint8
    assert res[0, 0] == 0       # Kẹp giá trị âm
    assert res[0, 1] == 0
    assert res[0, 2] == 128     # 0.5 * 255 = 127.5 -> 128
    assert res[1, 0] == 255
    assert res[1, 1] == 255     # Kẹp giá trị > 1.0


def test_save_and_load_png_jpg_bmp():
    """Kiểm tra nạp và lưu ảnh đa định dạng: .png, .jpg, .bmp."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Tạo ảnh mẫu RGB (50x60x3)
        original_rgb = np.random.randint(0, 256, size=(50, 60, 3), dtype=np.uint8)

        for ext in [".png", ".bmp"]:
            img_file = tmp_path / f"test_img{ext}"
            save_image(img_file, original_rgb)
            assert img_file.exists()

            # Nạp lại dưới dạng float32
            loaded_float = load_image(img_file, as_float=True)
            assert loaded_float.dtype == np.float32
            assert loaded_float.shape == (50, 60, 3)
            assert loaded_float.min() >= 0.0
            assert loaded_float.max() <= 1.0

            # Nạp lại dưới dạng uint8
            loaded_uint8 = load_image(img_file, as_float=False)
            assert loaded_uint8.dtype == np.uint8
            # Do định dạng nén không mất dữ liệu (lossless), mảng phải trùng khớp
            np.testing.assert_array_equal(original_rgb, loaded_uint8)

        # Kiểm tra file .jpg
        jpg_file = tmp_path / "test_img.jpg"
        save_image(jpg_file, original_rgb)
        loaded_jpg = load_image(jpg_file, as_float=False)
        assert loaded_jpg.shape == (50, 60, 3)


def test_load_dicom_file():
    """Kiểm tra nạp file DICOM (.dcm) và trích xuất ma trận pixel 2D."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dcm_file = Path(tmpdir) / "sample.dcm"

        # Tạo file DICOM chuẩn y tế
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(str(dcm_file), {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.PatientName = "Hospital^Patient"
        ds.PatientID = "MED-9988"
        ds.Modality = "CT"
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.SeriesInstanceUID = generate_uid()
        ds.StudyInstanceUID = generate_uid()
        ds.FrameOfReferenceUID = generate_uid()
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.SamplesPerPixel = 1
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.Rows = 32
        ds.Columns = 32
        ds.PixelSpacing = [1.0, 1.0]
        ds.RescaleIntercept = "-1024"
        ds.RescaleSlope = "1"

        raw_pixels = np.arange(32 * 32, dtype=np.uint16).reshape((32, 32))
        ds.PixelData = raw_pixels.tobytes()
        ds.save_as(str(dcm_file))

        # Nạp dữ liệu bằng io_handler
        dcm_matrix = load_dicom(dcm_file, as_float=True)
        assert dcm_matrix.shape == (32, 32)
        assert dcm_matrix.dtype == np.float32
        assert dcm_matrix.min() >= 0.0
        assert dcm_matrix.max() <= 1.0


def test_load_image_exif_orientation():
    """Kiểm tra load_image tự động chuẩn hóa góc xoay theo EXIF Orientation của điện thoại."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Tạo ảnh giả định hướng ngang (width=100, height=50)
        img = Image.new("RGB", (100, 50), color="blue")
        exif = img.getexif()
        # Tag EXIF 0x0112 = 6 (Rotate 90 CW - điện thoại chụp dọc)
        exif[0x0112] = 6
        file_path = Path(tmpdir) / "test_phone_photo.jpg"
        img.save(file_path, exif=exif)

        # Khi nạp qua load_image, ảnh phải được tự động xoay đứng thành (height=100, width=50)
        loaded = load_image(file_path, as_float=False)
        assert loaded.shape == (100, 50, 3)
