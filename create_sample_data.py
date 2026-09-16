"""
Script tạo dữ liệu mẫu thực tế cho 3 nhóm ảnh:
1. Ảnh Tự nhiên (Natural): Cảnh hoàng hôn, đồi núi, mặt trời, sông nước (RGB .png / .jpg)
2. Ảnh Công nghiệp (Industrial): Bo mạch điện tử PCB với IC chip, chân cắm, đường mạch (RGB / Grayscale .bmp / .png)
3. Ảnh Y tế (Medical): File DICOM (.dcm) chuẩn cấu trúc CT/X-quang lồng ngực với xương sườn, phổi, cột sống.
"""

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")


def create_natural_sample(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    w, h = 400, 300
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)

    # Bầu trời gradient hoàng hôn
    for y in range(h // 2 + 50):
        t = y / (h // 2 + 50)
        r = int(255 * (1 - 0.3 * t))
        g = int(120 * (1 - 0.4 * t) + 80 * t)
        b = int(60 * (1 - t) + 180 * t)
        draw.line([(0, y), (w, y)], fill=(r, g, b))

    # Mặt trời
    draw.ellipse([w // 2 - 40, 60, w // 2 + 40, 140], fill=(255, 230, 100))

    # Dãy núi xa
    mountain_pts = [(0, 160), (70, 110), (150, 150), (220, 95), (310, 145), (400, 120), (400, 200), (0, 200)]
    draw.polygon(mountain_pts, fill=(50, 70, 90))

    # Đồi gần
    hill_pts = [(0, 190), (90, 160), (200, 185), (290, 155), (400, 180), (400, 300), (0, 300)]
    draw.polygon(hill_pts, fill=(34, 110, 45))

    # Thêm chi tiết cây cối
    for x in range(30, 380, 45):
        draw.line([(x, 210), (x, 240)], fill=(80, 50, 20), width=3)
        draw.polygon([(x - 15, 220), (x, 180), (x + 15, 220)], fill=(20, 80, 30))

    img.save(output_dir / "natural_landscape.png")
    img.save(output_dir / "natural_landscape.jpg", quality=95)
    print(f"[OK] Đã tạo ảnh tự nhiên: {output_dir / 'natural_landscape.png'}")


def create_industrial_sample(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    w, h = 400, 350
    img = Image.new("RGB", (w, h), color=(15, 60, 30))  # Nền xanh lá PCB
    draw = ImageDraw.Draw(img)

    # Các đường mạch đồng mạ vàng (Traces)
    draw.line([(30, 50), (120, 50), (150, 80), (150, 150)], fill=(210, 175, 45), width=3)
    draw.line([(30, 70), (100, 70), (130, 100), (130, 150)], fill=(210, 175, 45), width=3)
    draw.line([(250, 150), (250, 80), (280, 50), (370, 50)], fill=(210, 175, 45), width=3)
    draw.line([(150, 220), (150, 280), (100, 310), (30, 310)], fill=(210, 175, 45), width=3)
    draw.line([(250, 220), (250, 280), (300, 310), (370, 310)], fill=(210, 175, 45), width=3)

    # Chip IC vi điều khiển ở trung tâm
    draw.rectangle([160, 140, 240, 220], fill=(25, 25, 25), outline=(60, 60, 60), width=2)
    # Các chân chip (Pins)
    for p in range(165, 235, 10):
        draw.rectangle([150, p, 160, p + 5], fill=(180, 180, 180))
        draw.rectangle([240, p, 250, p + 5], fill=(180, 180, 180))
        draw.rectangle([p, 130, p + 5, 140], fill=(180, 180, 180))
        draw.rectangle([p, 220, p + 5, 230], fill=(180, 180, 180))

    # Tụ điện & điện trở SMD
    for y in [80, 100, 120, 250, 270]:
        draw.rectangle([40, y, 70, y + 12], fill=(40, 40, 90), outline=(200, 200, 200), width=2)
        draw.rectangle([330, y, 360, y + 12], fill=(120, 70, 40), outline=(200, 200, 200), width=2)

    # Lỗ xuyên via (vias)
    for coord in [(120, 50), (100, 70), (280, 50), (100, 310), (300, 310), (100, 180), (300, 180)]:
        cx, cy = coord
        draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(210, 175, 45), outline=(10, 40, 20))
        draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(10, 40, 20))

    img.save(output_dir / "industrial_pcb.png")
    img.save(output_dir / "industrial_pcb.bmp")
    print(f"[OK] Đã tạo ảnh công nghiệp: {output_dir / 'industrial_pcb.png'}")


def create_medical_sample(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    dcm_path = output_dir / "chest_ct_scan.dcm"

    H, W = 256, 256
    # Mô phỏng lát cắt CT lồng ngực với đơn vị Hounsfield (-1000 HU đến +1000 HU)
    # Không khí = -1000, Mô mềm = ~40, Phổi = -600, Xương sườn/Cột sống = +800
    ct_data = np.full((H, W), -1000.0, dtype=np.float32)

    yy, xx = np.mgrid[:H, :W]
    center_y, center_x = H // 2, W // 2

    # Cơ thể người (hình elip mô mềm)
    body_mask = ((xx - center_x) ** 2) / (95 ** 2) + ((yy - center_y) ** 2) / (80 ** 2) <= 1.0
    ct_data[body_mask] = 40.0

    # Lớp mỡ dưới da
    fat_rim = ((xx - center_x) ** 2) / (92 ** 2) + ((yy - center_y) ** 2) / (77 ** 2) > 1.0
    ct_data[body_mask & fat_rim] = -80.0

    # Phổi trái & phải (-600 HU)
    lung_left = ((xx - (center_x - 45)) ** 2) / (32 ** 2) + ((yy - center_y) ** 2) / (50 ** 2) <= 1.0
    lung_right = ((xx - (center_x + 45)) ** 2) / (32 ** 2) + ((yy - center_y) ** 2) / (50 ** 2) <= 1.0
    ct_data[lung_left | lung_right] = -650.0

    # Mạch máu trong phổi
    vessels = (np.sin(xx * 0.2) * np.cos(yy * 0.2) > 0.6) & (lung_left | lung_right)
    ct_data[vessels] = 80.0

    # Cột sống (Spine) ở phía sau (+800 HU)
    spine = ((xx - center_x) ** 2) / (12 ** 2) + ((yy - (center_y + 45)) ** 2) / (12 ** 2) <= 1.0
    ct_data[spine] = 850.0

    # Ống tủy sống
    spinal_canal = ((xx - center_x) ** 2) / (4 ** 2) + ((yy - (center_y + 45)) ** 2) / (4 ** 2) <= 1.0
    ct_data[spinal_canal] = 15.0

    # Xương sườn (Ribs xung quanh lồng ngực)
    for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        rx = int(center_x + 82 * np.cos(angle))
        ry = int(center_y + 68 * np.sin(angle))
        rib = ((xx - rx) ** 2) + ((yy - ry) ** 2) <= 18
        ct_data[rib] = 750.0

    # Chuyển đổi sang uint16 với offset 1024 (chuẩn DICOM: pixel = HU + 1024)
    raw_pixels = np.clip(ct_data + 1024, 0, 4095).astype(np.uint16)

    # Đóng gói FileDataset chuẩn DICOM
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(dcm_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.PatientName = "Nguyen^Van^A"
    ds.PatientID = "MED-CHEST-2026"
    ds.PatientBirthDate = "19800101"
    ds.PatientSex = "M"
    ds.Modality = "CT"
    ds.StudyDescription = "CT Chest Routine"
    ds.SeriesDescription = "Lung 2.0mm"
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.FrameOfReferenceUID = generate_uid()
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SamplesPerPixel = 1
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.Rows = H
    ds.Columns = W
    ds.PixelSpacing = [0.75, 0.75]
    ds.RescaleIntercept = "-1024"
    ds.RescaleSlope = "1"
    ds.PixelData = raw_pixels.tobytes()

    ds.save_as(str(dcm_path))
    print(f"[OK] Đã tạo file y tế DICOM: {dcm_path}")


if __name__ == "__main__":
    base_data = Path("data")
    create_natural_sample(base_data / "natural")
    create_industrial_sample(base_data / "industrial")
    create_medical_sample(base_data / "medical")
