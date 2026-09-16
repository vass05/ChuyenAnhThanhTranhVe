"""
Script nghiệm thu Sprint 1: Xây dựng Động cơ Thuật toán Cốt lõi
Thực thi trên 3 nhóm dữ liệu ảnh: Tự nhiên, Công nghiệp, Y tế (DICOM).
Xác thực Definition of Done (DoD) và lưu kết quả trực quan vào output_sprint1/.
"""

import io
import sys
import time
from pathlib import Path

import numpy as np

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

from src.filters import gaussian_blur, get_gaussian_kernel_2d
from src.grayscale import to_grayscale
from src.io_handler import load_dicom, load_image, save_image, to_uint8


def run_pipeline_for_image(
    name: str,
    file_path: Path,
    is_dicom: bool = False,
    output_dir: Path = Path("output_sprint1")
):
    print("\n=======================================================")
    print(f"[*] BẮT ĐẦU XỬ LÝ: {name.upper()} ({file_path})")
    print("=======================================================")

    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # 1. Nạp và chuẩn hóa dữ liệu
    if is_dicom:
        img_raw = load_dicom(file_path, as_float=True)
    else:
        img_raw = load_image(file_path, as_float=True)

    t_load = time.perf_counter() - t0
    print(f" -> 1. Nạp ảnh: Shape = {img_raw.shape}, Dtype = {img_raw.dtype}, Min = {img_raw.min():.4f}, Max = {img_raw.max():.4f} (Thời gian: {t_load*1000:.2f}ms)")
    assert img_raw.dtype == np.float32, "Lỗi: Ảnh nạp chưa chuẩn hóa về float32"
    assert 0.0 <= img_raw.min() and img_raw.max() <= 1.0 + 1e-6, "Lỗi: Dải giá trị vượt ngoài [0.0, 1.0]"

    # Lưu ảnh gốc đã chuẩn hóa
    orig_save_path = output_dir / f"{name}_01_original.png"
    save_image(orig_save_path, img_raw)

    # 2. Chuyển đổi mức xám (Grayscale Engine)
    t1 = time.perf_counter()
    gray_img = to_grayscale(img_raw)
    t_gray = time.perf_counter() - t1
    print(f" -> 2. Mức xám (Grayscale): Shape = {gray_img.shape}, Dtype = {gray_img.dtype}, Min = {gray_img.min():.4f}, Max = {gray_img.max():.4f} (Thời gian: {t_gray*1000:.2f}ms)")
    assert gray_img.ndim == 2, f"Lỗi: Ảnh xám phải là ma trận 2D (H, W), nhận được ndim={gray_img.ndim}"
    assert gray_img.shape == img_raw.shape[:2], "Lỗi: Kích thước không gian bị thay đổi sau grayscale"

    gray_save_path = output_dir / f"{name}_02_grayscale.png"
    save_image(gray_save_path, gray_img)

    # 3. Làm mịn Gauss (Gaussian Blurring Engine)
    t2 = time.perf_counter()
    # Thử nghiệm với kernel 5x5 sigma=1.0 và 9x9 sigma=2.0
    blur_mild = gaussian_blur(gray_img, size=5, sigma=1.0)
    blur_strong = gaussian_blur(gray_img, size=9, sigma=2.0)
    t_blur = time.perf_counter() - t2
    print(f" -> 3. Làm mịn Gauss (Blur): Shape = {blur_mild.shape}, Min = {blur_mild.min():.4f}, Max = {blur_mild.max():.4f} (Thời gian: {t_blur*1000:.2f}ms)")
    assert blur_mild.shape == gray_img.shape, "Lỗi: Kích thước thay đổi sau Gaussian blur"
    assert blur_mild.dtype == np.float32

    # Làm mịn trực tiếp trên ảnh màu gốc (nếu là ảnh 3D RGB)
    if img_raw.ndim == 3:
        blur_color = gaussian_blur(img_raw, size=5, sigma=1.0)
        save_image(output_dir / f"{name}_03_blur_color.png", blur_color)

    save_image(output_dir / f"{name}_03_blur_sigma1.png", blur_mild)
    save_image(output_dir / f"{name}_04_blur_sigma2.png", blur_strong)

    # 4. Kiểm tra ma trận chuyển đổi uint8
    uint8_res = to_uint8(blur_mild)
    assert uint8_res.dtype == np.uint8
    assert 0 <= uint8_res.min() and uint8_res.max() <= 255

    print(f" [V] Hoàn thành pipeline cho {name} thành công. Kết quả đã lưu tại {output_dir}/")


def main():
    print("================================================================================")
    print("DEMO SPRINT 1: XÁC THỰC BỘ NHÂN THUẬT TOÁN XỬ LÝ MA TRẬN ẢNH NỘI BỘ (THUẦN NUMPY)")
    print("================================================================================")

    # Kiểm tra kernel Gauss
    print("\n[*] KIỂM TRA ĐẶC TÍNH KERNEL GAUSS 5x5 (sigma=1.0):")
    k = get_gaussian_kernel_2d(size=5, sigma=1.0)
    print(f"Ma trận Kernel Gauss 5x5:\n{np.round(k, 4)}")
    print(f"Tổng trọng số kernel: {np.sum(k):.6f} (Bảo toàn năng lượng ánh sáng)")

    output_base = Path("output_sprint1")

    # 1. Ảnh Tự nhiên (Natural)
    run_pipeline_for_image(
        name="natural",
        file_path=Path("data/natural/natural_landscape.png"),
        is_dicom=False,
        output_dir=output_base
    )

    # 2. Ảnh Công nghiệp (Industrial)
    run_pipeline_for_image(
        name="industrial",
        file_path=Path("data/industrial/industrial_pcb.png"),
        is_dicom=False,
        output_dir=output_base
    )

    # 3. Ảnh Y tế (Medical DICOM)
    run_pipeline_for_image(
        name="medical",
        file_path=Path("data/medical/chest_ct_scan.dcm"),
        is_dicom=True,
        output_dir=output_base
    )

    print("\n================================================================================")
    print(">>> TẤT CẢ 3 ĐỊNH DẠNG ẢNH ĐỀU ĐÃ ĐƯỢC XỬ LÝ VÀ NGHIỆM THU ĐẠT CHUẨN DoD! <<<")
    print(f"Toàn bộ ảnh đã được lưu trong thư mục: {output_base.resolve()}")
    print("================================================================================")


if __name__ == "__main__":
    main()
