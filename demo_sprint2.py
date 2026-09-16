"""
Demo nghiệm thu Sprint 2: Kiểm thử toàn diện trên 3 nhóm ảnh (Tự nhiên, Công nghiệp, Y tế).
Sinh ra các kết quả trung gian và sản phẩm nghệ thuật cuối cùng vào output_sprint2/.
"""

import io
import sys
import time
from pathlib import Path

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

from src.cartoon import cartoonify
from src.edges import compute_gradients, create_edge_mask
from src.filters import bilateral_filter
from src.grayscale import to_grayscale
from src.io_handler import load_dicom, load_image, save_image
from src.sketch import pencil_sketch


def process_image_sprint2(name: str, file_path: Path, is_dicom: bool = False, output_dir: Path = Path("output_sprint2")):
    print("\n================================================================================")
    print(f"[*] TIẾN HÀNH PIPELINE SPRINT 2: {name.upper()} ({file_path})")
    print("================================================================================")

    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    # 1. Nạp ảnh
    if is_dicom:
        img = load_dicom(file_path, as_float=True)
    else:
        img = load_image(file_path, as_float=True)

    save_image(output_dir / f"{name}_01_original.png", img)
    print(f" [1] Nạp ảnh: shape={img.shape}, dtype={img.dtype}, range=[{img.min():.3f}, {img.max():.3f}]")

    # 2. US-05: Trích xuất biên Sobel & Mask nét vẽ
    t0 = time.perf_counter()
    gray = to_grayscale(img)
    _gx, _gy, mag = compute_gradients(gray)
    edge_mask = create_edge_mask(mag, threshold=0.12, invert=True)  # Nền trắng nét đen
    t_edge = (time.perf_counter() - t0) * 1000
    save_image(output_dir / f"{name}_02_sobel_gradient.png", mag)
    save_image(output_dir / f"{name}_02_edge_mask.png", edge_mask)
    print(f" [2] US-05: Sobel Edge Magnitude & Edge Mask ({t_edge:.2f}ms)")

    # 3. US-06: Bộ lọc bảo toàn biên Bilateral Filter (2 lượt)
    t0 = time.perf_counter()
    smooth1 = bilateral_filter(img, d=7, sigma_s=7.0, sigma_r=0.10)
    smooth2 = bilateral_filter(smooth1, d=7, sigma_s=7.0, sigma_r=0.10)
    t_bilat = (time.perf_counter() - t0) * 1000
    save_image(output_dir / f"{name}_03_bilateral_pass1.png", smooth1)
    save_image(output_dir / f"{name}_03_bilateral_pass2.png", smooth2)
    print(f" [3] US-06: Bilateral Filter (2 passes) ({t_bilat:.2f}ms)")

    # 4. US-07: Tranh phác thảo chì (Pencil Sketch)
    t0 = time.perf_counter()
    sketch = pencil_sketch(img, blur_size=21, blur_sigma=10.0, as_float=True)
    t_sketch = (time.perf_counter() - t0) * 1000
    save_image(output_dir / f"{name}_04_pencil_sketch.png", sketch)
    print(f" [4] US-07: Pencil Sketch Art ({t_sketch:.2f}ms)")

    # 5. US-08: Tranh hoạt hình (Cartoonify)
    t0 = time.perf_counter()
    cartoon = cartoonify(img, num_levels=8, d=7, sigma_s=7.0, sigma_r=0.10, edge_threshold=0.12, as_float=True)
    t_cartoon = (time.perf_counter() - t0) * 1000
    save_image(output_dir / f"{name}_05_cartoonify.png", cartoon)
    print(f" [5] US-08: Cartoonify Art ({t_cartoon:.2f}ms)")

    # Kiểm tra tính toàn vẹn ma trận
    assert sketch.ndim == 2, "Lỗi: Sketch phải là ma trận 2D"
    assert 0.0 <= sketch.min() and sketch.max() <= 1.0 + 1e-6
    assert 0.0 <= cartoon.min() and cartoon.max() <= 1.0 + 1e-6

    total_t = (time.perf_counter() - t_start) * 1000
    print(f" [V] Hoàn tất pipeline cho {name} trong {total_t:.2f}ms. Đã lưu kết quả vào {output_dir}/")


def main():
    print("================================================================================")
    print("      DEMO SPRINT 2: PHÁT TRIỂN KỸ THUẬT NÂNG CAO & TẠO HIỆU ỨNG TRANH VẼ")
    print("                     (Bilateral, Sobel, Pencil Sketch, Cartoonify)")
    print("================================================================================")

    output_dir = Path("output_sprint2")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ảnh Tự nhiên
    process_image_sprint2(
        name="natural",
        file_path=Path("data/natural/natural_landscape.png"),
        is_dicom=False,
        output_dir=output_dir
    )

    # 2. Ảnh Công nghiệp
    process_image_sprint2(
        name="industrial",
        file_path=Path("data/industrial/industrial_pcb.png"),
        is_dicom=False,
        output_dir=output_dir
    )

    # 3. Ảnh Y tế DICOM
    process_image_sprint2(
        name="medical",
        file_path=Path("data/medical/chest_ct_scan.dcm"),
        is_dicom=True,
        output_dir=output_dir
    )

    print("\n================================================================================")
    print(">>> SPRINT 2 ĐÃ CHẠY THÀNH CÔNG TRÊN CẢ 3 ĐỊNH DẠNG ẢNH ĐẠT CHUẨN DoD! <<<")
    print(f"Toàn bộ ảnh kết quả được lưu tại: {output_dir.resolve()}")
    print("================================================================================")


if __name__ == "__main__":
    main()
