"""
CLI Runner cho Sprint 2: Tạo tranh vẽ phác thảo chì và hoạt hình.
Sử dụng:
    python run_sprint2.py --input <duong_dan_anh> [--output-dir <thu_muc_dich>]
Kết quả:
    Tự động sinh ra 2 file nghệ thuật: sketch_output.png và cartoon_output.png.
"""

import argparse
import io
import sys
import time
from pathlib import Path

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")

from src.cartoon import cartoonify
from src.io_handler import load_dicom, load_image, save_image
from src.sketch import pencil_sketch


def main():
    parser = argparse.ArgumentParser(
        description="Sprint 2 CLI Runner: Chuyển đổi ảnh thành tranh phác thảo chì (Sketch) và hoạt hình (Cartoonify)."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Đường dẫn đến file ảnh đầu vào (.png, .jpg, .bmp, .dcm)."
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./",
        help="Thư mục lưu trữ kết quả (mặc định: thư mục hiện tại)."
    )
    parser.add_argument(
        "--sketch-name",
        default="sketch_output.png",
        help="Tên file kết quả phác thảo chì (mặc định: sketch_output.png)."
    )
    parser.add_argument(
        "--cartoon-name",
        default="cartoon_output.png",
        help="Tên file kết quả tranh hoạt hình (mặc định: cartoon_output.png)."
    )
    parser.add_argument(
        "--num-levels",
        type=int,
        default=8,
        help="Số mức lượng tử hóa màu cho Cartoonify (mặc định: 8)."
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Không tìm thấy file đầu vào: {input_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sketch_path = output_dir / args.sketch_name
    cartoon_path = output_dir / args.cartoon_name

    print("================================================================================")
    print("        SPRINT 2: CHUYỂN ĐỔI ẢNH NGHỆ THUẬT (SKETCH & CARTOONIFY)")
    print("                    (Thuần NumPy - Không dùng OpenCV)")
    print("================================================================================")
    print(f"[*] Nạp ảnh từ: {input_path.resolve()}")

    t0 = time.perf_counter()
    if input_path.suffix.lower() == ".dcm":
        img = load_dicom(input_path, as_float=True)
    else:
        img = load_image(input_path, as_float=True)
    print(f" -> Nạp ảnh thành công: shape={img.shape}, dtype={img.dtype} ({ (time.perf_counter() - t0)*1000:.2f}ms)")

    # 1. Tạo hiệu ứng tranh chì (Pencil Sketch)
    print("\n[*] 1. Đang xử lý: Tranh phác thảo chì (Pencil Sketch)...")
    t_sketch = time.perf_counter()
    sketch_img = pencil_sketch(img, blur_size=21, blur_sigma=10.0, as_float=True)
    save_image(sketch_path, sketch_img)
    dt_sketch = (time.perf_counter() - t_sketch) * 1000
    print(f" [V] Đã lưu: {sketch_path.resolve()} (Thời gian xử lý: {dt_sketch:.2f}ms)")

    # 2. Tạo hiệu ứng tranh hoạt hình (Cartoonify)
    print("\n[*] 2. Đang xử lý: Tranh hoạt hình (Cartoonify / Cel-Shading)...")
    t_cartoon = time.perf_counter()
    cartoon_img = cartoonify(
        img,
        num_levels=args.num_levels,
        d=7,
        sigma_s=7.0,
        sigma_r=0.1,
        edge_threshold=0.12,
        as_float=True
    )
    save_image(cartoon_path, cartoon_img)
    dt_cartoon = (time.perf_counter() - t_cartoon) * 1000
    print(f" [V] Đã lưu: {cartoon_path.resolve()} (Thời gian xử lý: {dt_cartoon:.2f}ms)")

    print("\n================================================================================")
    print(f"Hoàn thành thành công! Tổng thời gian: { (time.perf_counter() - t0)*1000:.2f}ms")
    print(f"  - Phác thảo chì: {sketch_path}")
    print(f"  - Tranh hoạt hình: {cartoon_path}")
    print("================================================================================")


if __name__ == "__main__":
    main()
