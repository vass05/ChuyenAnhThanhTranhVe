# Chuyển Ảnh Thành Tranh Vẽ (Artistic Image Converter)

> **Đồ án môn học:** Xử lý ảnh số (Chương 5)  
> **Tác giả / Nhóm:** `vass05`  
> **Công nghệ:** 100% Thuần **Python & NumPy** — Tuyệt đối **Zero OpenCV** (`cv2`).

---

## Giới Thiệu Dự Án
**Chuyển Ảnh Thành Tranh Vẽ** là ứng dụng web tương tác trực quan giúp biến đổi các bức ảnh số thông thường thành các tác phẩm tranh vẽ nghệ thuật sống động. Toàn bộ các thuật toán xử lý ảnh số nền tảng và nâng cao (tích chập 2D, bộ lọc Gauss khả tách, bộ lọc song phương Bilateral, toán tử Sobel, Color Dodge, lượng tử hóa màu...) đều được tự cài đặt từ đầu bằng đại số tuyến tính ma trận NumPy vector hóa cao độ.

---

## 5 Phong Cách Tranh Nghệ Thuật
1. **Tranh Hoạt Hình (Cartoonify / Anime):** 
   - Làm phẳng khối màu bằng 2 lượt Bilateral Filter bảo toàn cạnh biên sắc nét.
   - Lượng tử hóa màu cel-shading bảo toàn sắc thái da người kết hợp hiệu ứng Anime Highlight Bloom.
2. **Tranh Phác Thảo Chì (Pencil Sketch):**
   - Multi-scale Color Dodge Blending bóc tách nét phác thảo thanh mảnh và bóng mờ than chì 4B-6B.
   - Tích hợp phủ bóng than chì tự nhiên và vi cấu trúc vân giấy ký họa (Paper Tooth Grain).
3. **Tranh Chì Màu (Color Pencil Sketch):**
   - Hòa trộn trừ sắc tố sáp màu trên mặt giấy trắng, giữ nguyên độ rực rỡ của cảnh vật.
4. **Tranh Màu Nước (Watercolor):**
   - Làm mịn loang màu nước mềm mại, tăng cường độ bão hòa sắc tố kết hợp viền cọ màu nước.
5. **Nét Vẽ Liền (Continuous Line Art):**
   - Trích xuất biên độ đa kênh RGB kết hợp khung hướng dẫn đa tỷ lệ (Multi-scale Guidance) bắt trọn cằm, mắt, mũi và dáng người.
   - Thuật toán đóng hình thái học (Morphological Closing) tự động hàn gắn nét đứt đoạn, tạo đường nét liền mạch sâu thẳm.

---

## Điểm Sáng Kỹ Thuật
- **Zero OpenCV:** Không sử dụng thư viện `cv2`, toàn bộ tính toán đều viết bằng toán học ma trận NumPy.
- **Tối ưu hóa Vector hóa SIMD:** Không dùng vòng lặp pixel $O(H \times W)$, tốc độ xử lý chỉ trong vài chục mili-giây.
- **Hỗ trợ ảnh Đa Nguồn:** Đọc ảnh số thông thường (JPG, PNG, BMP) và cả **ảnh y tế CT Scan lồng ngực DICOM 16-bit (`.dcm`)**.
- **Giao diện Web Trực Quan (Streamlit):**
  - Hỗ trợ song ngữ **Tiếng Việt - Tiếng Anh**.
  - So sánh trực quan Side-by-Side (Ảnh gốc vs Kết quả).
  - Tinh chỉnh thông số thời gian thực (độ sáng, tương phản, độ tươi màu, độ đậm nét).
  - Không bị mất ảnh hay reset thông số khi chuyển đổi ngôn ngữ hoặc chuyển màn hình.
  - Tải tranh vẽ chất lượng cao về máy (PNG).

---

## Hướng Dẫn Cài Đặt & Khởi Chạy

### 1. Cài đặt môi trường & thư viện
Yêu cầu **Python 3.10+**. Cài đặt các gói phụ thuộc cần thiết:
```bash
pip install numpy pillow streamlit pydicom pytest
```

### 2. Khởi chạy Ứng dụng Web
```bash
streamlit run app.py
```
Trình duyệt sẽ tự động mở giao diện tại địa chỉ: `http://localhost:8501`.

### 3. Chạy kiểm thử tự động (Unit Tests)
Dự án được bảo đảm chất lượng với **40/40 Unit Tests PASS 100%**:
```bash
pytest -v
```

---

## Cấu Trúc Thư Mục Dự Án
```text
ChuyenAnhThanhTranhVe/
│── app.py                   # Ứng dụng Web Streamlit chính diện (GUI)
│── DOCS_ALGORITHMS.md       # Tài liệu toán học & giải thuật chi tiết
│── SLIDES_PRESENTATION.md   # Slide báo cáo nghiệm thu đồ án
│── README.md                # Giới thiệu tổng quan dự án
│
├── src/                     # Động cơ thuật toán thuần NumPy
│   ├── __init__.py          # Export các hàm xử lý công khai
│   ├── io_handler.py        # Đọc/ghi ảnh PIL & ảnh y tế DICOM
│   ├── grayscale.py         # Chuyển đổi mức xám chuẩn ITU-R BT.601
│   ├── convolution.py       # Động cơ tích chập 2D & 1D vector hóa
│   ├── filters.py           # Bộ lọc Gaussian Blur & Bilateral Filter
│   ├── edges.py             # Toán tử Sobel & Pipeline Nét vẽ liền
│   ├── sketch.py            # Phác thảo chì than & Chì màu
│   ├── cartoon.py           # Pipeline hoạt hình hóa Cel-shading
│   ├── watercolor.py        # Pipeline hiệu ứng màu nước
│   ├── color_adjust.py      # Tinh chỉnh độ sáng, tương phản, độ tươi màu
│   └── resizer.py           # Thu phóng ảnh song tuyến tính (Bilinear)
│
├── tests/                   # 40 bài kiểm thử tự động (Pytest)
├── data/                    # Ảnh mẫu thử nghiệm (Tự nhiên, Công nghiệp, Y tế DICOM)
└── assets/                  # Logo và tài nguyên giao diện
```
