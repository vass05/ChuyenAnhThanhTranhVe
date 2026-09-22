# SLIDE BÁO CÁO NGHIỆM THU ĐỒ ÁN XỬ LÝ ẢNH (CHƯƠNG 5)
## ĐỀ TÀI: XÂY DỰNG PHẦN MỀM CHUYỂN ẢNH THÀNH TRANH VẼ
### Tác giả: vass05 | Công nghệ: Thuần NumPy (Zero OpenCV)

---

### SLIDE 1: GIỚI THIỆU & MỤC TIÊU ĐỒ ÁN
- **Tên đề tài:** Phần mềm chuyển ảnh số thành tranh vẽ nghệ thuật (Pencil Sketch & Cartoonify).
- **Mục tiêu kỹ thuật:**
  - Tự cài đặt toàn bộ thuật toán xử lý ảnh từ số 0 bằng ma trận **NumPy**.
  - Tuyệt đối **không dùng thư viện OpenCV (`cv2`)**.
  - Xây dựng giao diện người dùng trực quan, hỗ trợ nạp/lưu ảnh và xem kết quả thời gian thực.
  - Vượt qua kiểm nghiệm trên 3 lĩnh vực: **Tự nhiên, Công nghiệp, Y tế (CT DICOM)**.

---

### SLIDE 2: QUY TRÌNH SCRUM & DEFINITION OF DONE (DoD)
- **Chu kỳ Sprint:** 3 Sprints (6 tuần).
  - **Sprint 1:** Động cơ cốt lõi (I/O DICOM/PIL, Grayscale BT.601, 2D Convolution, Gaussian Blur).
  - **Sprint 2:** Kỹ thuật nâng cao (Sobel Edge, Bilateral Filter, Pencil Sketch, Cartoonify).
  - **Sprint 3:** Giao diện GUI tương tác, Tối ưu hiệu năng & Đóng gói sản phẩm.
- **Cam kết DoD đạt 100%:**
  - `grep -r "cv2"`: 0 kết quả sử dụng.
  - Vector hóa toàn bộ: Không dùng vòng lặp pixel $O(H \times W)$.
  - Đầy đủ Unit Test: **30/30 tests PASS 100%** qua `pytest`.

---

### SLIDE 3: KIẾN TRÚC THUẬT TOÁN (PIPELINE)
1. **Pipeline Tranh Phác Thảo Chì (Pencil Sketch):**
   - Ảnh gốc $\rightarrow$ Ảnh xám $I_{\text{gray}} \rightarrow$ Nghịch đảo $I_{\text{inv}} \rightarrow$ Gaussian Blur $I_{\text{blur}} \rightarrow$ **Color Dodge Blending**.
   - Công thức: $I_{\text{sketch}} = \min\left(255, \frac{I_{\text{gray}} \times 256}{255 - I_{\text{blur}} + 1}\right)$.
2. **Pipeline Tranh Hoạt Hình (Cartoonify):**
   - Ảnh gốc $\rightarrow$ 2 lượt **Bilateral Filter** $\rightarrow$ **Color Quantization** ($K=8$) $\rightarrow$ Phủ đè **Nét viền Sobel**.

---

### SLIDE 4: ĐIỂM SÁNG KỸ THUẬT: BILATERAL FILTER FROM SCRATCH
- Thách thức: Bộ lọc song phương thường tốn thời gian nếu duyệt từng pixel.
- **Giải pháp Vector hóa sáng tạo:**
  - Lặp qua $(2r + 1)^2$ độ lệch lân cận trong cửa sổ $d \times d$ (chỉ 25 hoặc 49 bước).
  - Nhân đồng thời trọng số không gian $w_s = \exp(-\Delta r^2 / 2\sigma_s^2)$ và trọng số màu sắc $w_r = \exp(-\Delta I^2 / 2\sigma_r^2)$ trên toàn tensor $H \times W$ cùng lúc bằng NumPy SIMD.
  - Tốc độ xử lý chỉ trong vài trăm mili-giây, bảo toàn cạnh biên sắc sảo và làm phẳng mịn khối màu.

---

### SLIDE 5: TỐI ƯU HÓA HIỆU NĂNG TƯƠNG TÁC (FAST PREVIEW)
- **Vấn đề:** Khi người dùng kéo thanh trượt (slider), render toàn bộ ảnh lớn 4K/Full HD liên tục sẽ gây đơ lag GUI.
- **Giải pháp (US-10):**
  - Tự động thu phóng ảnh về bản Preview $\le 400\text{px}$ bằng nội suy Bilinear thuần NumPy.
  - Tốc độ phản hồi tăng tốc **từ 1.8x đến 2.6x** ($< 50\text{ms}$).
  - Khi bấm *"Lưu ảnh"* hoặc *"Render Full Resolution"*, phần mềm tự động kết xuất $100\%$ độ phân giải gốc để đạt chất lượng tranh cao nhất.

---

### SLIDE 6: KẾT QUẢ NGHIỆM THU TRÊN 3 NHÓM ẢNH
1. **Ảnh Tự nhiên (Landscape):**
   - Nét chì mô tả sống động đường chân trời, bóng mờ đồi núi.
   - Tranh hoạt hình tạo các mảng màu hoàng hôn cel-shaded như phim anime.
2. **Ảnh Công nghiệp (PCB Circuit):**
   - Tách biệt rõ nét các đường mạch đồng mạ vàng, chân chip IC vi điều khiển.
   - Bản vẽ kỹ thuật nét chì trắng đen rõ ràng, không bị nhiễu hạt.
3. **Ảnh Y tế (Chest CT Scan DICOM 16-bit):**
   - Bóc tách đường viền xương sườn, cột sống và khối mô mềm lồng ngực.
   - Bộ lọc Bilateral khử sạch nhiễu hạt y tế mà không làm mất thông tin giải phẫu.

---

### SLIDE 7: HƯỚNG DẪN CHẠY PHẦN MỀM & KIỂM THỬ
1. **Khởi chạy Giao diện Ứng dụng (GUI Web App):**
   ```bash
   streamlit run app.py
   ```
   $\rightarrow$ Trình duyệt tự động mở giao diện tương tác Side-by-Side: kéo thả ảnh, chuyển đổi 5 phong cách tranh nghệ thuật, tinh chỉnh thông số thời gian thực và tải ảnh tranh vẽ về máy.
2. **Kiểm thử tự động toàn diện (Automated Testing):**
   ```bash
   pytest -v
   ```
   $\rightarrow$ Thực thi toàn bộ 41/41 bài kiểm thử Unit Test bảo đảm tính toàn vẹn và độ chính xác toán học của các giải thuật.

---

### SLIDE 8: TỔNG KẾT & KẾT LUẬN
- Đồ án đã hoàn thành trọn vẹn 100% mục tiêu cả về **mặt thuật toán toán học** lẫn **mặt ứng dụng phần mềm thực tế**.
- Mã nguồn sạch sẽ, vector hóa hoàn toàn, tuân thủ nghiêm ngặt tiêu chí **Zero OpenCV**.
- Cảm ơn Thầy/Cô và các bạn đã theo dõi!
