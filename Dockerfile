# Base image Python 3.11 slim tối ưu dung lượng
FROM python:3.11-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Ngăn tạo file .pyc và bật unbuffered log
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Cài đặt các thư viện phụ thuộc
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ mã nguồn vào container
COPY . .

# Mở cổng mặc định của Streamlit
EXPOSE 8501

# Lệnh khởi chạy ứng dụng web Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
