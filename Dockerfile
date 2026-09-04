# Mặc dù bạn yêu cầu không tạo/build image mới (do đó docker-compose.yml 
# sẽ map trực tiếp thư mục code vào image có sẵn), Dockerfile này vẫn được 
# tạo ra làm tài liệu tham khảo cho việc đóng gói ứng dụng sau này nếu cần.

FROM huyquang/deepstream-app:8.0

# Đặt thư mục làm việc
WORKDIR /app

# (Tuỳ chọn) Copy mã nguồn vào image thay vì dùng Volume nếu muốn build image độc lập:
# COPY . /app/

# Khởi chạy script chính của ứng dụng
CMD ["python3", "-m", "src.main"]
