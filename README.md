# 📌 DỰ ÁN NHẬN DIỆN TRÁI THANH LONG BẰNG HỌC SÂU

## 1. Giới thiệu
Dự án này sử dụng **mô hình học sâu (Deep Learning)** để phát hiện và phân loại trái thanh long trong hình ảnh. Ứng dụng bao gồm hai phần chính:

- **Backend (Python):** Xử lý ảnh và chạy mô hình học máy.  
- **Frontend (React + TypeScript):** Cho phép người dùng tải ảnh lên và xem kết quả dự đoán.

Trong dự án sử dụng 2 mô hình `.pth` đã huấn luyện sẵn:
- **Phân loại:** Nhận biết loại thanh long (`reject`, `good`, `immature`).  
- **Phát hiện:** Xác định vị trí trái thanh long trong ảnh.

---

## 2. Cấu trúc thư mục chính
- `classifier_model2.pth`: Mô hình phân loại trái thanh long.  
- `detector_model.pth`: Mô hình phát hiện vị trí trái thanh long.  
- `server.py`: File Python chạy backend.  
- `src/`: Chứa mã nguồn frontend (React).  
- `dragon-detection/`: Chứa mã Python xử lý mô hình (`predict.py`, `model.py`, `use_model.py`, … và file JSON cấu hình).  
- `index.html`: Trang HTML chính.  
- `package.json`, `package-lock.json`: Thông tin cấu hình frontend (Node.js).  
- `README.md`: Tài liệu hướng dẫn dự án (file này).  

---

## 3. Hướng dẫn cài đặt và chạy dự án:
#    a. Cài đặt môi trường Python:
	Mở terminal và chạy lệnh sau để cài các thư viện cần thiết: <br>
	pip install <tên_thư_viện>

#    b. Chạy server backend:
	Sau khi cài xong thư viện, chạy file server.py bằng lệnh: <br>
	python server.py 

#    c. Cài đặt và chạy frontend (React):
	Chuyển vào thư mục frontend (nơi có package.json), sau đó chạy: <br>
	npm install <br>
	npm run start <br>
	Trang web sẽ được chạy tại địa chỉ: http://localhost:5173

## 4. Thông tin mô hình sử dụng:
    • classifier_model2.pth: Dùng để phân loại các loại thanh long. Mô hình đã được huấn luyện trên tập dữ liệu hình ảnh thực tế.
    • detector_model.pth: Dùng để xác định vị trí (bounding box) của trái thanh long trong ảnh.

## 5. Yêu cầu hệ thống:
    • Python 3.8 trở lên
    • Node.js phiên bản 14 trở lên

## 6. Cách sử dụng:
    1 Mở trang web tại địa chỉ http://localhost:5173
    2 Tải lên một ảnh có chứa trái thanh long
    3 Hệ thống sẽ xử lý và hiển thị ảnh với các vị trí trái thanh long được phát hiện, đồng thời phân loại từng trái thanh long
    4 Có thể thử lại với nhiều ảnh khác nhau để kiểm tra

## 7. Các thư viện cần cài đặt
    • torch
    • numpy
    • opencv-python
    • pillow

8. Thông tin tác giả:
    • Họ tên: [Tên của bạn]
    • Email liên hệ: [Email nếu cần]
    • Ngày hoàn thành dự án: Tháng 9 năm 2025
