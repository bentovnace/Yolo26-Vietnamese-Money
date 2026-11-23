# Thư mục chứa Model Weights (YOLO26)

Thư mục này lưu trữ các trọng số (weights) của mô hình YOLO26 phục vụ nhận diện tiền Việt Nam.

## Cấu trúc:
- `yolo26n.pt`: Mô hình base pre-trained chuẩn từ Ultralytics (tự động dùng làm fallback nếu chưa có `best.pt`).
- `best.pt`: Trọng số tối ưu sau khi train trên Google Colab với bộ dữ liệu Roboflow `blindtrix/money_vn_regconization-2-cumhs`.

## Cách sử dụng:
1. Mở notebook `notebooks/train_yolo26_money_vn.ipynb` trên Google Colab.
2. Hoàn tất quá trình train (weights sẽ được lưu tại `runs/detect/yolo26_vietnamese_money/weights/best.pt`).
3. Tải file `best.pt` về máy.
4. Di chuyển file `best.pt` vào thư mục này (`models/best.pt`).
5. Nếu Docker đang chạy, gọi endpoint `/api/v1/reload-model` hoặc `make restart` để mô hình mới có hiệu lực ngay lập tức.
