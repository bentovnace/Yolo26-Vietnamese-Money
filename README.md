# 🇻🇳 YOLO26 Vietnamese Money Recognition & Smart Calculator
> Hệ thống Nhận diện 9 mệnh giá Tiền Việt Nam (1.000đ - 500.000đ) và Tự động tính tổng tiền bằng mô hình **YOLO26**, sẵn sàng triển khai qua **Docker**, **Makefile**, **FastAPI** và sổ tay huấn luyện trên **Google Colab**.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Ultralytics YOLO26](https://img.shields.io/badge/YOLO-YOLO26-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Roboflow Dataset](https://img.shields.io/badge/Roboflow-Dataset%20v2-purple.svg)](https://universe.roboflow.com/blindtrix/money_vn_regconization-2-cumhs)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Mục lục
- [Giới thiệu](#-giới-thiệu)
- [Thông tin Dataset Roboflow](#-thông-tin-dataset-roboflow)
- [Cấu trúc Thư mục](#-cấu-trúc-thư-mục)
- [Quy trình 1: Huấn luyện trên Google Colab](#-quy-trình-1-huấn-luyện-trên-google-colab)
- [Quy trình 2: Tự động hóa Docker qua Makefile](#-quy-trình-2-tự-động-hóa-docker-qua-makefile)
- [Giao diện Web UI & API](#-giao-diện-web-ui--api)
- [Sử dụng CLI Predictor](#-sử-dụng-cli-predictor)
- [Hướng dẫn Tạo Repo & Push lên GitHub](#-hướng-dẫn-tạo-repo--push-lên-github)

---

## 🌟 Giới thiệu

Dự án áp dụng thế hệ mô hình **YOLO26** mới nhất từ Ultralytics:
- **Kiến trúc NMS-Free (End-to-End):** Loại bỏ bước hậu xử lý Non-Maximum Suppression truyền thống, giúp tốc độ suy luận (inference) trên CPU và thiết bị biên nhanh hơn vượt trội.
- **Tự động tính tổng tiền:** Không chỉ vẽ bounding box quanh các tờ tiền, hệ thống còn tự động phân loại theo 9 mệnh giá, đếm số tờ mỗi loại và tính ra tổng giá trị tiền mặt (VNĐ).
- **Trọn gói sản phẩm:**
  - File Notebook huấn luyện hoàn chỉnh trên Google Colab (tự tải dataset bằng API key, train, xuất ONNX, đóng gói zip tải về).
  - Tự động đóng gói Docker container thông qua `Makefile`.
  - Web UI tương tác thời gian thực (hỗ trợ upload ảnh và chụp trực tiếp từ Webcam).
  - REST API chuẩn OpenAPI/Swagger cho ứng dụng di động hoặc web khác tích hợp.

---

## 📊 Thông tin Dataset Roboflow

- **Tên project:** `blindtrix / money_vn_regconization-2-cumhs`
- **Phiên bản:** Version 2
- **Định dạng:** `yolo26`
- **Tổng số ảnh:** **10,116 ảnh**
  - **Train:** 9,129 ảnh
  - **Validation:** 563 ảnh
  - **Test:** 424 ảnh
- **9 Mệnh giá tiền:**
  1. `1000` (1.000 VNĐ)
  2. `2000` (2.000 VNĐ)
  3. `5000` (5.000 VNĐ)
  4. `10000` (10.000 VNĐ)
  5. `20000` (20.000 VNĐ)
  6. `50000` (50.000 VNĐ)
  7. `100000` (100.000 VNĐ)
  8. `200000` (200.000 VNĐ)
  9. `500000` (500.000 VNĐ)

---

## 📂 Cấu trúc Thư mục

```text
Yolo26-Vietnamse-Money/
├── app/
│   ├── __init__.py
│   ├── detector.py          # Lớp YOLO26 Detector & tính tổng tiền VNĐ
│   ├── main.py              # FastAPI server & REST API endpoints
│   └── templates/
│       └── index.html       # Web UI trực quan (Tailwind CSS, Webcam, Analytics)
├── models/
│   ├── README.md            # Hướng dẫn lưu file best.pt
│   ├── yolo26n.pt           # Checkpoint base pretrained fallback
│   └── .gitkeep
├── notebooks/
│   └── train_yolo26_money_vn.ipynb   # Colab notebook hoàn chỉnh
├── tests/
│   └── test_detector.py     # Smoke test kiểm tra pipeline suy luận
├── Dockerfile               # Multi-stage Dockerfile cho production
├── docker-compose.yml       # Khởi chạy dịch vụ kèm volume mount
├── Makefile                 # Tự động hóa build, run, test, clean
├── predict_cli.py           # Công cụ CLI suy luận trên ảnh/thư mục/webcam
├── requirements.txt         # Thư viện Python phụ thuộc
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 🚀 Quy trình 1: Huấn luyện trên Google Colab

1. Mở file notebook: [`notebooks/train_yolo26_money_vn.ipynb`](notebooks/train_yolo26_money_vn.ipynb)
2. Tải notebook lên [Google Colab](https://colab.research.google.com/) và chọn Runtime GPU (`T4 GPU`).
3. Chạy tuần tự các ô lệnh:
   - Cài đặt thư viện (`roboflow`, `ultralytics`).
   - Tải dataset tự động bằng snippet Roboflow:
     ```python
     from roboflow import Roboflow
     rf = Roboflow(api_key="FtXoE5Gc1n3NlmiAnXvn")
     project = rf.workspace("blindtrix").project("money_vn_regconization-2-cumhs")
     version = project.version(2)
     dataset = version.download("yolo26")
     ```
   - Huấn luyện mô hình YOLO26:
     ```python
     from ultralytics import YOLO
     model = YOLO("yolo26n.pt")  # hoặc yolo26s.pt
     results = model.train(
         data=f"{dataset.location}/data.yaml",
         epochs=50,
         imgsz=640,
         batch=16,
         optimizer="auto",
         project="yolo26_vietnamese_money",
         name="train_run_1"
     )
     ```
   - Chạy ô cuối cùng để Colab tự động nén `best.pt` thành `yolo26_vietnamese_money_weights.zip` và kích hoạt tải về máy.
4. Sau khi tải về: giải nén và copy file `best.pt` vào thư mục `models/` trên máy tính:
   ```bash
   cp ~/Downloads/best.pt models/best.pt
   ```

---

## 🐳 Quy trình 2: Tự động hóa Docker qua Makefile

Dự án cung cấp sẵn `Makefile` với đầy đủ các lệnh tự động hóa:

| Lệnh Makefile | Chức năng |
|---|---|
| `make help` | Hiển thị bảng trợ giúp tất cả các lệnh |
| `make build` | Tự động build Docker Image `yolo26-vietnamese-money:latest` |
| `make run` | Khởi chạy container với port 8000 và mount thư mục `models/` |
| `make stop` | Dừng và xóa container đang chạy |
| `make restart` | Khởi động lại container (stop rồi run) |
| `make logs` | Xem stream log của container |
| `make status` | Kiểm tra trạng thái container |
| `make compose-up` | Khởi chạy toàn bộ hệ thống bằng Docker Compose |
| `make compose-down` | Dừng hệ thống Docker Compose |
| `make test` | Chạy smoke test kiểm tra detector |
| `make dev` | Chạy trực tiếp FastAPI local dev server (không cần docker) |

### Các bước khởi chạy nhanh:
```bash
# 1. Build image Docker
make build

# 2. Khởi chạy container
make run

# 3. Truy cập Web UI tại:
# http://localhost:8000
```

> **Lợi thế volume mount (`-v $(pwd)/models:/app/models`):** Bạn có thể cập nhật hoặc thay thế `models/best.pt` bất kỳ lúc nào mà không cần phải build lại image Docker!

---

## 🖥️ Giao diện Web UI & API

### 1. Web UI trực quan
Truy cập trình duyệt: `http://localhost:8000`
- **Kéo & thả ảnh** hoặc **Chụp từ Webcam trực tiếp**.
- **Điều chỉnh ngưỡng:** Confidence slider (0.10 - 0.95) & IoU slider.
- **Thống kê tổng quan:**
  - Tổng số tiền phát hiện (ví dụ: `580.000 VNĐ`).
  - Tổng số tờ tiền.
  - Phân tích chi tiết: số tờ theo từng mệnh giá (500k, 200k, 100k, 50k, 20k, 10k, 5k, 2k, 1k).
  - Bảng tọa độ bounding box và độ tin cậy của từng tờ tiền.

### 2. REST API Endpoints
Tài liệu Swagger OpenAPI: `http://localhost:8000/docs`

- `GET /health`: Kiểm tra trạng thái server và model đang tải (`best.pt` hay `yolo26n.pt`).
- `POST /api/v1/predict`: Nhận file ảnh và trả về JSON có cấu trúc gồm tọa độ, mệnh giá, số tờ và tổng tiền.
  ```json
  {
    "success": true,
    "total_notes": 4,
    "total_amount_vnd": 580000,
    "total_amount_formatted": "580.000 VNĐ",
    "breakdown": {
      "500000": 1,
      "50000": 1,
      "20000": 1,
      "10000": 1
    },
    "detections": [...]
  }
  ```
- `POST /api/v1/predict/visualize`: Nhận file ảnh và trả về ảnh JPEG đã được vẽ bounding box kèm thanh tiêu đề tóm tắt tổng số tiền.
- `POST /api/v1/reload-model`: Nạp lại file trọng số mới mà không cần restart server.

---

## 💻 Sử dụng CLI Predictor

Chạy suy luận trực tiếp trên dòng lệnh:

```bash
# Dự đoán trên 1 ảnh đơn lẻ
python predict_cli.py --source /path/to/money.jpg --weights models/best.pt --conf 0.3

# Dự đoán trên toàn bộ thư mục ảnh
python predict_cli.py --source /path/to/folder/ --weights models/best.pt --output-dir runs/predict
```

Kết quả in trên Terminal:
```text
============================================================
[*] YOLO26 VIETNAMESE MONEY RECOGNITION CLI
[*] Model: models/best.pt
[*] Processing 1 image(s)...
============================================================

[1/1] Analyzing: tien_viet_nam.jpg
    -> Phat hien: 3 to tien
    -> Tong gia tri: 750.000 VNĐ
    -> Chi tiet: 500,000d: 1 to, 200,000d: 1 to, 50,000d: 1 to
    -> Luu anh tai: runs/predict/pred_tien_viet_nam.jpg

============================================================
[SUMMARY] TONG KET TAT CA HINH ANH:
[*] Tong so to tien: 3 to
[*] TONG SO TIEN:    750.000 VNĐ
============================================================
```

---

## 📤 Hướng dẫn Tạo Repo & Push lên GitHub

Để đưa mã nguồn lên tài khoản GitHub của bạn (`bentovnace`):

### Cách 1: Sử dụng GitHub CLI (`gh`) (Khuyến nghị - Nhanh nhất)
```bash
# 1. Đăng nhập GitHub
gh auth login

# 2. Tự động tạo repo mới và push toàn bộ code trong 1 lệnh duy nhất:
gh repo create bentovnace/Yolo26-Vietnamese-Money --public --source=. --remote=origin --push
```

### Cách 2: Sử dụng Git truyền thống
```bash
# 1. Tạo repository mới trên web https://github.com/new với tên: Yolo26-Vietnamese-Money

# 2. Liên kết remote và đẩy code lên:
git remote add origin https://github.com/bentovnace/Yolo26-Vietnamese-Money.git
git branch -M main
git push -u origin main
```

---

## 📜 Giấy phép
Dự án được phân phối dưới giấy phép **MIT License**.
