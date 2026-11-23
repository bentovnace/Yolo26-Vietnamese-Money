# 📖 HƯỚNG DẪN SỬ DỤNG HỆ THỐNG NHẬN DIỆN & TÍNH TỔNG TIỀN VIỆT NAM (YOLO26)
> Tài liệu hướng dẫn đầy đủ dành cho bạn bè, đồng đội và người sử dụng.

---

## 🎯 Giới thiệu nhanh
Hệ thống sử dụng mô hình trí tuệ nhân tạo thế hệ mới nhất **YOLO26** (kiến trúc NMS-Free End-to-End từ Ultralytics) để:
- Tự động nhận diện **9 mệnh giá tiền Việt Nam**: `1.000đ`, `2.000đ`, `5.000đ`, `10.000đ`, `20.000đ`, `50.000đ`, `100.000đ`, `200.000đ`, `500.000đ`.
- Tự động phân loại, đếm số tờ từng loại và **tính tổng giá trị tiền mặt (VNĐ)** xuất hiện trong ảnh hoặc video.

---

## 📱 CÁCH 1: Dùng trực tiếp trên Trình duyệt (KHÔNG CẦN CÀI ĐẶT)
*Dành cho người dùng thông thường, bạn bè muốn test nhanh trên điện thoại hoặc máy tính.*

1. **Mở đường link ứng dụng:**
   👉 **[https://yolo26-vietnamese-money.onrender.com](https://yolo26-vietnamese-money.onrender.com)**
2. **Cách nhận diện tiền:**
   - **Kéo thả hoặc tải ảnh:** Bấm vào khung tải ảnh để chọn ảnh chụp tiền từ máy.
   - **Chụp trực tiếp từ Webcam:** Bấm nút **"Chụp từ Webcam"**, giơ tờ tiền trước camera và bấm **"Chụp ảnh"**.
3. **Xem kết quả:**
   - **Ô Tổng tiền:** Hiện to rõ tổng số tiền (ví dụ: `750.000 VNĐ - 3 tờ`).
   - **Các thẻ mệnh giá:** Thống kê chi tiết (ví dụ: `500.000đ: 1 tờ`, `200.000đ: 1 tờ`, `50.000đ: 1 tờ`).
   - **Ảnh kết quả:** Hiển thị khung chữ nhật (Bounding Box) đóng dấu vị trí từng tờ tiền cùng độ tin cậy (%).
   - **Thanh trượt điều chỉnh:** Bạn có thể kéo thanh **Ngưỡng tin cậy (Confidence)** từ `0.20` đến `0.50` tùy ý.

---

## 💻 CÁCH 2: Tự tải về và chạy trên máy tính cá nhân
*Dành cho bạn bè muốn chạy offline trên máy tính (hỗ trợ Windows, macOS, Linux).*

### Yêu cầu cấu hình máy:
- **RAM:** Từ 1 GB – 2 GB trở lên (rất nhẹ, laptop học tập chạy tốt).
- Đã cài **Python 3.10+** hoặc **Docker**.

### Bước 1: Tải mã nguồn về máy
Mở Terminal (hoặc Git Bash / Command Prompt) và gõ:
```bash
git clone https://github.com/bentovnace/Yolo26-Vietnamese-Money.git
cd Yolo26-Vietnamese-Money
```

### Bước 2: Khởi chạy ứng dụng
Bạn có thể chọn **1 trong 2 cách** sau:

#### ▶️ Cách 2.1: Chạy bằng Python (Khuyên dùng - Nhanh nhất, kết nối Render)
```bash
# 1. Cài thư viện
pip install -r requirements.txt

# 2. Khởi chạy Web Server Dev
make dev
```
Sau đó mở trình duyệt vào: **`http://localhost:8000`**
> 💡 **Tính năng Cloud-Only:** Khi chạy `make dev`, toàn bộ quá trình nhận diện và tính tổng tiền được chuyển tiếp 100% lên máy chủ Render Cloud `https://yolo26-vietnamese-money.onrender.com`. Máy cục bộ chỉ chạy giao diện điều khiển siêu nhẹ, không tiêu tốn RAM hay GPU của máy bạn!

#### ▶️ Cách 2.2: Chạy bằng Docker (Nếu máy đã cài Docker)
```bash
# Tự động build và chạy
make build
make run
```
Sau đó mở trình duyệt vào: **`http://localhost:8000`**

#### ▶️ Cách 2.3: Chạy nhận diện qua Terminal (CLI)
Nếu không muốn mở giao diện web mà chỉ muốn test 1 file ảnh:
```bash
python predict_cli.py --source /duong/dan/anh_tien.jpg
```

---

## 🔌 CÁCH 3: Tích hợp REST API vào Ứng dụng khác
*Dành cho lập trình viên muốn kết nối vào App Mobile (Flutter, React Native, iOS, Android) hoặc Website.*

- **Tài liệu Swagger UI:** [https://yolo26-vietnamese-money.onrender.com/docs](https://yolo26-vietnamese-money.onrender.com/docs)
- **Endpoint nhận diện:** `POST https://yolo26-vietnamese-money.onrender.com/api/v1/predict`

### Code mẫu gọi API bằng Python:
```python
import requests

url = "https://yolo26-vietnamese-money.onrender.com/api/v1/predict"
files = {"file": open("anh_tien.jpg", "rb")}
params = {"conf": 0.25}

response = requests.post(url, files=files, params=params)
data = response.json()

print(f"Tổng tiền: {data['total_amount_formatted']}")
print(f"Tổng số tờ: {data['total_notes']}")
print("Chi tiết:", data['breakdown'])
```

### Code mẫu gọi API bằng JavaScript / Node.js:
```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);

const response = await fetch("https://yolo26-vietnamese-money.onrender.com/api/v1/predict?conf=0.25", {
  method: "POST",
  body: formData
});
const result = await response.json();
console.log("Tổng tiền:", result.total_amount_formatted);
```

### Cấu trúc dữ liệu JSON trả về:
```json
{
  "success": true,
  "total_notes": 3,
  "total_amount_vnd": 750000,
  "total_amount_formatted": "750.000 VNĐ",
  "breakdown": {
    "500000": 1,
    "200000": 1,
    "50000": 1
  },
  "detections": [
    {
      "box": [120, 85, 450, 310],
      "denomination_str": "500000",
      "confidence": 0.9412,
      "amount_vnd": 500000,
      "formatted_amount": "500.000 VNĐ"
    }
  ]
}
```

---

## 🐍 CÁCH 4: Dùng mô hình trực tiếp trong Python (Không cần server)
Mô hình đã được lưu trữ công khai trên Hugging Face. Bất kỳ ai cũng có thể tải về chỉ bằng 2 dòng code:

```bash
pip install ultralytics
```

```python
from ultralytics import YOLO

# 1. Nạp mô hình trực tiếp từ Hugging Face
model = YOLO("https://huggingface.co/DangKhoa09/yolo26-vietnamese-money/resolve/main/best.pt")

# 2. Nhận diện trên ảnh
results = model.predict("anh_tien.jpg", conf=0.25)

# 3. Xem kết quả
results[0].show()
```

---

## 🎓 CÁCH 5: Tự huấn luyện lại mô hình trên Google Colab
Nếu bạn bè muốn xem quy trình huấn luyện hoặc train lại với số epoch cao hơn:
1. Mở file sổ tay: [`notebooks/train_yolo26_money_vn.ipynb`](https://github.com/bentovnace/Yolo26-Vietnamese-Money/blob/main/notebooks/train_yolo26_money_vn.ipynb)
2. Bấm vào nút **Open in Colab** ở đầu trang.
3. Chạy tuần tự các ô lệnh từ trên xuống dưới (đã tích hợp GPU, Roboflow dataset v2, theo dõi số liệu và tự động nén file kết quả).

---

## ❓ Câu hỏi thường gặp (FAQ)
- **Hỏi: Mô hình có nhận diện được tiền bị gấp đôi hoặc hơi nhàu không?**  
  *Đáp: Có, tập dữ liệu huấn luyện gồm hơn 10.100 ảnh có đầy đủ các góc chụp nghiêng, tiền gấp, tiền nhàu nát và ánh sáng yếu.*
- **Hỏi: Có phân biệt được tiền xu không?**  
  *Đáp: Hiện tại mô hình tập trung vào 9 mệnh giá tiền giấy và polymer đang lưu hành phổ biến nhất tại Việt Nam (1.000đ - 500.000đ).*
- **Hỏi: Làm sao để cập nhật file mô hình mới nhất?**  
  *Đáp: Chỉ cần đặt file `best.pt` vào thư mục `models/` trên máy, server sẽ tự động nhận diện mô hình mới ngay lập tức.*
