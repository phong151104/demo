# ⚡ Quick Start Guide

## 🚀 Chạy Ứng Dụng Nhanh (Chỉ Frontend)

### Windows:
```bash
# Double-click vào file:
RUN_APP.bat
```
hoặc mở Command Prompt và chạy:
```bash
.\RUN_APP.bat
```

### macOS/Linux:
```bash
chmod +x RUN_APP.sh
./RUN_APP.sh
```

### Hoặc chạy trực tiếp:
```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy app
streamlit run app.py
```

## 🌐 Truy Cập

Ứng dụng sẽ tự động mở tại: **http://localhost:8501**

---

## 📋 Hướng Dẫn Sử Dụng

### 1. Trang Chủ 🏠
- Xem tổng quan về hệ thống
- Kiểm tra trạng thái hiện tại
- Hướng dẫn nhanh 5 bước

### 2. Upload & EDA 📤
1. Click "Browse files" để upload file CSV
2. Xem dữ liệu mẫu và thống kê
3. Khám phá biểu đồ phân phối (Histogram, Boxplot, Correlation)
4. Xem phân tích tự động (mock AI analysis)

**Lưu ý**: Có thể tải file mẫu từ expander "Xem Định Dạng Mẫu"

### 3. Xử Lý & Chọn Biến ⚙️
1. **Tab Tiền Xử Lý**: Cấu hình xử lý giá trị thiếu, mã hóa, scaling
2. **Tab Binning**: Phân nhóm biến liên tục
3. **Tab Feature Importance**: Xem độ quan trọng của các biến
4. **Tab Chọn Biến**: Chọn features cho model (manual hoặc auto)

### 4. Huấn Luyện Mô Hình 🤖
1. **Tab Cấu Hình**:
   - Chọn loại mô hình (Logistic, Random Forest, XGBoost, LightGBM, v.v.)
   - Điều chỉnh tham số
   - Click "Huấn Luyện Mô Hình"
2. **Tab Kết Quả**: Xem ROC curve, Confusion Matrix, metrics
3. **Tab So Sánh**: So sánh nhiều mô hình

### 5. Giải Thích (SHAP) 🔍
1. Click "Khởi Tạo SHAP Explainer"
2. **Tab Global**: Xem feature importance tổng thể
3. **Tab Local**: Phân tích chi tiết từng mẫu
4. **Tab AI Interpretation**: Phân tích bằng ngôn ngữ tự nhiên (mock)

### 6. Dự Đoán & Gợi Ý 🎯
1. **Tab Nhập Thông Tin**: Điền form thông tin khách hàng
2. **Tab Kết Quả**: Xem điểm tín dụng, risk level, SHAP explanation
3. **Tab Gợi Ý**: Nhận gợi ý cải thiện từ AI (mock)

---

## 📊 Test Data

Nếu chưa có dữ liệu, download file mẫu từ trang "Upload & EDA" hoặc sử dụng bất kỳ dataset credit scoring nào có format:

```csv
feature1,feature2,feature3,...,target
value1,value2,value3,...,0
value1,value2,value3,...,1
```

**Gợi ý datasets công khai:**
- [UCI Credit Approval](https://archive.ics.uci.edu/ml/datasets/credit+approval)
- [Kaggle Credit Risk](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- [German Credit Data](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))

---

## ⚠️ Lưu Ý Quan Trọng

### ✅ Hiện tại (Đã hoàn thành):
- ✨ Giao diện đầy đủ 6 trang
- 🎨 Dark theme chuyên nghiệp
- 📊 Các biểu đồ tương tác (Plotly)
- 🔄 Quản lý session state giữa các trang
- 📋 Tất cả UI components và form inputs

### ⏳ Backend (Chưa triển khai):
- ❌ Data preprocessing logic
- ❌ Model training thực tế
- ❌ SHAP calculations
- ❌ LLM integration
- ❌ Model persistence

**Tất cả hiện đang dùng mock data và placeholder responses.**

---

## 🔧 Phát Triển Backend

Để bắt đầu phát triển backend, xem file:
- **DEVELOPER_GUIDE.md** - Hướng dẫn chi tiết từng module
- **backend/** - Cấu trúc thư mục đã sẵn sàng

Các file backend có TODO comments rõ ràng về chức năng cần implement.

---

## 🐛 Troubleshooting

### Lỗi: "streamlit: command not found"
```bash
pip install streamlit
```

### Lỗi: Port 8501 đang được sử dụng
```bash
streamlit run app.py --server.port 8502
```

### Lỗi: Module not found
```bash
pip install -r requirements.txt --force-reinstall
```

### Clear cache
```bash
streamlit cache clear
```

---

## 💡 Tips

1. **Session State**: Dữ liệu sẽ được lưu trong session. Refresh trang sẽ mất dữ liệu
2. **Navigation**: Sử dụng sidebar để chuyển trang
3. **Mock Data**: Tất cả kết quả hiện tại là mock - ngẫu nhiên mỗi lần
4. **Upload Data**: Để test đầy đủ flow, nên upload một file CSV thật

---

## 📞 Support

Nếu cần hỗ trợ:
1. Xem **README.md** cho thông tin tổng quan
2. Xem **DEVELOPER_GUIDE.md** cho hướng dẫn phát triển
3. Check issues trong repository

---

**Chúc bạn khám phá thành công! 🎉**

