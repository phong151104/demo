# 📦 Hướng Dẫn Cài Đặt

## Yêu cầu hệ thống

- **Python**: 3.10 - 3.11 (khuyến nghị 3.11)
- **RAM**: Tối thiểu 8GB
- **Disk**: 2GB trống

---

## 🚀 Cài đặt nhanh

### 1. Tạo môi trường ảo

```bash
# Dùng conda (khuyến nghị)
conda create -n demo python=3.11
conda activate demo

# Hoặc dùng venv
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Chạy ứng dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ mở tại: http://localhost:8501

---

## ⚠️ Xử lý lỗi thường gặp

### Lỗi 1: `cannot import name '_max_precision_float_dtype'`

**Nguyên nhân**: Xung đột phiên bản scikit-learn

**Giải pháp**:
```bash
pip install scikit-learn==1.5.2 scipy==1.11.4 numpy==1.26.2
```

---

### Lỗi 2: Xung đột protobuf khi cài optbinning

**Nguyên nhân**: `optbinning` yêu cầu `protobuf >= 5.26`, nhưng `google-generativeai` yêu cầu `protobuf < 5.0`

**Giải pháp 1** - Không dùng optbinning (khuyến nghị):
- Code đã có fallback sang Decision Tree, vẫn hoạt động tốt

**Giải pháp 2** - Nâng cấp google-generativeai:
```bash
pip install --upgrade google-generativeai
pip install optbinning
```

---

### Lỗi 3: `ModuleNotFoundError`

**Giải pháp**: Cài lại toàn bộ requirements
```bash
pip install -r requirements.txt --force-reinstall
```

---

## 📦 Cài đặt thêm (Tùy chọn)

### OptBinning (Optimal Binning nâng cao)

> ⚠️ **Lưu ý**: Có thể gây xung đột dependency

```bash
pip install optbinning
```

Nếu không cài, ứng dụng sẽ tự động dùng **Decision Tree fallback** - vẫn cho kết quả tốt.

---

## 🔧 Kiểm tra cài đặt

Chạy lệnh sau để kiểm tra các package quan trọng:

```bash
python -c "import streamlit; import pandas; import sklearn; import xgboost; print('✅ Tất cả package OK!')"
```

---

## 📁 Cấu trúc thư mục

```
demo/
├── app.py              # Entry point
├── requirements.txt    # Dependencies
├── INSTALL.md          # File này
├── README.md           # Giới thiệu dự án
├── backend/            # Logic xử lý
├── views/              # Giao diện Streamlit
└── utils/              # Tiện ích
```

---

## 💡 Mẹo

1. **Luôn dùng môi trường ảo** để tránh xung đột package
2. **Kiểm tra phiên bản Python** trước khi cài: `python --version`
3. **Cập nhật pip** trước khi cài: `pip install --upgrade pip`
