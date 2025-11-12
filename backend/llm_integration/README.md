# LLM Integration Module

Module tích hợp Large Language Models để phân tích tự động dữ liệu EDA và cung cấp insights.

## 📋 Tính Năng

### 1. EDA Data Collector
Thu thập toàn bộ thông tin từ quá trình EDA:
- Thông tin cơ bản (rows, columns, memory usage)
- Missing data analysis
- Numeric statistics (mean, median, outliers, skewness, kurtosis)
- Categorical statistics (cardinality, entropy, top values)
- Correlation analysis
- Data quality issues detection

### 2. LLM EDA Analyzer
Sử dụng LLM để phân tích và đưa ra nhận xét:
- Đánh giá chất lượng tổng thể
- Phân tích chi tiết từng loại vấn đề
- Đề xuất roadmap tiền xử lý
- Dự đoán khả năng xây dựng mô hình

## 🚀 Cách Sử Dụng

### Bước 1: Cấu hình API Key

Tạo file `.env` trong thư mục gốc:

```bash
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4

# Hoặc Anthropic Claude
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# Chọn provider
LLM_PROVIDER=openai
```

### Bước 2: Import và Sử Dụng

```python
from backend.llm_integration import analyze_eda_with_llm, get_eda_summary, LLMConfig
import pandas as pd

# Load data
df = pd.read_csv("data.csv")

# Kiểm tra cấu hình
if LLMConfig.is_configured():
    # Phân tích với LLM
    analysis = analyze_eda_with_llm(df)
    print(analysis)
else:
    # Chế độ template (không cần API key)
    analysis = analyze_eda_with_llm(df, api_key=None)
    print(analysis)

# Hoặc chỉ lấy EDA summary
summary_text = get_eda_summary(df, format="text")
summary_json = get_eda_summary(df, format="json")
```

### Bước 3: Tích hợp vào Streamlit

```python
import streamlit as st
from backend.llm_integration import analyze_eda_with_llm, LLMConfig

# Check configuration
is_configured = LLMConfig.is_configured()

if st.button("Phân tích AI"):
    with st.spinner("Đang phân tích..."):
        api_key = LLMConfig.get_api_key() if is_configured else None
        analysis = analyze_eda_with_llm(st.session_state.data, api_key=api_key)
        st.markdown(analysis)
```

## 📊 Thông Tin Thu Thập

### Basic Info
- Số dòng, số cột
- Kiểu dữ liệu của từng cột
- Memory usage

### Missing Data
- Tổng số giá trị thiếu
- Phân tích theo từng cột
- Số dòng hoàn chỉnh

### Numeric Statistics
Cho mỗi biến số:
- Mean, median, std, min, max
- Quantiles (Q1, Q3, IQR)
- Outliers count & percentage
- Skewness & kurtosis
- Coefficient of variation
- Zeros count

### Categorical Statistics
Cho mỗi biến phân loại:
- Số giá trị unique
- Giá trị phổ biến nhất
- Top 5 values
- High cardinality detection
- Entropy

### Correlations
- Ma trận tương quan
- High correlations (≥0.5)
- Average & max correlation

### Data Quality Issues
- Cột có >30% missing
- High cardinality categorical
- Potential ID columns
- Constant columns
- High outliers (>10%)
- Highly skewed (|skew| > 2)
- Duplicate rows

## 🎯 Output Format

LLM trả về phân tích theo cấu trúc:

```markdown
## 1. ĐÁNH GIÁ TỔNG QUAN
- Chất lượng tổng thể
- Điểm mạnh/yếu
- Mức độ sẵn sàng

## 2. PHÂN TÍCH CHI TIẾT
### 2.1 Dữ Liệu Thiếu
### 2.2 Biến Số
### 2.3 Biến Phân Loại
### 2.4 Tương Quan

## 3. VẤN ĐỀ CẦN ƯU TIÊN

## 4. ROADMAP TIỀN XỬ LÝ

## 5. KẾT LUẬN
```

## 🔧 Mở Rộng

### Thêm Provider Mới

```python
# backend/llm_integration/providers/custom_provider.py
class CustomLLMProvider:
    def __init__(self, api_key):
        self.api_key = api_key
    
    def analyze(self, prompt):
        # Implementation
        pass
```

### Custom Prompts

```python
analyzer = LLMEDAAnalyzer(api_key="...")

# Override prompt
custom_prompt = analyzer.create_analysis_prompt(eda_summary)
custom_prompt += "\n\nFocus on credit risk specific insights."

# Use custom prompt
# ... call LLM with custom_prompt
```

## 📝 Notes

- **Template Mode**: Khi không có API key, sử dụng phân tích template tự động
- **Cost**: GPT-4 có chi phí cao, xem xét sử dụng GPT-3.5-turbo cho demo
- **Privacy**: Không gửi dữ liệu nhạy cảm cho LLM API
- **Caching**: Cân nhắc cache kết quả để tiết kiệm cost

## 🔐 Security

- API key được lưu trong `.env`, không commit vào git
- `.env` đã được thêm vào `.gitignore`
- Sử dụng `python-dotenv` để load safely

## 📚 Dependencies

```txt
openai>=1.6.1          # OpenAI GPT
anthropic>=0.8.1       # Anthropic Claude (optional)
python-dotenv>=1.0.0   # Environment variables
scipy>=1.11.4          # Statistical analysis
```
