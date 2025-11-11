# 📁 Cấu Trúc Dự Án Chi Tiết

## 🌳 Tổng Quan Cây Thư Mục

```
credit-scoring-system/
│
├── 📄 app.py                          # File chính - Entry point của ứng dụng
├── 📄 requirements.txt                # Python dependencies
├── 📄 README.md                       # Documentation tổng quan
├── 📄 QUICKSTART.md                   # Hướng dẫn chạy nhanh
├── 📄 DEVELOPER_GUIDE.md              # Hướng dẫn phát triển backend
├── 📄 PROJECT_STRUCTURE.md            # File này - Mô tả cấu trúc
├── 📄 .gitignore                      # Git ignore rules
├── 📄 env.example                     # Template cho environment variables
├── 📄 RUN_APP.bat                     # Script chạy trên Windows
├── 📄 RUN_APP.sh                      # Script chạy trên macOS/Linux
│
├── 📁 .streamlit/                     # Streamlit configuration
│   └── 📄 config.toml                 # Theme và server settings
│
├── 📁 assets/                         # Tài nguyên tĩnh
│   └── 📄 .gitkeep                    # Placeholder (thêm logo.png vào đây)
│
├── 📁 pages/                          # Các trang của ứng dụng
│   ├── 📄 __init__.py                 # Package marker
│   ├── 📄 home.py                     # 🏠 Trang chủ
│   ├── 📄 upload_eda.py               # 📤 Upload & EDA
│   ├── 📄 feature_engineering.py      # ⚙️ Xử lý & chọn biến
│   ├── 📄 model_training.py           # 🤖 Huấn luyện mô hình
│   ├── 📄 shap_explanation.py         # 🔍 Giải thích SHAP
│   └── 📄 prediction.py               # 🎯 Dự đoán & gợi ý
│
├── 📁 utils/                          # Utilities và helpers
│   ├── 📄 __init__.py                 # Package marker
│   ├── 📄 ui_components.py            # UI components tùy chỉnh
│   └── 📄 session_state.py            # Quản lý session state
│
└── 📁 backend/                        # Backend logic (sẽ phát triển)
    ├── 📄 __init__.py                 # Package marker
    │
    ├── 📁 data_processing/            # Module xử lý dữ liệu
    │   └── 📄 __init__.py             # TODO: preprocessing functions
    │
    ├── 📁 models/                     # Module ML models
    │   └── 📄 __init__.py             # TODO: training & evaluation
    │
    ├── 📁 explainability/             # Module SHAP & giải thích
    │   └── 📄 __init__.py             # TODO: SHAP explainer
    │
    └── 📁 llm_integration/            # Module tích hợp LLM
        └── 📄 __init__.py             # TODO: LLM clients
```

---

## 📄 Chi Tiết Từng File

### 🔹 Root Files

#### `app.py` - Main Application
**Chức năng:**
- Entry point của toàn bộ ứng dụng
- Cấu hình Streamlit page config
- Render sidebar navigation
- Routing giữa các trang
- Load custom CSS

**Các thành phần chính:**
```python
st.set_page_config()           # Cấu hình trang
load_custom_css()              # Load CSS tùy chỉnh
render_header()                # Header chính
# Sidebar navigation với radio buttons
# Page routing logic
```

#### `requirements.txt` - Dependencies
**Thư viện chính:**
- `streamlit` - Framework web
- `pandas`, `numpy` - Xử lý dữ liệu
- `scikit-learn`, `xgboost`, `lightgbm`, `catboost` - ML models
- `shap` - Model explainability
- `plotly`, `matplotlib`, `seaborn` - Visualization
- (Optional) `openai`, `langchain` - LLM integration

#### `.streamlit/config.toml` - Theme Configuration
**Cấu hình:**
- Theme colors (dark mode)
- Server settings (port, headless)
- Browser settings

---

## 📁 Pages Module

### 🏠 `home.py` - Trang Chủ
**Nội dung:**
- Welcome message
- Tổng quan tính năng (5 features)
- Trạng thái phiên làm việc hiện tại
- Quick start guide (5 steps)
- Thông tin kỹ thuật

**Session State Dependencies:**
- `data` - Kiểm tra xem đã upload dữ liệu chưa
- `processed_data` - Kiểm tra xem đã xử lý chưa
- `model` - Kiểm tra xem đã train chưa
- `selected_features` - Số features đã chọn

---

### 📤 `upload_eda.py` - Upload & Exploratory Data Analysis
**4 Tabs:**

**Tab 1: Dữ Liệu Mẫu**
- File uploader (CSV)
- Dataframe viewer (head/tail/random)
- Metrics: số dòng, số cột, missing %, numeric cols

**Tab 2: Thống Kê Mô Tả**
- `describe()` statistics cho biến số
- Value counts cho biến phân loại
- Missing data analysis
- Download CSV button

**Tab 3: Phân Phối Dữ Liệu**
- Histogram với bins adjustable
- Boxplot cho outlier detection
- Correlation heatmap
- High correlation pairs

**Tab 4: Phân Tích AI**
- LLM analysis placeholder
- Mock analysis text
- Backend integration notes

**Session State Updates:**
- `st.session_state.data = uploaded_data`

---

### ⚙️ `feature_engineering.py` - Feature Engineering
**4 Tabs:**

**Tab 1: Tiền Xử Lý**
- Xử lý missing values (4 methods)
- Mã hóa categorical (4 methods)
- Scaling/normalization (3 methods)
- Outlier handling (3 methods)
- Data balancing (3 methods)

**Tab 2: Binning**
- Chọn biến để bin
- 3 methods: equal width/frequency/custom
- Slider cho số bins
- Visualization của binning
- Bin statistics table

**Tab 3: Feature Importance**
- Chọn method (RF, LightGBM, XGBoost, Logistic)
- Top N features slider
- Horizontal bar chart (sorted by importance)
- Mock importance values

**Tab 4: Chọn Biến**
- Chọn target variable
- 2 modes: manual/auto selection
- Multi-select cho features
- Summary metrics (numeric/categorical/total)
- Feature list display

**Session State Updates:**
- `st.session_state.selected_features = [...]`
- `st.session_state.processed_data = ...`

---

### 🤖 `model_training.py` - Model Training
**3 Tabs:**

**Tab 1: Cấu Hình Mô Hình**
- Model selection dropdown (6 models)
- Train/test split slider
- Random state input
- Model-specific hyperparameters
- Train button → saves to session state

**Tab 2: Kết Quả Đánh Giá**
- 5 metrics: Accuracy, Precision, Recall, F1, AUC
- Confusion Matrix heatmap
- ROC Curve với AUC
- Classification report table
- Precision-Recall curve
- Save model button

**Tab 3: So Sánh Mô Hình**
- Multi-select models
- Comparison table (all metrics)
- Bar chart comparison
- Multiple ROC curves overlay
- Best model recommendation

**Session State Updates:**
- `st.session_state.model = trained_model`
- `st.session_state.model_type = "XGBoost"`
- `st.session_state.model_metrics = {...}`

---

### 🔍 `shap_explanation.py` - SHAP Explanation
**3 Tabs:**

**Tab 1: Global Explanation**
- Initialize SHAP explainer button
- Feature importance bar chart (mean |SHAP|)
- Top features table with download
- SHAP value distribution (beeswarm simulation)
- Color-coded by feature value

**Tab 2: Local Explanation**
- Sample selection (by index/random/new input)
- Prediction info (probability, class)
- SHAP Waterfall plot
- Force plot (horizontal bar)
- Top positive/negative impacts tables

**Tab 3: AI Interpretation**
- Analysis type radio (global/local)
- Generate AI analysis button
- Mock LLM responses
- LLM settings (provider, temperature, tokens)
- Interactive Q&A section

**Session State Updates:**
- `st.session_state.explainer = shap_explainer`
- `st.session_state.shap_values = values`

---

### 🎯 `prediction.py` - Prediction & Recommendations
**3 Tabs:**

**Tab 1: Nhập Thông Tin**
- Comprehensive input form
- 3 sections: Personal Info, Additional Info, Financial Info
- 15+ input fields (number inputs, selectboxes, sliders)
- Submit button → calculates prediction

**Tab 2: Kết Quả Dự Đoán**
- 3 main metrics: Credit Score (300-850), Risk Level, Probability
- Credit score gauge chart
- Score interpretation
- SHAP explanation for prediction
- Feature impact bar chart

**Tab 3: Gợi Ý Cải Thiện**
- AI-generated recommendations button
- Detailed action plan (priority/long-term)
- Target score calculator
- 3-phase improvement roadmap
- Download report button

**Session State Updates:**
- `st.session_state.prediction_input = {...}`
- `st.session_state.prediction_result = {...}`

---

## 📁 Utils Module

### `ui_components.py` - UI Components
**Functions:**

1. `load_custom_css()` - Loads custom CSS styles
2. `render_header()` - Renders main app header
3. `render_metric_card()` - Gradient metric cards
4. `render_info_card()` - Information cards
5. `show_llm_analysis()` - LLM analysis display
6. `show_processing_placeholder()` - Backend placeholders

**CSS Classes:**
- `.main-header` - Gradient header
- `.info-card` - Information cards
- `.metric-card` - Metric display
- `.stButton > button` - Custom buttons
- Various Streamlit component overrides

### `session_state.py` - Session State Management
**Functions:**

1. `init_session_state()` - Initialize all session variables
2. `clear_session_state()` - Reset session
3. `get_session_info()` - Get current session status

**Session Variables:**
- `data` - Uploaded dataset
- `processed_data` - Processed dataset
- `selected_features` - List of selected features
- `model` - Trained model
- `model_type` - Model type name
- `model_metrics` - Evaluation metrics dict
- `explainer` - SHAP explainer object
- `shap_values` - SHAP values array
- `prediction_input` - User input for prediction
- `prediction_result` - Prediction results

---

## 📁 Backend Module (Cấu Trúc Sẵn Sàng)

### `backend/data_processing/`
**TODO: Implement**
- `preprocessing.py` - DataPreprocessor class
- `feature_engineering.py` - Feature creation & selection
- `binning.py` - Binning strategies

### `backend/models/`
**TODO: Implement**
- `trainer.py` - ModelTrainer class
- `evaluator.py` - ModelEvaluator class
- `comparison.py` - Model comparison utilities

### `backend/explainability/`
**TODO: Implement**
- `shap_explainer.py` - SHAPExplainer class
- `visualization.py` - SHAP plot generators

### `backend/llm_integration/`
**TODO: Implement**
- `llm_client.py` - LLM API clients
- `prompts.py` - Prompt templates
- `analyzers.py` - Analysis functions

---

## 🔄 Data Flow

```
1. Upload Data (upload_eda.py)
   ↓ saves to st.session_state.data
   
2. Process Data (feature_engineering.py)
   ↓ saves to st.session_state.processed_data
   ↓ saves to st.session_state.selected_features
   
3. Train Model (model_training.py)
   ↓ saves to st.session_state.model
   ↓ saves to st.session_state.model_metrics
   
4. Explain Model (shap_explanation.py)
   ↓ saves to st.session_state.explainer
   ↓ saves to st.session_state.shap_values
   
5. Make Prediction (prediction.py)
   ↓ uses all above session state
   ↓ saves to st.session_state.prediction_result
```

---

## 🎨 Styling & Theme

### Color Palette
- **Primary Gradient**: #667eea → #764ba2 (Purple-Blue)
- **Background**: #0E1117 (Dark)
- **Secondary Background**: #262730 (Dark Gray)
- **Text**: #FAFAFA (Light)
- **Accent**: #FF6B6B (Red)

### Typography
- **Font**: Sans Serif
- **Sizes**: 
  - Headers: 2-2.5rem
  - Body: 1rem
  - Captions: 0.9rem

### Components
- **Buttons**: Gradient background, hover effects
- **Cards**: Border-left accent, box shadows
- **Charts**: Plotly dark template
- **Tables**: Background gradients for emphasis

---

## 🚀 Expansion Points

### Dễ Mở Rộng
1. **Thêm trang mới**: Tạo file trong `pages/`, import trong `app.py`
2. **Thêm UI component**: Add function vào `utils/ui_components.py`
3. **Thêm backend module**: Tạo trong `backend/`, implement logic
4. **Thêm LLM provider**: Extend `llm_integration/`

### Scalability
- Multi-page architecture sẵn sàng
- Backend modules độc lập
- Session state centralized
- CSS modular và có thể override

---

## 📝 Conventions

### Naming
- **Files**: snake_case (e.g., `model_training.py`)
- **Functions**: snake_case (e.g., `render_header()`)
- **Classes**: PascalCase (e.g., `ModelTrainer`)
- **Constants**: UPPER_CASE (e.g., `MAX_FEATURES`)

### Imports
```python
# Standard library
import os
import sys

# Third party
import streamlit as st
import pandas as pd
import numpy as np

# Local
from utils.ui_components import render_header
from backend.models.trainer import ModelTrainer
```

### Comments
- Docstrings cho functions
- TODO comments cho backend placeholders
- Inline comments cho logic phức tạp

---

**Version**: 1.0.0  
**Last Updated**: 2025-01-11

