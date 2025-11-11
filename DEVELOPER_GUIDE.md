# 🔧 Developer Guide - Credit Scoring System

## 📋 Tổng Quan

Document này hướng dẫn phát triển backend cho hệ thống Credit Scoring đã có giao diện hoàn chỉnh.

## 🎯 Mục Tiêu Phát Triển

### Phase 1: Core Backend (Ưu tiên cao)
1. Data Processing Pipeline
2. Model Training & Evaluation
3. Basic SHAP Integration

### Phase 2: Advanced Features
1. Full SHAP Explainability
2. LLM Integration
3. Model Persistence & Versioning

### Phase 3: Production Ready
1. API Development
2. Database Integration
3. Monitoring & Logging
4. Security & Authentication

---

## 📦 Module 1: Data Processing

### File: `backend/data_processing/preprocessing.py`

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    """Class xử lý tiền xử lý dữ liệu"""
    
    def __init__(self):
        self.imputers = {}
        self.encoders = {}
        self.scalers = {}
    
    def handle_missing_values(self, df, strategy='mean'):
        """Xử lý giá trị thiếu"""
        # TODO: Implement
        pass
    
    def encode_categorical(self, df, method='onehot'):
        """Mã hóa biến phân loại"""
        # TODO: Implement
        pass
    
    def scale_features(self, df, method='standard'):
        """Scale các biến số"""
        # TODO: Implement
        pass
```

**Kết nối với UI**: 
- `pages/feature_engineering.py` Tab 1 (Tiền Xử Lý)
- Replace các `show_processing_placeholder()` bằng logic thực

---

## 🤖 Module 2: Model Training

### File: `backend/models/trainer.py`

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

class ModelTrainer:
    """Class huấn luyện mô hình"""
    
    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def prepare_data(self, X, y, test_size=0.2):
        """Chia train/test"""
        # TODO: Implement
        pass
    
    def train(self, params=None):
        """Huấn luyện mô hình"""
        # TODO: Implement
        pass
    
    def evaluate(self):
        """Đánh giá mô hình"""
        # TODO: Implement
        pass
```

**Kết nối với UI**: 
- `pages/model_training.py` Tab 1 (Cấu Hình)
- Button "Huấn Luyện Mô Hình" gọi `trainer.train()`

---

## 📊 Module 3: Model Evaluation

### File: `backend/models/evaluator.py`

```python
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    roc_curve,
    roc_auc_score
)
import numpy as np

class ModelEvaluator:
    """Class đánh giá mô hình"""
    
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
        self.y_pred_proba = None
    
    def get_confusion_matrix(self):
        """Tạo confusion matrix"""
        # TODO: Implement
        pass
    
    def get_roc_curve(self):
        """Tạo ROC curve data"""
        # TODO: Implement
        # Return: fpr, tpr, auc_score
        pass
    
    def get_classification_report(self):
        """Tạo classification report"""
        # TODO: Implement
        pass
```

**Kết nối với UI**: 
- `pages/model_training.py` Tab 2 (Kết Quả)
- Replace mock data bằng kết quả thực

---

## 🔍 Module 4: SHAP Explainability

### File: `backend/explainability/shap_explainer.py`

```python
import shap
import numpy as np

class SHAPExplainer:
    """Class giải thích mô hình bằng SHAP"""
    
    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train
        self.explainer = None
        self.shap_values = None
    
    def initialize_explainer(self):
        """Khởi tạo SHAP explainer"""
        # TreeExplainer for tree models
        # LinearExplainer for linear models
        # TODO: Implement
        pass
    
    def compute_shap_values(self, X):
        """Tính SHAP values"""
        # TODO: Implement
        pass
    
    def get_global_importance(self):
        """Feature importance toàn cục"""
        # TODO: Implement
        pass
    
    def get_local_explanation(self, sample_idx):
        """Giải thích cho một mẫu"""
        # TODO: Implement
        pass
```

**Kết nối với UI**: 
- `pages/shap_explanation.py` - Tất cả tabs
- Button "Khởi Tạo SHAP Explainer" gọi `initialize_explainer()`

---

## 🤖 Module 5: LLM Integration

### File: `backend/llm_integration/llm_client.py`

```python
from openai import OpenAI
import os

class LLMAnalyzer:
    """Class tích hợp LLM cho phân tích tự động"""
    
    def __init__(self, provider='openai'):
        self.provider = provider
        if provider == 'openai':
            self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def analyze_data_quality(self, data_summary):
        """Phân tích chất lượng dữ liệu"""
        prompt = f"""
        Phân tích dataset sau và đưa ra nhận xét:
        {data_summary}
        
        Hãy đánh giá:
        1. Chất lượng dữ liệu
        2. Phân phối các biến
        3. Vấn đề cần xử lý
        4. Khuyến nghị
        """
        # TODO: Implement
        pass
    
    def interpret_shap(self, shap_data):
        """Diễn giải SHAP values"""
        # TODO: Implement
        pass
    
    def generate_recommendations(self, customer_data, prediction):
        """Tạo gợi ý cải thiện"""
        # TODO: Implement
        pass
```

**Kết nối với UI**: 
- `pages/upload_eda.py` Tab 4 (AI Analysis)
- `pages/shap_explanation.py` Tab 3 (AI Interpretation)
- `pages/prediction.py` Tab 3 (Gợi Ý)

---

## 🔗 Integration Workflow

### 1. Upload & EDA Page

```python
# In pages/upload_eda.py
from backend.data_processing.preprocessing import DataPreprocessor
from backend.llm_integration.llm_client import LLMAnalyzer

# Replace placeholder
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.session_state.data = data
    
    # Generate AI analysis
    llm = LLMAnalyzer()
    analysis = llm.analyze_data_quality(data.describe().to_dict())
    st.markdown(analysis)
```

### 2. Feature Engineering Page

```python
# In pages/feature_engineering.py
from backend.data_processing.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()

if st.button("Áp Dụng Xử Lý Thiếu"):
    processed_data = preprocessor.handle_missing_values(
        st.session_state.data, 
        strategy=missing_method
    )
    st.session_state.processed_data = processed_data
```

### 3. Model Training Page

```python
# In pages/model_training.py
from backend.models.trainer import ModelTrainer
from backend.models.evaluator import ModelEvaluator

if st.button("Huấn Luyện Mô Hình"):
    trainer = ModelTrainer(model_type=model_type)
    trainer.prepare_data(X, y, test_size=test_size/100)
    trainer.train(params)
    
    st.session_state.model = trainer.model
    
    # Evaluate
    evaluator = ModelEvaluator(trainer.model, trainer.X_test, trainer.y_test)
    metrics = evaluator.get_classification_report()
    st.session_state.model_metrics = metrics
```

### 4. SHAP Page

```python
# In pages/shap_explanation.py
from backend.explainability.shap_explainer import SHAPExplainer

if st.button("Khởi Tạo SHAP Explainer"):
    explainer = SHAPExplainer(
        st.session_state.model,
        st.session_state.X_train
    )
    explainer.initialize_explainer()
    explainer.compute_shap_values(st.session_state.X_test)
    
    st.session_state.explainer = explainer
    st.session_state.shap_values = explainer.shap_values
```

---

## 📝 Testing Strategy

### Unit Tests
```python
# tests/test_preprocessing.py
import pytest
from backend.data_processing.preprocessing import DataPreprocessor

def test_handle_missing_values():
    preprocessor = DataPreprocessor()
    # Test logic
    pass
```

### Integration Tests
```python
# tests/test_integration.py
def test_full_pipeline():
    # Load data -> Process -> Train -> Evaluate -> Explain
    pass
```

---

## 🚀 Deployment Checklist

- [ ] Environment variables setup (.env)
- [ ] Dependencies installed (requirements.txt)
- [ ] All backend modules implemented
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] UI connected to backend
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Documentation updated

---

## 📚 Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **SHAP Docs**: https://shap.readthedocs.io
- **Scikit-learn**: https://scikit-learn.org
- **XGBoost**: https://xgboost.readthedocs.io
- **LightGBM**: https://lightgbm.readthedocs.io
- **OpenAI API**: https://platform.openai.com/docs

---

## 💡 Tips

1. **Start Small**: Implement một module trước, test kỹ, rồi chuyển sang module khác
2. **Use Session State**: Streamlit session state để lưu model và data giữa các trang
3. **Error Handling**: Wrap logic trong try-except và hiển thị lỗi thân thiện
4. **Progress Bars**: Dùng `st.progress()` cho các tác vụ lâu
5. **Caching**: Dùng `@st.cache_data` và `@st.cache_resource` cho hiệu năng

---

**Happy Coding! 🚀**

