"""
Dashboard - Home Page
"""

import streamlit as st
from utils.ui_components import render_info_card
from utils.session_state import init_session_state, get_session_info

def render():
    """Render dashboard page"""
    init_session_state()
    
    # Welcome section
    st.markdown("## Chào mừng đến với Hệ thống Chấm điểm Tín dụng")
    
    st.markdown("""
    Nền tảng toàn diện để phân tích, đánh giá và dự đoán khả năng tín dụng của khách hàng 
    sử dụng các thuật toán Machine Learning tiên tiến và AI có thể giải thích.
    """)
    
    st.markdown("---")
    
    # Features overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Các Tính Năng Chính")
        st.markdown("*Nhấn vào từng tính năng để chuyển đến trang tương ứng*")
        
        # Features with navigation mapping and descriptions
        features = [
            {
                "icon": "📤",
                "title": "Tải Dữ Liệu & Phân Tích",
                "desc": "Nhập và khám phá dữ liệu của bạn. Xem thống kê mô tả, phân phối, giá trị thiếu và tương quan.",
                "nav_key": "📊 Data Upload & Analysis",
                "btn_text": "Đi đến Tải Dữ Liệu & Phân Tích"
            },
            {
                "icon": "⚙️",
                "title": "Xử Lý Đặc Trưng", 
                "desc": "Xử lý giá trị thiếu, ngoại lai, mã hóa và chuẩn hóa. Chia thành tập Train/Valid/Test.",
                "nav_key": "⚙️ Feature Engineering",
                "btn_text": "Đi đến Xử Lý Đặc Trưng"
            },
            {
                "icon": "🧠",
                "title": "Huấn Luyện Mô Hình",
                "desc": "Huấn luyện các thuật toán ML: Logistic Regression, XGBoost, LightGBM, CatBoost.",
                "nav_key": "🧠 Model Training",
                "btn_text": "Đi đến Huấn Luyện Mô Hình"
            },
            {
                "icon": "💡",
                "title": "Giải Thích Mô Hình",
                "desc": "Giải thích mô hình với giá trị SHAP, độ quan trọng đặc trưng và biểu đồ lực.",
                "nav_key": "💡 Model Explanation",
                "btn_text": "Đi đến Giải Thích Mô Hình"
            },
            {
                "icon": "🎯",
                "title": "Dự Đoán & Tư Vấn",
                "desc": "Dự đoán điểm tín dụng cho khách hàng mới và nhận báo cáo tư vấn được tạo bởi AI.",
                "nav_key": "🎯 Prediction & Advisory",
                "btn_text": "Đi đến Dự Đoán & Tư Vấn"
            },
        ]
        
        for feature in features:
            # Create a card-like container for each feature
            with st.container():
                st.markdown(f"""
                <div style="background-color: #262730; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 3px solid #667eea;">
                    <h4 style="margin: 0; color: #667eea;">{feature['icon']} {feature['title']}</h4>
                    <p style="margin: 0.5rem 0; color: #aaa; font-size: 0.9rem;">{feature['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(feature['btn_text'], key=f"nav_{feature['title']}", width='stretch'):
                    st.session_state.nav_page = feature['nav_key']
                    st.rerun()
    
    with col2:
        st.markdown("### Trạng Thái Hiện Tại")
        
        session_info = get_session_info()
        
        # Data status
        if session_info['has_data']:
            st.success("● Đã tải dữ liệu thành công")
        else:
            st.warning("○ Chưa tải dữ liệu - Vui lòng tải từ 'Tải Dữ Liệu & Phân Tích'")
        
        # Processed data status
        if session_info['has_processed_data']:
            st.success("● Đã xử lý dữ liệu")
        else:
            st.info("○ Dữ liệu chưa được xử lý")
        
        # Model status
        if session_info['has_model']:
            st.success(f"● Đã huấn luyện mô hình ({st.session_state.model_type})")
        else:
            st.info("○ Chưa huấn luyện mô hình")
        
        # Features status
        if session_info['num_features'] > 0:
            st.success(f"● Đã chọn {session_info['num_features']} đặc trưng")
        else:
            st.info("○ Chưa chọn đặc trưng")

