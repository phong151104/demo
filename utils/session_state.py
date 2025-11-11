"""
Session State Management - Quản lý trạng thái phiên làm việc
"""

import streamlit as st

def init_session_state():
    """Khởi tạo session state"""
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = []
    
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    if 'model_type' not in st.session_state:
        st.session_state.model_type = None
    
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = {}
    
    if 'explainer' not in st.session_state:
        st.session_state.explainer = None
    
    if 'shap_values' not in st.session_state:
        st.session_state.shap_values = None

def clear_session_state():
    """Xóa toàn bộ session state"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def get_session_info():
    """Lấy thông tin về session hiện tại"""
    info = {
        "has_data": st.session_state.data is not None,
        "has_processed_data": st.session_state.processed_data is not None,
        "has_model": st.session_state.model is not None,
        "num_features": len(st.session_state.selected_features) if st.session_state.selected_features else 0,
    }
    return info

