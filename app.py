"""
Credit Scoring System - Main Application
Advanced Risk Assessment & Prediction Platform
"""

import streamlit as st
from pathlib import Path
import sys

# Enable debug logging
sys.stdout.flush()

# Cấu hình trang
st.set_page_config(
    page_title="Credit Scoring System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import custom CSS
from utils.ui_components import load_custom_css, render_header
import sys

# Enable logging

# Add anchor at top of page for scroll reset
st.markdown('<div id="top"></div>', unsafe_allow_html=True)

# CSS to help with scroll behavior
st.markdown("""
<style>
    /* Prevent scroll jump issues */
    .main .block-container {
        padding-top: 1rem;
    }
    /* Smooth scroll behavior */
    html {
        scroll-behavior: smooth;
    }
    /* Auto scroll to top on page load */
    .stApp {
        scroll-margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)

# Load CSS tùy chỉnh
try:
    load_custom_css()
except Exception as e:
    print(f"✗ CSS error: {e}", file=sys.stderr)

# Render header
try:
    render_header()
except Exception as e:
    print(f"✗ Header error: {e}", file=sys.stderr)
    st.markdown("# CREDIT SCORING SYSTEM")
    st.markdown("### Advanced Risk Assessment & Prediction Platform")
    st.markdown("---")

# Sidebar navigation
with st.sidebar:
    st.markdown("""
    <div style='text-align: left; padding: 1rem 0;'>
        <h2 style='margin: 0; color: #667eea; font-weight: 600;'>
            <span style='font-size: 1.8rem;'>🏦</span> Credit Scoring
        </h2>
        <p style='margin: 0.3rem 0 0 0; color: #aaa; font-size: 0.85rem;'>Risk Assessment Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Menu điều hướng
    st.markdown("### NAVIGATION")
    
    # Navigation options
    nav_options = ["🏠 Dashboard", "📊 Data Upload & Analysis", "⚙️ Feature Engineering", 
                   "🧠 Model Training", "💡 Model Explanation", "🎯 Prediction & Advisory"]
    
    # Get default index from session state if set
    default_index = 0
    if 'nav_page' in st.session_state and st.session_state.nav_page in nav_options:
        default_index = nav_options.index(st.session_state.nav_page)
        # Clear nav_page after using it
        del st.session_state.nav_page
    
    # Navigation with radio buttons
    page = st.radio(
        "Chọn chức năng:",
        nav_options,
        index=default_index,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Thông tin phiên làm việc
    with st.expander("▼ Session Status"):
        if 'data' in st.session_state and st.session_state.data is not None:
            st.success(f"● Đã tải dữ liệu: {len(st.session_state.data)} dòng")
        else:
            st.info("○ Chưa tải dữ liệu")
        
        if 'model' in st.session_state and st.session_state.model is not None:
            st.success("● Đã huấn luyện mô hình")
        else:
            st.info("○ Chưa huấn luyện mô hình")
        
        # Show configurations count
        total_configs = (
            len(st.session_state.get('missing_config', {})) +
            len(st.session_state.get('encoding_config', {})) +
            len(st.session_state.get('binning_config', {}))
        )
        if total_configs > 0:
            st.success(f"● {total_configs} cấu hình đã lưu")
            
            # Show breakdown
            if len(st.session_state.get('missing_config', {})) > 0:
                st.caption(f"  - Missing: {len(st.session_state.missing_config)}")
            if len(st.session_state.get('encoding_config', {})) > 0:
                st.caption(f"  - Encoding: {len(st.session_state.encoding_config)}")
            if len(st.session_state.get('binning_config', {})) > 0:
                st.caption(f"  - Binning: {len(st.session_state.binning_config)}")
        
        if 'processed_data' in st.session_state and st.session_state.processed_data is not None:
            st.success(f"● Đã xử lý: {len(st.session_state.processed_data)} dòng")
    
    st.markdown("---")
    st.caption("© 2025 Credit Scoring System v1.0")

# Định tuyến trang với logging
# Track page change to handle scroll
if 'current_page' not in st.session_state:
    st.session_state.current_page = page
    
page_changed = st.session_state.current_page != page
if page_changed:
    st.session_state.current_page = page
    # Add JavaScript to scroll to top when page changes
    st.markdown("""
    <script>
        window.parent.document.querySelector('section.main').scrollTo(0, 0);
    </script>
    """, unsafe_allow_html=True)

try:
    if page == "🏠 Dashboard":
        from views import home
        home.render()
    elif page == "📊 Data Upload & Analysis":
        from views import upload_eda
        upload_eda.render()
    elif page == "⚙️ Feature Engineering":
        from views import feature_engineering
        feature_engineering.render()
    elif page == "🧠 Model Training":
        from views import model_training
        model_training.render()
    elif page == "💡 Model Explanation":
        from views import shap_explanation
        shap_explanation.render()
    elif page == "🎯 Prediction & Advisory":
        from views import prediction
        prediction.render()
except Exception as e:
    import traceback
    traceback.print_exc(file=sys.stderr)
    st.error(f"Error loading page: {e}")
    st.error("Check terminal for details")

