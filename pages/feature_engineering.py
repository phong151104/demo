"""
Trang Xử Lý & Chọn Biến - Feature Engineering & Selection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.ui_components import show_processing_placeholder
from utils.session_state import init_session_state

def render():
    """Render trang xử lý và chọn biến"""
    init_session_state()
    
    st.markdown("## ⚙️ Xử Lý & Chọn Biến")
    st.markdown("Tiền xử lý dữ liệu và lựa chọn các đặc trưng quan trọng cho mô hình.")
    
    # Check if data exists
    if st.session_state.data is None:
        st.warning("⚠️ Chưa có dữ liệu. Vui lòng upload dữ liệu từ trang 'Upload & EDA' trước.")
        return
    
    data = st.session_state.data
    st.success(f"✅ Đang làm việc với dataset: {len(data)} dòng, {len(data.columns)} cột")
    
    st.markdown("---")
    
    # Tabs for different processing steps
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔧 Tiền Xử Lý",
        "📊 Binning",
        "⭐ Feature Importance",
        "✅ Chọn Biến"
    ])
    
    # Tab 1: Preprocessing
    with tab1:
        st.markdown("### 🔧 Các Bước Tiền Xử Lý")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 1️⃣ Xử Lý Giá Trị Thiếu")
            
            missing_data = data.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                st.warning(f"⚠️ Có {len(missing_data)} cột chứa giá trị thiếu")
                
                # Display missing data
                missing_df = pd.DataFrame({
                    'Cột': missing_data.index,
                    'Số lượng thiếu': missing_data.values,
                    'Tỷ lệ (%)': (missing_data.values / len(data) * 100).round(2)
                })
                st.dataframe(missing_df, use_container_width=True)
                
                # Missing handling options
                st.markdown("**Phương pháp xử lý:**")
                missing_method = st.radio(
                    "Chọn phương pháp:",
                    ["Mean/Median/Mode Imputation", "Drop Rows", "Drop Columns", "Forward/Backward Fill"],
                    key="missing_method"
                )
                
                if st.button("🔄 Áp Dụng Xử Lý Thiếu", key="apply_missing"):
                    with st.spinner("Đang xử lý..."):
                        show_processing_placeholder(f"Xử lý giá trị thiếu bằng {missing_method}")
                        st.success("✅ Đã xử lý giá trị thiếu!")
            else:
                st.success("✅ Không có giá trị thiếu trong dataset")
        
        with col2:
            st.markdown("#### 2️⃣ Mã Hóa Biến Phân Loại")
            
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if categorical_cols:
                st.info(f"📝 Có {len(categorical_cols)} biến phân loại cần mã hóa")
                
                # Display categorical columns
                for col in categorical_cols[:5]:  # Show first 5
                    unique_vals = data[col].nunique()
                    st.text(f"• {col}: {unique_vals} giá trị khác nhau")
                
                if len(categorical_cols) > 5:
                    st.text(f"... và {len(categorical_cols) - 5} cột khác")
                
                # Encoding options
                st.markdown("**Phương pháp mã hóa:**")
                encoding_method = st.selectbox(
                    "Chọn phương pháp:",
                    ["One-Hot Encoding", "Label Encoding", "Target Encoding", "Ordinal Encoding"],
                    key="encoding_method"
                )
                
                if st.button("🔄 Áp Dụng Mã Hóa", key="apply_encoding"):
                    with st.spinner("Đang mã hóa..."):
                        show_processing_placeholder(f"Mã hóa biến phân loại bằng {encoding_method}")
                        st.success("✅ Đã mã hóa biến phân loại!")
            else:
                st.success("✅ Không có biến phân loại cần mã hóa")
        
        st.markdown("---")
        
        # Additional preprocessing steps
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 3️⃣ Chuẩn Hóa/Scale")
            scaling_method = st.selectbox(
                "Phương pháp:",
                ["Standard Scaler", "Min-Max Scaler", "Robust Scaler", "No Scaling"],
                key="scaling_method"
            )
            
            if st.button("🔄 Áp Dụng Scaling", key="apply_scaling"):
                show_processing_placeholder(f"Scaling với {scaling_method}")
                st.success("✅ Đã scaling!")
        
        with col2:
            st.markdown("#### 4️⃣ Xử Lý Outliers")
            outlier_method = st.selectbox(
                "Phương pháp:",
                ["IQR Method", "Z-Score", "Winsorization", "Keep All"],
                key="outlier_method"
            )
            
            if st.button("🔄 Xử Lý Outliers", key="apply_outliers"):
                show_processing_placeholder(f"Xử lý outliers bằng {outlier_method}")
                st.success("✅ Đã xử lý outliers!")
        
        with col3:
            st.markdown("#### 5️⃣ Cân Bằng Dữ Liệu")
            balance_method = st.selectbox(
                "Phương pháp:",
                ["SMOTE", "Random Over-sampling", "Random Under-sampling", "No Balancing"],
                key="balance_method"
            )
            
            if st.button("🔄 Cân Bằng Dữ Liệu", key="apply_balance"):
                show_processing_placeholder(f"Cân bằng dữ liệu bằng {balance_method}")
                st.success("✅ Đã cân bằng dữ liệu!")
    
    # Tab 2: Binning
    with tab2:
        st.markdown("### 📊 Phân Nhóm (Binning) Biến Liên Tục")
        
        st.markdown("""
        <div style="background-color: #262730; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <p style="margin: 0;">💡 <strong>Binning</strong> giúp chuyển biến liên tục thành các nhóm rời rạc, 
            hữu ích cho việc phân tích và giải thích mô hình.</p>
        </div>
        """, unsafe_allow_html=True)
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                selected_col = st.selectbox("Chọn biến để binning:", numeric_cols, key="binning_col")
                
                binning_method = st.radio(
                    "Phương pháp binning:",
                    ["Equal Width", "Equal Frequency", "Custom"],
                    key="binning_method"
                )
                
                num_bins = st.slider("Số nhóm:", 2, 10, 5, key="num_bins")
                
                if st.button("🔄 Thực Hiện Binning", key="do_binning", type="primary"):
                    show_processing_placeholder(f"Binning biến {selected_col} thành {num_bins} nhóm")
                    st.success(f"✅ Đã tạo biến mới: {selected_col}_binned")
            
            with col2:
                # Visualize binning
                st.markdown("#### 📊 Trực Quan Hóa Binning")
                
                # Create sample bins for visualization
                col_data = data[selected_col].dropna()
                
                # Mock binning visualization
                fig = go.Figure()
                
                # Histogram
                fig.add_trace(go.Histogram(
                    x=col_data,
                    nbinsx=num_bins,
                    name='Distribution',
                    marker_color='#667eea',
                    opacity=0.7
                ))
                
                # Add bin edges as vertical lines (mock)
                bin_edges = np.linspace(col_data.min(), col_data.max(), num_bins + 1)
                for edge in bin_edges:
                    fig.add_vline(x=edge, line_dash="dash", line_color="red", opacity=0.5)
                
                fig.update_layout(
                    title=f"Binning visualization - {selected_col}",
                    xaxis_title=selected_col,
                    yaxis_title="Frequency",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Bin statistics
                st.markdown("#### 📊 Thống Kê Từng Nhóm")
                bin_stats = pd.DataFrame({
                    'Nhóm': [f"Bin {i+1}" for i in range(num_bins)],
                    'Khoảng': [f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})" for i in range(num_bins)],
                    'Số mẫu': np.random.randint(50, 200, num_bins),  # Mock data
                })
                st.dataframe(bin_stats, use_container_width=True)
        else:
            st.warning("⚠️ Không có biến số nào trong dataset")
    
    # Tab 3: Feature Importance
    with tab3:
        st.markdown("### ⭐ Mức Độ Quan Trọng Của Đặc Trưng")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("#### ⚙️ Cấu Hình")
            
            importance_method = st.selectbox(
                "Phương pháp tính:",
                ["Random Forest", "LightGBM", "XGBoost", "Logistic Regression (Coef)"],
                key="importance_method"
            )
            
            top_n = st.slider("Top N features:", 5, 30, 15, key="top_n_features")
            
            if st.button("🔄 Tính Feature Importance", key="calc_importance", type="primary"):
                with st.spinner("Đang tính toán..."):
                    show_processing_placeholder(f"Tính feature importance bằng {importance_method}")
                    st.success("✅ Đã tính xong!")
        
        with col2:
            st.markdown("#### 📊 Biểu Đồ Feature Importance")
            
            # Mock feature importance data
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                sample_features = numeric_cols[:min(top_n, len(numeric_cols))]
                importance_scores = np.random.random(len(sample_features))
                importance_scores = importance_scores / importance_scores.sum()  # Normalize
                
                # Sort by importance
                sorted_indices = np.argsort(importance_scores)[::-1]
                sorted_features = [sample_features[i] for i in sorted_indices]
                sorted_scores = importance_scores[sorted_indices]
                
                # Create bar chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=sorted_scores,
                    y=sorted_features,
                    orientation='h',
                    marker=dict(
                        color=sorted_scores,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Importance")
                    ),
                    text=[f"{score:.3f}" for score in sorted_scores],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title=f"Top {len(sorted_features)} Important Features",
                    xaxis_title="Importance Score",
                    yaxis_title="Features",
                    template="plotly_dark",
                    height=max(400, len(sorted_features) * 25),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 **Lưu ý**: Đây là dữ liệu mô phỏng. Backend sẽ tính toán importance thực tế từ mô hình.")
            else:
                st.warning("⚠️ Không có biến số để tính feature importance")
    
    # Tab 4: Feature Selection
    with tab4:
        st.markdown("### ✅ Chọn Đặc Trưng Cho Mô Hình")
        
        st.markdown("""
        <div style="background-color: #262730; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
            <p style="margin: 0;">📋 <strong>Chọn các đặc trưng</strong> bạn muốn sử dụng để huấn luyện mô hình. 
            Có thể dựa trên feature importance hoặc kiến thức nghiệp vụ.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get all columns except target
        all_cols = data.columns.tolist()
        
        # Assume last column is target (or let user select)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            target_col = st.selectbox(
                "Chọn biến mục tiêu (Target):",
                all_cols,
                index=len(all_cols) - 1 if len(all_cols) > 0 else 0,
                key="target_col"
            )
        
        with col2:
            st.metric("Số biến có sẵn", len(all_cols) - 1)
        
        # Available features (exclude target)
        available_features = [col for col in all_cols if col != target_col]
        
        # Feature selection
        st.markdown("#### 🎯 Chọn Đặc Trưng")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selection_mode = st.radio(
                "Chế độ chọn:",
                ["Chọn thủ công", "Chọn tự động (theo threshold)"],
                key="selection_mode"
            )
            
            if selection_mode == "Chọn tự động (theo threshold)":
                importance_threshold = st.slider(
                    "Ngưỡng importance:",
                    0.0, 1.0, 0.01, 0.01,
                    key="importance_threshold"
                )
                
                if st.button("🔄 Chọn Tự Động", key="auto_select"):
                    # Mock auto selection
                    num_selected = np.random.randint(5, min(15, len(available_features)))
                    selected = np.random.choice(available_features, num_selected, replace=False).tolist()
                    st.session_state.selected_features = selected
                    st.success(f"✅ Đã chọn tự động {len(selected)} đặc trưng!")
        
        with col2:
            # Manual selection
            if selection_mode == "Chọn thủ công":
                selected_features = st.multiselect(
                    "Chọn các đặc trưng:",
                    available_features,
                    default=st.session_state.selected_features if st.session_state.selected_features else available_features[:min(10, len(available_features))],
                    key="manual_features"
                )
                
                if st.button("💾 Lưu Lựa Chọn", key="save_selection", type="primary"):
                    st.session_state.selected_features = selected_features
                    st.success(f"✅ Đã lưu {len(selected_features)} đặc trưng!")
            else:
                # Display auto-selected features
                if st.session_state.selected_features:
                    st.multiselect(
                        "Đặc trưng đã chọn:",
                        available_features,
                        default=st.session_state.selected_features,
                        disabled=True,
                        key="auto_features_display"
                    )
        
        st.markdown("---")
        
        # Summary
        if st.session_state.selected_features:
            st.success(f"✅ **Đã chọn {len(st.session_state.selected_features)} đặc trưng cho mô hình**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                numeric_selected = len([f for f in st.session_state.selected_features 
                                       if f in data.select_dtypes(include=[np.number]).columns])
                st.metric("Biến số", numeric_selected)
            
            with col2:
                categorical_selected = len([f for f in st.session_state.selected_features 
                                           if f in data.select_dtypes(include=['object', 'category']).columns])
                st.metric("Biến phân loại", categorical_selected)
            
            with col3:
                st.metric("Tổng biến", len(st.session_state.selected_features))
            
            # Display selected features
            with st.expander("📋 Xem Danh Sách Đặc Trưng Đã Chọn"):
                for i, feat in enumerate(st.session_state.selected_features, 1):
                    st.text(f"{i}. {feat}")
        else:
            st.warning("⚠️ Chưa chọn đặc trưng nào. Vui lòng chọn ít nhất một đặc trưng.")

