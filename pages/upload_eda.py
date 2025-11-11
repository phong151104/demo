"""
Trang Upload & EDA - Upload dữ liệu và phân tích khám phá
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.ui_components import show_llm_analysis, show_processing_placeholder
from utils.session_state import init_session_state

def render():
    """Render trang Upload & EDA"""
    print("DEBUG: Starting upload_eda.render()")
    try:
        init_session_state()
        print("DEBUG: Session state initialized")
    except Exception as e:
        st.error(f"Error initializing session: {e}")
        print(f"ERROR: Session init failed: {e}")
        return
    
    st.markdown("## 📤 Upload Dữ Liệu & Phân Tích Khám Phá (EDA)")
    st.markdown("Tải lên file CSV chứa dữ liệu khách hàng và khám phá các thông tin quan trọng.")
    
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Chọn file CSV dữ liệu",
        type=['csv'],
        help="Upload file CSV chứa dữ liệu khách hàng với các đặc trưng và nhãn"
    )
    
    if uploaded_file is not None:
        try:
            # Load data with error handling
            data = pd.read_csv(uploaded_file, on_bad_lines='skip', encoding='utf-8')
            
            # Validate data
            if data.empty:
                st.error("❌ File is empty or invalid format")
                return
            
            if len(data) < 5:
                st.warning(f"⚠️ Dataset only has {len(data)} rows. Upload more data for better analysis.")
            
            st.session_state.data = data
            st.success(f"✅ Data loaded successfully! ({len(data)} rows, {len(data.columns)} columns)")
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs([
                "📋 Dữ Liệu Mẫu", 
                "📊 Thống Kê Mô Tả", 
                "📈 Phân Phối Dữ Liệu",
                "🤖 Phân Tích AI"
            ])
            
            # Tab 1: Sample Data
            with tab1:
                st.markdown("### 📋 Dữ Liệu Mẫu")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    num_rows = st.slider("Số dòng hiển thị:", 5, 100, 10)
                with col2:
                    view_mode = st.selectbox("Chế độ xem:", ["Đầu", "Cuối", "Ngẫu nhiên"])
                
                if view_mode == "Đầu":
                    st.dataframe(data.head(num_rows), use_container_width=True, height=400)
                elif view_mode == "Cuối":
                    st.dataframe(data.tail(num_rows), use_container_width=True, height=400)
                else:
                    st.dataframe(data.sample(min(num_rows, len(data))), use_container_width=True, height=400)
                
                # Data info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Tổng số dòng", f"{len(data):,}")
                with col2:
                    st.metric("📋 Tổng số cột", len(data.columns))
                with col3:
                    missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100)
                    st.metric("❓ Dữ liệu thiếu", f"{missing_pct:.1f}%")
                with col4:
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    st.metric("🔢 Cột số", len(numeric_cols))
            
            # Tab 2: Descriptive Statistics
            with tab2:
                st.markdown("### 📊 Thống Kê Mô Tả")
                
                # Numeric columns stats
                numeric_data = data.select_dtypes(include=[np.number])
                if not numeric_data.empty:
                    st.markdown("#### 🔢 Biến Số")
                    
                    stats_df = numeric_data.describe().T
                    stats_df['missing'] = data[numeric_data.columns].isnull().sum()
                    stats_df['missing_pct'] = (stats_df['missing'] / len(data) * 100).round(2)
                    
                    # Highlight styling
                    st.dataframe(
                        stats_df.style.background_gradient(cmap='viridis', subset=['mean', 'std']),
                        use_container_width=True
                    )
                    
                    # Download stats
                    csv = stats_df.to_csv(index=True).encode('utf-8')
                    st.download_button(
                        "📥 Tải Thống Kê (CSV)",
                        csv,
                        "statistics.csv",
                        "text/csv",
                        key='download-stats'
                    )
                
                # Categorical columns
                categorical_data = data.select_dtypes(include=['object', 'category'])
                if not categorical_data.empty:
                    st.markdown("#### 📝 Biến Phân Loại")
                    
                    cat_info = []
                    for col in categorical_data.columns:
                        cat_info.append({
                            'Tên cột': col,
                            'Số giá trị khác nhau': data[col].nunique(),
                            'Giá trị phổ biến nhất': data[col].mode()[0] if not data[col].mode().empty else 'N/A',
                            'Tần suất cao nhất': data[col].value_counts().iloc[0] if len(data[col].value_counts()) > 0 else 0,
                            'Thiếu': data[col].isnull().sum(),
                            'Tỷ lệ thiếu (%)': f"{data[col].isnull().sum() / len(data) * 100:.2f}"
                        })
                    
                    cat_df = pd.DataFrame(cat_info)
                    st.dataframe(cat_df, use_container_width=True)
            
            # Tab 3: Data Distribution
            with tab3:
                st.markdown("### 📈 Phân Phối Dữ Liệu")
                
                viz_type = st.radio(
                    "Chọn loại biểu đồ:",
                    ["Histogram", "Box Plot", "Correlation Heatmap"],
                    horizontal=True
                )
                
                if viz_type == "Histogram":
                    st.markdown("#### 📊 Histogram - Phân Phối Biến Số")
                    
                    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        selected_col = st.selectbox("Chọn biến để vẽ:", numeric_cols)
                        
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            bins = st.slider("Số bins:", 10, 100, 30)
                        with col2:
                            show_kde = st.checkbox("Hiện KDE", value=True)
                        
                        # Create histogram
                        fig = px.histogram(
                            data, 
                            x=selected_col,
                            nbins=bins,
                            title=f"Phân phối của {selected_col}",
                            marginal="box" if show_kde else None,
                            color_discrete_sequence=['#667eea']
                        )
                        
                        fig.update_layout(
                            template="plotly_dark",
                            height=500,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistics for selected column
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Mean", f"{data[selected_col].mean():.2f}")
                        with col2:
                            st.metric("Median", f"{data[selected_col].median():.2f}")
                        with col3:
                            st.metric("Std Dev", f"{data[selected_col].std():.2f}")
                        with col4:
                            st.metric("Min", f"{data[selected_col].min():.2f}")
                        with col5:
                            st.metric("Max", f"{data[selected_col].max():.2f}")
                
                elif viz_type == "Box Plot":
                    st.markdown("#### 📦 Box Plot - Phát Hiện Outliers")
                    
                    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        selected_cols = st.multiselect(
                            "Chọn các biến để so sánh:",
                            numeric_cols,
                            default=numeric_cols[:min(4, len(numeric_cols))]
                        )
                        
                        if selected_cols:
                            # Create box plot
                            fig = go.Figure()
                            
                            for col in selected_cols:
                                fig.add_trace(go.Box(
                                    y=data[col],
                                    name=col,
                                    boxmean='sd'
                                ))
                            
                            fig.update_layout(
                                title="Box Plot - Phân tích outliers",
                                template="plotly_dark",
                                height=500,
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Outlier detection info
                            st.info("💡 **Gợi ý**: Các điểm nằm ngoài 'râu' của box plot có thể là outliers cần xử lý.")
                
                else:  # Correlation Heatmap
                    st.markdown("#### 🔥 Ma Trận Tương Quan")
                    
                    numeric_data = data.select_dtypes(include=[np.number])
                    if not numeric_data.empty and len(numeric_data.columns) > 1:
                        corr_matrix = numeric_data.corr()
                        
                        # Create heatmap
                        fig = px.imshow(
                            corr_matrix,
                            text_auto='.2f',
                            aspect="auto",
                            color_continuous_scale='RdBu_r',
                            title="Ma trận tương quan giữa các biến",
                            zmin=-1,
                            zmax=1
                        )
                        
                        fig.update_layout(
                            template="plotly_dark",
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Find high correlations
                        st.markdown("#### 🔍 Các Cặp Biến Có Tương Quan Cao")
                        
                        threshold = st.slider("Ngưỡng tương quan:", 0.5, 0.95, 0.7, 0.05)
                        
                        high_corr = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                if abs(corr_matrix.iloc[i, j]) >= threshold:
                                    high_corr.append({
                                        'Biến 1': corr_matrix.columns[i],
                                        'Biến 2': corr_matrix.columns[j],
                                        'Tương quan': f"{corr_matrix.iloc[i, j]:.3f}"
                                    })
                        
                        if high_corr:
                            st.dataframe(pd.DataFrame(high_corr), use_container_width=True)
                        else:
                            st.info(f"Không tìm thấy cặp biến nào có tương quan >= {threshold}")
            
            # Tab 4: AI Analysis
            with tab4:
                st.markdown("### 🤖 Phân Tích Tự Động Bằng AI")
                
                st.markdown("""
                <div style="background-color: #262730; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;">
                    <h4 style="margin-top: 0; color: #667eea;">💡 Tính Năng AI Analysis</h4>
                    <p>Khu vực này sẽ hiển thị phân tích tự động từ LLM về:</p>
                    <ul>
                        <li>✨ Nhận xét về chất lượng dữ liệu</li>
                        <li>📊 Đánh giá phân phối các biến quan trọng</li>
                        <li>🔗 Phát hiện mối quan hệ giữa các biến</li>
                        <li>⚠️ Cảnh báo về outliers và dữ liệu bất thường</li>
                        <li>💡 Đề xuất các bước tiền xử lý</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if st.button("🔄 Tạo Phân Tích AI", use_container_width=True, type="primary"):
                    with st.spinner("🤖 AI đang phân tích dữ liệu..."):
                        # Placeholder response
                        placeholder_analysis = f"""
                        **📊 Tổng Quan Dữ Liệu:**
                        
                        Dataset có {len(data):,} mẫu với {len(data.columns)} đặc trưng. Dữ liệu có {data.isnull().sum().sum()} giá trị thiếu 
                        ({(data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100):.1f}% tổng số).
                        
                        **🔍 Phân Tích Chi Tiết:**
                        
                        - **Biến số**: Dataset có {len(data.select_dtypes(include=[np.number]).columns)} biến số. 
                          Phân phối của các biến cho thấy một số có độ lệch (skewness) cao, cần xem xét transform.
                        
                        - **Biến phân loại**: Có {len(data.select_dtypes(include=['object', 'category']).columns)} biến phân loại. 
                          Cần mã hóa (encoding) trước khi đưa vào mô hình.
                        
                        - **Outliers**: Một số biến có outliers đáng kể. Khuyến nghị sử dụng IQR method hoặc winsorization.
                        
                        **💡 Khuyến Nghị:**
                        
                        1. Xử lý giá trị thiếu bằng imputation hoặc loại bỏ
                        2. Chuẩn hóa/Scale các biến số trước khi training
                        3. Xem xét feature engineering để tạo biến mới
                        4. Kiểm tra imbalanced data nếu đây là bài toán classification
                        
                        ⚡ *Phân tích này là mô phỏng. Backend sẽ tích hợp LLM (OpenAI/LangChain) để phân tích thực tế.*
                        """
                        
                        show_llm_analysis(
                            "Phân tích dataset và đưa ra nhận xét",
                            placeholder_analysis
                        )
                
                st.markdown("---")
                show_processing_placeholder("Tích hợp LLM API (OpenAI GPT-4, Claude, hoặc local LLM)")
        
        except Exception as e:
            st.error(f"❌ Lỗi khi đọc file: {str(e)}")
    
    else:
        # Show sample format
        print("DEBUG: No file uploaded, showing sample format")
        st.info("📝 No file uploaded. Please select a CSV file.")
        
        with st.expander("📋 View Sample Format"):
            st.markdown("""
            CSV file should have the following format:
            
            | customer_id | age | income | credit_history | loan_amount | ... | default |
            |-------------|-----|--------|----------------|-------------|-----|---------|
            | 1001        | 35  | 50000  | good           | 10000       | ... | 0       |
            | 1002        | 42  | 75000  | excellent      | 15000       | ... | 0       |
            | 1003        | 28  | 30000  | poor           | 5000        | ... | 1       |
            
            - Last column is target: 0 = no default, 1 = default
            - Other columns are features
            """)
            
            # Simple basic sample only - no complex loading
            print("DEBUG: Creating basic sample data")
            sample_data = pd.DataFrame({
                'customer_id': range(1001, 1011),
                'age': np.random.randint(25, 65, 10),
                'income': np.random.randint(30000, 100000, 10),
                'credit_history': np.random.choice(['good', 'fair', 'poor'], 10),
                'loan_amount': np.random.randint(5000, 50000, 10),
                'default': np.random.choice([0, 1], 10, p=[0.8, 0.2])
            })
            
            print(f"DEBUG: Sample data created: {len(sample_data)} rows")
            
            csv = sample_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Sample Data",
                csv,
                "sample_credit_data.csv",
                "text/csv"
            )
            
            st.dataframe(sample_data, use_container_width=True)

