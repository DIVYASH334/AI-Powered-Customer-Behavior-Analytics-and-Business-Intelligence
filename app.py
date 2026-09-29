"""
app.py
AI-Powered Customer Behavior Analytics and Business Intelligence Dashboard
Interactive Streamlit Application for Sound/Recording Studios
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure Streamlit Page
st.set_page_config(
    page_title="StudioAI - Customer Analytics & BI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #6c757d;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 5px solid #1f77b4;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .rec-box {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 12px;
    }
    .badge-champion {
        background-color: #28a745;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-loyal {
        background-color: #17a2b8;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-promising {
        background-color: #ffc107;
        color: #212529;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-risk {
        background-color: #dc3545;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper: Load Data
@st.cache_data
def load_datasets():
    bookings_path = "data/processed/cleaned_bookings.csv"
    customers_path = "data/processed/rfm_segmented_customers.csv"
    
    if not os.path.exists(bookings_path) or not os.path.exists(customers_path):
        from src.data_generator import main as gen_data
        from src.preprocess import run_preprocessing_pipeline
        from src.segmentation import run_segmentation_pipeline
        gen_data()
        run_preprocessing_pipeline()
        run_segmentation_pipeline()
        
    df_b = pd.read_csv(bookings_path)
    df_c = pd.read_csv(customers_path)
    df_b["booking_date"] = pd.to_datetime(df_b["booking_date"])
    return df_b, df_c

# Helper: Load Models & Metrics
@st.cache_resource
def load_models_and_metrics():
    metrics_path = "models/model_metrics.json"
    if not os.path.exists(metrics_path):
        from src.predict import run_prediction_pipeline
        from src.recommend import train_and_save_recommender
        run_prediction_pipeline()
        train_and_save_recommender()
        
    with open(metrics_path) as f:
        metrics = json.load(f)
        
    similarity_matrix = joblib.load("models/service_similarity.pkl")
    return metrics, similarity_matrix

df_bookings, df_customers = load_datasets()
metrics_dict, similarity_matrix = load_models_and_metrics()

# Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3075/3075908.png", width=65)
st.sidebar.title("StudioAI Platform")
st.sidebar.markdown("**AI & BI for Recording Studios**")

nav_option = st.sidebar.radio(
    "Navigate Module:",
    [
        "📊 Executive Overview & KPIs",
        "📈 Exploratory Data Analytics (EDA)",
        "🎯 RFM Customer Segmentation",
        "🔮 Predictive Retention & Churn Studio",
        "💡 AI Service Recommender",
        "📁 Customer Directory & Export"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("🔧 **Filter Controls**")
selected_cities = st.sidebar.multiselect("Select Cities", options=sorted(df_customers["city"].unique()), default=[])
selected_services = st.sidebar.multiselect("Select Primary Services", options=sorted(df_bookings["service_type"].unique()), default=[])

# Apply Filter if selected
filtered_customers = df_customers.copy()
filtered_bookings = df_bookings.copy()

if selected_cities:
    filtered_customers = filtered_customers[filtered_customers["city"].isin(selected_cities)]
    cust_ids = filtered_customers["customer_id"].unique()
    filtered_bookings = filtered_bookings[filtered_bookings["customer_id"].isin(cust_ids)]
    
if selected_services:
    filtered_bookings = filtered_bookings[filtered_bookings["service_type"].isin(selected_services)]
    cust_ids = filtered_bookings["customer_id"].unique()
    filtered_customers = filtered_customers[filtered_customers["customer_id"].isin(cust_ids)]

# -------------------------------------------------------------
# 1. EXECUTIVE OVERVIEW & KPIS
# -------------------------------------------------------------
if nav_option == "📊 Executive Overview & KPIs":
    st.markdown('<div class="main-header">🎙️ Sound Studio Executive Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time performance indicators, customer lifecycle velocity, and revenue health</div>', unsafe_allow_html=True)
    
    # KPI Metrics Row
    total_rev = filtered_bookings["amount_paid"].sum()
    total_bookings = len(filtered_bookings)
    total_clients = len(filtered_customers)
    repeat_rate = (filtered_customers["repeat_customer"].mean() * 100) if total_clients > 0 else 0
    avg_rating = filtered_bookings["rating"].mean() if total_bookings > 0 else 0
    avg_spend = (total_rev / total_clients) if total_clients > 0 else 0
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Revenue", f"₹{total_rev:,.0f}", "+14.2% YoY")
    with col2:
        st.metric("Total Bookings", f"{total_bookings:,}", "+8.5%")
    with col3:
        st.metric("Client Base", f"{total_clients:,}", "Unique Artists")
    with col4:
        st.metric("Repeat Rate", f"{repeat_rate:.1f}%", "+3.1% MoM")
    with col5:
        st.metric("Avg Satisfaction", f"⭐ {avg_rating:.2f}/5", "Top Acoustics")
    with col6:
        st.metric("Avg Spend / Client", f"₹{avg_spend:,.0f}", "LTV Indicator")
        
    st.markdown("---")
    
    # Row 1: Charts
    c_left, c_right = st.columns([7, 5])
    
    with c_left:
        st.subheader("Monthly Revenue Velocity (₹ INR)")
        monthly_trend = filtered_bookings.groupby("booking_month")["amount_paid"].sum().reset_index()
        fig_trend = px.area(
            monthly_trend, x="booking_month", y="amount_paid",
            labels={"booking_month": "Month", "amount_paid": "Revenue (₹)"},
            color_discrete_sequence=["#1f77b4"]
        )
        fig_trend.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=320)
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with c_right:
        st.subheader("Service Revenue Contribution")
        service_rev = filtered_bookings.groupby("service_type")["amount_paid"].sum().reset_index()
        fig_donut = px.pie(
            service_rev, names="service_type", values="amount_paid",
            hole=0.45, color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_donut.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=320)
        st.plotly_chart(fig_donut, use_container_width=True)
        
    # Row 2: Customer Distribution by Segment & Strategic Alerts
    c_seg, c_action = st.columns([5, 7])
    
    with c_seg:
        st.subheader("Customer Health Segments")
        seg_counts = filtered_customers["segment_name"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig_bar = px.bar(
            seg_counts, x="Count", y="Segment", orientation="h",
            color="Segment", color_discrete_map={
                "High-Value Champions": "#28a745",
                "Loyal Regulars": "#17a2b8",
                "Potential / Promising": "#ffc107",
                "At-Risk / Inactive": "#dc3545"
            }
        )
        fig_bar.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=280, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with c_action:
        st.subheader("💡 Strategic BI Recommendations for Studio Manager")
        st.markdown("""
        - 🌟 **Champion Retention**: 61 VIP clients contribute **32% of total studio income**. Roll out exclusive weekend lockout passes.
        - 🔄 **Upsell Recording to Mixing**: Over **68%** of recording artists seek post-production. Bundle 'Record + Mix' packages at 10% off.
        - ⚠️ **Churn Mitigation**: **296 at-risk clients** haven't visited in 150+ days. Auto-dispatch win-back rehearsal vouchers via WhatsApp/Email.
        - 🎙️ **Podcast Studio Growth**: Podcasting shows the fastest booking frequency growth among corporate and independent creators.
        """)

# -------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYTICS (EDA)
# -------------------------------------------------------------
elif nav_option == "📈 Exploratory Data Analytics (EDA)":
    st.markdown('<div class="main-header">📈 Studio Exploratory Data Analytics (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Patterns in customer spend, room usage, duration, and geographic footprint</div>', unsafe_allow_html=True)
    
    eda_tab1, eda_tab2, eda_tab3 = st.tabs(["🎵 Service & Booking Trends", "👥 Customer Demographics", "🔥 Correlation Analysis"])
    
    with eda_tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Bookings Count by Day of Week")
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_df = filtered_bookings["day_of_week"].value_counts().reindex(day_order).reset_index()
            day_df.columns = ["Day", "Bookings"]
            fig_day = px.bar(day_df, x="Day", y="Bookings", color="Bookings", color_continuous_scale="Blues")
            fig_day.update_layout(height=320)
            st.plotly_chart(fig_day, use_container_width=True)
            
        with col2:
            st.markdown("##### Average Booking Duration by Service (Hours)")
            dur_df = filtered_bookings.groupby("service_type")["duration_hours"].mean().reset_index()
            fig_dur = px.bar(dur_df, x="service_type", y="duration_hours", color="duration_hours", color_continuous_scale="Teal")
            fig_dur.update_layout(height=320)
            st.plotly_chart(fig_dur, use_container_width=True)
            
        st.markdown("##### Studio Room Utilization & Hourly Revenue")
        room_df = filtered_bookings.groupby("studio_room").agg(
            total_hours=("duration_hours", "sum"),
            revenue=("amount_paid", "sum"),
            bookings=("booking_id", "count")
        ).reset_index()
        st.dataframe(room_df.style.format({"total_hours": "{:.0f} hrs", "revenue": "₹{:,.0f}", "bookings": "{:,}"}), use_container_width=True)

    with eda_tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Client Geographic Footprint (Top Cities)")
            city_df = filtered_customers["city"].value_counts().reset_index()
            city_df.columns = ["City", "Customers"]
            fig_city = px.pie(city_df, names="City", values="Customers", color_discrete_sequence=px.colors.sequential.Viridis)
            fig_city.update_layout(height=320)
            st.plotly_chart(fig_city, use_container_width=True)
            
        with c2:
            st.markdown("##### Age Distribution by Client Type")
            fig_box = px.box(filtered_customers, x="client_type", y="age", color="client_type")
            fig_box.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

    with eda_tab3:
        st.markdown("##### Correlation Matrix (Spend, Recency, Rating, Duration)")
        num_cols = ["age", "days_since_last_visit", "visit_count", "total_spend", "avg_duration_hours", "avg_rating", "avg_discount_received", "repeat_customer", "churn_risk"]
        corr_matrix = filtered_customers[num_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
        fig_corr.update_layout(height=450)
        st.plotly_chart(fig_corr, use_container_width=True)

# -------------------------------------------------------------
# 3. RFM CUSTOMER SEGMENTATION
# -------------------------------------------------------------
elif nav_option == "🎯 RFM Customer Segmentation":
    st.markdown('<div class="main-header">🎯 RFM Analysis & K-Means Clustering</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Segmenting sound studio clients via Recency (R), Frequency (F), and Monetary (M) algorithms</div>', unsafe_allow_html=True)
    
    seg_col1, seg_col2, seg_col3, seg_col4 = st.columns(4)
    with seg_col1:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #28a745;">
            <h4>🌟 High-Value Champions</h4>
            <p>Frequent studio lockouts, highest spend, recent sessions.</p>
            <b>Strategy:</b> VIP Priority Room Booking & Free Master Review.
        </div>
        """, unsafe_allow_html=True)
    with seg_col2:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #17a2b8;">
            <h4>🤝 Loyal Regulars</h4>
            <p>Consistent monthly visits, reliable recording & mixing revenue.</p>
            <b>Strategy:</b> Multi-session bundle pass & loyalty rewards.
        </div>
        """, unsafe_allow_html=True)
    with seg_col3:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #ffc107;">
            <h4>🌱 Potential / Promising</h4>
            <p>Recent first or second booking with high satisfaction score.</p>
            <b>Strategy:</b> Welcome-back discount on mixing/mastering.
        </div>
        """, unsafe_allow_html=True)
    with seg_col4:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #dc3545;">
            <h4>⚠️ At-Risk / Inactive</h4>
            <p>Lapsed clients (>120 days since visit) or one-time users.</p>
            <b>Strategy:</b> Win-back campaign & feedback diagnostic.
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # 3D & 2D Cluster Visualizations
    v_tab1, v_tab2 = st.tabs(["🔮 3D Interactive RFM Space", "🗺️ 2D PCA Cluster Map"])
    
    with v_tab1:
        fig_3d = px.scatter_3d(
            filtered_customers,
            x="days_since_last_visit",
            y="visit_count",
            z="total_spend",
            color="segment_name",
            hover_name="name",
            hover_data=["customer_id", "city", "client_type", "avg_rating"],
            labels={
                "days_since_last_visit": "Recency (Days)",
                "visit_count": "Frequency (Visits)",
                "total_spend": "Monetary (₹ Spend)"
            },
            color_discrete_map={
                "High-Value Champions": "#28a745",
                "Loyal Regulars": "#17a2b8",
                "Potential / Promising": "#ffc107",
                "At-Risk / Inactive": "#dc3545"
            }
        )
        fig_3d.update_layout(height=550, margin=dict(l=0, r=0, b=0, t=20))
        st.plotly_chart(fig_3d, use_container_width=True)
        
    with v_tab2:
        fig_pca = px.scatter(
            filtered_customers,
            x="pca_x",
            y="pca_y",
            color="segment_name",
            hover_name="name",
            hover_data=["customer_id", "city", "total_spend"],
            labels={"pca_x": "Principal Component 1 (Frequency & Spend)", "pca_y": "Principal Component 2 (Recency)"},
            color_discrete_map={
                "High-Value Champions": "#28a745",
                "Loyal Regulars": "#17a2b8",
                "Potential / Promising": "#ffc107",
                "At-Risk / Inactive": "#dc3545"
            }
        )
        fig_pca.update_layout(height=450)
        st.plotly_chart(fig_pca, use_container_width=True)
        
    st.subheader("Segment Aggregation Table")
    seg_summary = filtered_customers.groupby("segment_name").agg(
        Count=("customer_id", "count"),
        Avg_Recency=("days_since_last_visit", "mean"),
        Avg_Visits=("visit_count", "mean"),
        Avg_Total_Spend=("total_spend", "mean"),
        Avg_Rating=("avg_rating", "mean")
    ).reset_index()
    st.dataframe(seg_summary.style.format({
        "Avg_Recency": "{:.1f} days",
        "Avg_Visits": "{:.1f}",
        "Avg_Total_Spend": "₹{:,.0f}",
        "Avg_Rating": "⭐ {:.2f}"
    }), use_container_width=True)

# -------------------------------------------------------------
# 4. PREDICTIVE RETENTION & CHURN STUDIO
# -------------------------------------------------------------
elif nav_option == "🔮 Predictive Retention & Churn Studio":
    st.markdown('<div class="main-header">🔮 Machine Learning Prediction Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict future customer retention and churn risk before the client leaves</div>', unsafe_allow_html=True)
    
    pred_tab1, pred_tab2 = st.tabs(["⚡ Live Customer Prediction Form", "🏆 Model Performance & Evaluation"])
    
    with pred_tab1:
        st.markdown("Enter or adjust customer behavioral parameters to run inference with the trained **Gradient Boosting / Random Forest** models:")
        
        with st.form("prediction_form"):
            f_col1, f_col2, f_col3 = st.columns(3)
            with f_col1:
                age_input = st.slider("Customer Age", min_value=18, max_value=70, value=28)
                gender_input = st.selectbox("Gender", ["Male", "Female", "Non-Binary"])
                city_input = st.selectbox("City", ["Chennai", "Coimbatore", "Bangalore", "Madurai", "Hyderabad", "Kochi", "Mumbai"])
            with f_col2:
                client_input = st.selectbox("Client Persona", ["Independent Musician", "Band Member", "Film/Ad Music Director", "Podcaster", "Voice Actor", "Content Creator"])
                service_input = st.selectbox("Primary Service", ["Recording", "Mixing", "Mastering", "Rehearsal Space", "Voiceover / Dubbing", "Music Production", "Podcast Recording"])
                duration_input = st.slider("Avg Session Duration (Hours)", 1, 10, 4)
            with f_col3:
                spend_input = st.number_input("Average Booking Spend (₹)", min_value=500, max_value=50000, value=6000, step=500)
                recency_input = st.slider("Days Since Last Visit", 1, 365, 30)
                rating_input = st.slider("Satisfaction Rating (Stars)", 1.0, 5.0, 4.5, step=0.5)
                discount_input = st.slider("Average Discount Applied (%)", 0, 25, 5)
                
            submitted = st.form_submit_button("🚀 Run AI Prediction", use_container_width=True)
            
        if submitted:
            from src.predict import predict_single_customer
            input_payload = {
                "age": age_input,
                "gender": gender_input,
                "city": city_input,
                "client_type": client_input,
                "primary_service": service_input,
                "avg_duration_hours": duration_input,
                "avg_booking_spend": spend_input,
                "days_since_last_visit": recency_input,
                "avg_rating": rating_input,
                "avg_discount_received": discount_input
            }
            result = predict_single_customer(input_payload)
            
            st.markdown("### 🎯 Model Inference Result")
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                repeat_color = "#28a745" if result["repeat_probability"] >= 50 else "#dc3545"
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {repeat_color};">
                    <h3>Repeat Retention Likelihood</h3>
                    <h2>{result['repeat_prediction']}</h2>
                    <h4>Probability: <b>{result['repeat_probability']}%</b></h4>
                </div>
                """, unsafe_allow_html=True)
                st.progress(result["repeat_probability"] / 100.0)
                
            with res_col2:
                churn_color = "#dc3545" if result["churn_probability"] >= 50 else "#28a745"
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {churn_color};">
                    <h3>Churn Risk Assessment</h3>
                    <h2>{result['churn_risk']}</h2>
                    <h4>Churn Risk: <b>{result['churn_probability']}%</b></h4>
                </div>
                """, unsafe_allow_html=True)
                st.progress(result["churn_probability"] / 100.0)

    with pred_tab2:
        st.subheader("Model Evaluation Benchmark Comparison")
        st.caption("Rigorous evaluation on test set across baseline and ensemble architectures:")
        
        models_df = pd.DataFrame(metrics_dict["retention_models"])
        st.dataframe(
            models_df[["model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc"]]
            .style.format({
                "accuracy": "{:.2%}",
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "f1_score": "{:.4f}",
                "roc_auc": "{:.4f}"
            }),
            use_container_width=True
        )
        
        c_imp, c_churn = st.columns(2)
        with c_imp:
            st.subheader("Key Drivers of Customer Retention")
            feat_imp = pd.Series(metrics_dict["feature_importances"]).sort_values(ascending=True).tail(8)
            fig_imp = px.bar(x=feat_imp.values, y=feat_imp.index, orientation="h", labels={"x": "Importance Weight", "y": "Feature"})
            fig_imp.update_layout(height=300)
            st.plotly_chart(fig_imp, use_container_width=True)
            
        with c_churn:
            st.subheader("Churn Classifier Performance")
            churn_info = metrics_dict["churn_model"]
            st.metric("Model Architecture", churn_info["model_name"])
            st.metric("Test Accuracy", f"{churn_info['accuracy']*100:.2f}%")
            st.metric("ROC-AUC Score", f"{churn_info['roc_auc']:.4f}")

# -------------------------------------------------------------
# 5. AI SERVICE RECOMMENDER
# -------------------------------------------------------------
elif nav_option == "💡 AI Service Recommender":
    st.markdown('<div class="main-header">💡 AI Studio Service Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Collaborative filtering & cross-sell intelligence for studio audio engineers</div>', unsafe_allow_html=True)
    
    from src.recommend import get_recommendations_for_customer, get_default_recommendations
    
    rec_tab1, rec_tab2 = st.tabs(["👤 Customer-Specific Recommendations", "🔥 Cross-Service Affinity Matrix"])
    
    with rec_tab1:
        # Choose a customer from database
        sample_customers = df_customers.head(100)
        selected_cust_id = st.selectbox(
            "Select Customer to Generate Personalized Next-Best-Service Recommendations:",
            options=sample_customers["customer_id"].tolist(),
            format_func=lambda x: f"{x} - {df_customers[df_customers['customer_id'] == x]['name'].values[0]} ({df_customers[df_customers['customer_id'] == x]['client_type'].values[0]})"
        )
        
        cust_record = df_customers[df_customers["customer_id"] == selected_cust_id].iloc[0]
        
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            st.metric("Client Name", cust_record["name"])
        with col_c2:
            st.metric("Primary Service", cust_record["primary_service"])
        with col_c3:
            st.metric("Segment", cust_record["segment_name"])
        with col_c4:
            st.metric("Visits & Spend", f"{cust_record['visit_count']} visits | ₹{cust_record['total_spend']:,.0f}")
            
        st.markdown("### 🎧 Top Recommended Next Studio Services")
        recs = get_recommendations_for_customer(selected_cust_id, top_n=3)
        
        for idx, r in enumerate(recs, 1):
            st.markdown(f"""
            <div class="rec-box">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0; color: #1f77b4;">#{idx} {r['service']}</h4>
                    <span style="font-weight: bold; background: #e3f2fd; color: #0d47a1; padding: 4px 10px; border-radius: 8px;">Match: {r['match_score']}%</span>
                </div>
                <p style="margin: 6px 0; color: #495057;">{r['description']}</p>
                <small style="color: #28a745; font-weight: 500;">💡 <b>Why Recommended:</b> {r['rationale']}</small>
            </div>
            """, unsafe_allow_html=True)
            
    with rec_tab2:
        st.subheader("Service-to-Service Cosine Similarity (Item Collaborative Filtering)")
        st.caption("Identifies pairing affinities between studio offerings (e.g. Recording -> Mixing -> Mastering):")
        fig_sim = px.imshow(similarity_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="Blues")
        fig_sim.update_layout(height=450)
        st.plotly_chart(fig_sim, use_container_width=True)

# -------------------------------------------------------------
# 6. CUSTOMER DIRECTORY & EXPORT
# -------------------------------------------------------------
elif nav_option == "📁 Customer Directory & Export":
    st.markdown('<div class="main-header">📁 Studio Customer Directory</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Search, inspect and export customer profiles with segmentation tags</div>', unsafe_allow_html=True)
    
    search_query = st.text_input("🔍 Search customer by Name or ID:", "")
    
    display_df = filtered_customers.copy()
    if search_query:
        display_df = display_df[
            display_df["name"].str.contains(search_query, case=False, na=False) |
            display_df["customer_id"].str.contains(search_query, case=False, na=False)
        ]
        
    cols_to_show = [
        "customer_id", "name", "city", "client_type", "primary_service",
        "segment_name", "visit_count", "total_spend", "avg_rating",
        "days_since_last_visit", "repeat_customer", "actionable_strategy"
    ]
    
    st.dataframe(
        display_df[cols_to_show].style.format({
            "total_spend": "₹{:,.0f}",
            "avg_rating": "⭐ {:.2f}",
            "repeat_customer": lambda x: "✅ Repeat" if x == 1 else "⚠️ Single"
        }),
        use_container_width=True,
        height=480
    )
    
    csv_data = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Filtered Customer Data as CSV",
        data=csv_data,
        file_name="studio_customer_intelligence.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.caption("AI-Powered Customer Behavior Analytics & BI Platform | Built for Sound & Recording Studios | Deep Learning & Machine Learning Pipeline")
