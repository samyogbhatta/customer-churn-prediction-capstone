import os
import sys
import platform
import collections

# Set thread limits for OpenBLAS, MKL, OMP to prevent hangs on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Mock platform.uname and platform.machine to bypass slow/hanging system queries on Windows
UName = collections.namedtuple('uname_result', ['system', 'node', 'release', 'version', 'machine', 'processor'])
platform.uname = lambda: UName('Windows', 'node', '10', '10.0', 'AMD64', 'Intel')
platform.machine = lambda: 'AMD64'

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import joblib
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
from streamlit_option_menu import option_menu

# Import custom explainability and plotting modules
from src.explainability import ChurnExplainer, plot_dependence, plot_local_shap, plot_mean_bar, plot_summary, plot_waterfall
from src.preprocessing import NUMERICAL_COLS, CATEGORICAL_COLS, BINARY_COLS
from src.components.navigation import render_navigation
from src.components.report_generator import generate_report_pdf
from src.utils.excel_export import export_excel
import src.components.style as style

# Page Config
st.set_page_config(
    page_title="Telecom Churn Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS Styling (Crimson, Dark Blue, and White over Deep Midnight Canvas)
st.markdown("""
<style>
    /* Main application background and default text */
    .stApp {
        background-color: #090d16 !important;
        color: #f1f5f9 !important;
    }
    
    /* Clean block structure overrides */
    div[data-testid="stVerticalBlock"] > div {
        background-color: transparent;
    }
    
   /* Premium KPI Summary & Data Presentation Cards */
    .metric-card {
        background-color: #0a192f !important;
        border: 1px solid #172a45 !important;
        border-left: 4px solid #dc143c !important; /* Crimson Left Highlight Border */
        border-radius: 8px;
        padding: 10px 14px; 
        
        /* Centering text inside the box */
        text-align: center; /* <-- ADDED: Centers all values and labels perfectly */
        
        /* Width and alignment spacing overrides */
        width: 100%;         /* <-- UPDATED: Controls width relative to column slot */
        max-width: 190px;   /* <-- UPDATED: Restricts maximum card stretching width */
        margin: 0;     /* Aligns cards nicely within their native layouts */
        
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.45);
        margin-bottom: 12px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        border-color: #dc143c !important;
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 1.6rem; 
        font-weight: 700;
        color: #ffffff !important;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.75rem; 
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 2px;
        display: block;    /* Ensures text layout alignment behaviors remain block-centered */
    }
    
    /* Form Dropdown & Element styling overrides */
    div[data-baseweb="select"], div[data-baseweb="input"] {
        background-color: #0f172a !important;
        border-radius: 6px;
    }
    
    /* Global Section Breaks */
    hr {
        border-color: #1e293b !important;
        margin: 2rem 0;
    }

    /* Customizing fallback text and elements */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions for cached loading
@st.cache_data
def load_dataset(file_path):
    if not os.path.exists(file_path):
        return None
    return pd.read_csv(file_path)

@st.cache_resource
def load_explainer():
    try:
        return ChurnExplainer()
    except Exception as e:
        st.error(f"Failed to load model and explainer: {e}")
        return None

@st.cache_data
def load_model_metrics(metrics_path="models/metrics.json"):
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path, "r") as f:
        return json.load(f)

def get_human_readable_reasons(contributions, top_n=3, mode="risk"):
    reasons = []
    if mode == "risk":
        filtered = contributions[contributions["SHAP_Value"] > 0].copy()
    else:
        filtered = contributions[contributions["SHAP_Value"] < 0].copy()
        
    for _, row in filtered.head(top_n).iterrows():
        feat = row["Feature"]
        val = row["Feature_Value"]
        desc = ""
        if feat == "num_complaints_30d":
            desc = f"High number of complaints in last 30 days ({int(val)} complaints)"
        elif feat == "usage_drop_pct":
            desc = f"Usage dropped significantly by {val*100:.1f}% last month" if val > 0 else f"Usage increased by {abs(val)*100:.1f}% last month"
        elif feat == "recharge_drop_pct":
            desc = f"Recharge frequency/amount dropped by {val*100:.1f}% last month" if val > 0 else f"Recharge frequency/amount increased by {abs(val)*100:.1f}% last month"
        elif feat == "inactive_days":
            desc = f"Customer was inactive for {int(val)} days in the last 30 days"
        elif feat == "call_drop_rate":
            desc = f"High call drop rate ({val*100:.2f}%)"
        elif feat == "signal_strength_dbm":
            desc = f"Weak network signal strength ({int(val)} dBm)"
        elif feat == "last_recharge_days_ago":
            desc = f"Long gap since last recharge ({int(val)} days ago)"
        elif feat == "avg_recharge_amount_npr":
            desc = f"Low average recharge amount (Rs. {val:.1f})" if mode == "risk" else f"Healthy average recharge amount (Rs. {val:.1f})"
        elif feat == "tenure_days":
            desc = f"Short subscription tenure ({int(val)} days)" if mode == "risk" else f"Long-term loyal subscription tenure ({int(val)} days)"
        elif feat == "avg_data_speed_mbps":
            desc = f"Slow average data speed ({val:.2f} Mbps)" if mode == "risk" else f"Fast average data speed ({val:.2f} Mbps)"
        elif feat == "data_gb_30d":
            desc = f"Low data usage last month ({val:.2f} GB)" if mode == "risk" else f"High data usage last month ({val:.2f} GB)"
        elif feat == "calls_min_30d":
            desc = f"Low call duration last month ({val:.1f} minutes)" if mode == "risk" else f"High call duration last month ({val:.1f} minutes)"
        elif feat == "recharge_count_30d":
            desc = f"Few recharges in last 30 days ({int(val)} times)" if mode == "risk" else f"Frequent recharges in last 30 days ({int(val)} times)"
        elif feat in ["data_pack_active", "voice_pack_active", "vas_active", "roaming_active"]:
            pack_name = feat.replace("_active", "").replace("_", " ").title()
            desc = f"Active {pack_name} service" if val == 1 else f"No active {pack_name} service"
        elif "_" in feat:
            parts = feat.split("_")
            group = parts[0].title()
            category = "_".join(parts[1:])
            if val == 1:
                desc = f"{group} is {category}"
                
        if not desc:
            clean_feat = feat.replace("_", " ").title()
            desc = f"{clean_feat} is {val} (pushing risk up)" if mode == "risk" else f"{clean_feat} is {val} (holding risk down)"
        reasons.append(desc)
    return reasons

@st.cache_data
def process_uploaded_data(uploaded_df_raw, _explainer_instance):
    X_input = uploaded_df_raw.drop(columns=["customer_id", "churn"], errors="ignore")
    X_processed = _explainer_instance.get_preprocessed_df(X_input)
    probs = _explainer_instance.model.predict_proba(X_processed)[:, 1]
    return probs

@st.cache_data
def compute_uploaded_global_shap(uploaded_df_raw, _explainer_instance):
    X_input = uploaded_df_raw.drop(columns=["customer_id", "churn"], errors="ignore")
    X_processed = _explainer_instance.get_preprocessed_df(X_input)
    sample_size = min(200, len(X_processed))
    X_sample = X_processed.sample(n=sample_size, random_state=42) if len(X_processed) > sample_size else X_processed
    shap_vals, importance_df = _explainer_instance.get_global_explanations(X_sample)
    return shap_vals, importance_df, X_sample, len(X_sample)

# Load models and explainer
explainer = load_explainer()
metrics_data = load_model_metrics()

if explainer is None or metrics_data is None:
    st.warning("ML Pipeline needs to be executed before running the dashboard.")
    st.info("Please run the data generator and training pipeline in your terminal first:")
    st.code("python src/data_generator.py\npython src/train.py", language="bash")
    st.stop()

REQUIRED_RAW_FEATURES = [
    "age", "gender", "province", "district_type", "sim_type", "tenure_days",
    "calls_min_30d", "sms_count_30d", "data_gb_30d", "night_usage_pct",
    "last_recharge_days_ago", "avg_recharge_amount_npr", "recharge_count_30d",
    "signal_strength_dbm", "call_drop_rate", "avg_data_speed_mbps",
    "num_complaints_30d", "avg_resolution_time_hours",
    "data_pack_active", "voice_pack_active", "vas_active", "roaming_active",
    "usage_drop_pct", "recharge_drop_pct", "inactive_days"
]

def process_and_store_uploaded_data(uploaded_df_raw, filename):
    missing_cols = [col for col in REQUIRED_RAW_FEATURES if col not in uploaded_df_raw.columns]
    if missing_cols:
        return missing_cols
        
    uploaded_df = uploaded_df_raw.copy()
    if "customer_id" not in uploaded_df.columns:
        uploaded_df["customer_id"] = [f"NP-CUST-{i+1:05d}" for i in range(len(uploaded_df))]
        
    X_input = uploaded_df[REQUIRED_RAW_FEATURES]
    X_processed = explainer.get_preprocessed_df(X_input)
    probs = explainer.model.predict_proba(X_processed)[:, 1]
    
    uploaded_df["churn_probability"] = probs
    uploaded_df["Risk Score (%)"] = (probs * 100).round(1)
    uploaded_df["Risk Level"] = np.where(probs >= 0.7, "🔴 Critical", np.where(probs >= 0.3, "⚠️ Elevated", "🟢 Low"))
    
    if "churn" not in uploaded_df.columns or uploaded_df["churn"].isna().all():
        uploaded_df["churn"] = (probs >= 0.5).astype(int)
    else:
        uploaded_df["churn"] = uploaded_df["churn"].fillna(0).astype(int)
        
    st.session_state.uploaded_df = uploaded_df
    st.session_state.uploaded_filename = filename
    return None

st.title("📊 Nepal Telecom Churn Dashboard")
st.markdown("Developed Automated Customer Churn Prediction System Using Machine Learning and Explainable AI (SHAP)")

if "uploaded_df" not in st.session_state:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("📤 Get Started: Upload Your Dataset")
    
    col_upload, col_template = st.columns([2, 1])
    
    with col_template:
        st.markdown("### 📋 Required Columns")
        st.code("""
- Demographic: age, gender, province, district_type
- SIM info: sim_type, tenure_days
- Usage info: calls_min_30d, sms_count_30d, data_gb_30d
- Network info: signal_strength_dbm, call_drop_rate, avg_data_speed_mbps
- Recharges: avg_recharge_amount_npr, recharge_count_30d, last_recharge_days_ago
- Active Services: data_pack_active, voice_pack_active, vas_active, roaming_active
- Customer Experience: num_complaints_30d, avg_resolution_time_hours
- Trends: usage_drop_pct, recharge_drop_pct, inactive_days
- (Optional): customer_id, churn
        """, language="text")
        
        DATA_PATH = "data/telecom_churn_nepal.csv"
        df_demo = load_dataset(DATA_PATH)
        if df_demo is not None:
            sample_df = df_demo.copy()
            if "churn" in sample_df.columns:
                sample_df = sample_df.drop(columns=["churn"])
            sample_csv_data = sample_df.head(10).to_csv(index=False)
            st.download_button(
                label="📥 Download CSV Template",
                data=sample_csv_data,
                file_name="nepal_telecom_churn_template.csv",
                mime="text/csv"
            )
            
            st.markdown("---")
            if st.button("🚀 Load Demo Dataset"):
                errors = process_and_store_uploaded_data(df_demo, "telecom_churn_nepal_demo.csv")
                if errors:
                    st.error(f"Demo dataset is missing columns: {errors}")
                else:
                    st.success("Demo data loaded!")
                    st.rerun()
        else:
            st.info("Run `python src/data_generator.py` to generate the demo dataset.")
            
    with col_upload:
        st.markdown("### 📤 Drag and Drop CSV")
        uploaded_file = st.file_uploader("Upload CSV file containing subscriber records", type=["csv"])
        if uploaded_file is not None:
            try:
                uploaded_df_raw = pd.read_csv(uploaded_file)
                errors = process_and_store_uploaded_data(uploaded_df_raw, uploaded_file.name)
                if errors:
                    st.error("Missing required columns:")
                    st.write(errors)
                else:
                    st.success(f"Processed {len(uploaded_df_raw)} records.")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to process CSV: {e}")
        if "uploaded_df" not in st.session_state:
            st.stop()

app_mode = option_menu(
    menu_title=None,
    options=["Overview", "Customer List", "Customer Details", "Simulator"],
    icons=["bar-chart-line-fill", "table", "search", "cpu"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {
            "padding": "12px 0px", 
            "background-color": "#0f172a", 
            "border-radius": "6px",
            "border": "1px solid #1e293b"
        },
        "icon": {
            "color": "#ffffff", 
            "font-size": "14px"
        }, 
        "nav-link": {
            "font-size": "14px", 
            "text-align": "center", 
            "margin": "0px 15px", 
            "color": "#94a3b8",             
            "font-weight": "500",
            "border-radius": "50px",        
            "--hover-color": "#1e293b"      
        },
        "nav-link-selected": {
            "background-color": "#f90d3d",
            "color": "#ffffff",             
            "font-weight": "600",
            "border-radius": "50px",        
            "box-shadow": "0px 0px 12px rgba(220, 20, 60, 0.4)" 
        }
    }
)

if "gender_f" not in st.session_state:
    st.session_state.gender_f = "All"
if "province_f" not in st.session_state:
    st.session_state.province_f = "All"
if "sim_f" not in st.session_state:
    st.session_state.sim_f = "All"

filtered_df = st.session_state.uploaded_df.copy()
if st.session_state.gender_f != "All":
    filtered_df = filtered_df[filtered_df["gender"] == st.session_state.gender_f]
if st.session_state.province_f != "All":
    filtered_df = filtered_df[filtered_df["province"] == st.session_state.province_f]
if st.session_state.sim_f != "All":
    filtered_df = filtered_df[filtered_df["sim_type"] == st.session_state.sim_f]

if app_mode != "Simulator":
    total_customers = len(filtered_df)
    overall_churn_rate = (filtered_df["churn"].mean() * 100) if total_customers > 0 else 0.0
    model_accuracy = metrics_data.get("accuracy", 0.85)

    high_risk_revenue = 0.0
    if total_customers > 0:
        high_risk_revenue = filtered_df.loc[filtered_df["churn_probability"] >= 0.5, "avg_recharge_amount_npr"].sum()

    with st.container():
        st.markdown('<div style="padding: 0 10%;">', unsafe_allow_html=True)
        
        kpi_cols = st.columns(4, gap="small")
        with kpi_cols[0]:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{total_customers:,}</div><div class="metric-label">Total Customers</div></div>', unsafe_allow_html=True)
        with kpi_cols[1]:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{overall_churn_rate:.1f}%</div><div class="metric-label">Churn Rate</div></div>', unsafe_allow_html=True)
        with kpi_cols[2]:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{model_accuracy*100:.1f}%</div><div class="metric-label">Model Accuracy</div></div>', unsafe_allow_html=True)
        with kpi_cols[3]:
            st.markdown(f'<div class="metric-card"><div class="metric-value">Rs. {high_risk_revenue:,.0f}</div><div class="metric-label">Revenue at Risk</div></div>', unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
    overall_summary = {
        "total_customers": total_customers,
        "overall_churn_rate": overall_churn_rate,
        "model_accuracy": model_accuracy,
        "high_risk_revenue": high_risk_revenue,
    }
    st.markdown("<hr>", unsafe_allow_html=True)

if app_mode == "Overview":
    st.subheader("Overview & Demographics")
    st.markdown("### 🎛️ Dashboard Controls")
    
    meta_col1, meta_col2 = st.columns([2, 1])
    with meta_col1:
        st.markdown(f"**Dataset Loaded:** `{st.session_state.uploaded_filename}`")
    with meta_col2:
        st.markdown(f"**Subscribers in Segment:** `{len(filtered_df):,}` / `{len(st.session_state.uploaded_df):,}` total")
        
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)
    with ctrl_col1:
        genders = ["All"] + list(st.session_state.uploaded_df["gender"].unique())
        st.selectbox("Gender", genders, key="gender_f")
    with ctrl_col2:
        provinces = ["All"] + list(st.session_state.uploaded_df["province"].unique())
        st.selectbox("Province", provinces, key="province_f")
    with ctrl_col3:
        sim_types = ["All"] + list(st.session_state.uploaded_df["sim_type"].unique())
        st.selectbox("SIM Type", sim_types, key="sim_f")
    with ctrl_col4:
        st.markdown("<div style='padding-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reset & Upload New", use_container_width=True):
            for key in ["uploaded_df", "uploaded_filename", "gender_f", "province_f", "sim_f"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
            
    st.markdown("---")
    st.subheader("📊 Generate Executive Reports")

    if "pdf_report_bytes" not in st.session_state:
        st.session_state.pdf_report_bytes = None
    if "excel_report_bytes" not in st.session_state:
        st.session_state.excel_report_bytes = None
    if "report_filter_hash" not in st.session_state:
        st.session_state.report_filter_hash = ""

    current_filter_hash = f"{len(filtered_df)}_{overall_churn_rate:.2f}_{high_risk_revenue:.2f}"
    if st.session_state.report_filter_hash != current_filter_hash:
        st.session_state.pdf_report_bytes = None
        st.session_state.excel_report_bytes = None
        st.session_state.report_filter_hash = current_filter_hash

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("⚙️ Generate Reports (PDF & Excel)", use_container_width=True):
            with st.spinner("Generating reports..."):
                excel_path = os.path.join("reports", "at_risk_customers.xlsx")
                os.makedirs("reports", exist_ok=True)
                at_risk_df = filtered_df[filtered_df["churn_probability"] >= 0.5].copy()
                export_excel(at_risk_df, excel_path)
                with open(excel_path, "rb") as f:
                    st.session_state.excel_report_bytes = f.read()
                
                pdf_path = generate_report_pdf(overall_summary, filtered_df, explainer)
                with open(pdf_path, "rb") as f:
                    st.session_state.pdf_report_bytes = f.read()
                st.success("Reports generated successfully!")

    if st.session_state.pdf_report_bytes is not None and st.session_state.excel_report_bytes is not None:
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(label="📥 Save PDF Executive Report", data=st.session_state.pdf_report_bytes, file_name="churn_report.pdf", mime="application/pdf", use_container_width=True)
        with col_dl2:
            st.download_button(label="📥 Save Excel At-Risk List", data=st.session_state.excel_report_bytes, file_name="at_risk_customers.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    
    if len(filtered_df) == 0:
        st.warning("No customers match the current filter selection.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            province_churn = filtered_df.groupby("province")["churn"].mean().reset_index()
            province_churn["churn_pct"] = province_churn["churn"] * 100
            province_churn = province_churn.sort_values(by="churn_pct", ascending=False)
            
            fig_prov = px.bar(
                province_churn,
                x="province",
                y="churn_pct",
                color="churn_pct",
                color_continuous_scale=[[0, '#003893'], [0.5, '#ffffff'], [1, '#dc143c']],
                title="Observed Churn Rate by Province (%)",
                labels={"province": "Province", "churn_pct": "Churn Rate (%)"}
            )
            fig_prov.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_prov, use_container_width=True)

        with col2:
            corr_cols = [
                "tenure_days", "calls_min_30d", "data_gb_30d", "avg_recharge_amount_npr", 
                "signal_strength_dbm", "call_drop_rate", "num_complaints_30d", 
                "usage_drop_pct", "recharge_drop_pct", "inactive_days", "churn"
            ]
            corr_cols = [c for c in corr_cols if c in filtered_df.columns]
            corr_matrix = filtered_df[corr_cols].corr()
            corr_matrix.columns = [c.replace("_", " ").title() for c in corr_matrix.columns]
            corr_matrix.index = [c.replace("_", " ").title() for c in corr_matrix.index]
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale=[[0, '#dc143c'], [0.5, '#ffffff'], [1, '#003893']],
                title="Correlation Matrix (Key Features & Churn)"
            )
            fig_corr.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=500)
            st.plotly_chart(fig_corr, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig_sig = px.histogram(
                filtered_df,
                x="signal_strength_dbm",
                color=filtered_df["churn"].map({0: "Loyal", 1: "Churned"}),
                barmode="overlay",
                title="Signal Strength (dBm) Distribution by Churn",
                color_discrete_map={"Loyal": "#003893", "Churned": "#dc143c"},
                opacity=0.7
            )
            fig_sig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_sig, use_container_width=True)
            
        with col4:
            has_complaints_df = filtered_df[filtered_df["num_complaints_30d"] > 0]
            if len(has_complaints_df) > 0:
                fig_comp = px.scatter(
                    has_complaints_df,
                    x="num_complaints_30d",
                    y="avg_resolution_time_hours",
                    color=has_complaints_df["churn"].map({0: "Loyal", 1: "Churned"}),
                    title="Complaints Density vs Resolution Time (Hours)",
                    labels={"num_complaints_30d": "Number of Complaints", "avg_resolution_time_hours": "Resolution Time (Hrs)"},
                    color_discrete_map={"Loyal": "#003893", "Churned": "#dc143c"},
                    opacity=0.6
                )
                fig_comp.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.info("No customer complaints registered in the selected segment to display.")

        st.subheader("Global Explanations (AI-Driven Feature Importance)")
        
        @st.cache_data
        def compute_global_importance_filtered(_explainer, df_sample):       
            X_sample_raw = df_sample.drop(columns=["churn", "customer_id", "churn_probability", "Risk Score (%)", "Risk Level"], errors="ignore")
            X_sample_processed = _explainer.get_preprocessed_df(X_sample_raw)
            
            # FIX: Clean the raw processed columns to Title Case BEFORE computing SHAP and plotting
            X_sample_processed.columns = [c.replace("_", " ").title() for c in X_sample_processed.columns]
            
            shap_vals, importance_df = _explainer.get_global_explanations(X_sample_processed)
            return shap_vals, importance_df, X_sample_processed

        sample_size = min(300, len(filtered_df))
        df_sample = filtered_df.sample(n=sample_size, random_state=42)
        
        with st.spinner("Calculating global feature contributions..."):
            shap_vals, importance_df, X_sample_processed = compute_global_importance_filtered(explainer, df_sample)
        
        importance_df["Clean_Feature"] = importance_df["Feature"]
        
        fig_glob = px.bar(
            importance_df.head(15),
            y="Clean_Feature",
            x="Mean_Abs_SHAP",
            orientation="h",    
            title="Top 15 Global Predictive Features (Mean Absolute SHAP)",
            color="Mean_Abs_SHAP",
            color_continuous_scale=[[0, '#003893'], [1, '#dc143c']]
        )
        fig_glob.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(autorange="reversed"), height=400)
        
    # --- FIX 2: SYNCHRONIZE MATPLOTLIB CONFIGS ---
        plt.style.use('dark_background')
        plt.rcParams['figure.facecolor'] = '#090d16'
        plt.rcParams['axes.facecolor'] = '#090d16'
        plt.rcParams['text.color'] = '#ffffff'
        plt.rcParams['axes.labelcolor'] = '#ffffff'
        plt.rcParams['xtick.color'] = '#ffffff'
        plt.rcParams['ytick.color'] = '#ffffff'
        
        # Downsized text allocations to prevent collision on window minimize actions
        plt.rcParams['font.size'] = 7.5
        plt.rcParams['axes.labelsize'] = 6.8     # <-- SHRINKS THE Baseline X-AXIS LABELS PERFECTLY
        plt.rcParams['xtick.labelsize'] = 8.0
        plt.rcParams['ytick.labelsize'] = 8.0
        plt.rcParams['figure.autolayout'] = True
        
        # Generate the figures
        fig_summary = plot_summary(shap_vals, X_sample_processed)
        fig_bar = plot_mean_bar(shap_vals, X_sample_processed)
        
        # --- FIX 3: FORCE EXACT DPI & INCH MATRIX SIZES ---
        # A 6x4 inch canvas map at 100 DPI outputs exactly 600x400 pixels, perfectly matching Plotly
        for fig in [fig_summary, fig_bar]:
            if fig is not None:
                fig.set_size_inches(6.0, 4.0)
                fig.set_dpi(100)
                for ax in fig.get_axes():
                    ax.set_facecolor('#090d16')
                    ax.tick_params(colors='#ffffff', which='both', labelsize=8.0)
                    plt.setp(ax.get_yticklabels(), color='#ffffff', fontsize=8.0)
                    plt.setp(ax.get_xticklabels(), color='#ffffff', fontsize=8.0)
                    
                    # FORCE EXPLICIT MICRO SIZING FOR X-AXIS SHAP DESCRIPTION LABELS
                    ax.xaxis.label.set_size(6.8)  # <-- Overrides native package string overrides
                    ax.yaxis.label.set_size(8.0)  # Keeps Y-axis description matching feature sizes
                    ax.xaxis.label.set_color('#ffffff')
                    ax.yaxis.label.set_color('#ffffff')
                
                if len(fig.get_axes()) >= 2:
                    cb_ax = fig.get_axes()[-1]
                    cb_ax.yaxis.label.set_size(7.0)
                    cb_ax.yaxis.label.set_color('#ffffff')
                    cb_ax.tick_params(labelcolor='#ffffff', colors='#ffffff', labelsize=7.5)
        
        fig_display = X_sample_processed.copy()
        
        if "Age" in fig_display.columns:
            fig_dependence = plot_dependence("Age", shap_vals, fig_display) 
        else:
            fallback_feat = X_sample_processed.columns[0]
            fig_dependence = plot_dependence(fallback_feat, shap_vals, fig_display)

        if fig_dependence is not None:
            fig_dependence.set_size_inches(7.5, 5.0) # Matches identical aspect footprint
            for ax in fig_dependence.get_axes():
                ax.set_facecolor('#090d16')
                ax.tick_params(colors='#ffffff', which='both', labelsize=8.5)
                plt.setp(ax.get_yticklabels(), color='#ffffff', fontsize=8.5)
                plt.setp(ax.get_xticklabels(), color='#ffffff', fontsize=8.5)
                ax.xaxis.label.set_color('#ffffff')
                ax.yaxis.label.set_color('#ffffff')
                
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # --- ROW 1 ---
        col_row1_left, col_row1_right = st.columns(2)
        with col_row1_left:
            st.subheader("Top 15 Global Features")
            # Adjusted height within the chart declaration to standard dimensions
            fig_glob.update_layout(height=450, margin=dict(l=20, r=20, t=30, b=40))
            st.plotly_chart(fig_glob, use_container_width=True)
            
        with col_row1_right:
            st.subheader("Feature Distributions (Beeswarm)")
            # bbox_inches='tight' clips out any default ghost white margins on render
            st.pyplot(fig_summary, bbox_inches='tight')
            
        # Clear spatial structural row buffer 
        st.markdown("<br><br>", unsafe_allow_html=True) 
        
        # --- ROW 2 ---
        col_row2_left, col_row2_right = st.columns(2)
        with col_row2_left:
            st.subheader("Mean SHAP Importance")
            st.pyplot(fig_bar, bbox_inches='tight')
        with col_row2_right:
            st.subheader("Feature Dependence (Age)")
            st.pyplot(fig_dependence, bbox_inches='tight')

        # Global column padding stabilization
        st.markdown("""
        <style>
            div[data-testid="column"] {
                padding-left: 20px !important;
                padding-right: 20px !important;
            }
        </style>
        """, unsafe_allow_html=True)

elif app_mode == "Customer List":
    
    st.subheader("📋 Customer List")
    if len(filtered_df) == 0:
        st.warning("No customers match the active filters.")
    else:
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            risk_filter = st.multiselect("Filter by Risk Level", options=["🔴 Critical", "⚠️ Elevated", "🟢 Low"], default=["🔴 Critical", "⚠️ Elevated", "🟢 Low"])
        with col_f2:
            search_id = st.text_input("Search by Customer ID", "")
        
        display_df = filtered_df.copy()
        display_df = display_df[display_df["Risk Level"].isin(risk_filter)]
        if search_id:
            display_df = display_df[display_df["customer_id"].astype(str).str.contains(search_id, case=False)]
        
        cols_to_show = ["customer_id", "Risk Level", "Risk Score (%)", "age", "gender", "province", "tenure_days", "avg_recharge_amount_npr", "num_complaints_30d", "inactive_days"]
        if "churn" in display_df.columns:
            cols_to_show.append("churn")
            
        st.dataframe(display_df[cols_to_show].sort_values(by="Risk Score (%)", ascending=False), use_container_width=True, hide_index=True)

elif app_mode == "Customer Details":
    st.subheader("Customer Details")
    if len(filtered_df) == 0:
        st.warning("No customers match the active filters. Reset filters on the sidebar to explore.")
    else:
        selected_cust_id = st.selectbox("Select Customer ID to Analyze", filtered_df["customer_id"].values)
        customer_row = filtered_df[filtered_df["customer_id"] == selected_cust_id]
        col_detail1, col_detail2 = st.columns([1, 2])
        
        with col_detail1:
            st.markdown("### 📋 Demographic Profile")
            st.markdown(f"**Customer ID:** `{selected_cust_id}`")
            st.markdown(f"**Age:** {customer_row['age'].values[0]}")
            st.markdown(f"**Gender:** {customer_row['gender'].values[0]}")
            st.markdown(f"**Province:** {customer_row['province'].values[0]}")
            st.markdown(f"**District Type:** {customer_row['district_type'].values[0]}")
            st.markdown(f"**SIM Type:** {customer_row['sim_type'].values[0]}")
            st.markdown(f"**Tenure:** {customer_row['tenure_days'].values[0]} days")
            
            st.markdown("### 📱 Active Packages")
            st.write("Data Pack: ", "✅ Yes" if customer_row["data_pack_active"].values[0] == 1 else "❌ No")
            st.write("Voice Pack: ", "✅ Yes" if customer_row["voice_pack_active"].values[0] == 1 else "❌ No")
            st.write("VAS (Value Added): ", "✅ Yes" if customer_row["vas_active"].values[0] == 1 else "❌ No")
            st.write("Roaming: ", "✅ Yes" if customer_row["roaming_active"].values[0] == 1 else "❌ No")

        with col_detail2:
            st.markdown("### 📊 Usage & Service Quality Metrics")
            metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
            with metric_col_1:
                st.metric("Calls Last 30 Days", f"{customer_row['calls_min_30d'].values[0]:.1f} Min")
                st.metric("Average Recharge (NPR)", f"Rs. {customer_row['avg_recharge_amount_npr'].values[0]}")
                st.metric("Call Drop Rate", f"{customer_row['call_drop_rate'].values[0]*100:.2f}%")
            with metric_col_2:
                st.metric("Data Usage 30 Days", f"{customer_row['data_gb_30d'].values[0]:.2f} GB")
                st.metric("Recharges 30 Days", f"{customer_row['recharge_count_30d'].values[0]} times")
                st.metric("Avg Data Speed", f"{customer_row['avg_data_speed_mbps'].values[0]:.1f} Mbps")
            with metric_col_3:
                st.metric("Signal Strength", f"{customer_row['signal_strength_dbm'].values[0]} dBm")
                st.metric("Complaints 30 Days", f"{customer_row['num_complaints_30d'].values[0]}")
                st.metric("Resolution Time", f"{customer_row['avg_resolution_time_hours'].values[0]:.1f} Hrs")
                
            st.markdown("### 📈 Trend Indicators")
            trend_col_1, trend_col_2, trend_col_3 = st.columns(3)
            with trend_col_1:
                st.metric("Usage Drop %", f"{customer_row['usage_drop_pct'].values[0]*100:.1f}%")
            with trend_col_2:
                st.metric("Recharge Drop %", f"{customer_row['recharge_drop_pct'].values[0]*100:.1f}%")
            with trend_col_3:
                st.metric("Inactive Days", f"{customer_row['inactive_days'].values[0]} Days")

        st.markdown("<hr>", unsafe_allow_html=True)
        
        with st.spinner("Analyzing prediction explainability..."):
            raw_input_df = customer_row.drop(columns=["customer_id", "churn", "churn_probability", "Risk Score (%)", "Risk Level"], errors="ignore")
            explanation = explainer.explain_instance(raw_input_df)
            prob = explanation["probability"]
            contributions = explanation["contributions"]
            
        col_pred1, col_pred2 = st.columns([1, 1])
        with col_pred1:
            st.subheader("Risk Score Dial")            
            gauge_fig = go.Figure(go.Indicator(         
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#f8fafc"},
                    'bar': {'color': "#dc143c" if prob > 0.7 else ("#EF6C00" if prob > 0.3 else "#003893")},
                    'bgcolor': "#0f172a",
                    'borderwidth': 1,
                    'bordercolor': "#1e293b",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(0, 56, 147, 0.15)'},
                        {'range': [30, 70], 'color': 'rgba(239, 108, 0, 0.15)'},
                        {'range': [70, 100], 'color': 'rgba(220, 20, 60, 0.15)'}
                    ]
                }
            ))
            gauge_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=260, margin=dict(l=30, r=30, t=30, b=30))
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            actual_churn = customer_row["churn"].values[0]
            st.write(f"**Actual Status:** {'🔴 Churned' if actual_churn == 1 else '🟢 Loyal'}")
            
            if prob > 0.7:
                st.error("🔴 **CRITICAL CHURN RISK**: Immediate retention program recommended.")
            elif prob > 0.3:
                st.warning("⚠️ **ELEVATED CHURN RISK**: Monitor activity and send targeted loyalty campaign.")
            else:
                st.success("🟢 **LOW CHURN RISK**: Customer demonstrates healthy activity levels.")

        with col_pred2:
            st.markdown("### 🧠 Explainable AI: SHAP Contributions")
            st.markdown("Features pushing simulated risk **UP** are styled in Crimson 🔴; mitigating features are in Dark Blue 🔵.")
            
            local_shap_fig = plot_local_shap(contributions, max_display=8, theme_dark=True)
            if hasattr(local_shap_fig, "update_traces"):
                local_shap_fig.update_traces(marker_color=np.where(contributions.head(8)["SHAP_Value"] > 0, "#dc143c", "#003893"))
            st.plotly_chart(local_shap_fig, use_container_width=True)

            st.subheader("Prediction Path (Waterfall Plot)")
            # Set styles for local waterfall context
            plt.style.use('dark_background')
            plt.rcParams['figure.facecolor'] = '#0a192f'
            fig_waterfall = plot_waterfall(explanation["shap_values_obj"], None, max_display=8)
            st.pyplot(fig_waterfall)

        st.markdown("### 📋 Diagnostic Summary (Key Reasons)")
        risk_reasons = get_human_readable_reasons(contributions, top_n=3, mode="risk")
        mitigation_reasons = get_human_readable_reasons(contributions, top_n=3, mode="mitigation")
        
        col_reason1, col_reason2 = st.columns(2)
        with col_reason1:
            st.markdown("##### 🚨 Top Churn Risk Factors")
            if risk_reasons:
                for reason in risk_reasons:
                    st.markdown(f"- 🔴 {reason}")
            else:
                st.markdown("No significant positive risk drivers identified.")
        with col_reason2:
            st.markdown("##### 🛡️ Top Retention Factors (Holding Risk Down)")
            if mitigation_reasons:
                for reason in mitigation_reasons:
                    st.markdown(f"- 🟢 {reason}")
            else:
                st.markdown("No significant mitigating factors identified.")

elif app_mode == "Simulator":
    st.subheader("🔮 What-If Simulator")
    st.markdown("Manually input subscriber details below to calculate real-time churn risk.")
    
    col_sim_in1, col_sim_in2, col_sim_in3 = st.columns(3)
    with col_sim_in1:
        st.markdown("#### 👥 Demographics")
        sim_age = st.slider("Age", 18, 80, 35)
        sim_gender = st.selectbox("Gender ", ["Male", "Female", "Other"])
        sim_province = st.selectbox("Province ", ["Koshi", "Madhesh", "Bagmati", "Gandaki", "Lumbini", "Karnali", "Sudurpashchim"])
        sim_district = st.selectbox("District Type", ["Urban", "Semi-Urban", "Rural"])
        sim_sim = st.selectbox("SIM Type ", ["Prepaid", "Postpaid"])
        sim_tenure = st.slider("Tenure (Days)", 15, 1800, 365)

        st.markdown("#### 📦 Active Services")
        sim_data_pack = st.checkbox("Data Pack Active", value=True)
        sim_voice_pack = st.checkbox("Voice Pack Active", value=False)
        sim_vas = st.checkbox("VAS (Value Added) Active", value=False)
        sim_roaming = st.checkbox("Roaming Active", value=False)

    with col_sim_in2:
        st.markdown("#### 📈 Usage Activity")
        sim_calls = st.slider("Calls Last 30d (Minutes)", 0.0, 1200.0, 250.0)
        sim_sms = st.slider("SMS Last 30d (Count)", 0, 300, 30)
        sim_data = st.slider("Data Last 30d (GB)", 0.0, 150.0, 10.0)
        sim_night = st.slider("Night Usage %", 0.0, 100.0, 20.0)
        
        st.markdown("#### 💳 Recharges")
        sim_last_rech = st.slider("Last Recharge (Days Ago)", 0, 90, 8)
        sim_avg_rech = st.slider("Avg Recharge NPR", 20.0, 2000.0, 250.0)
        sim_rech_count = st.slider("Recharge Count 30d", 0, 20, 3)

    with col_sim_in3:
        st.markdown("#### 🌐 Network Quality")
        sim_signal = st.slider("Signal Strength (dBm)", -115, -50, -85)
        sim_drop = st.slider("Call Drop Rate (%)", 0.0, 25.0, 1.2) / 100.0
        sim_speed = st.slider("Avg Data Speed (Mbps)", 0.1, 120.0, 18.0)
        
        st.markdown("#### 💬 Complaints")
        sim_complaints = st.slider("Complaints Last 30d", 0, 10, 0)
        sim_resol = st.slider("Avg Resolution Time (Hrs)", 0.0, 120.0, 0.0)
        
        st.markdown("#### 📈 Account Trends")
        sim_usage_drop = st.slider("Usage Drop Last Month (%)", -50.0, 100.0, 5.0) / 100.0
        sim_rech_drop = st.slider("Recharge Drop Last Month (%)", -50.0, 100.0, 0.0) / 100.0
        sim_inactive = st.slider("Inactive Days Last 30d", 0, 30, 2)

    st.markdown("<hr>", unsafe_allow_html=True)
    
    sim_row = pd.DataFrame([{
        "customer_id": "NP-SIMULATED",
        "age": sim_age,
        "gender": sim_gender,
        "province": sim_province,
        "district_type": sim_district,
        "sim_type": sim_sim,
        "tenure_days": sim_tenure,
        "calls_min_30d": sim_calls,
        "sms_count_30d": sim_sms,
        "data_gb_30d": sim_data,
        "night_usage_pct": sim_night,
        "last_recharge_days_ago": sim_last_rech,
        "avg_recharge_amount_npr": sim_avg_rech,
        "recharge_count_30d": sim_rech_count,
        "signal_strength_dbm": sim_signal,
        "call_drop_rate": sim_drop,
        "avg_data_speed_mbps": sim_speed,
        "num_complaints_30d": sim_complaints,
        "avg_resolution_time_hours": sim_resol,
        "data_pack_active": 1 if sim_data_pack else 0,
        "voice_pack_active": 1 if sim_voice_pack else 0,
        "vas_active": 1 if sim_vas else 0,
        "roaming_active": 1 if sim_roaming else 0,
        "usage_drop_pct": sim_usage_drop,
        "recharge_drop_pct": sim_rech_drop,
        "inactive_days": sim_inactive
    }])
    
    with st.spinner("Simulating..."):
        explanation = explainer.explain_instance(sim_row)
        prob = explanation["probability"]
        contributions = explanation["contributions"]

    col_sim_res1, col_sim_res2 = st.columns([1, 1])
    with col_sim_res1:
        st.markdown("### 🧭 Simulated Risk Score Dial")
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#f8fafc"},
                'bar': {'color': "#dc143c" if prob > 0.7 else ("#EF6C00" if prob > 0.3 else "#003893")},
                'bgcolor': "#0f172a",
                'borderwidth': 1,
                'bordercolor': "#1e293b",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(0, 56, 147, 0.15)'},
                    {'range': [30, 70], 'color': 'rgba(239, 108, 0, 0.15)'},
                    {'range': [70, 100], 'color': 'rgba(220, 20, 60, 0.15)'}
                ]
            }
        ))
        gauge_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=260, margin=dict(l=30, r=30, t=30, b=30))
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        if prob > 0.7:
            st.error("🔴 **CRITICAL CHURN RISK (SIMULATED)**: Subscriber is highly likely to churn.")
        elif prob > 0.3:
            st.warning("⚠️ **ELEVATED CHURN RISK (SIMULATED)**: Moderate churn probability.")
        else:
            st.success("🟢 **LOW CHURN RISK (SIMULATED)**: Healthy profile.")

    with col_sim_res2:
        st.markdown("### 🧠 Explainable AI: SHAP Contributions")
        st.markdown("Features pushing simulated risk **UP** are styled in Crimson 🔴; mitigating features are in Dark Blue 🔵.")
        local_shap_fig = plot_local_shap(contributions, max_display=8, theme_dark=True)
        if hasattr(local_shap_fig, "update_traces"):
            local_shap_fig.update_traces(marker_color=np.where(contributions.head(8)["SHAP_Value"] > 0, "#dc143c", "#003893"))
        st.plotly_chart(local_shap_fig, use_container_width=True)