import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import json
import joblib

# Import custom explainability and plotting modules
from src.explainability import ChurnExplainer, plot_local_shap
from src.preprocessing import NUMERICAL_COLS, CATEGORICAL_COLS, BINARY_COLS

# Page Config
st.set_page_config(
    page_title="Telecom Churn Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Styling (Dark Slate Aesthetic)
st.markdown("""
<style>
    /* Metric Cards Styling */
    .metric-card-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        flex: 1;
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 12px;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Section dividers */
    hr {
        margin: 1.5rem 0;
        border-color: rgba(255, 255, 255, 0.1);
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

# Load data and assets
DATA_PATH = "data/telecom_churn_nepal.csv"
df_raw = load_dataset(DATA_PATH)
explainer = load_explainer()
metrics_data = load_model_metrics()

# Check if data and models are initialized
if df_raw is None or explainer is None or metrics_data is None:
    st.warning("⚠️ ML Pipeline needs to be executed before running the dashboard.")
    st.info("Please run the data generator and training pipeline in your terminal first:")
    st.code("python src/data_generator.py\npython src/train.py", language="bash")
    st.stop()

# Header Section
st.title("🇳🇵 Nepalese Telecom Churn Analytics & Explainable AI")
st.markdown("Developed Automated Customer Churn Prediction System Using Machine Learning and Explainable AI (SHAP)")

# Sidebar Navigation and Filters
st.sidebar.image("https://img.icons8.com/clouds/100/database.png", width=80)
st.sidebar.title("Dashboard Controls")

# Mode Selection
app_mode = st.sidebar.radio(
    "Select Interface Mode",
    ["Executive Overview", "🔍 Database Explorer", "🧪 What-If Simulator"]
)

# Sidebar Filters (Apply to Executive Overview and Database Explorer)
st.sidebar.subheader("Filter Data")
genders = ["All"] + list(df_raw["gender"].unique())
gender_filter = st.sidebar.selectbox("Gender", genders)

provinces = ["All"] + list(df_raw["province"].unique())
province_filter = st.sidebar.selectbox("Province", provinces)

sim_types = ["All"] + list(df_raw["sim_type"].unique())
sim_filter = st.sidebar.selectbox("SIM Type", sim_types)

# Filter Dataset
filtered_df = df_raw.copy()
if gender_filter != "All":
    filtered_df = filtered_df[filtered_df["gender"] == gender_filter]
if province_filter != "All":
    filtered_df = filtered_df[filtered_df["province"] == province_filter]
if sim_filter != "All":
    filtered_df = filtered_df[filtered_df["sim_type"] == sim_filter]

# Compute KPIs
total_customers = len(filtered_df)
overall_churn_rate = (filtered_df["churn"].mean() * 100) if total_customers > 0 else 0.0
model_accuracy = metrics_data.get("accuracy", 0.85)

# Calculate simulated Revenue at Risk
# We use the model to predict probability for each customer in filtered df
X_filtered_processed = explainer.get_preprocessed_df(filtered_df.drop(columns=["churn"]))
probs_filtered = explainer.model.predict_proba(X_filtered_processed)[:, 1]
high_risk_revenue = filtered_df.loc[probs_filtered > 0.5, "avg_recharge_amount_npr"].sum()

# Display KPI Section
kpi_cols = st.columns(4)

with kpi_cols[0]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_customers:,}</div>
        <div class="metric-label">Total Customers</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[1]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{overall_churn_rate:.1f}%</div>
        <div class="metric-label">Observed Churn Rate</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[2]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{model_accuracy*100:.1f}%</div>
        <div class="metric-label">Model Accuracy (XGB)</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[3]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">Rs. {high_risk_revenue:,.0f}</div>
        <div class="metric-label">Monthly Revenue at Risk</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ----------------- MODE 1: EXECUTIVE OVERVIEW -----------------
if app_mode == "📊 Executive Overview":
    st.subheader("📊 Executive Overview & Demographic Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn by Province Chart
        province_churn = filtered_df.groupby("province")["churn"].mean().reset_index()
        province_churn["churn_pct"] = province_churn["churn"] * 100
        province_churn = province_churn.sort_values(by="churn_pct", ascending=False)
        
        fig_prov = px.bar(
            province_churn,
            x="province",
            y="churn_pct",
            color="churn_pct",
            color_continuous_scale="Reds",
            title="Observed Churn Rate by Province (%)",
            labels={"province": "Province", "churn_pct": "Churn Rate (%)"}
        )
        fig_prov.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_prov, use_container_width=True)

    with col2:
        # Numerical Correlation Heatmap
        corr_cols = [
            "tenure_days", "calls_min_30d", "data_gb_30d", "avg_recharge_amount_npr", 
            "signal_strength_dbm", "call_drop_rate", "num_complaints_30d", 
            "usage_drop_pct", "recharge_drop_pct", "inactive_days", "churn"
        ]
        corr_matrix = filtered_df[corr_cols].corr()
        # Rename for clean mapping
        corr_matrix.columns = [c.replace("_", " ").title() for c in corr_matrix.columns]
        corr_matrix.index = [c.replace("_", " ").title() for c in corr_matrix.index]
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix (Key Features & Churn)"
        )
        fig_corr.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

    col3, col4 = st.columns(2)
    
    with col3:
        # Distribution of Signal Strength vs Churn
        fig_sig = px.histogram(
            filtered_df,
            x="signal_strength_dbm",
            color=filtered_df["churn"].map({0: "Loyal", 1: "Churned"}),
            barmode="overlay",
            title="Signal Strength (dBm) Distribution by Churn",
            color_discrete_map={"Loyal": "#2E7D32", "Churned": "#D32F2F"},
            opacity=0.7
        )
        fig_sig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sig, use_container_width=True)
        
    with col4:
        # Complaints & Resolution vs Churn
        fig_comp = px.scatter(
            filtered_df[filtered_df["num_complaints_30d"] > 0],
            x="num_complaints_30d",
            y="avg_resolution_time_hours",
            color=filtered_df[filtered_df["num_complaints_30d"] > 0]["churn"].map({0: "Loyal", 1: "Churned"}),
            title="Complaints Density vs Resolution Time (Hours)",
            labels={"num_complaints_30d": "Number of Complaints", "avg_resolution_time_hours": "Resolution Time (Hrs)"},
            color_discrete_map={"Loyal": "#2E7D32", "Churned": "#D32F2F"},
            opacity=0.6
        )
        fig_comp.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_comp, use_container_width=True)

    # Global SHAP feature importance
    st.subheader("Global Explanations (AI-Driven Feature Importance)")
    st.markdown("This chart displays the average absolute impact each feature has on the overall model predictions.")
    
    # Compute global importance on a random sample of 200 customers (to keep it fast)
    @st.cache_data
    def compute_global_importance(_explainer, df_sample):
        X_sample_processed = _explainer.get_preprocessed_df(df_sample.drop(columns=["churn"]))
        _, importance_df = _explainer.get_global_explanations(X_sample_processed)
        return importance_df

    sample_size = min(300, len(df_raw))
    df_sample = df_raw.sample(n=sample_size, random_state=42)
    importance_df = compute_global_importance(explainer, df_sample)
    
    # Clean feature names for displaying
    importance_df["Clean_Feature"] = importance_df["Feature"].apply(lambda x: x.replace("_", " ").title())
    
    fig_glob = px.bar(
        importance_df.head(15),
        y="Clean_Feature",
        x="Mean_Abs_SHAP",
        orientation="h",
        title="Top 15 Global Predictive Features (Mean Absolute SHAP)",
        color="Mean_Abs_SHAP",
        color_continuous_scale="Blues"
    )
    fig_glob.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed"),
        height=450
    )
    st.plotly_chart(fig_glob, use_container_width=True)


# ----------------- MODE 2: DATABASE EXPLORER -----------------
elif app_mode == "🔍 Database Explorer":
    st.subheader("🔍 Individual Customer Profiler & Prediction Explainer")
    
    if len(filtered_df) == 0:
        st.warning("No customers match the active filters. Reset filters on the sidebar to explore.")
    else:
        # Customer ID selector
        selected_cust_id = st.selectbox(
            "Select Customer ID to Analyze",
            filtered_df["customer_id"].values
        )
        
        customer_row = filtered_df[filtered_df["customer_id"] == selected_cust_id]
        
        # Display profile details
        col_detail1, col_detail2 = st.columns([1, 2])
        
        with col_detail1:
            st.markdown("### Demographic Profile")
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
            st.markdown("### Usage & Service Quality Metrics")
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
        
        # Run Model Prediction & SHAP
        with st.spinner("Analyzing prediction explainability..."):
            raw_input_df = customer_row.drop(columns=["churn"])
            explanation = explainer.explain_instance(raw_input_df)
            prob = explanation["probability"]
            contributions = explanation["contributions"]
            
        col_pred1, col_pred2 = st.columns([1, 1])
        
        with col_pred1:
            st.markdown("### 🧭 Risk Score Dial")
            
            # Speedometer Gauge
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#f8fafc"},
                    'bar': {'color': "#D32F2F" if prob > 0.7 else ("#EF6C00" if prob > 0.3 else "#2E7D32")},
                    'bgcolor': "#1e293b",
                    'borderwidth': 1,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(46, 125, 50, 0.15)'},
                        {'range': [30, 70], 'color': 'rgba(239, 108, 0, 0.15)'},
                        {'range': [70, 100], 'color': 'rgba(211, 47, 47, 0.15)'}
                    ]
                }
            ))
            gauge_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=260,
                margin=dict(l=30, r=30, t=30, b=30)
            )
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Text Output
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
            st.markdown("Features pushing the risk **UP** are in Red 🔴; features holding it **DOWN** are in Green 🟢.")
            local_shap_fig = plot_local_shap(contributions, max_display=8, theme_dark=True)
            st.plotly_chart(local_shap_fig, use_container_width=True)


# ----------------- MODE 3: WHAT-IF SIMULATOR -----------------
elif app_mode == "What-If Simulator":
    st.subheader("Customer Scenario Simulator (What-If Analysis)")
    st.markdown("Manually input subscriber details below to calculate real-time churn risk and see customized SHAP feature contributions.")
    
    col_sim_in1, col_sim_in2, col_sim_in3 = st.columns(3)
    
    with col_sim_in1:
        st.markdown("#### 👤 Demographics")
        sim_age = st.slider("Age", 18, 80, 35)
        sim_gender = st.selectbox("Gender ", ["Male", "Female", "Other"])
        sim_province = st.selectbox("Province ", ["Koshi", "Madhesh", "Bagmati", "Gandaki", "Lumbini", "Karnali", "Sudurpashchim"])
        sim_district = st.selectbox("District Type", ["Urban", "Semi-Urban", "Rural"])
        sim_sim = st.selectbox("SIM Type ", ["Prepaid", "Postpaid"])
        sim_tenure = st.slider("Tenure (Days)", 15, 1800, 365)

        st.markdown("#### Active Services")
        sim_data_pack = st.checkbox("Data Pack Active", value=True)
        sim_voice_pack = st.checkbox("Voice Pack Active", value=False)
        sim_vas = st.checkbox("VAS (Value Added) Active", value=False)
        sim_roaming = st.checkbox("Roaming Active", value=False)

    with col_sim_in2:
        st.markdown("#### 📱 Usage Activity")
        sim_calls = st.slider("Calls Last 30d (Minutes)", 0.0, 1200.0, 250.0)
        sim_sms = st.slider("SMS Last 30d (Count)", 0, 300, 30)
        sim_data = st.slider("Data Last 30d (GB)", 0.0, 150.0, 10.0)
        sim_night = st.slider("Night Usage %", 0.0, 100.0, 20.0)
        
        st.markdown("#### Recharges")
        sim_last_rech = st.slider("Last Recharge (Days Ago)", 0, 90, 8)
        sim_avg_rech = st.slider("Avg Recharge NPR", 20.0, 2000.0, 250.0)
        sim_rech_count = st.slider("Recharge Count 30d", 0, 20, 3)

    with col_sim_in3:
        st.markdown("#### Network Quality")
        sim_signal = st.slider("Signal Strength (dBm)", -115, -50, -85)
        sim_drop = st.slider("Call Drop Rate (%)", 0.0, 25.0, 1.2) / 100.0
        sim_speed = st.slider("Avg Data Speed (Mbps)", 0.1, 120.0, 18.0)
        
        st.markdown("#### Complaints")
        sim_complaints = st.slider("Complaints Last 30d", 0, 10, 0)
        sim_resol = st.slider("Avg Resolution Time (Hrs)", 0.0, 120.0, 0.0)
        
        st.markdown("#### Account Trends")
        sim_usage_drop = st.slider("Usage Drop Last Month (%)", -50.0, 100.0, 5.0) / 100.0
        sim_rech_drop = st.slider("Recharge Drop Last Month (%)", -50.0, 100.0, 0.0) / 100.0
        sim_inactive = st.slider("Inactive Days Last 30d", 0, 30, 2)

    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Pack parameters into a DataFrame
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
    
    # Run Prediction & SHAP
    with st.spinner("Simulating..."):
        explanation = explainer.explain_instance(sim_row)
        prob = explanation["probability"]
        contributions = explanation["contributions"]

    col_sim_res1, col_sim_res2 = st.columns([1, 1])
    
    with col_sim_res1:
        st.markdown("### Simulated Risk Score Dial")
        
        # Speedometer Gauge
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#f8fafc"},
                'bar': {'color': "#D32F2F" if prob > 0.7 else ("#EF6C00" if prob > 0.3 else "#2E7D32")},
                'bgcolor': "#1e293b",
                'borderwidth': 1,
                'bordercolor': "#334155",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(46, 125, 50, 0.15)'},
                    {'range': [30, 70], 'color': 'rgba(239, 108, 0, 0.15)'},
                    {'range': [70, 100], 'color': 'rgba(211, 47, 47, 0.15)'}
                ]
            }
        ))
        gauge_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=260,
            margin=dict(l=30, r=30, t=30, b=30)
        )
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        if prob > 0.7:
            st.error("🔴 **CRITICAL CHURN RISK (SIMULATED)**: Subscriber is highly likely to churn. Recommend immediate outreach.")
        elif prob > 0.3:
            st.warning("⚠️ **ELEVATED CHURN RISK (SIMULATED)**: Moderate churn probability. Monitor indicators and send special bundle packs.")
        else:
            st.success("🟢 **LOW CHURN RISK (SIMULATED)**: Healthy profile. Low churn probability.")

    with col_sim_res2:
        st.markdown("### Explainable AI: SHAP Contributions")
        st.markdown("Features pushing simulated risk **UP** are in Red 🔴; features holding it **DOWN** are in Green 🟢.")
        local_shap_fig = plot_local_shap(contributions, max_display=8, theme_dark=True)
        st.plotly_chart(local_shap_fig, use_container_width=True)
