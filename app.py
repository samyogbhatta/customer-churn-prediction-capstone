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

def get_human_readable_reasons(contributions, top_n=3, mode="risk"):
    """
    Translates SHAP contributions into friendly human-readable descriptions.
    mode can be "risk" (positive SHAP) or "mitigation" (negative SHAP).
    """
    reasons = []
    
    if mode == "risk":
        filtered = contributions[contributions["SHAP_Value"] > 0].copy()
    else:
        filtered = contributions[contributions["SHAP_Value"] < 0].copy()
        
    for _, row in filtered.head(top_n).iterrows():
        feat = row["Feature"]
        val = row["Feature_Value"]
        
        # Friendly description formatting
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
            if mode == "risk":
                desc = f"Low average recharge amount (Rs. {val:.1f})"
            else:
                desc = f"Healthy average recharge amount (Rs. {val:.1f})"
        elif feat == "tenure_days":
            if mode == "risk":
                desc = f"Short subscription tenure ({int(val)} days)"
            else:
                desc = f"Long-term loyal subscription tenure ({int(val)} days)"
        elif feat == "avg_data_speed_mbps":
            if mode == "risk":
                desc = f"Slow average data speed ({val:.2f} Mbps)"
            else:
                desc = f"Fast average data speed ({val:.2f} Mbps)"
        elif feat == "data_gb_30d":
            if mode == "risk":
                desc = f"Low data usage last month ({val:.2f} GB)"
            else:
                desc = f"High data usage last month ({val:.2f} GB)"
        elif feat == "calls_min_30d":
            if mode == "risk":
                desc = f"Low call duration last month ({val:.1f} minutes)"
            else:
                desc = f"High call duration last month ({val:.1f} minutes)"
        elif feat == "recharge_count_30d":
            if mode == "risk":
                desc = f"Few recharges in last 30 days ({int(val)} times)"
            else:
                desc = f"Frequent recharges in last 30 days ({int(val)} times)"
        elif feat in ["data_pack_active", "voice_pack_active", "vas_active", "roaming_active"]:
            pack_name = feat.replace("_active", "").replace("_", " ").title()
            if val == 1:
                desc = f"Active {pack_name} service"
            else:
                desc = f"No active {pack_name} service"
        # Categorical features
        elif "_" in feat:
            parts = feat.split("_")
            group = parts[0].title()
            category = "_".join(parts[1:])
            if val == 1:
                desc = f"{group} is {category}"
                
        # Fallback if no specific template
        if not desc:
            clean_feat = feat.replace("_", " ").title()
            if mode == "risk":
                desc = f"{clean_feat} is {val} (pushing risk up)"
            else:
                desc = f"{clean_feat} is {val} (holding risk down)"
                
        reasons.append(desc)
        
    return reasons

@st.cache_data
def process_uploaded_data(uploaded_df_raw, _explainer_instance):
    # Drop customer_id and churn if they exist
    X_input = uploaded_df_raw.drop(columns=["customer_id", "churn"], errors="ignore")
    X_processed = _explainer_instance.get_preprocessed_df(X_input)
    probs = _explainer_instance.model.predict_proba(X_processed)[:, 1]
    return probs

@st.cache_data
def compute_uploaded_global_shap(uploaded_df_raw, _explainer_instance):
    X_input = uploaded_df_raw.drop(columns=["customer_id", "churn"], errors="ignore")
    X_processed = _explainer_instance.get_preprocessed_df(X_input)
    # limit to 200 rows for speed
    sample_size = min(200, len(X_processed))
    X_sample = X_processed.sample(n=sample_size, random_state=42) if len(X_processed) > sample_size else X_processed
    _, importance_df = _explainer_instance.get_global_explanations(X_sample)
    return importance_df, len(X_sample)

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
    ["📊 Executive Overview", "🔍 Database Explorer", "📂 Batch Prediction & Risk Explorer", "🧪 What-If Simulator"]
)

# Sidebar Filters (Apply to Executive Overview and Database Explorer)
if app_mode in ["📊 Executive Overview", "🔍 Database Explorer"]:
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
else:
    filtered_df = df_raw.copy()

# Compute and display KPIs only for non-batch modes
if app_mode != "📂 Batch Prediction & Risk Explorer":
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
    st.subheader("🧠 Global Explanations (AI-Driven Feature Importance)")
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


# ----------------- MODE: BATCH PREDICTION & RISK EXPLORER -----------------
elif app_mode == "📂 Batch Prediction & Risk Explorer":
    st.subheader("📂 Customer Segment Batch Prediction & Risk Explainer")
    st.markdown("Upload a CSV dataset of subscribers to identify high-risk individuals, understand churn reasons, and analyze segment-level risk distribution.")

    # Generate and offer sample template CSV
    sample_df = df_raw.copy()
    if "churn" in sample_df.columns:
        sample_df = sample_df.drop(columns=["churn"])
    
    sample_csv_data = sample_df.head(10).to_csv(index=False)
    
    st.markdown("### 🛠️ Step 1: Download & Prepare Your Data")
    col_dl, col_info = st.columns([1, 2])
    with col_dl:
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=sample_csv_data,
            file_name="nepal_telecom_churn_template.csv",
            mime="text/csv",
            help="Download a pre-formatted CSV template with the required columns for batch prediction."
        )
    with col_info:
        st.info("💡 **Tip:** Use this template to format your database. Make sure all numerical and categorical columns are present. The `customer_id` is recommended but optional.")

    st.markdown("### 📤 Step 2: Upload Subscriber Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file containing subscriber records", type=["csv"])

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded dataset with **{len(uploaded_df)}** customer records.")
            
            # Validate required columns
            required_features = NUMERICAL_COLS + CATEGORICAL_COLS + BINARY_COLS
            missing_features = [col for col in required_features if col not in uploaded_df.columns]
            
            if missing_features:
                st.error("❌ **Invalid CSV Format!** The uploaded file is missing critical columns required by the model:")
                st.write(missing_features)
                st.warning("Please ensure your uploaded CSV matches the template columns exactly.")
                st.stop()
                
            # If customer_id is missing, auto-generate it
            if "customer_id" not in uploaded_df.columns:
                uploaded_df["customer_id"] = [f"UPLOAD-{i+1:04d}" for i in range(len(uploaded_df))]
                
            # Let's run predictions!
            with st.spinner("Analyzing subscriber risk profiles using ML pipeline..."):
                # Run cached batch processing
                probs = process_uploaded_data(uploaded_df, explainer)
                
                # Add risk details to uploaded dataframe
                uploaded_df["churn_probability"] = probs
                uploaded_df["Risk Score (%)"] = (probs * 100).round(1)
                uploaded_df["Risk Level"] = np.where(probs > 0.7, "🔴 Critical", np.where(probs > 0.3, "⚠️ Elevated", "🟢 Low"))
            
            # Set up Interactive Threshold Slider
            st.markdown("<hr>", unsafe_allow_html=True)
            st.subheader("⚡ Batch Risk Threshold Control")
            threshold = st.slider("Define Risk Warning Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05,
                                  help="Customers with a churn probability above this threshold will be flagged as 'At Risk'.")
            
            # Calculate KPIs
            total_uploaded = len(uploaded_df)
            at_risk_df = uploaded_df[uploaded_df["churn_probability"] >= threshold]
            at_risk_count = len(at_risk_df)
            avg_risk = uploaded_df["churn_probability"].mean() * 100
            revenue_at_risk = at_risk_df["avg_recharge_amount_npr"].sum() if "avg_recharge_amount_npr" in at_risk_df.columns else 0.0
            
            # Display KPI Cards
            st.markdown(f"""
            <div class="metric-card-container">
                <div class="metric-card">
                    <div class="metric-value">{total_uploaded:,}</div>
                    <div class="metric-label">Total Uploaded</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{at_risk_count:,} ({at_risk_count/total_uploaded*100:.1f}%)</div>
                    <div class="metric-label">At Risk (P ≥ {threshold:.0%})</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{avg_risk:.1f}%</div>
                    <div class="metric-label">Average Churn Risk</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">Rs. {revenue_at_risk:,.0f}</div>
                    <div class="metric-label">Revenue at Risk</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Sub-TABS
            tab_overview, tab_individual, tab_drivers = st.tabs([
                "📊 Batch Overview & Risk Registry",
                "🔍 Individual Customer Risk Explainer",
                "🌍 Overall Segment Drivers (Global SHAP)"
            ])
            
            # TAB 1: BATCH OVERVIEW & RISK REGISTRY
            with tab_overview:
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    # Risk Level Donut Chart
                    risk_counts = uploaded_df["Risk Level"].value_counts().reset_index()
                    risk_counts.columns = ["Risk Level", "Count"]
                    # Ensure all risk levels are represented for color mapping consistency
                    all_levels = pd.DataFrame({"Risk Level": ["🔴 Critical", "⚠️ Elevated", "🟢 Low"]})
                    risk_counts = all_levels.merge(risk_counts, on="Risk Level", how="left").fillna(0)
                    
                    fig_donut = px.pie(
                        risk_counts,
                        values="Count",
                        names="Risk Level",
                        hole=0.4,
                        title="Risk Segment Distribution",
                        color="Risk Level",
                        color_discrete_map={
                            "🔴 Critical": "#D32F2F",
                            "⚠️ Elevated": "#EF6C00",
                            "🟢 Low": "#2E7D32"
                        }
                    )
                    fig_donut.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_donut, use_container_width=True)
                    
                with col_c2:
                    # Histogram of Probabilities
                    fig_hist = px.histogram(
                        uploaded_df,
                        x="churn_probability",
                        nbins=20,
                        title="Distribution of Churn Probability Scores",
                        labels={"churn_probability": "Churn Probability"},
                        color_discrete_sequence=["#3b82f6"]
                    )
                    fig_hist.add_vline(x=threshold, line_width=2.5, line_color="#E53935", line_dash="dash",
                                       annotation_text=f"Threshold ({threshold:.0%})", annotation_position="top right")
                    fig_hist.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                           xaxis_tickformat=".0%")
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Model evaluation if target 'churn' is present
                if "churn" in uploaded_df.columns:
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.subheader("📈 Uploaded Segment Model Performance Evaluation")
                    st.markdown("Since the uploaded dataset contains actual target labels (`churn`), we can evaluate the model's accuracy on this segment:")
                    
                    from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
                    
                    y_true = uploaded_df["churn"].astype(int)
                    y_pred = (uploaded_df["churn_probability"] >= threshold).astype(int)
                    
                    acc = accuracy_score(y_true, y_pred)
                    rec = recall_score(y_true, y_pred)
                    prec = precision_score(y_true, y_pred)
                    cm = confusion_matrix(y_true, y_pred)
                    
                    m_cols = st.columns(3)
                    with m_cols[0]:
                        st.metric("Segment Accuracy", f"{acc*100:.1f}%")
                    with m_cols[1]:
                        st.metric("Segment Recall (Sensitivity)", f"{rec*100:.1f}%")
                    with m_cols[2]:
                        st.metric("Segment Precision", f"{prec*100:.1f}%")
                        
                    st.markdown(f"""
                    **Confusion Matrix (Threshold = {threshold:.2f}):**
                    - **True Negatives (Loyal predicted Loyal):** {cm[0][0]}
                    - **False Positives (Loyal predicted Churn):** {cm[0][1]}
                    - **False Negatives (Churn predicted Loyal):** {cm[1][0]}
                    - **True Positives (Churn predicted Churn):** {cm[1][1]}
                    """)
                    
                # Table Registry
                st.subheader("📋 Customer Risk Registry")
                st.markdown("Search, filter, and review the risk profiles of all uploaded subscriber records.")
                
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    risk_filter = st.multiselect(
                        "Filter by Risk Level",
                        options=["🔴 Critical", "⚠️ Elevated", "🟢 Low"],
                        default=["🔴 Critical", "⚠️ Elevated", "🟢 Low"]
                    )
                with col_f2:
                    search_id = st.text_input("Search by Customer ID", "")
                
                display_df = uploaded_df.copy()
                display_df = display_df[display_df["Risk Level"].isin(risk_filter)]
                if search_id:
                    display_df = display_df[display_df["customer_id"].astype(str).str.contains(search_id, case=False)]
                
                cols_to_show = ["customer_id", "Risk Level", "Risk Score (%)", "age", "gender", "province", 
                                "tenure_days", "avg_recharge_amount_npr", "num_complaints_30d", "inactive_days"]
                if "churn" in uploaded_df.columns:
                    cols_to_show.append("churn")
                    
                st.dataframe(
                    display_df[cols_to_show].sort_values(by="Risk Score (%)", ascending=False),
                    use_container_width=True,
                    hide_index=True
                )
                
            # TAB 2: INDIVIDUAL CUSTOMER RISK EXPLAINER
            with tab_individual:
                st.subheader("🔍 Explainable AI: Customer-Specific Churn Drivers")
                st.markdown("Select any customer from the uploaded dataset to dissect their risk score and view the exact reasons behind the prediction.")
                
                dropdown_df = uploaded_df.sort_values(by="churn_probability", ascending=False)
                cust_options = [
                    f"{row['customer_id']} ({row['Risk Level']} - {row['Risk Score (%)']:.1f}%)" 
                    for _, row in dropdown_df.iterrows()
                ]
                
                selected_cust_opt = st.selectbox("Select Customer to Explain", cust_options)
                
                if selected_cust_opt:
                    selected_id = selected_cust_opt.split(" ")[0]
                    cust_row = uploaded_df[uploaded_df["customer_id"] == selected_id]
                    
                    c_det1, c_det2 = st.columns([1, 2])
                    
                    with c_det1:
                        st.markdown("#### 📋 Subscriber Demographic Profile")
                        st.markdown(f"**Customer ID:** `{selected_id}`")
                        st.markdown(f"**Age:** {cust_row['age'].values[0]}")
                        st.markdown(f"**Gender:** {cust_row['gender'].values[0]}")
                        st.markdown(f"**Province:** {cust_row['province'].values[0]}")
                        st.markdown(f"**SIM Type:** {cust_row['sim_type'].values[0]}")
                        st.markdown(f"**Tenure:** {cust_row['tenure_days'].values[0]} Days")
                        st.markdown(f"**Average Recharge:** Rs. {cust_row['avg_recharge_amount_npr'].values[0]}")
                        
                    with c_det2:
                        st.markdown("#### 📱 Usage & Experience Indicators")
                        sc1, sc2, sc3 = st.columns(3)
                        with sc1:
                            st.metric("Calls Last 30d", f"{cust_row['calls_min_30d'].values[0]:.1f} Min")
                            st.metric("Call Drop Rate", f"{cust_row['call_drop_rate'].values[0]*100:.2f}%")
                        with sc2:
                            st.metric("Data Usage 30d", f"{cust_row['data_gb_30d'].values[0]:.2f} GB")
                            st.metric("Avg Speed", f"{cust_row['avg_data_speed_mbps'].values[0]:.1f} Mbps")
                        with sc3:
                            st.metric("Complaints 30d", f"{int(cust_row['num_complaints_30d'].values[0])}")
                            st.metric("Inactive Days", f"{int(cust_row['inactive_days'].values[0])} Days")
                            
                    st.markdown("<hr>", unsafe_allow_html=True)
                    
                    with st.spinner("Calculating SHAP values for selected subscriber..."):
                        raw_input_df = cust_row.drop(columns=["customer_id", "churn_probability", "Risk Score (%)", "Risk Level", "churn"], errors="ignore")
                        explanation = explainer.explain_instance(raw_input_df)
                        prob = explanation["probability"]
                        contributions = explanation["contributions"]
                    
                    col_dial, col_shap = st.columns([1, 1])
                    
                    with col_dial:
                        st.markdown("#### 🧭 Risk Score Dial")
                        gauge_fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prob * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#f8fafc"},
                                'bar': {'color': "#D32F2F" if prob >= 0.7 else ("#EF6C00" if prob >= 0.3 else "#2E7D32")},
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
                            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            height=250, margin=dict(l=20, r=20, t=20, b=20)
                        )
                        st.plotly_chart(gauge_fig, use_container_width=True)
                        
                        if prob >= 0.7:
                            st.error(f"🔴 **Critical Churn Risk ({prob*100:.1f}%)**: Highly likely to leave. Deploy active retention offers immediately.")
                        elif prob >= 0.3:
                            st.warning(f"⚠️ **Elevated Churn Risk ({prob*100:.1f}%)**: Moderate probability. Recommend targeted communication and service packs.")
                        else:
                            st.success(f"🟢 **Low Churn Risk ({prob*100:.1f}%)**: Active, loyal subscriber profile.")
                            
                    with col_shap:
                        st.markdown("#### 🧠 AI Explanation: SHAP Impact")
                        local_shap_fig = plot_local_shap(contributions, max_display=8, theme_dark=True)
                        st.plotly_chart(local_shap_fig, use_container_width=True)
                        
                    # Friendly Natural Language Reasons Why Predicted At Risk
                    st.markdown("#### 📋 Diagnostic Summary (Key Reasons)")
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
                            
            # TAB 3: OVERALL SEGMENT DRIVERS (GLOBAL SHAP)
            with tab_drivers:
                st.subheader("🌍 Overall Segment Churn Drivers (Segment Global SHAP)")
                st.markdown("This analysis compiles the predictions of all uploaded customers to show which features have the greatest influence on churn decisions for **this uploaded segment**.")
                
                with st.spinner("Analyzing global segment driver importance..."):
                    importance_up_df, sample_len = compute_uploaded_global_shap(uploaded_df, explainer)
                    
                importance_up_df["Clean_Feature"] = importance_up_df["Feature"].apply(lambda x: x.replace("_", " ").title())
                
                fig_up_glob = px.bar(
                    importance_up_df.head(15),
                    y="Clean_Feature",
                    x="Mean_Abs_SHAP",
                    orientation="h",
                    title=f"Top 15 Churn Drivers for Uploaded Segment (Sample Size: {sample_len} rows)",
                    color="Mean_Abs_SHAP",
                    color_continuous_scale="Reds"
                )
                fig_up_glob.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(autorange="reversed"), height=450
                )
                st.plotly_chart(fig_up_glob, use_container_width=True)
        except Exception as e:
            st.error(f"❌ An error occurred while processing the uploaded file: {e}")
            st.info("Please make sure the CSV contains all standard features of the subscriber churn dataset.")

# ----------------- MODE 3: WHAT-IF SIMULATOR -----------------
elif app_mode == "🧪 What-If Simulator":
    st.subheader("🧪 Customer Scenario Simulator (What-If Analysis)")
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

        st.markdown("#### 📦 Active Services")
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
        st.markdown("### 🧭 Simulated Risk Score Dial")
        
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
        st.markdown("### 🧠 Explainable AI: SHAP Contributions")
        st.markdown("Features pushing simulated risk **UP** are in Red 🔴; features holding it **DOWN** are in Green 🟢.")
        local_shap_fig = plot_local_shap(contributions, max_display=8, theme_dark=True)
        st.plotly_chart(local_shap_fig, use_container_width=True)
