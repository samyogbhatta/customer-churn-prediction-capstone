import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.explainability import plot_summary, plot_mean_bar, plot_dependence

def render_uploaded_data_overview(uploaded_df, explainer, metrics_data):
    """
    Render the Uploaded Data Overview page.
    """
    st.subheader("📊 Uploaded Data Insights & Business Metrics")

    # Determine if churn values are predicted or actual
    has_actual_churn = "churn" in uploaded_df.columns and not uploaded_df["churn"].isna().all()
    
    # Calculate metrics
    total_customers = len(uploaded_df)
    overall_churn_rate = (uploaded_df["churn"].mean() * 100) if total_customers > 0 else 0.0
    model_accuracy = metrics_data.get("accuracy", 0.85) if metrics_data else 0.85
    
    # Calculate revenue at risk
    high_risk_threshold = 0.5
    high_risk_mask = uploaded_df["churn_probability"] > high_risk_threshold
    high_risk_revenue = uploaded_df.loc[high_risk_mask, "avg_recharge_amount_npr"].sum() if "avg_recharge_amount_npr" in uploaded_df.columns else 0.0

    # Display KPI Section
    kpi_cols = st.columns(4)

    with kpi_cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_customers:,}</div>
            <div class="metric-label">Total Uploaded Customers</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[1]:
        label = "Observed Churn Rate" if has_actual_churn else "Predicted Churn Rate"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{overall_churn_rate:.1f}%</div>
            <div class="metric-label">{label}</div>
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

    # Demographic & Service Analysis
    st.subheader("Demographic & Service Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        province_churn = uploaded_df.groupby("province")["churn"].mean().reset_index()
        province_churn["churn_pct"] = province_churn["churn"] * 100
        province_churn = province_churn.sort_values(by="churn_pct", ascending=False)
        fig_prov = px.bar(
            province_churn,
            x="province",
            y="churn_pct",
            color="churn_pct",
            color_continuous_scale="Reds",
            title="Churn Rate by Province (%)",
            labels={"province": "Province", "churn_pct": "Churn Rate (%)"},
        )
        fig_prov.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_prov, use_container_width=True)

    with col2:
        corr_cols = [
            "tenure_days", "calls_min_30d", "data_gb_30d", "avg_recharge_amount_npr",
            "signal_strength_dbm", "call_drop_rate", "num_complaints_30d",
            "usage_drop_pct", "recharge_drop_pct", "inactive_days", "churn",
        ]
        corr_cols = [col for col in corr_cols if col in uploaded_df.columns]
        corr_matrix = uploaded_df[corr_cols].corr()
        corr_matrix.columns = [c.replace("_", " ").title() for c in corr_matrix.columns]
        corr_matrix.index = [c.replace("_", " ").title() for c in corr_matrix.index]
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix (Key Features & Churn)",
        )
        fig_corr.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig_sig = px.histogram(
            uploaded_df,
            x="signal_strength_dbm",
            color=uploaded_df["churn"].map({0: "Loyal", 1: "Churned"}),
            barmode="overlay",
            title="Signal Strength (dBm) Distribution by Churn",
            color_discrete_map={"Loyal": "#2E7D32", "Churned": "#D32F2F"},
            opacity=0.7,
        )
        fig_sig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sig, use_container_width=True)

    with col4:
        has_complaints_df = uploaded_df[uploaded_df["num_complaints_30d"] > 0]
        if len(has_complaints_df) > 0:
            fig_comp = px.scatter(
                has_complaints_df,
                x="num_complaints_30d",
                y="avg_resolution_time_hours",
                color=has_complaints_df["churn"].map({0: "Loyal", 1: "Churned"}),
                title="Complaints Density vs Resolution Time (Hours)",
                labels={"num_complaints_30d": "Number of Complaints", "avg_resolution_time_hours": "Resolution Time (Hrs)"},
                color_discrete_map={"Loyal": "#2E7D32", "Churned": "#D32F2F"},
                opacity=0.6,
            )
            fig_comp.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("No customer complaints registered in the uploaded dataset to display.")

    st.subheader("🧠 Global Explanations (AI-Driven Feature Importance)")
    st.markdown("This section analyzes the global feature contributions for the uploaded subscriber segment.")

    @st.cache_data
    def compute_uploaded_global_shap(uploaded_df_raw, _explainer_instance):
        X_input = uploaded_df_raw.drop(columns=["customer_id", "churn", "churn_probability", "Risk Score (%)", "Risk Level"], errors="ignore")
        X_processed = _explainer_instance.get_preprocessed_df(X_input)
        sample_size = min(300, len(X_processed))
        X_sample = X_processed.sample(n=sample_size, random_state=42) if len(X_processed) > sample_size else X_processed
        shap_vals, importance_df = _explainer_instance.get_global_explanations(X_sample)
        return shap_vals, importance_df, X_sample

    with st.spinner("Calculating SHAP values for the uploaded dataset..."):
        shap_vals, importance_df, X_sample_processed = compute_uploaded_global_shap(uploaded_df, explainer)

    importance_df["Clean_Feature"] = importance_df["Feature"].apply(lambda x: x.replace("_", " ").title())

    fig_glob = px.bar(
        importance_df.head(15),
        y="Clean_Feature",
        x="Mean_Abs_SHAP",
        orientation="h",
        title="Top 15 Predictive Features (Mean Absolute SHAP)",
        color="Mean_Abs_SHAP",
        color_continuous_scale="Blues",
    )
    fig_glob.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(autorange="reversed"), height=400)

    fig_summary = plot_summary(shap_vals, X_sample_processed)
    fig_bar = plot_mean_bar(shap_vals, X_sample_processed)
    fig_display = X_sample_processed.copy()
    if "age" in uploaded_df.columns:
        fig_display["age"] = uploaded_df.loc[fig_display.index, "age"]
        fig_dependence = plot_dependence("age", shap_vals, fig_display)
    else:
        fallback_feat = X_sample_processed.columns[0]
        fig_dependence = plot_dependence(fallback_feat, shap_vals, fig_display)

    st.markdown("<hr>", unsafe_allow_html=True)
    col_row1_left, col_row1_right = st.columns(2)
    with col_row1_left:
        st.subheader("🧠 Top 15 Global Features")
        st.plotly_chart(fig_glob, use_container_width=True)
        st.info("**What this shows:** This chart displays the overall predictive importance of features.")
    with col_row1_right:
        st.subheader("🐝 Feature Distributions (Beeswarm)")
        st.plotly_chart(fig_summary, use_container_width=True)
        st.info("**What this shows:** Every dot represents a single customer sample.")
    st.markdown("<br>", unsafe_allow_html=True)
    col_row2_left, col_row2_right = st.columns(2)
    with col_row2_left:
        st.subheader("📊 Mean SHAP Importance")
        st.plotly_chart(fig_bar, use_container_width=True)
        st.info("**What this shows:** A refined baseline representation of pure feature magnitude.")
    with col_row2_right:
        st.subheader("📈 Feature Dependence (Age)")
        st.plotly_chart(fig_dependence, use_container_width=True)
        st.info("**What this shows:** This highlights how the customer's exact age correlates to churn behavior.")
