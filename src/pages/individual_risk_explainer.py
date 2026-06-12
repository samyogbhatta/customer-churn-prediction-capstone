import streamlit as st
import plotly.graph_objects as go
from src.explainability import plot_local_shap, plot_waterfall


def render_individual_risk_explainer(filtered_df, explainer, metrics_data):
    """Render the Individual Risk Explainer page.

    Parameters
    ----------
    filtered_df : pd.DataFrame
        Data after applying sidebar filters.
    explainer : ChurnExplainer
        Preloaded explainer instance.
    metrics_data : dict
        Optional metrics (not used directly here but kept for API consistency).
    """
    st.subheader("Customer Profile & Prediction")
    
    if len(filtered_df) == 0:
        st.warning("No customers match the active filters. Reset filters on the sidebar to explore.")
        return
    
    selected_cust_id = st.selectbox(
        "Select Customer ID to Analyze",
        filtered_df["customer_id"].values
    )
    
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
        st.markdown("Features pushing the risk **UP** are in Red 🔴; features holding it **DOWN** are in Green 🟢.")
        local_shap_fig = plot_local_shap(contributions, max_display=8, theme_dark=True)
        st.plotly_chart(local_shap_fig, use_container_width=True)
        
        st.subheader("Prediction Path (Waterfall Plot)")
        fig_waterfall = plot_waterfall(explanation["shap_values_obj"], max_display=8)
        st.plotly_chart(fig_waterfall, use_container_width=True)
    
    st.markdown("### 📋 Diagnostic Summary (Key Reasons)")
    from src.explainability import get_human_readable_reasons
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
