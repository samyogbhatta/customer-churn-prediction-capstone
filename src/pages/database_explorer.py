import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.explainability import plot_local_shap, plot_waterfall

def get_human_readable_reasons(contributions, top_n=3, mode="risk"):
    """
    Translates SHAP contributions into friendly human-readable descriptions.
    """
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
        elif "_" in feat:
            parts = feat.split("_")
            group = parts[0].title()
            category = "_".join(parts[1:])
            if val == 1:
                desc = f"{group} is {category}"
                
        if not desc:
            clean_feat = feat.replace("_", " ").title()
            if mode == "risk":
                desc = f"{clean_feat} is {val} (pushing risk up)"
            else:
                desc = f"{clean_feat} is {val} (holding risk down)"
                
        reasons.append(desc)
        
    return reasons

def render_database_explorer(uploaded_df, explainer):
    """
    Render the Database Explorer page.
    """
    st.subheader("🔍 Individual Customer Profiler & Prediction Explainer")
    
    if len(uploaded_df) == 0:
        st.warning("No customers match the active filters. Reset filters on the sidebar to explore.")
        return

    # Selectbox for Customer ID
    selected_cust_id = st.selectbox(
        "Select Customer ID to Analyze",
        uploaded_df["customer_id"].values
    )
    
    customer_row = uploaded_df[uploaded_df["customer_id"] == selected_cust_id]

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
        # Drop model score/risk labels before computing SHAP
        drop_cols = ["churn", "churn_probability", "Risk Score (%)", "Risk Level"]
        raw_input_df = customer_row.drop(columns=drop_cols, errors="ignore")
        explanation = explainer.explain_instance(raw_input_df)
        prob = explanation["probability"]
        contributions = explanation["contributions"]
        
    col_pred1, col_pred2 = st.columns([1, 1])
    
    with col_pred1:
        st.markdown("### 🧭 Risk Score Dial")            
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
        
        actual_churn = customer_row["churn"].values[0] if "churn" in customer_row.columns else None
        if actual_churn is not None:
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

        st.markdown("#### 🌊 Prediction Path (Waterfall Plot)")
        fig_waterfall = plot_waterfall(explanation["shap_values_obj"], max_display=8)
        st.plotly_chart(fig_waterfall, use_container_width=True)

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
