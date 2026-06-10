import streamlit as st
import plotly.express as px
from src.explainability import plot_summary, plot_mean_bar, plot_dependence, plot_local_shap, plot_waterfall

def render_executive_overview(filtered_df, explainer, df_raw):
    """Render the Executive Overview page.
    Parameters
    ----------
    filtered_df: pd.DataFrame
        Data after applying sidebar filters.
    explainer: ChurnExplainer
        Preloaded explainer instance.
    df_raw: pd.DataFrame
        Full dataset (used for sampling in global explanations).
    """
    st.subheader("Executive Overview & Demographic Insights")

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
            labels={"province": "Province", "churn_pct": "Churn Rate (%)"},
        )
        fig_prov.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_prov, use_container_width=True)

    with col2:
        # Numerical Correlation Heatmap
        corr_cols = [
            "tenure_days", "calls_min_30d", "data_gb_30d", "avg_recharge_amount_npr",
            "signal_strength_dbm", "call_drop_rate", "num_complaints_30d",
            "usage_drop_pct", "recharge_drop_pct", "inactive_days", "churn",
        ]
        corr_matrix = filtered_df[corr_cols].corr()
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
        # Distribution of Signal Strength vs Churn
        fig_sig = px.histogram(
            filtered_df,
            x="signal_strength_dbm",
            color=filtered_df["churn"].map({0: "Loyal", 1: "Churned"}),
            barmode="overlay",
            title="Signal Strength (dBm) Distribution by Churn",
            color_discrete_map={"Loyal": "#2E7D32", "Churned": "#D32F2F"},
            opacity=0.7,
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
            opacity=0.6,
        )
        fig_comp.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_comp, use_container_width=True)

    st.subheader("Global Explanations (AI-Driven Feature Importance)")
    st.markdown("This chart displays the average absolute impact each feature has on the overall model predictions.")

    @st.cache_data
    def compute_global_importance(_explainer, df_sample):
        X_sample_processed = _explainer.get_preprocessed_df(df_sample.drop(columns=["churn"]))
        shap_vals, importance_df = _explainer.get_global_explanations(X_sample_processed)
        return shap_vals, importance_df, X_sample_processed

    sample_size = min(300, len(df_raw))
    df_sample = df_raw.sample(n=sample_size, random_state=42)
    shap_vals, importance_df, X_sample_processed = compute_global_importance(explainer, df_sample)
    importance_df["Clean_Feature"] = importance_df["Feature"].apply(lambda x: x.replace("_", " ").title())

    # Plot 1: Top 15 Global Features
    fig_glob = px.bar(
        importance_df.head(15),
        y="Clean_Feature",
        x="Mean_Abs_SHAP",
        orientation="h",
        title="Top 15 Global Predictive Features (Mean Absolute SHAP)",
        color="Mean_Abs_SHAP",
        color_continuous_scale="Blues",
    )
    fig_glob.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(autorange="reversed"), height=400)

    # Plot 2: Beeswarm
    fig_summary = plot_summary(shap_vals, X_sample_processed)
    # Plot 3: Mean SHAP Bar
    fig_bar = plot_mean_bar(shap_vals, X_sample_processed)
    # Plot 4: Feature Dependence (Age)
    fig_display = X_sample_processed.copy()
    fig_display["age"] = df_sample.loc[fig_display.index, "age"]
    fig_dependence = plot_dependence("age", shap_vals, fig_display)

    st.markdown("<hr>", unsafe_allow_html=True)
    col_row1_left, col_row1_right = st.columns(2)
    with col_row1_left:
        st.subheader("🧠 Top 15 Global Features")
        st.plotly_chart(fig_glob, use_container_width=True)
        st.info("**What this shows:** This chart displays the overall predictive importance of features. The longer the bar, the more weight the AI model places on this specific feature when determining whether *any* generic customer will churn or stay loyal.")
    with col_row1_right:
        st.subheader("🐝 Feature Distributions (Beeswarm)")
        st.plotly_chart(fig_summary, use_container_width=True)
        st.info("**What this shows:** Every dot represents a single customer sample. * Position (X-Axis): Dots on the right push risk UP (higher churn); dots on the left pull risk DOWN. * Color: Red represents high values of that feature, blue represents low values.")
    st.markdown("<br>", unsafe_allow_html=True)
    col_row2_left, col_row2_right = st.columns(2)
    with col_row2_left:
        st.subheader("📊 Mean SHAP Importance")
        st.plotly_chart(fig_bar, use_container_width=True)
        st.info("**What this shows:** A refined baseline representation of pure feature magnitude. It measures the average magnitude of change a feature introduces to the model's math, stripping away directionality to reveal raw predictive strength.")
    with col_row2_right:
        st.subheader("📈 Feature Dependence (Age)")
        st.plotly_chart(fig_dependence, use_container_width=True)
        st.info("**What this shows:** This highlights how the customer's exact age correlates to churn behavior. The Y-axis represents the SHAP risk impact. If you see the plot dip below 0 at certain ages, it indicates specific age bands that are mathematically more loyal to the network.")
