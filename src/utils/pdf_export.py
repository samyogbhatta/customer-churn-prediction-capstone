import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from fpdf import FPDF

class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        # Fall back to Arial since standard FPDF installation might not have DejaVu
        self.set_font('Arial', '', 12)

    def header(self):
        self.set_font('Arial', 'B', 10)
        self.set_text_color(100, 116, 139) # slate-500
        self.cell(0, 10, 'Nepal Telecom Churn Analytics Executive Report', ln=False, align='L')
        self.cell(0, 10, 'CONFIDENTIAL', ln=True, align='R')
        self.set_draw_color(226, 232, 240) # slate-200
        self.line(15, 20, 195, 20)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(148, 163, 184) # slate-400
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def add_section_title(self, title: str):
        self.set_font('Arial', 'B', 16)
        self.set_text_color(15, 23, 42) # slate-900
        self.cell(0, 10, title, ln=True)
        self.ln(3)

    def add_subsection_title(self, title: str):
        self.set_font('Arial', 'B', 13)
        self.set_text_color(30, 41, 59) # slate-800
        self.cell(0, 10, title, ln=True)
        self.ln(2)

    def add_paragraph(self, text: str):
        self.set_font('Arial', '', 11)
        self.set_text_color(51, 65, 85) # slate-700
        self.multi_cell(0, 6, text)
        self.ln(4)

def export_report(overall_summary: dict, filtered_df: pd.DataFrame, explainer, output_path: str):
    """Generate a high-quality visual PDF report containing graphs and detailed interpretations.

    Args:
        overall_summary: Dictionary with KPI metrics.
        filtered_df: DataFrame containing the filtered customer dataset.
        explainer: ChurnExplainer instance used to compute SHAP values.
        output_path: File path where the PDF will be saved.
    """
    # Create a temporary directory to save plots
    temp_dir = tempfile.mkdtemp()
    
    # ------------------ GENERATE CHARTS USING MATPLOTLIB/SEABORN ------------------
    # Apply standard styling
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12})
    
    # 1. Churn by Province
    province_churn = filtered_df.groupby("province")["churn"].mean().reset_index()
    province_churn["churn_pct"] = province_churn["churn"] * 100
    province_churn = province_churn.sort_values(by="churn_pct", ascending=False)
    
    fig1, ax1 = plt.subplots(figsize=(6, 3.5))
    sns.barplot(data=province_churn, x="province", y="churn_pct", ax=ax1, hue="province", palette="Oranges_r", legend=False)
    ax1.set_title("Observed Churn Rate by Province (%)", pad=10)
    ax1.set_xlabel("Province")
    ax1.set_ylabel("Churn Rate (%)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plot1_path = os.path.join(temp_dir, "province_churn.png")
    fig1.savefig(plot1_path, dpi=100)
    plt.close(fig1)
    
    # 2. Correlation Matrix Heatmap
    corr_cols = [
        "tenure_days", "calls_min_30d", "data_gb_30d", "avg_recharge_amount_npr", 
        "signal_strength_dbm", "call_drop_rate", "num_complaints_30d", 
        "usage_drop_pct", "recharge_drop_pct", "inactive_days", "churn"
    ]
    corr_cols = [c for c in corr_cols if c in filtered_df.columns]
    corr_matrix = filtered_df[corr_cols].corr()
    corr_matrix.columns = [c.replace("_", " ").title() for c in corr_matrix.columns]
    corr_matrix.index = [c.replace("_", " ").title() for c in corr_matrix.index]
    
    fig2, ax2 = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", ax=ax2, center=0, cbar=False)
    ax2.set_title("Correlation Matrix of Key Metrics & Churn", pad=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plot2_path = os.path.join(temp_dir, "correlation_matrix.png")
    fig2.savefig(plot2_path, dpi=100)
    plt.close(fig2)
    
    # 3. Signal Strength Distribution
    fig3, ax3 = plt.subplots(figsize=(6, 3.5))
    loyal_sig = filtered_df[filtered_df["churn"] == 0]["signal_strength_dbm"]
    churned_sig = filtered_df[filtered_df["churn"] == 1]["signal_strength_dbm"]
    ax3.hist(loyal_sig, bins=15, alpha=0.6, label="Loyal", color="#2E7D32")
    ax3.hist(churned_sig, bins=15, alpha=0.6, label="Churned", color="#D32F2F")
    ax3.set_title("Signal Strength (dBm) Distribution", pad=10)
    ax3.set_xlabel("Signal Strength (dBm)")
    ax3.set_ylabel("Subscriber Count")
    ax3.legend()
    plt.tight_layout()
    plot3_path = os.path.join(temp_dir, "signal_strength.png")
    fig3.savefig(plot3_path, dpi=100)
    plt.close(fig3)
    
    # 4. Complaints vs Resolution Time
    has_complaints_df = filtered_df[filtered_df["num_complaints_30d"] > 0]
    plot4_path = None
    if len(has_complaints_df) > 0:
        fig4, ax4 = plt.subplots(figsize=(6, 3.5))
        sns.scatterplot(
            data=has_complaints_df,
            x="num_complaints_30d",
            y="avg_resolution_time_hours",
            hue=has_complaints_df["churn"].map({0: "Loyal", 1: "Churned"}),
            palette={"Loyal": "#2E7D32", "Churned": "#D32F2F"},
            alpha=0.6,
            ax=ax4
        )
        ax4.set_title("Complaints Volume vs Avg Resolution Time (Hours)", pad=10)
        ax4.set_xlabel("Number of Complaints (Last 30 Days)")
        ax4.set_ylabel("Resolution Time (Hrs)")
        ax4.legend(title="Status")
        plt.tight_layout()
        plot4_path = os.path.join(temp_dir, "complaints_resolution.png")
        fig4.savefig(plot4_path, dpi=100)
        plt.close(fig4)
        
    # 5. SHAP Global Features Calculation & Plots
    # Check sample size
    sample_size = min(300, len(filtered_df))
    df_sample = filtered_df.sample(n=sample_size, random_state=42)
    
    # Preprocess and compute SHAP
    X_sample_raw = df_sample.drop(columns=["churn", "customer_id", "churn_probability", "Risk Score (%)", "Risk Level"], errors="ignore")
    X_sample_processed = explainer.get_preprocessed_df(X_sample_raw)
    shap_vals, importance_df = explainer.get_global_explanations(X_sample_processed)
    importance_df["Clean_Feature"] = importance_df["Feature"].apply(lambda x: x.replace("_", " ").title())
    
    # Top 15 SHAP bar plot
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    top_15 = importance_df.head(15).copy()
    sns.barplot(data=top_15, x="Mean_Abs_SHAP", y="Clean_Feature", ax=ax5, hue="Clean_Feature", palette="Blues_r", legend=False)
    ax5.set_title("Top 15 Global Predictive Features (Mean Absolute SHAP)", pad=10)
    ax5.set_xlabel("Mean Absolute SHAP Value (Impact Magnitude)")
    ax5.set_ylabel("")
    plt.tight_layout()
    plot5_path = os.path.join(temp_dir, "global_importance.png")
    fig5.savefig(plot5_path, dpi=100)
    plt.close(fig5)
    
    # SHAP Beeswarm Summary Plot
    fig6 = plt.figure(figsize=(7.5, 5.5))
    shap.summary_plot(shap_vals, X_sample_processed, max_display=15, show=False)
    plt.title("Feature Impact Distributions (SHAP Beeswarm)", fontsize=12, pad=15)
    plt.tight_layout()
    plot6_path = os.path.join(temp_dir, "beeswarm.png")
    fig6.savefig(plot6_path, dpi=100)
    plt.close(fig6)
    
    # Age Dependence Plot
    fig7 = plt.figure(figsize=(6.5, 4.5))
    if "age" in df_sample.columns:
        fig_display = X_sample_processed.copy()
        fig_display["age"] = df_sample.loc[fig_display.index, "age"]
        shap.dependence_plot("age", shap_vals, fig_display, show=False)
    else:
        fallback_feat = X_sample_processed.columns[0]
        shap.dependence_plot(fallback_feat, shap_vals, X_sample_processed, show=False)
    plt.title("Risk Impact Dependence Plot (Age)", fontsize=12, pad=15)
    plt.tight_layout()
    plot7_path = os.path.join(temp_dir, "dependence.png")
    fig7.savefig(plot7_path, dpi=100)
    plt.close(fig7)
    
    # ------------------ BUILD THE PDF REPORT ------------------
    pdf = PDFReport()
    
    # PAGE 1: TITLE PAGE & EXECUTIVE SUMMARY
    pdf.add_page()
    pdf.ln(15)
    
    # Main Report Title
    pdf.set_font('Arial', 'B', 24)
    pdf.set_text_color(15, 23, 42) # slate-900
    pdf.cell(0, 15, "Nepal Telecom Churn Analytics", ln=True, align='L')
    pdf.cell(0, 15, "Executive Report", ln=True, align='L')
    pdf.set_draw_color(220, 38, 38) # NT Red
    pdf.line(15, 60, 110, 60)
    pdf.ln(12)
    
    # Subtitle / Metadata
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(71, 85, 105) # slate-600
    pdf.cell(0, 6, "AI-Driven Churn Explanations & Segment Analysis", ln=True)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 6, f"Generated On: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.cell(0, 6, f"Scope: Filtered Customer Segment ({overall_summary['total_customers']:,} Subscribers)", ln=True)
    pdf.ln(10)
    
    # Executive Summary Text
    pdf.add_section_title("1. Executive Overview Summary")
    pdf.add_paragraph(
        "This report provides an in-depth analytical assessment of customer churn dynamics "
        "and risk drivers within the selected customer segment. Using state-of-the-art XGBoost machine learning "
        "models and SHAP (SHapley Additive exPlanations) explainability frameworks, we translate "
        "complex customer activity data into actionable operational insights for regional managers, customer "
        "retention teams, and corporate decision-makers."
    )
    
    # KPI metrics box
    pdf.set_fill_color(248, 250, 252) # slate-50
    pdf.set_draw_color(203, 213, 225) # slate-300
    pdf.rect(15, 115, 180, 48, style='FD')
    
    pdf.set_y(118)
    pdf.set_x(20)
    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 6, "Key Performance Indicators (KPIs):", ln=True)
    pdf.ln(2)
    
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(0, 6, f"  * Total Evaluated Customers:  {overall_summary['total_customers']:,}", ln=True)
    pdf.cell(0, 6, f"  * Segment Churn Rate (Observed Churn):  {overall_summary['overall_churn_rate']:.2f}%", ln=True)
    pdf.cell(0, 6, f"  * Model Prediction Accuracy (XGB):  {overall_summary['model_accuracy']*100:.1f}%", ln=True)
    pdf.cell(0, 6, f"  * Monthly Revenue At Churn Risk:  Rs. {overall_summary['high_risk_revenue']:,.2f}", ln=True)
    pdf.ln(4)
    
    pdf.set_font('Arial', 'I', 9)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, "Note: Churn Risk is defined as customers with predicted churn probability >= 50%.", ln=True)
    
    pdf.set_y(175)
    pdf.add_section_title("How to Use This Report")
    pdf.add_paragraph(
        "Use the charts and interpretations in this document to identify structural vulnerabilities in service quality, "
        "geographical regions requiring network reinforcement, and customer behavior patterns indicating imminent churn. "
        "Please note that the comprehensive list of at-risk customers has been exported to the companion Excel spreadsheet "
        "for direct telecommunication campaigns."
    )

    # PAGE 2: REGIONAL & CORRELATION INSIGHTS
    pdf.add_page()
    pdf.add_section_title("2. Regional Churn Analysis")
    pdf.image(plot1_path, x=17, y=pdf.get_y(), w=176)
    pdf.set_y(pdf.get_y() + 105)
    
    pdf.add_subsection_title("Interpretation & Business Actions:")
    pdf.add_paragraph(
        "This chart represents the observed rate of customer churn across the different provinces of Nepal. "
        "By understanding how churn is geographically distributed, regional managers can identify provinces "
        "that may be suffering from network infrastructure issues, low coverage, or localized competitive threats. "
        "High churn rates in specific provinces highlight a critical need for targeted network expansion, "
        "regional promotions, and local dealer engagement to retain subscribers."
    )

    # PAGE 3: FEATURE CORRELATION
    pdf.add_page()
    pdf.add_section_title("3. Linear Feature Correlations")
    pdf.image(plot2_path, x=30, y=pdf.get_y(), w=150)
    pdf.set_y(pdf.get_y() + 120)
    
    pdf.add_subsection_title("Interpretation & Business Actions:")
    pdf.add_paragraph(
        "The correlation matrix calculates the linear relationship between key customer metrics (such as tenure, "
        "call drops, complaints, and recharge behavior) and the final churn outcome. The values range from -1.0 to +1.0. "
        "A positive value (red) indicates that as the feature increases, churn increases (e.g., number of complaints "
        "or inactive days). A negative value (blue) indicates that as the feature increases, churn decreases (e.g., "
        "higher average recharge amount or longer tenure). This matrix is essential for identifying the strongest "
        "predictors of churn at a glance."
    )

    # PAGE 4: SIGNAL STRENGTH DISTRIBUTION
    pdf.add_page()
    pdf.add_section_title("4. Network Quality & Signal Strength Distribution")
    pdf.image(plot3_path, x=17, y=pdf.get_y(), w=176)
    pdf.set_y(pdf.get_y() + 105)
    
    pdf.add_subsection_title("Interpretation & Business Actions:")
    pdf.add_paragraph(
        "This histogram shows the distribution of signal strength (measured in dBm) for both loyal (green) and churned "
        "(red) customers. Network signal strength typically ranges from -50 dBm (excellent) to -110 dBm or lower (very "
        "poor/no signal). A clear shift of churned customers towards the lower end of the signal strength spectrum "
        "indicates that poor network quality is a primary driver of customer dissatisfaction and churn. Addressing "
        "coverage blindspots in these areas can directly improve customer retention."
    )

    # PAGE 5: CUSTOMER SERVICE EXPERIENCE
    if plot4_path:
        pdf.add_page()
        pdf.add_section_title("5. Service Quality & Complaint Resolution")
        pdf.image(plot4_path, x=17, y=pdf.get_y(), w=176)
        pdf.set_y(pdf.get_y() + 105)
        
        pdf.add_subsection_title("Interpretation & Business Actions:")
        pdf.add_paragraph(
            "This scatter plot visualizes customer complaints against the average resolution time in hours, colored "
            "by churn status. Points clustered in the upper-right quadrant represent customers with many complaints "
            "that took a long time to resolve. A high concentration of churned customers in this area highlights "
            "the direct negative impact of delayed service resolution on customer loyalty. Streamlining the ticketing "
            "system and reducing resolution times for high-value customers is a crucial retention strategy."
        )

    # PAGE 6: GLOBAL FEATURE IMPORTANCE (AI INSIGHTS)
    pdf.add_page()
    pdf.add_section_title("6. AI-Driven Global Feature Importance")
    pdf.image(plot5_path, x=17, y=pdf.get_y(), w=176)
    pdf.set_y(pdf.get_y() + 115)
    
    pdf.add_subsection_title("Interpretation & Business Actions:")
    pdf.add_paragraph(
        "This chart showcases the top 15 features that have the highest overall influence on the machine learning "
        "model's churn predictions, ranked by their Mean Absolute SHAP value. SHAP (SHapley Additive exPlanations) "
        "values measure the direct contribution of each feature to the model's output. Features at the top of the "
        "chart represent the most critical indicators that the AI looks at when determining whether a customer "
        "is likely to churn. Business units should base their predictive triggers on these high-ranking attributes."
    )

    # PAGE 7: SHAP BEESWARM
    pdf.add_page()
    pdf.add_section_title("7. Feature Distributions & Risk Impact (Beeswarm)")
    pdf.image(plot6_path, x=15, y=pdf.get_y(), w=180)
    pdf.set_y(pdf.get_y() + 130)
    
    pdf.add_subsection_title("Interpretation & Business Actions:")
    pdf.add_paragraph(
        "The Beeswarm plot provides a detailed view of how feature values drive the risk predictions up or down "
        "for individual customers. Each point represents a customer. The position on the X-axis indicates the impact "
        "(points on the right increase churn risk, points on the left reduce it). The color of the point indicates "
        "the feature value (red for high, blue for low). For example, a red point for 'inactive_days' on the right "
        "indicates that high inactivity heavily drives up churn risk, whereas a blue point indicates low inactivity "
        "reduces risk."
    )

    # PAGE 8: RISK DEPENDENCY ANALYSIS
    pdf.add_page()
    pdf.add_section_title("8. Risk Dependency Analysis")
    pdf.image(plot7_path, x=17, y=pdf.get_y(), w=176)
    pdf.set_y(pdf.get_y() + 125)
    
    pdf.add_subsection_title("Interpretation & Business Actions:")
    pdf.add_paragraph(
        "The feature dependence plot illustrates how the risk impact (SHAP value) of a single feature (in this case, "
        "customer age) varies across its range of values. Points below the zero line indicate age groups that are "
        "more likely to remain loyal to the network, while points above the zero line indicate age groups with a "
        "higher risk of churn. This allows marketing teams to design age-specific retention packages, such as youth-centric "
        "data bundles or senior-citizen loyalty programs, based on actual historical behavior."
    )

    # Ensure directory exists and write PDF
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(output_path)
    
    # ------------------ CLEAN UP TEMPORARY PLOT IMAGES ------------------
    try:
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
    except Exception as e:
        print(f"Error cleaning up temp files: {e}")
