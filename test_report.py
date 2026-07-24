import os
import pandas as pd
import numpy as np
from src.components.report_generator import generate_report_pdf
from src.explainability import ChurnExplainer

def run_test():
    # 1. Load the explainer
    print("Loading ChurnExplainer...")
    explainer = ChurnExplainer()
    
    # 2. Load the dataset
    print("Loading dataset...")
    df = pd.read_csv("data/telecom_churn_nepal.csv")
    
    # 3. Simulate processing (like in app.py)
    print("Processing dataset...")
    # Add dummy customer ID if not present
    if "customer_id" not in df.columns:
        df["customer_id"] = [f"NP-CUST-{i+1:05d}" for i in range(len(df))]
        
    REQUIRED_RAW_FEATURES = [
        "age", "gender", "province", "district_type", "sim_type", "tenure_days",
        "calls_min_30d", "sms_count_30d", "data_gb_30d", "night_usage_pct",
        "last_recharge_days_ago", "avg_recharge_amount_npr", "recharge_count_30d",
        "signal_strength_dbm", "call_drop_rate", "avg_data_speed_mbps",
        "num_complaints_30d", "avg_resolution_time_hours",
        "data_pack_active", "voice_pack_active", "vas_active", "roaming_active",
        "usage_drop_pct", "recharge_drop_pct", "inactive_days"
    ]
    
    X_input = df[REQUIRED_RAW_FEATURES]
    X_processed = explainer.get_preprocessed_df(X_input)
    probs = explainer.model.predict_proba(X_processed)[:, 1]
    
    df["churn_probability"] = probs
    df["Risk Score (%)"] = (probs * 100).round(1)
    df["Risk Level"] = np.where(probs >= 0.7, "🔴 Critical", np.where(probs >= 0.3, "⚠️ Elevated", "🟢 Low"))
    
    if "churn" not in df.columns or df["churn"].isna().all():
        df["churn"] = (probs >= 0.5).astype(int)
    else:
        df["churn"] = df["churn"].fillna(0).astype(int)
        
    # Take a sample for fast test run
    filtered_df = df.head(100).copy()
    
    # 4. Generate overall summary
    total_customers = len(filtered_df)
    overall_churn_rate = (filtered_df["churn"].mean() * 100) if total_customers > 0 else 0.0
    high_risk_revenue = filtered_df.loc[filtered_df["churn_probability"] >= 0.5, "avg_recharge_amount_npr"].sum()
    
    overall_summary = {
        "total_customers": total_customers,
        "overall_churn_rate": overall_churn_rate,
        "model_accuracy": 0.85,
        "high_risk_revenue": high_risk_revenue,
    }
    
    # 5. Run report generation
    print("Generating PDF report...")
    pdf_path = generate_report_pdf(overall_summary, filtered_df, explainer)
    print("Test completed successfully!")
    print("Generated PDF at:", os.path.abspath(pdf_path))

if __name__ == "__main__":
    run_test()
