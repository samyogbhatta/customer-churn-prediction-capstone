import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix

# Set publication style
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.style.use('seaborn-v0_8-whitegrid')

OUTPUT_DIR = os.path.join("outputs", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(fig, filename_base):
    """Saves figure in both 300 DPI PNG and vector PDF formats."""
    png_path = os.path.join(OUTPUT_DIR, f"{filename_base}.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"{filename_base}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filename_base}.png & {filename_base}.pdf")

def generate_all_figures():
    models_dir = "models"
    
    if not os.path.exists(os.path.join(models_dir, "processed_data.npz")):
        raise FileNotFoundError("Run src/train.py first to cache processed data.")
        
    data = np.load(os.path.join(models_dir, "processed_data.npz"), allow_pickle=True)
    X_train_orig = data["X_train_orig"]
    y_train_orig = data["y_train_orig"]
    X_train_smote = data["X_train_smote"]
    y_train_smote = data["y_train_smote"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    feature_names = list(data["feature_names"])
    
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    with open(os.path.join(models_dir, "metrics.json"), "r") as f:
        metrics = json.load(f)
        
    model = XGBClassifier()
    model.load_model(os.path.join(models_dir, "xgboost_churn_model.json"))
    
    y_prob = model.predict_proba(X_test_df)[:, 1]
    y_pred = model.predict(X_test_df)
    
    print("\n--------------------------------------------------")
    print("GENERATING BOTH COMBINED AND INDIVIDUAL SEPARATE FIGURES")
    print("--------------------------------------------------")

    # =========================================================================
    # 1. COMBINED MODEL PERFORMANCE DASHBOARD (4-Panel Unified Figure)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Model Evaluation & Performance Dashboard (Untouched Test Set)", fontsize=16, fontweight='bold', y=0.98)

    # Panel A: ROC Curve
    ax_roc = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = metrics["smote"]["roc_auc"]
    ax_roc.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'XGBoost + SMOTE (AUC = {auc_val:.4f})')
    ax_roc.plot([0, 1], [0, 1], color='#888888', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.5000)')
    ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    ax_roc.set_ylabel('True Positive Rate (Recall)', fontsize=11, fontweight='bold')
    ax_roc.set_title('(A) Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold', pad=10)
    ax_roc.legend(loc="lower right", frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax_roc.set_xlim([-0.01, 1.01])
    ax_roc.set_ylim([-0.01, 1.01])

    # Panel B: Precision-Recall Curve
    ax_pr = axes[0, 1]
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_prob)
    ap_val = metrics["smote"]["average_precision"]
    baseline_prec = np.sum(y_test == 1) / len(y_test)
    ax_pr.plot(rec_vals, prec_vals, color='#d95f02', lw=2.5, label=f'XGBoost + SMOTE (AP = {ap_val:.4f})')
    ax_pr.axhline(y=baseline_prec, color='#888888', lw=1.5, linestyle='--', label=f'Baseline ({baseline_prec:.1%})')
    ax_pr.set_xlabel('Recall', fontsize=11, fontweight='bold')
    ax_pr.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax_pr.set_title('(B) Precision-Recall Curve', fontsize=12, fontweight='bold', pad=10)
    ax_pr.legend(loc="lower left", frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax_pr.set_xlim([-0.01, 1.01])
    ax_pr.set_ylim([-0.01, 1.01])

    # Panel C: Confusion Matrix Heatmap
    ax_cm = axes[1, 0]
    cm = metrics["smote"]["confusion_matrix"]
    cm_arr = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    cm_norm = cm_arr.astype('float') / cm_arr.sum(axis=1)[:, np.newaxis]
    labels = np.array([[f"{cm_arr[0,0]}\n({cm_norm[0,0]:.1%})", f"{cm_arr[0,1]}\n({cm_norm[0,1]:.1%})"],
                       [f"{cm_arr[1,0]}\n({cm_norm[1,0]:.1%})", f"{cm_arr[1,1]}\n({cm_norm[1,1]:.1%})"]])
    sns.heatmap(cm_arr, annot=labels, fmt="", cmap="Blues", cbar=False,
                xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"], ax=ax_cm,
                annot_kws={"size": 12, "weight": "bold"})
    ax_cm.set_xlabel("Predicted Label", fontsize=11, fontweight='bold')
    ax_cm.set_ylabel("True Label", fontsize=11, fontweight='bold')
    ax_cm.set_title("(C) Confusion Matrix (Raw Counts & %)", fontsize=12, fontweight='bold', pad=10)

    # Panel D: Performance Metrics Comparison (Accuracy, Precision, Recall, F1, ROC-AUC)
    ax_metrics = axes[1, 1]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    m_smote = [metrics['smote']['accuracy'], metrics['smote']['precision'], metrics['smote']['recall'], metrics['smote']['f1_score'], metrics['smote']['roc_auc']]
    m_tomek = [metrics['smotetomek']['accuracy'], metrics['smotetomek']['precision'], metrics['smotetomek']['recall'], metrics['smotetomek']['f1_score'], metrics['smotetomek']['roc_auc']]
    
    x_m = np.arange(len(metric_names))
    width_m = 0.35
    r1 = ax_metrics.bar(x_m - width_m/2, m_smote, width_m, label='SMOTE', color='#2b5c8f')
    r2 = ax_metrics.bar(x_m + width_m/2, m_tomek, width_m, label='SMOTETomek', color='#d95f02')
    
    ax_metrics.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax_metrics.set_title('(D) Key Metrics Summary (SMOTE vs SMOTETomek)', fontsize=12, fontweight='bold', pad=10)
    ax_metrics.set_xticks(x_m)
    ax_metrics.set_xticklabels(metric_names, fontsize=10, fontweight='bold')
    ax_metrics.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    ax_metrics.set_ylim([0.65, 1.02])
    
    for r in r1 + r2:
        h = r.get_height()
        ax_metrics.annotate(f'{h:.3f}',
                            xy=(r.get_x() + r.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig(fig, "01_model_performance_dashboard")

    # =========================================================================
    # 2. STANDALONE SEPARATE FIGURES
    # =========================================================================

    # 2A. Standalone ROC Curve
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'XGBoost + SMOTE (AUC = {auc_val:.4f})')
    ax.plot([0, 1], [0, 1], color='#888888', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.5000)')
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc="lower right", frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    save_fig(fig, "02_roc_curve")

    # 2B. Standalone Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(rec_vals, prec_vals, color='#d95f02', lw=2.5, label=f'XGBoost + SMOTE (AP = {ap_val:.4f})')
    ax.axhline(y=baseline_prec, color='#888888', lw=1.5, linestyle='--', label=f'Baseline ({baseline_prec:.1%})')
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc="lower left", frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    save_fig(fig, "03_precision_recall_curve")

    # 2C. Standalone Confusion Matrix
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm_arr, annot=labels, fmt="", cmap="Blues", cbar=False,
                xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"], ax=ax,
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight='bold')
    ax.set_ylabel("True Label", fontsize=12, fontweight='bold')
    ax.set_title("Confusion Matrix (Test Set Counts & %)", fontsize=13, fontweight='bold', pad=12)
    save_fig(fig, "04_confusion_matrix")

    # 2D. Standalone Metrics Comparison Diagram (F1-Score, Accuracy, Precision, Recall, ROC-AUC)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    rects1 = ax.bar(x_m - width_m/2, m_smote, width_m, label='SMOTE', color='#2b5c8f')
    rects2 = ax.bar(x_m + width_m/2, m_tomek, width_m, label='SMOTETomek', color='#d95f02')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Metrics (Accuracy, Precision, Recall, F1, ROC-AUC)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x_m)
    ax.set_xticklabels(metric_names, fontsize=11, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
    ax.set_ylim([0.65, 1.02])
    for r in rects1 + rects2:
        h = r.get_height()
        ax.annotate(f'{h:.4f}', xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
    save_fig(fig, "05_f1_and_metrics_comparison")

    # 2E. Standalone Class Distribution (Before vs After SMOTE)
    fig, ax = plt.subplots(figsize=(7, 5))
    categories = ['Retained (0)', 'Churned (1)']
    orig_counts = [np.sum(y_train_orig == 0), np.sum(y_train_orig == 1)]
    smote_counts = [np.sum(y_train_smote == 0), np.sum(y_train_smote == 1)]
    
    x = np.arange(len(categories))
    width = 0.35
    r_orig = ax.bar(x - width/2, orig_counts, width, label='Original Training Fold', color='#2b5c8f')
    r_smote = ax.bar(x + width/2, smote_counts, width, label='After SMOTE Resampling', color='#d95f02')
    
    ax.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
    ax.set_title('Training Fold Class Distribution Before vs. After SMOTE', fontsize=13, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    for rect in r_orig + r_smote:
        h = rect.get_height()
        ax.annotate(f'{int(h):,}', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=10)
    ax.set_ylim(0, max(smote_counts) * 1.15)
    save_fig(fig, "06_class_distribution_smote")

    # 2F. Standalone Correlation Heatmap of Engineered Features
    from preprocessing import load_data, engineer_features
    df_raw = load_data(os.path.join("data", "nepal_telecom_churn_main.csv"))
    df_eng = engineer_features(df_raw)
    
    eng_cols = [
        "calls_per_day", "data_gb_per_day", "recharges_per_day",
        "avg_recharge_per_transaction", "complaint_density", "call_drop_severity",
        "total_active_packs", "churn_risk_interaction"
    ]
    corr_matrix = df_eng[eng_cols].corr()
    
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="vlag",
                vmin=-1, vmax=1, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Heatmap of Engineered Features", fontsize=13, fontweight='bold', pad=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    save_fig(fig, "07_correlation_heatmap_engineered_features")

    # 2G. Standalone Feature Importance (Gain-Based)
    booster = model.get_booster()
    importance_gain = booster.get_score(importance_type='gain')
    feat_imp_dict = {}
    for k, v in importance_gain.items():
        if k.startswith('f') and k[1:].isdigit():
            idx = int(k[1:])
            feat_imp_dict[feature_names[idx]] = v
        else:
            feat_imp_dict[k] = v
            
    imp_series = pd.Series(feat_imp_dict).sort_values(ascending=True).tail(15)
    
    fig, ax = plt.subplots(figsize=(8.5, 6))
    imp_series.plot(kind='barh', color='#2b5c8f', ax=ax, width=0.7)
    ax.set_xlabel('Gain (Average Loss Reduction per Split)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 XGBoost Feature Importances (Gain)', fontsize=13, fontweight='bold', pad=12)
    save_fig(fig, "08_xgboost_feature_importance")

    # Compute SHAP Values on representative test sample (~400 rows)
    explainer = shap.TreeExplainer(model)
    sample_size = min(400, len(X_test_df))
    X_sample = X_test_df.iloc[:sample_size]
    shap_obj = explainer(X_sample)
    
    shap_vals = shap_obj.values
    if len(shap_vals.shape) == 3:
        shap_vals = shap_vals[:, :, 1]

    # 2H. Standalone SHAP Beeswarm Summary Plot
    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_vals, X_sample, max_display=15, show=False)
    fig = plt.gcf()
    plt.title("SHAP Global Summary (Beeswarm)", fontsize=13, fontweight='bold', pad=12)
    save_fig(fig, "09_shap_summary_beeswarm")

    # 2I. Standalone SHAP Mean |Value| Bar Plot
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_vals, X_sample, max_display=15, plot_type="bar", show=False)
    fig = plt.gcf()
    plt.title("Global Feature Importance (Mean |SHAP Value|)", fontsize=13, fontweight='bold', pad=12)
    save_fig(fig, "10_shap_mean_bar")

    # 2J. Standalone SHAP Local Waterfall Plots
    high_risk_idx = np.where(y_prob > 0.7)[0]
    low_risk_idx = np.where(y_prob < 0.2)[0]
    
    target_high = high_risk_idx[0] if len(high_risk_idx) > 0 else np.argmax(y_prob)
    target_low = low_risk_idx[0] if len(low_risk_idx) > 0 else np.argmin(y_prob)
    
    fig = plt.figure(figsize=(8.5, 6))
    shap.waterfall_plot(shap_obj[target_high], max_display=10, show=False)
    plt.title(f"SHAP Local Waterfall: High-Risk Customer (Prob = {y_prob[target_high]:.1%})", fontsize=12, fontweight='bold', pad=12)
    save_fig(fig, "11_shap_waterfall_high_risk")

    fig = plt.figure(figsize=(8.5, 6))
    shap.waterfall_plot(shap_obj[target_low], max_display=10, show=False)
    plt.title(f"SHAP Local Waterfall: Low-Risk Customer (Prob = {y_prob[target_low]:.1%})", fontsize=12, fontweight='bold', pad=12)
    save_fig(fig, "12_shap_waterfall_low_risk")

    # 2K. Standalone SHAP Dependence Plots
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    top_3_indices = np.argsort(mean_abs_shap)[::-1][:3]
    top_3_features = [feature_names[i] for i in top_3_indices]
    
    for idx_f, feat_name in enumerate(top_3_features, start=1):
        fig = plt.figure(figsize=(8, 5))
        shap.dependence_plot(feat_name, shap_vals, X_sample, show=False)
        plt.title(f"SHAP Dependence Plot: {feat_name}", fontsize=13, fontweight='bold', pad=12)
        save_fig(fig, f"13_shap_dependence_{idx_f}_{feat_name}")

    # 2L. Standalone Hyperparameter Search Optimization Trajectory
    cv_scores = metrics.get("cv_results_auc_scores", [])
    if cv_scores:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', color='#1f77b4', linewidth=2, markersize=6)
        ax.axhline(y=max(cv_scores), color='#d95f02', linestyle='--', label=f'Best CV AUC: {max(cv_scores):.4f}')
        ax.set_xlabel('RandomizedSearchCV Candidate Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('3-Fold Cross-Validation ROC-AUC', fontsize=12, fontweight='bold')
        ax.set_title('Hyperparameter Optimization Trajectory (20 Candidates)', fontsize=13, fontweight='bold', pad=12)
        ax.set_xticks(range(1, len(cv_scores) + 1))
        ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=11)
        save_fig(fig, "14_hyperparameter_search_results")

    print("\nALL COMBINED AND SEPARATE INDIVIDUAL FIGURES SUCCESSFULLY GENERATED AND SAVED TO outputs/figures/")

if __name__ == "__main__":
    generate_all_figures()
