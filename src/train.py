import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import joblib

# Import preprocessing steps
from preprocessing import preprocess_and_save

def train_and_evaluate(data_path="data/nepal_telecom_churn_main.csv", models_dir="models", random_state=42):
    """Trains tuned XGBoost Classifier with SMOTE & SMOTETomek comparisons and exports artifacts."""
    os.makedirs(models_dir, exist_ok=True)
    
    print("==================================================")
    print("STEP 1: Stratified Split & Feature Preprocessing")
    print("==================================================")
    X_train, X_test, y_train, y_test, X_train_raw, X_test_raw, feature_names = preprocess_and_save(
        data_path, models_dir=models_dir, random_state=random_state
    )
    
    print(f"Training fold shape: {X_train.shape}")
    print(f"Test fold shape:     {X_test.shape}")
    
    # ---------------------------------------------------------
    # RESAMPLING: SMOTE vs SMOTETomek
    # ---------------------------------------------------------
    print("\n==================================================")
    print("STEP 2: Resampling Training Set (SMOTE vs SMOTETomek)")
    print("==================================================")
    
    train_counts_orig = y_train.value_counts().to_dict()
    print(f"Original Training Class Counts: Retained (0) = {train_counts_orig.get(0, 0)}, Churned (1) = {train_counts_orig.get(1, 0)}")
    
    # 1. SMOTE
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    X_train_smote = pd.DataFrame(X_train_smote, columns=feature_names)
    smote_counts = pd.Series(y_train_smote).value_counts().to_dict()
    print(f"After SMOTE:                   Retained (0) = {smote_counts.get(0, 0)}, Churned (1) = {smote_counts.get(1, 0)}")
    
    # 2. SMOTETomek
    smote_tomek = SMOTETomek(random_state=random_state)
    X_train_tomek, y_train_tomek = smote_tomek.fit_resample(X_train, y_train)
    X_train_tomek = pd.DataFrame(X_train_tomek, columns=feature_names)
    tomek_counts = pd.Series(y_train_tomek).value_counts().to_dict()
    print(f"After SMOTETomek:              Retained (0) = {tomek_counts.get(0, 0)}, Churned (1) = {tomek_counts.get(1, 0)}")
    
    # ---------------------------------------------------------
    # HYPERPARAMETER TUNING via RandomizedSearchCV
    # ---------------------------------------------------------
    print("\n==================================================")
    print("STEP 3: Hyperparameter Tuning via RandomizedSearchCV")
    print("==================================================")
    
    base_model = XGBClassifier(
        random_state=random_state,
        eval_metric="logloss"
    )
    
    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 1.0],
        'reg_lambda': [1.0, 5.0, 10.0]
    }
    
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring='roc_auc',
        cv=3,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    print("Tuning hyperparameters on SMOTE-resampled training data...")
    search.fit(X_train_smote, y_train_smote)
    
    best_model_smote = search.best_estimator_
    best_params = search.best_params_
    best_cv_auc = search.best_score_
    
    print("\n--------------------------------------------------")
    print("BEST HYPERPARAMETERS (SMOTE)")
    print("--------------------------------------------------")
    for param, val in sorted(best_params.items()):
        print(f"  {param:<20}: {val}")
    print(f"\nBest 3-Fold CV ROC-AUC: {best_cv_auc:.4f}")
    
    # Evaluate SMOTETomek model with the same search space for direct comparison
    print("\nTuning hyperparameters on SMOTETomek-resampled training data for comparison...")
    search_tomek = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring='roc_auc',
        cv=3,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )
    search_tomek.fit(X_train_tomek, y_train_tomek)
    best_model_tomek = search_tomek.best_estimator_
    
    # ---------------------------------------------------------
    # EVALUATION ON UNTOUCHED TEST SET
    # ---------------------------------------------------------
    print("\n==================================================")
    print("STEP 4: Model Evaluation on Untouched Test Set")
    print("==================================================")
    
    def eval_model(model, X_t, y_t):
        y_pred = model.predict(X_t)
        y_prob = model.predict_proba(X_t)[:, 1]
        cm = confusion_matrix(y_t, y_pred)
        return {
            "accuracy": float(accuracy_score(y_t, y_pred)),
            "precision": float(precision_score(y_t, y_pred)),
            "recall": float(recall_score(y_t, y_pred)),
            "f1_score": float(f1_score(y_t, y_pred)),
            "roc_auc": float(roc_auc_score(y_t, y_prob)),
            "average_precision": float(average_precision_score(y_t, y_prob)),
            "confusion_matrix": {
                "tn": int(cm[0][0]),
                "fp": int(cm[0][1]),
                "fn": int(cm[1][0]),
                "tp": int(cm[1][1])
            },
            "y_pred": y_pred.tolist(),
            "y_prob": y_prob.tolist()
        }
    
    metrics_smote = eval_model(best_model_smote, X_test, y_test)
    metrics_tomek = eval_model(best_model_tomek, X_test, y_test)
    
    print("\n### Performance Comparison: SMOTE vs. SMOTETomek ###")
    print(f"{'Metric':<20} | {'SMOTE':<12} | {'SMOTETomek':<12}")
    print("-" * 50)
    print(f"{'Accuracy':<20} | {metrics_smote['accuracy']:<12.4f} | {metrics_tomek['accuracy']:<12.4f}")
    print(f"{'Precision':<20} | {metrics_smote['precision']:<12.4f} | {metrics_tomek['precision']:<12.4f}")
    print(f"{'Recall':<20} | {metrics_smote['recall']:<12.4f} | {metrics_tomek['recall']:<12.4f}")
    print(f"{'F1-Score':<20} | {metrics_smote['f1_score']:<12.4f} | {metrics_tomek['f1_score']:<12.4f}")
    print(f"{'ROC-AUC':<20} | {metrics_smote['roc_auc']:<12.4f} | {metrics_tomek['roc_auc']:<12.4f}")
    print(f"{'Avg Precision (PR)':<20} | {metrics_smote['average_precision']:<12.4f} | {metrics_tomek['average_precision']:<12.4f}")
    
    print("\nClassification Report (SMOTE Pipeline):")
    print(classification_report(y_test, best_model_smote.predict(X_test)))
    
    print("Confusion Matrix (Raw Counts - SMOTE Pipeline):")
    cm_s = metrics_smote['confusion_matrix']
    print(f"  [[TN: {cm_s['tn']}, FP: {cm_s['fp']}], [FN: {cm_s['fn']}, TP: {cm_s['tp']}]]")
    
    # ---------------------------------------------------------
    # SAVE DELIVERABLES
    # ---------------------------------------------------------
    print("\n==================================================")
    print("STEP 5: Saving Pipeline Artifacts & Metrics")
    print("==================================================")
    
    # Load preprocessor artifact for single pipeline bundle
    preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.joblib"))
    
    # 1. Full pipeline via joblib
    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", best_model_smote)
    ])
    pipeline_path = "churn_xgb_pipeline.joblib"
    joblib.dump(full_pipeline, pipeline_path)
    print(f"Full pipeline saved to '{pipeline_path}'")
    
    # 2. Native model format for explainability app
    model_json_path = os.path.join(models_dir, "xgboost_churn_model.json")
    best_model_smote.save_model(model_json_path)
    print(f"XGBoost JSON model saved to '{model_json_path}'")
    
    # 3. Export metrics JSON & CSV
    cv_results = search.cv_results_
    cv_auc_scores = cv_results["mean_test_score"].tolist()
    
    metrics_summary = {
        "best_cv_roc_auc": float(best_cv_auc),
        "best_hyperparameters": best_params,
        "class_counts": {
            "original_train": train_counts_orig,
            "smote_train": smote_counts,
            "smotetomek_train": tomek_counts
        },
        "smote": {
            "accuracy": metrics_smote["accuracy"],
            "precision": metrics_smote["precision"],
            "recall": metrics_smote["recall"],
            "f1_score": metrics_smote["f1_score"],
            "roc_auc": metrics_smote["roc_auc"],
            "average_precision": metrics_smote["average_precision"],
            "confusion_matrix": metrics_smote["confusion_matrix"]
        },
        "smotetomek": {
            "accuracy": metrics_tomek["accuracy"],
            "precision": metrics_tomek["precision"],
            "recall": metrics_tomek["recall"],
            "f1_score": metrics_tomek["f1_score"],
            "roc_auc": metrics_tomek["roc_auc"],
            "average_precision": metrics_tomek["average_precision"],
            "confusion_matrix": metrics_tomek["confusion_matrix"]
        },
        "cv_results_auc_scores": cv_auc_scores,
        "feature_names": feature_names
    }
    
    metrics_json_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Metrics saved to '{metrics_json_path}'")
    
    metrics_df = pd.DataFrame([
        {"Method": "SMOTE", **metrics_smote},
        {"Method": "SMOTETomek", **metrics_tomek}
    ]).drop(columns=["y_pred", "y_prob", "confusion_matrix"])
    metrics_csv_path = os.path.join(models_dir, "metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Metrics CSV saved to '{metrics_csv_path}'")
    
    # Save preprocessed matrices for downstream figure generation
    np.savez_compressed(
        os.path.join(models_dir, "processed_data.npz"),
        X_train_orig=X_train.values,
        y_train_orig=y_train.values,
        X_train_smote=X_train_smote.values if isinstance(X_train_smote, pd.DataFrame) else X_train_smote,
        y_train_smote=y_train_smote.values if isinstance(y_train_smote, pd.Series) else y_train_smote,
        X_test=X_test.values,
        y_test=y_test.values,
        feature_names=feature_names
    )
    print("Processed datasets cached for figure generation.")
    
    return best_model_smote, full_pipeline, metrics_summary

if __name__ == "__main__":
    train_and_evaluate()
