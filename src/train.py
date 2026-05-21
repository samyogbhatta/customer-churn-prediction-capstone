import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import joblib

# Import preprocessing steps
from preprocessing import preprocess_and_save

def train_model(data_path, models_dir="models", random_state=42):
    """Trains the XGBoost Classifier and saves it along with evaluation metrics."""
    os.makedirs(models_dir, exist_ok=True)
    
    print("Step 1: Preprocessing raw dataset...")
    X_train, X_test, y_train, y_test, X_train_raw, X_test_raw, feature_names = preprocess_and_save(
        data_path, models_dir=models_dir, random_state=random_state
    )
    
    print("\nStep 2: Training XGBoost Classifier...")
    # Calculate scale_pos_weight to handle potential class imbalance (negative count / positive count)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    # Tuned hyperparameters for robust performance and reducing overfitting
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="logloss",
        use_label_encoder=False
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    
    print("\nStep 3: Evaluating Model Performance...")
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Save the model in JSON format (native XGBoost format, portable and clean)
    model_path = os.path.join(models_dir, "xgboost_churn_model.json")
    model.save_model(model_path)
    print(f"\nTrained model saved successfully to {model_path}")
    
    # Save metrics and class information for the dashboard
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": {
            "tn": int(cm[0][0]),
            "fp": int(cm[0][1]),
            "fn": int(cm[1][0]),
            "tp": int(cm[1][1])
        },
        "feature_names": feature_names
    }
    
    metrics_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Model evaluation metrics saved to {metrics_path}")
    
    return model

if __name__ == "__main__":
    data_path = os.path.join("data", "telecom_churn_nepal.csv")
    if not os.path.exists(data_path):
        print(f"Generating data first...")
        from data_generator import generate_telecom_data
        os.makedirs("data", exist_ok=True)
        df = generate_telecom_data(num_customers=5000)
        df.to_csv(data_path, index=False)
        
    train_model(data_path)
