import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import RandomizedSearchCV
from imblearn.combine import SMOTETomek
import joblib

# Import preprocessing stepsss
from preprocessing import preprocess_and_save

def train_model(data_path, models_dir="models", random_state=42):
    """Trains the XGBoost Classifier and saves it along with evaluation metrics."""
    os.makedirs(models_dir, exist_ok=True)
    
    print("Step 1: Preprocessing raw dataset...")
    X_train, X_test, y_train, y_test, X_train_raw, X_test_raw, feature_names = preprocess_and_save(
        data_path, models_dir=models_dir, random_state=random_state
    )
    
    print("\nStep 2: Training XGBoost Classifier...")
    # Apply SMOTETomek to balance training classes
    print("Applying SMOTETomek to balance training classes...")
    smote_tomek = SMOTETomek(random_state=random_state)
    X_train_res, y_train_res = smote_tomek.fit_resample(X_train, y_train)
    print(f"Resampled training set shape: {X_train_res.shape} (original: {X_train.shape})")
    
    # Run hyperparameter search to control overfitting and optimize performance
    print("Optimizing hyperparameters with RandomizedSearchCV...")
    base_model = XGBClassifier(
        random_state=random_state,
        eval_metric="logloss"
    )
    
    # Define hyperparameter grid for robust tuning
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'gamma': [0.0, 0.1, 0.2],
        'reg_alpha': [0.0, 0.1, 1.0],      # L1 regularization
        'reg_lambda': [1.0, 5.0, 10.0]     # L2 regularization
    }
    
    # Perform 3-fold cross validation search with 20 iterations
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring='roc_auc',
        cv=3,
        random_state=random_state,
        n_jobs=1,
        verbose=1
    )
    
    search.fit(X_train_res, y_train_res)
    model = search.best_estimator_
    
    print(f"Best hyperparameters found: {search.best_params_}")
    print(f"Best cross-validation ROC-AUC score: {search.best_score_:.4f}")
    
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
    
    # Save metrics and class information for the dashboards
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
        print(f"Generating data first....")
        from data_generator import generate_telecom_data
        os.makedirs("data", exist_ok=True)
        df = generate_telecom_data(num_customers=5000)
        df.to_csv(data_path, index=False)
        
    train_model(data_path)
