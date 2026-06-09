import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Define feature columns
NUMERICAL_COLS = [
    "age", "tenure_days", "calls_min_30d", "sms_count_30d", "data_gb_30d", 
    "night_usage_pct", "last_recharge_days_ago", "avg_recharge_amount_npr", 
    "recharge_count_30d", "signal_strength_dbm", "call_drop_rate", 
    "avg_data_speed_mbps", "num_complaints_30d", "avg_resolution_time_hours", 
    "usage_drop_pct", "recharge_drop_pct", "inactive_days",
    # Engineered Features
    "calls_per_day", "data_gb_per_day", "recharges_per_day",
    "avg_recharge_per_transaction", "complaint_density", "call_drop_severity",
    "total_active_packs", "churn_risk_interaction"
]

CATEGORICAL_COLS = [
    "gender", "province", "district_type", "sim_type"
]

BINARY_COLS = [
    "data_pack_active", "voice_pack_active", "vas_active", "roaming_active"
]

def load_data(file_path):
    """Loads dataset from CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    return pd.read_csv(file_path)

def engineer_features(df):
    """Generates interaction and derived features to capture non-linear relationships."""
    df_out = df.copy()
    
    # 1. Usage intensity
    df_out["calls_per_day"] = df_out["calls_min_30d"] / 30.0
    df_out["data_gb_per_day"] = df_out["data_gb_30d"] / 30.0
    df_out["recharges_per_day"] = df_out["recharge_count_30d"] / 30.0
    
    # 2. Recharge efficiency
    df_out["avg_recharge_per_transaction"] = df_out["avg_recharge_amount_npr"] / (df_out["recharge_count_30d"] + 1.0)
    
    # 3. Quality / complaint density
    df_out["complaint_density"] = df_out["num_complaints_30d"] * df_out["avg_resolution_time_hours"]
    df_out["call_drop_severity"] = df_out["call_drop_rate"] * df_out["num_complaints_30d"]
    
    # 4. Total services active
    df_out["total_active_packs"] = (
        df_out["data_pack_active"] + 
        df_out["voice_pack_active"] + 
        df_out["vas_active"] + 
        df_out["roaming_active"]
    ).astype(float)
    
    # 5. Combined risk interaction
    df_out["churn_risk_interaction"] = df_out["usage_drop_pct"] * df_out["recharge_drop_pct"] * df_out["inactive_days"]
    
    return df_out

def create_preprocessing_pipeline():
    """Creates a ColumnTransformer pipeline for feature scaling and encoding."""
    
    # Preprocessing for numerical features: impute missing values with median, then scale
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Preprocessing for categorical features: impute missing values with mode, then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # For binary columns, we just impute missing values with 0 if any (no scaling needed)
    binary_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0))
    ])
    
    # Combine transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
            ("bin", binary_transformer, BINARY_COLS)
        ],
        remainder="drop" # drop customer_id and target churn
    )
    
    return preprocessor

def preprocess_and_save(file_path, models_dir="models", test_size=0.2, random_state=42):
    """Loads raw data, fits preprocessor on train set, transforms data, and saves preprocessor."""
    os.makedirs(models_dir, exist_ok=True)
    
    # Load dataset
    df = load_data(file_path)
    
    # Run feature engineering on the raw dataframe
    df = engineer_features(df)
    
    # Split into features (X) and target (y)
    X = df.drop(columns=["churn"])
    y = df["churn"]
    
    # Stratified split to prevent data leakage and preserve class distribution
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Initialize and fit pipeline
    preprocessor = create_preprocessing_pipeline()
    
    # Fit preprocessor on training data only to avoid data leakage
    preprocessor.fit(X_train_raw)
    
    # Transform both training and test datasets
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    # Get feature names after transformation (useful for modeling and interpretability)
    # Numerical features stay the same
    num_features = NUMERICAL_COLS
    
    # One-hot encoded features
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_features = list(cat_encoder.get_feature_names_out(CATEGORICAL_COLS))
    
    # Binary features stay the same
    bin_features = BINARY_COLS
    
    feature_names = num_features + cat_features + bin_features
    
    # Save the fitted preprocessor
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Fitted preprocessor saved successfully to {preprocessor_path}")
    
    # Convert numpy arrays back to Pandas DataFrames/Series for easier down-stream tasks
    X_train_df = pd.DataFrame(X_train, columns=feature_names, index=X_train_raw.index)
    X_test_df = pd.DataFrame(X_test, columns=feature_names, index=X_test_raw.index)
    
    return X_train_df, X_test_df, y_train, y_test, X_train_raw, X_test_raw, feature_names

if __name__ == "__main__":
    # Test execution
    data_path = os.path.join("data", "telecom_churn_nepal.csv")
    if not os.path.exists(data_path):
        print(f"Generating data first...")
        from data_generator import generate_telecom_data
        os.makedirs("data", exist_ok=True)
        df = generate_telecom_data(num_customers=5000)
        df.to_csv(data_path, index=False)
        
    X_train, X_test, y_train, y_test, X_train_raw, X_test_raw, feature_names = preprocess_and_save(data_path)
    print(f"Preprocessing completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Number of preprocessed features: {len(feature_names)}")
