import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Core feature definitions for the Nepal Telecom Churn dataset schema
NUMERICAL_COLS = [
    "age", "tenure_days", "calls_min_30d", "sms_count_30d", "data_gb_30d", 
    "night_usage_pct", "last_recharge_days_ago", "avg_recharge_amount_npr", 
    "recharge_count_30d", "monthly_bill_npr", "signal_strength_dbm", "call_drop_rate", 
    "avg_data_speed_mbps", "num_complaints_30d", "avg_resolution_time_hours", 
    "usage_drop_pct", "recharge_drop_pct", "inactive_days",
    # Engineered Features
    "calls_per_day", "data_gb_per_day", "recharges_per_day",
    "avg_recharge_per_transaction", "complaint_density", "call_drop_severity",
    "total_active_packs", "churn_risk_interaction"
]

CATEGORICAL_COLS = [
    "gender", "province", "district_type", "sim_type", "recharge_segment"
]

BINARY_COLS = [
    "data_pack_active", "voice_pack_active", "vas_active", "roaming_active"
]

def load_data(file_path):
    """Loads dataset from CSV file safely."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    return pd.read_csv(file_path)

def sanitize_and_align_columns(df):
    """Aligns uploaded dataset column names and fills missing expected columns with sensible defaults."""
    df_out = df.copy()
    
    # Case-insensitive column matching
    col_map = {c.lower().strip(): c for c in df_out.columns}
    
    # Map common column aliases if uploaded
    aliases = {
        "cust_id": "customer_id",
        "id": "customer_id",
        "district_name": "district",
        "recharge_amt": "avg_recharge_amount_npr",
        "monthly_bill": "monthly_bill_npr"
    }
    
    rename_dict = {}
    for alias, target in aliases.items():
        if alias in col_map and target not in df_out.columns:
            rename_dict[col_map[alias]] = target
    if rename_dict:
        df_out = df_out.rename(columns=rename_dict)
        
    # Ensure all required raw numerical columns exist; if missing in an uploaded file, fill with default 0
    raw_numerics = [
        "age", "tenure_days", "calls_min_30d", "sms_count_30d", "data_gb_30d", 
        "night_usage_pct", "last_recharge_days_ago", "avg_recharge_amount_npr", 
        "recharge_count_30d", "monthly_bill_npr", "signal_strength_dbm", "call_drop_rate", 
        "avg_data_speed_mbps", "num_complaints_30d", "avg_resolution_time_hours", 
        "usage_drop_pct", "recharge_drop_pct", "inactive_days"
    ]
    for col in raw_numerics:
        if col not in df_out.columns:
            df_out[col] = 0.0
            
    # Binary columns default to 0
    for col in BINARY_COLS:
        if col not in df_out.columns:
            df_out[col] = 0
            
    # Categorical columns default to mode/Unknown
    for col in CATEGORICAL_COLS:
        if col not in df_out.columns:
            df_out[col] = "Unknown"
            
    return df_out

def engineer_features(df):
    """Generates interaction and derived features safely, avoiding divide-by-zero errors."""
    df_out = sanitize_and_align_columns(df)
    
    # 1. Usage intensity
    df_out["calls_per_day"] = df_out["calls_min_30d"].astype(float) / 30.0
    df_out["data_gb_per_day"] = df_out["data_gb_30d"].astype(float) / 30.0
    df_out["recharges_per_day"] = df_out["recharge_count_30d"].astype(float) / 30.0
    
    # 2. Recharge efficiency (safe division using +1)
    df_out["avg_recharge_per_transaction"] = df_out["avg_recharge_amount_npr"].astype(float) / (df_out["recharge_count_30d"].astype(float) + 1.0)
    
    # 3. Quality / complaint density
    df_out["complaint_density"] = df_out["num_complaints_30d"].astype(float) * df_out["avg_resolution_time_hours"].astype(float)
    df_out["call_drop_severity"] = df_out["call_drop_rate"].astype(float) * df_out["num_complaints_30d"].astype(float)
    
    # 4. Total services active
    df_out["total_active_packs"] = (
        df_out["data_pack_active"].astype(float) + 
        df_out["voice_pack_active"].astype(float) + 
        df_out["vas_active"].astype(float) + 
        df_out["roaming_active"].astype(float)
    )
    
    # 5. Combined risk interaction
    df_out["churn_risk_interaction"] = (
        df_out["usage_drop_pct"].astype(float) * 
        df_out["recharge_drop_pct"].astype(float) * 
        df_out["inactive_days"].astype(float)
    )
    
    return df_out

def create_preprocessing_pipeline():
    """Creates a ColumnTransformer pipeline for feature imputing and categorical one-hot encoding."""
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

# Core feature definitions for the Nepal Telecom Churn dataset schema
NUMERICAL_COLS = [
    "age", "tenure_days", "calls_min_30d", "sms_count_30d", "data_gb_30d", 
    "night_usage_pct", "last_recharge_days_ago", "avg_recharge_amount_npr", 
    "recharge_count_30d", "monthly_bill_npr", "signal_strength_dbm", "call_drop_rate", 
    "avg_data_speed_mbps", "num_complaints_30d", "avg_resolution_time_hours", 
    "usage_drop_pct", "recharge_drop_pct", "inactive_days",
    # Engineered Features
    "calls_per_day", "data_gb_per_day", "recharges_per_day",
    "avg_recharge_per_transaction", "complaint_density", "call_drop_severity",
    "total_active_packs", "churn_risk_interaction"
]

CATEGORICAL_COLS = [
    "gender", "province", "district_type", "sim_type", "recharge_segment"
]

BINARY_COLS = [
    "data_pack_active", "voice_pack_active", "vas_active", "roaming_active"
]

def load_data(file_path):
    """Loads dataset from CSV file safely."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    return pd.read_csv(file_path)

def sanitize_and_align_columns(df):
    """Aligns uploaded dataset column names and fills missing expected columns with sensible defaults."""
    df_out = df.copy()
    
    # Case-insensitive column matching
    col_map = {c.lower().strip(): c for c in df_out.columns}
    
    # Map common column aliases if uploaded
    aliases = {
        "cust_id": "customer_id",
        "id": "customer_id",
        "district_name": "district",
        "recharge_amt": "avg_recharge_amount_npr",
        "monthly_bill": "monthly_bill_npr"
    }
    
    rename_dict = {}
    for alias, target in aliases.items():
        if alias in col_map and target not in df_out.columns:
            rename_dict[col_map[alias]] = target
    if rename_dict:
        df_out = df_out.rename(columns=rename_dict)
        
    # Ensure all required raw numerical columns exist; if missing in an uploaded file, fill with default 0
    raw_numerics = [
        "age", "tenure_days", "calls_min_30d", "sms_count_30d", "data_gb_30d", 
        "night_usage_pct", "last_recharge_days_ago", "avg_recharge_amount_npr", 
        "recharge_count_30d", "monthly_bill_npr", "signal_strength_dbm", "call_drop_rate", 
        "avg_data_speed_mbps", "num_complaints_30d", "avg_resolution_time_hours", 
        "usage_drop_pct", "recharge_drop_pct", "inactive_days"
    ]
    for col in raw_numerics:
        if col not in df_out.columns:
            df_out[col] = 0.0
            
    # Binary columns default to 0
    for col in BINARY_COLS:
        if col not in df_out.columns:
            df_out[col] = 0
            
    # Categorical columns default to mode/Unknown
    for col in CATEGORICAL_COLS:
        if col not in df_out.columns:
            df_out[col] = "Unknown"
            
    return df_out

def engineer_features(df):
    """Generates interaction and derived features safely, avoiding divide-by-zero errors."""
    df_out = sanitize_and_align_columns(df)
    
    # 1. Usage intensity
    df_out["calls_per_day"] = df_out["calls_min_30d"].astype(float) / 30.0
    df_out["data_gb_per_day"] = df_out["data_gb_30d"].astype(float) / 30.0
    df_out["recharges_per_day"] = df_out["recharge_count_30d"].astype(float) / 30.0
    
    # 2. Recharge efficiency (safe division using +1)
    df_out["avg_recharge_per_transaction"] = df_out["avg_recharge_amount_npr"].astype(float) / (df_out["recharge_count_30d"].astype(float) + 1.0)
    
    # 3. Quality / complaint density
    df_out["complaint_density"] = df_out["num_complaints_30d"].astype(float) * df_out["avg_resolution_time_hours"].astype(float)
    df_out["call_drop_severity"] = df_out["call_drop_rate"].astype(float) * df_out["num_complaints_30d"].astype(float)
    
    # 4. Total services active
    df_out["total_active_packs"] = (
        df_out["data_pack_active"].astype(float) + 
        df_out["voice_pack_active"].astype(float) + 
        df_out["vas_active"].astype(float) + 
        df_out["roaming_active"].astype(float)
    )
    
    # 5. Combined risk interaction
    df_out["churn_risk_interaction"] = (
        df_out["usage_drop_pct"].astype(float) * 
        df_out["recharge_drop_pct"].astype(float) * 
        df_out["inactive_days"].astype(float)
    )
    
    return df_out

def create_preprocessing_pipeline():
    """Creates a ColumnTransformer pipeline for feature imputing and categorical one-hot encoding."""
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    binary_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
            ("bin", binary_transformer, BINARY_COLS)
        ],
        remainder="drop" # Safely drops customer_id, district, target churn, or extraneous columns
    )
    
    return preprocessor

def preprocess_and_save(file_path, models_dir="models", test_size=0.2, random_state=42):
    """Loads raw data, fits preprocessor on training set, transforms data, and exports preprocessor."""
    os.makedirs(models_dir, exist_ok=True)
    
    df = load_data(file_path)
    df = engineer_features(df)
    
    if "churn" not in df.columns:
        raise KeyError("Target column 'churn' missing from training dataset.")
        
    X = df.drop(columns=["churn"])
    y = df["churn"]
    
    # Stratified 80/20 train/test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(X_train_raw)
    
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    num_features = NUMERICAL_COLS
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_features = list(cat_encoder.get_feature_names_out(CATEGORICAL_COLS))
    bin_features = BINARY_COLS
    
    feature_names = num_features + cat_features + bin_features
    
    preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Fitted preprocessor saved successfully to {preprocessor_path}")
    
    X_train_df = pd.DataFrame(X_train, columns=feature_names, index=X_train_raw.index)
    X_test_df = pd.DataFrame(X_test, columns=feature_names, index=X_test_raw.index)
    
    return X_train_df, X_test_df, y_train, y_test, X_train_raw, X_test_raw, feature_names

def transform_uploaded_dataset(df_raw, preprocessor_path="models/preprocessor.joblib"):
    """Preprocesses any user-uploaded CSV dataset for inference, returning a cleaned DataFrame."""
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)
    
    df_engineered = engineer_features(df_raw)
    processed_arr = preprocessor.transform(df_engineered)
    
    feature_names = get_feature_names(preprocessor)
    
    return pd.DataFrame(processed_arr, columns=feature_names, index=df_raw.index)

if __name__ == "__main__":
    data_path = os.path.join("data", "nepal_telecom_churn_main.csv")
    X_train, X_test, y_train, y_test, X_train_raw, X_test_raw, feature_names = preprocess_and_save(data_path)
    print(f"Preprocessing completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Number of preprocessed features: {len(feature_names)}")
