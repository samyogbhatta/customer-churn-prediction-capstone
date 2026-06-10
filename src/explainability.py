import os
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier
import plotly.graph_objects as go
import joblib

try:
    from preprocessing import engineer_features
except ImportError:
    from src.preprocessing import engineer_features

class ChurnExplainer:
    def __init__(self, model_path="models/xgboost_churn_model.json", preprocessor_path="models/preprocessor.joblib"):
        """Initializes the SHAP Explainer with the trained XGBoost model and preprocessor."""
        # Load the model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")
        self.model = XGBClassifier()
        self.model.load_model(model_path)
        
        # Load the preprocessor
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}. Run training first.")
        self.preprocessor = joblib.load(preprocessor_path)
        
        # Initialize SHAP TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)
        
        # Get feature names after transformation
        # We extract this from the preprocessor ColumnTransformer
        num_features = self.preprocessor.transformers_[0][2]
        
        cat_encoder = self.preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_cols = self.preprocessor.transformers_[1][2]
        cat_features = list(cat_encoder.get_feature_names_out(cat_cols))
        
        bin_features = self.preprocessor.transformers_[2][2]
        
        self.feature_names = num_features + cat_features + bin_features

    def get_preprocessed_df(self, raw_df):
        """Converts raw customer DataFrame into preprocessed DataFrame with feature names."""
        engineered_df = engineer_features(raw_df)
        processed_arr = self.preprocessor.transform(engineered_df)
        return pd.DataFrame(processed_arr, columns=self.feature_names, index=raw_df.index)

    def explain_instance(self, raw_customer_row):
        """Calculates SHAP values for a single customer row.
        
        Returns:
            dict containing base value, prediction probability, SHAP values, and feature contributions.
        """
        # Engineer features to get the unscaled values
        engineered_row = engineer_features(raw_customer_row)
        
        # Transform the single row
        X_processed = self.get_preprocessed_df(raw_customer_row)
        
        # Get probability prediction
        prob = self.model.predict_proba(X_processed)[0, 1]
        
        # Compute SHAP values
        shap_values_obj = self.explainer(X_processed)
        
        # For tree classifiers, explainer output shap_values can be 2D (one per class) or 1D
        # For binary classification, typically shape is (num_instances, num_features)
        # expected_value can be a list or single float
        shap_vals = shap_values_obj.values[0]
        base_value = self.explainer.expected_value
        
        if isinstance(base_value, (list, np.ndarray)) and len(base_value) > 1:
            # Multi-class format or two outputs for binary class
            shap_vals = shap_vals[:, 1] if len(shap_vals.shape) > 1 else shap_vals
            base_value = base_value[1]
            
        # Create a DataFrame of feature contributions
        # Map them back to features
        contributions = pd.DataFrame({
            "Feature": self.feature_names,
            "SHAP_Value": shap_vals,
            "Absolute_SHAP": np.abs(shap_vals)
        })
        
        # We can enrich this with original values
        # Let's map features back to original or preprocessed values
        orig_values = {}
        for feature in self.feature_names:
            # Handle One-Hot Encoded features
            if "_" in feature and any(c in feature for c in ["gender_", "province_", "district_type_", "sim_type_"]):
                orig_values[feature] = X_processed[feature].values[0]
            else:
                if feature in engineered_row.columns:
                    orig_values[feature] = engineered_row[feature].values[0]
                else:
                    orig_values[feature] = X_processed[feature].values[0]
                    
        contributions["Feature_Value"] = contributions["Feature"].map(orig_values)
        
        # Sort by absolute SHAP value
        contributions = contributions.sort_values(by="Absolute_SHAP", ascending=False).reset_index(drop=True)
        
        return {
            "base_value": float(base_value),
            "probability": float(prob),
            "contributions": contributions,
            "shap_values_obj": shap_values_obj
        }

    def get_global_explanations(self, X_sample_processed):
        """Computes SHAP values for a sample dataset to show global summary."""
        # Calculate SHAP values
        shap_values_obj = self.explainer(X_sample_processed)
        
        # Handle shape mismatch (e.g. 3D output)
        shap_vals = shap_values_obj.values
        if len(shap_vals.shape) == 3:
            shap_vals = shap_vals[:, :, 1]
            
        # Calculate average absolute SHAP values for feature importance
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        importance_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Mean_Abs_SHAP": mean_abs_shap
        }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
        
        return shap_vals, importance_df

def plot_local_shap(contributions, max_display=10, theme_dark=True):
    """Generates an interactive Plotly horizontal bar chart for local SHAP contributions.
    
    Shows features driving the prediction towards churn (Red) or loyalty (Green).
    """
    # Take the top N contributing features
    df_plot = contributions.head(max_display).copy()
    
    # Reverse order so the largest is at the top of the horizontal bar chart
    df_plot = df_plot.iloc[::-1]
    
    # Set colors
    colors = ["#FF4B4B" if val > 0 else "#2E7D32" for val in df_plot["SHAP_Value"]]
    
    # Clean label formatting (e.g. province_Bagmati -> Province: Bagmati)
    clean_names = []
    for feat, val in zip(df_plot["Feature"], df_plot["Feature_Value"]):
        name = feat.replace("_", " ").title()
        if "Province" in name:
            name = f"Province: {name.split(' ')[-1]}"
        elif "Gender" in name:
            name = f"Gender: {name.split(' ')[-1]}"
        elif "Sim Type" in name:
            name = f"SIM: {name.split(' ')[-1]}"
        elif "District Type" in name:
            name = f"District: {name.split(' ')[-1]}"
            
        # Add actual value in parentheses if numeric
        if isinstance(val, (int, float)) and not np.isnan(val):
            if val == 1 or val == 0:
                if feat in ["data_pack_active", "voice_pack_active", "vas_active", "roaming_active"]:
                    clean_names.append(f"{name} ({'Yes' if val == 1 else 'No'})")
                    continue
            if isinstance(val, float):
                if val.is_integer():
                    clean_names.append(f"{name} ({int(val)})")
                else:
                    clean_names.append(f"{name} ({val:.2f})")
            else:
                clean_names.append(f"{name} ({val})")
        else:
            clean_names.append(f"{name}")
            
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=clean_names,
        x=df_plot["SHAP_Value"],
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=1, color="rgba(255,255,255,0.15)")
        ),
        hovertemplate="Feature: %{y}<br>SHAP Value: %{x:.4f}<extra></extra>"
    ))
    
    # Dynamic layout styling
    bg_color = "rgba(17,17,17,0)" # transparent
    text_color = "#E0E0E0" if theme_dark else "#111111"
    grid_color = "rgba(255,255,255,0.08)" if theme_dark else "rgba(0,0,0,0.08)"
    
    fig.update_layout(
        title=dict(
            text="Top Feature Contributions to Churn Prediction",
            font=dict(size=14, color=text_color),
            x=0.0
        ),
        xaxis=dict(
            title=dict(
                text="SHAP Value (Impact on Prediction Log-Odds)",
                font=dict(size=11, color=text_color)
            ),
            tickfont=dict(color=text_color),
            gridcolor=grid_color,
            zerolinecolor=text_color,
            zerolinewidth=1.5
        ),
        yaxis=dict(
            tickfont=dict(size=11, color=text_color),
            automargin=True
        ),
        margin=dict(l=20, r=20, t=40, b=40),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        height=380
    )
    
    # Add vertical line at X = 0
    fig.add_vline(x=0, line_width=1.5, line_color="#888888", line_dash="solid")
    
    return fig

if __name__ == "__main__":
    # Test explainability script
    try:
        explainer = ChurnExplainer()
        print("Explainer initialized successfully.")
        
        # Test with a dummy customer raw row
        dummy_row = pd.DataFrame([{
            "age": 35,
            "gender": "Male",
            "province": "Bagmati",
            "district_type": "Urban",
            "sim_type": "Prepaid",
            "tenure_days": 100,
            "calls_min_30d": 120.0,
            "sms_count_30d": 10,
            "data_gb_30d": 4.5,
            "night_usage_pct": 15.0,
            "last_recharge_days_ago": 5,
            "avg_recharge_amount_npr": 150.0,
            "recharge_count_30d": 2,
            "signal_strength_dbm": -85,
            "call_drop_rate": 0.02,
            "avg_data_speed_mbps": 12.5,
            "num_complaints_30d": 1,
            "avg_resolution_time_hours": 24.0,
            "data_pack_active": 1,
            "voice_pack_active": 0,
            "vas_active": 0,
            "roaming_active": 0,
            "usage_drop_pct": 0.1,
            "recharge_drop_pct": 0.0,
            "inactive_days": 2
        }])
        
        exp_res = explainer.explain_instance(dummy_row)
        print(f"Customer Churn Probability: {exp_res['probability']:.4f}")
        print("Top 5 contributing features:")
        print(exp_res["contributions"].head(5))
        
    except Exception as e:
        print(f"Error testing explainer: {e}")
    # Existing code up to line 264 remains unchanged

# ---------------------------------------------------------------------------
# Additional SHAP plotting utilities required by the Streamlit app
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
from plotly.tools import mpl_to_plotly

def plot_summary(shap_values, X_processed, max_display=20):
    """Generates a SHAP summary (beeswarm) plot and returns a Plotly Figure.
    """
    shap.summary_plot(shap_values, X_processed, max_display=max_display, show=False)
    fig = plt.gcf()
    return mpl_to_plotly(fig)

def plot_mean_bar(shap_values, X_processed, max_display=20):
    """Generates a SHAP mean absolute bar plot and returns a Plotly Figure.
    """
    shap.summary_plot(shap_values, X_processed, max_display=max_display, plot_type="bar", show=False)
    fig = plt.gcf()
    return mpl_to_plotly(fig)

def plot_dependence(feature_name, shap_values, X_processed, interaction_index=None):
    """Generates a SHAP dependence plot for a given feature.
    """
    shap.dependence_plot(feature_name, shap_values, X_processed, interaction_index=interaction_index, show=False)
    fig = plt.gcf()
    return mpl_to_plotly(fig)

def plot_waterfall(shap_values_instance, X_processed_instance, max_display=10):
    """Generates a SHAP waterfall plot for a single instance.
    """
    shap.waterfall_plot(shap_values_instance, max_display=max_display, show=False)
    fig = plt.gcf()
    return mpl_to_plotly(fig)
