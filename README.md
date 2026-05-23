# Telecom Customer Churn Prediction & Explainable AI Dashboard

This project is a complete end-to-end Machine Learning and Explainable AI (XAI) system tailored for customer churn prediction in the Nepalese telecommunication sector (e.g. Ncell, NTC contexts). It trains a tuned XGBoost Classifier to predict churn and integrates SHAP (SHapley Additive exPlanations) to explain individual predictions in real-time. The final dashboard is built using Streamlit.

---

## Key Features

1. **Synthetic Data Generator (`src/data_generator.py`)**
   - Synthesizes realistic subscriber profiles using Nepalese demographics (Provinces 1-7, district classifications: Urban, Rural, Semi-Urban).
   - Generates usage details (voice calls, SMS count, data GB, night usage percentage) and recharges in Nepalese Rupees (NPR).
   - Simulates realistic network quality issues (signal strength, call drop rate, average speed) and complaints logs.
   - Calculates churn outcomes using a correlated logit function.

2. **Preprocessing Pipeline (`src/preprocessing.py`)**
   - Clean and scale numerical features using `StandardScaler` inside a scikit-learn `Pipeline`.
   - Encodes categorical values using `OneHotEncoder`.
   - Saves the preprocessor to `models/preprocessor.joblib` to prevent data leakage during deployment.
   - Uses stratified splitting (80% train, 20% test) to preserve churn balance.

3. **Model Training & Tuning (`src/train.py`)**
   - Fits a tuned `XGBClassifier` using hyperparameters that reduce overfitting (tuned max depth, learning rate, subsampling, and features fraction).
   - Utilizes `scale_pos_weight` to automatically balance minority-class churn samples.
   - Evaluates performance metrics: **Accuracy, Precision, Recall, F1-Score, and ROC-AUC**.
   - Exports the model in native format to `models/xgboost_churn_model.json` and metric reports to `models/metrics.json`.

4. **Explainable AI (SHAP) (`src/explainability.py`)**
   - Sets up a SHAP `TreeExplainer` for model logic.
   - Includes custom, interactive **Plotly SHAP bar charts** mapping log-odds contributions directly to clean feature names and raw values for superior styling.

5. **Streamlit Dashboard (`app.py`)**
   - **Executive Overview Mode**: Shows high-level KPIs, monthly revenue at risk, churn rate by province, correlation heatmaps, and global SHAP feature importance.
   - **Database Explorer Mode**: Lets you select and view specific subscriber profiles, their churn risk dial speedometer, and active services.
   - **What-If Simulator Mode**: Lets you tweak demographic, recharge, and usage parameters using sliders to observe real-time predictions and local SHAP explanations.

---

## Project Structure

```
customer-churn-prediction-capstone/
├── data/
│   └── telecom_churn_nepal.csv         # Generated Nepal telecom churn dataset
├── models/
│   ├── xgboost_churn_model.json        # Pre-trained XGBoost model
│   ├── preprocessor.joblib             # Saved preprocessor pipeline
│   └── metrics.json                    # Saved validation metrics and feature names
├── src/
│   ├── data_generator.py               # Generates synthetic dataset
│   ├── preprocessing.py                # Preprocesses, scales, and encodes dataset
│   ├── train.py                        # Fits XGBoost classifier and exports metrics
│   └── explainability.py               # Handles SHAP explanations and Plotly viz
├── app.py                              # Main interactive Streamlit dashboard
├── requirements.txt                    # Python dependencies
└── README.md                           # Documentation & user instructions
```

---

## Local Installation & Setup (Python version 3.12.0)

Follow these steps to run the pipeline and dashboard locally:

### 1. Set Up Environment & Install Dependencies
Create a virtual environment (optional but recommended) and install packages:
```bash
# Using standard cmd/PowerShell on Windows
python -m venv venv
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Generate the Dataset
Create the synthetic telecom dataset:
```bash
python src/data_generator.py
```
This writes the file `data/telecom_churn_nepal.csv` containing 5,000 customers.

### 3. Train and Evaluate the Model
Fit the XGBoost classifier and export variables:
```bash
python src/train.py
```
This outputs classification metrics and confusion matrix values, saving files to `models/`.

### 4. Run the Streamlit Dashboard
Launch the web interface:
```bash
streamlit run app.py
```
The dashboard will open automatically in your default browser at `http://localhost:8501`.
