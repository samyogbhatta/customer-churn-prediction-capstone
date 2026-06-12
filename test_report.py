import os
import pandas as pd
from src.components.report_generator import generate_report_pdf

# Dummy data
overall_summary = {
    "Total Customers": 100,
    "Overall Churn Rate (%)": "10%",
    "Model Accuracy": "85%",
    "Revenue at Risk (Rs.)": "50000"
}

# Create dummy DataFrames
at_risk_df = pd.DataFrame({
    "CustomerID": [1, 2],
    "Risk Level": ["🔴 Critical", "⚠️ Elevated"],
    "Score": [0.9, 0.8]
})

detailed_df = pd.DataFrame({
    "CustomerID": range(1, 6),
    "Risk Level": ["🔴 Critical", "⚠️ Elevated", "🟢 Low", "🟢 Low", "⚠️ Elevated"],
    "Score": [0.9, 0.8, 0.3, 0.2, 0.7]
})

output_path = os.path.join('reports', 'test_report.pdf')

pdf_path = generate_report_pdf(overall_summary, at_risk_df, detailed_df)
print('Generated PDF at:', pdf_path)
