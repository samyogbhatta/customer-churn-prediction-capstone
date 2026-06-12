import os
import pandas as pd
from src.utils.pdf_export import export_report

def generate_report_pdf(overall_summary: dict, filtered_df: pd.DataFrame, explainer, output_dir: str = "reports") -> str:
    """Generate a PDF report and return the file path.

    Parameters
    ----------
    overall_summary: dict
        High‑level metrics such as total customers, churn rate, etc.
    filtered_df: pd.DataFrame
        Filtered customer records dataset.
    explainer: ChurnExplainer
        ChurnExplainer object for SHAP calculations.
    output_dir: str, optional
        Directory where the PDF will be saved. Defaults to "reports".

    Returns
    -------
    str
        Absolute path to the generated PDF file.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Build a filename using the uploaded dataset name if available in session state
    filename = "churn_report.pdf"
    output_path = os.path.join(output_dir, filename)
    export_report(overall_summary, filtered_df, explainer, output_path)
    return output_path

