import pandas as pd
import os

def export_excel(df: pd.DataFrame, output_path: str) -> str:
    """Export a DataFrame to an Excel file.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to export.
    output_path: str
        Destination file path.
    Returns
    -------
    str
        Path to the written Excel file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='AtRisk')
    return output_path
