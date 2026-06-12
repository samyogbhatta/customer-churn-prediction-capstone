import os
from fpdf import FPDF
import pandas as pd

class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        font_path = os.path.join(os.path.dirname(__file__), 'DejaVuSans.ttf')
        if os.path.exists(font_path):
            self.add_font('DejaVu', '', font_path, uni=True)
            self.set_font('DejaVu', '', 12)
        else:
            self.set_font('Arial', '', 12)

    def header(self):
        self.set_font('DejaVu', 'B', 14) if 'DejaVu' in self.fonts else self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Customer Churn Prediction Report', ln=True, align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', 'I', 10) if 'DejaVu' in self.fonts else self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def add_section_title(self, title: str):
        self.set_font('DejaVu', 'B', 13) if 'DejaVu' in self.fonts else self.set_font('Arial', 'B', 13)
        self.cell(0, 10, title, ln=True)
        self.ln(2)
        self.set_font('DejaVu', '', 12) if 'DejaVu' in self.fonts else self.set_font('Arial', '', 12)

    def add_paragraph(self, text: str):
        self.multi_cell(0, 8, text)
        self.ln(3)

    def add_dataframe(self, df: pd.DataFrame, title: str = None):
        if title:
            self.add_section_title(title)
        # Table header
        col_width = self.w / (len(df.columns) + 1)
        self.set_fill_color(200, 200, 200)
        for col in df.columns:
            self.cell(col_width, 8, str(col), border=1, fill=True, align='C')
        self.ln()
        # Table rows
        self.set_fill_color(255, 255, 255)
        for _, row in df.iterrows():
            for item in row:
                # Remove non‑ASCII characters (e.g., emojis) to avoid encoding errors with Helvetica
                clean_text = str(item).encode('ascii', 'ignore').decode()
                self.cell(col_width, 8, clean_text, border=1, align='C')
            self.ln()
        self.ln(5)

def export_report(overall_summary: dict, at_risk_df: pd.DataFrame, detailed_df: pd.DataFrame, output_path: str):
    """Generate a PDF report containing the churn prediction results.

    Args:
        overall_summary: Dictionary with high‑level metrics (e.g., total customers, at‑risk count, risk score avg).
        at_risk_df: DataFrame listing customers identified as at risk.
        detailed_df: DataFrame with the full prediction details for all customers.
        output_path: Destination file path for the generated PDF.
    """
    pdf = PDFReport()
    pdf.add_page()

    # Overall summary section
    pdf.add_section_title('Overall Summary')
    for key, value in overall_summary.items():
        pdf.add_paragraph(f"{key}: {value}")

    # At‑risk customers section
    pdf.add_section_title('Customers At Risk')
    if not at_risk_df.empty:
        if len(at_risk_df) > 100:
            pdf.add_paragraph(f"Showing top 100 of {len(at_risk_df)} at-risk customer records. For the full list, please refer to the Excel export.")
            pdf.add_dataframe(at_risk_df.head(100))
        else:
            pdf.add_dataframe(at_risk_df)
    else:
        pdf.add_paragraph('No customers were flagged as at risk.')

    # Detailed results section
    pdf.add_section_title('Detailed Prediction Results')
    if len(detailed_df) > 100:
        pdf.add_paragraph(f"Showing top 100 of {len(detailed_df)} customer records. For the full list, please refer to the Excel export.")
        pdf.add_dataframe(detailed_df.head(100))
    else:
        pdf.add_dataframe(detailed_df)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(output_path)
