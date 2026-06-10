import streamlit as st
from streamlit_option_menu import option_menu

def render_navigation():
    """Render the top navigation menu and return the selected mode string."""
    app_mode = option_menu(
        menu_title=None,
        options=["Executive Overview", "Database Explorer", "Batch Prediction & Risk Explorer", "What-If Simulator"],
        icons=["bar-chart-line-fill", "search", "exclamation-triangle", "brain"],
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {
                "padding": "12px 0px",
                "background-color": "#0f172a",
                "border-radius": "0px"
            },
            "icon": {"color": "#ffffff", "font-size": "14px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "center",
                "margin": "0px 15px",
                "color": "#94a3b8",
                "font-weight": "500",
                "border-radius": "50px",
                "--hover-color": "#1e293b"
            },
            "nav-link-selected": {
                "background-color": "#dc2626",
                "color": "#ffffff",
                "font-weight": "600",
                "border-radius": "50px",
                "box-shadow": "0px 0px 12px rgba(220, 38, 38, 0.4)"
            }
        }
    )
    return app_mode
