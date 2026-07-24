# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

def is_dark_theme() -> bool:
    """Return True if the current Streamlit theme is dark.

    Streamlit exposes the selected theme via ``st.get_option('theme.base')``
    which returns either ``'dark'`` or ``'light'``.
    """
    try:
        theme = st.get_option('theme.base')
        return theme == 'dark'
    except Exception:
        # Fallback to dark theme for safety if Streamlit option is unavailable.
        return True

def apply_mpl_theme(fig: plt.Figure, dark: bool = None):
    """Apply light/dark style to a Matplotlib figure.

    If ``dark`` is None, the current Streamlit theme is detected via
    ``is_dark_theme()``.
    """
    if dark is None:
        dark = is_dark_theme()
    # Define colors
    if dark:
        facecolor = '#0f172a'
        edgecolor = '#1e293b'
        textcolor = '#e0e0e0'
    else:
        facecolor = '#ffffff'
        edgecolor = '#e5e7eb'
        textcolor = '#111111'
    fig.patch.set_facecolor(facecolor)
    for ax in fig.get_axes():
        ax.set_facecolor(facecolor)
        ax.tick_params(colors=textcolor)
        ax.xaxis.label.set_color(textcolor)
        ax.yaxis.label.set_color(textcolor)
        ax.title.set_color(textcolor)
        for spine in ax.spines.values():
            spine.set_edgecolor(edgecolor)
    return fig

def get_plotly_template(dark: bool) -> str:
    """Return Plotly template name based on dark mode."""
    return 'plotly_dark' if dark else 'plotly'

def inject_css(dark: bool) -> str:
    """Return CSS string to style Streamlit components according to the theme.
    The CSS is injected via `st.markdown(..., unsafe_allow_html=True)`.
    """
    if dark:
        bg = "#0e1117"
        text = "#e0e0e0"
        card = "#1e222a"
        shadow = "0 4px 14px rgba(0,0,0,0.45)"
    else:
        bg = "#f8f9fa"
        text = "#212529"
        card = "#ffffff"
        shadow = "0 4px 14px rgba(0,0,0,0.08)"
    return f"""
    <style>
    .stApp {{ background-color: {bg}; color: {text}; }}
    .streamlit-expanderHeader, .st-bf {{ color: {text}; }}
    .stButton > button {{ background-color: {card}; border: none; box-shadow: {shadow}; color: {text}; }}
    </style>
    """
