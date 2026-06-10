def get_css():
    """Return the custom CSS for the app as a string."""
    return """
    <style>
        /* Metric Cards Styling */
        .metric-card-container {
            display: flex;
            justify-content: space-between;
            gap: 15px;
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: #1e293b;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #334155;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
            flex: 1;
            transition: transform 0.2s, border-color 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: #3b82f6;
        }
        .metric-value {
            font-size: 28px;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 4px;
        }
        .metric-label {
            font-size: 12px;
            font-weight: 600;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        /* Section dividers */
        hr {
            margin: 1.5rem 0;
            border-color: rgba(255, 255, 255, 0.1);
        }
    </style>
    """
