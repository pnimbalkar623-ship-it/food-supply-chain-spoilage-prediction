"""Main entry point for the Food Supply Chain Spoilage Prediction app."""
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.streamlit_app import apply_page_style

st.set_page_config(
    page_title="Food Spoilage Prediction",
    page_icon="🍏",
    layout="wide",
)

page_definitions = [
    ("Home", "🏠", "pages/0_Home.py"),
    ("Food Image Analysis", "🖼️", "pages/1_Food_Image_Analysis.py"),
    ("Shipment Risk Analysis", "🚚", "pages/2_Shipment_Risk_Analysis.py"),
    ("Dataset Explorer", "📊", "pages/4_Dataset_Explorer.py"),
    ("Model Performance", "📈", "pages/3_Dashboard_Model_Performance.py"),
]
pages = [
    st.Page(
        str(PROJECT_ROOT / page_path),
        title=title,
        icon=icon,
        url_path=title.lower().replace(" ", "-"),
    )
    for title, icon, page_path in page_definitions
]

apply_page_style()

with st.sidebar:
    st.title("🍏 Food Freshness")
    st.caption("Supply chain spoilage dashboard")
    st.divider()

current_page = st.navigation(pages, position="sidebar")
current_page.run()
