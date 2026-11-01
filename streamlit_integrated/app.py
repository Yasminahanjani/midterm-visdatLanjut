"""Integrasi animasi patahan dan tujuh grafik Plotly pada HTML terbaru."""
from pathlib import Path
import re

import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

from visualizations import load_and_clean, make_figures

BASE = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Data Story Gempa",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown("""
<style>
/* Kunci scroll halaman Streamlit luar: hanya halaman cerita di iframe yang scroll. */
html, body, #root, [data-testid="stApp"],
[data-testid="stAppViewContainer"], [data-testid="stMain"] {
    height: 100dvh !important;
    max-height: 100dvh !important;
    overflow: hidden !important;
    overscroll-behavior: none !important;
}
header[data-testid="stHeader"], footer, [data-testid="stDecoration"] {
    display: none !important;
}
[data-testid="stMainBlockContainer"], .block-container {
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
    height: 100dvh !important;
    overflow: hidden !important;
}
/* Hindari tinggi iframe tetap (900px) yang memicu scroll kedua. */
[data-testid="stElementContainer"]:has(iframe) {
    height: 100dvh !important;
    overflow: hidden !important;
}
[data-testid="stElementContainer"] iframe,
iframe[title="streamlit.components.v1.html"] {
    display: block !important;
    width: 100% !important;
    height: 100dvh !important;
    border: 0 !important;
}
[data-testid="stAppViewContainer"] {background: #EAF6FC;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Membaca data gempa...")
def get_data():
    return load_and_clean(BASE / "data" / "katalog_gempa.csv")

@st.cache_data(show_spinner="Membuat visualisasi...")
def get_figures(data):
    return make_figures(data)


def assemble_story(template: str, figures: list) -> str:
    """Mengisi VIZ_01..VIZ_07 pada HTML yang diunggah, bukan membuat ulang scene."""
    html = template
    for i, fig in enumerate(figures, start=1):
        responsive_fig = fig.to_dict()
        responsive_fig.setdefault("layout", {}).pop("width", None)
        responsive_fig["layout"]["autosize"] = True
        
        plot = pio.to_html(
            responsive_fig,
            full_html=False,
            include_plotlyjs=True if i == 1 else False,
            config={"responsive": True, "displaylogo": False, "scrollZoom": False},
        )
        opening = f"<!--VIZ_{i:02d}_START-->"
        closing = f"<!--VIZ_{i:02d}_END-->"
        pattern = re.escape(opening) + r".*?" + re.escape(closing)
        if not re.search(pattern, html, flags=re.DOTALL):
            raise ValueError(f"Placeholder visualisasi {i} tidak ditemukan dalam story.html")
        html = re.sub(pattern, lambda _m: opening + plot + closing, html,
                      count=1, flags=re.DOTALL)
    html = html.replace("<!--PLOTLY_JS-->", "")
    html = html.replace("</style>", """
    .chart .plotly-graph-div {width:100%!important;max-width:100%!important;transform:none!important}
    .chart {min-width:0;width:100%}
    @media(max-width:700px){.txt{padding:0 4px}.chart{border-radius:8px}}
    </style>""", 1)
    return html


csv_path = BASE / "data" / "katalog_gempa.csv"
if not csv_path.is_file():
    st.error("Tidak menemukan data/katalog_gempa.csv")
    st.stop()

try:
    df = get_data()
    figures, facts = get_figures(df)
    source = (BASE / "story.html").read_text(encoding="utf-8")
    page = assemble_story(source, figures)
    components.html(page, height=900, scrolling=True)
except Exception as exc:
    st.exception(exc)
