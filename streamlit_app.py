"""Streamlit app — Financial News Sentiment Analysis."""

import base64
import functools
import io
import json
import os
import sys
import time
from pathlib import Path

# Fix SSL cert path for curl_cffi on Windows (spaces in OneDrive path break it)
import certifi
import shutil
_cert_src = certifi.where()
if " " in _cert_src:
    _cert_dst = os.path.join(os.environ.get("TEMP", "C:\\Temp"), "cacert.pem")
    if not os.path.exists(_cert_dst) or os.path.getmtime(_cert_src) > os.path.getmtime(_cert_dst):
        shutil.copy2(_cert_src, _cert_dst)
    os.environ["CURL_CA_BUNDLE"] = _cert_dst
    os.environ["REQUESTS_CA_BUNDLE"] = _cert_dst

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import MODELS_DIR, RESULTS_DIR
from src.inference import build_pipeline, get_device, predict
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Sentiment Analysis",
    page_icon="F",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS  = {"positive": "#16a34a", "neutral": "#2563eb", "negative": "#dc2626"}
SYMBOLS = {"positive": "+", "neutral": "~", "negative": "-"}

ALL_EXAMPLES = [
    ("positive", "The company announced record profits and raised its annual dividend by 15%."),
    ("positive", "Net sales in the third quarter increased by 12% year-on-year to EUR 205 million."),
    ("positive", "Operating profit rose to EUR 13.1 million, representing 7.7% of net sales."),
    ("neutral",  "Revenue remained flat compared to the previous quarter."),
    ("neutral",  "The company plans to open two new offices in the Nordic region by 2027."),
    ("neutral",  "Annual general meeting will take place on March 15 at the Helsinki headquarters."),
    ("negative", "CEO resigned amid an accounting fraud scandal, sending shares tumbling 30%."),
    ("negative", "Acquisition talks with a major competitor have reportedly collapsed."),
    ("negative", "Operating loss widened to EUR 8 million due to declining demand."),
]

PAGES = [
    ("Analyze",           "M12 20h9M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"),
    ("Batch Analysis",    "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z M14 2v6h6 M16 13H8 M16 17H8 M10 9H8"),
    ("Live News",         "M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8z M12 12m-3 0a3 3 0 1 0 6 0a3 3 0 1 0-6 0"),
    ("Model Leaderboard", "M8 6l4-4 4 4 M4 18h4v-4 M20 18h-4v-4 M12 2v10 M18 22V12 M6 22V12"),
    ("Rendu",             "M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z M2 10h20"),
    ("Project",           "M12 12m-10 0a10 10 0 1 0 20 0a10 10 0 1 0-20 0 M12 16v-4 M12 8h.01"),
]

# ── Detect available models ──────────────────────────────────────────
AVAILABLE_MODELS = {}
for name in ["FinBERT", "BERT", "DistilBERT"]:
    p = MODELS_DIR / name
    if p.exists() and (p / "config.json").exists():
        AVAILABLE_MODELS[name] = p

MODEL_META = {
    "FinBERT":    {"desc": "Financial domain transformer",  "badge": "Best"},
    "BERT":       {"desc": "General-purpose transformer",   "badge": ""},
    "DistilBERT": {"desc": "Fast distilled transformer",    "badge": "Fast"},
}

# ── Model loading (cached per model name) ────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    path = AVAILABLE_MODELS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForSequenceClassification.from_pretrained(str(path))
    device = get_device()
    pipe = build_pipeline(model, tokenizer, device)
    return pipe, device

# ── Leaderboard data ─────────────────────────────────────────────────
@st.cache_data
def load_leaderboard():
    path = RESULTS_DIR / "leaderboard.json"
    if not path.exists():
        return pd.DataFrame()
    data = json.loads(path.read_text())
    df = pd.DataFrame(data).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    df.index += 1
    return df

lb = load_leaderboard()

# ── LIME ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_lime_explainer():
    from lime.lime_text import LimeTextExplainer
    return LimeTextExplainer(class_names=["positive", "neutral", "negative"])

def lime_predict_proba(texts, _pipe):
    results = []
    for text in texts:
        out = _pipe(text[:512])
        scores_list = out[0] if out and isinstance(out[0], list) else out
        score_map = {r["label"]: r["score"] for r in scores_list}
        results.append([
            score_map.get("positive", 0),
            score_map.get("neutral", 0),
            score_map.get("negative", 0),
        ])
    return np.array(results)

def run_lime(text, label, _pipe):
    explainer = get_lime_explainer()
    label_idx = {"positive": 0, "neutral": 1, "negative": 2}[label]
    predict_fn = functools.partial(lime_predict_proba, _pipe=_pipe)
    exp = explainer.explain_instance(
        text, predict_fn, num_features=15, num_samples=200, labels=[label_idx],
    )
    return exp.as_list(label=label_idx)

def render_lime_html(text, word_weights):
    weight_map = {w.lower(): wt for w, wt in word_weights}
    tokens = text.split()
    parts = []
    for token in tokens:
        clean = token.strip(".,;:!?\"'()-").lower()
        w = weight_map.get(clean, 0)
        if abs(w) < 0.01:
            parts.append(f'<span style="padding:1px 2px;">{token}</span>')
        else:
            intensity = min(abs(w) * 5, 0.8)
            bg = f"rgba(22,163,74,{intensity})" if w > 0 else f"rgba(220,38,38,{intensity})"
            parts.append(
                f'<span style="background:{bg};padding:2px 5px;border-radius:4px;margin:0 1px;">{token}</span>'
            )
    return '<div style="font-size:15px;line-height:2.2;padding:8px 0;">' + " ".join(parts) + '</div>'


# ── Global CSS injection ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ========== FOUNDATION ========== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    .stApp {
        background: #f8fafc;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* ========== SIDEBAR ========== */
    section[data-testid="stSidebar"] { background: #0f172a; }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] .stMarkdown div { color: #e2e8f0; }
    section[data-testid="stSidebar"] hr { border-color: #1e293b; margin: 12px 0; }

    section[data-testid="stSidebar"] button[kind="secondary"] {
        background: transparent !important; border: none !important;
        color: #94a3b8 !important; text-align: left !important;
        padding: 10px 14px !important; border-radius: 8px !important;
        font-size: 14px !important; font-weight: 500 !important;
        transition: all 0.15s !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background: #1e293b !important; color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"].active-nav,
    section[data-testid="stSidebar"] button[kind="primary"] {
        background: linear-gradient(135deg, #1e3a5f, #1e293b) !important;
        color: #fff !important; font-weight: 600 !important; border: none !important;
        box-shadow: 0 2px 8px rgba(59,130,246,0.15) !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label {
        color: #475569 !important; font-size: 11px !important;
        text-transform: uppercase; letter-spacing: 1.2px;
    }
    section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
        background: #1e293b !important; border-color: #334155 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] * { color: #e2e8f0 !important; }
    .sidebar-label {
        font-size: 11px; font-weight: 600; color: #475569;
        text-transform: uppercase; letter-spacing: 1.2px; padding: 16px 0 6px;
    }

    /* ========== PAGE HEADER ========== */
    .page-header {
        padding: 32px 0 24px; text-align: center;
        border-bottom: 1px solid #e2e8f0; margin-bottom: 28px;
    }
    .page-header h1 {
        font-size: 28px; font-weight: 800; color: #0f172a;
        letter-spacing: -0.5px; margin: 0 0 6px;
    }
    .page-header p {
        font-size: 15px; color: #64748b; margin: 0;
        max-width: 600px; margin-left: auto; margin-right: auto;
    }

    /* ========== CARDS ========== */
    .card {
        background: #fff; border: 1px solid #e2e8f0; border-radius: 14px;
        padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.06);
        transition: box-shadow 0.2s;
    }
    .card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.06), 0 1px 3px rgba(0,0,0,0.08); }
    .card h3 { font-size: 16px; font-weight: 700; color: #0f172a; margin: 0 0 12px; }
    .card-muted { background: #f8fafc; border-color: #f1f5f9; }

    /* ========== METRICS ========== */
    [data-testid="stMetric"] {
        background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 18px 20px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    [data-testid="stMetricLabel"] {
        font-size: 12px !important; font-weight: 600 !important;
        text-transform: uppercase; letter-spacing: 0.8px;
        color: #64748b !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 24px !important; font-weight: 800 !important;
        color: #0f172a !important;
    }

    /* ========== BUTTONS ========== */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        border: none !important; color: #fff !important;
        font-weight: 600 !important; font-size: 14px !important;
        padding: 10px 24px !important; border-radius: 10px !important;
        box-shadow: 0 2px 8px rgba(59,130,246,0.3) !important;
        transition: all 0.2s !important; letter-spacing: 0.2px;
    }
    .stButton button[kind="primary"]:hover {
        box-shadow: 0 4px 16px rgba(59,130,246,0.4) !important;
        transform: translateY(-1px);
    }
    .stButton button[kind="secondary"],
    .stMainMenu button,
    div[data-testid="stMainMenu"] button {
        border-radius: 10px !important; font-weight: 500 !important;
        transition: all 0.15s !important;
    }
    /* Example chip buttons (main area only, not sidebar) */
    .main .stButton button[kind="secondary"] {
        background: #fff !important; border: 1px solid #e2e8f0 !important;
        color: #475569 !important; font-size: 12px !important;
        padding: 8px 14px !important; border-radius: 10px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
    }
    .main .stButton button[kind="secondary"]:hover {
        background: #f1f5f9 !important; border-color: #cbd5e1 !important;
        color: #1e293b !important;
    }

    /* ========== INPUTS ========== */
    .stTextArea textarea,
    .stTextInput input {
        border: 1.5px solid #e2e8f0 !important; border-radius: 12px !important;
        font-size: 15px !important; padding: 14px 16px !important;
        background: #fff !important; color: #1e293b !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    .stTextArea textarea:focus,
    .stTextInput input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
    }
    .stTextArea textarea::placeholder,
    .stTextInput input::placeholder {
        color: #94a3b8 !important;
    }

    /* ========== FILE UPLOADER ========== */
    [data-testid="stFileUploader"] {
        border: 2px dashed #cbd5e1 !important; border-radius: 14px !important;
        padding: 28px !important; background: #fafbfc !important;
        transition: border-color 0.2s !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #3b82f6 !important; background: #f0f7ff !important;
    }

    /* ========== TABS ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0; border-bottom: 2px solid #e2e8f0; padding: 0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px !important; font-size: 14px !important;
        font-weight: 500 !important; color: #64748b !important;
        border-bottom: 2px solid transparent; margin-bottom: -2px;
        transition: all 0.15s !important;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #1e293b !important; }
    .stTabs [aria-selected="true"] {
        color: #1d4ed8 !important; font-weight: 600 !important;
        border-bottom-color: #3b82f6 !important;
    }

    /* ========== DATAFRAMES ========== */
    [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0; border-radius: 12px;
        overflow: hidden;
    }

    /* ========== ALERTS ========== */
    [data-testid="stAlert"] {
        border-radius: 12px !important; font-size: 14px !important;
        border-left-width: 4px !important;
    }

    /* ========== PROGRESS ========== */
    .stProgress > div > div {
        border-radius: 6px; height: 6px !important;
    }
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #1d4ed8) !important;
        border-radius: 6px;
    }

    /* ========== DOWNLOAD BUTTON ========== */
    [data-testid="stDownloadButton"] button {
        background: #fff !important; border: 1.5px solid #e2e8f0 !important;
        border-radius: 10px !important; font-weight: 600 !important;
        color: #1e293b !important; transition: all 0.15s !important;
    }
    [data-testid="stDownloadButton"] button:hover {
        border-color: #3b82f6 !important; color: #1d4ed8 !important;
        box-shadow: 0 2px 8px rgba(59,130,246,0.12) !important;
    }

    /* ========== DIVIDERS ========== */
    .main hr {
        border: none; height: 1px; background: #e2e8f0; margin: 28px 0;
    }

    /* ========== SECTION TITLES ========== */
    .section-title {
        font-size: 15px; font-weight: 700; color: #0f172a;
        margin: 0 0 16px; padding-bottom: 10px;
        border-bottom: 2px solid #e2e8f0;
        letter-spacing: -0.2px;
    }

    /* ========== SPINNER ========== */
    .stSpinner > div { border-top-color: #3b82f6 !important; }

    /* ========== SELECTBOX (main area) ========== */
    .main .stSelectbox [data-baseweb="select"] {
        background: #fff !important; border: 1.5px solid #e2e8f0 !important;
        border-radius: 12px !important; transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    .main .stSelectbox [data-baseweb="select"]:hover {
        border-color: #cbd5e1 !important;
    }
    .main .stSelectbox [data-baseweb="select"]:focus-within {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
    }
    .main .stSelectbox [data-baseweb="select"] * {
        color: #1e293b !important;
    }
    .main .stSelectbox label {
        font-size: 13px !important; font-weight: 600 !important;
        color: #475569 !important;
    }

    /* Dropdown menu */
    [data-baseweb="popover"] {
        border-radius: 12px !important; border: 1px solid #e2e8f0 !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08) !important;
        overflow: hidden !important;
    }
    [data-baseweb="popover"] li {
        font-size: 14px !important; padding: 10px 14px !important;
        transition: background 0.1s !important;
    }
    [data-baseweb="popover"] li:hover {
        background: #f1f5f9 !important;
    }
    [data-baseweb="popover"] li[aria-selected="true"] {
        background: #eff6ff !important; color: #1d4ed8 !important;
        font-weight: 600 !important;
    }

    /* ========== HIDE STREAMLIT DEFAULT ELEMENTS ========== */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)


def page_header(title, subtitle=""):
    """Render a consistent page header across all pages."""
    sub = f'<p>{subtitle}</p>' if subtitle else ''
    st.markdown(f'<div class="page-header"><h1>{title}</h1>{sub}</div>', unsafe_allow_html=True)


# Shared Plotly layout defaults
PLOTLY_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, -apple-system, sans-serif", color="#334155"),
    margin=dict(t=16, b=36, l=12, r=12),
    xaxis=dict(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0"),
    yaxis=dict(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0"),
)


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state["page"] = "Analyze"
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = list(AVAILABLE_MODELS.keys())[0]

with st.sidebar:
    # Brand — candlestick chart icon for financial analysis
    st.markdown("""
    <div style="text-align:center; padding:20px 0 24px;">
        <div style="
            display:inline-flex; align-items:center; justify-content:center;
            width:48px; height:48px; border-radius:12px;
            background:linear-gradient(135deg,#0ea5e9,#1d4ed8);
            color:white; margin-bottom:8px;
            box-shadow:0 4px 16px rgba(14,165,233,0.35);
        ">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                <path d="M6 4v16"/>
                <rect x="4" y="8" width="4" height="6" rx="0.5" fill="currentColor" stroke="none" opacity="0.9"/>
                <path d="M12 2v20"/>
                <rect x="10" y="5" width="4" height="8" rx="0.5" fill="currentColor" stroke="none" opacity="0.9"/>
                <path d="M18 6v12"/>
                <rect x="16" y="9" width="4" height="5" rx="0.5" fill="currentColor" stroke="none" opacity="0.9"/>
            </svg>
        </div>
        <div style="font-size:15px; font-weight:700; color:#f1f5f9; letter-spacing:-0.3px;">FinSentiment</div>
        <div style="font-size:11px; color:#64748b;">NLP Analysis Platform</div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation — real st.button calls
    st.markdown('<div class="sidebar-label">Navigation</div>', unsafe_allow_html=True)

    for page_name, _ in PAGES:
        is_active = st.session_state["page"] == page_name
        btn_type = "primary" if is_active else "secondary"
        if st.button(page_name, key=f"nav_{page_name}", use_container_width=True, type=btn_type):
            st.session_state["page"] = page_name
            st.rerun()

    # Load model
    with st.spinner(f"Loading {st.session_state['selected_model']}..."):
        pipe, device = load_model(st.session_state["selected_model"])
    model_name = st.session_state["selected_model"]

    # Status
    st.markdown('<div class="sidebar-label">Status</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="padding:10px 14px; background:#1e293b; border-radius:8px; font-size:12px;">
        <div style="display:flex; justify-content:space-between; padding:4px 0;">
            <span style="color:#64748b;">Model</span>
            <span style="color:#e2e8f0; font-weight:600;">{model_name}</span>
        </div>
        <div style="display:flex; justify-content:space-between; padding:4px 0;">
            <span style="color:#64748b;">Device</span>
            <span style="color:#e2e8f0; font-weight:600;">{device.type.upper()}</span>
        </div>
        <div style="display:flex; justify-content:space-between; padding:4px 0;">
            <span style="color:#64748b;">Max tokens</span>
            <span style="color:#e2e8f0; font-weight:600;">128</span>
        </div>
        <div style="display:flex; justify-content:space-between; padding:4px 0;">
            <span style="color:#64748b;">Metric</span>
            <span style="color:#e2e8f0; font-weight:600;">Macro-F1</span>
        </div>
        <div style="display:flex; justify-content:space-between; padding:4px 0;">
            <span style="color:#64748b;">Split</span>
            <span style="color:#e2e8f0; font-weight:600;">70 / 10 / 20</span>
        </div>
        <div style="display:flex; justify-content:space-between; padding:4px 0;">
            <span style="color:#64748b;">Framework</span>
            <span style="color:#e2e8f0; font-weight:600;">PyTorch + HF</span>
        </div>
        <div style="display:flex; justify-content:space-between; padding:4px 0;">
            <span style="color:#64748b;">Env</span>
            <span style="color:#e2e8f0; font-weight:600;">Python 3.11 / uv</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

page = st.session_state["page"]

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — ANALYZE
# ══════════════════════════════════════════════════════════════════════
if page == "Analyze":

    page_header(
        "Analyze a Financial Headline",
        f"Try an example or type your own -- powered by {model_name}",
    )

    # ── Example chips ────────────────────────────────────────────
    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
    chip_cols = st.columns(3, gap="small")
    def _set_example(text):
        st.session_state["headline_input"] = text

    for i, (sent, ex) in enumerate(ALL_EXAMPLES):
        with chip_cols[i % 3]:
            st.button(ex, key=f"ex_{i}", use_container_width=True,
                      on_click=_set_example, args=(ex,))

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Main: Input left + Result right ──────────────────────────
    col_in, col_out = st.columns([3, 2], gap="large")

    with col_in:
        if "headline_input" not in st.session_state:
            st.session_state["headline_input"] = ""
        headline = st.text_area(
            label="headline",
            placeholder="Type or paste a financial news headline...",
            height=140,
            label_visibility="collapsed",
            key="headline_input",
        )

        # Buttons row: Analyze + Model selector + Clear
        btn_cols = st.columns([3, 2, 1])
        with btn_cols[0]:
            run = st.button("Analyze", type="primary", use_container_width=True)
        with btn_cols[1]:
            sel_model = st.selectbox(
                "model",
                list(AVAILABLE_MODELS.keys()),
                index=list(AVAILABLE_MODELS.keys()).index(st.session_state["selected_model"]),
                format_func=lambda m: f"{m} -- {MODEL_META.get(m, {}).get('desc', '')}",
                label_visibility="collapsed",
                key="model_sel",
            )
            if sel_model != st.session_state["selected_model"]:
                st.session_state["selected_model"] = sel_model
                st.rerun()
        with btn_cols[2]:
            def _clear_input():
                st.session_state["headline_input"] = ""
            st.button("Clear", use_container_width=True, on_click=_clear_input)

        # Voice input: inject a <script> into the PARENT page so that
        # SpeechRecognition runs in the parent JS context (not an iframe).
        # This gives it real mic permissions and a valid user-gesture on click.
        # components.html() is used only as a bootstrap loader (height=0).
        components.html("""
<script>
(function(){
    var d = window.parent.document;

    // Clean previous injections
    var ids = ['voice-mic-btn','voice-mic-css','voice-init-script'];
    ids.forEach(function(id){ var el = d.getElementById(id); if(el) el.remove(); });

    // Pulse animation CSS — injected into parent <head>
    var css = d.createElement('style');
    css.id = 'voice-mic-css';
    css.textContent = '@keyframes vmic-pulse{0%,100%{box-shadow:0 0 0 0 rgba(220,38,38,0.4)}50%{box-shadow:0 0 0 12px rgba(220,38,38,0)}}';
    d.head.appendChild(css);

    // Build the main logic as a function, convert to string, inject as
    // a <script> in the parent document so it executes in parent context.
    var sc = d.createElement('script');
    sc.id = 'voice-init-script';
    sc.textContent = '(' + function(){
        /* ---- runs in Streamlit parent page context ---- */
        var ta = document.querySelector('textarea');
        if(!ta) return;

        // Position the button inside the textarea's Streamlit wrapper
        var container = ta.closest('[data-testid="stTextArea"]');
        if(!container) container = ta.parentElement.parentElement;
        container.style.position = 'relative';

        // Mic button
        var btn = document.createElement('div');
        btn.id = 'voice-mic-btn';
        btn.title = 'Click to speak (Chrome / Edge)';
        btn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>';
        btn.style.cssText = 'position:absolute;bottom:14px;right:14px;z-index:9999;width:36px;height:36px;border-radius:50%;background:#f1f5f9;color:#64748b;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all 0.2s;box-shadow:0 2px 6px rgba(0,0,0,0.1);border:1px solid #e2e8f0;';

        btn.onmouseenter = function(){ if(!this._on) this.style.background='#e2e8f0'; };
        btn.onmouseleave = function(){ if(!this._on) this.style.background='#f1f5f9'; };

        container.appendChild(btn);

        // React-compatible value setter for Streamlit's textarea
        var setter = Object.getOwnPropertyDescriptor(
            HTMLTextAreaElement.prototype, 'value'
        ).set;

        var rec = null, isOn = false;

        btn.addEventListener('click', function(){
            var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
            if(!SR){ alert('Speech recognition not supported. Use Chrome or Edge.'); return; }
            if(isOn && rec){ rec.stop(); return; }

            rec = new SR();
            rec.lang = 'en-US';
            rec.continuous = true;
            rec.interimResults = true;
            isOn = true;
            btn._on = true;
            btn.style.background = '#fee2e2';
            btn.style.color = '#dc2626';
            btn.style.animation = 'vmic-pulse 1.5s ease-in-out infinite';
            btn.style.borderColor = '#fecaca';

            rec.onresult = function(e){
                var t = '';
                for(var i = 0; i < e.results.length; i++){
                    t += e.results[i][0].transcript;
                }
                setter.call(ta, t);
                ta.dispatchEvent(new Event('input', {bubbles:true}));
                ta.dispatchEvent(new Event('change', {bubbles:true}));
            };
            var stopRec = function(){
                isOn = false;
                btn._on = false;
                btn.style.background = '#f1f5f9';
                btn.style.color = '#64748b';
                btn.style.animation = 'none';
                btn.style.borderColor = '#e2e8f0';
                // Focus then blur the textarea so Streamlit commits
                // the value to its internal widget state (session_state)
                ta.focus();
                ta.dispatchEvent(new Event('change', {bubbles:true}));
                ta.blur();
            };
            rec.onend = stopRec;
            rec.onerror = stopRec;
            rec.start();
        });
    } + ')();';
    d.body.appendChild(sc);
})();
</script>
""", height=0)

    with col_out:
        result_container = st.container()

    # ── Run analysis ─────────────────────────────────────────────
    analysis_result = None

    if run and headline.strip():
        t0 = time.time()
        result = predict(headline, pipe)
        ms = (time.time() - t0) * 1000
        lbl  = result["label"]
        conf = result["confidence"]
        sc   = result["scores"]
        col  = COLORS[lbl]
        analysis_result = result

        with result_container:
            st.markdown(f"""
            <div class="card" style="
                border-color:{col}30; text-align:center;
                background:linear-gradient(160deg,{col}04,{col}10);
            ">
                <div style="
                    display:inline-flex; align-items:center; justify-content:center;
                    width:56px; height:56px; border-radius:14px;
                    background:{col}; color:white;
                    font-family:monospace; font-size:1.5rem; font-weight:800;
                    margin-bottom:14px; box-shadow:0 6px 16px {col}30;
                ">{SYMBOLS[lbl]}</div>
                <div style="font-size:13px; font-weight:600; color:{col};
                    text-transform:uppercase; letter-spacing:1px;">{lbl}</div>
                <div style="font-size:2.4rem; font-weight:800; color:{col}; margin:4px 0 6px;
                    letter-spacing:-1px;">{conf:.1%}</div>
                <div style="font-size:12px; color:#94a3b8;">
                    {model_name} &middot; {device.type.upper()} &middot; {ms:.0f} ms
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            sorted_scores = sorted(sc.items(), key=lambda x: x[1], reverse=True)
            bars_html = ""
            for l, s in sorted_scores:
                c = COLORS[l]
                pct = s * 100
                bars_html += f"""
                <div style="margin:10px 0;">
                    <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:5px;">
                        <span style="font-weight:600; color:{c};">[{SYMBOLS[l]}] {l.capitalize()}</span>
                        <span style="color:#64748b; font-weight:600; font-variant-numeric:tabular-nums;">{s:.1%}</span>
                    </div>
                    <div style="background:#f1f5f9; border-radius:8px; height:10px; overflow:hidden;">
                        <div style="background:linear-gradient(90deg,{c},{c}dd); width:{pct}%;
                            height:100%; border-radius:8px; transition:width 0.5s ease;"></div>
                    </div>
                </div>"""
            st.markdown(f'<div class="card" style="padding:18px 20px;">{bars_html}</div>', unsafe_allow_html=True)

    elif run:
        with result_container:
            st.warning("Please enter a headline first.")
    else:
        with result_container:
            st.markdown(f"""
            <div class="card card-muted" style="
                border:2px dashed #e2e8f0; text-align:center; padding:56px 24px;
            ">
                <div style="
                    width:52px; height:52px; border-radius:14px; background:#e2e8f0;
                    color:#94a3b8; display:inline-flex; align-items:center; justify-content:center;
                    font-size:22px; font-weight:700; margin-bottom:14px;
                ">?</div>
                <div style="font-size:15px; font-weight:700; color:#64748b;">Waiting for input</div>
                <div style="font-size:13px; color:#94a3b8; margin-top:6px;">
                    Select an example or type a headline
                </div>
                <div style="font-size:12px; color:#cbd5e1; margin-top:14px;
                    background:#f1f5f9; display:inline-block; padding:4px 12px; border-radius:6px;">
                    {model_name}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Compare All Models ────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-title">Compare All Models</div>', unsafe_allow_html=True)
    st.markdown(
        '<span style="color:#64748b; font-size:14px;">'
        "Run the same headline through all available models and compare their predictions side by side.</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    compare_text = headline.strip() if headline.strip() else None
    if compare_text:
        compare_btn = st.button("Compare all models", type="primary")
    else:
        compare_btn = False
        st.info("Enter a headline above to enable model comparison.")

    if compare_btn and compare_text:
        model_cols = st.columns(len(AVAILABLE_MODELS), gap="medium")
        all_results = {}

        for idx, (mname, mpath) in enumerate(AVAILABLE_MODELS.items()):
            with st.spinner(f"Running {mname}..."):
                m_pipe, m_dev = load_model(mname)
                t0 = time.time()
                res = predict(compare_text, m_pipe)
                ms = (time.time() - t0) * 1000
                all_results[mname] = res

            lbl, conf, sc = res["label"], res["confidence"], res["scores"]
            col = COLORS[lbl]

            with model_cols[idx]:
                # Result card
                st.markdown(f"""
                <div class="card" style="border-color:{col}30; text-align:center;
                    background:linear-gradient(160deg,{col}04,{col}10); padding:20px 16px;">
                    <div style="font-size:13px; font-weight:700; color:#0f172a; margin-bottom:10px;">{mname}</div>
                    <div style="display:inline-flex; align-items:center; justify-content:center;
                        width:44px; height:44px; border-radius:12px;
                        background:{col}; color:white;
                        font-family:monospace; font-size:1.2rem; font-weight:800;
                        margin-bottom:10px; box-shadow:0 4px 12px {col}30;">{SYMBOLS[lbl]}</div>
                    <div style="font-size:12px; font-weight:600; color:{col};
                        text-transform:uppercase; letter-spacing:1px;">{lbl}</div>
                    <div style="font-size:1.8rem; font-weight:800; color:{col}; margin:2px 0;
                        letter-spacing:-1px;">{conf:.1%}</div>
                    <div style="font-size:11px; color:#94a3b8;">{ms:.0f} ms</div>
                </div>
                """, unsafe_allow_html=True)

                # Score bars
                bars_html = ""
                for l, s in sorted(sc.items(), key=lambda x: x[1], reverse=True):
                    c = COLORS[l]
                    pct = s * 100
                    bars_html += f"""
                    <div style="margin:6px 0;">
                        <div style="display:flex; justify-content:space-between; font-size:11px; margin-bottom:3px;">
                            <span style="font-weight:600; color:{c};">[{SYMBOLS[l]}]</span>
                            <span style="color:#64748b; font-weight:600;">{s:.1%}</span>
                        </div>
                        <div style="background:#f1f5f9; border-radius:6px; height:6px; overflow:hidden;">
                            <div style="background:{c}; width:{pct}%; height:100%; border-radius:6px;"></div>
                        </div>
                    </div>"""
                st.markdown(f'<div class="card" style="padding:12px 14px; margin-top:8px;">{bars_html}</div>',
                            unsafe_allow_html=True)

        # LIME comparison
        st.markdown("---")
        st.markdown('<div class="section-title">Keyword Influence Analysis (LIME)</div>', unsafe_allow_html=True)
        st.markdown(
            '<span style="color:#64748b; font-size:14px;">'
            "Which words push each model toward its prediction? "
            "Green = supports, red = opposes.</span>",
            unsafe_allow_html=True,
        )

        lime_cols = st.columns(len(AVAILABLE_MODELS), gap="medium")
        for idx, (mname, _) in enumerate(AVAILABLE_MODELS.items()):
            with lime_cols[idx]:
                st.markdown(f"**{mname}**")
                with st.spinner("LIME..."):
                    try:
                        m_pipe, _ = load_model(mname)
                        res = all_results[mname]
                        ww = run_lime(compare_text, res["label"], m_pipe)
                        html = render_lime_html(compare_text, ww)
                        st.markdown(html, unsafe_allow_html=True)

                        if ww:
                            ww_df = pd.DataFrame(ww, columns=["word", "weight"]).sort_values("weight")
                            fig_lime = go.Figure(go.Bar(
                                x=ww_df["weight"], y=ww_df["word"], orientation="h",
                                marker_color=["#16a34a" if w > 0 else "#dc2626" for w in ww_df["weight"]],
                            ))
                            fig_lime.update_layout(
                                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                font=PLOTLY_LAYOUT["font"],
                                xaxis_title="", yaxis_title="",
                                height=max(200, len(ww_df) * 20),
                                margin=dict(t=8, b=24, l=8, r=8),
                            )
                            st.plotly_chart(fig_lime, use_container_width=True)
                    except Exception as e:
                        st.warning(f"LIME failed: {e}")


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — BATCH ANALYSIS
# ══════════════════════════════════════════════════════════════════════
elif page == "Batch Analysis":
    page_header(
        "Batch Analysis",
        f"Upload a CSV of financial headlines -- {model_name} will classify each one and produce a report.",
    )

    uploaded = st.file_uploader(
        "Upload CSV", type=["csv"],
        help="CSV must contain a column named 'headline'. One headline per row.",
    )

    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        headline_col = None
        for candidate in ["headline", "Headline", "HEADLINE", "text", "Text", "sentence", "title"]:
            if candidate in df_in.columns:
                headline_col = candidate
                break

        if headline_col is None:
            st.error(
                f"Could not find a headline column. Found: {list(df_in.columns)}. "
                "Rename the column to **headline**."
            )
            st.stop()

        headlines = df_in[headline_col].dropna().astype(str).tolist()
        st.info(f"Found **{len(headlines)}** headlines in column `{headline_col}`.")

        if st.button("Run batch analysis", type="primary"):
            results = []
            progress = st.progress(0, text="Analyzing headlines...")

            for i, h in enumerate(headlines):
                r = predict(h, pipe)
                results.append({
                    "Headline": h, "Sentiment": r["label"], "Confidence": r["confidence"],
                    "P(positive)": r["scores"].get("positive", 0),
                    "P(neutral)": r["scores"].get("neutral", 0),
                    "P(negative)": r["scores"].get("negative", 0),
                })
                progress.progress((i + 1) / len(headlines), text=f"Analyzing... {i+1}/{len(headlines)}")

            progress.empty()
            df_out = pd.DataFrame(results)
            st.success(f"Analysis complete. {len(df_out)} headlines classified.")
            st.markdown("---")

            counts = df_out["Sentiment"].value_counts()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total", len(df_out))
            c2.metric("[+] Positive", int(counts.get("positive", 0)))
            c3.metric("[~] Neutral", int(counts.get("neutral", 0)))
            c4.metric("[-] Negative", int(counts.get("negative", 0)))
            st.markdown("---")

            col_pie, col_hist = st.columns(2, gap="large")
            with col_pie:
                st.markdown('<div class="section-title">Sentiment Distribution</div>', unsafe_allow_html=True)
                fig_pie = go.Figure(go.Pie(
                    labels=[k.capitalize() for k in ["positive", "neutral", "negative"]],
                    values=[int(counts.get(k, 0)) for k in ["positive", "neutral", "negative"]],
                    marker=dict(colors=[COLORS["positive"], COLORS["neutral"], COLORS["negative"]]),
                    hole=0.45, textinfo="label+percent+value",
                ))
                fig_pie.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=300)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_hist:
                st.markdown('<div class="section-title">Confidence Distribution</div>', unsafe_allow_html=True)
                fig_hist = go.Figure()
                for sent in ["positive", "neutral", "negative"]:
                    mask = df_out["Sentiment"] == sent
                    if mask.any():
                        fig_hist.add_trace(go.Histogram(
                            x=df_out.loc[mask, "Confidence"], name=sent.capitalize(),
                            marker_color=COLORS[sent], opacity=0.7, nbinsx=20,
                        ))
                fig_hist.update_layout(
                    **PLOTLY_LAYOUT, barmode="overlay",
                    xaxis_title="Confidence", yaxis_title="Count", height=300,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown('<div class="section-title">Results Table</div>', unsafe_allow_html=True)
            st.dataframe(df_out, use_container_width=True, height=400)

            csv_buffer = io.StringIO()
            df_out.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv_buffer.getvalue(),
                file_name="sentiment_analysis_results.csv",
                mime="text/csv",
            )


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — LIVE NEWS
# ══════════════════════════════════════════════════════════════════════
elif page == "Live News":
    page_header(
        "Live News",
        "Enter a stock ticker to fetch the latest Yahoo Finance headlines and analyze their sentiment in real time.",
    )

    col_ticker, col_btn = st.columns([2, 1])
    with col_ticker:
        ticker_input = st.text_input(
            "Stock ticker", placeholder="e.g. AAPL, TSLA, MSFT, LVMH.PA",
            label_visibility="collapsed",
        )
    with col_btn:
        fetch_btn = st.button("Fetch & Analyze", type="primary", use_container_width=True)

    if fetch_btn and ticker_input.strip():
        ticker_sym = ticker_input.strip().upper()

        with st.spinner(f"Fetching news for {ticker_sym}..."):
            try:
                import yfinance as yf
                ticker = yf.Ticker(ticker_sym)
                news = ticker.news

                if not news:
                    st.warning(f"No news found for **{ticker_sym}**. Check the ticker symbol.")
                    st.stop()

                headlines = []
                for item in news:
                    if not isinstance(item, dict):
                        continue
                    title = None
                    content = item.get("content")
                    if isinstance(content, dict):
                        title = content.get("title")
                    if not title:
                        title = item.get("title") or item.get("headline")
                    if title and isinstance(title, str) and title.strip():
                        headlines.append(title.strip())

                if not headlines:
                    st.warning(f"No headlines could be extracted for **{ticker_sym}**.")
                    st.stop()

                st.info(f"Found **{len(headlines)}** headlines for **{ticker_sym}**.")
            except Exception as e:
                st.error(f"Failed to fetch news: {e}")
                st.stop()

        results = []
        progress = st.progress(0, text="Analyzing headlines...")
        for i, h in enumerate(headlines):
            r = predict(h, pipe)
            results.append({"Headline": h, "Sentiment": r["label"], "Confidence": r["confidence"]})
            progress.progress((i + 1) / len(headlines), text=f"Analyzing... {i+1}/{len(headlines)}")
        progress.empty()

        df_news = pd.DataFrame(results)
        st.markdown("---")

        counts = df_news["Sentiment"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Headlines", len(df_news))
        c2.metric("[+] Positive", int(counts.get("positive", 0)))
        c3.metric("[~] Neutral", int(counts.get("neutral", 0)))
        c4.metric("[-] Negative", int(counts.get("negative", 0)))

        pos_pct = counts.get("positive", 0) / len(df_news)
        neg_pct = counts.get("negative", 0) / len(df_news)
        sentiment_score = pos_pct - neg_pct

        if sentiment_score > 0.15:
            overall, overall_color = "Predominantly Positive", COLORS["positive"]
        elif sentiment_score < -0.15:
            overall, overall_color = "Predominantly Negative", COLORS["negative"]
        else:
            overall, overall_color = "Mixed / Neutral", COLORS["neutral"]

        st.markdown(
            f'<div class="card" style="text-align:center; padding:18px 24px; margin:16px 0;'
            f' border-left:4px solid {overall_color}; background:linear-gradient(135deg,{overall_color}06,{overall_color}10);">'
            f'<span style="font-size:18px; font-weight:700; color:{overall_color}; letter-spacing:-0.3px;">'
            f'{ticker_sym} -- {overall}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        col_chart, col_table = st.columns([1, 2], gap="large")
        with col_chart:
            st.markdown('<div class="section-title">Sentiment Distribution</div>', unsafe_allow_html=True)
            fig = go.Figure(go.Pie(
                labels=[k.capitalize() for k in ["positive", "neutral", "negative"]],
                values=[int(counts.get(k, 0)) for k in ["positive", "neutral", "negative"]],
                marker=dict(colors=[COLORS["positive"], COLORS["neutral"], COLORS["negative"]]),
                hole=0.45, textinfo="label+percent+value",
            ))
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.markdown('<div class="section-title">Headlines</div>', unsafe_allow_html=True)
            rows_html = ""
            for _, row in df_news.iterrows():
                sent = row["Sentiment"]
                conf = row["Confidence"]
                c = COLORS[sent]
                rows_html += (
                    f'<div style="padding:10px 14px; border-bottom:1px solid #f1f5f9; display:flex; align-items:center; gap:10px;">'
                    f'<span style="color:{c}; font-weight:700; font-family:monospace; font-size:13px; flex-shrink:0;">[{SYMBOLS[sent]}]</span>'
                    f'<span style="color:#1e293b; font-size:14px; flex:1;">{row["Headline"]}</span>'
                    f'<span style="color:#94a3b8; font-size:12px; font-weight:600; flex-shrink:0; font-variant-numeric:tabular-nums;">{conf:.0%}</span></div>'
                )
            st.markdown(f'<div class="card" style="padding:0; overflow:hidden;">{rows_html}</div>', unsafe_allow_html=True)

    elif fetch_btn:
        st.warning("Please enter a ticker symbol.")


# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL LEADERBOARD
# ══════════════════════════════════════════════════════════════════════
elif page == "Model Leaderboard":
    page_header(
        "Model Leaderboard",
        "All models trained on the same 70/10/20 stratified split (seed=42). Ranked by macro-F1.",
    )

    if lb.empty:
        st.warning("No results found. Run training scripts first.")
    else:
        best = lb.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Model", best["model"])
        c2.metric("Best Macro-F1", f"{best['f1_macro']:.1%}")
        c3.metric("Best Accuracy", f"{best['accuracy']:.1%}")
        c4.metric("Models Benchmarked", len(lb))
        st.markdown("---")

        TYPE_COLORS = {"transformer": "#2563eb", "classical": "#f59e0b", "deep_learning": "#dc2626"}

        col_bar, col_scatter = st.columns(2, gap="large")
        with col_bar:
            st.markdown('<div class="section-title">Macro-F1 by Model</div>', unsafe_allow_html=True)
            fig = go.Figure()
            for _, row in lb.iterrows():
                c = TYPE_COLORS.get(row.get("type", ""), "#888")
                fig.add_trace(go.Bar(
                    x=[row["model"]], y=[row["f1_macro"]], marker_color=c,
                    text=[f"{row['f1_macro']:.1%}"], textposition="outside", showlegend=False,
                ))
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=PLOTLY_LAYOUT["font"],
                yaxis=dict(range=[0, 1.05], tickformat=".0%", gridcolor="#f1f5f9"),
                xaxis=dict(tickangle=-15, gridcolor="#f1f5f9"),
                margin=PLOTLY_LAYOUT["margin"], height=320, bargap=0.35,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_scatter:
            st.markdown('<div class="section-title">Accuracy vs Macro-F1</div>', unsafe_allow_html=True)
            fig2 = go.Figure()
            for _, row in lb.iterrows():
                c = TYPE_COLORS.get(row.get("type", ""), "#888")
                fig2.add_trace(go.Scatter(
                    x=[row["accuracy"]], y=[row["f1_macro"]],
                    mode="markers+text", marker=dict(size=14, color=c),
                    text=[row["model"]], textposition="top center",
                    textfont=dict(size=9), showlegend=False,
                ))
            fig2.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=PLOTLY_LAYOUT["font"],
                xaxis=dict(title="Accuracy", tickformat=".0%", gridcolor="#f1f5f9", range=[0.5, 0.97]),
                yaxis=dict(title="Macro-F1", tickformat=".0%", gridcolor="#f1f5f9", range=[0.2, 1.0]),
                margin=dict(t=16, b=40, l=40, r=12), height=320,
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            '<div style="text-align:center; padding:8px 0; font-size:13px; color:#64748b;">'
            "<span style='color:#2563eb'>&#9632;</span> Transformer &nbsp;&nbsp;&nbsp;"
            "<span style='color:#f59e0b'>&#9632;</span> Classical ML &nbsp;&nbsp;&nbsp;"
            "<span style='color:#dc2626'>&#9632;</span> Deep Learning</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.markdown('<div class="section-title">Full Benchmark Table</div>', unsafe_allow_html=True)
        display = lb.copy()
        type_map = {"classical": "Classical ML", "deep_learning": "Deep Learning", "transformer": "Transformer"}
        display["type"] = display["type"].map(type_map).fillna(display["type"])
        for col_name in ["accuracy", "f1_macro", "f1_weighted"]:
            if col_name in display.columns:
                display[col_name] = display[col_name].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "--")
        if "train_time_s" in display.columns:
            display["train_time_s"] = display["train_time_s"].apply(
                lambda x: f"{x:.0f}s" if pd.notna(x) and x > 0 else "<1s"
            )
        cols_show = {"model": "Model", "type": "Category", "accuracy": "Accuracy",
                     "f1_macro": "Macro-F1", "f1_weighted": "Weighted-F1",
                     "params": "Parameters", "train_time_s": "Train Time"}
        display = display[[c for c in cols_show if c in display.columns]].rename(columns=cols_show)
        st.dataframe(display, use_container_width=True)
        st.caption(f"Active model (currently serving): {model_name}")


# ══════════════════════════════════════════════════════════════════════
# PAGE 5 — RENDU
# ══════════════════════════════════════════════════════════════════════
elif page == "Rendu":
    page_header(
        "Rendu",
        "Slides de presentation et rapport ecrit du projet.",
    )

    _PDF_FILES = [
        ("Slides de presentation", Path("presentation/presentation_deep_nlp.pdf")),
        ("Rapport", Path("rapport/rapport_deep_nlp.pdf")),
    ]

    for _pdf_title, _pdf_path in _PDF_FILES:
        if _pdf_path.exists():
            _pdf_bytes = _pdf_path.read_bytes()
            _pdf_b64 = base64.b64encode(_pdf_bytes).decode()
            _uid = _pdf_title.replace(" ", "_").lower()
            components.html(
                f"""
                <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
                <style>
                    * {{ margin:0; padding:0; box-sizing:border-box; }}
                    body {{ background:transparent; }}
                    #wrap_{_uid} {{
                        background:#1e293b;
                        border-radius:10px;
                        overflow:hidden;
                        border:1px solid #334155;
                        font-family:Inter,system-ui,sans-serif;
                    }}
                    #bar_{_uid} {{
                        display:flex; justify-content:space-between; align-items:center;
                        padding:8px 16px; background:#0f172a;
                        border-bottom:1px solid #334155;
                    }}
                    #bar_{_uid} .title {{ color:#94a3b8; font-size:13px; font-weight:500; }}
                    #bar_{_uid} .nav {{ display:flex; align-items:center; gap:6px; }}
                    #bar_{_uid} .nav button {{
                        background:#334155; color:#e2e8f0; border:none;
                        padding:5px 12px; border-radius:5px; cursor:pointer;
                        font-size:12px; font-family:inherit;
                    }}
                    #bar_{_uid} .nav button:disabled {{ opacity:0.35; cursor:default; }}
                    #bar_{_uid} .nav span {{ color:#94a3b8; font-size:12px; min-width:60px; text-align:center; }}
                    #bar_{_uid} .fs {{
                        background:linear-gradient(135deg,#6366f1,#8b5cf6);
                        color:#fff; border:none; padding:6px 14px;
                        border-radius:6px; cursor:pointer;
                        font-size:12px; font-weight:500; font-family:inherit;
                    }}
                    #view_{_uid} {{
                        height:550px; overflow:auto;
                        display:flex; justify-content:center; align-items:flex-start;
                        background:#4a5568;
                    }}
                    #view_{_uid} canvas {{ display:block; }}
                </style>
                <div id="wrap_{_uid}">
                    <div id="bar_{_uid}">
                        <span class="title">{_pdf_title}</span>
                        <div class="nav">
                            <button id="prev_{_uid}" onclick="go_{_uid}(-1)">Prev</button>
                            <span id="info_{_uid}">...</span>
                            <button id="next_{_uid}" onclick="go_{_uid}(1)">Next</button>
                        </div>
                        <button class="fs" onclick="full_{_uid}()">Fullscreen</button>
                    </div>
                    <div id="view_{_uid}">
                        <canvas id="cv_{_uid}"></canvas>
                    </div>
                </div>
                <script>
                (function() {{
                    var pdfjsLib = window["pdfjs-dist/build/pdf"];
                    pdfjsLib.GlobalWorkerOptions.workerSrc =
                        "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

                    var pdfDoc = null, pg = 1, rendering = false, pending = null;
                    var cv = document.getElementById("cv_{_uid}");
                    var ctx = cv.getContext("2d");
                    var viewer = document.getElementById("view_{_uid}");

                    function render(num) {{
                        rendering = true;
                        pdfDoc.getPage(num).then(function(page) {{
                            var w = viewer.clientWidth - 16;
                            var vp1 = page.getViewport({{ scale: 1 }});
                            var scale = w / vp1.width;
                            var vp = page.getViewport({{ scale: scale }});
                            cv.width = vp.width;
                            cv.height = vp.height;
                            page.render({{ canvasContext: ctx, viewport: vp }}).promise.then(function() {{
                                rendering = false;
                                if (pending !== null) {{ render(pending); pending = null; }}
                            }});
                            document.getElementById("info_{_uid}").textContent =
                                num + " / " + pdfDoc.numPages;
                            document.getElementById("prev_{_uid}").disabled = (num <= 1);
                            document.getElementById("next_{_uid}").disabled = (num >= pdfDoc.numPages);
                        }});
                    }}

                    window.go_{_uid} = function(d) {{
                        var n = pg + d;
                        if (n < 1 || n > pdfDoc.numPages) return;
                        pg = n;
                        if (rendering) {{ pending = pg; }} else {{ render(pg); }}
                    }};

                    window.full_{_uid} = function() {{
                        var el = document.getElementById("wrap_{_uid}");
                        if (!document.fullscreenElement) {{
                            el.requestFullscreen().then(function() {{
                                viewer.style.height = "calc(100vh - 45px)";
                                setTimeout(function() {{ render(pg); }}, 150);
                            }});
                        }} else {{
                            document.exitFullscreen();
                        }}
                    }};

                    document.addEventListener("fullscreenchange", function() {{
                        if (!document.fullscreenElement) {{
                            viewer.style.height = "550px";
                            setTimeout(function() {{ render(pg); }}, 150);
                        }}
                    }});

                    // Decode base64 and load
                    var raw = atob("{_pdf_b64}");
                    var arr = new Uint8Array(raw.length);
                    for (var i = 0; i < raw.length; i++) arr[i] = raw.charCodeAt(i);
                    pdfjsLib.getDocument({{ data: arr }}).promise.then(function(pdf) {{
                        pdfDoc = pdf;
                        render(1);
                    }});
                }})();
                </script>
                """,
                height=610,
            )
            st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        else:
            st.warning(f"Fichier introuvable : {_pdf_path}")


# ══════════════════════════════════════════════════════════════════════
# PAGE 7 — PROJECT
# ══════════════════════════════════════════════════════════════════════
elif page == "Project":
    page_header(
        "Project",
        "Projet Deep Learning / NLP -- Universite Paris 1 Pantheon-Sorbonne.",
    )

    col_a, col_b = st.columns([3, 2], gap="large")
    with col_a:
        st.markdown('<div class="section-title">Le projet</div>', unsafe_allow_html=True)
        st.markdown("""
        Objectif : comparer plusieurs approches NLP pour la classification de sentiment
        sur des textes financiers, en illustrant la progression des performances de la
        generation bag-of-words aux transformers specialises.
        """)

        st.markdown('<div class="section-title">Limitations</div>', unsafe_allow_html=True)
        st.warning("""
        - Dataset petit (~4 800 exemples) -> generalisation limitee
        - News europeennes des annees 2000-2010
        - Ne pas utiliser pour des decisions financieres reelles
        - Performances reduites hors domaine (reseaux sociaux, transcriptions d'appels)
        """)

        st.markdown('<div class="section-title">References</div>', unsafe_allow_html=True)
        for ref in [
            "Malo et al. (2014). *Good debt or bad debt.* JASIST, 65(4).",
            "Araci (2019). *FinBERT.* arXiv:1908.10063.",
            "Devlin et al. (2019). *BERT.* NAACL-HLT.",
            "Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.",
        ]:
            st.markdown(f"- {ref}")

    with col_b:
        st.markdown('<div class="section-title">Dataset -- FinancialPhraseBank</div>', unsafe_allow_html=True)
        st.markdown("Malo et al. (2014) -- 4 846 phrases en anglais, annotees par 16 experts.")

        c1, c2 = st.columns(2)
        c1.metric("Phrases", "4 846")
        c2.metric("Classes", "3")

        fig = go.Figure(go.Pie(
            labels=["Neutral (59.4%)", "Positive (28.1%)", "Negative (12.5%)"],
            values=[2879, 1363, 604],
            marker=dict(colors=[COLORS["neutral"], COLORS["positive"], COLORS["negative"]]),
            hole=0.45, textinfo="label+percent",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=260)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Split utilise</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Split": ["Train", "Val", "Test"],
            "Samples": [3391, 485, 970],
            "Share": ["70%", "10%", "20%"],
        }), use_container_width=True, hide_index=True)
