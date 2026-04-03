#!/usr/bin/env python
"""Gradio demo — Financial News Sentiment Analysis.

Usage:
    python app.py
"""

import json
import sys
import time
from pathlib import Path

import gradio as gr
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import LABEL_NAMES, NUM_CLASSES, RESULTS_DIR
from src.inference import build_pipeline, get_device, load_best_model, predict

# ── Load model at startup ─────────────────────────────────────────────

print("Loading model...")
model, tokenizer, model_name = load_best_model()
device = get_device()
pipe = build_pipeline(model, tokenizer, device)
print(f"Ready — serving {model_name}")

# ── Load leaderboard data ─────────────────────────────────────────────

_leaderboard_path = RESULTS_DIR / "leaderboard.json"
if _leaderboard_path.exists():
    _leaderboard_raw = json.loads(_leaderboard_path.read_text())
    _leaderboard_raw.sort(key=lambda x: x.get("f1_macro", 0), reverse=True)
else:
    _leaderboard_raw = []

# ── Sentiment visual markers (no emojis — using ASCII/Unicode symbols) ─

MARKERS = {
    "positive": {"symbol": "+", "color": "#1a7a3a", "label": "Positive"},
    "neutral":  {"symbol": "~", "color": "#2c5ea0", "label": "Neutral"},
    "negative": {"symbol": "-", "color": "#a82a2a", "label": "Negative"},
}

# ── Predict function ──────────────────────────────────────────────────

def analyze(text: str):
    if not text or not text.strip():
        return (
            _empty_verdict(),
            _empty_bars(),
            "",
        )

    t0 = time.time()
    result = predict(text, pipe)
    latency_ms = (time.time() - t0) * 1000

    verdict_html = _build_verdict(result["label"], result["confidence"])
    bars_html = _build_bars(result["scores"])
    meta = f"Inference: {latency_ms:.0f} ms  ·  Model: {model_name}  ·  Device: {device.type.upper()}"

    return verdict_html, bars_html, meta


def _empty_verdict():
    return '<div style="text-align:center; color:#888; padding:24px 0;">Enter a headline above to begin analysis.</div>'


def _empty_bars():
    return ""


def _build_verdict(label: str, confidence: float):
    m = MARKERS.get(label, MARKERS["neutral"])
    pct = f"{confidence:.1%}"
    return f"""
    <div style="text-align:center; padding:20px 0;">
        <div style="
            display:inline-block;
            width:56px; height:56px; line-height:56px;
            border-radius:50%;
            background:{m['color']};
            color:#fff;
            font-size:28px; font-weight:700;
            font-family: 'Courier New', monospace;
            margin-bottom:12px;
        ">{m['symbol']}</div>
        <div style="font-size:22px; font-weight:600; color:{m['color']}; margin-top:8px;">
            {m['label']}
        </div>
        <div style="font-size:14px; color:#666; margin-top:4px;">
            Confidence: {pct}
        </div>
    </div>
    """


def _build_bars(scores: dict):
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    rows = []
    for label, score in ordered:
        m = MARKERS.get(label, MARKERS["neutral"])
        pct = score * 100
        rows.append(f"""
        <div style="margin-bottom:10px;">
            <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:3px;">
                <span style="font-weight:500;">[{m['symbol']}] {m['label']}</span>
                <span style="color:#555;">{pct:.1f}%</span>
            </div>
            <div style="background:#e8e8e8; border-radius:4px; height:10px; overflow:hidden;">
                <div style="
                    width:{pct:.1f}%;
                    height:100%;
                    background:{m['color']};
                    border-radius:4px;
                    transition: width 0.4s ease;
                "></div>
            </div>
        </div>
        """)
    return "\n".join(rows)


# ── Leaderboard table builder ─────────────────────────────────────────

def _build_leaderboard_html():
    if not _leaderboard_raw:
        return "<p>No benchmark results found. Run training scripts first.</p>"

    rows = []
    for i, entry in enumerate(_leaderboard_raw):
        rank = i + 1
        name = entry["model"]
        acc = entry.get("accuracy", 0)
        f1 = entry.get("f1_macro", 0)
        model_type = entry.get("type", "")
        params = entry.get("params", "--")
        train_t = entry.get("train_time_s", None)

        type_labels = {
            "classical": "Classical ML",
            "deep_learning": "Deep Learning",
            "transformer": "Transformer",
        }
        type_display = type_labels.get(model_type, model_type)

        train_display = f"{train_t:.0f}s" if train_t and train_t > 0 else "<1s"

        is_active = name == model_name
        highlight = ' style="background:#f0f4ff; font-weight:600;"' if is_active else ""
        active_marker = ' (active)' if is_active else ""

        rows.append(f"""
            <tr{highlight}>
                <td style="text-align:center;">{rank}</td>
                <td>{name}{active_marker}</td>
                <td style="text-align:center;">{type_display}</td>
                <td style="text-align:center;">{acc:.1%}</td>
                <td style="text-align:center; font-weight:600;">{f1:.1%}</td>
                <td style="text-align:center;">{params}</td>
                <td style="text-align:center;">{train_display}</td>
            </tr>
        """)

    return f"""
    <table style="width:100%; border-collapse:collapse; font-size:13px;">
        <thead>
            <tr style="border-bottom:2px solid #333; text-align:center;">
                <th style="padding:8px 4px;">#</th>
                <th style="padding:8px 4px; text-align:left;">Model</th>
                <th style="padding:8px 4px;">Category</th>
                <th style="padding:8px 4px;">Accuracy</th>
                <th style="padding:8px 4px;">Macro-F1</th>
                <th style="padding:8px 4px;">Parameters</th>
                <th style="padding:8px 4px;">Train Time</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>
    <p style="font-size:11px; color:#888; margin-top:8px;">
        Ranked by macro-F1 (accounts for class imbalance). The active model is highlighted.
    </p>
    """


# ── Dataset overview ──────────────────────────────────────────────────

DATASET_HTML = """
<div style="font-size:13px; line-height:1.7;">
    <p>
        <strong>FinancialPhraseBank</strong> is a dataset of 4,846 English-language sentences
        from financial news articles, annotated for sentiment by 16 domain experts with
        backgrounds in finance and business.
    </p>
    <p>
        The annotation task asked whether a given sentence would influence an investor's
        perception of the mentioned company in a positive, negative, or neutral direction.
        Only sentences with majority annotator agreement are included.
    </p>

    <table style="width:100%; border-collapse:collapse; margin:12px 0;">
        <thead>
            <tr style="border-bottom:2px solid #333;">
                <th style="padding:6px 8px; text-align:left;">Class</th>
                <th style="padding:6px 8px; text-align:center;">Count</th>
                <th style="padding:6px 8px; text-align:center;">Share</th>
                <th style="padding:6px 8px; text-align:left;">Distribution</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="padding:4px 8px;">[~] Neutral</td>
                <td style="text-align:center;">2,879</td>
                <td style="text-align:center;">59.4%</td>
                <td>
                    <div style="background:#e8e8e8; border-radius:3px; height:8px; width:100%;">
                        <div style="width:59.4%; height:100%; background:#2c5ea0; border-radius:3px;"></div>
                    </div>
                </td>
            </tr>
            <tr>
                <td style="padding:4px 8px;">[+] Positive</td>
                <td style="text-align:center;">1,363</td>
                <td style="text-align:center;">28.1%</td>
                <td>
                    <div style="background:#e8e8e8; border-radius:3px; height:8px; width:100%;">
                        <div style="width:28.1%; height:100%; background:#1a7a3a; border-radius:3px;"></div>
                    </div>
                </td>
            </tr>
            <tr>
                <td style="padding:4px 8px;">[-] Negative</td>
                <td style="text-align:center;">604</td>
                <td style="text-align:center;">12.5%</td>
                <td>
                    <div style="background:#e8e8e8; border-radius:3px; height:8px; width:100%;">
                        <div style="width:12.5%; height:100%; background:#a82a2a; border-radius:3px;"></div>
                    </div>
                </td>
            </tr>
        </tbody>
    </table>

    <p>
        The significant class imbalance (neutral sentences dominate) makes
        <strong>macro-F1</strong> the appropriate primary metric, as it weights each class
        equally regardless of sample size. A model that simply predicts "neutral" every time
        would score ~59% accuracy but near-zero macro-F1.
    </p>

    <p style="font-size:12px; color:#666; margin-top:8px;">
        Source: Malo et al. (2014), "Good debt or bad debt: Detecting semantic orientations in economic texts."
        Journal of the Association for Information Science and Technology, 65(4), 782-796.
    </p>
</div>
"""

# ── Methodology section ───────────────────────────────────────────────

METHODOLOGY_HTML = """
<div style="font-size:13px; line-height:1.7;">
    <p>
        This project benchmarks three generations of NLP approaches on the same dataset
        and evaluation protocol, making the progression in capability visible and measurable.
    </p>

    <h4 style="margin:16px 0 6px; border-bottom:1px solid #ddd; padding-bottom:4px;">
        Classical Machine Learning
    </h4>
    <p>
        <strong>TF-IDF + Logistic Regression / Naive Bayes</strong> represent the bag-of-words
        paradigm. Text is converted to sparse numerical vectors using term frequency-inverse
        document frequency weighting (bigrams, 20K features, sublinear TF). A linear classifier
        then maps these vectors to sentiment labels. These methods are fast and interpretable
        but cannot capture word order, context, or semantic meaning beyond surface-level
        co-occurrence.
    </p>

    <h4 style="margin:16px 0 6px; border-bottom:1px solid #ddd; padding-bottom:4px;">
        Deep Learning
    </h4>
    <p>
        <strong>Bidirectional LSTM</strong> processes text as a sequence, learning to weight
        each token based on its surrounding context in both directions. Unlike TF-IDF,
        it can capture word order and local dependencies. However, with only ~5K training
        samples and no pre-trained knowledge, the LSTM struggles to generalize, particularly
        on the minority negative class.
    </p>

    <h4 style="margin:16px 0 6px; border-bottom:1px solid #ddd; padding-bottom:4px;">
        Transformer Models
    </h4>
    <p>
        <strong>DistilBERT</strong> and <strong>BERT</strong> are general-purpose language models
        pre-trained on large English corpora (Wikipedia, BookCorpus). They bring rich linguistic
        knowledge that can be fine-tuned for downstream tasks with relatively little labeled data.
    </p>
    <p>
        <strong>FinBERT</strong> extends this idea with <em>domain-specific pre-training</em>:
        it was further trained on a large corpus of financial communications (earnings calls,
        analyst reports, financial news) before being fine-tuned here. This domain knowledge
        gives it an edge in understanding financial language nuances — terms like "restructuring"
        or "write-down" carry implicit sentiment that general models may not fully capture.
    </p>

    <h4 style="margin:16px 0 6px; border-bottom:1px solid #ddd; padding-bottom:4px;">
        Evaluation Protocol
    </h4>
    <p>
        All models are evaluated on the same held-out test set (20% of data, stratified split,
        seed 42). The primary metric is <strong>macro-averaged F1</strong>, which computes F1
        independently for each class and averages them, giving equal importance to the minority
        negative class (12.5% of data). Accuracy is reported alongside for reference.
    </p>
</div>
"""

# ── Example headlines, grouped by expected sentiment ──────────────────

EXAMPLES_POSITIVE = [
    "The company announced record profits and raised its annual dividend by 15%.",
    "Net sales in the third quarter increased by 12% year-on-year to EUR 205 million.",
    "Operating profit rose to EUR 13.1 million, representing 7.7% of net sales.",
    "The board approved a share buyback program worth USD 500 million.",
]

EXAMPLES_NEUTRAL = [
    "Revenue remained flat compared to the previous quarter.",
    "The company plans to open two new offices in the Nordic region by 2027.",
    "Annual general meeting will take place on March 15 at the Helsinki headquarters.",
    "The firm employs approximately 3,200 people across 12 countries.",
]

EXAMPLES_NEGATIVE = [
    "CEO resigned amid an accounting fraud scandal, sending shares tumbling 30%.",
    "Acquisition talks with a major competitor have reportedly collapsed.",
    "The firm cut 500 jobs as part of its ongoing restructuring plan.",
    "Operating loss widened to EUR 8 million due to declining demand.",
]


# ── Custom CSS ────────────────────────────────────────────────────────

CUSTOM_CSS = """
    .main-title {
        font-size: 26px;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 2px;
        letter-spacing: -0.3px;
    }
    .subtitle {
        font-size: 14px;
        color: #555;
        margin-bottom: 16px;
    }
    .section-header {
        font-size: 15px;
        font-weight: 600;
        color: #1a1a2e;
        margin: 8px 0 4px;
        padding-bottom: 4px;
        border-bottom: 1px solid #e0e0e0;
    }
    .info-text {
        font-size: 12px;
        color: #777;
        margin-top: 4px;
    }
    footer { display: none !important; }
"""


# ── Build the interface ───────────────────────────────────────────────

with gr.Blocks(
    title="Financial Sentiment Analysis",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.slate,
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ),
) as demo:

    # ── Header ─────────────────────────────────────────────────────
    gr.HTML("""
        <div style="padding:8px 0 4px;">
            <div class="main-title">Financial News Sentiment Analysis</div>
            <div class="subtitle">
                Classifying financial headlines as positive, neutral, or negative
                using transformer-based language models fine-tuned on FinancialPhraseBank.
            </div>
        </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Analyze ─────────────────────────────────────────
        with gr.Tab("Analyze"):

            with gr.Row(equal_height=False):

                with gr.Column(scale=3):
                    gr.HTML('<div class="section-header">Input</div>')
                    inp = gr.Textbox(
                        label="Financial news headline",
                        placeholder="Paste or type a financial news headline here...",
                        lines=3,
                        show_label=False,
                    )
                    with gr.Row():
                        btn = gr.Button("Analyze", variant="primary", scale=2)
                        clear_btn = gr.Button("Clear", variant="secondary", scale=1)

                    meta_display = gr.Textbox(
                        label="",
                        interactive=False,
                        show_label=False,
                        container=False,
                        elem_classes=["info-text"],
                    )

                with gr.Column(scale=2):
                    gr.HTML('<div class="section-header">Result</div>')
                    verdict_display = gr.HTML(value=_empty_verdict())
                    gr.HTML('<div class="section-header">Class Probabilities</div>')
                    bars_display = gr.HTML(value=_empty_bars())

            # ── Examples ───────────────────────────────────────────
            gr.HTML('<div class="section-header" style="margin-top:16px;">Example Headlines</div>')

            with gr.Row():
                with gr.Column():
                    gr.HTML('<div style="font-size:12px; color:#1a7a3a; font-weight:600; margin-bottom:4px;">[+] Typically Positive</div>')
                    for ex in EXAMPLES_POSITIVE:
                        gr.Button(ex, variant="secondary", size="sm").click(
                            fn=lambda t=ex: analyze(t), outputs=[verdict_display, bars_display, meta_display]
                        )
                with gr.Column():
                    gr.HTML('<div style="font-size:12px; color:#2c5ea0; font-weight:600; margin-bottom:4px;">[~] Typically Neutral</div>')
                    for ex in EXAMPLES_NEUTRAL:
                        gr.Button(ex, variant="secondary", size="sm").click(
                            fn=lambda t=ex: analyze(t), outputs=[verdict_display, bars_display, meta_display]
                        )
                with gr.Column():
                    gr.HTML('<div style="font-size:12px; color:#a82a2a; font-weight:600; margin-bottom:4px;">[-] Typically Negative</div>')
                    for ex in EXAMPLES_NEGATIVE:
                        gr.Button(ex, variant="secondary", size="sm").click(
                            fn=lambda t=ex: analyze(t), outputs=[verdict_display, bars_display, meta_display]
                        )

        # ── Tab 2: Leaderboard ─────────────────────────────────────
        with gr.Tab("Model Leaderboard"):
            gr.HTML("""
                <div style="padding:8px 0;">
                    <div class="section-header">Benchmark Comparison</div>
                    <p style="font-size:13px; color:#555; margin:4px 0 12px;">
                        All models trained and evaluated on the same data split
                        (70/10/20 stratified, seed 42) using identical preprocessing
                        where applicable. Transformer models fine-tuned for 5 epochs
                        with early stopping.
                    </p>
                </div>
            """)
            gr.HTML(_build_leaderboard_html())

        # ── Tab 3: Dataset ─────────────────────────────────────────
        with gr.Tab("Dataset"):
            gr.HTML(f"""
                <div style="padding:8px 0;">
                    <div class="section-header">FinancialPhraseBank</div>
                </div>
            """)
            gr.HTML(DATASET_HTML)

        # ── Tab 4: Methodology ─────────────────────────────────────
        with gr.Tab("Methodology"):
            gr.HTML(f"""
                <div style="padding:8px 0;">
                    <div class="section-header">Approach and Model Comparison</div>
                </div>
            """)
            gr.HTML(METHODOLOGY_HTML)

        # ── Tab 5: About ───────────────────────────────────────────
        with gr.Tab("About"):
            gr.HTML(f"""
                <div style="padding:8px 0; font-size:13px; line-height:1.7;">
                    <div class="section-header">About This Project</div>
                    <p>
                        This application is part of a sentiment analysis study conducted
                        at Universite Paris 1 Pantheon-Sorbonne, benchmarking multiple NLP
                        approaches — from classical machine learning to domain-specific
                        transformer models — on financial text classification.
                    </p>
                    <p>
                        The goal is both practical and pedagogical: to produce a working
                        sentiment classifier for financial headlines, and to illustrate
                        how successive generations of NLP techniques (bag-of-words,
                        recurrent networks, pre-trained transformers, domain-adapted
                        transformers) compare on the same task under controlled conditions.
                    </p>

                    <div class="section-header" style="margin-top:16px;">Technical Details</div>
                    <table style="font-size:13px; border-collapse:collapse; width:100%;">
                        <tr><td style="padding:4px 8px; color:#666; width:160px;">Active model</td>
                            <td style="padding:4px 8px; font-weight:500;">{model_name}</td></tr>
                        <tr><td style="padding:4px 8px; color:#666;">Inference device</td>
                            <td style="padding:4px 8px;">{device.type.upper()}</td></tr>
                        <tr><td style="padding:4px 8px; color:#666;">Max input length</td>
                            <td style="padding:4px 8px;">128 tokens</td></tr>
                        <tr><td style="padding:4px 8px; color:#666;">Training split</td>
                            <td style="padding:4px 8px;">70% train / 10% validation / 20% test</td></tr>
                        <tr><td style="padding:4px 8px; color:#666;">Primary metric</td>
                            <td style="padding:4px 8px;">Macro-averaged F1 score</td></tr>
                        <tr><td style="padding:4px 8px; color:#666;">Framework</td>
                            <td style="padding:4px 8px;">PyTorch + HuggingFace Transformers</td></tr>
                    </table>

                    <div class="section-header" style="margin-top:16px;">Limitations</div>
                    <p>
                        This model was trained on a relatively small dataset of financial
                        sentences (~4,800 samples). It performs well on headlines similar in
                        style and domain to the training data but may not generalize to other
                        text types (social media posts, earnings call transcripts, non-English
                        text). Predictions should be treated as one input among many in any
                        decision-making process, not as definitive assessments.
                    </p>

                    <div class="section-header" style="margin-top:16px;">References</div>
                    <ul style="font-size:12px; color:#555;">
                        <li>Malo, P. et al. (2014). "Good debt or bad debt: Detecting semantic orientations
                            in economic texts." JASIST, 65(4), 782-796.</li>
                        <li>Araci, D. (2019). "FinBERT: Financial Sentiment Analysis with Pre-Trained
                            Language Models." arXiv:1908.10063.</li>
                        <li>Devlin, J. et al. (2019). "BERT: Pre-training of Deep Bidirectional
                            Transformers for Language Understanding." NAACL-HLT.</li>
                    </ul>
                </div>
            """)

    # ── Wire up events ─────────────────────────────────────────────
    btn.click(fn=analyze, inputs=inp, outputs=[verdict_display, bars_display, meta_display])
    inp.submit(fn=analyze, inputs=inp, outputs=[verdict_display, bars_display, meta_display])
    clear_btn.click(
        fn=lambda: ("", _empty_verdict(), _empty_bars(), ""),
        outputs=[inp, verdict_display, bars_display, meta_display],
    )


if __name__ == "__main__":
    demo.launch(share=False)
