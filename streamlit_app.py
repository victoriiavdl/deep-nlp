"""Streamlit app — Financial News Sentiment Analysis."""

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_javascript import st_javascript

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import RESULTS_DIR
from src.inference import build_pipeline, get_device, load_best_model, predict

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Sentiment Analysis",
    page_icon="F",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS  = {"positive": "#16a34a", "neutral": "#2563eb", "negative": "#dc2626"}
SYMBOLS = {"positive": "+", "neutral": "~", "negative": "-"}

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

# ── Model loading ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading FinBERT model...")
def load_model():
    model, tokenizer, name = load_best_model()
    device = get_device()
    pipe = build_pipeline(model, tokenizer, device)
    return pipe, name, device

pipe, model_name, device = load_model()

# ── Leaderboard data ──────────────────────────────────────────────────
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

# ── LIME explainer (cached) ──────────────────────────────────────────
@st.cache_resource
def get_lime_explainer():
    from lime.lime_text import LimeTextExplainer
    return LimeTextExplainer(class_names=["positive", "neutral", "negative"])

def lime_predict_proba(texts):
    """Adapter: LIME expects a function that returns (n_samples, n_classes) array."""
    results = []
    for text in texts:
        out = pipe(text[:512], top_k=None)[0]
        score_map = {r["label"]: r["score"] for r in out}
        results.append([
            score_map.get("positive", 0),
            score_map.get("neutral", 0),
            score_map.get("negative", 0),
        ])
    return np.array(results)

def run_lime(text, label):
    """Run LIME and return list of (word, weight) tuples."""
    explainer = get_lime_explainer()
    label_idx = {"positive": 0, "neutral": 1, "negative": 2}[label]
    exp = explainer.explain_instance(
        text, lime_predict_proba,
        num_features=15, num_samples=200,
        labels=[label_idx],
    )
    return exp.as_list(label=label_idx)

def render_lime_html(text, word_weights, label):
    """Build HTML with words highlighted by importance."""
    weight_map = {}
    for word, weight in word_weights:
        w_lower = word.lower()
        weight_map[w_lower] = weight

    tokens = text.split()
    html_parts = []
    for token in tokens:
        clean = token.strip(".,;:!?\"'()-").lower()
        w = weight_map.get(clean, 0)
        if abs(w) < 0.01:
            html_parts.append(f'<span style="padding:1px 2px;">{token}</span>')
        else:
            intensity = min(abs(w) * 5, 0.8)
            if w > 0:
                bg = f"rgba(22, 163, 74, {intensity})"
            else:
                bg = f"rgba(220, 38, 38, {intensity})"
            html_parts.append(
                f'<span style="background:{bg}; padding:1px 4px; border-radius:3px; '
                f'margin:0 1px;">{token}</span>'
            )

    return (
        f'<div style="font-size:15px; line-height:2; padding:8px 0;">'
        + " ".join(html_parts)
        + '</div>'
    )


# ── Web Speech API component ─────────────────────────────────────────

SPEECH_HTML = """
<div id="speech-container" style="text-align:center;">
    <button id="speech-btn" onclick="toggleSpeech()" style="
        background: #2563eb; color: white; border: none; border-radius: 50%;
        width: 56px; height: 56px; font-size: 22px; cursor: pointer;
        transition: all 0.2s; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    ">&#9673;</button>
    <div id="speech-status" style="font-size:12px; color:#888; margin-top:6px;">
        Click to start recording
    </div>
    <div id="speech-transcript" style="
        font-size:14px; color:#ccc; margin-top:10px; min-height:24px;
        font-style:italic;
    "></div>
</div>
<script>
let recognition = null;
let isListening = false;
let finalTranscript = '';

function toggleSpeech() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        document.getElementById('speech-status').innerText = 'Speech recognition not supported in this browser.';
        return;
    }
    if (isListening) {
        stopSpeech();
    } else {
        startSpeech();
    }
}

function startSpeech() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.continuous = false;
    recognition.interimResults = true;

    const btn = document.getElementById('speech-btn');
    const status = document.getElementById('speech-status');
    const transcriptDiv = document.getElementById('speech-transcript');

    btn.style.background = '#dc2626';
    status.innerText = 'Listening...';
    isListening = true;
    finalTranscript = '';
    transcriptDiv.innerText = '';

    recognition.onresult = function(event) {
        let interim = '';
        let final_ = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            if (event.results[i].isFinal) {
                final_ += event.results[i][0].transcript;
            } else {
                interim += event.results[i][0].transcript;
            }
        }
        if (final_) finalTranscript += final_;
        transcriptDiv.innerText = finalTranscript + interim;
    };

    recognition.onend = function() {
        btn.style.background = '#2563eb';
        isListening = false;
        if (finalTranscript.trim()) {
            status.innerText = 'Transcript ready. Sending...';
            transcriptDiv.style.color = '#fff';
            // Send to Streamlit via query params trick
            const encoded = encodeURIComponent(finalTranscript.trim());
            window.parent.postMessage({type: 'streamlit:setComponentValue', value: finalTranscript.trim()}, '*');
        } else {
            status.innerText = 'No speech detected. Click to try again.';
        }
    };

    recognition.onerror = function(event) {
        btn.style.background = '#2563eb';
        isListening = false;
        status.innerText = 'Error: ' + event.error + '. Try again.';
    };

    recognition.start();
}

function stopSpeech() {
    if (recognition) recognition.stop();
}
</script>
"""


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Financial Sentiment Analysis")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Analyze", "Batch Analysis", "Live News", "Model Leaderboard", "Dataset", "Methodology", "Rendu", "About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(f"**Model** : `{model_name}`")
    st.markdown(f"**Device** : `{device.type.upper()}`")
    st.markdown(f"**Dataset** : FinancialPhraseBank")
    st.markdown(f"**Split** : 70 / 10 / 20")

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — ANALYZE
# ══════════════════════════════════════════════════════════════════════
if page == "Analyze":
    st.title("Analyze a Headline")
    st.markdown(
        "Classify a financial headline as **positive**, **neutral**, or **negative** "
        "using FinBERT — a transformer fine-tuned on financial text."
    )
    st.markdown("---")

    # ── Text input ────────────────────────────────────────────────────
    col_in, col_out = st.columns([3, 2], gap="large")

    with col_in:
        st.markdown("### Text input")
        init_val = st.session_state.pop("example", "")

        # Check if voice transcript arrived
        if "voice_transcript" in st.session_state and st.session_state["voice_transcript"]:
            init_val = st.session_state.pop("voice_transcript")

        headline = st.text_area(
            label="headline",
            value=init_val,
            placeholder="Paste or type a financial news headline here...",
            height=110,
            label_visibility="collapsed",
        )
        c1, c2 = st.columns([2, 1])
        run   = c1.button("Analyze", type="primary", use_container_width=True)
        clear = c2.button("Clear", use_container_width=True)
        if clear:
            st.session_state.pop("voice_transcript", None)
            st.rerun()

    with col_out:
        st.markdown("### Result")
        result_slot = st.empty()
        result_slot.info("Enter a headline to begin analysis.")

    # ── Run analysis ──────────────────────────────────────────────
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

        with col_out:
            result_slot.empty()
            st.markdown(f"""
            <div style="text-align:center; padding:16px 0 8px;">
                <div style="font-size:3rem; font-weight:700; color:{col}; font-family:monospace;">
                    [{SYMBOLS[lbl]}]
                </div>
                <div style="font-size:1.4rem; font-weight:700; color:{col};">{lbl.capitalize()}</div>
                <div style="font-size:0.85rem; color:#666; margin-top:4px;">Confidence : {conf:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Class probabilities**")
            for l, s in sorted(sc.items(), key=lambda x: x[1], reverse=True):
                st.markdown(f"[{SYMBOLS[l]}] {l.capitalize()}")
                st.progress(s)
                st.caption(f"{s:.1%}")

            st.caption(f"Inference: {ms:.0f} ms · Model: {model_name} · Device: {device.type.upper()}")

    elif run:
        with col_out:
            result_slot.warning("Please enter a headline first.")

    # ── LIME Explainability ──────────────────────────────────────
    if analysis_result and headline.strip():
        st.markdown("---")
        st.markdown("### Keyword influence analysis")
        st.markdown(
            "Which words contributed most to the prediction? "
            "Green = pushes toward predicted class, red = pushes against."
        )
        with st.spinner("Computing word importance (LIME)..."):
            try:
                word_weights = run_lime(headline, analysis_result["label"])
                html = render_lime_html(headline, word_weights, analysis_result["label"])
                st.markdown(html, unsafe_allow_html=True)

                # Show top words as a bar chart
                if word_weights:
                    ww_df = pd.DataFrame(word_weights, columns=["word", "weight"])
                    ww_df = ww_df.sort_values("weight")
                    fig_lime = go.Figure(go.Bar(
                        x=ww_df["weight"],
                        y=ww_df["word"],
                        orientation="h",
                        marker_color=[
                            "#16a34a" if w > 0 else "#dc2626"
                            for w in ww_df["weight"]
                        ],
                    ))
                    fig_lime.update_layout(
                        xaxis_title="Contribution to prediction",
                        yaxis_title="",
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        margin=dict(t=10, b=30, l=10, r=10),
                        height=max(250, len(ww_df) * 22),
                    )
                    st.plotly_chart(fig_lime, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not compute explanations: {e}")

    # ── Voice input (Web Speech API) ─────────────────────────────
    st.markdown("---")
    st.markdown("### Voice input")
    st.markdown(
        "Click the microphone, speak a financial headline in English. "
        "The transcript appears in real time. Works on Chrome and Edge."
    )

    # Use streamlit-javascript to get the transcript
    transcript = st_javascript("""
    await new Promise((resolve) => {
        // Check if we already have a running instance
        if (window._speechResolve) {
            resolve("");
            return;
        }

        const container = document.createElement('div');
        container.style.textAlign = 'center';
        container.style.padding = '16px 0';

        const btn = document.createElement('button');
        btn.innerHTML = '&#9673; Click to speak';
        btn.style.cssText = 'background:#2563eb; color:white; border:none; border-radius:28px; padding:12px 28px; font-size:15px; cursor:pointer; transition:all 0.2s; box-shadow:0 2px 8px rgba(0,0,0,0.15);';

        const status = document.createElement('div');
        status.style.cssText = 'font-size:12px; color:#888; margin-top:8px;';
        status.innerText = 'Click the button, then speak clearly.';

        const transcript = document.createElement('div');
        transcript.style.cssText = 'font-size:14px; margin-top:10px; min-height:24px; font-style:italic; color:#aaa;';

        container.appendChild(btn);
        container.appendChild(status);
        container.appendChild(transcript);

        // Find the iframe body to append to
        document.body.appendChild(container);

        let finalText = '';

        btn.onclick = () => {
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
                status.innerText = 'Not supported in this browser. Use Chrome or Edge.';
                resolve("");
                return;
            }

            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            const recognition = new SpeechRecognition();
            recognition.lang = 'en-US';
            recognition.continuous = false;
            recognition.interimResults = true;

            btn.style.background = '#dc2626';
            btn.innerHTML = '&#9673; Listening...';
            status.innerText = 'Speak now...';
            finalText = '';

            recognition.onresult = (event) => {
                let interim = '';
                let final_ = '';
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    if (event.results[i].isFinal) {
                        final_ += event.results[i][0].transcript;
                    } else {
                        interim += event.results[i][0].transcript;
                    }
                }
                if (final_) finalText += final_;
                transcript.innerText = finalText + interim;
            };

            recognition.onend = () => {
                btn.style.background = '#2563eb';
                btn.innerHTML = '&#9673; Click to speak';
                if (finalText.trim()) {
                    status.innerText = 'Transcript captured.';
                    transcript.style.color = '#fff';
                    resolve(finalText.trim());
                } else {
                    status.innerText = 'No speech detected. Try again.';
                    resolve("");
                }
            };

            recognition.onerror = (event) => {
                btn.style.background = '#2563eb';
                btn.innerHTML = '&#9673; Click to speak';
                status.innerText = 'Error: ' + event.error;
                resolve("");
            };

            recognition.start();
        };
    });
    """)

    if transcript and isinstance(transcript, str) and transcript.strip():
        st.info(f"**Transcript :** {transcript}")
        t0 = time.time()
        voice_result = predict(transcript, pipe)
        ms = (time.time() - t0) * 1000
        lbl  = voice_result["label"]
        conf = voice_result["confidence"]
        sc   = voice_result["scores"]
        col  = COLORS[lbl]

        st.markdown(f"""
        <div style="text-align:center; padding:12px 0;">
            <div style="font-size:2rem; font-weight:700; color:{col}; font-family:monospace;">
                [{SYMBOLS[lbl]}]
            </div>
            <div style="font-size:1.2rem; font-weight:700; color:{col};">{lbl.capitalize()}</div>
            <div style="font-size:0.8rem; color:#666;">Confidence : {conf:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        for l, s in sorted(sc.items(), key=lambda x: x[1], reverse=True):
            st.markdown(f"[{SYMBOLS[l]}] {l.capitalize()}")
            st.progress(s)
            st.caption(f"{s:.1%}")

        st.caption(f"Inference: {ms:.0f} ms · Model: {model_name}")

        # LIME for voice result too
        with st.spinner("Computing word importance (LIME)..."):
            try:
                word_weights = run_lime(transcript, voice_result["label"])
                html = render_lime_html(transcript, word_weights, voice_result["label"])
                st.markdown(html, unsafe_allow_html=True)
            except Exception:
                pass

    # ── Examples ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Example headlines")
    st.markdown("Click a headline to load it.")

    col_pos, col_neu, col_neg = st.columns(3, gap="medium")
    examples = [
        (col_pos, "positive", "[+] Typically Positive", EXAMPLES_POSITIVE),
        (col_neu, "neutral",  "[~] Typically Neutral",  EXAMPLES_NEUTRAL),
        (col_neg, "negative", "[-] Typically Negative", EXAMPLES_NEGATIVE),
    ]
    for col_el, sentiment, label, exs in examples:
        with col_el:
            st.markdown(f"<span style='color:{COLORS[sentiment]}; font-weight:600;'>{label}</span>", unsafe_allow_html=True)
            for ex in exs:
                if st.button(ex, key=f"ex_{ex[:25]}", use_container_width=True):
                    st.session_state["example"] = ex
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — BATCH ANALYSIS
# ══════════════════════════════════════════════════════════════════════
elif page == "Batch Analysis":
    st.title("Batch Analysis")
    st.markdown(
        "Upload a CSV file containing financial headlines. "
        "FinBERT will classify each headline and produce a downloadable report."
    )
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="CSV must contain a column named 'headline'. One headline per row.",
    )

    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        # Try to find the headline column
        headline_col = None
        for candidate in ["headline", "Headline", "HEADLINE", "text", "Text", "sentence", "title"]:
            if candidate in df_in.columns:
                headline_col = candidate
                break

        if headline_col is None:
            st.error(
                f"Could not find a headline column. Found columns: {list(df_in.columns)}. "
                "Please rename the column containing headlines to **headline**."
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
                    "Headline": h,
                    "Sentiment": r["label"],
                    "Confidence": r["confidence"],
                    "P(positive)": r["scores"].get("positive", 0),
                    "P(neutral)": r["scores"].get("neutral", 0),
                    "P(negative)": r["scores"].get("negative", 0),
                })
                progress.progress((i + 1) / len(headlines), text=f"Analyzing... {i+1}/{len(headlines)}")

            progress.empty()
            df_out = pd.DataFrame(results)

            st.success(f"Analysis complete. {len(df_out)} headlines classified.")
            st.markdown("---")

            # ── Summary metrics ──────────────────────────────────
            counts = df_out["Sentiment"].value_counts()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total", len(df_out))
            c2.metric("[+] Positive", int(counts.get("positive", 0)))
            c3.metric("[~] Neutral", int(counts.get("neutral", 0)))
            c4.metric("[-] Negative", int(counts.get("negative", 0)))

            st.markdown("---")

            # ── Visualisations ───────────────────────────────────
            col_pie, col_hist = st.columns(2, gap="large")

            with col_pie:
                st.markdown("#### Sentiment distribution")
                fig_pie = go.Figure(go.Pie(
                    labels=[k.capitalize() for k in ["positive", "neutral", "negative"]],
                    values=[int(counts.get(k, 0)) for k in ["positive", "neutral", "negative"]],
                    marker=dict(colors=[COLORS["positive"], COLORS["neutral"], COLORS["negative"]]),
                    hole=0.45,
                    textinfo="label+percent+value",
                ))
                fig_pie.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=300)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_hist:
                st.markdown("#### Confidence score distribution")
                fig_hist = go.Figure()
                for sent in ["positive", "neutral", "negative"]:
                    mask = df_out["Sentiment"] == sent
                    if mask.any():
                        fig_hist.add_trace(go.Histogram(
                            x=df_out.loc[mask, "Confidence"],
                            name=sent.capitalize(),
                            marker_color=COLORS[sent],
                            opacity=0.7,
                            nbinsx=20,
                        ))
                fig_hist.update_layout(
                    barmode="overlay",
                    xaxis_title="Confidence",
                    yaxis_title="Count",
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    margin=dict(t=10, b=30, l=10, r=10),
                    height=300,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # ── Full table ───────────────────────────────────────
            st.markdown("#### Results table")
            st.dataframe(df_out, use_container_width=True, height=400)

            # ── Download button ──────────────────────────────────
            csv_buffer = io.StringIO()
            df_out.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download results as CSV",
                data=csv_buffer.getvalue(),
                file_name="sentiment_analysis_results.csv",
                mime="text/csv",
            )


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — LIVE NEWS (Yahoo Finance)
# ══════════════════════════════════════════════════════════════════════
elif page == "Live News":
    st.title("Live News — Yahoo Finance")
    st.markdown(
        "Enter a stock ticker to fetch the latest news headlines from Yahoo Finance "
        "and analyze their sentiment in real time."
    )
    st.markdown("---")

    col_ticker, col_btn = st.columns([2, 1])
    with col_ticker:
        ticker_input = st.text_input(
            "Stock ticker",
            placeholder="e.g. AAPL, TSLA, MSFT, LVMH.PA",
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
                    title = None
                    if isinstance(item, dict):
                        title = item.get("title") or item.get("headline")
                    if title:
                        headlines.append(title)

                if not headlines:
                    st.warning(f"No headlines could be extracted for **{ticker_sym}**.")
                    st.stop()

                st.info(f"Found **{len(headlines)}** headlines for **{ticker_sym}**.")

            except Exception as e:
                st.error(f"Failed to fetch news: {e}")
                st.stop()

        # Analyze each headline
        results = []
        progress = st.progress(0, text="Analyzing headlines...")
        for i, h in enumerate(headlines):
            r = predict(h, pipe)
            results.append({
                "Headline": h,
                "Sentiment": r["label"],
                "Confidence": r["confidence"],
            })
            progress.progress((i + 1) / len(headlines), text=f"Analyzing... {i+1}/{len(headlines)}")
        progress.empty()

        df_news = pd.DataFrame(results)

        st.markdown("---")

        # ── Summary ──────────────────────────────────────────────
        counts = df_news["Sentiment"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Headlines", len(df_news))
        c2.metric("[+] Positive", int(counts.get("positive", 0)))
        c3.metric("[~] Neutral", int(counts.get("neutral", 0)))
        c4.metric("[-] Negative", int(counts.get("negative", 0)))

        # Overall sentiment gauge
        pos_pct = counts.get("positive", 0) / len(df_news)
        neg_pct = counts.get("negative", 0) / len(df_news)
        sentiment_score = pos_pct - neg_pct  # -1 to +1

        if sentiment_score > 0.15:
            overall = "Predominantly Positive"
            overall_color = COLORS["positive"]
        elif sentiment_score < -0.15:
            overall = "Predominantly Negative"
            overall_color = COLORS["negative"]
        else:
            overall = "Mixed / Neutral"
            overall_color = COLORS["neutral"]

        st.markdown(
            f'<div style="text-align:center; padding:12px; margin:12px 0; '
            f'border-left:4px solid {overall_color}; background:rgba(0,0,0,0.02);">'
            f'<span style="font-size:18px; font-weight:600; color:{overall_color};">'
            f'{ticker_sym} — {overall}</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ── Distribution chart ───────────────────────────────────
        col_chart, col_table = st.columns([1, 2], gap="large")

        with col_chart:
            st.markdown("#### Sentiment distribution")
            fig = go.Figure(go.Pie(
                labels=[k.capitalize() for k in ["positive", "neutral", "negative"]],
                values=[int(counts.get(k, 0)) for k in ["positive", "neutral", "negative"]],
                marker=dict(colors=[COLORS["positive"], COLORS["neutral"], COLORS["negative"]]),
                hole=0.45,
                textinfo="label+percent+value",
            ))
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.markdown("#### Headlines")
            for _, row in df_news.iterrows():
                sent = row["Sentiment"]
                conf = row["Confidence"]
                c = COLORS[sent]
                st.markdown(
                    f'<div style="padding:6px 0; border-bottom:1px solid #eee;">'
                    f'<span style="color:{c}; font-weight:600; font-family:monospace;">[{SYMBOLS[sent]}]</span> '
                    f'{row["Headline"]} '
                    f'<span style="color:#888; font-size:12px;">({conf:.0%})</span></div>',
                    unsafe_allow_html=True,
                )

    elif fetch_btn:
        st.warning("Please enter a ticker symbol.")


# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL LEADERBOARD
# ══════════════════════════════════════════════════════════════════════
elif page == "Model Leaderboard":
    st.title("Model Leaderboard")
    st.markdown(
        "All models trained on the same **70/10/20 stratified split** (seed=42). "
        "Ranked by **macro-F1** — accounts for class imbalance."
    )
    st.markdown("---")

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
            st.markdown("#### Macro-F1 by model")
            fig = go.Figure()
            for _, row in lb.iterrows():
                c = TYPE_COLORS.get(row.get("type", ""), "#888")
                fig.add_trace(go.Bar(
                    x=[row["model"]], y=[row["f1_macro"]],
                    marker_color=c,
                    text=[f"{row['f1_macro']:.1%}"],
                    textposition="outside",
                    showlegend=False,
                ))
            fig.update_layout(
                yaxis=dict(range=[0, 1.05], tickformat=".0%", gridcolor="#f0f0f0"),
                xaxis=dict(tickangle=-15),
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=10, b=10, l=10, r=10), height=320, bargap=0.35,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_scatter:
            st.markdown("#### Accuracy vs Macro-F1")
            fig2 = go.Figure()
            for _, row in lb.iterrows():
                c = TYPE_COLORS.get(row.get("type", ""), "#888")
                fig2.add_trace(go.Scatter(
                    x=[row["accuracy"]], y=[row["f1_macro"]],
                    mode="markers+text",
                    marker=dict(size=14, color=c),
                    text=[row["model"]], textposition="top center",
                    textfont=dict(size=9),
                    showlegend=False,
                ))
            fig2.update_layout(
                xaxis=dict(title="Accuracy", tickformat=".0%", gridcolor="#f0f0f0", range=[0.5, 0.97]),
                yaxis=dict(title="Macro-F1", tickformat=".0%", gridcolor="#f0f0f0", range=[0.2, 1.0]),
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=10, b=40, l=40, r=10), height=320,
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            "<span style='color:#2563eb'>&#9632;</span> Transformer &nbsp;&nbsp;"
            "<span style='color:#f59e0b'>&#9632;</span> Classical ML &nbsp;&nbsp;"
            "<span style='color:#dc2626'>&#9632;</span> Deep Learning",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.markdown("#### Full benchmark table")
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

        st.markdown("---")
        st.markdown("#### Key takeaways")
        k1, k2 = st.columns(2, gap="large")
        with k1:
            st.success("**FinBERT #1 (0.877 macro-F1)** — Le pre-entrainement sur corpus financier lui permet de comprendre les nuances semantiques du domaine.")
            st.success("**DistilBERT > BERT** — Avec ~5K exemples, le modele allege generalise mieux que BERT complet.")
        with k2:
            st.info("**Baselines : accuracy trompeuse** — 72% accuracy mais 0.60 macro-F1. Ces modeles predisent surtout *neutral*.")
            st.error("**BiLSTM : 0% recall sur *negative*** — Sans pre-entrainement, le modele ne generalise pas sur la classe minoritaire.")


# ══════════════════════════════════════════════════════════════════════
# PAGE 5 — DATASET
# ══════════════════════════════════════════════════════════════════════
elif page == "Dataset":
    st.title("Dataset — FinancialPhraseBank")
    st.markdown("**Malo et al. (2014)** — 4 846 phrases en anglais extraites de news financieres, annotees par 16 experts.")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total phrases", "4 846")
    c2.metric("Classes", "3")
    c3.metric("Annotateurs", "16")
    c4.metric("Langue", "Anglais")

    st.markdown("---")
    col_pie, col_info = st.columns([2, 3], gap="large")

    with col_pie:
        st.markdown("#### Distribution des classes")
        fig = go.Figure(go.Pie(
            labels=["Neutral", "Positive", "Negative"],
            values=[2879, 1363, 604],
            marker=dict(colors=[COLORS["neutral"], COLORS["positive"], COLORS["negative"]]),
            hole=0.45,
            textinfo="label+percent",
        ))
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=280)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(pd.DataFrame({
            "Classe": ["[~] Neutral", "[+] Positive", "[-] Negative"],
            "Nombre": [2879, 1363, 604],
            "Part": ["59.4%", "28.1%", "12.5%"],
        }), use_container_width=True, hide_index=True)

    with col_info:
        st.markdown("#### Tache d'annotation")
        st.markdown("""
        > *"Cette phrase, du point de vue d'un investisseur, donne-t-elle une impression
        positive, negative ou neutre de l'entreprise mentionnee ?"*

        Seules les phrases avec **accord majoritaire** sont incluses.
        """)

        st.warning(
            "Un modele predisant toujours **'neutral'** obtiendrait **59% d'accuracy** "
            "mais un **macro-F1 ~ 0** — d'ou l'importance du macro-F1 comme metrique principale."
        )

        st.markdown("#### Split utilise")
        st.dataframe(pd.DataFrame({
            "Split": ["Train", "Val", "Test"],
            "Samples": [3391, 485, 970],
            "Share": ["70%", "10%", "20%"],
        }), use_container_width=True, hide_index=True)

        st.markdown("#### Exemples par classe")
        t1, t2, t3 = st.tabs(["Positive", "Neutral", "Negative"])
        with t1:
            for ex in EXAMPLES_POSITIVE: st.markdown(f"- {ex}")
        with t2:
            for ex in EXAMPLES_NEUTRAL: st.markdown(f"- {ex}")
        with t3:
            for ex in EXAMPLES_NEGATIVE: st.markdown(f"- {ex}")

    st.markdown("---")
    st.caption("Source : Malo, P. et al. (2014). Good debt or bad debt. JASIST, 65(4), 782-796.")


# ══════════════════════════════════════════════════════════════════════
# PAGE 6 — METHODOLOGY
# ══════════════════════════════════════════════════════════════════════
elif page == "Methodology":
    st.title("Methodology")
    st.markdown("Comparaison de **trois generations d'approches NLP** sur le meme protocole d'evaluation.")
    st.markdown("---")

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown("### 1. Classical ML — TF-IDF + Classifier")
        st.markdown("""
        Text → vecteurs TF-IDF creux (unigrammes + bigrammes, 20K features, TF sublineaire) → classifieur lineaire.

        **Avantages** : rapide, interpretable, pas de GPU.
        **Limites** : ignore l'ordre des mots, le contexte, la semantique.
        """)

        st.markdown("### 2. Deep Learning — Bidirectional LSTM")
        st.markdown("""
        Embedding (64d, 15K vocab) → BiLSTM 2 couches → Dense → Softmax.

        **Avantages** : capture l'ordre et les dependances locales.
        **Limites** : sans pre-entrainement + ~5K exemples → 0% recall sur *negative*.
        """)

        st.markdown("### 3. Transformers — BERT / DistilBERT / FinBERT")
        st.markdown("""
        Pre-entraines sur larges corpus → fine-tuning (5 epochs, lr=2e-5, early stopping).

        **FinBERT** : further pre-entraine sur corpus financier (rapports d'analystes, earnings calls).
        Comprend les nuances implicites : *restructuring*, *write-down*, *impairment charge*...
        """)

    with col_r:
        st.markdown("### Protocole d'evaluation")
        st.info("""
        **Meme split pour tous**
        70% train / 10% val / 20% test — stratifie, seed=42

        **Metrique principale : Macro-F1**
        F1 par classe moyenne equitablement
        → *negative* (12.5%) = meme poids que *neutral* (59.4%)

        **Metrique secondaire : Accuracy**
        Reportee pour reference — trompeuse sur donnees desequilibrees
        """)

        st.markdown("### References")
        for ref in [
            "Malo et al. (2014). *Good debt or bad debt.* JASIST, 65(4).",
            "Araci (2019). *FinBERT.* arXiv:1908.10063.",
            "Devlin et al. (2019). *BERT.* NAACL-HLT.",
            "Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.",
        ]:
            st.markdown(f"- {ref}")


# ══════════════════════════════════════════════════════════════════════
# PAGE 7 — RENDU
# ══════════════════════════════════════════════════════════════════════
elif page == "Rendu":
    st.title("Rendu")
    st.markdown("Slides de presentation et rapport ecrit du projet.")
    st.markdown("---")

    col_s, col_r2 = st.columns(2, gap="large")
    with col_s:
        st.markdown("### Slides de presentation")
        st.info("Les slides seront ajoutees ici une fois finalisees.")

    with col_r2:
        st.markdown("### Rapport Overleaf")
        st.info("Le rapport LaTeX (Overleaf) sera ajoute ici une fois finalise.")

    st.markdown("---")
    st.markdown("### Structure du rapport")
    st.dataframe(pd.DataFrame({
        "Section": ["Introduction", "Dataset", "Modeles", "Experiences", "Analyse", "Conclusion"],
        "Contenu": [
            "Contexte, problematique, objectifs",
            "FinancialPhraseBank, statistiques, desequilibre",
            "Description des 6 approches benchmarkees",
            "Protocole, hyperparametres, resultats",
            "Comparaison, interpretation, limites",
            "Bilan, perspectives",
        ],
    }), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 8 — ABOUT
# ══════════════════════════════════════════════════════════════════════
elif page == "About":
    st.title("About")
    st.markdown("---")

    col_a, col_b = st.columns([3, 2], gap="large")

    with col_a:
        st.markdown("### Le projet")
        st.markdown("""
        Projet de cours **Deep Learning / NLP** — Universite Paris 1 Pantheon-Sorbonne.

        Objectif : comparer plusieurs approches NLP pour la classification de sentiment
        sur des textes financiers, en illustrant la progression des performances de la
        generation bag-of-words aux transformers specialises.
        """)

        st.markdown("### Limitations")
        st.warning("""
        - Dataset petit (~4 800 exemples) → generalisation limitee
        - News europeennes des annees 2000-2010
        - Ne pas utiliser pour des decisions financieres reelles
        - Performances reduites hors domaine (reseaux sociaux, transcriptions d'appels)
        """)

    with col_b:
        st.markdown("### Details techniques")
        st.dataframe(pd.DataFrame({
            "Parametre": ["Modele actif", "Device", "Max tokens", "Split", "Metrique", "Framework", "Env"],
            "Valeur": [model_name, device.type.upper(), "128", "70/10/20", "Macro-F1", "PyTorch + HuggingFace", "uv / Python 3.11"],
        }), use_container_width=True, hide_index=True)

        st.markdown("### References")
        for ref in [
            "Malo et al. (2014). JASIST, 65(4).",
            "Araci (2019). FinBERT. arXiv:1908.10063.",
            "Devlin et al. (2019). BERT. NAACL-HLT.",
            "Vaswani et al. (2017). NeurIPS.",
        ]:
            st.markdown(f"- {ref}")
