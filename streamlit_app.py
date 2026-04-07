"""Streamlit app — Financial News Sentiment Analysis."""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import speech_recognition as sr
import streamlit as st
from audio_recorder_streamlit import audio_recorder

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import RESULTS_DIR
from src.inference import build_pipeline, get_device, load_best_model, predict

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Sentiment Analysis",
    page_icon="📊",
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

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Financial Sentiment Analysis")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Analyze", "Model Leaderboard", "Dataset", "Methodology", "Rendu", "About"],
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
            st.rerun()

    with col_out:
        st.markdown("### Result")
        result_slot = st.empty()
        result_slot.info("Enter a headline to begin analysis.")

    if run and headline.strip():
        t0 = time.time()
        result = predict(headline, pipe)
        ms = (time.time() - t0) * 1000
        lbl  = result["label"]
        conf = result["confidence"]
        sc   = result["scores"]
        col  = COLORS[lbl]

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

    # ── Voice input ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Voice input")
    st.markdown(
        "Click the mic, speak a financial headline in English, "
        "then wait ~2 seconds of silence."
    )

    col_mic, col_voice_result = st.columns([1, 3], gap="large")
    with col_mic:
        audio_bytes = audio_recorder(
            text="",
            recording_color="#dc2626",
            neutral_color="#2563eb",
            icon_size="2x",
            pause_threshold=2.5,
        )

    with col_voice_result:
        if audio_bytes:
            with st.spinner("Transcribing..."):
                try:
                    recognizer = sr.Recognizer()
                    audio_data = sr.AudioData(audio_bytes, sample_rate=44100, sample_width=2)
                    transcript = recognizer.recognize_google(audio_data, language="en-US")

                    st.info(f"**Transcript :** {transcript}")

                    t0 = time.time()
                    result = predict(transcript, pipe)
                    ms = (time.time() - t0) * 1000
                    lbl  = result["label"]
                    conf = result["confidence"]
                    sc   = result["scores"]
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

                except sr.UnknownValueError:
                    st.warning("Could not understand audio. Speak clearly and try again.")
                except sr.RequestError:
                    st.error("Speech recognition unavailable. Check your internet connection.")
        else:
            st.markdown("*No audio recorded yet.*")

    # ── Examples ──────────────────────────────────────────────────────
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
# PAGE 2 — MODEL LEADERBOARD
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
            "<span style='color:#2563eb'>■</span> Transformer &nbsp;&nbsp;"
            "<span style='color:#f59e0b'>■</span> Classical ML &nbsp;&nbsp;"
            "<span style='color:#dc2626'>■</span> Deep Learning",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.markdown("#### Full benchmark table")
        display = lb.copy()
        type_map = {"classical": "Classical ML", "deep_learning": "Deep Learning", "transformer": "Transformer"}
        display["type"] = display["type"].map(type_map).fillna(display["type"])
        for col in ["accuracy", "f1_macro", "f1_weighted"]:
            if col in display.columns:
                display[col] = display[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "--")
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
            st.success("**FinBERT #1 (0.877 macro-F1)** — Le pré-entraînement sur corpus financier lui permet de comprendre les nuances sémantiques du domaine.")
            st.success("**DistilBERT > BERT** — Avec ~5K exemples, le modèle allégé généralise mieux que BERT complet.")
        with k2:
            st.info("**Baselines : accuracy trompeuse** — 72% accuracy mais 0.60 macro-F1. Ces modèles prédisent surtout *neutral*.")
            st.error("**BiLSTM : 0% recall sur *negative*** — Sans pré-entraînement, le modèle ne généralise pas sur la classe minoritaire.")

# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — DATASET
# ══════════════════════════════════════════════════════════════════════
elif page == "Dataset":
    st.title("Dataset — FinancialPhraseBank")
    st.markdown("**Malo et al. (2014)** — 4 846 phrases en anglais extraites de news financières, annotées par 16 experts.")
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
        st.markdown("#### Tâche d'annotation")
        st.markdown("""
        > *"Cette phrase, du point de vue d'un investisseur, donne-t-elle une impression
        positive, négative ou neutre de l'entreprise mentionnée ?"*

        Seules les phrases avec **accord majoritaire** sont incluses.
        """)

        st.warning(
            "Un modèle prédisant toujours **'neutral'** obtiendrait **59% d'accuracy** "
            "mais un **macro-F1 ≈ 0** — d'où l'importance du macro-F1 comme métrique principale."
        )

        st.markdown("#### Split utilisé")
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
# PAGE 4 — METHODOLOGY
# ══════════════════════════════════════════════════════════════════════
elif page == "Methodology":
    st.title("Methodology")
    st.markdown("Comparaison de **trois générations d'approches NLP** sur le même protocole d'évaluation.")
    st.markdown("---")

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown("### 1. Classical ML — TF-IDF + Classifier")
        st.markdown("""
        Text → vecteurs TF-IDF creux (unigrammes + bigrammes, 20K features, TF sublinéaire) → classifieur linéaire.

        **Avantages** : rapide, interprétable, pas de GPU.
        **Limites** : ignore l'ordre des mots, le contexte, la sémantique.
        """)

        st.markdown("### 2. Deep Learning — Bidirectional LSTM")
        st.markdown("""
        Embedding (64d, 15K vocab) → BiLSTM 2 couches → Dense → Softmax.

        **Avantages** : capture l'ordre et les dépendances locales.
        **Limites** : sans pré-entraînement + ~5K exemples → 0% recall sur *negative*.
        """)

        st.markdown("### 3. Transformers — BERT / DistilBERT / FinBERT")
        st.markdown("""
        Pré-entraînés sur larges corpus → fine-tuning (5 epochs, lr=2e-5, early stopping).

        **FinBERT** : further pré-entraîné sur corpus financier (rapports d'analystes, earnings calls).
        Comprend les nuances implicites : *restructuring*, *write-down*, *impairment charge*...
        """)

    with col_r:
        st.markdown("### Protocole d'évaluation")
        st.info("""
        **Même split pour tous**
        70% train / 10% val / 20% test — stratifié, seed=42

        **Métrique principale : Macro-F1**
        F1 par classe moyenné équitablement
        → *negative* (12.5%) = même poids que *neutral* (59.4%)

        **Métrique secondaire : Accuracy**
        Reportée pour référence — trompeuse sur données déséquilibrées
        """)

        st.markdown("### Références")
        for ref in [
            "Malo et al. (2014). *Good debt or bad debt.* JASIST, 65(4).",
            "Araci (2019). *FinBERT.* arXiv:1908.10063.",
            "Devlin et al. (2019). *BERT.* NAACL-HLT.",
            "Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.",
        ]:
            st.markdown(f"- {ref}")

# ══════════════════════════════════════════════════════════════════════
# PAGE 5 — RENDU
# ══════════════════════════════════════════════════════════════════════
elif page == "Rendu":
    st.title("Rendu")
    st.markdown("Slides de présentation et rapport écrit du projet.")
    st.markdown("---")

    col_s, col_r2 = st.columns(2, gap="large")
    with col_s:
        st.markdown("### Slides de présentation")
        st.info("Les slides seront ajoutées ici une fois finalisées.")

    with col_r2:
        st.markdown("### Rapport Overleaf")
        st.info("Le rapport LaTeX (Overleaf) sera ajouté ici une fois finalisé.")

    st.markdown("---")
    st.markdown("### Structure du rapport")
    st.dataframe(pd.DataFrame({
        "Section": ["Introduction", "Dataset", "Modèles", "Expériences", "Analyse", "Conclusion"],
        "Contenu": [
            "Contexte, problématique, objectifs",
            "FinancialPhraseBank, statistiques, déséquilibre",
            "Description des 6 approches benchmarkées",
            "Protocole, hyperparamètres, résultats",
            "Comparaison, interprétation, limites",
            "Bilan, perspectives",
        ],
    }), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 6 — ABOUT
# ══════════════════════════════════════════════════════════════════════
elif page == "About":
    st.title("About")
    st.markdown("---")

    col_a, col_b = st.columns([3, 2], gap="large")

    with col_a:
        st.markdown("### Le projet")
        st.markdown("""
        Projet de cours **Deep Learning / NLP** — Université Paris 1 Panthéon-Sorbonne.

        Objectif : comparer plusieurs approches NLP pour la classification de sentiment
        sur des textes financiers, en illustrant la progression des performances de la
        génération bag-of-words aux transformers spécialisés.
        """)

        st.markdown("### Limitations")
        st.warning("""
        - Dataset petit (~4 800 exemples) → généralisation limitée
        - News européennes des années 2000-2010
        - Ne pas utiliser pour des décisions financières réelles
        - Performances réduites hors domaine (réseaux sociaux, transcriptions d'appels)
        """)

    with col_b:
        st.markdown("### Détails techniques")
        st.dataframe(pd.DataFrame({
            "Paramètre": ["Modèle actif", "Device", "Max tokens", "Split", "Métrique", "Framework", "Env"],
            "Valeur": [model_name, device.type.upper(), "128", "70/10/20", "Macro-F1", "PyTorch + HuggingFace", "uv · Python 3.11"],
        }), use_container_width=True, hide_index=True)

        st.markdown("### Références")
        for ref in [
            "Malo et al. (2014). JASIST, 65(4).",
            "Araci (2019). FinBERT. arXiv:1908.10063.",
            "Devlin et al. (2019). BERT. NAACL-HLT.",
            "Vaswani et al. (2017). NeurIPS.",
        ]:
            st.markdown(f"- {ref}")
