"""OpinionMetrix AI sentiment-analysis Streamlit application."""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent
SRC_DIR = APP_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from db import save_prediction  # noqa: E402
from download_models import download_models  # noqa: E402
from predict_transformer import TransformerSentimentPredictor as SentimentPredictor  # noqa: E402
from styles import inject_css, render_header, render_result_card, sentiment_color  # noqa: E402

st.set_page_config(
    page_title="OpinionMetrix AI Sentiment Transformer",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def load_predictor():
    model_dir = download_models()
    return SentimentPredictor(model_dir=model_dir)


inject_css()

if "history" not in st.session_state:
    st.session_state["history"] = []
if "sample_text" not in st.session_state:
    st.session_state["sample_text"] = ""

try:
    predictor = load_predictor()
    model_loaded = True
    model_error = None
except Exception as exc:
    predictor = None
    model_loaded = False
    model_error = str(exc)

with st.sidebar:
    st.markdown("### ℹ️ About this model")
    st.write(
        "A fine-tuned Transformer model trained on labeled customer reviews "
        "to predict Positive, Negative, or Neutral sentiment."
    )
    st.markdown(
        "The Transformer uses light cleaning that removes HTML, URLs, and "
        "extra whitespace while preserving useful word order and punctuation."
    )

    st.divider()
    st.markdown("### 🧪 Try a sample review")
    samples = [
        "Absolutely love this product, best purchase I've made all year!",
        "Terrible quality, broke within a week, avoid this seller.",
        "It's fine, does what it says but nothing more.",
    ]
    for index, sample in enumerate(samples):
        if st.button(sample, key=f"sample_{index}"):
            st.session_state["sample_text"] = sample
            st.rerun()

    st.divider()
    st.markdown("### 🕘 Recent predictions")
    if st.session_state["history"]:
        for item in reversed(st.session_state["history"][-5:]):
            color = sentiment_color(item["sentiment"])
            st.markdown(
                f'<div class="history-row"><span class="history-text">'
                f'{item["text"]}</span><span style="color:{color}; font-weight:600;">'
                f'{item["sentiment"]}</span></div>',
                unsafe_allow_html=True,
            )
        if st.button("Clear history", width="stretch"):
            st.session_state["history"] = []
            st.rerun()
    else:
        st.caption("Your last few predictions will show up here.")

render_header()

if not model_loaded:
    st.error("The sentiment model could not be loaded.")
    st.warning("The model download or validation step failed. Check the deployment logs for the underlying error.")
    st.code(model_error or "Unknown model-loading error")
    st.stop()

st.markdown('<div class="input-card">', unsafe_allow_html=True)
text_input = st.text_area(
    "Enter a customer review",
    height=140,
    placeholder="e.g. The product quality is amazing and delivery was super fast!",
    value=st.session_state["sample_text"],
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 3])
with col1:
    predict_clicked = st.button("🔍  Predict Sentiment", type="primary", width="stretch")
with col2:
    st.caption(f"{len(text_input)} characters")
st.markdown("</div>", unsafe_allow_html=True)

if predict_clicked:
    if not text_input.strip():
        st.warning("Please enter a review to analyze.")
    else:
        try:
            with st.spinner("Analyzing sentiment..."):
                result = predictor.predict(text_input)
        except Exception:
            st.error("Prediction failed. Check the application logs for details.")
            st.stop()

        sentiment = result["sentiment"]
        confidence = result["confidence"]
        probs = result["probabilities"]

        try:
            save_prediction(text_input, sentiment, probs)
        except Exception:
            st.warning("Prediction was generated, but database persistence failed.")

        st.session_state["history"].append(
            {
                "text": text_input.strip()[:60],
                "sentiment": sentiment,
                "confidence": confidence,
            }
        )
        st.session_state["sample_text"] = ""

        render_result_card(sentiment, confidence)
        st.markdown("#### Prediction breakdown")

        df = pd.DataFrame(
            {"Label": list(probs.keys()), "Probability": list(probs.values())}
        ).sort_values("Probability", ascending=True)
        color_scale = alt.Scale(
            domain=["Positive", "Negative", "Neutral"],
            range=["#34d399", "#f87171", "#fbbf24"],
        )
        chart = (
            alt.Chart(df)
            .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8, height=26)
            .encode(
                x=alt.X("Probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Label:N", sort="-x", title=None),
                color=alt.Color("Label:N", scale=color_scale, legend=None),
                tooltip=[alt.Tooltip("Label:N"), alt.Tooltip("Probability:Q", format=".1%")],
            )
            .properties(height=120)
        )
        st.altair_chart(chart, width="stretch")
