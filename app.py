#!/usr/bin/env python3
"""App Streamlit : détection d'anomalies économiques sur les séries INSEE (BDM).

Lancer :  streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from radar_eco_insee.data import INSEEBDMClient, SeriesData
from radar_eco_insee.detection import detect_anomalies
from radar_eco_insee.explain import (
    OLLAMA_MODELS,
    OPENROUTER_MODELS,
    LLMExplainer,
    RuleBasedExplainer,
    model_options,
    resolve_model,
)
from radar_eco_insee.report import detections_dataframe, make_plotly_figure

CONFIG_PATH = ROOT / "config" / "series.yaml"


def _config(key: str, default: str = "") -> str:
    """Secret Streamlit avec repli sur variable d'environnement."""
    try:
        return str(st.secrets.get(key, os.environ.get(key, default)))
    except Exception:
        return os.environ.get(key, default)


def llm_provider() -> str:
    """Fournisseur LLM : openai si une clé est configurée, sinon ollama."""
    provider = _config("LLM_PROVIDER", "")
    if provider in ("openai", "ollama"):
        return provider
    return "openai" if _config("OPENAI_API_KEY") else "ollama"


def llm_default_model(provider: str) -> str:
    if provider == "openai":
        return _config("OPENAI_MODEL", "mistralai/mistral-medium-3-5")
    return _config("OLLAMA_MODEL", "qwen2.5:1.5b")

st.set_page_config(page_title="Radar Éco INSEE", page_icon="📈", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch(series_id: str) -> tuple[str, str, str, str, int, pd.DataFrame]:
    """Récupération mise en cache (les DataFrames sont sérialisables par Streamlit)."""
    series = INSEEBDMClient().fetch_series(series_id)
    return (
        series.id,
        series.title_fr,
        series.freq,
        series.unit_measure,
        series.seasonal_period,
        series.data,
    )


def to_series(record: tuple) -> SeriesData:
    id_, title, freq, unit, seasonal_period, df = record
    return SeriesData(id=id_, title_fr=title, freq=freq, unit_measure=unit, seasonal_period=seasonal_period, data=df)


def load_config_series() -> list[dict]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f).get("series", [])


st.title("📈 Radar Éco INSEE")
st.caption("Détection d'anomalies économiques sur les séries de la BDM · points anormaux + ruptures de niveau · explication en français")

config_series = load_config_series()
labels = {f"{e['label']}  ·  `{e['id']}`": e["id"] for e in config_series}

with st.sidebar:
    st.header("Paramètres")
    choice = st.radio(
        "Série",
        ["Séries du catalogue"] + ["Autre ID BDM..."],
    )
    if choice == "Séries du catalogue":
        selected = st.selectbox("Choisir une série", list(labels))
        series_id = labels[selected]
    else:
        series_id = st.text_input("Identifiant BDM (ex. 001688537)", value="001688537").strip()

    n_points = st.slider("Points anormaux (top |z|)", 1, 20, 5, 1)
    penalty = st.slider("Sensibilité ruptures (variance expliquée)", 0.0, 0.05, 0.004, 0.001)

    provider = llm_provider()
    use_llm = st.checkbox("Expliquer avec un LLM")
    if use_llm:
        catalog = OPENROUTER_MODELS if provider == "openai" else OLLAMA_MODELS
        default_model = llm_default_model(provider)
        options, default_index = model_options(provider, default_model)
        llm_model = resolve_model(
            catalog,
            default_model,
            st.selectbox("Modèle", options, index=default_index, help=f"Fournisseur : {provider}"),
        )
        explain_llm = st.button("Expliquer avec le LLM", type="secondary", use_container_width=True)
    else:
        llm_model = None
        explain_llm = False

if not series_id:
    st.info("Saisissez un identifiant BDM pour lancer l'analyse.")
    st.stop()

with st.spinner("Récupération des données INSEE..."):
    try:
        record = cached_fetch(series_id)
    except Exception as exc:
        st.error(f"Impossible de récupérer la série `{series_id}` : {exc}")
        st.stop()
    series = to_series(record)

with st.spinner("Détection des anomalies..."):
    result = detect_anomalies(series, n_points=n_points, penalty=penalty)

series_key = (series_id, n_points, round(penalty, 3), llm_model)
llm_state = st.session_state.setdefault("llm_explanation", {"key": None, "text": None})

rule_text = RuleBasedExplainer().explain(result)
explanation = rule_text
llm_used = False

if use_llm and explain_llm:
    with st.spinner("Génération de l'explication LLM..."):
        llm_explainer = LLMExplainer(provider=provider, model=llm_model, api_key=_config("OPENAI_API_KEY"))
        if not llm_explainer.is_available():
            st.warning("LLM injoignable — explication par règles utilisée à la place.")
        else:
            llm_text = llm_explainer.explain(result, rule_text)
            if llm_explainer.last_error:
                st.warning(f"LLM indisponible ({llm_explainer.last_error}) — explication par règles affichée.")
            else:
                explanation = llm_text
                llm_used = True
                st.session_state["llm_explanation"] = {"key": series_key, "text": llm_text}
elif use_llm and llm_state.get("key") == series_key and llm_state.get("text"):
    explanation = llm_state["text"]
    llm_used = True

st.subheader(series.title_fr)
col1, col2, col3 = st.columns(3)
col1.metric("Observations", f"{result.n_obs}", help=series.frequency_label)
col2.metric("Points anormaux", f"{len(result.point_anomalies)}")
col3.metric("Ruptures de niveau", f"{len(result.level_shifts)}")

st.markdown("### Explication")
if llm_used:
    st.caption(f"Explication générée avec le modèle **{llm_model}**")
st.markdown(explanation)

st.markdown("### Graphique")
st.plotly_chart(make_plotly_figure(result), use_container_width=True)

points_df, shifts_df = detections_dataframe(result)
if not points_df.empty:
    st.markdown("### Points anormaux")
    st.dataframe(points_df, use_container_width=True, hide_index=True)
if not shifts_df.empty:
    st.markdown("### Ruptures de niveau")
    st.dataframe(shifts_df, use_container_width=True, hide_index=True)

with st.expander("Données brutes"):
    st.dataframe(series.data[["period", "date", "value"]], use_container_width=True, hide_index=True)
