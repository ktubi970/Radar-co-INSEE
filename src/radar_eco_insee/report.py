"""Génération de rapports Markdown avec graphiques (matplotlib)."""

from __future__ import annotations

import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .data import SeriesData
from .detection import DetectionResult
from .explain import LLMExplainer, RuleBasedExplainer

PLOT_FILENAME = "plot.png"


def make_plot(result: DetectionResult) -> plt.Figure:
    """Dessine le graphique de la série + anomalies, retourne la figure matplotlib."""
    df = result.series.data
    dates = df["date"]

    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    ax.plot(dates, result.series.data["value"], label="Valeurs", color="#1f77b4", lw=1.4)
    ax.plot(dates, result.trend, label="Tendance", color="#ff7f0e", lw=2.0)
    if result.seasonal is not None:
        ax.plot(
            dates,
            result.adjusted,
            label="Désaisonnalisée (tendance + résidu)",
            color="#2ca02c",
            lw=1.0,
            alpha=0.7,
        )

    for a in result.point_anomalies:
        x = pd.Timestamp(a.date)
        ax.scatter([x], [a.value], color="red", zorder=5, s=70, marker="o")
        ax.annotate(
            a.period,
            (x, a.value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="red",
        )

    for s in result.level_shifts:
        x = pd.Timestamp(s.date)
        ax.axvline(x, color="purple", ls="--", lw=1.2, alpha=0.8)
        ax.annotate(
            f"Rupture {s.period}",
            (x, ax.get_ylim()[1] * 0.9),
            textcoords="offset points",
            xytext=(-70, 0),
            fontsize=8,
            color="purple",
        )

    ax.set_title(result.series.title_fr)
    ax.set_xlabel("Période")
    ax.set_ylabel(result.series.unit_measure or "Valeur")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def _save_plot(result: DetectionResult, output_dir: Path) -> str:
    """Sauvegarde le graphique de la série + anomalies, retourne le nom du fichier."""
    fig = make_plot(result)
    plot_path = output_dir / PLOT_FILENAME
    fig.savefig(plot_path)
    plt.close(fig)
    return PLOT_FILENAME


def detections_dataframe(result: DetectionResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tableaux récapitulatifs des détections (points anormaux, ruptures de niveau)."""
    points = pd.DataFrame(
        [
            {
                "Période": a.period,
                "Valeur": a.value,
                "Attendu": round(a.expected, 2),
                "Écart (z)": round(a.z_score, 2),
                "Sévérité": a.severity,
            }
            for a in result.point_anomalies
        ]
    )
    shifts = pd.DataFrame(
        [
            {
                "Période": s.period,
                "Niveau avant": round(s.mean_before, 2),
                "Niveau après": round(s.mean_after, 2),
                "Direction": s.direction,
                "Amplitude (%)": round(s.magnitude_pct, 1),
            }
            for s in result.level_shifts
        ]
    )
    return points, shifts


def build_report(
    result: DetectionResult,
    explainer: RuleBasedExplainer | LLMExplainer,
    output_dir: Path,
) -> Path:
    """Écrit le rapport Markdown d'une série dans `output_dir`, retourne son chemin."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_name = _save_plot(result, output_dir)

    rule_explainer = explainer if isinstance(explainer, RuleBasedExplainer) else RuleBasedExplainer()
    rule_text = rule_explainer.explain(result)
    explanation = explainer.explain(result, rule_text) if not isinstance(explainer, RuleBasedExplainer) else rule_text

    lines = [
        f"# {result.series.title_fr}",
        "",
        f"- **Identifiant BDM** : `{result.series.id}`",
        f"- **Fréquence** : {result.series.frequency_label}",
        f"- **Observations** : {result.n_obs}",
        f"- **Unité** : {result.series.unit_measure or '—'}",
        f"- **Analyse** : {datetime.date.today().isoformat()}",
        "",
        "## Détections",
        "",
        explanation,
        "",
        "## Graphique",
        "",
        f"![Série et anomalies]({plot_name})",
        "",
    ]
    report_path = output_dir / "rapport.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
