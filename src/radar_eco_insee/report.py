"""Graphiques Plotly interactifs et tableaux de détection."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .detection import DetectionResult


def make_plotly_figure(result: DetectionResult) -> go.Figure:
    """Dessine le graphique interactif (Plotly) de la série + anomalies."""
    df = result.series.data
    dates = df["date"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=dates, y=df["value"], name="Valeurs", mode="lines",
                   line=dict(color="#1f77b4", width=1.4))
    )
    fig.add_trace(
        go.Scatter(x=dates, y=result.trend, name="Tendance", mode="lines",
                   line=dict(color="#ff7f0e", width=2.0))
    )
    if result.seasonal is not None:
        fig.add_trace(
            go.Scatter(x=dates, y=result.adjusted,
                       name="Désaisonnalisée (tendance + résidu)", mode="lines",
                       line=dict(color="#2ca02c", width=1.0, dash="dot"), opacity=0.7)
        )

    if result.point_anomalies:
        fig.add_trace(
            go.Scatter(
                x=[pd.Timestamp(a.date) for a in result.point_anomalies],
                y=[a.value for a in result.point_anomalies],
                name="Points anormaux", mode="markers",
                marker=dict(color="red", size=12, line=dict(color="black", width=1)),
                customdata=[
                    [a.period, round(a.z_score, 2), a.severity]
                    for a in result.point_anomalies
                ],
                hovertemplate=(
                    "Période : %{customdata[0]}<br>"
                    "Valeur : %{y}<br>"
                    "Écart (z) : %{customdata[1]}<br>"
                    "Sévérité : %{customdata[2]}<extra></extra>"
                ),
            )
        )

    for s in result.level_shifts:
        x = pd.Timestamp(s.date).isoformat()
        fig.add_vline(x=x, line=dict(color="purple", dash="dash", width=1.2), opacity=0.8)
        fig.add_annotation(
            x=x, y=1, yref="y domain", text=f"Rupture {s.period}",
            showarrow=False, xanchor="left", yanchor="top", font=dict(color="purple", size=11),
        )

    fig.update_layout(
        title=result.series.title_fr,
        xaxis_title="Période",
        yaxis_title=result.series.unit_measure or "Valeur",
        legend=dict(orientation="h", y=1.12, x=0),
        hovermode="x unified",
        margin=dict(t=80, b=40, l=60, r=30),
    )
    return fig


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
