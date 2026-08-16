"""Détection d'anomalies et de ruptures de niveau sur une série temporelle.

Méthode :
1. Décomposition STL (saison + tendance + résidu) ; en repli, lissage par médiane
   glissante quand la série est trop courte ou non saisonnière.
2. Points anormaux : score z robuste (médiane/MAD) sur les résidus.
3. Ruptures de niveau : segmentation binaire avec coût L2 (implémentée ici,
   sans dépendance externe) sur la série désaisonnalisée.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import SeriesData

MAD_Z_SCALE = 1.4826  # facteur de normalisation de la MAD vers l'écart-type


@dataclass
class PointAnomaly:
    """Valeur anormalement haute ou basse par rapport à la tendance saisonnière."""

    period: str
    date: str
    value: float
    expected: float
    residual: float
    z_score: float
    severity: str  # 'moyenne' | 'forte'


@dataclass
class LevelShift:
    """Rupture de niveau (changement de régime) détectée."""

    date: str
    period: str
    mean_before: float
    mean_after: float
    direction: str  # 'hausse' | 'baisse'
    magnitude_pct: float


@dataclass
class DetectionResult:
    """Résultat complet de l'analyse d'une série."""

    series: SeriesData
    n_obs: int
    trend: np.ndarray
    seasonal: np.ndarray | None
    residual: np.ndarray
    adjusted: np.ndarray  # série désaisonnalisée (tendance + résidu)
    point_anomalies: list[PointAnomaly] = field(default_factory=list)
    level_shifts: list[LevelShift] = field(default_factory=list)

    @property
    def has_anomalies(self) -> bool:
        return bool(self.point_anomalies or self.level_shifts)


def _mad(values: np.ndarray) -> float:
    return float(np.median(np.abs(values - np.median(values))))


def _stl_residuals(values: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Retourne (tendance, saisonnière ou None, résidu)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    seasonal = None

    if period >= 2 and n >= 2 * period + 1:
        try:
            from statsmodels.tsa.seasonal import STL

            res = STL(values, period=period, robust=True).fit()
            return (
                np.asarray(res.trend),
                np.asarray(res.seasonal),
                np.asarray(res.resid),
            )
        except Exception:
            pass

    # Repli : tendance par médiane glissante, résidu = valeur - tendance
    window = max(3, min(period, n // 3)) if period >= 2 else max(3, min(7, n // 3))
    trend = pd.Series(values).rolling(window, center=True, min_periods=1).median().to_numpy()
    return trend, seasonal, values - trend


def _z_scores(residual: np.ndarray) -> np.ndarray:
    center = float(np.median(residual))
    spread = _mad(residual)
    if spread < 1e-12:
        spread = float(np.std(residual))
    if spread < 1e-12:
        return np.zeros_like(residual, dtype=float)
    return (residual - center) / (spread * MAD_Z_SCALE)


def detect_point_anomalies(z_scores: np.ndarray, n_points: int) -> list[tuple[int, float]]:
    """Retourne les indices (et z-scores) des `n_points` valeurs les plus extrêmes.

    Les doublons trop proches (fenêtre ~ racine carrée du nombre d'observations)
    sont écartés pour ne pas compter un même événement deux fois.
    """
    z_scores = np.nan_to_num(z_scores, nan=0.0)
    window = max(2, int(np.ceil(len(z_scores) ** 0.5)))
    order = sorted(range(len(z_scores)), key=lambda i: -abs(z_scores[i]))
    kept: list[int] = []
    for idx in order:
        if all(abs(idx - other) > window for other in kept):
            kept.append(idx)
        if len(kept) >= n_points:
            break
    return sorted((idx, float(z_scores[idx])) for idx in kept)


# ---------------------------------------------------------------- segmentation

def _seg_cost_cumsum(cum: np.ndarray, cum2: np.ndarray, lo: int, hi: int) -> float:
    """Coût L2 (somme des écarts quadratiques à la moyenne) de cum[lo:hi]."""
    k = hi - lo
    if k <= 0:
        return 0.0
    s = cum[hi - 1] - (cum[lo - 1] if lo > 0 else 0.0)
    s2 = cum2[hi - 1] - (cum2[lo - 1] if lo > 0 else 0.0)
    return float(s2 - s * s / k)


def _best_split(arr: np.ndarray, min_seg: int) -> tuple[float, int] | None:
    """Meilleure coupe (gain de coût, index) de `arr` en deux segments ≥ min_seg."""
    n = len(arr)
    if n < 2 * min_seg:
        return None
    cum = np.cumsum(arr)
    cum2 = np.cumsum(arr**2)
    base = _seg_cost_cumsum(cum, cum2, 0, n)
    best: tuple[float, int] | None = None
    for i in range(min_seg, n - min_seg + 1):
        gain = base - _seg_cost_cumsum(cum, cum2, 0, i) - _seg_cost_cumsum(cum, cum2, i, n)
        if best is None or gain > best[0]:
            best = (float(gain), i)
    return best


def binary_segmentation(values: np.ndarray, min_seg: int, penalty: float) -> list[int]:
    """Détecte des ruptures de niveau par segmentation binaire (coût L2 + pénalité).

    `penalty` est exprimé en fraction de la variance totale de la série (0..1) :
    indépendant de l'échelle, donc comparable entre séries. Une coupe est
    acceptée si elle explique plus de variance que `penalty`.
    Les segments sont traités par gain décroissant (résultat déterministe).
    """
    n = len(values)
    if n < 2 * min_seg:
        return []

    values = np.asarray(values, dtype=float)
    base_cost = float(np.sum((values - np.mean(values)) ** 2))
    threshold = penalty * base_cost if base_cost > 1e-12 else 0.0
    segments: list[tuple[int, int]] = [(0, n)]
    breaks: list[int] = []

    while True:
        best: tuple[float, int, int, int] | None = None  # (gain, lo, hi, split)
        for lo, hi in segments:
            res = _best_split(values[lo:hi], min_seg)
            if res is None:
                continue
            gain, rel = res
            if best is None or gain > best[0]:
                best = (gain, lo, hi, lo + rel)

        if best is None or best[0] <= threshold:
            break

        gain, lo, hi, split = best
        segments.remove((lo, hi))
        segments.append((lo, split))
        segments.append((split, hi))
        breaks.append(split)

    return sorted(breaks)


# ----------------------------------------------------------------- détection

def detect_anomalies(
    series: SeriesData,
    n_points: int = 5,
    penalty: float = 0.004,
) -> DetectionResult:
    """Analyse complète d'une série : points anormaux + ruptures de niveau.

    `n_points` : nombre de points anormaux les plus extrêmes à retenir (top |z|).
    `penalty`  : fraction de variance minimale expliquée pour accepter une rupture.
    """
    df = series.data
    values = df["value"].to_numpy(dtype=float)
    period = series.seasonal_period

    trend, seasonal, residual = _stl_residuals(values, period)
    adjusted = (trend + residual) if seasonal is not None else (trend + residual)
    z = _z_scores(residual)

    # Points anormaux
    point_anomalies: list[PointAnomaly] = []
    for idx, z_score in detect_point_anomalies(z, n_points):
        point_anomalies.append(
            PointAnomaly(
                period=df["period"].iloc[idx],
                date=str(df["date"].iloc[idx].date()),
                value=float(values[idx]),
                expected=float(trend[idx]),
                residual=float(residual[idx]),
                z_score=float(z_score),
                severity="forte" if abs(z_score) >= 4.0 else "moyenne",
            )
        )

    # Ruptures de niveau sur la série désaisonnalisée
    min_seg = max(3, period if period >= 2 else 3)
    level_shifts: list[LevelShift] = []
    for idx in binary_segmentation(adjusted, min_seg, penalty):
        mean_before = float(np.mean(adjusted[max(0, idx - min_seg):idx]))
        mean_after = float(np.mean(adjusted[idx:idx + min_seg]))
        direction = "hausse" if mean_after > mean_before else "baisse"
        magnitude_pct = (mean_after - mean_before) / abs(mean_before) * 100 if mean_before else 0.0
        level_shifts.append(
            LevelShift(
                date=str(df["date"].iloc[idx].date()),
                period=df["period"].iloc[idx],
                mean_before=mean_before,
                mean_after=mean_after,
                direction=direction,
                magnitude_pct=float(magnitude_pct),
            )
        )

    return DetectionResult(
        series=series,
        n_obs=len(values),
        trend=trend,
        seasonal=seasonal,
        residual=residual,
        adjusted=adjusted,
        point_anomalies=point_anomalies,
        level_shifts=level_shifts,
    )
