"""Tests unitaires de la détection et de l'explication."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from radar_eco_insee.data import SeriesData, parse_period
from radar_eco_insee.detection import (
    binary_segmentation,
    detect_anomalies,
    detect_point_anomalies,
)
from radar_eco_insee.explain import RuleBasedExplainer
from radar_eco_insee.report import detections_dataframe


def make_series(values, freq="T", unit="POURCENT", title="Série test") -> SeriesData:
    periods = []
    for i in range(len(values)):
        year = 2000 + i // 4
        q = i % 4 + 1
        periods.append(f"{year}-Q{q}")
    df = pd.DataFrame(
        {
            "date": pd.to_datetime([parse_period(p) for p in periods]),
            "period": periods,
            "value": values,
        }
    )
    return SeriesData(
        id="TEST",
        title_fr=title,
        freq=freq,
        unit_measure=unit,
        seasonal_period={"M": 12, "T": 4, "S": 2, "A": 1}[freq],
        data=df,
    )


class TestParsePeriod(unittest.TestCase):
    def test_quarter(self):
        self.assertEqual(parse_period("2026-Q1"), "2026-01-01")
        self.assertEqual(parse_period("2026-Q4"), "2026-10-01")

    def test_month(self):
        self.assertEqual(parse_period("2025-05"), "2025-05-01")

    def test_year(self):
        self.assertEqual(parse_period("2025"), "2025-01-01")


class TestPointAnomalyDetection(unittest.TestCase):
    def test_spike_detected(self):
        residual = np.zeros(100)
        residual[50] = 8.0  # écart énorme -> z élevé
        z = residual / 1.4826
        indices = detect_point_anomalies(residual, z, z_threshold=2.5)
        self.assertEqual(len(indices), 1)
        self.assertEqual(indices[0][0], 50)


class TestBinarySegmentation(unittest.TestCase):
    def test_level_shift_detected(self):
        n = 120
        rng = np.random.default_rng(42)
        values = np.concatenate([rng.normal(10, 1, n // 2), rng.normal(14, 1, n // 2)])
        breaks = binary_segmentation(values, min_seg=4, penalty=20.0)
        self.assertTrue(breaks, "aucune rupture détectée")
        self.assertEqual(breaks[0], n // 2)

    def test_no_false_positive_on_flat(self):
        rng = np.random.default_rng(7)
        values = rng.normal(10, 1, 80)
        breaks = binary_segmentation(values, min_seg=4, penalty=50.0)
        self.assertEqual(breaks, [])


class TestFullDetection(unittest.TestCase):
    def test_seasonal_spike_and_shift(self):
        rng = np.random.default_rng(1)
        n = 160  # 40 ans de trimestriel
        season = np.tile([2.0, 1.0, -1.0, -2.0], n // 4)
        trend = np.linspace(20, 30, n)
        noise = rng.normal(0, 0.3, n)
        values = season + trend + noise
        values[120] += 5.0  # pic anormal
        values[40:] += 6.0  # rupture de niveau (à partir de la période 40)

        series = make_series(values)
        result = detect_anomalies(series, z_threshold=2.5, penalty=20.0)

        point_dates = [a.period for a in result.point_anomalies]
        self.assertIn("2030-Q1", point_dates)  # pic en indice 120 -> année 2030

        shift_dates = [s.period for s in result.level_shifts]
        self.assertTrue(any(s.period == "2010-Q1" for s in result.level_shifts), shift_dates)


class TestExplainer(unittest.TestCase):
    def test_french_output(self):
        values = np.concatenate([np.full(80, 10.0), np.full(80, 14.0)])
        series = make_series(values, title="Taux de chômage test")
        result = detect_anomalies(series)
        explainer = RuleBasedExplainer()
        text = explainer.explain(result)
        self.assertIn("Taux de chômage test", text)
        self.assertIn("rupture de niveau", text)


class TestReportDataFrames(unittest.TestCase):
    def test_detections_dataframe(self):
        values = np.concatenate([np.full(80, 10.0), np.full(80, 14.0)])
        values[40] += 5.0
        series = make_series(values)
        result = detect_anomalies(series)
        points, shifts = detections_dataframe(result)
        self.assertEqual(list(shifts.columns), ["Période", "Niveau avant", "Niveau après", "Direction", "Amplitude (%)"])
        self.assertEqual(list(points.columns), ["Période", "Valeur", "Attendu", "Écart (z)", "Sévérité"])
        self.assertFalse(points.empty)


if __name__ == "__main__":
    unittest.main()
