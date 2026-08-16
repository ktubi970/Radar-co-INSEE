"""Tests unitaires de la détection et de l'explication."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd
import requests

from radar_eco_insee.data import SeriesData, parse_period
from radar_eco_insee.detection import (
    binary_segmentation,
    detect_anomalies,
    detect_point_anomalies,
)
from radar_eco_insee.explain import (
    CUSTOM_MODEL,
    OLLAMA_MODELS,
    OPENROUTER_MODELS,
    LLMExplainer,
    RuleBasedExplainer,
    model_options,
)
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

    def test_title_appears_once(self):
        values = np.concatenate([np.full(80, 10.0), np.full(80, 14.0)])
        values[40] += 5.0
        series = make_series(values, title="Taux de chômage test")
        result = detect_anomalies(series)
        explainer = RuleBasedExplainer()
        text = explainer.explain(result)
        self.assertEqual(text.count("Taux de chômage test"), 1)


class TestLLMExplainer(unittest.TestCase):
    def make_result(self):
        values = np.concatenate([np.full(80, 10.0), np.full(80, 14.0)])
        values[40] += 5.0
        series = make_series(values)
        return detect_anomalies(series)

    def test_openai_available_with_key(self):
        explainer = LLMExplainer(provider="openai", api_key="sk-test")
        self.assertTrue(explainer.is_available())

    def test_openai_unavailable_without_key(self):
        explainer = LLMExplainer(provider="openai", api_key="")
        self.assertFalse(explainer.is_available())
        self.assertEqual(explainer.explain(self.make_result(), "règles"), "règles")

    def test_missing_key_records_error(self):
        explainer = LLMExplainer(provider="openai", api_key="")
        explainer.is_available()
        self.assertIn("clé", explainer.last_error.lower())

    def test_openai_explain_uses_chat_completions(self):
        result = self.make_result()
        explainer = LLMExplainer(
            provider="openai",
            base_url="https://example.com/api/v1",
            api_key="sk-test",
            model="modèle-test",
        )
        with mock.patch("radar_eco_insee.explain.requests.post") as mock_post:
            resp = mock.Mock()
            resp.status_code = 200
            resp.raise_for_status.return_value = None
            resp.json.return_value = {"choices": [{"message": {"content": "Hypothèse A."}}]}
            mock_post.return_value = resp
            out = explainer.explain(result, "règles")
        self.assertIn("Hypothèse A.", out)
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "https://example.com/api/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer sk-test")
        self.assertEqual(kwargs["json"]["model"], "modèle-test")

    def test_openai_explain_falls_back_on_error(self):
        result = self.make_result()
        explainer = LLMExplainer(provider="openai", api_key="sk-test")
        with mock.patch(
            "radar_eco_insee.explain.requests.post", side_effect=requests.RequestException
        ):
            out = explainer.explain(result, "règles")
        self.assertEqual(out, "règles")

    def test_openai_records_error_on_failure(self):
        result = self.make_result()
        explainer = LLMExplainer(provider="openai", api_key="sk-test")
        with mock.patch(
            "radar_eco_insee.explain.requests.post", side_effect=requests.RequestException("boom")
        ):
            explainer.explain(result, "règles")
        self.assertIn("boom", explainer.last_error)

    def test_openai_http_error_includes_body(self):
        result = self.make_result()
        explainer = LLMExplainer(provider="openai", api_key="sk-test")
        with mock.patch("radar_eco_insee.explain.requests.post") as mock_post:
            resp = mock.Mock()
            resp.status_code = 404
            resp.text = '{"error": {"message": "Unknown model: foo/bar"}}'
            mock_post.return_value = resp
            explainer.explain(result, "règles")
        self.assertIn("404", explainer.last_error)
        self.assertIn("foo/bar", explainer.last_error)


class TestModelOptions(unittest.TestCase):
    def test_openai_known_model_defaults_to_it(self):
        options, index = model_options("openai", "google/gemini-3.7-flash")
        self.assertEqual(index, 0)
        self.assertEqual(options[index], "google/gemini-3.7-flash")

    def test_unknown_model_defaults_to_custom(self):
        options, index = model_options("openai", "foo/bar")
        self.assertEqual(options[index], CUSTOM_MODEL)

    def test_ollama_uses_ollama_catalog(self):
        options, index = model_options("ollama", "qwen2.5:7b")
        self.assertEqual(options[index], "qwen2.5:7b")
        self.assertIn("llama3.2:3b", options)
        self.assertNotIn("deepseek/deepseek-v3.2", options)

    def test_catalogs_are_non_empty(self):
        self.assertTrue(OPENROUTER_MODELS)
        self.assertTrue(OLLAMA_MODELS)


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
