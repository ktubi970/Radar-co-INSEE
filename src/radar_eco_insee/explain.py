"""Explication des anomalies détectées en langage naturel (français).

Deux niveaux :
- `RuleBasedExplainer` : phrases générées à partir des statistiques détectées
  (déterministe, aucune dépendance, toujours disponible).
- `LLMExplainer` : enrichit le texte avec des hypothèses causales plausibles en
  interrogeant un LLM local (Ollama). Optionnel : si Ollama n'est pas lancé, on
  retombe silencieusement sur l'explication par règles.
"""

from __future__ import annotations

import os
from typing import Optional

import requests

from .detection import DetectionResult, LevelShift, PointAnomaly

CUSTOM_MODEL = "Autre (modèle personnalisé)..."

OPENROUTER_MODELS = [
    "google/gemini-3.7-flash",
    "google/gemini-3.5-flash",
    "google/gemini-3.1-flash-lite",
    "deepseek/deepseek-v3.2",
    "meta-llama/llama-3.3-70b-instruct",
    "anthropic/claude-sonnet-5",
]

OLLAMA_MODELS = [
    "qwen2.5:7b",
    "qwen2.5:1.5b",
    "llama3.2:3b",
    "mistral:7b",
]


def model_options(provider: str, current: str | None = None) -> tuple[list[str], int]:
    """Retourne (options de sélection, index par défaut) pour un fournisseur."""
    catalog = OPENROUTER_MODELS if provider == "openai" else OLLAMA_MODELS
    options = catalog + [CUSTOM_MODEL]
    if current and current in catalog:
        return options, catalog.index(current)
    return options, len(options) - 1


def _round(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _format_period(period: str) -> str:
    if "Q" in period:
        year, quarter = period.split("-Q")
        return {1: "T1", 2: "T2", 3: "T3", 4: "T4"}[int(quarter)] + f" {year}"
    return period


class RuleBasedExplainer:
    """Génère des explications déterministes en français."""

    def explain_point(self, anomaly: PointAnomaly) -> str:
        delta = anomaly.value - anomaly.expected
        sign = "supérieur" if delta >= 0 else "inférieur"
        direction = "en hausse" if delta >= 0 else "en baisse"
        txt = (
            f"**{_format_period(anomaly.period)}** : valeur de **{_round(anomaly.value)}**, "
            f"soit un niveau {sign} de **{_round(abs(delta))}** à la tendance "
            f"saisonnière attendue ({_round(anomaly.expected)}), un écart de "
            f"**{_round(abs(anomaly.z_score), 1)} écart(s)-type** (sévérité {anomaly.severity}). "
            f"La série est anormalement {direction} à cette période."
        )
        return txt

    def explain_shift(self, shift: LevelShift) -> str:
        verbe = "passe" if shift.direction == "hausse" else "chute"
        txt = (
            f"**{_format_period(shift.period)}** : rupture de niveau détectée. "
            f"Le niveau moyen {verbe} de **{_round(shift.mean_before)}** à "
            f"**{_round(shift.mean_after)}**, soit une {shift.direction} de "
            f"**{_round(abs(shift.magnitude_pct), 1)} %**."
        )
        return txt

    def explain(self, result: DetectionResult) -> str:
        title = result.series.title_fr
        if not result.has_anomalies:
            return (
                f"Aucune anomalie significative détectée sur « {title} » "
                f"({result.n_obs} observations)."
            )
        lines = [f"**Série** : {title}"]
        if result.point_anomalies:
            lines.append("")
            lines.append("**Points anormaux** (écart à la tendance saisonnière) :")
            lines.extend(self.explain_point(a) for a in result.point_anomalies)
        if result.level_shifts:
            lines.append("")
            lines.append("**Ruptures de niveau** (changement de régime) :")
            lines.extend(self.explain_shift(s) for s in result.level_shifts)
        return "\n".join(lines)


class LLMExplainer:
    """Enrichit les explications avec un LLM, si disponible.

    Deux fournisseurs :
    - `"ollama"` (défaut) : LLM local (Ollama), endpoint `/api/generate`.
    - `"openai"` : API compatible OpenAI (ex. OpenRouter), endpoint `/chat/completions`,
      clé API obligatoire.
    """

    DEFAULT_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
    OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "google/gemini-3.7-flash")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

    def __init__(
        self,
        provider: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        self.provider = (provider or self.DEFAULT_PROVIDER).lower()
        if self.provider == "openai":
            self.base_url = (base_url or self.OPENAI_BASE_URL).rstrip("/")
            self.model = model or self.OPENAI_MODEL
            self.api_key = api_key or self.OPENAI_API_KEY
        else:
            self.provider = "ollama"
            self.base_url = (base_url or self.OLLAMA_BASE_URL).rstrip("/")
            self.model = model or self.OLLAMA_MODEL
            self.api_key = None
        self._available: Optional[bool] = None
        self.last_error: Optional[str] = None

    def is_available(self) -> bool:
        if self._available is None:
            if self.provider == "openai":
                self._available = bool(self.api_key)
                if not self._available:
                    self.last_error = "Clé API manquante (OPENAI_API_KEY)."
            else:
                try:
                    r = requests.get(f"{self.base_url}/api/tags", timeout=3)
                    self._available = r.status_code == 200
                    if not self._available:
                        self.last_error = f"Réponse inattendue de {self.base_url}."
                except requests.RequestException as exc:
                    self._available = False
                    self.last_error = f"Ollama injoignable : {exc}"
        return self._available

    def _chat(self, prompt: str) -> str:
        """Interroge le modèle et retourne sa réponse (chaîne vide si vide)."""
        if self.provider == "openai":
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        r = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def explain(self, result: DetectionResult, rule_text: str) -> str:
        """Génère des hypothèses causales plausibles, sans remplacer les faits."""
        if not self.is_available():
            return rule_text

        point_summary = "; ".join(
            f"{a.period}: {a.value} (écart {a.z_score:.1f} sigma, {a.severity})"
            for a in result.point_anomalies[:5]
        )
        shift_summary = "; ".join(
            f"{s.period}: {s.direction} de {s.magnitude_pct:.1f} %"
            for s in result.level_shifts[:5]
        )
        prompt = (
            "Tu es un économiste. Voici des anomalies détectées sur une série INSEE.\n"
            f"Série : {result.series.title_fr} (unité : {result.series.unit_measure}).\n"
            f"Points anormaux : {point_summary or 'aucun'}\n"
            f"Ruptures de niveau : {shift_summary or 'aucune'}\n"
            "Propose 2 à 4 hypothèses explicatives plausibles en français, en une "
            "phrase chacune, et termine par un avertissement indiquant que ces "
            "hypothèses doivent être vérifiées avec d'autres sources. "
            "Reste factuel, n'invente pas de chiffres."
        )
        try:
            hypotheses = self._chat(prompt)
            if not hypotheses:
                self.last_error = "Réponse LLM vide."
                return rule_text
            return f"{rule_text}\n\n**Hypothèses d'explication (LLM) :**\n{hypotheses}"
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            self.last_error = str(exc)
            return rule_text
