#!/usr/bin/env python3
"""Analyse des séries INSEE (BDM) : détection d'anomalies + rapport.

Exemples :
    python scripts/run_report.py
    python scripts/run_report.py --series 001688537
    python scripts/run_report.py --series 001688537 001763865 --z 3.0
    python scripts/run_report.py --llm --model qwen2.5:7b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from radar_eco_insee.data import INSEEBDMClient
from radar_eco_insee.detection import detect_anomalies
from radar_eco_insee.explain import LLMExplainer, RuleBasedExplainer
from radar_eco_insee.report import build_report


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", nargs="*", help="Identifiants BDM (ex. 001688537). Par défaut : config/series.yaml")
    parser.add_argument("--config", type=Path, default=Path("config/series.yaml"), help="Fichier de configuration des séries")
    parser.add_argument("--out", type=Path, default=Path("reports"), help="Dossier de sortie des rapports")
    parser.add_argument("--z", type=float, default=None, help="Seuil |z| pour les points anormaux (défaut : 2.5)")
    parser.add_argument("--penalty", type=float, default=None, help="Pénalité de segmentation (défaut : 25.0)")
    parser.add_argument("--llm", action="store_true", help="Enrichir avec un LLM local (Ollama)")
    parser.add_argument("--model", type=str, default=None, help="Modèle Ollama (ex. qwen2.5:7b)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    defaults = cfg.get("defaults", {})
    series_list = cfg.get("series", [])

    if args.series:
        series_list = [{"id": sid, "label": sid} for sid in args.series]

    z_threshold = args.z if args.z is not None else defaults.get("z_threshold", 2.5)
    penalty = args.penalty if args.penalty is not None else defaults.get("penalty", 25.0)

    explainer: RuleBasedExplainer | LLMExplainer
    if args.llm:
        explainer = LLMExplainer(model=args.model or LLMExplainer.OLLAMA_MODEL)
        if not explainer.is_available():
            print("! LLM injoignable : utilisation de l'explication par règles.", file=sys.stderr)
            explainer = RuleBasedExplainer()
    else:
        explainer = RuleBasedExplainer()

    client = INSEEBDMClient()
    out_root = args.out

    for entry in series_list:
        series_id = entry["id"]
        print(f"[{series_id}] Récupération...")
        try:
            series = client.fetch_series(series_id)
        except Exception as exc:
            print(f"[{series_id}] Échec de la récupération : {exc}", file=sys.stderr)
            continue

        n = len(series.data)
        if n < defaults.get("min_points", 40):
            print(f"[{series_id}] Attention : seulement {n} observations.", file=sys.stderr)

        print(f"[{series_id}] Analyse ({n} obs, {series.frequency_label})...")
        result = detect_anomalies(series, z_threshold=z_threshold, penalty=penalty)
        print(
            f"[{series_id}] -> {len(result.point_anomalies)} point(s) anormal(aux), "
            f"{len(result.level_shifts)} rupture(s) de niveau"
        )

        series_dir = out_root / series_id
        report_path = build_report(result, explainer, series_dir)
        print(f"[{series_id}] Rapport : {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
