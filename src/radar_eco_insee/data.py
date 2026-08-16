"""Client minimal pour l'API SDMX de la Banque de données macro-économiques (BDM) de l'INSEE."""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import pandas as pd
import requests

BASE_URL = "https://www.bdm.insee.fr/series/sdmx/data/SERIES_BDM/{series_id}"
TIMEOUT_SECONDS = 30

# Fréquence -> périodicité (nombre d'observations par an)
FREQ_SEASONAL = {"M": 12, "T": 4, "S": 2, "A": 1}


def _local_name(tag: str) -> str:
    """Retourne le nom local d'un tag XML (après l'éventuel namespace)."""
    return tag.rsplit("}", 1)[-1]


def parse_period(period: str) -> str:
    """Transforme un TIME_PERIOD SDMX ('2026-Q1', '2026-05', '2026') en date ISO 'YYYY-MM-DD'."""
    if "Q" in period:
        year, quarter = period.split("-Q")
        month = {"1": "01", "2": "04", "3": "07", "4": "10"}[quarter]
        return f"{year}-{month}-01"
    parts = period.split("-")
    if len(parts) == 2:
        return f"{parts[0]}-{parts[1]}-01"
    return f"{parts[0]}-01-01"


def parse_frequency(freq: str) -> str:
    return {"M": "Mensuelle", "T": "Trimestrielle", "S": "Semestrielle", "A": "Annuelle"}.get(freq, freq)


@dataclass
class SeriesData:
    """Série temporelle récupérée depuis la BDM."""

    id: str
    title_fr: str
    freq: str  # 'M', 'T', 'S', 'A'
    unit_measure: str
    seasonal_period: int
    data: pd.DataFrame  # index datetime (période), colonne 'value'

    @property
    def frequency_label(self) -> str:
        return parse_frequency(self.freq)


class INSEEBDMClient:
    """Récupère les séries de la BDM au format SDMX 2.1 et les convertit en DataFrame."""

    def fetch_series(self, series_id: str) -> SeriesData:
        url = BASE_URL.format(series_id=series_id)
        response = requests.get(url, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        root = ET.fromstring(response.content)

        series_el = None
        for el in root.iter():
            if _local_name(el.tag) == "Series":
                series_el = el
                break
        if series_el is None:
            raise ValueError(f"Série {series_id} introuvable dans la réponse SDMX")

        freq = series_el.get("FREQ", "A")
        obs: list[tuple[str, float]] = []
        for el in series_el.iter():
            if _local_name(el.tag) == "Obs":
                period = el.get("TIME_PERIOD")
                value = el.get("OBS_VALUE")
                if period and value is not None:
                    try:
                        obs.append((period, float(value)))
                    except ValueError:
                        continue

        if not obs:
            raise ValueError(f"Série {series_id} vide")

        records = [
            {"date": parse_period(period), "period": period, "value": value}
            for period, value in obs
        ]
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)

        return SeriesData(
            id=series_id,
            title_fr=series_el.get("TITLE_FR", f"Série {series_id}"),
            freq=freq,
            unit_measure=series_el.get("UNIT_MEASURE", ""),
            seasonal_period=FREQ_SEASONAL.get(freq, 1),
            data=df,
        )

    def fetch_series_to_csv(self, series_id: str, path: str) -> None:
        series = self.fetch_series(series_id)
        series.data.to_csv(path, index=False)
