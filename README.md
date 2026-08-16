# Radar Éco INSEE

Détection d'anomalies économiques dans les séries temporelles macro-économiques
de l'INSEE (BDM) pour détecter les **points anormaux** et les **ruptures de
niveau**, puis en produire une **explication en langage naturel** (français),
le tout sous forme de rapport Markdown avec graphiques.

Les résultats réels collent aux grandes inflexions de l'économie française
(crise de 2009, choc COVID début 2020, poussée inflationniste 2022-2023, …).

## Exemple de résultat (taux de chômage des moins de 25 ans, source INSEE)

> **T4 2020** : valeur de **20,50**, soit un niveau inférieur de **1,25** à la
> tendance saisonnière attendue (21,75), un écart de **4,4 écart(s)-type**
> (sévérité forte). La série est anormalement en baisse à cette période.
>
> **T1 2009** : rupture de niveau détectée. Le niveau moyen passe de **20,18**
> à **25,02**, soit une hausse de **24,0 %**.

## Méthode

1. **Récupération** des séries via l'API SDMX publique de la BDM
   (`https://www.bdm.insee.fr/series/sdmx/data/SERIES_BDM/<ID>`) — sans clé.
2. **Décomposition** STL (saison + tendance + résidu) ; repli sur une tendance
   par médiane glissante pour les séries trop courtes ou non saisonnières.
3. **Points anormaux** : score z robuste (médiane/MAD) sur les résidus, avec
   suppression des doublons trop proches.
4. **Ruptures de niveau** : segmentation binaire avec coût L2 et pénalité,
   implémentée ici en `numpy` pur (sans dépendance externe), appliquée à la
   série désaisonnalisée.
5. **Explication NLP** : phrases générées en français à partir des statistiques
   détectées. Enrichissement optionnel par un **LLM local (Ollama)** qui propose
   des hypothèses causales, avec avertissement de vérification.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### CLI

```bash
# Toutes les séries de config/series.yaml
python scripts/run_report.py

# Séries spécifiques
python scripts/run_report.py --series 001688537 001763865

# Seuil de détection personnalisé
python scripts/run_report.py --z 3.0 --penalty 30.0

# Enrichissement avec un LLM local (Ollama lancé sur localhost:11434)
python scripts/run_report.py --llm --model qwen2.5:7b
```

Rapports générés dans `reports/<ID_BDM>/` : `rapport.md` + `plot.png`.

### App Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py
```

Ouvrir http://localhost:8501 : sélection d'une série (catalogue ou ID libre),
curseurs `z`/`penalty`, analyse en un clic (explication, graphique, tableaux),
et enrichissement optionnel par un LLM (Ollama en local ou API hébergée).

#### LLM hébergé (OpenRouter) dans l'app déployée

Le fournisseur LLM est choisi automatiquement : `openai` (API compatible
OpenAI, ex. OpenRouter) si une clé est configurée, sinon `ollama`. En
déploiement Streamlit Cloud, ajouter dans **Paramètres → Secrets** :

```toml
OPENAI_API_KEY = "sk-or-..."
OPENAI_MODEL = "google/gemini-2.0-flash-001"   # optionnel
LLM_PROVIDER = "openai"                          # optionnel
```

Localement, le fournisseur se règle par variables d'environnement
(`LLM_PROVIDER`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`).

### Docker (app + LLM local embarqué)

L'app et Ollama sont conteneurisés ensemble ; le modèle défini par
`OLLAMA_MODEL` (défaut `qwen2.5:1.5b`) est téléchargé au démarrage.

```bash
docker compose up --build
```

- App Streamlit : http://localhost:8501
- Ollama (API) : http://localhost:11434

### Ajouter une série

Trouver un identifiant sur le [catalogue INSEE](https://catalogue-donnees.insee.fr/)
puis l'ajouter à `config/series.yaml` :

```yaml
- id: "001688537"
  label: "Taux de chômage (moins de 25 ans, CVS)"
  description: "Taux de chômage BIT, France hors Mayotte, données CVS (trimestriel)."
```

## Structure

```
├── config/series.yaml        # séries par défaut + paramètres
├── app.py                    # app Streamlit
├── Dockerfile                # image de l'app Streamlit
├── docker-compose.yml        # app + service Ollama (LLM local)
├── scripts/run_report.py     # CLI principale
├── src/radar_eco_insee/
│   ├── data.py               # client SDMX BDM (fetch + parsing XML)
│   ├── detection.py          # STL, z-scores, segmentation binaire
│   ├── explain.py            # explications en français (règles + LLM optionnel)
│   └── report.py             # graphiques matplotlib + rapport Markdown + tableaux
├── tests/test_detection.py   # tests unitaires (unittest)
└── reports/                  # rapports générés
```

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Pistes d'amélioration

- API REST (FastAPI) + petit dashboard web pour naviguer dans les séries.
- Enrichissement LLM avec récupération d'actualités (chocs exogènes expliqués).
- Détection multi-séries croisée (corrélations, anomalies simultanées).
- Publication des séries non arrêtées et surveillance mensuelle automatisée.
