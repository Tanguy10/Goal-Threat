"""Central paths for the Goal Threat project.

All data, models and results are resolved relative to the repository root so that
scripts run identically from any working directory (run them as
``python src/<script>.py`` from the repo root).
"""
import sys
from pathlib import Path

# Scripts log with UTF-8 characters; keep stdout/stderr UTF-8 on legacy consoles.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
METRICA_DIR = DATA_DIR / "metrica"
PROCESSED_DIR = DATA_DIR / "processed"
STATSBOMB_360_DIR = DATA_DIR / "statsbomb_360"

MODELS_DIR = ROOT / "models"

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
HEATMAPS_DIR = RESULTS_DIR / "heatmaps"
GOAL_THREAT_MAPS_DIR = RESULTS_DIR / "goal_threat_maps"

for _d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, HEATMAPS_DIR, GOAL_THREAT_MAPS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
