"""
pipeline/config_loader.py
─────────────────────────
Central config loader. Load config.yaml sekali, pakai di mana saja.

Usage:
    from pipeline.config_loader import config
    threshold = config["thresholds"]["faithfulness_evidence"]
    term_groups = config["technical_term_groups"]
"""
import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def _load_config() -> dict:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"config.yaml tidak ditemukan di {_CONFIG_PATH}\n"
            f"Pastikan file ada di root proyek."
        )
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Singleton — load sekali saat import, pakai berkali-kali tanpa re-read disk
config = _load_config()