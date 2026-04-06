"""Application settings loaded from optional local config (not committed)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AppConfig:
    google_maps_api_key: str | None = None


def load_app_config() -> AppConfig:
    """Load settings from config.local.yaml; env GOOGLE_MAPS_API_KEY overrides the file."""
    key: str | None = None
    local_path = _CONFIG_DIR / "config.local.yaml"
    if local_path.is_file():
        with open(local_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        raw = data.get("google_maps_api_key")
        if raw is not None:
            s = str(raw).strip()
            if s:
                key = s
    env_key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if env_key:
        key = env_key
    return AppConfig(google_maps_api_key=key)
