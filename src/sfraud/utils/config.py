"""Config loading helpers. Every script reads its parameters from configs/config.yaml
so there is exactly one place to change hyperparameters or paths."""

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


def load_config(path: "str | Path | None" = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(relative: str) -> Path:
    """Resolve a path from config.yaml (given relative to repo root) to an absolute Path."""
    p = Path(relative)
    return p if p.is_absolute() else REPO_ROOT / p


def figures_dir() -> Path:
    d = REPO_ROOT / "reports" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def metrics_dir() -> Path:
    d = REPO_ROOT / "reports" / "metrics"
    d.mkdir(parents=True, exist_ok=True)
    return d


def samples_dir() -> Path:
    d = REPO_ROOT / "reports" / "samples"
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoints_dir() -> Path:
    d = REPO_ROOT / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d
