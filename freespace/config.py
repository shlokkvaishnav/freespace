"""Config loading: one YAML file, read into nested attribute dicts."""
from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class Cfg(dict):
    """dict with attribute access, recursive."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Cfg:
    with open(path or DEFAULT_PATH) as f:
        cfg = yaml.safe_load(f)
    for key, val in (overrides or {}).items():
        section, name = key.split(".", 1)
        cfg[section][name] = val
    return Cfg(cfg)
