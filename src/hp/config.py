"""Configuration loading with dotted-path overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = _REPO_ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(text: str) -> Any:
    """Parse a CLI ``key=value`` string into a natural python type."""
    lowered = text.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null", "~"}:
        return None
    # lists like [1,2,3]
    if text.strip().startswith("[") and text.strip().endswith("]"):
        inner = text.strip()[1:-1].strip()
        if not inner:
            return []
        return [_coerce(part) for part in inner.split(",")]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _set_path(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


class Config(dict):
    """Dict with ``cfg.io.max_keyframes`` style attribute access."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load default config, merge an optional YAML file, then apply overrides.

    Overrides look like ``change.voxel_size=0.04``.
    """
    base_path = Path(path) if path else DEFAULT_CONFIG
    if not base_path.exists():
        raise FileNotFoundError(f"config not found: {base_path}")
    with open(base_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    if path and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        # merge a user config on top of defaults
        with open(DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
            defaults = yaml.safe_load(fh) or {}
        cfg = _deep_merge(defaults, cfg)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got: {item!r}")
        key, value = item.split("=", 1)
        _set_path(cfg, key.strip(), _coerce(value))

    return Config(cfg)
