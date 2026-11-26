"""Load JSON or YAML configuration and merge it with command-line settings. YAML requires PyYAML. Named sections let several scripts share one configuration file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_config_file(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {str(p)}")

    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")

    if suffix == ".json":
        obj = json.loads(text)
    elif suffix in (".yml", ".yaml"):
        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError('YAML configuration requires PyYAML.') from e
        obj = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported configuration extension: {suffix}. Expected .json, .yml, or .yaml.")

    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError('The configuration must contain a top-level mapping.')
    return obj


def select_section(cfg: Dict[str, Any], section: Optional[str]) -> Dict[str, Any]:
    """Return the named mapping when present or use the full configuration."""
    if not section:
        return cfg
    v = cfg.get(section)
    if isinstance(v, dict):
        return v
    return cfg


def merge_config_and_cli(cfg: Dict[str, Any], cli_ns) -> Dict[str, Any]:
    """Apply non-None values from the argparse namespace over the configuration defaults."""
    merged: Dict[str, Any] = dict(cfg)
    for k, v in vars(cli_ns).items():
        if v is not None:
            merged[k] = v
    return merged
