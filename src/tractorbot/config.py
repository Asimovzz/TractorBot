from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: Optional[Union[str, os.PathLike]] = None) -> Dict[str, Any]:
    """Load a TractorBot YAML configuration file."""
    config_path = Path(path or os.environ.get("TRACTORBOT_CONFIG", DEFAULT_CONFIG_PATH))
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    config["_config_path"] = str(config_path)
    return config


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries without mutating either input."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def build_training_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the flat config shape expected by Actor and Learner."""
    rl_config = dict(config.get("rl", {}))
    training_config = dict(config.get("training", {}))
    merged = deep_merge(rl_config, training_config)

    if "model_pool" in merged:
        pool = merged.pop("model_pool") or {}
        merged.setdefault("model_pool_size", pool.get("size"))
        merged.setdefault("model_pool_name", pool.get("name"))

    merged["experiment"] = dict(config.get("experiment", {}))
    merged["env"] = dict(config.get("env", {}))
    merged["opponents"] = dict(config.get("opponents", {}))
    merged["model"] = dict(config.get("model", {}))
    merged["_config_path"] = config.get("_config_path")

    return {key: value for key, value in merged.items() if value is not None}
