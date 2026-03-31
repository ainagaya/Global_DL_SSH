from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = repo_root() / config_file

    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config["__config_path__"] = str(config_file)
    config["__repo_root__"] = str(repo_root())
    return config


def resolve_path(path_value: str | Path, config: Dict[str, Any]) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return Path(config["__repo_root__"]) / path


def ensure_dir(path_value: str | Path, config: Dict[str, Any]) -> Path:
    path = resolve_path(path_value, config)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_split_config(config: Dict[str, Any], split_name: str) -> Dict[str, Any]:
    splits = config.get("splits", {})
    if split_name not in splits:
        raise KeyError(f"Unknown split '{split_name}'. Available splits: {sorted(splits)}")
    return splits[split_name]
