from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict

from src.config_utils import resolve_path


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _stringify_param(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    return json.dumps(value, sort_keys=True, default=_json_default)


def flatten_config(config: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
    flat: Dict[str, str] = {}
    for key, value in config.items():
        if str(key).startswith("__"):
            continue
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_config(value, full_key))
        else:
            flat[full_key] = _stringify_param(value)
    return flat


class MLflowTracker:
    def __init__(self, config: Dict[str, Any], stage: str):
        self.config = config
        self.stage = stage
        tracking_cfg = config.get("tracking", {}).get("mlflow", {})
        self.enabled = bool(tracking_cfg.get("enabled", False))
        self._tracking_cfg = tracking_cfg
        self._mlflow = None
        self._active_run = None

        if self.enabled:
            try:
                import mlflow
            except ImportError as exc:
                raise ImportError(
                    "MLflow tracking is enabled in the config, but `mlflow` is not installed."
                ) from exc
            self._mlflow = mlflow

    def start_run(self, run_name: str | None = None, extra_tags: Dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return

        tracking_uri = self._tracking_cfg.get("tracking_uri")
        if tracking_uri:
            uri_value = str(tracking_uri)
            if "://" not in uri_value:
                uri_value = str(resolve_path(uri_value, self.config))
            self._mlflow.set_tracking_uri(uri_value)

        experiment_name = self._tracking_cfg.get("experiment_name", "Global_DL_SSH")
        self._mlflow.set_experiment(experiment_name)

        tags = {"stage": self.stage}
        tags.update(self._tracking_cfg.get("tags", {}))
        if extra_tags:
            tags.update(extra_tags)

        resolved_run_name = run_name or self._tracking_cfg.get("run_name")
        self._active_run = self._mlflow.start_run(run_name=resolved_run_name, tags=tags)

    def log_config(self) -> None:
        if not self.enabled:
            return
        if self._tracking_cfg.get("log_params", True):
            self._mlflow.log_params(flatten_config(self.config))
        if self._tracking_cfg.get("log_config_artifact", True):
            self.log_artifact(self.config["__config_path__"], artifact_subdir="config")

    def log_metrics(self, metrics: Dict[str, float], step: int | None = None) -> None:
        if not self.enabled:
            return
        clean_metrics = {key: float(value) for key, value in metrics.items()}
        self._mlflow.log_metrics(clean_metrics, step=step)

    def log_artifact(self, artifact_path: str | Path, artifact_subdir: str | None = None, **kwargs: Any) -> None:
        if not self.enabled:
            return
        target_subdir = artifact_subdir if artifact_subdir is not None else kwargs.get("artifact_path")
        self._mlflow.log_artifact(str(artifact_path), artifact_path=target_subdir)

    def set_tag(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        self._mlflow.set_tag(key, value)

    def end_run(self) -> None:
        if not self.enabled:
            return
        if self._active_run is not None:
            self._mlflow.end_run()
            self._active_run = None
