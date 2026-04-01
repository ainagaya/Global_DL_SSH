from __future__ import annotations

from pathlib import Path

from src.config_utils import ensure_dir, get_split_config, load_config, resolve_path


def test_load_config_adds_internal_paths():
    config = load_config("configs/oceantaco.yaml")

    assert "__config_path__" in config
    assert "__repo_root__" in config
    assert config["data"]["sequence_length"] == 5


def test_resolve_path_uses_repo_root(base_config):
    resolved = resolve_path("model_weights", base_config)

    assert resolved.is_absolute()
    assert resolved.name == "model_weights"


def test_ensure_dir_creates_directory(tmp_path, base_config):
    base_config["__repo_root__"] = str(tmp_path)

    created = ensure_dir("artifacts/checkpoints", base_config)

    assert created == tmp_path / "artifacts" / "checkpoints"
    assert created.exists()
    assert created.is_dir()


def test_get_split_config_returns_named_split(base_config):
    split = get_split_config(base_config, "train")

    assert split["strategy"] == "training"
    assert split["n_queries"] == 256
