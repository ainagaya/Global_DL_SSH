from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.config_utils import load_config


@pytest.fixture
def base_config():
    config = load_config(Path("configs/oceantaco.yaml"))
    return copy.deepcopy(config)
