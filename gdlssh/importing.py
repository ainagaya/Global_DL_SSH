from __future__ import annotations

import importlib
from typing import Any


def import_from_string(spec: str) -> Any:
    try:
        module_name, attr_name = spec.split(":", maxsplit=1)
    except ValueError as exc:
        raise ValueError(f"Invalid import spec '{spec}'. Expected 'module:attribute'.") from exc
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)
