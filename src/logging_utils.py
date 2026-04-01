from __future__ import annotations

import logging
from typing import Any, Dict


def configure_logging(config: Dict[str, Any]) -> None:
    logging_cfg = config.get("logging", {})
    level_name = str(logging_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
