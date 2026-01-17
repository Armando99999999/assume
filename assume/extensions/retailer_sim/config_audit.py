"""Lightweight config audit helpers."""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

LOGGER = logging.getLogger(__name__)


KNOWN_TOP_LEVEL_KEYS = {
    "decision_making",
    "market",
    "simulation",
    "retailer_logic",
    "macro_imbalance_forecast",
    "forecast_mapping",
    "forecasting",
    "information_flow",
    "data_availability",
    "logging",
}

def audit_config(config: Mapping[str, object]) -> None:
    """Emit warnings for unexpected or deprecated configuration keys."""
    for key in config.keys():
        if key.startswith("_"):
            continue
        if key not in KNOWN_TOP_LEVEL_KEYS:
            LOGGER.warning("Config section '%s' is not recognised and will be ignored.", key)
