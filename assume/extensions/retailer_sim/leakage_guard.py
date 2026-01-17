# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Runtime guard that prevents data leakage across market phases."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Optional

import pandas as pd


class DataLeakageError(RuntimeError):
    """Raised when a forbidden column is accessed for the current phase."""


class Phase(str, Enum):
    MGP = "MGP"
    MI = "MI"
    MSD = "MSD"


# Colonne che contengono dati "reali" e non devono essere lette in fasi previsionali.
REAL_ONLY_COLUMNS = {
    "actual_consumption_MWh",
    "actual_consumption_delayed_MWh",
    "actual_imbalance_MWh",
    "SBIL_MWH",
    "MGP_PRICE_NORD",
    "MI_PRICE_NORD",
    "msd_price_real_tmp",
}


@dataclass
class GuardConfig:
    """Struttura dati leggera per configurare rapidamente il guard."""

    enabled: bool = False
    debug: bool = False


class DataLeakageGuard:
    """Centralised policy enforcement for per-phase data availability."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        debug: bool = False,
        real_columns: Optional[Iterable[str]] = None,
        session_definitions: Optional[Iterable[Dict[str, float]]] = None,
    ) -> None:
        """
        Parametri:
          - enabled: attiva/disattiva completamente la guardia;
          - debug: produce log quando le colonne vengono ammesse;
          - real_columns: elenco di colonne "tabù" finché non siamo in MSD;
          - session_definitions: metadati sulle sessioni MI (per la regola delle 4h).
        """
        self.enabled = bool(enabled)
        self.debug = bool(debug)
        self.logger = logging.getLogger(__name__)
        self.real_columns = set(real_columns or REAL_ONLY_COLUMNS)
        self.session_definitions = list(session_definitions or [])

    @classmethod
    def from_config(
        cls,
        config: Optional[Dict[str, object]],
        *,
        session_definitions: Optional[Iterable[Dict[str, float]]] = None,
    ) -> "DataLeakageGuard":
        """Factory che legge il blocco di configurazione `data_leakage_guard`."""
        cfg = config or {}
        return cls(
            enabled=cfg.get("enabled", False),
            debug=cfg.get("debug", False),
            real_columns=cfg.get("real_columns"),
            session_definitions=session_definitions,
        )

    # ------------------------------------------------------------------ #
    # Column access helpers
    # ------------------------------------------------------------------ #
    def check_column(
        self,
        *,
        phase: Phase,
        column: Optional[str],
        timestamp: Optional[pd.Timestamp] = None,
        delivery_time: Optional[pd.Timestamp] = None,
    ) -> None:
        """Verifica che la colonna richiesta sia disponibile nella fase corrente."""
        if not self.enabled or not column:
            return
        if phase != Phase.MSD and column in self.real_columns:
            raise DataLeakageError(
                f"Column '{column}' is not available during phase {phase.value} "
                f"(timestamp={timestamp}, delivery_time={delivery_time})."
            )
        if self.debug:
            self.logger.debug(
                "LeakageGuard allowed column=%s phase=%s ts=%s delivery=%s",
                column,
                phase.value,
                timestamp,
                delivery_time,
            )

    # ------------------------------------------------------------------ #
    # Session window enforcement (~4h rule)
    # ------------------------------------------------------------------ #
    def ensure_session_window(self, session_name: str, delivery_time: pd.Timestamp) -> None:
        """Applica la regola delle ~4 ore fra apertura sessione e slot consegnato."""
        if not self.enabled:
            return
        session = self._find_session(session_name)
        if session is None:
            return
        hour_value = delivery_time.hour + delivery_time.minute / 60
        if not self._slot_in_window(
            hour_value,
            float(session.get("coverage_start_hour", 0.0)),
            float(session.get("coverage_end_hour", 24.0)),
        ):
            raise DataLeakageError(
                f"Session {session_name} cannot trade slot {delivery_time}: "
                "coverage window constraint (~4h rule) violated."
            )

    def _find_session(self, session_name: str) -> Optional[Dict[str, float]]:
        """Restituisce la definizione di sessione (case-insensitive)."""
        target = session_name.upper()
        for entry in self.session_definitions:
            if entry.get("name", "").upper() == target:
                return entry
        return None

    @staticmethod
    def _slot_in_window(hour_value: float, start: float, end: float) -> bool:
        """Controlla se uno slot (espresso in ore float) cade dentro l'intervallo coperto."""
        window_start = start
        window_end = end
        if window_end <= window_start:
            window_end += 24
        hour = hour_value
        if hour < window_start:
            hour += 24
        return window_start <= hour < window_end
