# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Logica di offerta condivisa usata sia nello standalone sia nel World adapter.

Walkthrough in stile colloquio tecnico
--------------------------------------
1. ``StrategyInputs`` e' il contenitore unico dei segnali di mercato: ogni ambiente
   (standalone o world) istanzia questa struttura per ciascuno slot e la passa alla
   strategia di bidding, mantenendo il codice decisionale agnostico rispetto alla
   sorgente dati.
2. ``BiddingStrategy`` definisce l'interfaccia fra strategia e simulatore: metodi
   distinti per pianificazione MGP e correzioni MI, con estensioni concrete come
   ``RandomBiddingStrategy`` e ``SimpleRetailStrategy``.
3. ``build_strategy`` e' la factory centrale che restituisce l'implementazione scelta
   in configurazione, garantendo coerenza tra il mondo ASSUME e il simulatore
   standalone.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd


@dataclass
class StrategyInputs:
    """
    Confezione di segnali a disposizione del retailer per uno slot di consegna.

    Il motivo per cui esiste: mantenere una API identica tra l'ambiente standalone e
    l'adapter World. Ogni layer di integrazione crea la struttura, la popola con i
    dati (domanda prevista, prezzi, forecast macro) e la passa alla stessa strategia.
    """

    timestamp: pd.Timestamp
    demand_forecast_MWh: float
    mgp_price_EUR_MWh: float
    mi_price_EUR_MWh: float
    macro_price_EUR_MWh: float
    imbalance_forecast_MWh: float
    macro_imbalance_MWh: float
    imbalance_coeff: float
    msd_price_estimate_EUR_MWh: float
    macro_forecast_mean_MWh: float = float("nan")
    macro_forecast_low_MWh: float = float("nan")
    macro_forecast_high_MWh: float = float("nan")
    macro_forecast_sign: float = float("nan")

    def for_intraday(self, *, mi_price_EUR_MWh: float, macro_band: Optional[float] = None) -> "StrategyInputs":
        """
        Restituisce una copia con il nuovo prezzo MI e, facoltativamente, aggiorna la
        banda del forecast macro (utile quando sessioni differenti hanno bande dedicate).
        """
        payload = {"mi_price_EUR_MWh": mi_price_EUR_MWh}
        if macro_band is not None and not np.isnan(self.macro_forecast_mean_MWh):
            payload["macro_forecast_low_MWh"] = self.macro_forecast_mean_MWh - macro_band
            payload["macro_forecast_high_MWh"] = self.macro_forecast_mean_MWh + macro_band
        return replace(self, **payload)


def estimate_msd_price(
    *,
    mgp_price: float,
    macro_price: float,
    imbalance_coeff: float,
    imbalance_penalty: float = 0.0,
) -> float:
    """
    Stima il prezzo di sbilanciamento MSD usando la formula italiana semplificata:
    ``prezzo = Pz + (Pz - Pmz) * coefficiente + penalita``. Il ritorno e' clippato
    a zero per evitare valori negativi.
    """
    delta = mgp_price - macro_price
    price = mgp_price + delta * imbalance_coeff + imbalance_penalty
    return price


class BiddingStrategy:
    """API base: definisce cosa una strategia deve implementare per MGP e MI."""

    def supports_second_session(self) -> bool:
        """Per default supportiamo solo MI1."""
        return False

    def plan_day_ahead(self, inputs: StrategyInputs) -> float:
        """Volume MGP (MWh) desiderato per lo slot specifico."""
        raise NotImplementedError

    def plan_intraday(
        self,
        inputs: StrategyInputs,
        *,
        contracted_volume_MWh: float,
        session: int = 1,
    ) -> float:
        """Aggiustamento MI (MWh) per la sessione richiesta."""
        raise NotImplementedError


class RandomBiddingStrategy(BiddingStrategy):
    """
    Strategia di riferimento che ignora i fondamentali e genera ordini casuali.

    Utile come baseline o per testare il flusso end-to-end quando non ci interessa
    ottimizzare rispetto ai segnali di mercato.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        cfg = dict(config or {})
        self.max_buy = float(cfg.get("max_buy_volume_MWh", 50.0))
        self.max_sell = abs(float(cfg.get("max_sell_volume_MWh", 50.0)))
        self.max_intraday = float(cfg.get("max_intraday_volume_MWh", min(self.max_buy, self.max_sell)))
        self.rng = np.random.default_rng(cfg.get("random_seed", 0))

    def supports_second_session(self) -> bool:
        return True

    def plan_day_ahead(self, inputs: StrategyInputs) -> float:
        return float(self.rng.uniform(-self.max_sell, self.max_buy))

    def plan_intraday(
        self,
        inputs: StrategyInputs,
        *,
        contracted_volume_MWh: float,
        session: int = 1,
    ) -> float:
        return float(self.rng.uniform(-self.max_intraday, self.max_intraday))


class SimpleRetailStrategy(BiddingStrategy):
    """
    Heuristica deterministica che copre la domanda prevista e limita l'attivita' MI.

    Tutte le scelte (frazioni di copertura, bias macro, limiti agli aggiustamenti) sono
    configurabili e mirano a riprodurre un comportamento "ragionevole" del retailer.
    """

    COVERAGE_PRESETS = {
        # Copertura robusta, poca dipendenza dall'intraday.
        "conservative": {
            "cover_fraction": 1.05,
            "mgp_lower_fraction": 0.9,
            "mgp_upper_fraction": 1.2,
            "mi_shift_fraction": 0.0,
            "mi_resell_fraction": 0.0,
        },
        # Profilo standard: copertura vicina a 1:1, intraday moderato.
        "balanced": {
            "cover_fraction": 1.0,
            "mgp_lower_fraction": 0.9,
            "mgp_upper_fraction": 1.1,
            "mi_shift_fraction": 0.05,
            "mi_resell_fraction": 0.0,
        },
        # Copre meno in MGP per lasciare spazio al MI.
        "aggressive": {
            "cover_fraction": 0.95,
            "mgp_lower_fraction": 0.85,
            "mgp_upper_fraction": 1.05,
            "mi_shift_fraction": 0.0,
            "mi_resell_fraction": 0.05,
        },
    }

    INTRADAY_PRESETS = {
        "cautious": {
            "mi_correction_fraction": 0.2,
            "mi2_fraction": 0.2,
            "max_intraday_volume_MWh": 200.0,
            "mi_price_ratio_limit": 2.0,
        },
        "moderate": {
            "mi_correction_fraction": 0.35,
            "mi2_fraction": 0.3,
            "max_intraday_volume_MWh": 400.0,
            "mi_price_ratio_limit": 2.5,
        },
        "aggressive": {
            "mi_correction_fraction": 0.6,
            "mi2_fraction": 0.5,
            "max_intraday_volume_MWh": 600.0,
            "mi_price_ratio_limit": 3.0,
        },
    }

    MACRO_PRESETS = {
        "off": {
            "use_macro_imbalance_forecast": False,
            "macro_bias_factor": 0.0,
            "force_macro_opposite_intraday": False,
        },
        "balanced": {
            "use_macro_imbalance_forecast": True,
            "macro_bias_factor": 0.1,
            "force_macro_opposite_intraday": True,
        },
        "opposite": {
            "use_macro_imbalance_forecast": True,
            "macro_bias_factor": 0.2,
            "force_macro_opposite_intraday": True,
        },
    }

    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        cfg = self._apply_presets(dict(config or {}))
        self.cover_fraction = float(cfg.get("cover_fraction", 0.98))
        self.lower_fraction = float(cfg.get("mgp_lower_fraction", 0.9))
        self.upper_fraction = float(cfg.get("mgp_upper_fraction", 1.05))
        self.cover_interval_enabled = bool(cfg.get("cover_interval", True))
        self.mi_correction_fraction = float(cfg.get("mi_correction_fraction", 0.6))
        self.mi2_factor = float(cfg.get("mi2_fraction", 0.5))
        self.price_ratio_limit = float(cfg.get("mi_price_ratio_limit", 1.2))
        self.max_intraday_volume = float(cfg.get("max_intraday_volume_MWh", 120.0))
        self.use_macro_forecast = bool(cfg.get("use_macro_imbalance_forecast", False))
        self.macro_bias_factor = float(cfg.get("macro_bias_factor", 0.05))
        self.mi_shift_fraction = float(cfg.get("mi_shift_fraction", 0.0))
        self.mi_resell_fraction = float(cfg.get("mi_resell_fraction", 0.0))
        self.mi_price_signal_threshold = float(cfg.get("mi_price_signal_threshold", 0.03))
        # Se attivo, le correzioni MI vengono orientate in direzione opposta al macro-sbilanciamento previsto.
        self.force_macro_opposite_intraday = bool(cfg.get("force_macro_opposite_intraday", False))

    @classmethod
    def _apply_presets(cls, cfg: Mapping[str, Any]) -> dict[str, Any]:
        """
        Riduce la necessità di parametri granulari applicando preset di copertura,
        intraday e macro. I parametri espliciti nel config sovrascrivono i preset.
        """
        payload = dict(cfg)
        coverage_key = str(payload.get("coverage_profile") or payload.get("profile") or "").lower()
        intraday_key = str(payload.get("intraday_mode") or "").lower()
        macro_key = str(payload.get("macro_mode") or "").lower()

        def merge(preset: Mapping[str, Any]) -> None:
            for k, v in preset.items():
                payload.setdefault(k, v)

        if coverage_key in cls.COVERAGE_PRESETS:
            merge(cls.COVERAGE_PRESETS[coverage_key])
        if intraday_key in cls.INTRADAY_PRESETS:
            merge(cls.INTRADAY_PRESETS[intraday_key])
        if macro_key in cls.MACRO_PRESETS:
            merge(cls.MACRO_PRESETS[macro_key])
        # Se l'utente fornisce una cover_fraction, generiamo lower/upper coerenti
        # solo se non sono stati impostati esplicitamente.
        if "cover_fraction" in payload:
            cover = float(payload["cover_fraction"])
            payload.setdefault("mgp_lower_fraction", max(0.0, cover * 0.9))
            payload.setdefault("mgp_upper_fraction", max(cover, cover * 1.1))
        return payload

    def supports_second_session(self) -> bool:
        return True

    # --------------------------- Day-ahead planning ---------------------------
    def plan_day_ahead(self, inputs: StrategyInputs) -> float:
        """
        Determina il volume MGP obiettivo:
        - copertura target = domanda * ``cover_fraction``;
        - clamp tra lower/upper fraction per evitare under/over coverage;
        - aggiustamento opzionale basato sul forecast macro.
        """
        demand = max(inputs.demand_forecast_MWh, 0.0)
        target = demand * self.cover_fraction
        if self.cover_interval_enabled:
            lower = demand * self.lower_fraction
            upper = demand * self.upper_fraction
        else:
            lower = upper = demand
        if target < lower:
            target = lower
        if target > upper:
            target = upper
        if self.use_macro_forecast and not np.isnan(inputs.macro_forecast_sign):
            target *= 1.0 + self.macro_bias_factor * inputs.macro_forecast_sign
        price_signal = self._mi_price_signal(inputs)
        macro_sign = inputs.macro_forecast_sign if self.use_macro_forecast else float("nan")
        msd_price = inputs.msd_price_estimate_EUR_MWh
        if self._should_shift_to_mi(price_signal, macro_sign, inputs.mi_price_EUR_MWh, inputs.mgp_price_EUR_MWh, msd_price):
            target -= demand * self.mi_shift_fraction
        elif self._should_resell_on_mi(
            price_signal, macro_sign, inputs.mi_price_EUR_MWh, inputs.mgp_price_EUR_MWh, msd_price
        ):
            target += demand * self.mi_resell_fraction
        target = min(max(target, lower), upper)
        return target

    # --------------------------- Intraday corrections ------------------------
    def plan_intraday(
        self,
        inputs: StrategyInputs,
        *,
        contracted_volume_MWh: float,
        session: int = 1,
    ) -> float:
        """
        Esegue un aggiustamento MI proporzionale al residuo domanda - contratti,
        attenuandolo con:
            * price cap: niente MI se il prezzo e' troppo alto rispetto al MSD stimato;
            * frazione di correzione diversa per sessione (MI2 piu' cauta);
            * limite assoluto sui MWh intraday;
            * bias macro opzionale.
        """
        residual = inputs.demand_forecast_MWh - contracted_volume_MWh
        if residual == 0.0:
            return 0.0

        reference = max(inputs.msd_price_estimate_EUR_MWh, 1.0)
        if inputs.mi_price_EUR_MWh > reference * self.price_ratio_limit:
            return 0.0

        base_fraction = self.mi_correction_fraction
        fraction = base_fraction if session == 1 else base_fraction * self.mi2_factor
        adjustment = residual * fraction
        limit = min(abs(residual) * fraction, self.max_intraday_volume)

        if self.use_macro_forecast and not np.isnan(inputs.macro_forecast_sign):
            adjustment *= 1.0 + self.macro_bias_factor * inputs.macro_forecast_sign

        if self.force_macro_opposite_intraday and not np.isnan(inputs.macro_forecast_sign) and inputs.macro_forecast_sign != 0:
            desired_sign = -np.sign(inputs.macro_forecast_sign)
            if not self._macro_opposite_price_ok(desired_sign, inputs):
                return 0.0
            adjustment = desired_sign * min(abs(adjustment), limit)

        return max(-limit, min(adjustment, limit))

    @staticmethod
    def _mi_price_signal(inputs: StrategyInputs) -> float:
        mgp_price = max(inputs.mgp_price_EUR_MWh, 1.0)
        if not np.isfinite(inputs.mi_price_EUR_MWh):
            return float("nan")
        return (inputs.mi_price_EUR_MWh - inputs.mgp_price_EUR_MWh) / mgp_price

    def _should_shift_to_mi(
        self,
        price_signal: float,
        macro_sign: float,
        mi_price: float,
        mgp_price: float,
        msd_price: float,
    ) -> bool:
        if self.mi_shift_fraction <= 0:
            return False
        if not np.isnan(macro_sign) and macro_sign > 0:
            return False
        cheaper_than_mgp = not np.isnan(price_signal) and price_signal <= -abs(self.mi_price_signal_threshold)
        cheaper_than_msd = np.isfinite(msd_price) and np.isfinite(mi_price) and mi_price < msd_price
        if cheaper_than_mgp:
            return True
        if mi_price > mgp_price and cheaper_than_msd:
            return True
        return False

    def _should_resell_on_mi(
        self,
        price_signal: float,
        macro_sign: float,
        mi_price: float,
        mgp_price: float,
        msd_price: float,
    ) -> bool:
        if self.mi_resell_fraction <= 0:
            return False
        if np.isnan(price_signal) or price_signal <= abs(self.mi_price_signal_threshold):
            return False
        if not np.isnan(macro_sign) and macro_sign < 0:
            return False
        if not np.isfinite(mi_price) or mi_price <= mgp_price:
            return False
        if np.isfinite(msd_price) and mi_price <= msd_price:
            return False
        return True

    def _macro_opposite_price_ok(self, desired_sign: float, inputs: StrategyInputs) -> bool:
        """
        Valuta se conviene correggere in MI rispetto a MGP/MSD, quando forziamo la
        direzione opposta al macro-sbilanciamento.
        """
        mi_price = inputs.mi_price_EUR_MWh
        mgp_price = inputs.mgp_price_EUR_MWh
        msd_price = inputs.msd_price_estimate_EUR_MWh
        if not np.isfinite(mi_price) or not np.isfinite(mgp_price):
            return False
        reference_buy_cap = min(mgp_price, msd_price if np.isfinite(msd_price) else mgp_price) * self.price_ratio_limit
        if desired_sign > 0:
            # vogliamo comprare: solo se il MI non è sensibilmente più caro dei riferimenti
            return mi_price <= reference_buy_cap
        reference_sell_floor = max(mgp_price, msd_price if np.isfinite(msd_price) else mgp_price) / max(self.price_ratio_limit, 1e-6)
        # vendiamo solo se il MI paga almeno quanto MGP/MSD (al netto del ratio)
        return mi_price >= reference_sell_floor


def apply_retailer_logic_presets(logic_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """
    Applica preset al blocco retailer_logic per ridurre i parametri da tarare.
    I valori espliciti nel config sovrascrivono sempre quelli del preset.
    """
    presets = {
        "conservative": {
            "mgp_min_fraction": 0.9,
            "mgp_max_fraction": 1.15,
            "gme_balance_tolerance_MWh": 0.5,
            "mi1_max_fraction": 0.2,
            "mi2_max_fraction": 0.2,
            "mi1_smoothing": 0.95,
            "mi2_smoothing": 0.9,
            "long_position_credit_factor": 0.3,
            "imbalance_coeff_default": 0.5,
        },
        "balanced": {
            "mgp_min_fraction": 0.85,
            "mgp_max_fraction": 1.25,
            "gme_balance_tolerance_MWh": 1.0,
            "mi1_max_fraction": 0.25,
            "mi2_max_fraction": 0.25,
            "mi1_smoothing": 0.9,
            "mi2_smoothing": 0.85,
            "long_position_credit_factor": 0.35,
            "imbalance_coeff_default": 0.55,
        },
        "aggressive": {
            "mgp_min_fraction": 0.8,
            "mgp_max_fraction": 1.35,
            "gme_balance_tolerance_MWh": 2.0,
            "mi1_max_fraction": 0.35,
            "mi2_max_fraction": 0.3,
            "mi1_smoothing": 0.85,
            "mi2_smoothing": 0.8,
            "long_position_credit_factor": 0.4,
            "imbalance_coeff_default": 0.6,
        },
    }
    terna_presets = {
        "conservative": {
            "same_direction_penalty_EUR_MWh": 10.0,
            "opposite_direction_bonus_EUR_MWh": 8.0,
            "neutral_direction_bonus_EUR_MWh": 0.0,
            "off_hours_penalty_factor": 0.4,
            "reference_volume_MWh": 30.0,
            "imbalance_pass_through_fraction": 0.4,
            "sale_price_floor_EUR_MWh": 40.0,
            "sale_price_cap_EUR_MWh": 150.0,
            "coverage_shortfall_markup_EUR_MWh": 10.0,
            "coverage_shortfall_tolerance_MWh": 0.2,
        },
        "balanced": {
            "same_direction_penalty_EUR_MWh": 15.0,
            "opposite_direction_bonus_EUR_MWh": 12.0,
            "neutral_direction_bonus_EUR_MWh": 0.0,
            "off_hours_penalty_factor": 0.4,
            "reference_volume_MWh": 20.0,
            "imbalance_pass_through_fraction": 0.5,
            "sale_price_floor_EUR_MWh": 30.0,
            "sale_price_cap_EUR_MWh": 200.0,
            "coverage_shortfall_markup_EUR_MWh": 20.0,
            "coverage_shortfall_tolerance_MWh": 0.5,
        },
        "aggressive": {
            "same_direction_penalty_EUR_MWh": 20.0,
            "opposite_direction_bonus_EUR_MWh": 15.0,
            "neutral_direction_bonus_EUR_MWh": 5.0,
            "off_hours_penalty_factor": 0.5,
            "reference_volume_MWh": 15.0,
            "imbalance_pass_through_fraction": 0.6,
            "sale_price_floor_EUR_MWh": 25.0,
            "sale_price_cap_EUR_MWh": 250.0,
            "coverage_shortfall_markup_EUR_MWh": 25.0,
            "coverage_shortfall_tolerance_MWh": 0.5,
        },
    }
    msd_presets = {
        "conservative": {
            "price_sensitivity": 0.3,
            "additional_penalty_EUR_MWh": 5.0,
            "long_position_credit_factor": 0.3,
        },
        "balanced": {
            "price_sensitivity": 0.4,
            "additional_penalty_EUR_MWh": 10.0,
            "long_position_credit_factor": 0.2,
        },
        "aggressive": {
            "price_sensitivity": 0.5,
            "additional_penalty_EUR_MWh": 12.0,
            "long_position_credit_factor": 0.2,
        },
    }
    payload = dict(logic_cfg or {})
    profile = str(payload.get("retailer_profile") or payload.get("profile") or "").lower()

    if profile in presets:
        for key, value in presets[profile].items():
            payload.setdefault(key, value)

    terna_profile = str(payload.get("terna_profile") or profile).lower()
    terna_cfg = dict(payload.get("terna_agent", {}))
    if terna_profile in terna_presets:
        for key, value in terna_presets[terna_profile].items():
            terna_cfg.setdefault(key, value)
    payload["terna_agent"] = terna_cfg

    msd_profile = str(payload.get("msd_profile") or profile).lower()
    msd_cfg = dict(payload.get("msd_settlement", {}))
    if msd_profile in msd_presets:
        for key, value in msd_presets[msd_profile].items():
            msd_cfg.setdefault(key, value)
    payload["msd_settlement"] = msd_cfg
    return payload

    def supports_second_session(self) -> bool:
        return True

    # --------------------------- Day-ahead planning ---------------------------
    def plan_day_ahead(self, inputs: StrategyInputs) -> float:
        """
        Determina il volume MGP obiettivo:
        - copertura target = domanda * ``cover_fraction``;
        - clamp tra lower/upper fraction per evitare under/over coverage;
        - aggiustamento opzionale basato sul forecast macro.
        """
        demand = max(inputs.demand_forecast_MWh, 0.0)
        target = demand * self.cover_fraction
        if self.cover_interval_enabled:
            lower = demand * self.lower_fraction
            upper = demand * self.upper_fraction
        else:
            lower = upper = demand
        if target < lower:
            target = lower
        if target > upper:
            target = upper
        if self.use_macro_forecast and not np.isnan(inputs.macro_forecast_sign):
            target *= 1.0 + self.macro_bias_factor * inputs.macro_forecast_sign
        price_signal = self._mi_price_signal(inputs)
        macro_sign = inputs.macro_forecast_sign if self.use_macro_forecast else float("nan")
        msd_price = inputs.msd_price_estimate_EUR_MWh
        if self._should_shift_to_mi(price_signal, macro_sign, inputs.mi_price_EUR_MWh, inputs.mgp_price_EUR_MWh, msd_price):
            target -= demand * self.mi_shift_fraction
        elif self._should_resell_on_mi(
            price_signal, macro_sign, inputs.mi_price_EUR_MWh, inputs.mgp_price_EUR_MWh, msd_price
        ):
            target += demand * self.mi_resell_fraction
        target = min(max(target, lower), upper)
        return target

    # --------------------------- Intraday corrections ------------------------
    def plan_intraday(
        self,
        inputs: StrategyInputs,
        *,
        contracted_volume_MWh: float,
        session: int = 1,
    ) -> float:
        """
        Esegue un aggiustamento MI proporzionale al residuo domanda - contratti,
        attenuandolo con:
            * price cap: niente MI se il prezzo e' troppo alto rispetto al MSD stimato;
            * frazione di correzione diversa per sessione (MI2 piu' cauta);
            * limite assoluto sui MWh intraday;
            * bias macro opzionale.
        """
        residual = inputs.demand_forecast_MWh - contracted_volume_MWh
        if residual == 0.0:
            return 0.0

        reference = max(inputs.msd_price_estimate_EUR_MWh, 1.0)
        if inputs.mi_price_EUR_MWh > reference * self.price_ratio_limit:
            return 0.0

        base_fraction = self.mi_correction_fraction
        fraction = base_fraction if session == 1 else base_fraction * self.mi2_factor
        adjustment = residual * fraction
        limit = min(abs(residual) * fraction, self.max_intraday_volume)

        if self.use_macro_forecast and not np.isnan(inputs.macro_forecast_sign):
            adjustment *= 1.0 + self.macro_bias_factor * inputs.macro_forecast_sign

        if self.force_macro_opposite_intraday and not np.isnan(inputs.macro_forecast_sign) and inputs.macro_forecast_sign != 0:
            desired_sign = -np.sign(inputs.macro_forecast_sign)
            if not self._macro_opposite_price_ok(desired_sign, inputs):
                return 0.0
            adjustment = desired_sign * min(abs(adjustment), limit)

        return max(-limit, min(adjustment, limit))

    @staticmethod
    def _mi_price_signal(inputs: StrategyInputs) -> float:
        mgp_price = max(inputs.mgp_price_EUR_MWh, 1.0)
        if not np.isfinite(inputs.mi_price_EUR_MWh):
            return float("nan")
        return (inputs.mi_price_EUR_MWh - inputs.mgp_price_EUR_MWh) / mgp_price

    def _should_shift_to_mi(
        self,
        price_signal: float,
        macro_sign: float,
        mi_price: float,
        mgp_price: float,
        msd_price: float,
    ) -> bool:
        if self.mi_shift_fraction <= 0:
            return False
        if not np.isnan(macro_sign) and macro_sign > 0:
            return False
        cheaper_than_mgp = not np.isnan(price_signal) and price_signal <= -abs(self.mi_price_signal_threshold)
        cheaper_than_msd = np.isfinite(msd_price) and np.isfinite(mi_price) and mi_price < msd_price
        if cheaper_than_mgp:
            return True
        if mi_price > mgp_price and cheaper_than_msd:
            return True
        return False

    def _should_resell_on_mi(
        self,
        price_signal: float,
        macro_sign: float,
        mi_price: float,
        mgp_price: float,
        msd_price: float,
    ) -> bool:
        if self.mi_resell_fraction <= 0:
            return False
        if np.isnan(price_signal) or price_signal <= abs(self.mi_price_signal_threshold):
            return False
        if not np.isnan(macro_sign) and macro_sign < 0:
            return False
        if not np.isfinite(mi_price) or mi_price <= mgp_price:
            return False
        if np.isfinite(msd_price) and mi_price <= msd_price:
            return False
        return True

    def _macro_opposite_price_ok(self, desired_sign: float, inputs: StrategyInputs) -> bool:
        """
        Valuta se conviene correggere in MI rispetto a MGP/MSD, quando forziamo la
        direzione opposta al macro-sbilanciamento.
        """
        mi_price = inputs.mi_price_EUR_MWh
        mgp_price = inputs.mgp_price_EUR_MWh
        msd_price = inputs.msd_price_estimate_EUR_MWh
        if not np.isfinite(mi_price) or not np.isfinite(mgp_price):
            return False
        reference_buy_cap = min(mgp_price, msd_price if np.isfinite(msd_price) else mgp_price) * self.price_ratio_limit
        if desired_sign > 0:
            # vogliamo comprare: solo se il MI non è sensibilmente più caro dei riferimenti
            return mi_price <= reference_buy_cap
        reference_sell_floor = max(mgp_price, msd_price if np.isfinite(msd_price) else mgp_price) / max(self.price_ratio_limit, 1e-6)
        # vendiamo solo se il MI paga almeno quanto MGP/MSD (al netto del ratio)
        return mi_price >= reference_sell_floor


def build_strategy(decision_cfg: Mapping[str, Any], sim_cfg: Mapping[str, Any]) -> BiddingStrategy:
    """
    Legge ``strategy_type`` dalla configurazione e restituisce
    l'istanza corrispondente. Il codice chiamante (standalone o world) riceve lo
    stesso oggetto, assicurando identico comportamento strategico in entrambi gli
    ambienti.
    """
    del sim_cfg  # parametro mantenuto per compatibilita' futura
    strategy_type = str(decision_cfg.get("strategy_type", "simple")).lower()
    if strategy_type == "random":
        return RandomBiddingStrategy(decision_cfg.get("random", {}))
    if strategy_type == "simple":
        return SimpleRetailStrategy(decision_cfg.get("simple", {}))
    raise ValueError(f"Unsupported strategy_type '{strategy_type}'.")
