# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unita' retailer usata nell'adapter ASSUME (logica identica alla versione standalone).

Walkthrough in stile colloquio tecnico
--------------------------------------
1. ``RetailerUnit`` replica la stessa logica del retailer standalone cosi' da
   riutilizzare strategie e dataset senza duplicazioni. Tutte le serie temporali
   provengono dal ``Forecaster`` condiviso e quindi gli input sono perfettamente
   allineati tra gli scenari.
2. Il costruttore raggruppa step di configurazione affini (limiti di potenza,
   previsioni di domanda, prezzi, predittori di sbilanciamento, forecast macro,
   parametri MSD e metadati delle sessioni MI). Le funzioni di utilita' rendono
   esplicito ogni blocco senza alterare il comportamento.
3. I metodi pubblici espongono solamente lo stato della unita' (previsioni,
   envelope min/max, suggerimenti di liquidita', posizioni contrattate) cosi'
   che gli altri componenti ASSUME possano interrogarla. La logica delle offerte
   resta immutata rispetto al simulatore standalone.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from assume.common.base import SupportsMinMax
from assume.common.fast_pandas import FastSeries
from assume.common.forecasts import Forecaster

from ..decision import StrategyInputs, apply_retailer_logic_presets
from ..standalone.market import MSDSettlement
from ..terna_agent import TernaLikeBalancingAgent


class RetailerUnit(SupportsMinMax):
    """
    Rappresenta l'unita' lato domanda che reimpiega la strategia retailer
    standalone dentro il mondo ASSUME: mantiene intatte serie previsionali e
    parametri della controparte originale, cosi' che l'interfaccia
    ``SupportsMinMax`` possa esporre min/max, costi marginali e posizioni
    contrattate senza alterare la logica di bidding riutilizzata.
    """

    def __init__(
        self,
        id: str,
        unit_operator: str,
        technology: str,
        bidding_strategies: Dict[str, str],
        max_power: float,
        min_power: float,
        forecaster: Forecaster,
        node: str = "node0",
        location: tuple[float, float] = (0.0, 0.0),
        forecast_column: str = "",
        price_columns: Optional[Dict[str, str]] = None,
        macro_price_column: str = "macro_price",
        imbalance_coeff_column: str = "imbalance_coeff",
        imbalance_forecast_column: Optional[str] = None,
        macro_imbalance_column: Optional[str] = None,
        actual_consumption_column: Optional[str] = None,
        retailer_logic: Optional[Dict[str, Any]] = None,
        imbalance_penalty_cost_per_MWh: float = 0.0,
        intraday_sessions: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Inizializza la unita' retailer mantenendo una corrispondenza 1:1 con il
        simulatore standalone: allinea setup e serie previsionali per evitare
        divergenze tra ambienti e suddivide il costruttore in blocchi tematici
        che caricano dati dal ``Forecaster`` e configurano gli stessi parametri
        operativi dell'originale.

        Walkthrough operativo:
            1. Invochiamo ``SupportsMinMax`` per popolare i metadati base.
            2. Definiamo i limiti di potenza equivalenti alla domanda massima/minima.
            3. Carichiamo serie di domanda, consumi reali e metadati di slot dal forecaster.
            4. Agganciamo tutte le serie di prezzo (MGP, MI, macro).
            5. Importiamo predittori di sbilanciamento e opzioni di forecast macro.
            6. Configuriamo insiemi MSD e agente Terna per replicare penali e aggiustamenti.
            7. Prepariamo metadati delle sessioni intraday (eligibilita' e liquidita').
            8. Azzeriamo il libro mastro delle posizioni contrattate.

        La logica di ottimizzazione resta identica: rendiamo soltanto esplicite le
        dipendenze dati gestite dalla classe.
        """
        self.forecast_column = forecast_column or id
        super().__init__(
            id=id,
            unit_operator=unit_operator,
            technology=technology,
            bidding_strategies=bidding_strategies,
            forecaster=forecaster,
            node=node,
            location=location,
            **kwargs,
        )

        self.logic_config = apply_retailer_logic_presets(retailer_logic or {})
        self.default_imbalance_fraction = float(self.logic_config.get("default_imbalance_fraction", 0.05))

        self._setup_power_envelope(min_power=min_power, max_power=max_power)
        self._initialize_consumption_forecasts(actual_consumption_column=actual_consumption_column)
        self._configure_price_signals(price_columns=price_columns, macro_price_column=macro_price_column)
        self._configure_imbalance_predictors(
            imbalance_coeff_column=imbalance_coeff_column,
            imbalance_forecast_column=imbalance_forecast_column,
        )
        self._configure_macro_forecast(
            macro_imbalance_column=macro_imbalance_column,
            imbalance_forecast_column=imbalance_forecast_column,
        )
        self._configure_msd_and_terna(imbalance_penalty_cost_per_MWh=imbalance_penalty_cost_per_MWh)
        self._configure_session_metadata(intraday_sessions=intraday_sessions)
        self._initialize_contract_ledger()

    # ------------------------------------------------------------------ #
    # Interfaccia pubblica interrogata dal mondo di simulazione ASSUME.
    # ------------------------------------------------------------------ #

    def execute_current_dispatch(self, start: datetime, end: datetime) -> np.ndarray:
        """
        Restituisce il vettore di domanda prevista compreso tra ``start`` ed ``end``,
        permettendo al mondo ASSUME di conoscere il profilo di consumo atteso; i dati
        provengono dalla stessa serie ``self.volume`` usata nello standalone, cosi' le
        due ambientazioni condividono esattamente le curve.
        """
        return self.volume.loc[start:end]

    def calculate_min_max_power(
        self,
        start: datetime,
        end: datetime,
        product_type: str = "energy",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcola envelope minimi e massimi identici al residuo della domanda, poiche'
        il retailer puo' modulare solo la quota non ancora coperta: sottrae dalle
        previsioni l'energia gia' allocata in ``self.outputs`` e usa l'intervallo
        risultante come margine di regolazione simmetrico.
        """
        end_excl = end - self.index.freq
        residual = self.volume.loc[start:end_excl] - self.outputs[product_type].loc[start:end_excl]
        return residual, residual

    def calculate_marginal_cost(self, start: datetime, power: float, market_id: Optional[str] = None) -> float:
        """
        Fornisce il prezzo di riferimento usato per valorizzare un'offerta, cercando la
        serie configurata per ``market_id`` (default MGP) e leggendo il valore al
        timestamp richiesto; se manca la serie dedicata si ripiega su ``self.price``.
        """
        series = self.price_series.get((market_id or "MGP").upper())
        if series is None:
            return float(self.price.at[start])
        return float(series.at[start])

    def get_demand_forecast(self, timestamp: datetime) -> float:
        """
        Restituisce la previsione di domanda per un singolo timestamp prelevandola da
        ``self.volume`` (la stessa serie usata dallo standalone), cosi' le strategie
        ottengono il valore puntuale senza dover gestire l'intera serie.
        """
        return float(self.volume.at[timestamp])

    def is_session_slot_eligible(self, timestamp: datetime, session_name: str) -> bool:
        """
        Indica se una sessione MI puo' trattare lo slot richiesto consultando la serie
        di eligibilita'; in assenza del dato (serie mancante o valore non trovato) la
        funzione ritorna True per rimanere compatibile con simulazioni prive di metadati.
        """
        series = self.session_flag_series.get(session_name.upper())
        if series is None:
            return True
        try:
            value = float(series.at[timestamp])
        except KeyError:
            # L'assenza di dato viene interpretata come slot ammesso per non bloccare trading.
            return True
        return value >= 0.5

    def get_session_liquidity(self, timestamp: datetime, session_name: str) -> float:
        """
        Restituisce un suggerimento di liquidita' per la sessione indicata recuperando
        la serie dedicata, clampando i valori nell'intervallo [0, 1] e ripiegando su 1.0
        quando i dati sono mancanti o ``NaN`` cosi' da non irrigidire eccessivamente la
        strategia.
        """
        series = self.session_liquidity_series.get(session_name.upper())
        if series is None:
            return 1.0
        try:
            value = float(series.at[timestamp])
        except KeyError:
            value = 1.0
        if np.isnan(value):
            return 1.0
        return float(min(max(value, 0.0), 1.0))

    def regulate_day_ahead_volume(self, desired_volume_MWh: float, demand_forecast_MWh: float) -> float:
        """
        Applica al volume richiesto le salvaguardie del MGP standalone, imponendo
        frazioni minime/massime di copertura rispetto alla domanda, impedendo vendite
        indesiderate e garantendo un ordine positivo quando la domanda attesa e' sopra
        zero; se la domanda e' nulla ci si limita a evitare volumi negativi.
        """
        if demand_forecast_MWh <= 0.0:
            # Con domanda nulla ci limitiamo a impedire volumi negativi.
            return max(desired_volume_MWh, 0.0)

        min_frac = float(self.logic_config.get("mgp_min_fraction", 0.82))
        max_frac = float(self.logic_config.get("mgp_max_fraction", 0.95))
        sell_frac = max(float(self.logic_config.get("mgp_sell_fraction", 0.0)), 0.0)

        lower_buy = demand_forecast_MWh * min_frac
        upper_buy = demand_forecast_MWh * max_frac
        sell_cap = demand_forecast_MWh * sell_frac

        regulated = desired_volume_MWh
        if regulated >= 0:
            regulated = max(regulated, lower_buy)
        regulated = min(regulated, upper_buy)
        regulated = max(regulated, -sell_cap)
        regulated = max(regulated, 0.0)
        if regulated <= 0.0:
            regulated = lower_buy
        return regulated

    def regulate_intraday_volume(
        self,
        *,
        session_index: int,
        desired_volume_MWh: float,
        demand_forecast_MWh: float,
        contracted_before_MWh: float,
        liquidity_hint: float = 1.0,
    ) -> float:
        """
        Replica le salvaguardie intraday dello standalone applicando tolleranze per
        evitare micro-ordini, frazioni massime dipendenti dalla sessione, controlli di
        direzione (niente acquisti se gia' lunghi e viceversa) e fattori di smoothing
        combinati con gli hint di liquidita' per prevenire reazioni eccessive.
        """
        tolerance = float(self.logic_config.get("gme_balance_tolerance_MWh", 0.1))
        residual = demand_forecast_MWh - contracted_before_MWh
        if abs(residual) <= tolerance:
            return 0.0

        # Step 1: determiniamo parametri di sessione (frazione massima e smoothing).
        liquidity_hint = float(min(max(liquidity_hint, 0.0), 1.0))
        if session_index <= 1:
            max_fraction = float(self.logic_config.get("mi1_max_fraction", 0.6))
            smoothing = float(self.logic_config.get("mi1_smoothing", 0.95))
        else:
            max_fraction = float(self.logic_config.get("mi2_max_fraction", 0.3))
            smoothing = float(self.logic_config.get("mi2_smoothing", 0.85))

        # Step 2: permettiamo al massimo una frazione della deviazione residua.
        allowed = abs(residual) * max_fraction * liquidity_hint
        if allowed <= tolerance:
            return 0.0

        regulated = max(-allowed, min(desired_volume_MWh, allowed))
        if residual > 0 and regulated < 0:
            regulated = 0.0
        elif residual < 0 and regulated > 0:
            regulated = 0.0

        # Step 3: evitiamo overshoot correggendo eventuali sorpassi della domanda target.
        final_position = contracted_before_MWh + regulated
        if residual > 0 and final_position > demand_forecast_MWh:
            regulated = max(0.0, demand_forecast_MWh - contracted_before_MWh)
        elif residual < 0 and final_position < demand_forecast_MWh:
            regulated = min(0.0, demand_forecast_MWh - contracted_before_MWh)

        # Step 4: applichiamo smoothing e liquidity hint per imitare la cautela dello standalone.
        smoothing = float(min(max(smoothing, 0.0), 1.0))
        regulated *= smoothing * liquidity_hint
        return regulated

    def record_contracted_volume(self, timestamp: datetime, market_label: str, volume: float) -> None:
        """
        Aggiorna il libro mastro delle posizioni contrattate per un determinato
        timestamp/mercato normalizzando l'etichetta e sommando il volume, cosi' che le
        stime successive (ad es. MSD) conoscano sempre la posizione netta cumulata.
        """
        plan = self._contract_plan.setdefault(timestamp, {})
        label = market_label.upper()
        plan[label] = plan.get(label, 0.0) + volume

    def get_contracted_volume(self, timestamp: datetime) -> float:
        """
        Restituisce la posizione contrattata complessiva per lo slot richiesto sommando
        tutti i volumi registrati nei diversi mercati, informazione indispensabile per
        valutare il residuo rispetto alla domanda.
        """
        plan = self._contract_plan.get(timestamp, {})
        return float(sum(plan.values()))

    def build_strategy_inputs(
        self,
        timestamp: datetime,
        *,
        mi_price_market: str = "MI",
        phase: str = "mgp",
    ) -> StrategyInputs:
        """
        Assembla il dataclass ``StrategyInputs`` richiesto dalla logica standalone,
        raccogliendo tutte le serie (domanda, prezzi MGP/MI/macro, coefficienti,
        forecast di sbilanciamento) e calcolando la stima del prezzo MSD tramite i
        metodi interni prima di restituire l'istanza popolata.
        """
        demand = float(self.volume.at[timestamp])
        mgp_price = float(self.price_series.get("MGP", self.price).at[timestamp])
        mi_price = float(self.price_series.get(mi_price_market.upper(), self.price).at[timestamp])
        macro_price = float(self.macro_price.at[timestamp])
        imbalance_coeff = float(self.imbalance_coeff.at[timestamp])
        imbalance_forecast = float(self.imbalance_forecast.at[timestamp])
        macro_imbalance = float(self.macro_imbalance.at[timestamp])
        macro_sign = self._macro_forecast_sign(macro_imbalance)

        band = self.macro_band_mgp if phase.lower() == "mgp" else self.macro_band_mi
        if not self.macro_forecast_enabled:
            band = 0.0
        macro_low = macro_imbalance - band
        macro_high = macro_imbalance + band

        normalized_forecast = self._normalize_imbalance_forecast(imbalance_forecast, fallback=demand)
        # Riutilizziamo la stessa pipeline standalone per stimare il prezzo MSD che alimenta la strategia.
        msd_price = self._estimate_msd_price(
            timestamp=timestamp,
            demand=demand,
            mgp_price=mgp_price,
            macro_price=macro_price,
            imbalance_coeff=imbalance_coeff,
            imbalance_input=normalized_forecast,
            macro_imbalance=macro_imbalance,
        )

        return StrategyInputs(
            timestamp=pd.Timestamp(timestamp),
            demand_forecast_MWh=demand,
            mgp_price_EUR_MWh=mgp_price,
            mi_price_EUR_MWh=mi_price,
            macro_price_EUR_MWh=macro_price,
            imbalance_forecast_MWh=imbalance_forecast,
            macro_imbalance_MWh=macro_imbalance,
            imbalance_coeff=imbalance_coeff,
            msd_price_estimate_EUR_MWh=msd_price,
            macro_forecast_mean_MWh=macro_imbalance,
            macro_forecast_low_MWh=macro_low,
            macro_forecast_high_MWh=macro_high,
            macro_forecast_sign=macro_sign,
        )

    # ------------------------------------------------------------------ #
    # Helper interni che mantengono dichiarativo il costruttore.
    # ------------------------------------------------------------------ #

    def _setup_power_envelope(self, *, min_power: float, max_power: float) -> None:
        """
        Configura i limiti di potenza richiesti da ``SupportsMinMax`` usando il massimo
        assoluto tra ``min_power`` e ``max_power`` come rampa simmetrica, cosi' da
        rappresentare correttamente una unita' esclusivamente di consumo.
        """
        limit = max(abs(min_power), abs(max_power))
        self.max_power = max_power
        self.min_power = min_power
        self.ramp_down = limit
        self.ramp_up = limit

    def _initialize_consumption_forecasts(self, *, actual_consumption_column: Optional[str]) -> None:
        """
        Carica dal ``Forecaster`` le serie di domanda prevista, eventuali consumi reali
        e i metadati temporali: ``self.volume`` punta sempre al forecast scelto mentre
        ``self.actual_consumption`` riusa tale serie quando non e' disponibile una
        colonna dedicata; viene inoltre caricata ``slot_hour_float``.
        """
        self.volume = self._load_series(self.forecast_column)
        self.actual_consumption = (
            self._load_series(actual_consumption_column) if actual_consumption_column else self.volume
        )
        self.slot_hour_float = self._load_series("slot_hour_float")

    def _configure_price_signals(
        self,
        *,
        price_columns: Optional[Dict[str, str]],
        macro_price_column: str,
    ) -> None:
        """
        Prepara le serie di prezzo interrogate dalla strategia salvandole in un
        dizionario per mercato (in maiuscolo) e garantendo un fallback costante di
        3000 EUR/MWh per il MGP, cosi' da mantenere coerenza di costo marginale tra
        standalone e adapter.
        """
        self.price_series: Dict[str, FastSeries] = {}
        for market_id, column in (price_columns or {}).items():
            self.price_series[market_id.upper()] = self._load_series(column)
        self.price = self.price_series.get("MGP", FastSeries(index=self.index, value=3000.0))
        self.macro_price = self._load_series(macro_price_column)

    def _configure_imbalance_predictors(
        self,
        *,
        imbalance_coeff_column: str,
        imbalance_forecast_column: Optional[str],
    ) -> None:
        """
        Carica coefficienti e forecast di sbilanciamento necessari alla stima MSD,
        riutilizzando il forecast di domanda quando non viene indicata una colonna
        dedicata, in modo che i calcoli coincidano con quelli della strategia standalone.
        """
        self.imbalance_coeff = self._load_series(imbalance_coeff_column)
        self.imbalance_forecast = (
            self._load_series(imbalance_forecast_column) if imbalance_forecast_column else self.volume
        )

    def _configure_macro_forecast(
        self,
        *,
        macro_imbalance_column: Optional[str],
        imbalance_forecast_column: Optional[str],
    ) -> None:
        """
        Inizializza la serie di sbilanciamento macro e le relative bande leggendo la
        configurazione, scegliendo la colonna piu' appropriata (dedicata o fallback sul
        forecast di domanda) e, quando il macro forecast e' abilitato, verificando che il
        nome della colonna non suggerisca la presenza di dati reali.
        """
        macro_cfg = dict(self.logic_config.get("macro_imbalance_forecast", {}))
        self.macro_forecast_enabled = bool(macro_cfg.get("enabled"))
        macro_column = (
            macro_cfg.get("source_column")
            or macro_imbalance_column
            or imbalance_forecast_column
            or self.forecast_column
        )
        if self.macro_forecast_enabled:
            self._guard_forecast_column(macro_column)
        self.macro_imbalance = self._load_series(macro_column)
        self.macro_band_mgp = float(macro_cfg.get("band_mgp", 0.0))
        self.macro_band_mi = float(macro_cfg.get("band_mi", self.macro_band_mgp))

    def _configure_msd_and_terna(self, *, imbalance_penalty_cost_per_MWh: float) -> None:
        """
        Configura il modello di settlement MSD e l'eventuale agente Terna-like leggendo
        sensibilita', penali aggiuntive e fattore di credito dalla configurazione,
        inizializzando ``MSDSettlement`` con tali valori e istanziando
        ``TernaLikeBalancingAgent`` quando previsto.
        """
        msd_cfg = dict(self.logic_config.get("msd_settlement", {}))
        sensitivity = float(msd_cfg.get("price_sensitivity", self.logic_config.get("msd_price_sensitivity", 0.45)))
        additional_penalty = float(imbalance_penalty_cost_per_MWh) + float(
            msd_cfg.get("additional_penalty_EUR_MWh", self.logic_config.get("msd_additional_penalty_EUR_MWh", 0.0))
        )
        credit_factor = float(
            msd_cfg.get("long_position_credit_factor", self.logic_config.get("long_position_credit_factor", 1.0))
        )
        self.msd_settlement = MSDSettlement(
            price_sensitivity=sensitivity,
            additional_penalty_EUR_MWh=additional_penalty,
            long_position_credit_factor=credit_factor,
        )
        self.terna_agent = TernaLikeBalancingAgent(self.logic_config.get("terna_agent", {}))

    def _configure_session_metadata(self, *, intraday_sessions: Optional[List[Dict[str, Any]]]) -> None:
        """
        Carica i metadati di eligibilita' e liquidita' delle sessioni MI clonando la
        lista fornita, normalizzando i nomi in maiuscolo e creando per ciascuna due
        serie (flag e hint) con fallback permissivi quando i dati non sono disponibili.
        """
        self.session_definitions = list(intraday_sessions or [])
        self.session_order = {
            session.get("name", "").upper(): idx + 1 for idx, session in enumerate(self.session_definitions)
        }
        self.session_flag_series: Dict[str, FastSeries] = {}
        self.session_liquidity_series: Dict[str, FastSeries] = {}
        for session in self.session_definitions:
            name = session.get("name", "")
            if not name:
                continue
            upper = name.upper()
            self.session_flag_series[upper] = self._load_series_or_default(f"is_{name}_eligible", default_value=1.0)
            liquidity = float(session.get("liquidity", 1.0))
            self.session_liquidity_series[upper] = self._load_series_or_default(
                f"{name}_liquidity_hint",
                default_value=liquidity,
            )

    def _initialize_contract_ledger(self) -> None:
        """
        Reinizializza la struttura che traccia i volumi contrattati per timestamp,
        creando un dizionario vuoto ``_contract_plan`` cosi' che ogni simulazione parta
        da posizioni nulle.
        """
        self._contract_plan: Dict[datetime, Dict[str, float]] = {}

    def _estimate_msd_price(
        self,
        *,
        timestamp: datetime,
        demand: float,
        mgp_price: float,
        macro_price: float,
        imbalance_coeff: float,
        imbalance_input: float,
        macro_imbalance: float,
    ) -> float:
        """
        Calcola il prezzo MSD previsto invocando ``MSDSettlement.estimate_price`` con i
        parametri correnti e, se e' presente l'agente Terna-like, applica l'eventuale
        aggiustamento basato sul residuo e sul segno macro prima di restituire il valore.
        """
        msd_price = self.msd_settlement.estimate_price(
            mgp_price=mgp_price,
            macrozone_price=macro_price,
            imbalance_coeff=imbalance_coeff,
            imbalance_input_MWh=imbalance_input,
        )
        if self.terna_agent is not None:
            contracted = self.get_contracted_volume(timestamp)
            residual = demand - contracted
            adjustment = self.terna_agent.evaluate(
                timestamp=pd.Timestamp(timestamp),
                residual_MWh=residual,
                macro_imbalance_MWh=macro_imbalance,
                base_price_EUR_MWh=msd_price,
                mgp_price_EUR_MWh=mgp_price,
            )
            msd_price = adjustment.adjusted_price_EUR_MWh
        return float(msd_price)

    def _normalize_imbalance_forecast(self, value: float, *, fallback: float) -> float:
        """
        Normalizza il forecast di sbilanciamento sostituendo i ``NaN`` con
        ``fallback * default_imbalance_fraction`` per mantenere stabili le stime MSD
        anche quando mancano dati previsionali.
        """
        if np.isnan(value):
            return fallback * self.default_imbalance_fraction
        return float(value)

    def _load_series(self, column: str) -> FastSeries:
        """
        Recupera dal ``Forecaster`` la colonna indicata e la restituisce come
        ``FastSeries``, convertendo automaticamente array o Serie pandas per uniformare
        la gestione degli indici all'interno della classe.
        """
        series = self.forecaster[column]
        if isinstance(series, FastSeries):
            return series
        return FastSeries(index=self.index, value=series)

    def _load_series_or_default(self, column: str, *, default_value: float) -> FastSeries:
        """
        Carica la colonna richiesta dal forecaster e, se non esiste, restituisce una
        ``FastSeries`` costante con ``default_value`` costruita sul medesimo indice,
        cosi' la strategia dispone comunque di una serie valida.
        """
        try:
            return self._load_series(column)
        except KeyError:
            return FastSeries(index=self.index, value=default_value)

    def _guard_forecast_column(self, column: Optional[str]) -> None:
        """
        Verifica che il nome della colonna scelta per il forecast macro non suggerisca la
        presenza di dati reali (parole chiave come ``real`` o ``actual``) e in tal caso
        solleva un ``ValueError`` per evitare leakage informativo.
        """
        if not column:
            return
        lowered = str(column).lower()
        if "real" in lowered or "actual" in lowered:
            raise ValueError(f"La colonna di forecast macro '{column}' sembra contenere dati reali.")

    @staticmethod
    def _macro_forecast_sign(value: float) -> float:
        """Converte lo sbilanciamento macro previsto nel segno -1/0/1 usato dalla strategia."""
        if value > 0:
            return 1.0
        if value < 0:
            return -1.0
        return 0.0
