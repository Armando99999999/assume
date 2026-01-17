# Guida simulatore retailer (standalone & world) — versione estesa

Questa guida approfondisce ogni componente del simulatore, i profili disponibili e come i parametri del `config.yaml` si combinano per determinare risultati e sensitività. Include esempi d’uso e note operative per evitare che alcune modifiche restino senza effetto.

## Panoramica dei due ambienti
- **Standalone** (`standalone/main_simulation.py`): legge dataset/forecast dal CSV, esegue clearing MGP/MI con il motore interno e produce `hourly_results.csv`, `aggregated_totals.*`, `orders.json`. È il riferimento per confronti rapidi e test di strategia.
- **World adapter** (`world_adapter/run_world.py`): costruisce mercati e unità in ASSUME World, usa il clearing interno del world, quindi ricostruisce gli output in formato identico allo standalone (`world_hourly_results.csv`, `world_totals.*`, `world_orders.json`). Usa `reference_df` solo per macro price/coeff/forecast, non per sovrascrivere i prezzi di clearing.
- **Strategia unica** (`decision.py`): la stessa istanza di strategia è condivisa; differenze fra standalone e world derivano solo dai prezzi di clearing, dalla tariffa cliente (flag `align_tariff_with_standalone`) e dai dataset di input.

## Novità e fix principali
- **MI price zero fix**: nello standalone `mi_price_EUR_MWh` viene riempito con `mi_market_price_EUR_MWh` quando non c’è stato trade; il prezzo di mercato MI è salvato in output.
- **Preset espliciti**: profili coverage/intraday/macro più chiari (`profile`, `intraday_mode`, `macro_mode`) con override granulari (`mi_shift_fraction`, `mi_correction_fraction`, `mi2_fraction`, `mi_price_signal_threshold`).
- **Profili Terna/MSD**: `retailer_profile`, `terna_profile`, `msd_profile` applicano preset; gli override nel config li sostituiscono puntualmente.
- **Tariffa cliente**: `align_tariff_with_standalone` (true) forza `tariff = purchase_price + margin` anche nel world per confronti “apples-to-apples”. Se false, il world usa il pass-through Terna (tariffa dipende da costo sbilanciamento).
- **Output arricchito**: in standalone vengono salvate `mi_market_price_EUR_MWh` e la `mi_price_EUR_MWh` clippata; world/standalone hanno struttura identica di hourly/totali/ordini.

## Come eseguire (passo-passo)
1) Verifica `simulation.input_csv_path` e le colonne mappate. Se usi ML, genera `forecast_inputs.csv` (o il file con suffisso del modello) prima di lanciare.
2) Standalone: `python -m assume.extensions.retailer_sim.standalone.main_simulation` (usa `config.yaml`). Trovi output in `outputs/`.
3) World: `python -m assume.extensions.retailer_sim.world_adapter.run_world` e leggi `outputs/retailer_world/`.
4) Confronti: affianca `hourly_results.csv` e `retailer_world/world_hourly_results.csv`. Ricorda: world usa prezzi di clearing interni; standalone usa il market interno e, se MI non scambia, riempie MI price con il prezzo di mercato.

## Profili e comportamento (strategia, Terna, MSD)
### Strategia `simple` (decision_making.simple)
- **Coverage profile (`profile`)**
  - `conservative`: copertura ~105% domanda, range MGP 0.9–1.2; intraday quasi nullo.
  - `balanced`: copertura ~100%, range 0.9–1.1; intraday moderato.
  - `aggressive`: copertura ~95%, range 0.85–1.05; lascia più spazio all’MI, possibile resell.
- **Intraday mode (`intraday_mode`)**
  - `cautious`: `mi_correction_fraction` ~0.2, `mi2_fraction` ~0.2, `max_intraday_volume` ridotto, `mi_price_ratio_limit` 2.0.
  - `moderate`: frazioni ~0.35/0.3, volumi medi, ratio 2.5.
  - `aggressive`: frazioni ~0.6/0.5, volumi alti, ratio 3.0.
- **Macro mode (`macro_mode`)**
  - `off`: niente macro bias, nessuna forzatura intraday opposta.
  - `balanced`: bias macro leggero (0.1), intraday forzato opposto se prezzi ok.
  - `opposite`: bias macro più forte (0.2), intraday opposto se conveniente.

**Dipendenza dagli input**
- `cover_fraction` moltiplica la domanda prevista; lower/upper frazioni tagliano il target per evitare under/over coverage.
- `mi_price_signal_threshold` confronta MI vs MGP: se MI è più basso di soglia negativa, può scattare lo shift; se è più alto di soglia positiva, può scattare il resell (se macro non negativo).
- `mi_shift_fraction` sposta quota di domanda sul MI solo se MI è conveniente e il macro-sign non è positivo.
- `mi_correction_fraction` e `mi2_fraction` applicano una quota del residuo; sono ulteriormente limitate da `mi_price_ratio_limit`, `max_intraday_volume` e, nel retailer, dalla `gme_balance_tolerance_MWh`.
- `force_macro_opposite_intraday` richiede prezzi MI non troppo cari rispetto a MGP/MSD (ratio limit).

### Profili Terna (`retailer_logic.terna_profile`)
- **conservative**: penalità basse, bonus piccoli, `reference_volume_MWh` più alto, `imbalance_pass_through_fraction` più bassa (0.4).
- **balanced**: penalità/bonus medi, reference volume ~20 MWh, pass-through 0.5.
- **aggressive**: penalità più alte ma bonus maggiori, reference volume più basso, pass-through 0.6.

**Dipendenza dagli input**
- Penalità/bonus si applicano in funzione del segno dello sbilanciamento e del segno macro (`macro_imbalance_MWh`), e sono scalati per il volume relativo a `reference_volume_MWh`.
- Se `align_tariff_with_standalone` è false, la tariffa cliente usa `compute_sale_tariff` e incorpora il costo di sbilanciamento e la frazione pass-through.

### Profili MSD (`retailer_logic.msd_profile`)
- **conservative**: `price_sensitivity` bassa (0.3), `additional_penalty` bassa, `long_position_credit_factor` 0.3.
- **balanced**: sensibilità 0.4, penalità 10, credito 0.2.
- **aggressive**: sensibilità 0.5, penalità 12, credito 0.2.

**Dipendenza dagli input**
- Il prezzo MSD stimato cresce con `imbalance_penalty_cost_per_MWh` (market) e con il delta MGP–macrozone price moltiplicato per `imbalance_coeff` (da dati o fallback). Più alto il coefficiente, più ripida la risposta dello sbilanciamento.

## Parametri del config.yaml (dettaglio completo)
### decision_making
- `strategy_type`: `simple` (heuristica) o `random` (baseline casuale).
- `random.*`: limiti MGP/MI e `random_seed`.
- `simple.profile`, `simple.intraday_mode`, `simple.macro_mode`: applicano i preset sopra descritti.
- `simple.cover_fraction`: override del preset; se presente, rigenera `mgp_lower_fraction`/`mgp_upper_fraction` se non impostati a mano.
- `simple.mgp_lower_fraction` / `mgp_upper_fraction`: clamp sul target MGP rispetto alla domanda.
- `simple.mi_shift_fraction`: quota di domanda spostata su MI se MI più conveniente e macro non positivo.
- `simple.mi_resell_fraction`: quota aggiuntiva se MI paga molto più di MGP e macro non negativo.
- `simple.mi_correction_fraction`: quota del residuo corretta in MI1.
- `simple.mi2_fraction`: fattore moltiplicativo per MI2 rispetto a MI1.
- `simple.mi_price_ratio_limit`: filtro prezzo MI vs MSD/MGP; se MI è troppo caro, la correzione è zero.
- `simple.mi_price_signal_threshold`: soglia % vs MGP per shift/resell.
- `simple.force_macro_opposite_intraday`: se true, le correzioni MI vanno in segno opposto al macro, ma solo se il prezzo MI è accettabile.

### market
- `transaction_cost_per_MWh` / `transaction_cost_intraday_per_MWh`: costi di transazione applicati a MGP/MI.
- `imbalance_penalty_cost_per_MWh`: base per stima MSD.
- `global_capacity`: capacità totale del mercato sintetico.
- `price_slope_scale`: scala la ripidità delle curve prezzo (MGP base 4/18, MI base 5/22).
- `price_slope_threshold`: quota di capacità oltre cui scatta la pendenza ripida.
- `mgp.capacity` / `mgp.volatility_EUR_MWh`: parametri MGP.
- `mi.capacity` / `mi.volatility_EUR_MWh`: parametri MI.

### simulation
- `start_datetime` / `end_datetime`: intervallo simulato.
- `time_step_minutes`: granularità (15 minuti).
- `real_market`: true = 7 sessioni MI reali; false = schema compatto a 2 sessioni.
- `timestamp_column`, `consumption_col`, `actual_consumption_col`, `mgp_price_col`, `mi_price_col`, `imbalance_col`, `macro_imbalance_col`, `macrozone_price_col`, `imbalance_coeff_col`, `mi2_price_col`: mapping colonne del CSV.
- `input_csv_path`: dataset di ingresso; `output_folder`: cartella output.

### retailer_logic
- `retailer_profile`: preset coverage/regolatori vendite surplus (vedi sopra).
- `terna_profile`: preset penali/bonus (vedi sopra).
- `msd_profile`: preset MSD (vedi sopra).
- `imbalance_coeff_default`: fallback per coefficiente sbilanciamento se la colonna manca.
- `max_imbalance_forecast_MWh`: clip del forecast sbilanciamento (riduce outlier e penalità).
- `macrozone_price_window_hours`: media mobile per macro price.
- `mgp_sell_fraction`: quota massima vendibile in MGP (di default 0).
- `surplus_sale_fraction`: quota di surplus liquidata nello slot.
- `surplus_sale_price`: `mgp`/`mi`/`msd`/`custom`/`equilibrium`; `surplus_sale_custom_price` se custom.
- `customer_margin_EUR_MWh`: margine unitario sulla tariffa.
- `gme_balance_tolerance_MWh`: soglia residuo prima di correggere in MI (più bassa = più correzioni).
- `align_tariff_with_standalone`: true per tariffa flat (purchase+margin) anche nel world.
- `terna_agent.*` (override): penalità/bonus direzionali, `reference_volume_MWh`, `imbalance_pass_through_fraction`.
- `msd_settlement.*`: sensibilità prezzo, penalità addizionale, fattore credito posizioni lunghe.

### macro_imbalance_forecast
- `enabled`: abilita uso del forecast macro.
- `source_column`: colonna del forecast macro (rinominata dal loader).
- `band_mgp` / `band_mi`: ampiezza banda per intervalli macro in MGP/MI.
- `leakage_guard.forbid_real_macro_imbalance`: blocca uso di dati reali come forecast.

### forecast_mapping
- Mapping colonne reali per metriche (`*_actual_col`) e naive lag.

### forecasting
- `mode`: `baseline` o `advanced`.
- `baseline_lag_steps`, `horizon_steps`: parametri naive.
- `ml_model`: modello ML in advanced (es. XGBoost); se presente, cerca il CSV con suffisso modello.
- `include_lstm`, `enable_model_optimization`: toggle di modelli e ottimizzazione.
- `output_csv`: percorso forecast generato.
- `advanced_settings.*`: target, range train/validation/test, `exogenous_columns`.

### information_flow
- Ritardi informativi: `consumption_delay_slots`, `imbalance_delay_slots`, `mi_price_delay_hours`, `mi_session_delays_hours.*`, `mgp_price_delay_hours`, `mgp_offer_day_offset`, `mgp_offer_deadline_hour`.

## Output attesi e lettura
- **Standalone**: `outputs/hourly_results.csv`, `aggregated_totals.*`, `orders.json`, più `mi_market_price_EUR_MWh` e `mi_price_EUR_MWh` clippato.
- **World**: `outputs/retailer_world/world_hourly_results.csv`, `world_totals.*`, `world_orders.json`; ricostruiti da `market_meta/dispatch` con prezzi di clearing interni.
- **Confronti**: differenze principali derivano da (a) prezzi MI/imbalance di clearing vs dataset, (b) tariffa cliente se `align_tariff_with_standalone` è false, (c) outlier tagliati da `max_imbalance_forecast_MWh` e dalla tolleranza GME che limita le correzioni MI.

## Suggerimenti operativi e diagnostica
- Se cambi parametri e non vedi effetti: verifica `mi_price_ratio_limit` e `gme_balance_tolerance_MWh` (possono annullare le correzioni), controlla che il preset non stia sovrascrivendo valori mancanti, e che il file config caricato sia quello usato dal run.
- Per aumentare attività MI: abbassa `gme_balance_tolerance_MWh`, alza `mi_price_ratio_limit`, aumenta `mi_correction_fraction`/`mi2_fraction`, e assicurati che i prezzi MI non siano filtrati perché troppo alti rispetto a MSD/MGP.
- Per ridurre penalità sbilanciamento: abbassa `imbalance_penalty_cost_per_MWh` (market), usa un profilo Terna più conservativo, riduci `imbalance_coeff_default` o fornisci coeff più bassi nel dataset, e riduci `max_imbalance_forecast_MWh` se hai outlier. 
