# ASSUME Retailer Simulation: guida tecnica discorsiva
_Auto-generated from repository sources_

## Introduzione e scopo della guida

Questa guida descrive in modo discorsivo e tracciabile il funzionamento del simulatore del retailer contenuto nel pacchetto `assume/extensions/retailer_sim`. L'obiettivo e' spiegare l'architettura, il flusso dati, la logica di decisione e la metodologia di settlement degli sbilanciamenti, senza fare assunzioni non supportate dal codice. Ogni affermazione e' ricondotta a file e simboli reali presenti nel repository. La guida copre sia la versione *Standalone* sia il *World adapter*, evidenziando come le stesse regole di bidding vengano riutilizzate in due contesti di mercato diversi.

## Modello concettuale e metodologia

### Che cosa viene simulato

Il simulatore rappresenta un retailer che deve coprire una domanda di energia per slot temporale. Il retailer acquista volumi nel mercato day-ahead (MGP) e, se necessario, corregge la posizione sui mercati intraday (MI1..MI7 o MI1..MI2). A valle, eventuali sbilanciamenti (differenza fra consumo reale e volumi contrattati) vengono regolati al prezzo MSD stimato, con eventuali penalita' o incentivi legati al segno del macro-sbilanciamento. Questo flusso e' implementato in modo esplicito in `assume/extensions/retailer_sim/standalone/retailer.py` e `assume/extensions/retailer_sim/standalone/market.py`.

### Significato di "forecast-driven decision"

La strategia usa una struttura dati comune, `StrategyInputs`, che incapsula previsione di domanda, prezzi di mercato, macro prezzi, sbilanciamento previsto e una stima del prezzo MSD. Le regole di decisione non ottimizzano via solver: applicano euristiche deterministiche (o randomiche) basate su queste variabili. La decisione di spostare volumi dal MGP al MI, o di correggere la posizione intraday, e' guidata da soglie prezzo e dal segno del macro-sbilanciamento (`assume/extensions/retailer_sim/decision.py`).

### Standalone vs World adapter

Lo stesso nucleo decisionale viene riusato in due modalita' di esecuzione:
- **Standalone**: mercati e clearing sono simulati localmente con classi dedicate (`DayAheadMarket`, `IntraDayMarket`) e la contabilita' e' calcolata nello stesso processo.
- **World adapter**: la strategia viene adattata alle API ASSUME tramite `RetailerMGPStrategy` e `RetailerMIStrategy`. Le offerte sono poi eseguite nel *World* di ASSUME, che produce esportazioni standard ricostruite in seguito.

La motivazione di questa duplicita' e' mantenere una logica di bidding identica pur cambiando il contesto di mercato: nel World si beneficia di un clearing multi-unita' e di esportazioni standard, mentre nel Standalone si ottiene un controllo locale completo e rapido.

## File ausiliari e pipeline dati

### Aggregation e POD/PUN

`aggregation.py` e `pod_dataset.py` costituiscono una pipeline di preparazione dati:
- `aggregation.py` carica le curve POD orarie, calcola il portafoglio mensile, unisce PUN (mensile o orario) e festivita', e produce `pods_hourly.csv` con segnali F1/F2/F3 e prezzo PUN.
- `pod_dataset.py` ricampiona `ITALY_NORD_DATASET_ENRICHED.csv` a orario con regole diverse per energia/prezzi/flag, unisce `pods_hourly.csv` e sostituisce colonne chiave (`cluster_total_load_MW`, `MGP_PRICE_NORD`) con i valori POD/PUN, producendo `ITALY_NORD_HOURLY_WITH_PODS_PUN.csv`.

Questi script non sono necessari per l'esecuzione del simulatore, ma sono fondamentali se si vuole replicare il dataset usato negli esempi.

### Forecasting e gestione ritardi

`generate_forecasts.py` addestra modelli, seleziona il migliore per RMSE/MAE e produce `forecast_inputs.csv` compatibile con il simulatore. L'uso dei ritardi informativi e' implementato in `_attach_actual_signals`, che shift-a serie reali di carico, sbilanciamento e prezzi in base a `information_flow`.

### Audit config e guard anti-leakage

`config_audit.py` segnala chiavi YAML inattese. `leakage_guard.py` fornisce una policy centralizzata per impedire l'accesso a colonne reali in fasi previsionali e per rispettare vincoli di finestra MI, ma non risulta chiamato dai loop di simulazione principali.

## File dati e asset

### File di configurazione e dataset di base

I file dati sono gestiti come input esterni, tipicamente CSV:
- `assume/extensions/retailer_sim/config.yaml`: definisce mapping colonne, parametri di mercato e logica retailer.
- `assume/extensions/retailer_sim/ITALY_NORD_DATASET_ENRICHED.csv`: dataset arricchito su cui lavora `pod_dataset.py`.
- `assume/extensions/retailer_sim/POD_CURVA_EFFETTIVA_ORARIA_MTA2.csv`: curve POD orarie usate da `aggregation.py`.
- `assume/extensions/retailer_sim/PUN_ORARIO_FLG_012019_122025.csv`: input PUN mensile/orario usato da `aggregation.py`.

I percorsi sono codificati negli script di pipeline dati, non nella logica di simulazione.

### Forecast inputs generati

`generate_forecasts.py` scrive un file `forecast_inputs.csv` nel path definito da `forecasting.output_csv`. In modalita' `advanced` il file puo' includere un suffisso con il nome del modello scelto. Le colonne prodotte includono, quando disponibili: `load_cluster_forecast`, `sbil_forecasted`, `mi1..miN`, `mgp`, `msd_forecast_EUR_MWh` e colonne di disponibilita' costruite da `information_flow` (ad esempio prezzi realizzati ritardati).

### Artifact di training

Molti modelli di forecasting espongono metodi `save_metrics`/`save_model` che scrivono file di diagnostica (es. `model_details_*.txt`) o scaler (`*.pkl`). L'effettiva creazione di questi file dipende dal codice chiamante, che non e' sempre presente negli script principali.

## Dettagli dei modelli di forecasting

Questa sezione descrive i modelli presenti nel repository, sulla base del codice disponibile. Non implica che tutti siano usati nella pipeline principale.

### NAIVE_model.py

`NAIVE_Predictor` implementa forecast semplici: persistence (ultimo valore), media e stagionale. Il metodo `forecast` costruisce indici temporali coerenti con la frequenza inferita. Il modello espone anche plotting e salvataggio metriche (`model_details_NAIVE.txt`).

### MLFORECASTCopia.py

`MLFORECAST_Predictor` usa la libreria `mlforecast` con lag e rolling mean. I modelli supportati includono LightGBM, XGBoost, RandomForest e LinearRegression. Le opzioni di ottimizzazione usano `GridSearchCV` su un set di iperparametri (per LGBM e XGBoost). La funzione `backtest` fa rolling update del modello e riallinea le previsioni alle date originali del test.

### NEURALFORECASTCopia.py

`NEURALFORECAST_Predictor` usa `NeuralForecast` con un modello LSTM. Supporta un ramo ottimizzato con tuning di iperparametri (input size, layer, learning rate, batch size) e un ramo non ottimizzato con valori base. Gestisce esogene storiche se presenti e produce backtest con step unitario.

### STATSFORECASTCopia.py

`STATSFORECAST_Predictor` usa la libreria `StatsForecast` e include modelli come Naive, SeasonalNaive, WindowAverage, RandomWalkWithDrift, HistoricAverage, AutoETS, ARIMA e SARIMA. Per ARIMA/SARIMA con esogene, la logica di forecast e' iterativa one-step con aggiornamento della storia. Il modello supporta una fase di cross-validation se `optimization` e' abilitato.

### ARIMA_model.py e SARIMA_model.py

Entrambi i moduli usano `statsmodels.SARIMAX`. La selezione automatica dei parametri e' presente ma commentata; i parametri di default in training sono fissi (ad esempio ordine (4,1,4) in entrambi). I metodi di test supportano open-loop one-step o multi-step. Espongono anche metodi di salvataggio metriche.

### XGB_model.py

`XGB_Predictor` costruisce feature temporali sinusoidali (giorno della settimana, ora, giorno del mese) e addestra un `XGBRegressor` con iperparametri fissati nel codice. La previsione e' point-wise sul set di test. Il modello prevede funzioni di plotting e salvataggio metriche.

### LSTM_torch.py

`LSTM_Predictor` implementa un LSTM in PyTorch con data windowing e training tramite `DataLoader`. E' presente una fase opzionale di validazione con ottimizzazione bayesiana (hyperopt). La classe produce curve di loss e supporta test con batch prediction. I dettagli architetturali dipendono dalla classe interna `LSTM_Network` (definita nello stesso file).

### MLFORECAST.py e NEURALFORECAST.py

Questi file definiscono versioni alternative dei predittori (con suffisso "2" nelle classi). Non risultano importati da `generate_forecasts.py`, ma potrebbero essere usati in script o notebook esterni.

## Flusso di configurazione

### Caricamento e normalizzazione

La configurazione e' caricata da `assume/extensions/retailer_sim/config.yaml` con `load_config`. L'audit `config_audit.audit_config` emette warning per chiavi non riconosciute. La funzione `load_dataframe` arricchisce la config con il passo temporale inferito e con le definizioni delle sessioni MI (array `_intraday_sessions`). Se sono presenti range di test nelle impostazioni di forecast (`forecasting.advanced_settings.test_range`), il dataframe viene filtrato a quella finestra tramite `filter_dataframe_for_simulation`.

### Parametri chiave e uso nel codice

Di seguito una sintesi testuale (non esaustiva) dei parametri principali e del loro uso. Per dettagli puntuali, si vedano le funzioni e classi citate:
- **decision_making.***: determina quale strategia istanziare e con quali preset/frazioni. E' consumato in `decision.py`.
- **market.***: controlla costi di transazione, capacita' e slope prezzi in `standalone/main_simulation.py` e `standalone/market.py`.
- **simulation.***: mappa colonne del CSV e definisce time step, finestra, output, reale vs compatto per sessioni MI.
- **retailer_logic.***: applica preset per regolatori MGP/MI, parametri Terna-like, margine cliente e MSD.
- **macro_imbalance_forecast.***: abilita il macro forecast e definisce banda (MGP/MI), usata sia per segnali macro sia per far rispettare il segno in intraday.
- **forecasting.*** e **forecast_mapping.***: guidano il merge di `forecast_inputs` e i fallback naive.

## Flusso dati end-to-end

### Dati di input e segnali disponibili

Il dataset di input contiene colonne per domanda, prezzi, sbilanciamenti e variabili esogene. Il caricamento e la normalizzazione avvengono in `assume/extensions/retailer_sim/data_utils.py`. Alcune colonne possono mancare: in tal caso il loader le crea e le riempie con fallback, garantendo che il simulatore non si fermi per assenza di segnali. Il risultato e' un dataframe uniforme in cui ogni slot ha almeno: timestamp, domanda prevista, prezzi MGP/MI, prezzo macro, coefficiente sbilanciamento, sbilanciamento previsto e metadati di sessione.

Un aspetto metodologico importante e' la distinzione tra:
- **forecast** (previsioni): usato per pianificare MGP/MI;
- **actual** (consumo reale): usato solo per calcolare lo sbilanciamento a fine slot.

Nel codice questa separazione e' visibile nelle colonne `consumption_col` vs `actual_consumption_col` e nelle funzioni di `imbalance` (`assume/extensions/retailer_sim/standalone/retailer.py`).

### Forecast inputs e scelta del modello

Il loader puo' integrare un file esterno `forecast_inputs.csv` (o un file con suffisso di modello) se presente. La selezione dipende da:
- `forecasting.mode`;
- `forecasting.ml_model` (solo se `mode=advanced`);
- esistenza fisica del file con suffisso (`forecast_inputs_<MODEL>.csv`).

Questa logica e' in `data_utils._load_forecast_inputs`. Se non esiste alcun file di forecast, il loader applica un mapping naive di colonne (`apply_naive_forecasts`).

### Creazione di StrategyInputs e causalita'

`StrategyInputs` viene popolato:
- in Standalone, dentro `Retailer.simulate`, usando la riga corrente del dataframe;
- nel World, tramite `RetailerUnit.build_strategy_inputs`, usando serie allineate del Forecaster.

Entrambe le strade calcolano una stima MSD prima di decidere i volumi. Questo implica che la strategia dispone di un indicatore di costo di sbilanciamento stimato, che guida sia la scelta MGP (shift/resell) sia le correzioni MI.

### Ritardi informativi

I ritardi di informazione (consumi, prezzi, sbilanciamento) non sono enforceati dal simulatore al runtime; vengono invece applicati nel pre-processing dei forecast, in `assume/extensions/retailer_sim/generate_forecasts.py`. Per questa ragione la corretta causalita' dipende dal fatto che i forecast inputs vengano generati con le stesse regole di ritardo configurate in `information_flow`.

## Logica decisionale

### Interfaccia comune delle strategie

Le strategie implementano `BiddingStrategy` (`assume/extensions/retailer_sim/decision.py`): una API minimale con `plan_day_ahead` e `plan_intraday`. Questa scelta evita che lo stesso algoritmo sia duplicato nei due contesti di esecuzione e consente a Standalone e World di usare la stessa logica con regolatori diversi.

### Strategia random

`RandomBiddingStrategy` genera volumi casuali entro limiti configurati e non considera prezzi o domanda. La casualita' e' deterministica se `random_seed` e' fissato. I volumi random vengono comunque sottoposti a regolatori MGP/MI (Standalone o World), per cui il comportamento finale e' spesso piu' vincolato rispetto alla generazione iniziale.

### Strategia rule-based (SimpleRetailStrategy)

La strategia rule-based e' un insieme di euristiche:
- stima un target MGP come frazione della domanda prevista, con clamp tra soglie minime e massime;
- applica un bias in base al segno del macro-imbalance se attivo;
- valuta un segnale di prezzo MI vs MGP e MSD per spostare volumi da MGP a MI (shift) o aggiungere volumi da rivendere (resell);
- calcola correzioni intraday proporzionali al residuo, ma limitate da ratio prezzi, limiti assoluti e limiti per sessione.

### Regolatori MGP/MI

La logica di regolazione e' stata introdotta per evitare che una strategia (anche rule-based) produca volumi non coerenti con un retailer reale. In Standalone, i regolatori sono in `Retailer._apply_mgp_regulator` e `Retailer._apply_intraday_regulator`. Nel World adapter, gli stessi principi sono replicati in `RetailerUnit.regulate_day_ahead_volume` e `RetailerUnit.regulate_intraday_volume`:
- minimo e massimo di copertura MGP rispetto alla domanda;
- divieto di vendite MGP se non previste;
- tolleranza per evitare micro-ordini MI;
- frazioni massime per sessione e smoothing, con scaling per liquidita'.

### Pseudocodice dei loop principali

**Caption: Standalone: ciclo di simulazione (schema)**
```python
for each day:
  for each slot:
    inputs = StrategyInputs(...)
  for each slot:
    target = strategy.plan_day_ahead(inputs)
    mgp_volume = apply_mgp_regulator(target, demand_forecast)
    execute_day_ahead_bid(mgp_volume, mgp_price)
  for each intraday session:
    for each eligible slot:
      inputs_session = inputs.for_intraday(mi_price, macro_band)
      desired = strategy.plan_intraday(inputs_session, contracted_before, session)
      mi_volume = apply_intraday_regulator(desired, demand_forecast, contracted_before)
      if abs(mi_volume) > 0: execute_intraday_bid(mi_volume)
  for each slot:
    imbalance = actual_consumption - contracted_total
    msd_price = compute_imbalance_price(...)
    imbalance_cost = evaluate_imbalance_cost(...)
    surplus = max(contracted_total - actual_consumption, 0)
    record HourlyResult
```

**Caption: World adapter: generazione offerte (schema)**
```python
# MGP
for each product slot:
  inputs = unit.build_strategy_inputs(timestamp, mi_price_market=session_ref, phase="mgp")
  target = strategy.plan_day_ahead(inputs)
  final = unit.regulate_day_ahead_volume(target, demand_forecast)
  unit.record_contracted_volume(timestamp, "MGP", final)
  emit order

# MI
for each product slot:
  if not unit.is_session_slot_eligible(timestamp, session_name): continue
  inputs = unit.build_strategy_inputs(timestamp, mi_price_market=market_id, phase="mi")
  desired = strategy.plan_intraday(inputs, contracted_before, session_index)
  final = unit.regulate_intraday_volume(session_index, desired, demand_forecast,
                                        contracted_before, liquidity_hint)
  if abs(final) > 0: record and emit order
```

## Esecuzione di mercato e settlement

### Mercati MGP/MI e clearing

Nel Standalone i mercati sono simulati con un clearing lineare. Il volume puo' essere clippato da un cap (locale o condiviso). Il prezzo di clearing e' il prezzo di riferimento piu' un incremento proporzionale alla saturazione. Questa logica e' codificata in `assume/extensions/retailer_sim/standalone/market.py` (`DayAheadMarket`, `IntraDayMarket`, `ClearingCoordinator`).

### Formula MSD e costo dello sbilanciamento

Il modello MSD usa una formula semplificata:
```text
price = mgp + (mgp - macro) * coeff + sens * |imbalance| + penalty
```

La formula e' implementata in `MSDSettlement.estimate_price` e il costo e' `abs(imbalance) * price`, con fattore di credito sulle posizioni long. Questo valore entra nel costo orario e influenza il profitto.

### Aggiustamenti Terna-like

`TernaLikeBalancingAgent` modifica il prezzo MSD in base alla direzione dello sbilanciamento rispetto al macro-sbilanciamento e a un volume di riferimento. Inoltre applica moltiplicatori su ore non lavorative e possibili penalita' per shortfall. In Standalone questa logica e' usata nella funzione `Retailer._compute_imbalance_price`; nel World adapter e' riusata in `RetailerUnit._estimate_msd_price` e nella ricostruzione output in `output_utils.py`.

### Gestione surplus

Quando il retailer ha contratti superiori al consumo reale, il surplus non e' venduto a un mercato con acquirente reale ma viene valorizzato al prezzo scelto dalla regola `surplus_sale_price`. La quantita' liquidata e' controllata da `surplus_sale_fraction`. Questa scelta e' utile per modellare il comportamento "price-taker" del surplus senza complicare il clearing.

## Output e dashboard

### Istruzioni di esecuzione (Standalone e World adapter)

Gli entry-point CLI non espongono flag per cambiare la configurazione: usano sempre `assume/extensions/retailer_sim/config.yaml` tramite `load_config`. Per usare un config diverso e' necessario sostituire quel file o modificare il path nel codice. Da root del repository:
**Caption: Esecuzione degli ambienti**
```bash
# Standalone (mercati e clearing locali)
python -m assume.extensions.retailer_sim.standalone.main_simulation

# World adapter (clearing interno al World ASSUME)
python -m assume.extensions.retailer_sim.world_adapter.run_world

# Variante equivalente
python -m assume.extensions.retailer_sim.world_adapter.run_world2
```

Gli output finiscono nella cartella `simulation.output_folder` (default `assume/extensions/retailer_sim/outputs`), con una sottocartella `outputs/retailer_world` per il World adapter.

### Output Standalone

`main_simulation.save_outputs` salva: `hourly_results.csv`, `aggregated_totals.json/csv` e `orders.json`. Le colonne di output sono definite da `HourlyResult.as_dict`. Il log ordini converte MWh per slot in MW (in base al time step) per mantenere la compatibilita' con formati di mercato.

### Output World e ricostruzione

Nel World adapter, l'output del World viene ricostruito da `output_utils.build_world_hourly_dataframe`, che unisce dispatch, prezzi e (se disponibile) un dataframe di riferimento con segnali macro e domanda. Il risultato e' salvato in `outputs/retailer_world` insieme a totali e ordini convertiti in un formato simile allo Standalone.

### Output di simulazione

I file di output generati dal codice includono:
- Standalone: `outputs/hourly_results.csv`, `outputs/aggregated_totals.json`, `outputs/aggregated_totals.csv`, `outputs/orders.json` (`standalone/main_simulation.py`).
- World adapter: `outputs/retailer_world/world_hourly_results.csv`, `outputs/retailer_world/world_totals.json`, `outputs/retailer_world/world_totals.csv`, `outputs/retailer_world/world_orders.json` (`output_utils.py`).

La ricostruzione World legge anche file generati dal core ASSUME (es. `market_dispatch.csv`, `market_meta.csv`, `market_orders.csv`) per ricavare volumi e prezzi.

### Dashboard

`plot_dashboard.py` produce un grafico stile Grafana con prezzi, volumi e KPI. Il file PNG di output e' salvato nel path passato al CLI (default `assume/extensions/retailer_sim/outputs/dashboard.png`).

### Dashboard e asset grafici

Lo script `standalone/plot_dashboard.py` salva una figura PNG. Il path di default e' `assume/extensions/retailer_sim/outputs/dashboard.png`, ma puo' essere sostituito via CLI. Il file e' prodotto solo se si esegue esplicitamente lo script.

### Istruzioni per il plot (dashboard)

Il plot usa `standalone/plot_dashboard.py` e supporta sia i risultati standalone sia gli export del World adapter. Esempi d'uso:
**Caption: Plot dashboard (standalone vs world)**
```bash
# Standalone: usa hourly_results.csv e aggregated_totals.json di default
python -m assume.extensions.retailer_sim.standalone.plot_dashboard

# Standalone con percorsi espliciti
python -m assume.extensions.retailer_sim.standalone.plot_dashboard \
  --results assume/extensions/retailer_sim/outputs/hourly_results.csv \
  --totals assume/extensions/retailer_sim/outputs/aggregated_totals.json \
  --output assume/extensions/retailer_sim/outputs/dashboard.png

# World adapter: punta alla cartella retailer_world
python -m assume.extensions.retailer_sim.standalone.plot_dashboard \
  --world-export assume/extensions/retailer_sim/outputs/retailer_world
```

Il comando per il World adapter usa `world_totals.json` se presente; in caso contrario, e' possibile passare `--totals`. Se `world_hourly_results.csv` non esiste, lo script lo ricostruisce a partire dagli export del World.

## Mappa del repository (parti rilevanti)

### Struttura generale

Il pacchetto `assume/extensions/retailer_sim` include:
- un nucleo di simulazione (strategie, retailer, mercato);
- un adapter per il mondo ASSUME (World);
- utility per caricare dati, produrre forecast, aggregare dataset e ricostruire output;
- moduli di forecasting sperimentali (ML, neural, statistici) e script ausiliari;
- file di configurazione e dataset di supporto.

La sezione seguente elenca tutti i file di codice principali, inclusi gli ausiliari come `aggregation.py`.

### Catalogo dei file di codice

| File | Ruolo e contenuto rilevante |
| --- | --- |
| `assume/extensions/retailer_sim/__init__.py` | Marca il pacchetto e abilita import coerenti. |
| `assume/extensions/retailer_sim/decision.py` | Cuore della logica di bidding: `BiddingStrategy`, `RandomBiddingStrategy`, `SimpleRetailStrategy`, `StrategyInputs`, factory `build_strategy`, preset e regole di shifting MI. |
| `assume/extensions/retailer_sim/data_utils.py` | Lettura config, preparazione dataframe, merge con forecast inputs, generazione sessioni MI, fill di colonne mancanti, window temporale. |
| `assume/extensions/retailer_sim/output_utils.py` | Scrittura output standalone, ricostruzione output World, calcolo totali, conversione ordini. |
| `assume/extensions/retailer_sim/config_audit.py` | Audit leggero dei blocchi YAML: warning su chiavi non riconosciute. |
| `assume/extensions/retailer_sim/leakage_guard.py` | Definisce un guard contro leakage dati per fase (MGP/MI/MSD); non risulta collegato al runtime. |
| `assume/extensions/retailer_sim/standalone/main_simulation.py` | Entry point Standalone: crea strategia, mercati, retailer, esegue simulazione, salva output. |
| `assume/extensions/retailer_sim/standalone/retailer.py` | Loop principale di simulazione Standalone: costruzione `StrategyInputs`, ordine MGP/MI, regolatori, calcolo sbilanciamento, surplus e tariffa. |
| `assume/extensions/retailer_sim/standalone/market.py` | Mercati MGP/MI con clearing lineare, coordinatore capacita', modello MSD (formula prezzo e costo). |
| `assume/extensions/retailer_sim/standalone/strategies.py` | Wrapper che re-esporta le strategie condivise da `decision.py`. |
| `assume/extensions/retailer_sim/standalone/plot_dashboard.py` | Dashboard Matplotlib basata sui risultati orari; supporta anche output World ricostruiti. |
| `assume/extensions/retailer_sim/world_adapter/world_simulation.py` | Costruzione del World ASSUME da dataframe: forecaster, mercati, unita', strategia condivisa. |
| `assume/extensions/retailer_sim/world_adapter/retailer_strategy.py` | Wrapper ASSUME delle strategie (MGP/MI) e strategia di balancing Terna-like. |
| `assume/extensions/retailer_sim/world_adapter/retailer_unit.py` | Unita' retailer nel World: ledger posizioni, regulator MGP/MI, build di `StrategyInputs`. |
| `assume/extensions/retailer_sim/world_adapter/terna_unit.py` | Unita' Terna-like minimalista nel World; fornisce capacita' e prezzi. |
| `assume/extensions/retailer_sim/world_adapter/virtual_supply_unit.py` | Alias storico: re-esporta `TernaBalancingUnit`. |
| `assume/extensions/retailer_sim/world_adapter/run_world.py` | CLI per eseguire il World adapter e esportare output. |
| `assume/extensions/retailer_sim/world_adapter/run_world2.py` | Variante equivalente di `run_world.py`. |
| `assume/extensions/retailer_sim/terna_agent.py` | Logica Terna-like per penalita'/incentivi e prezzi di sbilanciamento. |
| `assume/extensions/retailer_sim/generate_forecasts.py` | Orchestratore di forecasting: addestra modelli, seleziona migliori, produce `forecast_inputs.csv`. |
| `assume/extensions/retailer_sim/retailer_forecasts.py` | Orchestratore alternativo: valuta modelli su load e SBIL, seleziona per MAE. |
| `assume/extensions/retailer_sim/aggregation.py` | Script di aggregazione POD+PUN: normalizza POD orari, unisce PUN e festivita', genera `pods_hourly.csv`. |
| `assume/extensions/retailer_sim/pod_dataset.py` | Script di merge dataset: porta `ITALY_NORD_DATASET_ENRICHED.csv` ad orario, unisce POD/PUN, produce `ITALY_NORD_HOURLY_WITH_PODS_PUN.csv`. |
| `assume/extensions/retailer_sim/Predictor.py` | Classe base astratta per predittori (interfaccia `train_model`, `plot_predictions`). |
| `assume/extensions/retailer_sim/NAIVE_model.py` | Implementazione di forecast naive (mean/persistence/seasonal), plotting e salvataggio metriche. |
| `assume/extensions/retailer_sim/MLFORECAST.py` | Classe `MLFORECAST_Predictor2`: modelli ML (wrapper alternativo). |
| `assume/extensions/retailer_sim/MLFORECASTCopia.py` | Classe `MLFORECAST_Predictor`: modelli ML, usata da `generate_forecasts.py`. |
| `assume/extensions/retailer_sim/NEURALFORECAST.py` | Classe `NEURALFORECAST_Predictor2`: modelli neural forecast (wrapper alternativo). |
| `assume/extensions/retailer_sim/NEURALFORECASTCopia.py` | Classe `NEURALFORECAST_Predictor`: modelli neural, usata da `generate_forecasts.py`. |
| `assume/extensions/retailer_sim/STATSFORECASTCopia.py` | Classe `STATSFORECAST_Predictor`: modelli statistici su StatsForecast. |
| `assume/extensions/retailer_sim/STATSFORECASTlast.py` | Variante di `STATSFORECAST_Predictor` con implementazione alternativa. |
| `assume/extensions/retailer_sim/ARIMA_model.py` | Classe `ARIMA_Predictor`: forecast ARIMA (pmdarima/statsmodels). |
| `assume/extensions/retailer_sim/SARIMA_model.py` | Classe `SARIMA_Predictor`: forecast SARIMA. |
| `assume/extensions/retailer_sim/LSTM_torch.py` | Classe `LSTM_Predictor`: modello LSTM in PyTorch. |
| `assume/extensions/retailer_sim/XGB_model.py` | Classe `XGB_Predictor`: modello XGBoost. |

**Nota sui file di forecasting duplicati.**

Nel codice coesistono file con suffissi *Copia*, *last* o versioni con suffisso numerico. Il repository non documenta in modo esplicito la differenza funzionale; la guida indica solo quali versioni sono effettivamente importate dagli orchestratori (ad esempio `generate_forecasts.py` importa `MLFORECASTCopia.py` e `NEURALFORECASTCopia.py`).

## Checklist di riproducibilita'

Per riprodurre una simulazione in modo coerente, e' consigliabile fissare e documentare:
- seed random (per `RandomBiddingStrategy`);
- dataset di input e relativa versione;
- time step effettivo (derivato dal timestamp o forzato in config);
- file di forecast inputs usato (incluso suffisso del modello);
- range temporali (start/end, test range);
- parametri di sessione MI (real_market true/false);
- parametri MSD e Terna-like che influenzano i costi.

## Open questions / To confirm

Questi punti non sono completamente chiariti dal codice e dovrebbero essere confermati con i maintainer:
- **Unita' di misura**: molte colonne hanno suffisso `_MW` ma sono trattate come MWh per slot. Il codice non applica conversioni esplicite se non per la scrittura degli ordini. Confermare la convenzione attesa.
- **Macro vs retailer imbalance**: nel World adapter `macro_imbalance_column` e' settata a `imbalance_forecast_MWh` in `world_simulation.py`. Confermare se macro e retailer sbilanciamento devono essere serie distinte.
- **Leakage guard**: esiste `leakage_guard.py` e un blocco di config, ma non risulta integrato nel runtime. Confermare se si tratta di funzionalita' non ancora collegata.
- **Versioni dei modelli**: file con suffisso *Copia*/*last* non sono spiegati. Confermare quale versione e' considerata "ufficiale" per la tesi.
- **Predictors/ directory**: `retailer_forecasts.py` importa moduli da `assume/extensions/retailer_sim/Predictors`, ma nel repository corrente risultano presenti solo artefatti `__pycache__`. Verificare se i sorgenti mancano o risiedono altrove.

