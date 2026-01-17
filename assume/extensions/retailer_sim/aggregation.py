from pathlib import Path

import pandas as pd
import yaml
import warnings

# Toggle: keep the extra hour (KWH_25) as a distinct timestamp.
KEEP_KWH_25 = False
# PUN input configuration.
PUN_FILE = "PUN_ORARIO_FLG_012019_122025.csv"
PUN_COLUMNS = {
    "PUN MENSILE F1": "pun_f1_EUR_MWh",
    "PUN MENSILE F2": "pun_f2_EUR_MWh",
    "PUN MENSILE F3": "pun_f3_EUR_MWh",
    "PUN MENSILE FU": "price_MGP_EUR_MWh",
}

# --- LOAD POD FILE ---
base_dir = Path(__file__).resolve().parent
pods_path = base_dir / "POD_CURVA_EFFETTIVA_ORARIA_MTA2.csv"
if not pods_path.exists():
    raise FileNotFoundError(f"Missing POD file at {pods_path}")
pods = pd.read_csv(pods_path)

# --- MONTHLY PORTFOLIO FROM KWH_MESE ---
def parse_datetime_dual(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=False)
    if parsed.isna().any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fallback = pd.to_datetime(series[parsed.isna()], errors="coerce", dayfirst=True)
        parsed.loc[parsed.isna()] = fallback
    return parsed

validity = (
    pods[["POD", "KWH_MESE", "DATA_INIZIO_VALIDITA", "DATA_FINE_VALIDITA"]]
    .dropna(subset=["KWH_MESE", "DATA_INIZIO_VALIDITA", "DATA_FINE_VALIDITA"])
    .drop_duplicates()
)
validity["KWH_MESE"] = pd.to_numeric(validity["KWH_MESE"], errors="coerce")
validity = validity.dropna(subset=["KWH_MESE"])
validity["start"] = parse_datetime_dual(validity["DATA_INIZIO_VALIDITA"])
validity["end"] = parse_datetime_dual(validity["DATA_FINE_VALIDITA"])
validity = validity.dropna(subset=["start", "end"]).reset_index(drop=True)
validity["year"] = validity["end"].dt.year
validity["month"] = validity["end"].dt.month
# Take the latest validity per POD and month, then sum across PODs.
latest_per_pod = (
    validity.sort_values(["year", "month", "POD", "end", "start"])
    .drop_duplicates(subset=["year", "month", "POD"], keep="last")
)
monthly_portfolio = (
    latest_per_pod
    .groupby(["year", "month"], as_index=False)["KWH_MESE"]
    .sum()
    .rename(columns={"KWH_MESE": "POD_PORTFOLIO_KWH_MONTH"})
)

# --- LOAD CONFIG ---
config_path = base_dir / "config.yaml"
if not config_path.exists():
    raise FileNotFoundError(f"Missing config file at {config_path}")
with config_path.open("r", encoding="utf-8") as fh:
    config = yaml.safe_load(fh) or {}
sim_cfg = config.get("simulation", {})
forecast_cfg = config.get("forecasting", {}).get("advanced_settings", {})
ranges = []
for key in ("train_range", "validation_range", "valid_range", "test_range"):
    value = forecast_cfg.get(key)
    if isinstance(value, list) and len(value) >= 2:
        start_val = pd.to_datetime(value[0], errors="coerce")
        end_val = pd.to_datetime(value[1], errors="coerce")
        if not pd.isna(start_val) and not pd.isna(end_val):
            ranges.append((start_val, end_val))
if ranges:
    start_dt = min(r[0] for r in ranges)
    end_dt = max(r[1] for r in ranges)
else:
    start_dt = pd.to_datetime(sim_cfg.get("start_datetime"), errors="coerce")
    end_dt = pd.to_datetime(sim_cfg.get("end_datetime"), errors="coerce")
if pd.isna(start_dt) or pd.isna(end_dt):
    raise ValueError("Invalid simulation start/end datetime in config.yaml")
if end_dt == end_dt.normalize():
    end_dt = end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

# --- PREPARE HOURLY DATA ---
kwh_cols = [f"KWH_{i:02d}" for i in range(1, 25)]
if KEEP_KWH_25:
    kwh_cols.append("KWH_25")
kwh_cols = [c for c in kwh_cols if c in pods.columns]

pods_long = pods.melt(
    id_vars=["POD", "ANNO", "MESE", "GIORNO"],
    value_vars=kwh_cols,
    var_name="ORA",
    value_name="KWH"
)
pods_long["KWH"] = pd.to_numeric(pods_long["KWH"], errors="coerce").fillna(0.0)
pods_long["hour_idx"] = pd.to_numeric(
    pods_long["ORA"].str.replace("KWH_", "", regex=False),
    errors="coerce"
)
hour_base = pods_long["hour_idx"].clip(upper=24) - 1
pods_long["ts"] = pd.to_datetime(
    dict(
        year=pods_long["ANNO"],
        month=pods_long["MESE"],
        day=pods_long["GIORNO"],
        hour=hour_base
    ),
    errors="coerce",
    utc=False
)
if KEEP_KWH_25:
    kwh_25_mask = pods_long["hour_idx"] == 25
    pods_long.loc[kwh_25_mask, "ts"] = pods_long.loc[kwh_25_mask, "ts"] + pd.Timedelta(minutes=30)
pods_long = pods_long.dropna(subset=["ts"])

pods_hourly = (
    pods_long
    .groupby("ts", as_index=False)["KWH"]
    .sum()
    .rename(columns={"KWH": "POD_PORTFOLIO_KWH"})
)
pods_hourly = pods_hourly.loc[
    (pods_hourly["ts"] >= start_dt) & (pods_hourly["ts"] <= end_dt)
].reset_index(drop=True)

pods_hourly["year"] = pods_hourly["ts"].dt.year
pods_hourly["month"] = pods_hourly["ts"].dt.month
pods_hourly = pods_hourly.merge(monthly_portfolio, on=["year", "month"], how="left")

# --- LOAD PUN (monthly or hourly) ---
pun_path = base_dir / PUN_FILE
if not pun_path.exists():
    raise FileNotFoundError(f"Missing PUN file at {pun_path}")
pun_df = pd.read_csv(pun_path)
pun_columns = [c for c in PUN_COLUMNS.keys() if c in pun_df.columns]
missing = [c for c in ("PUN MENSILE F1", "PUN MENSILE F2", "PUN MENSILE F3") if c not in pun_df.columns]
if missing:
    raise ValueError(f"PUN file missing required columns: {', '.join(missing)}")
if "ANNO" in pun_df.columns and "MESE" in pun_df.columns:
    pun_monthly = pun_df[["ANNO", "MESE"] + pun_columns].copy()
    pun_monthly = pun_monthly.rename(columns={"ANNO": "year", "MESE": "month", **PUN_COLUMNS})
    for col in PUN_COLUMNS.values():
        if col in pun_monthly.columns:
            pun_monthly[col] = pd.to_numeric(pun_monthly[col], errors="coerce")
    pods_hourly = pods_hourly.merge(pun_monthly, on=["year", "month"], how="left")
else:
    pun_ts_col = None
    for candidate in ("timestamp", "ts", "DATA_ORA", "DATA", "data", "datetime"):
        if candidate in pun_df.columns:
            pun_ts_col = candidate
            break
    if pun_ts_col is None:
        raise ValueError("PUN file format not recognized (no monthly or timestamp columns found).")
    pun_ts = pd.to_datetime(pun_df[pun_ts_col], errors="coerce")
    pun_ts_alt = pd.to_datetime(pun_df[pun_ts_col], errors="coerce", dayfirst=True)
    pun_df["timestamp"] = pun_ts.fillna(pun_ts_alt)
    pun_df = pun_df.dropna(subset=["timestamp"])
    if not pun_columns:
        raise ValueError("PUN file has no price columns.")
    pun_df = pun_df[["timestamp"] + pun_columns].copy()
    pun_df = pun_df.rename(columns=PUN_COLUMNS)
    for col in PUN_COLUMNS.values():
        if col in pun_df.columns:
            pun_df[col] = pd.to_numeric(pun_df[col], errors="coerce")
    pods_hourly = pods_hourly.rename(columns={"ts": "timestamp"})
    pods_hourly = pods_hourly.merge(pun_df, on="timestamp", how="left")
    pods_hourly = pods_hourly.rename(columns={"timestamp": "ts"})

pods_hourly = pods_hourly.drop(columns=["year", "month"])

# --- LOAD HOLIDAYS FOR FASCE ---
italy_path = base_dir / "ITALY_NORD_DATASET_ENRICHED.csv"
if not italy_path.exists():
    for parent in base_dir.parents:
        candidate = parent / "ITALY_NORD_DATASET_ENRICHED.csv"
        if candidate.exists():
            italy_path = candidate
            break
if not italy_path.exists():
    raise FileNotFoundError(f"Missing holidays file at {italy_path}")
italy_df = pd.read_csv(italy_path)
italy_df.columns = italy_df.columns.str.strip()
if "is_holiday" not in italy_df.columns:
    raise ValueError("Holidays file missing required column: is_holiday")
ts_candidates = ("timestamp", "ts", "datetime", "data", "DATA_ORA", "ORAINI")
italy_ts_col = next((c for c in ts_candidates if c in italy_df.columns), None)
if italy_ts_col is None:
    raise ValueError("Holidays file missing a recognizable timestamp column")
italy_subset = italy_df[[italy_ts_col, "is_holiday"]].copy()
italy_subset["timestamp"] = parse_datetime_dual(italy_subset[italy_ts_col])
italy_subset = italy_subset.dropna(subset=["timestamp"])
italy_subset = italy_subset[["timestamp", "is_holiday"]]

pods_hourly = pods_hourly.rename(columns={"ts": "timestamp"})
pods_hourly = pods_hourly.merge(italy_subset, on="timestamp", how="left")
pods_hourly["is_holiday"] = pods_hourly["is_holiday"].fillna(0).astype(int)

hour = pods_hourly["timestamp"].dt.hour
weekday = pods_hourly["timestamp"].dt.dayofweek
f3_mask = (pods_hourly["is_holiday"] == 1) | (weekday == 6) | (hour < 7) | (hour >= 23)
f1_mask = (~f3_mask) & (weekday <= 4) & (hour >= 8) & (hour < 19)
f2_mask = ~(f1_mask | f3_mask)
pods_hourly["F1_flag"] = f1_mask.astype(int)
pods_hourly["F2_flag"] = f2_mask.astype(int)
pods_hourly["F3_flag"] = f3_mask.astype(int)
if not ((pods_hourly["F1_flag"] + pods_hourly["F2_flag"] + pods_hourly["F3_flag"]) == 1).all():
    raise ValueError("Invalid one-hot fascia assignment")

pods_hourly["price_PUN_EUR_MWh"] = (
    pods_hourly["F1_flag"] * pods_hourly["pun_f1_EUR_MWh"]
    + pods_hourly["F2_flag"] * pods_hourly["pun_f2_EUR_MWh"]
    + pods_hourly["F3_flag"] * pods_hourly["pun_f3_EUR_MWh"]
)

# --- OUTPUT COLUMNS ---
pods_hourly = pods_hourly.rename(columns={"price_MGP_EUR_MWh": "PUN MENSILE FU"})
pods_hourly["POD_PORTFOLIO_MWH"] = pods_hourly["POD_PORTFOLIO_KWH"] / 1000.0
pods_hourly["POD_PORTFOLIO_MWH_MONTH"] = pods_hourly["POD_PORTFOLIO_KWH_MONTH"] / 1000.0
pods_hourly = pods_hourly.sort_values("timestamp").reset_index(drop=True)
output_columns = [
    "timestamp",
    "POD_PORTFOLIO_MWH",
    "POD_PORTFOLIO_MWH_MONTH",
    "PUN MENSILE FU",
    "pun_f1_EUR_MWh",
    "pun_f2_EUR_MWh",
    "pun_f3_EUR_MWh",
    "F1_flag",
    "F2_flag",
    "F3_flag",
    "price_PUN_EUR_MWh",
]
pods_hourly = pods_hourly[[c for c in output_columns if c in pods_hourly.columns]]

output_dir = base_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "pods_hourly.csv"
pods_hourly.to_csv(output_path, index=False)
print(pods_hourly.head(10))
print(f"Saved: {output_path}")
