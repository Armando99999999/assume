from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

# Input files
ENRICHED_PATH = BASE_DIR / "ITALY_NORD_DATASET_ENRICHED.csv"
PODS_HOURLY_CANDIDATES = [
    BASE_DIR / "outputs" / "pods_hourly.csv",
    BASE_DIR / "world_adapter" / "pods_hourly.csv",
    BASE_DIR / "pods_hourly.csv",
]

# Output
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "ITALY_NORD_HOURLY_WITH_PODS_PUN.csv"


def to_datetime_floor_hour(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s, errors="coerce")
    return ts.dt.floor("H")


def build_hourly_from_enriched(df_15m: pd.DataFrame) -> pd.DataFrame:
    """
    Porta ITALY_NORD_DATASET_ENRICHED (tipicamente 15-min) a frequenza oraria.
    Regole:
      - colonne *_MWH e SBIL_MWH: SUM (energia)
      - colonne *_MW, *_MWQH: MEAN (potenza)
      - prezzi (contiene 'PRICE'): MEAN
      - flag is_*: MAX
      - calendar (day_of_week, month, etc): FIRST
      - altri numerici: MEAN
    """
    if "ORAINI" not in df_15m.columns:
        raise ValueError("ITALY_NORD_DATASET_ENRICHED.csv: missing 'ORAINI' timestamp column")

    df = df_15m.copy()
    df["timestamp"] = to_datetime_floor_hour(df["ORAINI"])
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")

    # Build aggregation map
    agg = {}

    for col in df.columns:
        if col == "ORAINI":
            continue

        c = col.upper()

        # Flags and calendar-like fields
        if col.startswith("is_"):
            agg[col] = "max"
            continue
        if col in {"hour", "minute", "day_of_week", "day_of_year", "week_of_year", "month"}:
            agg[col] = "first"
            continue

        # Prices
        if "PRICE" in c:
            agg[col] = "mean"
            continue

        # Energy (MWH)
        if c.endswith("_MWH") or c == "SBIL_MWH":
            agg[col] = "sum"
            continue

        # Power-like
        if c.endswith("_MW") or c.endswith("_MWQH"):
            agg[col] = "mean"
            continue

        # Purchases/sales volumes: treat as energy-like (sum)
        if any(k in c for k in ["PURCHASE", "SALES"]):
            agg[col] = "sum"
            continue

        # default
        if pd.api.types.is_numeric_dtype(df[col]):
            agg[col] = "mean"
        else:
            agg[col] = "first"

    hourly = df.resample("H").agg(agg).reset_index()
    hourly = hourly.rename(columns={"timestamp": "ORAINI"})  # mantieni nome timestamp coerente con dataset originale
    return hourly


def main():
    if not ENRICHED_PATH.exists():
        raise FileNotFoundError(f"Missing {ENRICHED_PATH}")
    pods_path = next((p for p in PODS_HOURLY_CANDIDATES if p.exists()), None)
    if pods_path is None:
        checked = ", ".join(str(p) for p in PODS_HOURLY_CANDIDATES)
        raise FileNotFoundError(f"Missing pods_hourly.csv (checked: {checked})")

    # Load
    nord = pd.read_csv(ENRICHED_PATH)
    pods = pd.read_csv(pods_path)

    # Make hourly from enriched
    nord_h = build_hourly_from_enriched(nord)

    # Prepare pods timestamps (hourly)
    if "timestamp" not in pods.columns:
        raise ValueError("pods_hourly.csv must contain 'timestamp'")
    pods = pods.copy()
    pods["timestamp"] = to_datetime_floor_hour(pods["timestamp"])
    pods = pods.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Align nord_h timestamps too (ORAINI is hourly already after resample)
    nord_h = nord_h.copy()
    nord_h["ORAINI"] = to_datetime_floor_hour(nord_h["ORAINI"])
    nord_h = nord_h.dropna(subset=["ORAINI"]).sort_values("ORAINI")

    # Merge (left = keep all columns from nord_h)
    merged = nord_h.merge(
        pods[["timestamp", "POD_PORTFOLIO_MWH", "POD_PORTFOLIO_MWH_MONTH", "price_PUN_EUR_MWh"]],
        left_on="ORAINI",
        right_on="timestamp",
        how="left",
    )
    merged = merged.drop(columns=["timestamp"])

    # --- Apply replacements/additions ---
    # 1) cluster_total_load_MW -> POD_PORTFOLIO_MWH (sostituzione "contenuto")
    if "cluster_total_load_MW" in merged.columns:
        merged["cluster_total_load_MW"] = merged["POD_PORTFOLIO_MWH"]
    else:
        # Se non esiste, creala (così resti compatibile col tuo framework)
        merged["cluster_total_load_MW"] = merged["POD_PORTFOLIO_MWH"]

    # 2) MGP_PRICE_NORD -> price_PUN_EUR_MWh (sostituzione "contenuto")
    if "MGP_PRICE_NORD" in merged.columns:
        merged["MGP_PRICE_NORD"] = merged["price_PUN_EUR_MWh"]
    else:
        merged["MGP_PRICE_NORD"] = merged["price_PUN_EUR_MWh"]

    # 3) Add monthly portfolio column
    if "POD_PORTFOLIO_MWH_MONTH" not in merged.columns:
        merged["POD_PORTFOLIO_MWH_MONTH"] = pd.NA

    # Optional: if you want MI_PRICE_NORD aligned too (you didn't ask, so commented)
    # if "MI_PRICE_NORD" in merged.columns:
    #     merged["MI_PRICE_NORD"] = merged["price_PUN_EUR_MWh"]

    # Save
    merged = merged.sort_values("ORAINI").drop_duplicates(subset=["ORAINI"], keep="first")
    merged = merged.drop(columns=["POD_PORTFOLIO_MWH", "price_PUN_EUR_MWh"], errors="ignore")
    merged.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")
    print("Rows:", len(merged), "Cols:", len(merged.columns))
    print(merged.head(5))


if __name__ == "__main__":
    main()
