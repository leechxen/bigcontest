from __future__ import annotations

from pathlib import Path
import glob
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELING = ROOT / "modeling"
CARD_PATH = MODELING / "daily_card_agg.csv"
WEATHER_PATH = MODELING / "weather_daily.csv"
OUTPUT = MODELING / "modeling_dataset_v1.csv"
HEAVY_RAIN_3H_MM = 60.0

# Station-based proxies, not exact administrative-area identities.
OBS_STATION_TO_REGION = {
    "400": "서울 강남구",  # Gangnam station proxy for Seoul Gangnam-gu
    "101": "강원 춘천시",  # Chuncheon station proxy for Chuncheon-si
}

CARD_REQUIRED = [
    "TA_YMD", "MCT_SGG_CD", "MCT_RY_CD", "TS_AT", "USE_CNT", "avg_price"
]
WEATHER_REQUIRED = [
    "TA_YMD", "MCT_SGG_CD", "heatwave", "cold_wave",
    "avg_temp", "min_temp", "max_temp", "rainfall",
    "rain_observation_hours", "rain_valid_hours", "rainfall_complete",
]

def require_columns(df: pd.DataFrame, required: list[str], path: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}; found {list(df.columns)}")

def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT}")

    card = pd.read_csv(CARD_PATH, encoding="utf-8-sig")
    weather = pd.read_csv(WEATHER_PATH, encoding="utf-8-sig")
    require_columns(card, CARD_REQUIRED, CARD_PATH)
    require_columns(weather, WEATHER_REQUIRED, WEATHER_PATH)

    print("=== Input schema and key check ===")
    print(f"{CARD_PATH.name}: rows={len(card)}, dtypes={card.dtypes.astype(str).to_dict()}")
    print(f"{WEATHER_PATH.name}: rows={len(weather)}, dtypes={weather.dtypes.astype(str).to_dict()}")

    for df, name in ((card, CARD_PATH.name), (weather, WEATHER_PATH.name)):
        df["TA_YMD"] = pd.to_numeric(df["TA_YMD"], errors="raise").astype("int64")
        df["MCT_SGG_CD"] = df["MCT_SGG_CD"].astype("string").str.strip()
        if df[["TA_YMD", "MCT_SGG_CD"]].isna().any().any():
            raise ValueError(f"{name}: null join keys found")
    card["MCT_RY_CD"] = card["MCT_RY_CD"].astype("string").str.strip()

    if card.duplicated(["TA_YMD", "MCT_SGG_CD", "MCT_RY_CD"]).any():
        raise ValueError("Card input has duplicate date-region-industry keys.")
    if weather.duplicated(["TA_YMD", "MCT_SGG_CD"]).any():
        raise ValueError("Weather input has duplicate date-region keys.")

    print("card TA_YMD dtype:", card["TA_YMD"].dtype)
    print("weather TA_YMD dtype:", weather["TA_YMD"].dtype)
    print("card regions:", sorted(card["MCT_SGG_CD"].dropna().unique().tolist()))
    print("weather regions:", sorted(weather["MCT_SGG_CD"].dropna().unique().tolist()))

    # Calculate a daily observed exposure from three consecutive hourly values.
    # A 3-hour window is assigned to the calendar date of its ending timestamp.
    # A day is 1 if any valid window reaches 60 mm; 0 only when the full
    # calendar day is observed and no window reaches 60 mm; otherwise it is NA.
    hourly_files = sorted(glob.glob(str(ROOT / "weather_data" / "OBS_*_시간별_TIM_*.csv")))
    hourly_parts = []
    for file_name in hourly_files:
        path = Path(file_name)
        hourly = pd.read_csv(path, encoding="cp949")
        required_hourly = ["지점", "지점명", "일시", "강수량(mm)"]
        require_columns(hourly, required_hourly, path)
        station_ids = hourly["지점"].astype(str).str.strip().unique()
        if len(station_ids) != 1 or station_ids[0] not in OBS_STATION_TO_REGION:
            continue
        station_id = station_ids[0]
        region = OBS_STATION_TO_REGION[station_id]
        timestamps = pd.to_datetime(hourly["일시"], errors="raise")
        rain = pd.to_numeric(hourly["강수량(mm)"], errors="coerce")
        series = pd.Series(rain.to_numpy(), index=timestamps).sort_index()
        if series.index.has_duplicates:
            raise ValueError(f"{path.name}: duplicate hourly timestamps")
        series = series.asfreq("h")
        rolling_3h = series.rolling(window=3, min_periods=3).sum()

        for day, day_series in series.groupby(series.index.normalize()):
            day_windows = rolling_3h.loc[rolling_3h.index.normalize() == day].dropna()
            complete_day = (
                len(day_series) == 24
                and day_series.notna().sum() == 24
                and day_series.index.hour.nunique() == 24
            )
            hit = bool((day_windows >= HEAVY_RAIN_3H_MM).any())
            if hit:
                heavy_rain = 1
            elif complete_day:
                heavy_rain = 0
            else:
                heavy_rain = pd.NA
            hourly_parts.append({
                "TA_YMD": int(day.strftime("%Y%m%d")),
                "MCT_SGG_CD": region,
                "heavy_rain": heavy_rain,
                "rain_3h_max_mm": day_windows.max() if len(day_windows) else pd.NA,
                "rain_3h_valid_windows": int(len(day_windows)),
            })

    if not hourly_parts:
        raise FileNotFoundError("No matching hourly OBS rainfall files found.")
    rain3 = pd.DataFrame(hourly_parts)
    rain3["heavy_rain"] = rain3["heavy_rain"].astype("Int64")
    if rain3.duplicated(["TA_YMD", "MCT_SGG_CD"]).any():
        raise ValueError("Derived 3-hour rainfall data has duplicate date-region keys.")

    # Use only the requested weather fields plus observation completeness measures.
    weather_columns = [
        "TA_YMD", "MCT_SGG_CD", "heatwave", "cold_wave",
        "avg_temp", "min_temp", "max_temp", "rainfall",
        "rain_observation_hours", "rain_valid_hours", "rainfall_complete",
    ]
    weather_for_join = weather[weather_columns].merge(
        rain3,
        on=["TA_YMD", "MCT_SGG_CD"],
        how="left",
        validate="one_to_one",
    )

    merged = card.merge(
        weather_for_join,
        on=["TA_YMD", "MCT_SGG_CD"],
        how="left",
        validate="many_to_one",
        indicator="_weather_match",
    )
    matched = merged["_weather_match"].eq("both")
    unmatched = merged.loc[~matched, ["TA_YMD", "MCT_SGG_CD"]].drop_duplicates()
    merged["heatwave"] = merged["heatwave"].astype("Int64")
    merged["cold_wave"] = merged["cold_wave"].astype("Int64")
    merged["heavy_rain"] = merged["heavy_rain"].astype("Int64")

    output_columns = [
        "TA_YMD", "MCT_SGG_CD", "MCT_RY_CD",
        "TS_AT", "USE_CNT", "avg_price",
        "heatwave", "heavy_rain", "cold_wave",
        "avg_temp", "min_temp", "max_temp", "rainfall",
        "rain_observation_hours", "rain_valid_hours", "rainfall_complete",
        "rain_3h_max_mm", "rain_3h_valid_windows",
    ]
    result = merged[output_columns].copy()

    print("\n=== V1 validation ===")
    print(f"rows: {len(result)}")
    print(f"date range: {result['TA_YMD'].min()} .. {result['TA_YMD'].max()}")
    print(f"regions: {result['MCT_SGG_CD'].nunique()}")
    print(f"industries: {result['MCT_RY_CD'].nunique()}")
    print("duplicate TA_YMD × MCT_SGG_CD × MCT_RY_CD rows:",
          int(result.duplicated(["TA_YMD", "MCT_SGG_CD", "MCT_RY_CD"]).sum()))
    print(f"weather key match: {int(matched.sum())}/{len(result)} "
          f"({matched.mean():.2%}); unmatched card rows={int((~matched).sum())}")
    if len(unmatched):
        print("unmatched date-region keys (not zero-filled):")
        print(unmatched.to_string(index=False))
    print("missing values:")
    print(result[[
        "heatwave", "heavy_rain", "cold_wave",
        "avg_temp", "min_temp", "max_temp", "rainfall"
    ]].isna().sum().to_string())
    print("temperature missing total:",
          int(result[["avg_temp", "min_temp", "max_temp"]].isna().sum().sum()))

    print("regional weather match and heavy_rain event dates:")
    for region, group in merged.groupby("MCT_SGG_CD", dropna=False):
        region_match = group["_weather_match"].eq("both")
        events = group.loc[group["heavy_rain"].eq(1), "TA_YMD"].nunique()
        print(f"  {region}: rows={len(group)}, match={region_match.mean():.2%}, "
              f"heavy_rain region-days={events}")

    print("industry row counts:")
    print(result.groupby("MCT_RY_CD", dropna=False).size().sort_index().to_string())

    print("card input rows:", len(card))
    print("V1 output rows:", len(result))
    if len(card) != len(result):
        print("row-count difference reason: left join row count changed; inspect join-key cardinality.")
        raise ValueError("V1 row count differs from card input row count.")

    print("heavy_rain definition: observed 3-hour rolling rainfall >= 60 mm => 1.")
    print("Zero is assigned only on fully observed calendar days with no qualifying window; "
          "incomplete days without a hit remain missing.")
    print("This is an analysis exposure variable, not an official heavy-rain warning occurrence.")
    print("proxy mapping: Seoul Gangnam-gu -> Gangnam(400); Chuncheon-si -> Chuncheon(101).")
    print("source files:")
    print(f"  {CARD_PATH.relative_to(ROOT)}")
    print(f"  {WEATHER_PATH.relative_to(ROOT)}")
    for file_name in hourly_files:
        print(f"  {Path(file_name).relative_to(ROOT)}")

    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT}")
    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print("output:", OUTPUT.relative_to(ROOT))

if __name__ == "__main__":
    main()
