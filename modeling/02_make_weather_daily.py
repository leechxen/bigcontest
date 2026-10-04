from __future__ import annotations

from pathlib import Path
import glob
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WEATHER = ROOT / "weather_data"
OUTPUT = ROOT / "modeling" / "weather_daily.csv"

# These are station-based proxies, not claims of exact administrative-area identity.
REGIONS = {
    "서울 강남구": {"issue_station": "서울(108)", "obs_id": 400, "obs_name": "강남"},
    "강원 춘천시": {"issue_station": "북춘천(93)", "obs_id": 101, "obs_name": "춘천"},
}
QUANTILES = (0.90, 0.95, 0.99)

def read_issue(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", encoding="cp949", dtype=str)

def require_columns(df: pd.DataFrame, columns: list[str], path: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{Path(path).name}: required columns not found: {missing}; actual={list(df.columns)}")

def main() -> None:
    if OUTPUT.exists() and "--overwrite" not in sys.argv:
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT}")

    hw_files = sorted(glob.glob(str(WEATHER / "ISSUE_HW_DAY_2025-0[789]*")))
    cw_files = sorted(glob.glob(str(WEATHER / "ISSUE_CW_DAY_2025-1[12]*")))
    obs_files = {
        region: next(iter(sorted(WEATHER.glob(f"OBS_{cfg['obs_name']}_시간별_TIM_*.csv"))), None)
        for region, cfg in REGIONS.items()
    }
    if len(hw_files) != 3 or len(cw_files) != 2 or any(path is None for path in obs_files.values()):
        raise FileNotFoundError("Expected ISSUE_HW (Jul-Sep), ISSUE_CW (Nov-Dec), and both hourly OBS files.")

    issue_rows = []
    temp_rows = []
    raw_files = [Path(p) for p in hw_files + cw_files + list(obs_files.values())]

    for path in hw_files:
        df = read_issue(path)
        require_columns(df, ["일시", "지점", "폭염여부(O/X)", "폭염특보(O/X)", "폭염영향예보(단계)",
                             "평균기온(°C)", "최저기온(°C)", "최고기온(°C)"], path)
        for region, cfg in REGIONS.items():
            sub = df.loc[df["지점"].str.strip() == cfg["issue_station"]].copy()
            if sub.empty:
                raise ValueError(f"{Path(path).name}: station missing: {cfg['issue_station']}")
            sub["TA_YMD"] = pd.to_datetime(sub["일시"], errors="raise").dt.strftime("%Y%m%d").astype("int64")
            flag = sub["폭염여부(O/X)"].str.strip()
            if not flag.isin(["O", "X"]).all():
                raise ValueError(f"Unexpected heatwave flag values: {sorted(flag.unique())}")
            issue_rows.append(pd.DataFrame({
                "TA_YMD": sub["TA_YMD"], "MCT_SGG_CD": region,
                "heatwave": flag.map({"O": 1, "X": 0}).astype("int64"),
                "heat_alert_status": sub["폭염특보(O/X)"].str.strip(),
                "heat_impact_forecast": sub["폭염영향예보(단계)"].str.strip(),
            }))
            temp_rows.append(pd.DataFrame({
                "TA_YMD": sub["TA_YMD"], "MCT_SGG_CD": region,
                "avg_temp": pd.to_numeric(sub["평균기온(°C)"], errors="coerce"),
                "min_temp": pd.to_numeric(sub["최저기온(°C)"], errors="coerce"),
                "max_temp": pd.to_numeric(sub["최고기온(°C)"], errors="coerce"),
            }))

    for path in cw_files:
        df = read_issue(path)
        require_columns(df, ["일시", "지점", "한파특보(O/X)", "한파영향예보(단계)",
                             "일평균기온(°C)", "일최저기온(°C)", "일최고기온(°C)"], path)
        for region, cfg in REGIONS.items():
            sub = df.loc[df["지점"].str.strip() == cfg["issue_station"]].copy()
            if sub.empty:
                raise ValueError(f"{Path(path).name}: station missing: {cfg['issue_station']}")
            sub["TA_YMD"] = pd.to_datetime(sub["일시"], errors="raise").dt.strftime("%Y%m%d").astype("int64")
            status = sub["한파특보(O/X)"].str.strip()
            allowed = {"X", "주의보", "경보"}
            if not set(status.dropna().unique()).issubset(allowed):
                raise ValueError(f"Unexpected cold-wave status values: {sorted(status.dropna().unique())}")
            issue_rows.append(pd.DataFrame({
                "TA_YMD": sub["TA_YMD"], "MCT_SGG_CD": region,
                "cold_wave": status.map({"X": 0, "주의보": 1, "경보": 1}).astype("Int64"),
                "cold_alert_status": status,
                "cold_impact_forecast": sub["한파영향예보(단계)"].str.strip(),
            }))
            temp_rows.append(pd.DataFrame({
                "TA_YMD": sub["TA_YMD"], "MCT_SGG_CD": region,
                "avg_temp": pd.to_numeric(sub["일평균기온(°C)"], errors="coerce"),
                "min_temp": pd.to_numeric(sub["일최저기온(°C)"], errors="coerce"),
                "max_temp": pd.to_numeric(sub["일최고기온(°C)"], errors="coerce"),
            }))

    for frame in issue_rows + temp_rows:
        if frame.duplicated(["TA_YMD", "MCT_SGG_CD"]).any():
            raise ValueError("Duplicate date-region key in selected ISSUE station rows.")

    obs_daily = []
    candidate_frames = {}
    for region, path in obs_files.items():
        df = pd.read_csv(path, encoding="cp949")
        require_columns(df, ["지점", "지점명", "일시", "강수량(mm)"], path)
        if not (df["지점"].astype(str) == str(REGIONS[region]["obs_id"])).all():
            raise ValueError(f"{Path(path).name}: unexpected station ID")
        dt = pd.to_datetime(df["일시"], errors="raise")
        if dt.duplicated().any():
            raise ValueError(f"{Path(path).name}: duplicate hourly timestamps")
        rain = pd.to_numeric(df["강수량(mm)"], errors="coerce")
        tmp = pd.DataFrame({"date": dt.dt.normalize(), "hour": dt.dt.hour, "rain": rain})
        rows = []
        for date, group in tmp.groupby("date"):
            complete = (len(group) == 24 and group["hour"].nunique() == 24 and group["rain"].notna().sum() == 24)
            rows.append({
                "TA_YMD": int(date.strftime("%Y%m%d")), "MCT_SGG_CD": region,
                "rainfall": group["rain"].sum() if complete else pd.NA,
                "rain_observation_hours": len(group),
                "rain_valid_hours": int(group["rain"].notna().sum()),
                "rainfall_complete": int(complete),
            })
        daily = pd.DataFrame(rows)
        obs_daily.append(daily)
        valid = pd.to_numeric(daily.loc[daily["rainfall_complete"] == 1, "rainfall"], errors="coerce").dropna()
        candidate_frames[region] = valid

    min_date = min(pd.to_datetime(d.iloc[:, 2]).min() for d in [
        pd.read_csv(path, encoding="cp949") for path in obs_files.values()
    ]).normalize()
    max_date = max(pd.to_datetime(d.iloc[:, 2]).max() for d in [
        pd.read_csv(path, encoding="cp949") for path in obs_files.values()
    ]).normalize()
    dates = pd.date_range(min_date, max_date, freq="D")
    base = pd.MultiIndex.from_product(
        [dates.strftime("%Y%m%d").astype("int64"), REGIONS.keys()],
        names=["TA_YMD", "MCT_SGG_CD"]
    ).to_frame(index=False)

    result = base
    issue_data = pd.concat(issue_rows, ignore_index=True)
    if issue_data.duplicated(["TA_YMD", "MCT_SGG_CD"]).any():
        raise ValueError("Duplicate event date-region key in selected ISSUE rows.")
    result = result.merge(issue_data, on=["TA_YMD", "MCT_SGG_CD"], how="left", validate="one_to_one")
    temperatures = pd.concat(temp_rows, ignore_index=True)
    if temperatures.duplicated(["TA_YMD", "MCT_SGG_CD"]).any():
        raise ValueError("Duplicate temperature date-region key.")
    result = result.merge(temperatures, on=["TA_YMD", "MCT_SGG_CD"], how="left", validate="one_to_one")
    rainfall = pd.concat(obs_daily, ignore_index=True)
    result = result.merge(rainfall, on=["TA_YMD", "MCT_SGG_CD"], how="left", validate="one_to_one")
    result["heatwave"] = result["heatwave"].astype("Int64")
    result["cold_wave"] = result["cold_wave"].astype("Int64")
    # No heavy-rain threshold has been approved, so retain the variable as missing.
    result["heavy_rain"] = pd.Series(pd.NA, index=result.index, dtype="Int64")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    columns = ["TA_YMD", "MCT_SGG_CD", "heatwave", "heat_alert_status", "heat_impact_forecast",
               "heavy_rain", "cold_wave", "cold_alert_status", "cold_impact_forecast",
               "avg_temp", "min_temp", "max_temp", "rainfall",
               "rain_observation_hours", "rain_valid_hours", "rainfall_complete"]
    result = result[columns].sort_values(["TA_YMD", "MCT_SGG_CD"]).reset_index(drop=True)
    result.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print("=== weather_daily V1 validation ===")
    print(f"rows: {len(result)}")
    print(f"date range: {pd.to_datetime(result['TA_YMD'].astype(str), format='%Y%m%d').min().date()} .. {pd.to_datetime(result['TA_YMD'].astype(str), format='%Y%m%d').max().date()}")
    print("rows by region:")
    print(result.groupby("MCT_SGG_CD").size().to_string())
    print("missing values:")
    print(result.isna().sum().to_string())
    print(f"heatwave days (sum over region-days): {int(result['heatwave'].sum())}")
    print(f"cold_wave days (sum over region-days): {int(result['cold_wave'].sum())}")
    print(f"rainfall observable days: {int(result['rainfall_complete'].fillna(0).sum())}")
    print("heavy_rain candidate thresholds: region-specific empirical quantiles among complete daily rainfall totals; heavy_rain remains NA pending human choice.")
    for region, values in candidate_frames.items():
        print(f"[{region}] complete days={len(values)}")
        for q in QUANTILES:
            threshold = values.quantile(q)
            count = int((values >= threshold).sum())
            print(f"  q{int(q*100)} threshold={threshold:.3f} mm, days >= threshold={count}/{len(values)}")
    key_dups = int(result.duplicated(["TA_YMD", "MCT_SGG_CD"]).sum())
    print(f"duplicate date-region keys: {key_dups}")
    print("source files:")
    for path in raw_files:
        print(f"  {path.relative_to(ROOT)}")
    print(f"output: {OUTPUT.relative_to(ROOT)}")
    print("proxy mapping: Seoul Gangnam-gu -> ISSUE Seoul(108), OBS Gangnam(400); Chuncheon-si -> ISSUE North Chuncheon(93), OBS Chuncheon(101). These are station-based proxies, not exact administrative-area equivalence.")

if __name__ == "__main__":
    main()
