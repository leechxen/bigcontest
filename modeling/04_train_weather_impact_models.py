from __future__ import annotations

"""Reproducible Y1 weather-impact analysis with grouped split and leakage audits."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "modeling" / "modeling_dataset_v2_filtered.csv"
OUTPUT_DIR = ROOT / "modeling" / "weather_impact_results"
KEY = ["MCT_SGG_CD", "MCT_RY_CD"]
HAZARDS = {"heatwave": (7, 8, 9), "cold_wave": (11, 12)}
SPLITS = ("train", "validation", "test")
RATIOS = np.array([0.70, 0.15, 0.15])


def log(message: str) -> None:
    print(f"\n=== {message} ===", flush=True)


def build_components(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make Monday weeks, per-region contiguous event episodes, and union groups."""
    date_table = pd.DataFrame({"date": sorted(df["date"].unique())})
    date_table["week"] = date_table["date"] - pd.to_timedelta(date_table["date"].dt.weekday, unit="D")
    weeks = sorted(date_table["week"].unique())
    week_index = {pd.Timestamp(w): i for i, w in enumerate(weeks)}
    parent = list(range(len(weeks)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    daily = df.groupby(["date", "MCT_SGG_CD"], as_index=False)[list(HAZARDS)].first()
    # Flags are duplicated over industry rows; reject inconsistent duplicates.
    distinct = df.groupby(["date", "MCT_SGG_CD"])[list(HAZARDS)].nunique(dropna=True)
    if (distinct > 1).any().any():
        raise ValueError("Event flag differs between industries for the same date-region.")

    episode_rows = []
    for hazard in HAZARDS:
        for region, group in daily.groupby("MCT_SGG_CD", sort=True):
            event_dates = list(group.loc[group[hazard].eq(1), "date"].sort_values())
            runs: list[list[pd.Timestamp]] = []
            for day in event_dates:
                day = pd.Timestamp(day)
                if not runs or (day - runs[-1][-1]).days != 1:
                    runs.append([day])
                else:
                    runs[-1].append(day)
            for run_no, run in enumerate(runs, start=1):
                episode_id = f"{hazard}|{region}|{run[0]:%Y%m%d}|{run_no:02d}"
                run_weeks = sorted({week_index[pd.Timestamp(day - pd.Timedelta(days=day.weekday()))] for day in run})
                for idx in run_weeks[1:]:
                    union(run_weeks[0], idx)
                episode_rows.append({
                    "episode_id": episode_id, "hazard": hazard, "region": region,
                    "start_date": run[0], "end_date": run[-1], "days": len(run),
                    "weeks": "|".join(pd.Timestamp(weeks[i]).strftime("%Y-%m-%d") for i in run_weeks),
                })

    root_to_component: dict[int, str] = {}
    for i in range(len(weeks)):
        root = find(i)
        root_to_component.setdefault(root, f"G{len(root_to_component)+1:02d}")
    week_table = pd.DataFrame({"week": [pd.Timestamp(w) for w in weeks]})
    week_table["component"] = [root_to_component[find(i)] for i in range(len(weeks))]
    date_table = date_table.merge(week_table, on="week", validate="many_to_one")
    return date_table, pd.DataFrame(episode_rows)


def assign_splits(date_table: pd.DataFrame, df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Deterministic randomized local search over indivisible week/episode components."""
    comp_stats = date_table.groupby("component").agg(weeks=("week", "nunique"), dates=("date", "nunique")).reset_index()
    daily_flags = df.groupby("date", as_index=False)[list(HAZARDS)].max()
    daily_flags = date_table[["date", "component"]].merge(daily_flags, on="date", validate="one_to_one")
    ev = daily_flags.groupby("component")[list(HAZARDS)].sum()
    comp_stats = comp_stats.set_index("component").join(ev).reset_index()
    comps = comp_stats["component"].tolist()
    if len(comps) < 3:
        raise ValueError("Fewer than three indivisible components; cannot create three splits.")
    dates = comp_stats["dates"].to_numpy(float)
    events = comp_stats[list(HAZARDS)].to_numpy(float)
    n_dates, n_events = dates.sum(), events.sum(axis=0)
    rng = np.random.default_rng(seed)

    def score(a: np.ndarray) -> float:
        sizes = np.array([dates[a == s].sum() for s in range(3)]) / n_dates
        event_sizes = np.array([events[a == s].sum(axis=0) for s in range(3)])
        event_shares = event_sizes / np.maximum(n_events, 1)[None, :]
        # Strongly prioritize presence of both hazards in every split.
        missing = int((event_sizes == 0).sum())
        return float(4.0 * np.square(sizes - RATIOS).sum()
                     + 2.0 * np.square(event_shares - RATIOS[:, None]).sum()
                     + 100.0 * missing)

    best_a, best_score = None, float("inf")
    # Randomized greedy seeds followed by single-component local moves.
    for restart in range(2500):
        order = np.arange(len(comps))
        rng.shuffle(order)
        a = np.full(len(comps), -1, dtype=int)
        for i in order:
            candidates = []
            for s in range(3):
                a[i] = s
                candidates.append(score(a))
            a[i] = int(np.argmin(candidates))
            if restart % 3 == 1 and rng.random() < 0.12:
                a[i] = int(rng.integers(3))
        # Ensure initial assignment covers each hazard in every split if possible.
        for _ in range(100):
            current = score(a)
            if current < 0.2:
                break
            improved = False
            for i in rng.permutation(len(comps)):
                old = a[i]
                for new in rng.permutation(3):
                    if new == old:
                        continue
                    a[i] = new
                    value = score(a)
                    if value + 1e-12 < current:
                        current, old, improved = value, new, True
                        break
                    a[i] = old
                if improved:
                    break
            if not improved:
                break
        value = score(a)
        if value < best_score:
            best_a, best_score = a.copy(), value

    assert best_a is not None
    split_names = np.array(SPLITS)
    comp_stats["split"] = split_names[best_a]
    if (comp_stats.groupby("split")[list(HAZARDS)].sum() == 0).any().any():
        raise ValueError("Could not place both hazards in all splits without splitting groups.")
    return date_table.merge(comp_stats[["component", "split"]], on="component", validate="many_to_one")


def audit_split(df: pd.DataFrame, date_table: pd.DataFrame, episodes: pd.DataFrame) -> None:
    assigned = df.merge(date_table[["date", "week", "component", "split"]], on="date", validate="many_to_one")
    for field in ("date", "week", "component"):
        counts = assigned.groupby(field)["split"].nunique()
        if counts.max() != 1:
            raise ValueError(f"Split leakage: {field} assigned to multiple splits.")
    # All dates making up every event episode must have one split.
    for row in episodes.itertuples(index=False):
        dates = pd.date_range(row.start_date, row.end_date, freq="D")
        hit = assigned.loc[assigned["date"].isin(dates) & assigned["MCT_SGG_CD"].eq(row.region)]
        if hit["split"].nunique() != 1:
            raise ValueError(f"Split leakage in episode {row.episode_id}.")
    sizes = assigned.drop_duplicates("date").groupby("split").size().reindex(SPLITS, fill_value=0)
    log("5-6. Week + Episode split and leakage audit")
    print("dates by split:", sizes.to_dict(), "shares:", (sizes / sizes.sum()).round(4).to_dict())
    print("week counts:", date_table.groupby("split")["week"].nunique().reindex(SPLITS).to_dict())
    print("episode counts:", episodes.merge(date_table[["date", "split"]], left_on="start_date", right_on="date", how="left").groupby("split").size().reindex(SPLITS, fill_value=0).to_dict())
    print("event region-days by split:", assigned.drop_duplicates(["date", "MCT_SGG_CD"]).groupby("split")[list(HAZARDS)].apply(lambda x: x.eq(1).sum()).reindex(SPLITS).to_dict())
    print("date/week/component/episode intersections: none")


def add_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    out = df.copy()
    baselines: dict[str, pd.DataFrame] = {}
    out["month"] = out["date"].dt.month.astype("int8")
    out["weekday"] = out["date"].dt.weekday.astype("int8")
    for hazard, months in HAZARDS.items():
        eligible = out["month"].isin(months) & out[hazard].eq(0) & out["split"].eq("train")
        base = out.loc[eligible].groupby(KEY, as_index=False)["TS_AT"].mean().rename(columns={"TS_AT": "baseline"})
        base["baseline_source_split"] = "train"
        baselines[hazard] = base
        target = out["month"].isin(months) & out[hazard].eq(1)
        out = out.merge(base[KEY + ["baseline"]].rename(columns={"baseline": f"baseline_{hazard}"}), on=KEY, how="left", validate="many_to_one")
        base_col = f"baseline_{hazard}"
        bad = target & (out[base_col].isna() | out[base_col].le(0))
        if bad.any():
            examples = out.loc[bad, KEY].drop_duplicates().head().to_dict("records")
            raise ValueError(f"No positive Train-only baseline for {hazard}; examples={examples}")
        out[f"Y1_{hazard}"] = np.where(target, 100.0 * (out["TS_AT"] - out[base_col]) / out[base_col], np.nan)
        if not base["baseline_source_split"].eq("train").all():
            raise ValueError("Baseline source split must be Train only.")
        if not out.loc[eligible, f"Y1_{hazard}"].isna().all():
            # Only event rows carry targets; baseline general days are not modeled.
            raise ValueError("Unexpected targets assigned to baseline rows.")
        print(f"{hazard}: Train general-day baseline groups={len(base)}, eligible rows={int(eligible.sum())}, target rows={int(target.sum())}")
    return out, baselines


def add_calendar_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Date keyed lag and strict complete-calendar rolling means through t-1."""
    out = df.copy()
    full_days = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    feature_rows = []
    for key, group in df.groupby(KEY, sort=False):
        series = group.set_index("date")["TS_AT"].sort_index().reindex(full_days)
        shifted = series.shift(1)
        rec = pd.DataFrame({"date": full_days})
        rec["lag1"] = series.reindex(full_days - pd.Timedelta(days=1)).to_numpy()
        rec["lag7"] = series.reindex(full_days - pd.Timedelta(days=7)).to_numpy()
        rec["rolling7"] = shifted.rolling(7, min_periods=7).mean().to_numpy()
        rec["rolling28"] = shifted.rolling(28, min_periods=28).mean().to_numpy()
        rec[KEY[0]], rec[KEY[1]] = key
        feature_rows.append(rec)
    calendar = pd.concat(feature_rows, ignore_index=True)
    out = out.merge(calendar, on=["date", *KEY], how="left", validate="one_to_one")

    # Independent date-key audit of lag values and rolling windows.
    lookup = df.set_index([*KEY, "date"])["TS_AT"]
    for idx, row in out.iterrows():
        key = (row[KEY[0]], row[KEY[1]])
        day = row["date"]
        for col, offset in (("lag1", 1), ("lag7", 7)):
            src = day - pd.Timedelta(days=offset)
            expected = lookup.get((*key, src), np.nan)
            actual = row[col]
            if pd.isna(expected):
                if pd.notna(actual):
                    raise ValueError(f"{col} should be missing at row {idx}.")
            elif pd.isna(actual) or not np.isclose(actual, expected):
                raise ValueError(f"{col} date-key mismatch at row {idx}.")
    # Independently reconstruct both windows from keyed source dates. A real
    # zero remains a valid observation; a missing date/value invalidates the window.
    for key, group in out.groupby(KEY, sort=False):
        history = df.loc[(df[KEY[0]].eq(key[0])) & (df[KEY[1]].eq(key[1])), ["date", "TS_AT"]]
        sales_by_date = dict(zip(history["date"], history["TS_AT"]))
        for row in group.itertuples(index=False):
            day = row.date
            for width, column in ((7, "rolling7"), (28, "rolling28")):
                source_dates = pd.date_range(day - pd.Timedelta(days=width), day - pd.Timedelta(days=1), freq="D")
                values = [sales_by_date.get(source_day, np.nan) for source_day in source_dates]
                actual = getattr(row, column)
                expected = float(np.mean(values)) if len(values) == width and np.isfinite(values).all() else np.nan
                if pd.isna(expected):
                    if pd.notna(actual):
                        raise ValueError(f"{column} must remain missing unless every prior calendar day exists ({key}, {day.date()}).")
                elif pd.isna(actual) or not np.isclose(actual, expected):
                    raise ValueError(f"{column} mismatch against explicit t-{width}..t-1 lookup ({key}, {day.date()}).")
    return out, ["lag1", "lag7", "rolling7", "rolling28"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    log("1-2. Input schema, duplicate keys, missingness")
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    required = {"TA_YMD", "MCT_SGG_CD", "MCT_RY_CD", "TS_AT", "heatwave", "cold_wave", "avg_temp", "min_temp", "max_temp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    print("source:", args.input, "rows:", len(df), "columns:", len(df.columns))
    print("key duplicates:", int(df.duplicated(["TA_YMD", *KEY]).sum()))
    if df.duplicated(["TA_YMD", *KEY]).any():
        raise ValueError("Duplicate date-region-industry key.")
    df["date"] = pd.to_datetime(df["TA_YMD"].astype(str), format="%Y%m%d", errors="raise")
    df["MCT_SGG_CD"] = df["MCT_SGG_CD"].astype(str)
    df["MCT_RY_CD"] = df["MCT_RY_CD"].astype(str)
    if df[KEY + ["date", "TS_AT"]].isna().any().any():
        raise ValueError("Null key/date/sales values found.")
    dates_per_series = df.groupby(KEY)["date"].nunique()
    full_n = (df["date"].max() - df["date"].min()).days + 1
    print("date range:", df.date.min().date(), "..", df.date.max().date(), "unique dates:", df.date.nunique())
    print("rows/date range:", int(df.groupby("date").size().min()), "..", int(df.groupby("date").size().max()))
    print("series with missing calendar dates:", int((dates_per_series < full_n).sum()), "/", len(dates_per_series))
    print("event missing counts:", df[list(HAZARDS)].isna().sum().to_dict(), "event rows:", {h: int(df[h].eq(1).sum()) for h in HAZARDS})
    # Heavy rain and precipitation fields are deliberately never selected below.

    log("3-4. Monday week and contiguous regional episodes")
    date_table, episodes = build_components(df)
    print("weeks:", date_table.week.nunique(), "episodes:", len(episodes), "indivisible components:", date_table.component.nunique())
    print(episodes[["episode_id", "start_date", "end_date", "days"]].to_string(index=False))

    date_table = assign_splits(date_table, df, args.seed)
    audit_split(df, date_table, episodes)
    df = df.merge(date_table[["date", "week", "component", "split"]], on="date", validate="many_to_one")
    df, baselines = add_targets(df)
    log("7-8. Train-only seasonal baseline and Y1")
    print("Baseline candidates restricted to split=train and event flag exactly 0; NA flags excluded.")

    df, lag_features = add_calendar_features(df)
    log("9-10. Calendar-keyed lag/rolling and feature leakage audit")
    print("lag1 source: t-1; lag7 source: t-7; rolling7: t-7..t-1; rolling28: t-28..t-1")
    print("rolling minimum observed calendar days: exactly 7/7 and 28/28; any missing day leaves feature NA.")
    print("No lag/rolling values are zero-filled. Independent exact-date lag audit passed.")
    print("Model pipeline: numeric missing values imputed with Train-event medians; categories with Train-event mode.")

    # Models use event-day observations only, separately for each hazard.
    numeric = ["avg_temp", "min_temp", "max_temp", "month", "weekday", *lag_features]
    categorical = KEY
    forbidden = {"TS_AT", "USE_CNT", "avg_price", "heavy_rain", "rainfall", "Y"}
    features = numeric + categorical
    if forbidden.intersection(features):
        raise ValueError("Forbidden outcome/rainfall field selected as feature.")
    model_specs = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.8, random_state=args.seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.04, max_depth=2, loss="huber", random_state=args.seed),
    }
    metric_rows, importance_rows, subgroup_rows = [], [], []
    for hazard in HAZARDS:
        log(f"11-14. {hazard}: train, validation/test, interpretation")
        target_col = f"Y1_{hazard}"
        event = df[hazard].eq(1) & df[target_col].notna()
        work = df.loc[event].copy()
        if work.empty:
            raise ValueError(f"No target event rows for {hazard}.")
        train = work[work.split.eq("train")]
        if train.empty:
            raise ValueError(f"No training event rows for {hazard}.")
        preprocessor = ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ])
        for model_name, estimator in model_specs.items():
            pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
            pipe.fit(train[features], train[target_col])
            train_mean = float(train[target_col].mean())
            for split in SPLITS:
                part = work[work.split.eq(split)]
                if part.empty:
                    raise ValueError(f"No {hazard} event rows in {split}; refusing partial evaluation.")
                pred = pipe.predict(part[features]) if split != "train" else pipe.predict(train[features])
                actual = part[target_col].to_numpy()
                metric_rows.append({
                    "hazard": hazard, "model": model_name, "split": split, "n": len(part),
                    "MAE": mean_absolute_error(actual, pred),
                    "RMSE": mean_squared_error(actual, pred) ** 0.5,
                    "R2": r2_score(actual, pred) if len(actual) > 1 else np.nan,
                    "train_mean_baseline_MAE": mean_absolute_error(actual, np.full(len(part), train_mean)),
                    "train_mean_baseline_RMSE": mean_squared_error(actual, np.full(len(part), train_mean)) ** 0.5,
                    "train_mean_baseline_R2": r2_score(actual, np.full(len(part), train_mean)) if len(actual) > 1 else np.nan,
                })
            test = work[work.split.eq("test")]
            pi = permutation_importance(pipe, test[features], test[target_col], n_repeats=10, random_state=args.seed, scoring="neg_mean_absolute_error", n_jobs=-1)
            importance_rows.extend({"hazard": hazard, "model": model_name, "feature": f, "importance_mean": m, "importance_std": s} for f, m, s in zip(features, pi.importances_mean, pi.importances_std))
            if model_name == "Random Forest":
                pred_test = pipe.predict(test[features])
                subgroup = test[[*KEY, target_col]].copy()
                subgroup["prediction"] = pred_test
                agg = subgroup.groupby(KEY).agg(n=(target_col, "size"), observed_mean_Y1=(target_col, "mean"), predicted_mean_Y1=("prediction", "mean"), observed_sd_Y1=(target_col, "std")).reset_index()
                agg["hazard"] = hazard
                subgroup_rows.extend(agg.to_dict("records"))

    metrics = pd.DataFrame(metric_rows)
    importances = pd.DataFrame(importance_rows)
    subgroups = pd.DataFrame(subgroup_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    importances.to_csv(args.output_dir / "permutation_importance.csv", index=False, encoding="utf-8-sig")
    subgroups.to_csv(args.output_dir / "test_region_industry_response.csv", index=False, encoding="utf-8-sig")
    episodes.to_csv(args.output_dir / "episodes.csv", index=False, encoding="utf-8-sig")
    date_table.to_csv(args.output_dir / "split_by_date.csv", index=False, encoding="utf-8-sig")
    for hazard, base in baselines.items():
        base.to_csv(args.output_dir / f"baseline_{hazard}.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "run_config.json").write_text(json.dumps({
        "input": str(args.input), "seed": args.seed, "ratios": dict(zip(SPLITS, RATIOS.tolist())),
        "features": features, "excluded_features": sorted(forbidden),
        "rolling_min_periods": {"rolling7": 7, "rolling28": 28},
        "model_missing_policy": "Features remain NA at construction; numeric median and categorical most-frequent imputation fit on Train event rows only",
        "lag_availability_assumption": "Historical observed TS_AT through t-1 is available even when its date belongs to Validation/Test; retrospective sequential-day analysis",
        "baseline_rule": "Train split, seasonal months, event flag exactly 0; NA excluded",
        "split_group_rule": "Monday weeks unioned across contiguous regional hazard episodes",
        "models": list(model_specs),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log("Train/Validation/Test metrics")
    print(metrics[metrics.split.isin(["train", "validation", "test"])].round(4).to_string(index=False))
    log("Test permutation importance (top 10 per hazard/model)")
    print(importances.sort_values(["hazard", "model", "importance_mean"], ascending=[True, True, False]).groupby(["hazard", "model"]).head(10).round(4).to_string(index=False))
    log("Largest absolute observed subgroup responses in Test (descriptive, RF predictions alongside)")
    subgroups["abs_observed_mean"] = subgroups.observed_mean_Y1.abs()
    print(subgroups.sort_values("abs_observed_mean", ascending=False).groupby("hazard").head(10).drop(columns="abs_observed_mean").round(3).to_string(index=False))
    print("\nResults written to:", args.output_dir)


if __name__ == "__main__":
    main()
