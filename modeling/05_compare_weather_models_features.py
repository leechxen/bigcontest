from __future__ import annotations

"""Compare model families and three feature sets for disaster-day Y1."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "modeling" / "modeling_dataset_v2_filtered.csv"
FLOW_DIR = ROOT / "flow_data"
OUTPUT_DIR = ROOT / "modeling" / "weather_model_comparison"
BASE_SCRIPT = Path(__file__).with_name("04_train_weather_impact_models.py")

# BLOCK_CD begins with the legacy municipal administrative code in these files.
# The remaining neighboring prefixes are intentionally not assigned to either region.
FLOW_PREFIX_TO_REGION = {"11230": "서울 강남구", "32010": "강원 춘천시"}
FLOW_DAY_COLS = {
    0: "FLOW_POP_CNT_MON", 1: "FLOW_POP_CNT_TUS", 2: "FLOW_POP_CNT_WED",
    3: "FLOW_POP_CNT_THU", 4: "FLOW_POP_CNT_FRI", 5: "FLOW_POP_CNT_SAT",
    6: "FLOW_POP_CNT_SUN",
}
FORBIDDEN_FEATURES = {"TS_AT", "USE_CNT", "avg_price", "Y", "heavy_rain", "rainfall"}


def load_base_module():
    spec = importlib.util.spec_from_file_location("weather_base_modeling", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared modeling utilities from {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_flow_weekday_feature() -> tuple[pd.DataFrame, dict]:
    files = sorted(FLOW_DIR.glob("flow_wkdy_pop_*.csv"))
    if not files:
        raise FileNotFoundError(f"No weekday flow data found under {FLOW_DIR}")
    usecols = ["STD_YM", "BLOCK_CD", *FLOW_DAY_COLS.values()]
    parts = []
    coverage = []
    unknown_prefixes: set[str] = set()
    for path in files:
        flow = pd.read_csv(path, sep="|", encoding="utf-8-sig", usecols=usecols,
                           dtype={"STD_YM": "string", "BLOCK_CD": "string"})
        flow["prefix"] = flow["BLOCK_CD"].str[:5]
        unknown_prefixes.update(flow.loc[~flow["prefix"].isin(FLOW_PREFIX_TO_REGION), "prefix"].dropna().unique())
        flow = flow.loc[flow["prefix"].isin(FLOW_PREFIX_TO_REGION)].copy()
        flow["region"] = flow["prefix"].map(FLOW_PREFIX_TO_REGION)
        for col in FLOW_DAY_COLS.values():
            flow[col] = pd.to_numeric(flow[col], errors="coerce")
        if flow[list(FLOW_DAY_COLS.values())].isna().any().any():
            raise ValueError(f"Missing/non-numeric weekday flow values in {path.name}")
        coverage.append({
            "file": path.name, "month": str(flow["STD_YM"].iloc[0]) if len(flow) else path.stem[-6:],
            "matched_grid_rows": len(flow), "matched_blocks": int(flow["BLOCK_CD"].nunique()),
            "prefixes": "|".join(sorted(flow["prefix"].unique())),
        })
        parts.append(flow[["STD_YM", "region", "BLOCK_CD", *FLOW_DAY_COLS.values()]])

    all_flow = pd.concat(parts, ignore_index=True)
    monthly = all_flow.groupby(["STD_YM", "region"], as_index=False)[list(FLOW_DAY_COLS.values())].sum()
    long = monthly.melt(id_vars=["STD_YM", "region"], var_name="flow_day", value_name="flow_pop_weekday")
    day_from_col = {col: day for day, col in FLOW_DAY_COLS.items()}
    long["weekday"] = long["flow_day"].map(day_from_col).astype("int8")
    long["month"] = long["STD_YM"].str[-2:].astype("int8")
    if long.duplicated(["region", "month", "weekday"]).any():
        raise ValueError("Weekday flow aggregation has duplicate region-month-weekday keys.")
    metadata = {
        "source_files": [p.name for p in files],
        "prefix_to_region": FLOW_PREFIX_TO_REGION,
        "ignored_block_prefixes": sorted(unknown_prefixes),
        "coverage": coverage,
        "definition": "Monthly weekday flow-population values summed over matching 50m BLOCK_CD cells; selected by date month and weekday. This is a monthly weekday profile, not a daily observed count.",
    }
    return long[["region", "month", "weekday", "flow_pop_weekday"]], metadata


def build_models(seed: int, optional: dict[str, bool]) -> tuple[dict[str, object], dict[str, str]]:
    models: dict[str, object] = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.8, random_state=seed, n_jobs=1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.04, max_depth=2, loss="huber", random_state=seed),
        "Extra Trees": ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.8, random_state=seed, n_jobs=1),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=2.0, random_state=seed),
    }
    import_errors = {}
    if optional["xgboost"]:
        try:
            from xgboost import XGBRegressor
            models["XGBoost"] = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, objective="reg:squarederror", random_state=seed, n_jobs=-1)
        except Exception as exc:
            import_errors["XGBoost"] = repr(exc)
    if optional["lightgbm"]:
        try:
            from lightgbm import LGBMRegressor
            models["LightGBM"] = LGBMRegressor(n_estimators=300, num_leaves=15, learning_rate=0.04, reg_lambda=5.0, verbosity=-1, random_state=seed, n_jobs=-1)
        except Exception as exc:
            import_errors["LightGBM"] = repr(exc)
    if optional["catboost"]:
        try:
            from catboost import CatBoostRegressor
            models["CatBoost"] = CatBoostRegressor(iterations=300, depth=6, learning_rate=0.04, loss_function="RMSE", verbose=False, random_seed=seed, thread_count=-1)
        except Exception as exc:
            import_errors["CatBoost"] = repr(exc)
    return models, import_errors


def get_feature_sets() -> dict[str, dict[str, list[str]]]:
    weather = ["avg_temp", "min_temp", "max_temp"]
    categorical = ["MCT_SGG_CD", "MCT_RY_CD"]
    history = ["lag1", "lag7", "rolling7", "rolling28"]
    return {
        "A_weather_region_industry": {"numeric": weather, "categorical": categorical},
        "B_plus_past_sales": {"numeric": weather + history, "categorical": categorical},
        "C_plus_flow_population": {"numeric": weather + history + ["flow_pop_weekday"], "categorical": categorical},
    }


def metrics_for(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    return (mean_absolute_error(y, pred), mean_squared_error(y, pred) ** 0.5,
            r2_score(y, pred) if len(y) > 1 else np.nan)


def feature_group(feature: str) -> str:
    if feature in {"avg_temp", "min_temp", "max_temp"}:
        return "weather"
    if feature in {"lag1", "lag7", "rolling7", "rolling28"}:
        return "past_sales"
    if feature == "MCT_RY_CD":
        return "industry"
    if feature == "MCT_SGG_CD":
        return "region"
    if feature == "flow_pop_weekday":
        return "flow_population"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    args = parser.parse_args()
    if args.permutation_repeats < 1:
        raise ValueError("--permutation-repeats must be at least 1")

    base = load_base_module()
    optional = {name: importlib.util.find_spec(name) is not None for name in ("xgboost", "lightgbm", "catboost", "shap")}
    print("Optional packages:", optional)
    if not optional["xgboost"]:
        print("XGBoost skipped: package is not installed.")
    if not optional["lightgbm"]:
        print("LightGBM skipped: package is not installed.")
    if not optional["catboost"]:
        print("CatBoost skipped: package is not installed.")
    if not optional["shap"]:
        print("SHAP skipped: package is not installed; permutation importance will be used.")

    print("\n=== 1-2. Read-only source checks ===")
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    required = {"TA_YMD", "MCT_SGG_CD", "MCT_RY_CD", "TS_AT", "heatwave", "cold_wave", "avg_temp", "min_temp", "max_temp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required source columns missing: {sorted(missing)}")
    if df.duplicated(["TA_YMD", *base.KEY]).any():
        raise ValueError("Duplicate date-region-industry rows found.")
    df["date"] = pd.to_datetime(df["TA_YMD"].astype(str), format="%Y%m%d", errors="raise")
    df["MCT_SGG_CD"] = df["MCT_SGG_CD"].astype(str)
    df["MCT_RY_CD"] = df["MCT_RY_CD"].astype(str)
    print(f"source rows={len(df):,}; date range={df.date.min().date()}..{df.date.max().date()}; key duplicates=0")

    print("\n=== 3-6. Shared week/episode split and leakage checks ===")
    date_table, episodes = base.build_components(df)
    date_table = base.assign_splits(date_table, df, args.seed)
    base.audit_split(df, date_table, episodes)
    df = df.merge(date_table[["date", "week", "component", "split"]], on="date", validate="many_to_one")

    print("\n=== Feature Set C. Aggregate monthly weekday flow profile ===")
    flow, flow_metadata = load_flow_weekday_feature()
    df["month"] = df["date"].dt.month.astype("int8")
    df["weekday"] = df["date"].dt.weekday.astype("int8")
    df = df.merge(flow, left_on=["MCT_SGG_CD", "month", "weekday"], right_on=["region", "month", "weekday"], how="left", validate="many_to_one")
    unmatched_flow = int(df["flow_pop_weekday"].isna().sum())
    print("flow rows by month/region:", flow.groupby(["month", "region"]).size().to_dict())
    print("model rows without flow value:", unmatched_flow, "/", len(df))
    if unmatched_flow:
        print("Feature Set C will preserve these missing values for Train-only imputation.")
    df.drop(columns=["region"], inplace=True)

    print("\n=== 7-10. Train-only baseline, Y1, and pre-date features ===")
    df, baselines = base.add_targets(df)
    df, history_features = base.add_calendar_features(df)
    print("Y1 baselines use only Train rows where seasonal event flag == 0; NA event flags are excluded.")
    print("Lag/rolling date audit passed: lag1=t-1, lag7=t-7, rolling7=t-7..t-1, rolling28=t-28..t-1.")
    print("Current TS_AT/USE_CNT/avg_price, Y, and all rainfall fields are excluded from predictors.")

    feature_sets = get_feature_sets()
    for name, spec in feature_sets.items():
        all_features = spec["numeric"] + spec["categorical"]
        forbidden = FORBIDDEN_FEATURES.intersection(all_features)
        if forbidden:
            raise ValueError(f"Forbidden feature(s) in {name}: {sorted(forbidden)}")
        absent = set(all_features) - set(df.columns)
        if absent:
            raise ValueError(f"Feature(s) missing for {name}: {sorted(absent)}")
    print("Feature sets:", {k: v["numeric"] + v["categorical"] for k, v in feature_sets.items()})

    models, optional_import_errors = build_models(args.seed, optional)
    metric_rows, prediction_parts, importance_rows = [], [], []
    for hazard in base.HAZARDS:
        target_col = f"Y1_{hazard}"
        event_mask = df[hazard].eq(1) & df[target_col].notna()
        work = df.loc[event_mask].copy()
        print(f"\n=== Model comparison: disaster={hazard}; event rows={len(work):,}; split rows="
              f"{work.groupby('split').size().reindex(base.SPLITS).to_dict()} ===")
        for feature_set, spec in feature_sets.items():
            numeric, categorical = spec["numeric"], spec["categorical"]
            features = numeric + categorical
            train = work.loc[work["split"].eq("train")]
            if train.empty:
                raise ValueError(f"No Train event rows for {hazard}.")
            preprocess = ColumnTransformer([
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]), numeric),
                ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
            ])
            for model_name, estimator_template in models.items():
                # Every model and feature set receives the identical split labels.
                from sklearn.base import clone
                pipeline = Pipeline([("prep", clone(preprocess)), ("model", clone(estimator_template))])
                pipeline.fit(train[features], train[target_col])
                for split in base.SPLITS:
                    part = work.loc[work["split"].eq(split)]
                    if part.empty:
                        raise ValueError(f"No {hazard} event rows in split={split}.")
                    prediction = pipeline.predict(part[features])
                    y = part[target_col].to_numpy()
                    mae, rmse, r2 = metrics_for(y, prediction)
                    train_mean = float(train[target_col].mean())
                    base_mae, base_rmse, base_r2 = metrics_for(y, np.full(len(y), train_mean))
                    metric_rows.append({
                        "disaster_type": hazard, "model": model_name, "feature_set": feature_set,
                        "split": split, "n": len(part), "mae": mae, "rmse": rmse, "r2": r2,
                        "train_mean_baseline_mae": base_mae,
                        "train_mean_baseline_rmse": base_rmse,
                        "train_mean_baseline_r2": base_r2,
                    })
                    prediction_parts.append(pd.DataFrame({
                        "date": part["date"].dt.strftime("%Y-%m-%d").to_numpy(),
                        "region": part["MCT_SGG_CD"].to_numpy(), "industry": part["MCT_RY_CD"].to_numpy(),
                        "disaster_type": hazard, "actual_y1": y, "predicted_y1": prediction,
                        "model": model_name, "feature_set": feature_set, "split": split,
                    }))

                test = work.loc[work["split"].eq("test")]
                perm = permutation_importance(
                    pipeline, test[features], test[target_col], n_repeats=args.permutation_repeats,
                    random_state=args.seed, scoring="neg_mean_absolute_error", n_jobs=1,
                )
                importance_rows.extend({
                    "disaster_type": hazard, "model": model_name, "feature_set": feature_set,
                    "feature": feature, "feature_group": feature_group(feature),
                    "importance_mean": mean, "importance_std": std,
                    "n_test_rows": len(test), "permutation_repeats": args.permutation_repeats,
                } for feature, mean, std in zip(features, perm.importances_mean, perm.importances_std))
                print(f"  completed {feature_set} | {model_name}")

    long_metrics = pd.DataFrame(metric_rows)
    comparison_rows = []
    for (hazard, model, feature_set), group in long_metrics.groupby(["disaster_type", "model", "feature_set"]):
        by_split = group.set_index("split")
        row = {"disaster_type": hazard, "model": model, "feature_set": feature_set}
        for split, prefix in (("train", "train"), ("validation", "val"), ("test", "test")):
            for metric in ("mae", "rmse", "r2"):
                row[f"{prefix}_{metric}"] = by_split.loc[split, metric]
            row[f"{prefix}_n"] = int(by_split.loc[split, "n"])
            for metric in ("mae", "rmse", "r2"):
                row[f"{prefix}_train_mean_baseline_{metric}"] = by_split.loc[split, f"train_mean_baseline_{metric}"]
        row["test_minus_train_mae"] = row["test_mae"] - row["train_mae"]
        row["test_minus_validation_mae"] = row["test_mae"] - row["val_mae"]
        row["train_minus_test_r2"] = row["train_r2"] - row["test_r2"]
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows).sort_values(["disaster_type", "feature_set", "test_mae"])
    predictions = pd.concat(prediction_parts, ignore_index=True)
    importances = pd.DataFrame(importance_rows).sort_values(["disaster_type", "feature_set", "model", "importance_mean"], ascending=[True, True, True, False])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    importances.to_csv(args.output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(args.output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    episodes.to_csv(args.output_dir / "episodes.csv", index=False, encoding="utf-8-sig")
    date_table.to_csv(args.output_dir / "split_by_date.csv", index=False, encoding="utf-8-sig")
    flow_metadata["matched_feature_rows"] = int(df["flow_pop_weekday"].notna().sum())
    flow_metadata["unmatched_feature_rows"] = unmatched_flow
    (args.output_dir / "flow_feature_metadata.json").write_text(json.dumps(flow_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    for hazard, baseline in baselines.items():
        baseline.to_csv(args.output_dir / f"baseline_{hazard}.csv", index=False, encoding="utf-8-sig")

    config = {
        "input": str(args.input), "seed": args.seed,
        "split": {"method": "Week + Episode-aware grouped stratified split", "ratios_target": {"train": 0.70, "validation": 0.15, "test": 0.15}, "shared_across_all_models_and_feature_sets": True},
        "baseline": "Train rows only, seasonal months, event flag exactly 0; missing event flags excluded",
        "target": "Y1 = 100 * (event-day TS_AT - Train seasonal normal-day baseline) / baseline",
        "feature_sets": {name: spec["numeric"] + spec["categorical"] for name, spec in feature_sets.items()},
        "excluded_features": sorted(FORBIDDEN_FEATURES),
        "lag_and_rolling": {"lag1": "calendar t-1", "lag7": "calendar t-7", "rolling7": "t-7..t-1; all 7 daily values required", "rolling28": "t-28..t-1; all 28 daily values required", "zero_fill": False},
        "missing_input_policy": "Keep missing during feature construction; numeric median and categorical most-frequent imputation are fit separately on each disaster/feature-set Train rows only.",
        "flow_feature": flow_metadata["definition"],
        "flow_region_link": "BLOCK_CD first-five-character prefix mapping 11230->Gangnam-gu and 32010->Chuncheon-si; no exact merchant/block spatial crosswalk is present.",
        "models_run": list(models), "optional_packages_installed": optional,
        "optional_model_import_errors": optional_import_errors,
        "permutation_importance_repeats": args.permutation_repeats,
        "shap": {"available": optional["shap"], "used": False if not optional["shap"] else "not configured in this run"},
        "interpretation_note": "Predictive explanation and permutation importance are not causal evidence. Y1 uses Train normal-day means; flow is a monthly weekday profile, not a daily observed flow count.",
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Validation/Test MAE leaders by disaster and feature set ===")
    for hazard in comparison.disaster_type.unique():
        for feature_set in feature_sets:
            subset = comparison.loc[comparison.disaster_type.eq(hazard) & comparison.feature_set.eq(feature_set)]
            val_best = subset.sort_values(["val_mae", "test_mae"]).iloc[0]
            test_best = subset.sort_values(["test_mae", "val_mae"]).iloc[0]
            print(f"{hazard} | {feature_set}: best validation={val_best.model} (MAE {val_best.val_mae:.2f}; test {val_best.test_mae:.2f}); "
                  f"best test={test_best.model} (MAE {test_best.test_mae:.2f}; validation {test_best.val_mae:.2f})")
    print("\n=== Feature group permutation importance: best 3 models per disaster/feature set ===")
    grouped_imp = importances.groupby(["disaster_type", "feature_set", "model", "feature_group"], as_index=False)["importance_mean"].sum()
    for hazard in base.HAZARDS:
        best = comparison.loc[comparison.disaster_type.eq(hazard)].sort_values(["val_mae", "test_mae"]).groupby("feature_set").head(3)
        selected = best[["model", "feature_set"]].drop_duplicates()
        selected["disaster_type"] = hazard
        view = grouped_imp.merge(selected, on=["disaster_type", "model", "feature_set"])
        print(f"{hazard}:\n", view.sort_values(["feature_set", "importance_mean"], ascending=[True, False]).round(4).to_string(index=False))
    print("\nOutput directory:", args.output_dir)


if __name__ == "__main__":
    main()
