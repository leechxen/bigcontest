from __future__ import annotations

"""Diagnostics for the established Gradient Boosting + Feature Set B model."""

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "modeling" / "modeling_dataset_v2_filtered.csv"
COMPARISON_DIR = ROOT / "modeling" / "weather_model_comparison_extended"
OUTPUT_DIR = ROOT / "modeling" / "analysis_gb_feature_b"
PREDICTION_SOURCE = COMPARISON_DIR / "predictions_extended.csv"
EPISODE_SOURCE = COMPARISON_DIR / "episodes.csv"
KEY = ["MCT_SGG_CD", "MCT_RY_CD"]
HAZARDS = {"heatwave": (7, 8, 9), "cold_wave": (11, 12)}
NUMERIC = ["avg_temp", "min_temp", "max_temp", "lag1", "lag7", "rolling7", "rolling28"]
CATEGORICAL = ["MCT_SGG_CD", "MCT_RY_CD"]
FEATURES = NUMERIC + CATEGORICAL
FEATURE_GROUPS = {
    "weather": ["avg_temp", "min_temp", "max_temp"],
    "lag": ["lag1", "lag7"],
    "rolling": ["rolling7", "rolling28"],
    "region": ["MCT_SGG_CD"],
    "industry": ["MCT_RY_CD"],
}
SEED = 20261003


def load_base():
    path = Path(__file__).with_name("04_train_weather_impact_models.py")
    spec = importlib.util.spec_from_file_location("weather_base_analysis", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared utilities from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "n": int(len(y)), "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else np.nan,
    }


def make_pipeline() -> Pipeline:
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ])
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.04, max_depth=2, loss="huber", random_state=SEED)
    return Pipeline([("prep", prep), ("model", model)])


def grouped_permutation(pipe: Pipeline, x: pd.DataFrame, y: np.ndarray, group_map: dict[str, list[str]],
                        n_repeats: int, seed: int, hazard: str, sample: str) -> list[dict]:
    base_mae = mean_absolute_error(y, pipe.predict(x))
    rng = np.random.default_rng(seed)
    output = []
    for group, cols in group_map.items():
        values = []
        for _ in range(n_repeats):
            order = rng.permutation(len(x))
            shuffled = x.copy()
            shuffled.loc[:, cols] = x[cols].to_numpy()[order]
            values.append(mean_absolute_error(y, pipe.predict(shuffled)) - base_mae)
        output.append({
            "disaster_type": hazard, "sample": sample, "feature_group": group,
            "n_rows": len(x), "baseline_mae": base_mae,
            "importance_mean_mae_increase": float(np.mean(values)),
            "importance_sd_across_permutations": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "permutation_repeats": n_repeats,
        })
    return output


def single_feature_permutation(pipe: Pipeline, x: pd.DataFrame, y: np.ndarray, hazard: str,
                               sample: str, n_repeats: int = 30) -> list[dict]:
    base_mae = mean_absolute_error(y, pipe.predict(x))
    rng = np.random.default_rng(SEED + 51)
    output = []
    for feature in FEATURES:
        values = []
        for _ in range(n_repeats):
            order = rng.permutation(len(x))
            shuffled = x.copy()
            shuffled[feature] = x[feature].to_numpy()[order]
            values.append(mean_absolute_error(y, pipe.predict(shuffled)) - base_mae)
        output.append({
            "disaster_type": hazard, "sample": sample, "feature": feature,
            "feature_group": next(g for g, cols in FEATURE_GROUPS.items() if feature in cols),
            "n_rows": len(x), "baseline_mae": base_mae,
            "importance_mean_mae_increase": float(np.mean(values)),
            "importance_sd_across_permutations": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "permutation_repeats": n_repeats,
        })
    return output


def make_episode_map(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ep in episodes.itertuples(index=False):
        for day in pd.date_range(ep.start_date, ep.end_date, freq="D"):
            rows.append({"disaster_type": ep.hazard, "region": ep.region, "date": day, "episode_id": ep.episode_id})
    mapping = pd.DataFrame(rows)
    if mapping.duplicated(["disaster_type", "region", "date"]).any():
        raise ValueError("Overlapping episode definitions found for a disaster-region-date.")
    return mapping


def summarize_episodes(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dated = pred.groupby(["disaster_type", "episode_id", "split", "date"], as_index=False).agg(
        date_actual_mean=("actual_y1", "mean"), date_predicted_mean=("predicted_y1", "mean"),
        industries_observed=("industry", "nunique"), rows=("actual_y1", "size"),
    )
    episode_day_means = dated.groupby(["disaster_type", "episode_id", "split"], as_index=False).agg(
        event_days=("date", "nunique"), actual_y1_day_equal_mean=("date_actual_mean", "mean"),
        predicted_y1_day_equal_mean=("date_predicted_mean", "mean"),
        actual_y1_day_equal_median=("date_actual_mean", "median"),
        predicted_y1_day_equal_median=("date_predicted_mean", "median"),
        mean_absolute_daily_error=("date_actual_mean", lambda x: np.nan),
        industries_max_day=("industries_observed", "max"),
        rows=("rows", "sum"),
    )
    # Add row-weighted means and error after retaining episode/day equal weights above.
    row_stats = pred.groupby(["disaster_type", "episode_id", "split"], as_index=False).agg(
        actual_y1_row_mean=("actual_y1", "mean"), predicted_y1_row_mean=("predicted_y1", "mean"),
        actual_y1_row_median=("actual_y1", "median"), actual_y1_row_sd=("actual_y1", "std"),
        industry_count=("industry", "nunique"),
        industries_negative_mean=("actual_y1", lambda x: int(x.groupby(pred.loc[x.index, "industry"]).mean().lt(0).sum())),
        industries_positive_mean=("actual_y1", lambda x: int(x.groupby(pred.loc[x.index, "industry"]).mean().gt(0).sum())),
    )
    result = episode_day_means.merge(row_stats, on=["disaster_type", "episode_id", "split"], validate="one_to_one")
    result["episode_error_y1"] = result["predicted_y1_day_equal_mean"] - result["actual_y1_day_equal_mean"]
    result["episode_absolute_error_y1"] = result["episode_error_y1"].abs()
    result.drop(columns=["mean_absolute_daily_error"], inplace=True)
    result = result.sort_values(["disaster_type", "split", "episode_id"])
    return result, dated


def distribution_table(work: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (hazard, split), group in work.groupby(["disaster_type", "split"]):
        y = group["actual_y1"]
        rows.append({
            "disaster_type": hazard, "split": split, "n_industry_rows": len(y),
            "n_episodes": group["episode_id"].nunique(), "n_event_dates": group["date"].nunique(),
            "mean": y.mean(), "median": y.median(), "sd": y.std(),
            "p01": y.quantile(.01), "p05": y.quantile(.05), "p10": y.quantile(.10),
            "p25": y.quantile(.25), "p75": y.quantile(.75), "p90": y.quantile(.90),
            "p95": y.quantile(.95), "p99": y.quantile(.99), "min": y.min(), "max": y.max(),
        })
    return pd.DataFrame(rows)


def aggregate_entity_episode(pred: pd.DataFrame, entity: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    within = pred.groupby(["disaster_type", "split", "episode_id", entity], as_index=False).agg(
        actual_mean_y1=("actual_y1", "mean"), predicted_mean_y1=("predicted_y1", "mean"),
        actual_median_y1=("actual_y1", "median"), actual_sd_y1=("actual_y1", "std"),
        observation_rows=("actual_y1", "size"), event_dates=("date", "nunique"),
    )
    within["error_y1"] = within["predicted_mean_y1"] - within["actual_mean_y1"]
    summary = within.groupby(["disaster_type", "split", entity]).agg(
        episode_count=("episode_id", "nunique"),
        mean_episode_actual_y1=("actual_mean_y1", "mean"),
        median_episode_actual_y1=("actual_mean_y1", "median"),
        sd_episode_actual_y1=("actual_mean_y1", "std"),
        p10_episode_actual_y1=("actual_mean_y1", lambda x: x.quantile(.10)),
        p90_episode_actual_y1=("actual_mean_y1", lambda x: x.quantile(.90)),
        mean_episode_predicted_y1=("predicted_mean_y1", "mean"),
        mean_episode_error_y1=("error_y1", "mean"),
        episodes_actual_negative=("actual_mean_y1", lambda x: int(x.lt(0).sum())),
        episodes_predicted_negative=("predicted_mean_y1", lambda x: int(x.lt(0).sum())),
        episodes_sign_agree=("actual_mean_y1", lambda x: 0),
    ).reset_index()
    # Explicitly calculate the count of same-sign episode means.
    signs = within.assign(sign_agrees=np.sign(within.actual_mean_y1) == np.sign(within.predicted_mean_y1)).groupby(["disaster_type", "split", entity])["sign_agrees"].sum()
    summary = summary.drop(columns="episodes_sign_agree").merge(signs.rename("episodes_sign_agree").reset_index(), on=["disaster_type", "split", entity], validate="one_to_one")
    return within, summary


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis folder: {OUTPUT_DIR}")
    shap_available = importlib.util.find_spec("shap") is not None
    if shap_available:
        import shap  # noqa: F401
    else:
        print("SHAP not installed; no installation attempted. Using permutation importance.")

    base = load_base()
    df = pd.read_csv(SOURCE, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["TA_YMD"].astype(str), format="%Y%m%d", errors="raise")
    df["MCT_SGG_CD"] = df["MCT_SGG_CD"].astype(str)
    df["MCT_RY_CD"] = df["MCT_RY_CD"].astype(str)
    date_table, episodes = base.build_components(df)
    date_table = base.assign_splits(date_table, df, SEED)
    base.audit_split(df, date_table, episodes)
    df = df.merge(date_table[["date", "week", "component", "split"]], on="date", validate="many_to_one")
    df, _ = base.add_targets(df)
    df, _ = base.add_calendar_features(df)

    episode_source = pd.read_csv(EPISODE_SOURCE, encoding="utf-8-sig", parse_dates=["start_date", "end_date"])
    episode_map = make_episode_map(episode_source)
    prediction_source = pd.read_csv(PREDICTION_SOURCE, encoding="utf-8-sig", parse_dates=["date"])
    saved = prediction_source.loc[
        prediction_source["model"].eq("Gradient Boosting")
        & prediction_source["feature_set"].eq("B_plus_past_sales")
    ].copy()
    if saved.empty:
        raise ValueError("Reference Gradient Boosting + Feature Set B predictions not found.")
    saved["split"] = saved["split"].astype(str)
    saved = saved.merge(episode_map, on=["disaster_type", "region", "date"], how="left", validate="many_to_one")
    if saved["episode_id"].isna().any():
        raise ValueError("Some reference event predictions do not map to an episode.")

    # Refit exactly the established Gradient Boosting + B pipeline and check it
    # against the saved prediction artifact before computing diagnostics.
    comparison_rows = []
    group_importance_rows, feature_importance_rows = [], []
    outlier_sensitivity_rows, outlier_detail_rows = [], []
    modeled_predictions = []
    training_distribution = {}
    for hazard in HAZARDS:
        ycol = f"Y1_{hazard}"
        work = df.loc[df[hazard].eq(1) & df[ycol].notna()].copy()
        train = work.loc[work["split"].eq("train")]
        pipe = make_pipeline()
        pipe.fit(train[FEATURES], train[ycol])
        for split in base.SPLITS:
            part = work.loc[work["split"].eq(split)].copy()
            pred = pipe.predict(part[FEATURES])
            observed = part[ycol].to_numpy()
            record = {"disaster_type": hazard, "split": split, **metrics(observed, pred)}
            comparison_rows.append(record)
            part["actual_y1"] = observed
            part["predicted_y1_refit"] = pred
            part["model_error_y1"] = pred - observed
            part["model_abs_error_y1"] = np.abs(pred - observed)
            modeled_predictions.append(part[["date", "MCT_SGG_CD", "MCT_RY_CD", "split", "actual_y1", "predicted_y1_refit", "model_error_y1", "model_abs_error_y1"]].assign(disaster_type=hazard))

        saved_h = saved.loc[saved.disaster_type.eq(hazard)].copy()
        test_refit = work.loc[work["split"].eq("test")].copy()
        test_refit["predicted_y1_refit"] = pipe.predict(test_refit[FEATURES])
        check = saved_h.loc[saved_h["split"].eq("test")].merge(
            test_refit[["date", *KEY, ycol, "predicted_y1_refit"]],
            left_on=["date", "region", "industry"], right_on=["date", *KEY], validate="one_to_one",
        )
        max_prediction_diff = float(np.abs(check["predicted_y1"] - check["predicted_y1_refit"]).max())
        if max_prediction_diff > 1e-7:
            raise ValueError(f"Reference predictions do not reproduce for {hazard}: max diff={max_prediction_diff}")

        tr_y = train[ycol].to_numpy()
        low, high = np.quantile(tr_y, [.01, .99])
        training_distribution[hazard] = {"train_p01": float(low), "train_p99": float(high), "train_n": len(train)}
        test = work.loc[work["split"].eq("test")].copy()
        test_x, test_y = test[FEATURES], test[ycol].to_numpy()
        test_pred = pipe.predict(test_x)
        in_range_mask = (test_y >= low) & (test_y <= high)
        sample_specs = [("test_all", np.ones(len(test), dtype=bool)), ("test_inside_train_p01_p99", in_range_mask)]
        for sample_name, mask in sample_specs:
            x_sub, y_sub = test_x.loc[mask], test_y[mask]
            if len(x_sub) == 0:
                continue
            group_importance_rows.extend(grouped_permutation(pipe, x_sub, y_sub, FEATURE_GROUPS, 50, SEED + (1 if hazard == "heatwave" else 2), hazard, sample_name))
            feature_importance_rows.extend(single_feature_permutation(pipe, x_sub, y_sub, hazard, sample_name, n_repeats=30))
            pred_sub = pipe.predict(x_sub)
            outlier_sensitivity_rows.append({
                "disaster_type": hazard, "sample": sample_name,
                **metrics(y_sub, pred_sub), "train_p01": low, "train_p99": high,
                "excluded_test_rows": int((~mask).sum()),
            })
        # Winsorized diagnostic clips both observed values and predictions to
        # training-only 1st/99th percentile bounds; it is not a new fitted model.
        clipped_y, clipped_pred = np.clip(test_y, low, high), np.clip(test_pred, low, high)
        outlier_sensitivity_rows.append({
            "disaster_type": hazard, "sample": "test_winsorized_train_p01_p99",
            **metrics(clipped_y, clipped_pred), "train_p01": low, "train_p99": high,
            "excluded_test_rows": 0,
        })
        detail = test[["date", *KEY, "split"]].copy()
        detail["disaster_type"] = hazard
        detail["actual_y1"] = test_y
        detail["predicted_y1"] = test_pred
        detail["error_y1"] = test_pred - test_y
        detail["abs_error_y1"] = np.abs(detail["error_y1"])
        detail["outside_train_p01_p99"] = (~in_range_mask)
        detail["train_p01"] = low
        detail["train_p99"] = high
        outlier_detail_rows.extend(detail.to_dict("records"))

    predictions = pd.concat(modeled_predictions, ignore_index=True)
    predictions = predictions.merge(episode_map, left_on=["disaster_type", "MCT_SGG_CD", "date"], right_on=["disaster_type", "region", "date"], how="left", validate="many_to_one").drop(columns="region")
    # Rebuilt model predictions must reproduce the saved predictions for every split.
    all_saved = saved.merge(
        predictions[["date", *KEY, "disaster_type", "split", "predicted_y1_refit"]],
        left_on=["date", "region", "industry", "disaster_type", "split"],
        right_on=["date", *KEY, "disaster_type", "split"], validate="one_to_one",
    )
    all_diff = float(np.abs(all_saved.predicted_y1 - all_saved.predicted_y1_refit).max())
    if all_diff > 1e-7:
        raise ValueError(f"Saved predictions do not reproduce across all splits: max diff={all_diff}")

    episodes_result, dated_episode_result = summarize_episodes(saved)
    industry_episode, industry_summary = aggregate_entity_episode(saved, "industry")
    region_episode, region_summary = aggregate_entity_episode(saved, "region")
    distributions = distribution_table(saved)
    comparison = pd.DataFrame(comparison_rows)

    OUTPUT_DIR.mkdir(parents=True)
    episodes_result.to_csv(OUTPUT_DIR / "episode_summary.csv", index=False, encoding="utf-8-sig")
    dated_episode_result.to_csv(OUTPUT_DIR / "episode_day_summary.csv", index=False, encoding="utf-8-sig")
    industry_episode.to_csv(OUTPUT_DIR / "industry_episode_comparison.csv", index=False, encoding="utf-8-sig")
    industry_summary.to_csv(OUTPUT_DIR / "industry_episode_summary.csv", index=False, encoding="utf-8-sig")
    region_episode.to_csv(OUTPUT_DIR / "region_episode_comparison.csv", index=False, encoding="utf-8-sig")
    region_summary.to_csv(OUTPUT_DIR / "region_episode_summary.csv", index=False, encoding="utf-8-sig")
    distributions.to_csv(OUTPUT_DIR / "y1_distribution_by_split.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(comparison_rows).to_csv(OUTPUT_DIR / "gb_feature_b_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(group_importance_rows).to_csv(OUTPUT_DIR / "feature_group_permutation_importance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(feature_importance_rows).to_csv(OUTPUT_DIR / "feature_permutation_importance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(outlier_sensitivity_rows).to_csv(OUTPUT_DIR / "outlier_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(outlier_detail_rows).to_csv(OUTPUT_DIR / "test_outlier_rows.csv", index=False, encoding="utf-8-sig")

    # Actual-versus-predicted scatter for each disaster and split.
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    colors = {"train": "#4C78A8", "validation": "#F2A541", "test": "#D45087"}
    for ax, hazard in zip(axes, HAZARDS):
        part = predictions[predictions.disaster_type.eq(hazard)]
        lo = min(part.actual_y1.min(), part.predicted_y1_refit.min())
        hi = max(part.actual_y1.max(), part.predicted_y1_refit.max())
        for split in base.SPLITS:
            sample = part[part.split.eq(split)]
            ax.scatter(sample.actual_y1, sample.predicted_y1_refit, s=10, alpha=.3, label=f"{split} (n={len(sample)})", color=colors[split])
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
        ax.set(title=hazard, xlabel="Actual Y1 (% points)", ylabel="Predicted Y1 (% points)")
        ax.legend(frameon=False)
    fig.suptitle("Gradient Boosting + Feature Set B: actual vs predicted")
    fig.savefig(OUTPUT_DIR / "actual_vs_predicted_scatter.png", dpi=180)
    plt.close(fig)

    # Episode means are date-equal-weighted so a longer episode does not become
    # many independent event counts. Show per-episode actual/predicted means.
    plot_ep = episodes_result.copy()
    plot_ep["label"] = plot_ep["episode_id"].str.replace("heatwave|", "HW·", regex=False).str.replace("cold_wave|", "CW·", regex=False)
    plot_ep = plot_ep.sort_values(["disaster_type", "split", "episode_id"])
    fig_h = max(7, .34 * len(plot_ep))
    fig, ax = plt.subplots(figsize=(13, fig_h), constrained_layout=True)
    yloc = np.arange(len(plot_ep))
    ax.scatter(plot_ep.actual_y1_day_equal_mean, yloc, label="Actual episode mean", marker="o", s=35)
    ax.scatter(plot_ep.predicted_y1_day_equal_mean, yloc, label="Predicted episode mean", marker="x", s=42)
    ax.axvline(0, color="grey", linewidth=1)
    ax.set_yticks(yloc, plot_ep["label"])
    ax.set_xlabel("Y1 (% points), event-day means averaged within episode")
    ax.set_title("Episode-level actual and predicted response")
    ax.legend(frameon=False)
    fig.savefig(OUTPUT_DIR / "episode_actual_predicted.png", dpi=180)
    plt.close(fig)

    group_imp = pd.DataFrame(group_importance_rows)
    full_imp = group_imp[group_imp["sample"].eq("test_all")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, hazard in zip(axes, HAZARDS):
        sample = full_imp[full_imp.disaster_type.eq(hazard)].sort_values("importance_mean_mae_increase")
        ax.barh(sample.feature_group, sample.importance_mean_mae_increase, xerr=sample.importance_sd_across_permutations, color="#4C78A8", alpha=.85)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set(title=hazard, xlabel="Test MAE increase after joint group permutation")
    fig.suptitle("Test permutation importance by feature group")
    fig.savefig(OUTPUT_DIR / "feature_group_importance.png", dpi=180)
    plt.close(fig)

    config = {
        "reference": "Gradient Boosting + B_plus_past_sales from the existing extended comparison",
        "source_csv_modified": False, "split_y1_baseline_episode_definitions_changed": False,
        "split_seed": SEED,
        "saved_prediction_reproduction_max_abs_difference": all_diff,
        "shap_available": shap_available,
        "shap_action": "Not run; SHAP is absent and was not installed. Permutation importance was used.",
        "permutation": {"group_repeats": 50, "single_feature_repeats": 30, "score": "increase in Test MAE after permutation; positive means prediction error increased"},
        "outlier_rule": "Train Y1 1st and 99th percentiles; report full Test, Test within these bounds, and a winsorized diagnostic (clip both actual/predicted). No refitting or trimming of training rows.",
        "episode_summary_rule": "Event-day mean across available industry rows, then equal-weight average across dates inside the existing regional hazard episode. Episodes, not industry rows, are the event units.",
        "interpretation_limits": [
            "Permutation importance is not causal and correlated features can share or mask importance.",
            "The Test split contains very few independent event region-days; permutation and episode diagnostics are unstable.",
            "Training predictions are in-sample and must not be interpreted as generalization performance.",
            "Observed same-day temperatures and historical sales imply retrospective impact estimation, not pre-event forecasting.",
        ],
        "train_y1_bounds": training_distribution,
    }
    (OUTPUT_DIR / "analysis_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved prediction reproduction max absolute difference:", all_diff)
    print("\nReference model metrics:")
    print(comparison.round(4).to_string(index=False))
    print("\nTest group importance:")
    print(full_imp[["disaster_type", "feature_group", "importance_mean_mae_increase", "importance_sd_across_permutations"]].round(3).to_string(index=False))
    print("\nTest episode summary:")
    print(episodes_result[episodes_result.split.eq("test")][["disaster_type", "episode_id", "event_days", "industry_count", "actual_y1_day_equal_mean", "predicted_y1_day_equal_mean", "episode_error_y1", "industries_negative_mean"]].round(2).to_string(index=False))
    print("\nOutlier sensitivity:")
    print(pd.DataFrame(outlier_sensitivity_rows).round(3).to_string(index=False))
    print("\nAnalysis outputs:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
