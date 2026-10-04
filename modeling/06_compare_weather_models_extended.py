from __future__ import annotations

"""Run the established comparison in a new output folder with optional boosters."""

import argparse
import importlib.metadata as metadata
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPARISON_SCRIPT = Path(__file__).with_name("05_compare_weather_models_features.py")
DEFAULT_INPUT = ROOT / "modeling" / "modeling_dataset_v2_filtered.csv"
DEFAULT_OUTPUT = ROOT / "modeling" / "weather_model_comparison_extended"
CORE_PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "joblib"]
BOOST_PACKAGES = ["xgboost", "lightgbm", "catboost"]


def get_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def import_check(name: str) -> tuple[bool, str | None]:
    try:
        module = __import__(name)
        return True, getattr(module, "__version__", get_version(name))
    except Exception as exc:
        return False, repr(exc)


def write_package_versions(path: Path) -> dict:
    packages = {}
    for name in ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "xgboost", "lightgbm", "catboost", "narwhals", "shap", "numba", "slicer", "tqdm", "cloudpickle", "llvmlite"]:
        module_name = {"scikit-learn": "sklearn"}.get(name, name)
        available, detail = import_check(module_name)
        packages[name] = {"distribution_version": get_version(name), "import_ok": available, "import_version_or_error": detail}
    try:
        import pip
        pip_version = pip.__version__
    except Exception:
        pip_version = None
    data = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "pip_version": pip_version,
        "packages": packages,
        "installation_actions": [
            {"package": "xgboost", "installed_version": packages["xgboost"]["distribution_version"], "method": "pip --no-deps --only-binary=:all:"},
            {"package": "lightgbm", "installed_version": packages["lightgbm"]["distribution_version"], "method": "pip --no-deps --only-binary=:all:"},
            {"package": "catboost", "installed_version": packages["catboost"]["distribution_version"], "method": "pip --no-deps --only-binary=:all:"},
            {"package": "narwhals", "installed_version": packages["narwhals"]["distribution_version"], "reason": "LightGBM 4.7.0 import required this missing dependency; installed without changing other packages"},
            {"package": "shap", "status": "not installed; not attempted", "reason": "Optional; supporting packages numba, slicer, tqdm, cloudpickle, llvmlite are absent, so installation would require expanding the dependency set. Permutation importance is used instead."},
        ],
        "core_package_versions_before_booster_install": {
            "numpy": "2.3.3", "pandas": "2.3.3", "scikit-learn": "1.7.2", "scipy": "1.16.2", "joblib": "1.5.2",
        },
        "core_package_versions_after_install": {name: packages[name]["distribution_version"] for name in CORE_PACKAGES},
        "core_dependencies_changed": False,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def add_stability_assessment(path: Path, config_path: Path) -> None:
    import pandas as pd

    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["train_to_validation_mae_change_pct"] = (frame["val_mae"] / frame["train_mae"] - 1.0) * 100.0
    frame["validation_to_test_mae_change_pct"] = (frame["test_mae"] / frame["val_mae"] - 1.0) * 100.0
    frame["train_to_test_mae_change_pct"] = (frame["test_mae"] / frame["train_mae"] - 1.0) * 100.0
    frame["train_to_validation_r2_drop"] = frame["train_r2"] - frame["val_r2"]
    frame["validation_to_test_r2_drop"] = frame["val_r2"] - frame["test_r2"]

    def assess(row) -> str:
        flags = []
        if row.train_mae < row.val_mae * 0.85 and row.train_mae < row.test_mae * 0.85 and row.train_r2 - max(row.val_r2, row.test_r2) >= 0.20:
            flags.append("train_only_fit")
        if row.test_mae > row.val_mae * 1.15 and row.val_r2 - row.test_r2 >= 0.15:
            flags.append("validation_to_test_drop")
        if row.val_mae > row.train_mae * 1.15 and row.train_r2 - row.val_r2 >= 0.15:
            flags.append("train_to_validation_drop")
        mae_span = (max(row.train_mae, row.val_mae, row.test_mae) / min(row.train_mae, row.val_mae, row.test_mae)) - 1.0
        r2_span = max(row.train_r2, row.val_r2, row.test_r2) - min(row.train_r2, row.val_r2, row.test_r2)
        if mae_span <= 0.20 and r2_span <= 0.25:
            flags.append("comparatively_stable_by_threshold")
        return "|".join(flags) if flags else "mixed_or_no_large_gap_by_threshold"

    frame["stability_assessment"] = frame.apply(assess, axis=1)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["stability_assessment_thresholds"] = {
        "train_only_fit": "Train MAE at least 15% lower than both validation and test, and Train R2 exceeds both by >=0.20",
        "validation_to_test_drop": "Test MAE > Validation MAE by 15% and Validation R2 exceeds Test R2 by >=0.15",
        "train_to_validation_drop": "Validation MAE > Train MAE by 15% and Train R2 exceeds Validation R2 by >=0.15",
        "comparatively_stable": "max/min MAE <= 1.20 and R2 range <= 0.25",
        "note": "Descriptive flags with explicit thresholds; review the numeric metrics rather than treating labels as model selection.",
    }
    config["shap_analysis"] = "Not run: SHAP and its supporting packages are absent. Permutation importance is included."
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutation-repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing results directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    versions = write_package_versions(args.output_dir / "package_versions.json")

    spec = importlib.util.spec_from_file_location("weather_feature_comparison", COMPARISON_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {COMPARISON_SCRIPT}")
    comparison = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = comparison
    spec.loader.exec_module(comparison)

    # Reuse the established processing, split, target, model and importance code.
    original_argv = sys.argv[:]
    sys.argv = [
        str(COMPARISON_SCRIPT), "--input", str(args.input), "--seed", str(args.seed),
        "--output-dir", str(args.output_dir), "--permutation-repeats", str(args.permutation_repeats),
    ]
    try:
        comparison.main()
    finally:
        sys.argv = original_argv

    names = {
        "model_comparison.csv": "model_comparison_extended.csv",
        "feature_importance.csv": "feature_importance_extended.csv",
        "predictions.csv": "predictions_extended.csv",
        "run_config.json": "run_config_extended.json",
    }
    for old_name, new_name in names.items():
        old_path = args.output_dir / old_name
        if not old_path.exists():
            raise FileNotFoundError(f"Expected comparison output was not created: {old_path}")
        old_path.replace(args.output_dir / new_name)
    add_stability_assessment(args.output_dir / names["model_comparison.csv"], args.output_dir / names["run_config.json"])

    config_path = args.output_dir / names["run_config.json"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["package_versions_file"] = "package_versions.json"
    config["package_versions"] = {name: versions["packages"][name]["distribution_version"] for name in ["xgboost", "lightgbm", "catboost"]}
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nExtended comparison outputs saved separately under:", args.output_dir)


if __name__ == "__main__":
    main()
