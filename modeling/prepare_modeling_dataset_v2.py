from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "modeling" / "modeling_dataset_v1.csv"
DEFAULT_OUTPUT = ROOT / "modeling" / "modeling_dataset_v2_filtered.csv"
KEY = ["MCT_SGG_CD", "MCT_RY_CD"]
REQUIRED_COLUMNS = {"TA_YMD", *KEY, "TS_AT"}
DEFAULT_MIN_OCTOBER_DAYS = 18


def prepare_dataset(
    frame: pd.DataFrame,
    minimum_october_days: int = DEFAULT_MIN_OCTOBER_DAYS,
) -> pd.DataFrame:
    """Add October sales baselines and retain groups with enough October data."""
    if minimum_october_days < 1:
        raise ValueError("minimum_october_days must be at least 1.")

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    result = frame.copy()
    if result[["TA_YMD", *KEY]].isna().any().any():
        raise ValueError("Input contains null date, region, or industry keys.")
    if result.duplicated(["TA_YMD", *KEY]).any():
        raise ValueError("Input contains duplicate date-region-industry keys.")

    result["date"] = pd.to_datetime(
        result["TA_YMD"].astype(str), format="%Y%m%d", errors="raise"
    )
    result["TS_AT"] = pd.to_numeric(result["TS_AT"], errors="raise")
    if result["TS_AT"].isna().any():
        raise ValueError("Input contains missing card sales (TS_AT).")

    october_rows = result.loc[result["date"].dt.month.eq(10)]
    if october_rows.empty:
        raise ValueError("Input has no October rows for the seasonal baseline.")

    baselines = october_rows.groupby(KEY, as_index=False).agg(
        october_mean=("TS_AT", "mean"),
        october_n=("TS_AT", "count"),
    )
    if (baselines["october_mean"] <= 0).any():
        raise ValueError("October baseline must be positive for every group.")

    result = result.merge(baselines, on=KEY, how="left", validate="many_to_one")
    eligible = result["october_n"].ge(minimum_october_days)
    result = result.loc[eligible].copy()
    if result.empty:
        raise ValueError(
            "No rows remain after applying the minimum October observation rule."
        )

    result["Y"] = 100.0 * (
        result["TS_AT"] - result["october_mean"]
    ) / result["october_mean"]
    result = result.drop(columns="date")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add October sales baselines and filter groups with too few October "
            "observations."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--minimum-october-days",
        type=int,
        default=DEFAULT_MIN_OCTOBER_DAYS,
        help="Minimum non-missing October TS_AT observations per region-industry.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input dataset not found: {args.input}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output: {args.output}; "
            "pass --overwrite to replace it."
        )

    source = pd.read_csv(args.input, encoding="utf-8-sig")
    result = prepare_dataset(source, args.minimum_october_days)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")

    source_groups = source[KEY].drop_duplicates().shape[0]
    result_groups = result[KEY].drop_duplicates().shape[0]
    print(f"Input rows: {len(source):,}; region-industry groups: {source_groups:,}")
    print(
        f"Output rows: {len(result):,}; retained groups: {result_groups:,}; "
        f"minimum October days: {args.minimum_october_days}"
    )
    print(f"Saved: {args.output}")
    print(
        "Y is relative to the full October mean for each region-industry. "
        "Model scripts 04-07 calculate their own Train-only Y1 target."
    )


if __name__ == "__main__":
    main()
