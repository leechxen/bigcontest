from __future__ import annotations

import csv
import math
from collections import defaultdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARD_DIR = ROOT / "card_data"
OUTPUT = ROOT / "modeling" / "daily_card_agg.csv"
KEY_COLUMNS = ("TA_YMD", "MCT_SGG_CD", "MCT_RY_CD")
REQUIRED_COLUMNS = set(KEY_COLUMNS) | {"TS_AT", "USE_CNT", "TIME_GB"}


def source_card_file() -> Path:
    candidates = sorted(CARD_DIR.glob("*.txt")) + sorted(CARD_DIR.glob("*.tsv"))
    for path in candidates:
        with path.open("r", encoding="cp949", newline="") as f:
            header = next(csv.reader(f, delimiter="\t"), [])
        if REQUIRED_COLUMNS.issubset(set(header)):
            return path
    raise FileNotFoundError("카드 1 데이터(TIME_GB 포함)를 card_data/에서 찾지 못했습니다.")


def parse_decimal(value: str) -> Decimal | None:
    value = value.strip()
    if not value:
        return None
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def main() -> None:
    source = source_card_file()
    # Each value stores sums and missing-value counts. Decimal avoids rounding
    # monetary/count totals during aggregation.
    groups: dict[tuple[str, str, str], list] = {}
    raw_rows = 0
    skipped_missing_key_rows = 0
    invalid_dates = 0
    invalid_sales = 0
    invalid_use_counts = 0
    missing_sales = 0
    missing_use_counts = 0
    zero_raw_use_counts = 0
    raw_key_occurrences: set[tuple[str, str, str]] = set()
    repeated_target_key_rows = 0
    min_date = None
    max_date = None

    with source.open("r", encoding="cp949", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        absent = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if absent:
            raise ValueError(f"필수 컬럼이 없습니다: {sorted(absent)}")
        for row in reader:
            raw_rows += 1
            key = tuple((row.get(col) or "").strip() for col in KEY_COLUMNS)
            if any(not part for part in key):
                skipped_missing_key_rows += 1
                continue
            date_value = key[0]
            try:
                parsed_date = datetime.strptime(date_value, "%Y%m%d").date()
                min_date = parsed_date if min_date is None or parsed_date < min_date else min_date
                max_date = parsed_date if max_date is None or parsed_date > max_date else max_date
            except ValueError:
                invalid_dates += 1

            if key in raw_key_occurrences:
                repeated_target_key_rows += 1
            else:
                raw_key_occurrences.add(key)

            if key not in groups:
                groups[key] = [Decimal(0), 0, Decimal(0), 0, 0]
            state = groups[key]
            sales = parse_decimal(row.get("TS_AT") or "")
            use_count = parse_decimal(row.get("USE_CNT") or "")
            if sales is None:
                missing_sales += 1
                state[1] += 1
                if (row.get("TS_AT") or "").strip():
                    invalid_sales += 1
            else:
                state[0] += sales
            if use_count is None:
                missing_use_counts += 1
                state[3] += 1
                if (row.get("USE_CNT") or "").strip():
                    invalid_use_counts += 1
            else:
                state[2] += use_count
                if use_count == 0:
                    zero_raw_use_counts += 1
            state[4] += 1

    output_rows = []
    missing_agg_sales = 0
    missing_agg_use_counts = 0
    zero_agg_use_counts = 0
    missing_or_nonfinite_avg = 0
    infinite_avg = 0
    for key, state in sorted(groups.items()):
        sales_sum, sales_missing, use_sum, use_missing, _ = state
        sales_out = "" if sales_missing else decimal_text(sales_sum)
        use_out = "" if use_missing else decimal_text(use_sum)
        if sales_missing:
            missing_agg_sales += 1
        if use_missing:
            missing_agg_use_counts += 1
        avg_out = ""
        if not sales_missing and not use_missing and use_sum != 0:
            avg = sales_sum / use_sum
            if avg.is_finite():
                avg_out = decimal_text(avg)
            else:
                infinite_avg += 1
        if not avg_out:
            missing_or_nonfinite_avg += 1
        if not use_missing and use_sum == 0:
            zero_agg_use_counts += 1
        output_rows.append((*key, sales_out, use_out, avg_out))

    output_columns = [*KEY_COLUMNS, "TS_AT", "USE_CNT", "avg_price"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(output_columns)
        writer.writerows(output_rows)

    output_keys = [(r[0], r[1], r[2]) for r in output_rows]
    duplicate_output_keys = len(output_keys) - len(set(output_keys))
    regions = {r[1] for r in output_rows}
    industries = {r[2] for r in output_rows}

    print("=== 일별 카드 기본 집계 검증 ===")
    print(f"사용 원본: {source.relative_to(ROOT)} (TIME_GB 포함 카드 1만 사용; 카드 2와 합산하지 않음)")
    print("집계 단위: TA_YMD × MCT_SGG_CD × MCT_RY_CD")
    print(f"집계 전 행 수: {raw_rows:,}")
    print(f"집계 후 행 수: {len(output_rows):,}")
    print(f"집계 키 중복 입력 행(같은 키의 추가 관측치): {repeated_target_key_rows:,}")
    print(f"집계 후 키 중복: {duplicate_output_keys:,}")
    print(f"원본 TS_AT 결측/비수치: {missing_sales:,} (비수치: {invalid_sales:,})")
    print(f"원본 USE_CNT 결측/비수치: {missing_use_counts:,} (비수치: {invalid_use_counts:,})")
    print(f"집계 결과 TS_AT 결측 행: {missing_agg_sales:,}")
    print(f"집계 결과 USE_CNT 결측 행: {missing_agg_use_counts:,}")
    print(f"원본 USE_CNT=0 행: {zero_raw_use_counts:,}")
    print(f"집계 USE_CNT=0 행: {zero_agg_use_counts:,}")
    print(f"avg_price 결측/비유한 행: {missing_or_nonfinite_avg:,}")
    print(f"avg_price 무한값 행: {infinite_avg:,}")
    print(f"잘못된 날짜 행: {invalid_dates:,}; 키 결측으로 제외한 행: {skipped_missing_key_rows:,}")
    print(f"날짜 범위: {min_date} .. {max_date}")
    print(f"지역 고유값 수: {len(regions):,}")
    print(f"업종 고유값 수: {len(industries):,}")
    print("생성 컬럼: " + ", ".join(output_columns))
    print(f"저장 파일: {OUTPUT.relative_to(ROOT)}")
    print("사건 변수 생성 및 SK 통신인구 merge: 없음")
    print("\n샘플 (최대 10행):")
    for row in output_rows[:10]:
        print("\t".join(row))


if __name__ == "__main__":
    main()
