"""Create a cleaned PathSnap copy while preserving all road-network IDs.

Policy: discard malformed/non-finite GPS rows, out-of-range coordinates, and
numeric duplicate observations (same longitude, latitude, and timestamp).
The first text record is retained verbatim. Track order, road files, and route
labels are preserved. Gaps are classified in a separate report; points are
never interpolated automatically.

Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from decimal import Decimal, InvalidOperation
from pathlib import Path

PARTITIONS = ("train", "student_test")
SUFFIXES = (".track", ".nodes", ".arcs", ".route")


def gap_classification(seconds: float) -> str:
    """Classify adjacent sample intervals; thresholds are explicit and configurable by policy."""
    if seconds <= 0:
        return "nonincreasing_timestamp"
    if seconds <= 2:
        return "regular"
    if seconds <= 10:
        return "irregular_2_to_10s"
    if seconds <= 60:
        return "large_10_to_60s"
    return "extreme_over_60s"


def clean_track(source: Path, target: Path, gap_rows: list[dict[str, object]]) -> dict[str, int]:
    seen: set[tuple[Decimal, Decimal, Decimal]] = set()
    counts = {
        "input_nonblank_rows": 0,
        "malformed_or_nonfinite_rows_removed": 0,
        "invalid_coordinate_rows_removed": 0,
        "numeric_duplicate_rows_removed": 0,
        "output_rows": 0,
        "nonincreasing_timestamp_pairs": 0,
        "irregular_gap_pairs_2_to_10s": 0,
        "large_gap_pairs_10_to_60s": 0,
        "extreme_gap_pairs_over_60s": 0,
    }
    previous: tuple[int, float] | None = None
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8-sig") as src, target.open("w", encoding="utf-8", newline="\n") as dst:
        for source_line, raw_line in enumerate(src, start=1):
            text = raw_line.strip()
            if not text:
                continue
            counts["input_nonblank_rows"] += 1
            parts = text.split()
            if len(parts) != 3:
                counts["malformed_or_nonfinite_rows_removed"] += 1
                continue
            try:
                numeric_key = tuple(Decimal(value) for value in parts)
                lon, lat, timestamp = (float(value) for value in numeric_key)
            except (ValueError, InvalidOperation, OverflowError):
                counts["malformed_or_nonfinite_rows_removed"] += 1
                continue
            if not all(value.is_finite() for value in numeric_key) or not all(
                math.isfinite(value) for value in (lon, lat, timestamp)
            ):
                counts["malformed_or_nonfinite_rows_removed"] += 1
                continue
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                counts["invalid_coordinate_rows_removed"] += 1
                continue
            if numeric_key in seen:
                counts["numeric_duplicate_rows_removed"] += 1
                continue
            seen.add(numeric_key)

            if previous is not None:
                previous_row, previous_timestamp = previous
                delta = timestamp - previous_timestamp
                kind = gap_classification(delta)
                if kind != "regular":
                    gap_rows.append({
                        "partition": source.parent.parent.name,
                        "trip_id": source.parent.name,
                        "previous_source_row": previous_row,
                        "current_source_row": source_line,
                        "gap_seconds": delta,
                        "gap_class": kind,
                        "handling": "preserved; downstream code must handle explicitly; no interpolation",
                    })
                    if kind == "nonincreasing_timestamp":
                        counts["nonincreasing_timestamp_pairs"] += 1
                    elif kind == "irregular_2_to_10s":
                        counts["irregular_gap_pairs_2_to_10s"] += 1
                    elif kind == "large_10_to_60s":
                        counts["large_gap_pairs_10_to_60s"] += 1
                    else:
                        counts["extreme_gap_pairs_over_60s"] += 1
            previous = (source_line, timestamp)
            # Keep the first valid numeric record exactly as supplied.
            dst.write(text + "\n")
            counts["output_rows"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Source folder containing train/ and student_test/.")
    parser.add_argument("--output-root", type=Path, required=True, help="New output folder for cleaned partitions and reports.")
    args = parser.parse_args()

    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"Output folder is not empty; choose a new path: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    log_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []
    for partition in PARTITIONS:
        source_part = args.data_root / partition
        if not source_part.is_dir():
            raise SystemExit(f"Missing partition directory: {source_part}")
        for trip_dir in sorted(path for path in source_part.iterdir() if path.is_dir()):
            trip_id = trip_dir.name
            out_dir = args.output_root / partition / trip_id
            out_dir.mkdir(parents=True, exist_ok=True)
            for suffix in SUFFIXES:
                source = trip_dir / f"{trip_id}{suffix}"
                if not source.is_file():
                    raise SystemExit(f"Missing required trip file: {source}")
                target = out_dir / source.name
                if suffix == ".track":
                    counts = clean_track(source, target, gap_rows)
                    log_rows.append({"partition": partition, "trip_id": trip_id, **counts})
                else:
                    # Row positions are IDs: .arcs references .nodes; .route references .arcs.
                    shutil.copy2(source, target)

    if not log_rows:
        raise SystemExit("No trips found.")
    log_path = args.output_root / "cleaning_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    gap_path = args.output_root / "gps_gap_report.csv"
    gap_fields = ["partition", "trip_id", "previous_source_row", "current_source_row", "gap_seconds", "gap_class", "handling"]
    with gap_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=gap_fields)
        writer.writeheader()
        writer.writerows(gap_rows)
    print(f"Cleaned data written to: {args.output_root.resolve()}")
    print(f"Cleaning log written to: {log_path.resolve()}")
    print(f"Classified GPS gaps written to: {gap_path.resolve()} ({len(gap_rows)} flagged intervals)")


if __name__ == "__main__":
    main()

