"""Create a cleaned copy of PathSnap data while preserving road/route IDs.

Cleaning policy:
* drop malformed or non-finite track rows and out-of-range GPS coordinates;
* drop only exact duplicate track rows (same longitude, latitude and timestamp);
* preserve input order and never renumber/deduplicate nodes or arcs because those
  row positions are IDs used by .arcs and .route;
* copy .route unchanged and report non-increasing timestamps for review rather
  than silently reordering trajectory observations.

Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

PARTITIONS = ("train", "student_test")
SUFFIXES = (".track", ".nodes", ".arcs", ".route")


def clean_track(source: Path, target: Path) -> dict[str, int]:
    seen: set[tuple[str, str, str]] = set()
    counts = {
        "input_nonblank_rows": 0,
        "malformed_or_nonfinite_rows_removed": 0,
        "invalid_coordinate_rows_removed": 0,
        "exact_duplicate_rows_removed": 0,
        "output_rows": 0,
        "nonincreasing_timestamp_pairs": 0,
    }
    previous_timestamp: float | None = None
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8-sig") as src, target.open("w", encoding="utf-8", newline="\n") as dst:
        for raw_line in src:
            text = raw_line.strip()
            if not text:
                continue
            counts["input_nonblank_rows"] += 1
            parts = text.split()
            if len(parts) != 3:
                counts["malformed_or_nonfinite_rows_removed"] += 1
                continue
            try:
                lon, lat, timestamp = (float(value) for value in parts)
            except ValueError:
                counts["malformed_or_nonfinite_rows_removed"] += 1
                continue
            if not all(math.isfinite(value) for value in (lon, lat, timestamp)):
                counts["malformed_or_nonfinite_rows_removed"] += 1
                continue
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                counts["invalid_coordinate_rows_removed"] += 1
                continue

            exact_row = tuple(parts)
            if exact_row in seen:
                counts["exact_duplicate_rows_removed"] += 1
                continue
            seen.add(exact_row)

            if previous_timestamp is not None and timestamp <= previous_timestamp:
                counts["nonincreasing_timestamp_pairs"] += 1
            previous_timestamp = timestamp
            dst.write("\t".join(parts) + "\n")
            counts["output_rows"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Source folder containing train/ and student_test/.")
    parser.add_argument("--output-root", type=Path, required=True, help="New output folder for cleaned partitions.")
    args = parser.parse_args()

    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit(f"Output folder is not empty; choose a new path: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    log_rows: list[dict[str, object]] = []
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
                    counts = clean_track(source, target)
                    log_rows.append({"partition": partition, "trip_id": trip_id, **counts})
                else:
                    # Keep row IDs stable: .arcs references .nodes, .route references .arcs.
                    shutil.copy2(source, target)

    log_path = args.output_root / "cleaning_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Cleaned data written to: {args.output_root.resolve()}")
    print(f"Cleaning log written to: {log_path.resolve()}")


if __name__ == "__main__":
    main()
