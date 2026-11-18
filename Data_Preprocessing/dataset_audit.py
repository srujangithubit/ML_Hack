"""Audit PathSnap trip files and write a per-trip inventory CSV.

Expected layout: DATA_ROOT/{train,student_test}/TRIP_ID/TRIP_ID.{track,nodes,arcs,route}
Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from statistics import median

PARTITIONS = ("train", "student_test")
EXPECTED_SUFFIXES = (".track", ".nodes", ".arcs", ".route")


def parse_rows(path: Path, expected_columns: int) -> tuple[list[tuple[float, ...]], int, int, int]:
    """Return parsed numeric rows, malformed count, blank-line count, exact duplicate count."""
    rows: list[tuple[float, ...]] = []
    malformed = blanks = duplicates = 0
    seen: set[tuple[str, ...]] = set()
    with path.open("r", encoding="utf-8-sig") as stream:
        for raw_line in stream:
            text = raw_line.strip()
            if not text:
                blanks += 1
                continue
            parts = tuple(text.split())
            if len(parts) != expected_columns:
                malformed += 1
                continue
            try:
                values = tuple(float(part) for part in parts)
            except ValueError:
                malformed += 1
                continue
            if not all(math.isfinite(value) for value in values):
                malformed += 1
                continue
            if parts in seen:
                duplicates += 1
            else:
                seen.add(parts)
            rows.append(values)
    return rows, malformed, blanks, duplicates


def weak_components(node_count: int, arcs: list[tuple[float, ...]]) -> tuple[int, int, int]:
    """Return weak component count, isolated node count and invalid endpoint count."""
    parent = list(range(node_count))
    rank = [0] * node_count
    degree = [0] * node_count
    invalid = 0

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        if rank[root_left] < rank[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        if rank[root_left] == rank[root_right]:
            rank[root_left] += 1

    for source, target in arcs:
        if not source.is_integer() or not target.is_integer():
            invalid += 1
            continue
        source_id, target_id = int(source), int(target)
        if not (0 <= source_id < node_count and 0 <= target_id < node_count):
            invalid += 1
            continue
        degree[source_id] += 1
        degree[target_id] += 1
        union(source_id, target_id)
    components = len({find(node_id) for node_id in range(node_count)}) if node_count else 0
    isolated = sum(value == 0 for value in degree)
    return components, isolated, invalid


def audit_trip(partition: str, trip_dir: Path) -> dict[str, object]:
    trip_id = trip_dir.name
    paths = {suffix: trip_dir / f"{trip_id}{suffix}" for suffix in EXPECTED_SUFFIXES}
    missing_files = [suffix for suffix, path in paths.items() if not path.is_file()]
    if missing_files:
        return {"partition": partition, "trip_id": trip_id, "missing_files": ";".join(missing_files)}

    track, bad_track, blank_track, dup_track = parse_rows(paths[".track"], 3)
    nodes, bad_nodes, blank_nodes, dup_nodes = parse_rows(paths[".nodes"], 2)
    arcs, bad_arcs, blank_arcs, dup_arcs = parse_rows(paths[".arcs"], 2)
    route, bad_route, blank_route, dup_route = parse_rows(paths[".route"], 1)

    timestamps = [row[2] for row in track]
    intervals = [right - left for left, right in zip(timestamps, timestamps[1:])]
    nonincreasing = sum(value <= 0 for value in intervals)
    valid_gps = [row for row in track if -180 <= row[0] <= 180 and -90 <= row[1] <= 90]
    invalid_coords = len(track) - len(valid_gps)
    node_count = len(nodes)
    components, isolated, invalid_endpoints = weak_components(node_count, arcs)

    integer_route = [int(row[0]) for row in route if row[0].is_integer()]
    invalid_route_ids = sum(not 0 <= arc_id < len(arcs) for arc_id in integer_route)
    discontinuities = 0
    for first_id, second_id in zip(integer_route, integer_route[1:]):
        if not (0 <= first_id < len(arcs) and 0 <= second_id < len(arcs)):
            continue
        first, second = arcs[first_id], arcs[second_id]
        if not (first[1].is_integer() and second[0].is_integer() and int(first[1]) == int(second[0])):
            discontinuities += 1

    lon = [row[0] for row in valid_gps]
    lat = [row[1] for row in valid_gps]
    gaps_gt_2 = sum(value > 2 for value in intervals)
    gaps_gt_10 = sum(value > 10 for value in intervals)
    row: dict[str, object] = {
        "partition": partition,
        "trip_id": trip_id,
        "missing_files": "",
        "track_rows": len(track),
        "track_malformed_rows": bad_track,
        "track_blank_lines": blank_track,
        "track_exact_duplicate_rows": dup_track,
        "invalid_gps_coordinates": invalid_coords,
        "nodes": node_count,
        "nodes_malformed_rows": bad_nodes,
        "nodes_blank_lines": blank_nodes,
        "nodes_exact_duplicate_rows": dup_nodes,
        "arcs": len(arcs),
        "arcs_malformed_rows": bad_arcs,
        "arcs_blank_lines": blank_arcs,
        "arcs_exact_duplicate_rows": dup_arcs,
        "invalid_arc_endpoints": invalid_endpoints,
        "weakly_connected_components": components,
        "isolated_nodes": isolated,
        "route_rows": len(route),
        "route_malformed_rows": bad_route,
        "route_blank_lines": blank_route,
        "route_exact_duplicate_rows": dup_route,
        "route_arc_ids_out_of_range": invalid_route_ids,
        "route_consecutive_discontinuities": discontinuities,
        "timestamp_nonincreasing_pairs": nonincreasing,
        "timestamp_gap_pairs_gt_2s": gaps_gt_2,
        "timestamp_gap_pairs_gt_10s": gaps_gt_10,
        "timestamp_min_s": min(timestamps) if timestamps else "",
        "timestamp_max_s": max(timestamps) if timestamps else "",
        "timestamp_interval_median_s": median(intervals) if intervals else "",
        "timestamp_interval_max_s": max(intervals) if intervals else "",
        "gps_longitude_min": min(lon) if lon else "",
        "gps_longitude_max": max(lon) if lon else "",
        "gps_latitude_min": min(lat) if lat else "",
        "gps_latitude_max": max(lat) if lat else "",
        "track_bytes": paths[".track"].stat().st_size,
        "nodes_bytes": paths[".nodes"].stat().st_size,
        "arcs_bytes": paths[".arcs"].stat().st_size,
        "route_bytes": paths[".route"].stat().st_size,
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Folder containing train/ and student_test/.")
    parser.add_argument("--output", type=Path, default=Path("trip_inventory.csv"), help="Output CSV path.")
    args = parser.parse_args()

    results: list[dict[str, object]] = []
    for partition in PARTITIONS:
        part_dir = args.data_root / partition
        if not part_dir.is_dir():
            raise SystemExit(f"Missing partition directory: {part_dir}")
        trip_dirs = sorted(path for path in part_dir.iterdir() if path.is_dir())
        results.extend(audit_trip(partition, trip_dir) for trip_dir in trip_dirs)

    if not results:
        raise SystemExit("No trip folders found.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    for partition in PARTITIONS:
        group = [row for row in results if row.get("partition") == partition]
        sum_field = lambda field: sum(int(row.get(field, 0) or 0) for row in group)
        print(
            f"{partition}: trips={len(group)}, GPS={sum_field('track_rows')}, "
            f"nodes={sum_field('nodes')}, arcs={sum_field('arcs')}, routes={sum_field('route_rows')}"
        )
        print(
            f"  malformed rows: track={sum_field('track_malformed_rows')}, "
            f"nodes={sum_field('nodes_malformed_rows')}, arcs={sum_field('arcs_malformed_rows')}, "
            f"route={sum_field('route_malformed_rows')}"
        )
        print(
            f"  duplicate rows: track={sum_field('track_exact_duplicate_rows')}, "
            f"nodes={sum_field('nodes_exact_duplicate_rows')}, arcs={sum_field('arcs_exact_duplicate_rows')}"
        )
        print(
            f"  invalid GPS={sum_field('invalid_gps_coordinates')}, "
            f"bad arc endpoints={sum_field('invalid_arc_endpoints')}, "
            f"isolated nodes={sum_field('isolated_nodes')}, "
            f"route discontinuities={sum_field('route_consecutive_discontinuities')}"
        )
    print(f"Inventory written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
