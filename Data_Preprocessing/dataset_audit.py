"""Audit PathSnap data and write trip, GPS-gap, and route-issue reports.

Expected layout: DATA_ROOT/{train,student_test}/TRIP_ID/TRIP_ID.{track,nodes,arcs,route}
Only Python's standard library is required. This read-only audit never edits source files.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import median

PARTITIONS = ("train", "student_test")
EXPECTED = (".track", ".nodes", ".arcs", ".route")


def classify_gap(seconds: float) -> str:
    if seconds <= 0:
        return "nonincreasing_timestamp"
    if seconds <= 2:
        return "regular"
    if seconds <= 10:
        return "irregular_2_to_10s"
    if seconds <= 60:
        return "large_10_to_60s"
    return "extreme_over_60s"


def read_float_rows(path: Path, columns: int, preserve_ids: bool = False):
    rows, malformed, blanks, duplicates = [], 0, 0, 0
    seen = set()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            text = line.strip()
            if not text:
                blanks += 1
                continue
            parts = text.split()
            if len(parts) != columns:
                malformed += 1
                if preserve_ids:
                    rows.append(None)
                continue
            try:
                exact_numeric = tuple(Decimal(value) for value in parts)
                values = tuple(float(value) for value in exact_numeric)
            except (InvalidOperation, ValueError, OverflowError):
                malformed += 1
                if preserve_ids:
                    rows.append(None)
                continue
            if not all(value.is_finite() for value in exact_numeric) or not all(math.isfinite(value) for value in values):
                malformed += 1
                if preserve_ids:
                    rows.append(None)
                continue
            if exact_numeric in seen:
                duplicates += 1
            else:
                seen.add(exact_numeric)
            rows.append(values)
    return rows, malformed, blanks, duplicates


def read_integer_rows(path: Path, columns: int):
    """Keep a placeholder per nonblank row so malformed rows do not shift IDs."""
    rows, malformed, noninteger, blanks, duplicates = [], 0, 0, 0, 0
    noninteger_positions, seen = [], set()
    entry_index = 0
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            text = line.strip()
            if not text:
                blanks += 1
                continue
            entry_index += 1
            parts = text.split()
            if len(parts) != columns:
                malformed += 1
                rows.append(None)
                continue
            try:
                decimals = tuple(Decimal(value) for value in parts)
            except InvalidOperation:
                malformed += 1
                rows.append(None)
                continue
            if not all(value.is_finite() for value in decimals):
                malformed += 1
                rows.append(None)
                continue
            if not all(value == value.to_integral_value() for value in decimals):
                noninteger += 1
                noninteger_positions.append((entry_index, line_number))
                rows.append(None)
                continue
            values = tuple(int(value) for value in decimals)
            if values in seen:
                duplicates += 1
            else:
                seen.add(values)
            rows.append(values)
    return rows, malformed, noninteger, blanks, duplicates, noninteger_positions


def relation(previous: tuple[int, int], current: tuple[int, int]) -> str:
    if previous[1] == current[0]:
        return "directed_continuation"
    if previous[0] == current[1]:
        return "reverse_direction_match"
    if set(previous) & set(current):
        return "shared_endpoint_other_orientation"
    return "no_shared_endpoint"


def audit_trip(partition: str, folder: Path):
    trip_id = folder.name
    paths = {suffix: folder / f"{trip_id}{suffix}" for suffix in EXPECTED}
    missing = [suffix for suffix, path in paths.items() if not path.is_file()]
    if missing:
        return {"partition": partition, "trip_id": trip_id, "missing_files": ";".join(missing)}, [], []

    track, bad_track, blank_track, dup_track = read_float_rows(paths[".track"], 3)
    nodes, bad_nodes, blank_nodes, dup_nodes = read_float_rows(paths[".nodes"], 2, preserve_ids=True)
    arcs, bad_arcs, noninteger_arcs, blank_arcs, dup_arcs, bad_arc_lines = read_integer_rows(paths[".arcs"], 2)
    route, bad_route, noninteger_route, blank_route, dup_route, bad_route_lines = read_integer_rows(paths[".route"], 1)

    invalid_gps = sum(not (-180 <= row[0] <= 180 and -90 <= row[1] <= 90) for row in track)
    invalid_nodes = sum(node is None or not (-180 <= node[0] <= 180 and -90 <= node[1] <= 90) for node in nodes)
    valid_arcs = [arc for arc in arcs if arc is not None]
    arc_rows_invalid = len(arcs) - len(valid_arcs)
    invalid_endpoints = sum(not (0 <= arc[0] < len(nodes) and 0 <= arc[1] < len(nodes)) for arc in valid_arcs)
    endpoint_valid_arcs = [arc for arc in valid_arcs if 0 <= arc[0] < len(nodes) and 0 <= arc[1] < len(nodes)]

    parent = list(range(len(nodes)))
    rank = [0] * len(nodes)
    degree = [0] * len(nodes)

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    for source, target in endpoint_valid_arcs:
        degree[source] += 1
        degree[target] += 1
        union(source, target)
    components = len({find(node_id) for node_id in range(len(nodes))}) if nodes else 0
    isolated = sum(value == 0 for value in degree)
    self_loops = sum(source == target for source, target in valid_arcs)
    zero_length = sum(
        nodes[source] is not None and nodes[target] is not None and nodes[source] == nodes[target]
        for source, target in endpoint_valid_arcs
    )

    route_issues = []
    for route_position, line_number in bad_route_lines:
        route_issues.append({
            "partition": partition, "trip_id": trip_id, "issue_type": "noninteger_route_id",
            "route_position": route_position, "source_line": line_number, "previous_arc_id": "",
            "current_arc_id": "", "previous_endpoints": "", "current_endpoints": "",
            "classification": "route ID must be an integer value",
        })
    for arc_position, line_number in bad_arc_lines:
        route_issues.append({
            "partition": partition, "trip_id": trip_id, "issue_type": "noninteger_arc_endpoint",
            "route_position": "", "source_line": line_number, "previous_arc_id": "",
            "current_arc_id": arc_position, "previous_endpoints": "", "current_endpoints": "",
            "classification": "arc endpoints must be integer node IDs",
        })

    valid_route_refs = []
    out_of_range = invalid_arc_refs = invalid_endpoint_refs = 0
    for position, entry in enumerate(route, start=1):
        if entry is None:
            continue
        arc_id = entry[0]
        if not 0 <= arc_id < len(arcs):
            out_of_range += 1
            route_issues.append({
                "partition": partition, "trip_id": trip_id, "issue_type": "route_arc_id_out_of_range",
                "route_position": position, "source_line": "", "previous_arc_id": "",
                "current_arc_id": arc_id, "previous_endpoints": "", "current_endpoints": "",
                "classification": f"valid row IDs range from 0 through {len(arcs)-1}",
            })
            continue
        arc = arcs[arc_id]
        if arc is None:
            invalid_arc_refs += 1
            route_issues.append({
                "partition": partition, "trip_id": trip_id, "issue_type": "route_references_invalid_arc_row",
                "route_position": position, "source_line": "", "previous_arc_id": "",
                "current_arc_id": arc_id, "previous_endpoints": "", "current_endpoints": "",
                "classification": "referenced arc row is malformed or contains non-integer endpoints",
            })
            continue
        if not (0 <= arc[0] < len(nodes) and 0 <= arc[1] < len(nodes)):
            invalid_endpoint_refs += 1
            route_issues.append({
                "partition": partition, "trip_id": trip_id, "issue_type": "route_references_invalid_node_id",
                "route_position": position, "source_line": "", "previous_arc_id": "",
                "current_arc_id": arc_id, "previous_endpoints": "", "current_endpoints": f"{arc[0]}->{arc[1]}",
                "classification": "referenced arc has an endpoint outside the node table",
            })
            continue
        valid_route_refs.append((position, arc_id, arc))

    route_transition_counts = Counter()
    # Only compare adjacent entries in the original route sequence. Never bridge a bad ID.
    for index in range(1, len(route)):
        previous_entry, current_entry = route[index - 1], route[index]
        if previous_entry is None or current_entry is None:
            continue
        previous_id, current_id = previous_entry[0], current_entry[0]
        if not (0 <= previous_id < len(arcs) and 0 <= current_id < len(arcs)):
            continue
        previous_arc, current_arc = arcs[previous_id], arcs[current_id]
        if previous_arc is None or current_arc is None:
            continue
        if not all(0 <= node_id < len(nodes) for node_id in (*previous_arc, *current_arc)):
            continue
        kind = relation(previous_arc, current_arc)
        if kind != "directed_continuation":
            route_transition_counts[kind] += 1
            route_issues.append({
                "partition": partition, "trip_id": trip_id, "issue_type": "route_transition_review",
                "route_position": index + 1, "source_line": "", "previous_arc_id": previous_id,
                "current_arc_id": current_id, "previous_endpoints": f"{previous_arc[0]}->{previous_arc[1]}",
                "current_endpoints": f"{current_arc[0]}->{current_arc[1]}", "classification": kind,
            })

    gap_report, gap_counts = [], Counter()
    timestamps = [row[2] for row in track]
    intervals = [right - left for left, right in zip(timestamps, timestamps[1:])]
    for index, seconds in enumerate(intervals, start=1):
        kind = classify_gap(seconds)
        if kind != "regular":
            gap_counts[kind] += 1
            gap_report.append({
                "partition": partition, "trip_id": trip_id, "previous_record": index,
                "current_record": index + 1, "gap_seconds": seconds, "gap_class": kind,
                "handling": "preserve; downstream code must handle explicitly; no interpolation",
            })
    valid_gps = [row for row in track if -180 <= row[0] <= 180 and -90 <= row[1] <= 90]
    row = {
        "partition": partition, "trip_id": trip_id, "missing_files": "",
        "track_rows": len(track), "track_malformed_or_nonfinite_rows": bad_track,
        "track_blank_lines": blank_track, "track_numeric_duplicate_rows": dup_track,
        "invalid_gps_coordinates": invalid_gps,
        "nodes": len(nodes), "nodes_malformed_or_nonfinite_rows": bad_nodes,
        "nodes_blank_lines": blank_nodes, "nodes_numeric_duplicate_rows": dup_nodes,
        "invalid_node_coordinates": invalid_nodes,
        "arcs": len(arcs), "arcs_malformed_rows": bad_arcs, "arc_noninteger_rows": noninteger_arcs,
        "arcs_blank_lines": blank_arcs, "invalid_arc_rows": arc_rows_invalid,
        "duplicate_directed_arcs": dup_arcs, "arc_self_loops": self_loops,
        "zero_length_arcs": zero_length, "invalid_arc_endpoints": invalid_endpoints,
        "weakly_connected_components": components, "isolated_nodes": isolated,
        "route_rows": len(route), "route_malformed_rows": bad_route,
        "route_noninteger_values": noninteger_route, "route_blank_lines": blank_route,
        "route_numeric_duplicate_entries": dup_route, "route_valid_integer_rows": sum(entry is not None for entry in route),
        "route_arc_ids_out_of_range": out_of_range, "route_references_invalid_arc_rows": invalid_arc_refs,
        "route_references_invalid_endpoint_arcs": invalid_endpoint_refs,
        "route_transition_review_count": sum(route_transition_counts.values()),
        "route_reverse_direction_matches": route_transition_counts["reverse_direction_match"],
        "route_other_shared_endpoint": route_transition_counts["shared_endpoint_other_orientation"],
        "route_no_shared_endpoint": route_transition_counts["no_shared_endpoint"],
        "timestamp_nonincreasing_pairs": gap_counts["nonincreasing_timestamp"],
        "timestamp_gap_pairs_2_to_10s": gap_counts["irregular_2_to_10s"],
        "timestamp_gap_pairs_10_to_60s": gap_counts["large_10_to_60s"],
        "timestamp_gap_pairs_over_60s": gap_counts["extreme_over_60s"],
        "timestamp_min_s": min(timestamps) if timestamps else "",
        "timestamp_max_s": max(timestamps) if timestamps else "",
        "timestamp_interval_median_s": median(intervals) if intervals else "",
        "timestamp_interval_max_s": max(intervals) if intervals else "",
        "gps_longitude_min": min((row[0] for row in valid_gps), default=""),
        "gps_longitude_max": max((row[0] for row in valid_gps), default=""),
        "gps_latitude_min": min((row[1] for row in valid_gps), default=""),
        "gps_latitude_max": max((row[1] for row in valid_gps), default=""),
        "track_bytes": paths[".track"].stat().st_size, "nodes_bytes": paths[".nodes"].stat().st_size,
        "arcs_bytes": paths[".arcs"].stat().st_size, "route_bytes": paths[".route"].stat().st_size,
    }
    return row, gap_report, route_issues


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Folder containing train/ and student_test/.")
    parser.add_argument("--output", type=Path, default=Path("trip_inventory.csv"), help="Trip inventory CSV.")
    parser.add_argument("--gap-report", type=Path, default=None, help="Detailed GPS gap report; defaults beside inventory.")
    parser.add_argument("--route-report", type=Path, default=None, help="Detailed route issue report; defaults beside inventory.")
    args = parser.parse_args()

    inventory, gaps, route_issues = [], [], []
    for partition in PARTITIONS:
        part_dir = args.data_root / partition
        if not part_dir.is_dir():
            raise SystemExit(f"Missing partition directory: {part_dir}")
        for trip_dir in sorted(path for path in part_dir.iterdir() if path.is_dir()):
            summary, trip_gaps, trip_route_issues = audit_trip(partition, trip_dir)
            inventory.append(summary)
            gaps.extend(trip_gaps)
            route_issues.extend(trip_route_issues)
    if not inventory:
        raise SystemExit("No trip folders found.")

    write_csv(args.output, inventory, list(inventory[0]))
    gap_path = args.gap_report or args.output.with_name("gps_gap_report.csv")
    route_path = args.route_report or args.output.with_name("route_issues.csv")
    write_csv(gap_path, gaps, ["partition", "trip_id", "previous_record", "current_record", "gap_seconds", "gap_class", "handling"])
    write_csv(route_path, route_issues, ["partition", "trip_id", "issue_type", "route_position", "source_line", "previous_arc_id", "current_arc_id", "previous_endpoints", "current_endpoints", "classification"])

    for partition in PARTITIONS:
        group = [item for item in inventory if item["partition"] == partition]
        def total(key: str) -> int:
            return sum(int(item.get(key, 0) or 0) for item in group)
        print(f"{partition}: trips={len(group)}, GPS={total('track_rows')}, nodes={total('nodes')}, arcs={total('arcs')}, routes={total('route_rows')}")
        print(f"  numeric duplicate GPS={total('track_numeric_duplicate_rows')}; invalid GPS coords={total('invalid_gps_coordinates')}")
        print(f"  invalid node coords={total('invalid_node_coordinates')}; invalid arc endpoints={total('invalid_arc_endpoints')}; zero-length arcs={total('zero_length_arcs')}")
        print(f"  duplicate directed arcs={total('duplicate_directed_arcs')}; bad/noninteger arc rows={total('arcs_malformed_rows') + total('arc_noninteger_rows')}")
        print(f"  route out-of-range IDs={total('route_arc_ids_out_of_range')}; noninteger entries={total('route_noninteger_values')}; transitions for review={total('route_transition_review_count')}")
        print(f"  gap pairs: 2-10s={total('timestamp_gap_pairs_2_to_10s')}; 10-60s={total('timestamp_gap_pairs_10_to_60s')}; >60s={total('timestamp_gap_pairs_over_60s')}; nonincreasing={total('timestamp_nonincreasing_pairs')}")
    print(f"Inventory: {args.output.resolve()}")
    print(f"GPS gap report: {gap_path.resolve()}")
    print(f"Route issue report: {route_path.resolve()}")


if __name__ == "__main__":
    main()

