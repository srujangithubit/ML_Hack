#!/usr/bin/env python3
"""PathSnap Member 1B: spatial candidate-road generation and feature engineering.

Input: cleaned_dataset/{train,student_test}/{trip_id}/{trip_id}.{track,nodes,arcs,route}
Output: one row per GPS observation / candidate directed arc.

The .route file is read and used only for training labels, after geographic
candidate generation. It is never used for candidate selection or inference.
Labels are approximate route-order labels; unmatched observations stay -1.
"""
from __future__ import annotations

import argparse
import math
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import LineString, STRtree, points

EARTH_R_M = 6_371_008.8


def read_numeric(path: Path, ncols: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    if path.stat().st_size == 0:
        if path.suffix == ".route":
            return np.empty((0, ncols), dtype=float)
        raise ValueError(f"Required data file is empty: {path}")
    data = np.loadtxt(path, dtype=float, ndmin=2)
    if data.shape[1] != ncols:
        raise ValueError(f"{path}: expected {ncols} columns, found {data.shape[1]}")
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: non-finite values; check Member 1A cleaning")
    return data


def project_local(lonlat: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    """Local equirectangular projection, returning metres."""
    lon = np.radians(lonlat[:, 0] - lon0)
    lat = np.radians(lonlat[:, 1] - lat0)
    return np.column_stack((
        EARTH_R_M * math.cos(math.radians(lat0)) * lon,
        EARTH_R_M * lat,
    ))


def bearing_xy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Clockwise bearing from north, in degrees [0, 360)."""
    dx, dy = b[..., 0] - a[..., 0], b[..., 1] - a[..., 1]
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def angle_difference(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def point_segment_features(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Exact projected distance and projection fraction on finite segment AB."""
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    ap = p - a
    numer = np.einsum("ij,ij->i", ap, ab)
    fraction = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 1e-12)
    fraction = np.clip(fraction, 0.0, 1.0)
    projected = a + fraction[:, None] * ab
    return np.linalg.norm(p - projected, axis=1), fraction


def load_trip(trip_dir: Path, include_route: bool):
    tid = trip_dir.name
    track = read_numeric(trip_dir / f"{tid}.track", 3)
    nodes = read_numeric(trip_dir / f"{tid}.nodes", 2)
    arcs_float = read_numeric(trip_dir / f"{tid}.arcs", 2)
    if not np.equal(arcs_float, np.floor(arcs_float)).all():
        raise ValueError(f"{tid}: arc endpoints must be integer node IDs")
    arcs = arcs_float.astype(np.int64)
    if len(arcs) and (arcs.min() < 0 or arcs.max() >= len(nodes)):
        raise ValueError(f"{tid}: arc endpoint outside node table")
    route = np.empty(0, dtype=np.int64)
    if include_route:
        route_path = trip_dir / f"{tid}.route"
        route_float = read_numeric(route_path, 1).reshape(-1)
        if not np.equal(route_float, np.floor(route_float)).all():
            raise ValueError(f"{tid}: route entries must be integer arc IDs")
        route = route_float.astype(np.int64)
        if len(route) and (route.min() < 0 or route.max() >= len(arcs)):
            raise ValueError(f"{tid}: route arc ID outside arc table")
    return track, nodes, arcs, route


def motion_features(track: np.ndarray, gps_xy: np.ndarray):
    n = len(track)
    time_delta = np.full(n, np.nan)
    step_distance = np.full(n, np.nan)
    speed = np.full(n, np.nan)
    heading = np.full(n, np.nan)
    if n < 2:
        return time_delta, step_distance, speed, heading
    dts = np.diff(track[:, 2])
    steps = np.linalg.norm(np.diff(gps_xy, axis=0), axis=1)
    bearings = bearing_xy(gps_xy[:-1], gps_xy[1:])
    time_delta[1:] = dts
    step_distance[1:] = steps
    valid_speed = np.isfinite(dts) & (dts > 0)
    speed[1:][valid_speed] = steps[valid_speed] / dts[valid_speed]
    moving = steps > 0.05
    heading[1:][moving] = bearings[moving]
    if moving[0]:
        heading[0] = bearings[0]
    return time_delta, step_distance, speed, heading


def add_training_labels(frame: pd.DataFrame, route: np.ndarray, max_distance_m: float):
    """Attach cautious labels by monotonic matching to the ordered reference route.

    Only candidates occurring at/after the previous matched route position can
    be selected. If no candidate is close enough, all candidates for that GPS
    observation remain label=-1 (unknown), not fabricated negatives.
    """
    out = frame.copy()
    out["label"] = -1
    out["label_status"] = "unmatched_reference"
    if out.empty or len(route) == 0:
        return out

    route_positions: dict[int, list[int]] = {}
    for pos, arc_id in enumerate(route.tolist()):
        route_positions.setdefault(int(arc_id), []).append(pos)

    # Limit route-index jumps so a crossing far ahead in the route cannot
    # pull the alignment forward and make later GPS observations appear out of order.
    gps_count = max(1, int(out["gps_point_id"].nunique()))
    max_route_advance = max(2, min(50, int(math.ceil(12.0 * len(route) / gps_count + 5.0))))
    previous_pos = 0
    for _, indices in out.groupby("gps_point_id", sort=True).groups.items():
        group = out.loc[indices]
        options = []
        for row_index, row in group.iterrows():
            possible = route_positions.get(int(row["candidate_arc_id"]), [])
            offset = bisect_left(possible, previous_pos)
            if offset == len(possible):
                continue
            route_pos = possible[offset]
            distance = float(row["distance_to_road_m"])
            if route_pos <= previous_pos + max_route_advance and distance <= max_distance_m:
                options.append((distance, route_pos, row_index))
        if options:
            options.sort(key=lambda item: (item[0], item[1]))
            _, selected_route_pos, selected_row = options[0]
            out.loc[indices, "label"] = 0
            out.loc[selected_row, "label"] = 1
            out.loc[indices, "label_status"] = "route_aligned_approx"
            previous_pos = selected_route_pos
    return out


def process_trip(trip_dir: Path, partition: str, radius_m: float,
                 max_candidates: int, label_max_distance_m: float):
    tid = trip_dir.name
    track, nodes_lonlat, arcs, route = load_trip(trip_dir, include_route=(partition == "train"))
    if not len(track) or not len(nodes_lonlat) or not len(arcs):
        raise ValueError(f"{tid}: .track, .nodes and .arcs must be non-empty")

    # Metric trip-local projection: do not compute distances in raw degrees.
    lon0, lat0 = float(np.mean(track[:, 0])), float(np.mean(track[:, 1]))
    gps_xy = project_local(track[:, :2], lon0, lat0)
    nodes_xy = project_local(nodes_lonlat, lon0, lat0)
    from_ids, to_ids = arcs[:, 0], arcs[:, 1]
    seg_a, seg_b = nodes_xy[from_ids], nodes_xy[to_ids]
    lengths = np.linalg.norm(seg_b - seg_a, axis=1)
    road_headings = bearing_xy(seg_a, seg_b)

    # STRtree dwithin searches actual finite segments, not just their midpoints.
    geometries = np.asarray([LineString((a, b)) for a, b in zip(seg_a, seg_b)], dtype=object)
    tree = STRtree(geometries)
    gps_points = points(gps_xy[:, 0], gps_xy[:, 1])
    pairs = tree.query(gps_points, predicate="dwithin", distance=radius_m)
    per_point: list[list[int]] = [[] for _ in range(len(track))]
    if pairs.size:
        for gps_index, arc_index in zip(pairs[0].tolist(), pairs[1].tolist()):
            per_point[gps_index].append(arc_index)

    dt, step_m, speed, gps_heading = motion_features(track, gps_xy)
    rows = []
    for gps_index, found in enumerate(per_point):
        fallback = len(found) == 0
        arc_positions = np.asarray(
            [int(tree.nearest(gps_points[gps_index]))] if fallback else found,
            dtype=np.int64,
        )
        gps_repeat = np.repeat(gps_xy[gps_index][None, :], len(arc_positions), axis=0)
        distances, fractions = point_segment_features(gps_repeat, seg_a[arc_positions], seg_b[arc_positions])
        within = distances <= radius_m
        if not within.any():
            within[np.argmin(distances)] = True
        arc_positions, distances, fractions = arc_positions[within], distances[within], fractions[within]
        order = np.argsort(distances, kind="stable")[:max_candidates]
        arc_positions, distances, fractions = arc_positions[order], distances[order], fractions[order]

        for rank, (arc_pos, distance, fraction) in enumerate(zip(arc_positions, distances, fractions), 1):
            gps_h = float(gps_heading[gps_index]) if np.isfinite(gps_heading[gps_index]) else np.nan
            road_h = float(road_headings[arc_pos])
            rows.append({
                "partition": partition, "trip_id": tid, "gps_point_id": gps_index,
                "timestamp_s": float(track[gps_index, 2]),
                "gps_lon": float(track[gps_index, 0]), "gps_lat": float(track[gps_index, 1]),
                "candidate_arc_id": int(arc_pos),
                "from_node_id": int(from_ids[arc_pos]), "to_node_id": int(to_ids[arc_pos]),
                "distance_to_road_m": float(distance),
                "candidate_projection_fraction": float(fraction),
                "candidate_road_length_m": float(lengths[arc_pos]),
                "gps_step_distance_m": float(step_m[gps_index]) if np.isfinite(step_m[gps_index]) else np.nan,
                "time_delta_s": float(dt[gps_index]) if np.isfinite(dt[gps_index]) else np.nan,
                "gps_speed_mps": float(speed[gps_index]) if np.isfinite(speed[gps_index]) else np.nan,
                "gps_heading_deg": gps_h, "road_heading_deg": road_h,
                "heading_difference_deg": angle_difference(gps_h, road_h) if np.isfinite(gps_h) else np.nan,
                "candidate_rank": rank,
                "candidate_count_within_radius": 0 if fallback else len(found),
                "candidate_is_fallback": fallback,
            })

    frame = pd.DataFrame(rows)
    if partition == "train":
        frame = add_training_labels(frame, route, label_max_distance_m)
    else:
        frame["label"] = -1
        frame["label_status"] = "not_used_for_inference"

    if partition == "train" and not frame.empty:
        point_labels = frame.groupby("gps_point_id")["label"].max()
        aligned = int((point_labels == 1).sum())
        unmatched = int((point_labels == -1).sum())
        coverage = aligned / len(point_labels) if len(point_labels) else np.nan
        positive_distance = float(frame.loc[frame.label == 1, "distance_to_road_m"].mean()) if aligned else np.nan
    else:
        aligned = unmatched = 0
        coverage = positive_distance = np.nan

    summary = {
        "partition": partition, "trip_id": tid,
        "gps_observations": len(track), "road_nodes": len(nodes_lonlat), "road_arcs": len(arcs),
        "candidate_rows": len(frame),
        "gps_points_with_candidates": int(frame.gps_point_id.nunique()) if not frame.empty else 0,
        "mean_candidates_per_gps": float(frame.groupby("gps_point_id").size().mean()) if not frame.empty else 0.0,
        "route_aligned_points": aligned, "unmatched_points": unmatched,
        "route_candidate_coverage": coverage, "mean_positive_distance_m": positive_distance,
        "fallback_candidate_points": int(frame.loc[frame.candidate_is_fallback, "gps_point_id"].nunique()) if not frame.empty else 0,
    }
    return frame, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Member 1A cleaned_dataset folder containing train/ and student_test/")
    parser.add_argument("--output-dir", type=Path, default=Path("Member_1B/outputs"))
    parser.add_argument("--radius-m", type=float, default=100.0)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--label-max-distance-m", type=float, default=100.0)
    parser.add_argument("--partitions", nargs="+", choices=("train", "student_test"),
                        default=("train", "student_test"))
    args = parser.parse_args()
    if min(args.radius_m, args.label_max_distance_m) <= 0 or args.max_candidates <= 0:
        parser.error("radius, label-max-distance and max-candidates must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for partition in args.partitions:
        folder = args.data_root / partition
        if not folder.is_dir():
            raise FileNotFoundError(f"Partition directory not found: {folder}")
        trip_dirs = sorted(p for p in folder.iterdir() if p.is_dir())
        print(f"\nProcessing {partition}: {len(trip_dirs)} trips")
        frames = []
        for index, trip_dir in enumerate(trip_dirs, 1):
            frame, summary = process_trip(trip_dir, partition, args.radius_m,
                                          args.max_candidates, args.label_max_distance_m)
            if not frame.empty:
                frames.append(frame)
            summaries.append(summary)
            print(f"[{index}/{len(trip_dirs)}] {trip_dir.name}: {summary['candidate_rows']:,} candidates; "
                  f"route-aligned={summary['route_aligned_points']}")
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        output = args.output_dir / f"{partition}_candidates.csv"
        combined.to_csv(output, index=False)
        print(f"Saved {output} ({len(combined):,} rows)")

    summary_path = args.output_dir / "candidate_generation_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")
    print("Reminder: route-derived labels are approximate; inspect alignment diagnostics before training.")


if __name__ == "__main__":
    main()
