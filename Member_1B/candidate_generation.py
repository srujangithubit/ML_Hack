"""PathSnap Member 1B: candidate generation and feature engineering.

Input layout:
  cleaned_dataset/{train,student_test}/{trip_id}/{trip_id}.{track,nodes,arcs,route}

The script creates one row per (GPS observation, candidate directed arc).
It never uses route labels for candidate selection or for inference features.
For train only, labels are attached by a conservative route-order alignment:
the route arc must appear among the candidates and its segment must lie within
LABEL_MAX_DISTANCE_M. Ambiguous/unmatched observations get label=-1 and are
excluded from supervised training by default.

Dependencies: numpy, pandas, scipy
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EARTH_R = 6_371_008.8
DEFAULT_RADIUS_M = 60.0
DEFAULT_MAX_CANDIDATES = 8
LABEL_MAX_DISTANCE_M = 60.0


def read_numeric(path: Path, cols: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.loadtxt(path, dtype=float, ndmin=2)
    if arr.size == 0:
        return np.empty((0, cols), dtype=float)
    if arr.shape[1] != cols:
        raise ValueError(f"{path}: expected {cols} columns, got {arr.shape[1]}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{path}: non-finite values found; run Member 1A cleaning first")
    return arr


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_deg(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def angle_diff_deg(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def local_xy(lon, lat, lon0, lat0):
    """Local equirectangular projection in metres, accurate for a trip area."""
    x = np.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_R
    return np.column_stack((x, y))


def point_segment_projection(px, py, ax, ay, bx, by):
    """Vector projection of point P onto finite segment AB; returns distance and t."""
    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        t = np.zeros_like(px)
    else:
        t = np.clip(((px - ax) * vx + (py - ay) * vy) / denom, 0.0, 1.0)
    qx, qy = ax + t * vx, ay + t * vy
    return np.hypot(px - qx, py - qy), t


def load_trip(trip_dir: Path):
    tid = trip_dir.name
    track = read_numeric(trip_dir / f"{tid}.track", 3)
    nodes = read_numeric(trip_dir / f"{tid}.nodes", 2)
    arcs = read_numeric(trip_dir / f"{tid}.arcs", 2).astype(np.int64)
    route_path = trip_dir / f"{tid}.route"
    route = read_numeric(route_path, 1).reshape(-1).astype(np.int64) if route_path.exists() else np.empty(0, dtype=np.int64)

    if len(arcs) and (arcs.min() < 0 or arcs.max() >= len(nodes)):
        raise ValueError(f"{tid}: arc endpoint outside node table")
    if len(route) and (route.min() < 0 or route.max() >= len(arcs)):
        raise ValueError(f"{tid}: route arc ID outside arc table")
    return track, nodes, arcs, route


def build_candidates(track, nodes, arcs, trip_id, partition,
                     radius_m=DEFAULT_RADIUS_M, max_candidates=DEFAULT_MAX_CANDIDATES):
    if len(track) == 0 or len(nodes) == 0 or len(arcs) == 0:
        return pd.DataFrame()

    lon0, lat0 = float(np.mean(track[:, 0])), float(np.mean(track[:, 1]))
    node_xy = local_xy(nodes[:, 0], nodes[:, 1], lon0, lat0)
    gps_xy = local_xy(track[:, 0], track[:, 1], lon0, lat0)

    start_xy = node_xy[arcs[:, 0]]
    end_xy = node_xy[arcs[:, 1]]
    midpoint_xy = (start_xy + end_xy) / 2.0
    half_lengths = np.linalg.norm(end_xy - start_xy, axis=1) / 2.0

    # Query segment midpoints using radius + maximum half segment length.
    tree = cKDTree(midpoint_xy)
    max_half = float(half_lengths.max()) if len(half_lengths) else 0.0
    search_r = radius_m + max_half
    neighbor_lists = tree.query_ball_point(gps_xy, r=search_r)

    timestamps = track[:, 2]
    n = len(track)
    gps_speed = np.zeros(n, dtype=float)
    gps_heading = np.zeros(n, dtype=float)
    time_delta = np.zeros(n, dtype=float)
    gps_step_distance = np.zeros(n, dtype=float)

    for i in range(n):
        j = i + 1 if i < n - 1 else i - 1
        if n < 2:
            continue
        dt = timestamps[j] - timestamps[i]
        if j < i:
            dt = -dt
        if dt <= 0:
            continue
        d = float(haversine_m(track[i, 0], track[i, 1], track[j, 0], track[j, 1]))
        gps_step_distance[i] = d
        time_delta[i] = dt
        gps_speed[i] = d / dt
        gps_heading[i] = float(bearing_deg(track[i, 0], track[i, 1], track[j, 0], track[j, 1]))

    rows = []
    for i, arc_candidates in enumerate(neighbor_lists):
        if not arc_candidates:
            continue
        ids = np.asarray(arc_candidates, dtype=np.int64)
        px, py = gps_xy[i]
        dist, frac = point_segment_projection(
            np.full(len(ids), px), np.full(len(ids), py),
            start_xy[ids, 0], start_xy[ids, 1], end_xy[ids, 0], end_xy[ids, 1]
        )
        keep = dist <= radius_m
        ids, dist, frac = ids[keep], dist[keep], frac[keep]
        if len(ids) == 0:
            continue

        road_heading = bearing_deg(
            nodes[arcs[ids, 0], 0], nodes[arcs[ids, 0], 1],
            nodes[arcs[ids, 1], 0], nodes[arcs[ids, 1], 1]
        )
        hd = angle_diff_deg(gps_heading[i], road_heading)
        order = np.lexsort((hd, dist))[:max_candidates]
        ids, dist, frac, road_heading, hd = ids[order], dist[order], frac[order], road_heading[order], hd[order]

        for rank, (aid, d, t, rh, hdiff) in enumerate(zip(ids, dist, frac, road_heading, hd), start=1):
            rows.append({
                "partition": partition,
                "trip_id": trip_id,
                "gps_point_id": i,
                "timestamp": float(timestamps[i]),
                "gps_lon": float(track[i, 0]),
                "gps_lat": float(track[i, 1]),
                "candidate_arc_id": int(aid),
                "distance_to_road_m": float(d),
                "gps_step_distance_m": float(gps_step_distance[i]),
                "time_delta_s": float(time_delta[i]),
                "gps_speed_mps": float(gps_speed[i]),
                "gps_heading_deg": float(gps_heading[i]),
                "road_heading_deg": float(rh),
                "heading_difference_deg": float(hdiff),
                "candidate_fraction": float(t),
                "candidate_rank": int(rank),
                "candidate_count": int(len(ids)),
                "from_node_id": int(arcs[aid, 0]),
                "to_node_id": int(arcs[aid, 1]),
            })
    return pd.DataFrame(rows)


def attach_training_labels(candidates: pd.DataFrame, route: np.ndarray,
                           max_label_distance_m=LABEL_MAX_DISTANCE_M) -> pd.DataFrame:
    """Conservative route-order alignment, not a GPS-to-route row-index assumption.

    For each GPS point, use the closest route arc that is among nearby candidates,
    while preserving non-decreasing route position. If no plausible route arc is
    present, retain label=-1 (unknown). The caller excludes unknown labels.
    """
    if candidates.empty:
        candidates["label"] = pd.Series(dtype="int64")
        candidates["label_status"] = pd.Series(dtype="object")
        return candidates

    candidates = candidates.copy()
    candidates["label"] = -1
    candidates["label_status"] = "unmatched"

    route_positions: dict[int, list[int]] = {}
    for pos, arc_id in enumerate(route.tolist()):
        route_positions.setdefault(int(arc_id), []).append(pos)

    prev_route_pos = 0
    for gps_id, indices in candidates.groupby("gps_point_id", sort=True).groups.items():
        group = candidates.loc[indices]
        plausible = group[group["distance_to_road_m"] <= max_label_distance_m]
        options = []
        for idx, row in plausible.iterrows():
            for rp in route_positions.get(int(row["candidate_arc_id"]), []):
                if rp >= prev_route_pos:
                    options.append((float(row["distance_to_road_m"]), rp, idx))
        if not options:
            continue
        options.sort()
        best_distance, best_pos, best_idx = options[0]
        candidates.loc[indices, "label"] = 0
        candidates.loc[best_idx, "label"] = 1
        candidates.loc[indices, "label_status"] = "aligned"
        prev_route_pos = best_pos
    return candidates


def process_partition(root: Path, partition: str, output_dir: Path,
                      radius_m: float, max_candidates: int):
    part_dir = root / partition
    if not part_dir.is_dir():
        raise FileNotFoundError(f"Missing partition: {part_dir}")
    outputs = []
    summary = []
    trips = sorted(p for p in part_dir.iterdir() if p.is_dir())
    for trip_dir in trips:
        track, nodes, arcs, route = load_trip(trip_dir)
        df = build_candidates(track, nodes, arcs, trip_dir.name, partition,
                               radius_m=radius_m, max_candidates=max_candidates)
        if partition == "train":
            df = attach_training_labels(df, route)
        else:
            df["label"] = -1
            df["label_status"] = "not_available"
        outputs.append(df)
        summary.append({
            "partition": partition,
            "trip_id": trip_dir.name,
            "gps_points": len(track),
            "candidate_rows": len(df),
            "gps_points_with_candidates": int(df["gps_point_id"].nunique()) if not df.empty else 0,
            "positive_labels": int((df.get("label", pd.Series(dtype=int)) == 1).sum()),
            "unmatched_gps_points": int((df.groupby("gps_point_id")["label"].max() == -1).sum()) if partition == "train" and not df.empty else 0,
        })
        print(f"[{partition}] {trip_dir.name}: GPS={len(track):,}, candidates={len(df):,}")
    all_df = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(output_dir / f"{partition}_candidates.csv", index=False)
    pd.DataFrame(summary).to_csv(output_dir / f"{partition}_candidate_summary.csv", index=False)
    return all_df, pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Path to Data_Preprocessing/cleaned_dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("Member_1B/outputs"))
    parser.add_argument("--radius-m", type=float, default=DEFAULT_RADIUS_M)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--partitions", nargs="+", default=["train", "student_test"],
                        choices=["train", "student_test"])
    args = parser.parse_args()
    if args.radius_m <= 0 or args.max_candidates <= 0:
        parser.error("--radius-m and --max-candidates must be positive")

    for partition in args.partitions:
        process_partition(args.data_root, partition, args.output_dir,
                          args.radius_m, args.max_candidates)
    print(f"Done. Outputs saved under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
