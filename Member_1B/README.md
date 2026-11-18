# Member 1B — Candidate Generation & Feature Engineering

This module converts Member 1A's cleaned trips into a candidate-level ML dataset.

## Input layout

- Data_Preprocessing/cleaned_dataset/train/<trip_id>/<trip_id>.track
- Data_Preprocessing/cleaned_dataset/train/<trip_id>/<trip_id>.nodes
- Data_Preprocessing/cleaned_dataset/train/<trip_id>/<trip_id>.arcs
- Data_Preprocessing/cleaned_dataset/train/<trip_id>/<trip_id>.route
- Data_Preprocessing/cleaned_dataset/student_test/<trip_id>/... (no route labels are read)

The row index in .nodes is the node ID. The row index in .arcs is the directed candidate_arc_id; its two columns contain from_node_id and to_node_id.

## Install and run (from repository root)

    python -m pip install -r Member_1B/requirements.txt
    python Member_1B/candidate_generation.py --data-root Data_Preprocessing/cleaned_dataset --output-dir Member_1B/outputs

Optional parameters:

    --radius-m 100 --max-candidates 8 --label-max-distance-m 100

## Outputs

- outputs/train_candidates.csv: candidate-level features plus approximate training labels.
- outputs/student_test_candidates.csv: inference features only; labels are -1 and the .route file is not read.
- outputs/candidate_generation_summary.csv: trip-level candidate coverage and alignment diagnostics.

## Features

GPS position/time; directed candidate arc ID and endpoint node IDs; perpendicular distance to the finite road segment; projection fraction; road length; GPS step distance; time delta; GPS speed; GPS heading; road heading; circular heading difference; candidate rank/count; fallback indicator.

All spatial distances use a trip-local metric projection in metres, not raw latitude/longitude degrees. Candidate generation uses a Shapely STRtree dwithin query on actual line segments, so long roads are not missed just because their midpoints are far from a GPS point.

## Label and leakage policy

Candidates and inference features are generated from GPS observations and the road network only. Only after candidate generation, training .route arc IDs are aligned monotonically to candidate arc IDs. An observation with no plausible route-matching candidate is marked unknown (label=-1), rather than assigning fabricated negatives. The supplied route is reference information and may not be perfect; inspect alignment diagnostics before training.

For student-test and evaluator-test predictions, do not use .route labels or route-derived information. Keep evaluator ground truth outside the prediction pipeline.
