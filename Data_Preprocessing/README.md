# Member 1A: Dataset analysis and cleaning

This folder contains the dataset audit, conservative cleaning script, trip inventory, EDA notebook, and cleaned copies of the `train` and `student_test` partitions.

## Folder contents

- `dataset_audit.py` — standard-library audit for file structure, numerically equivalent duplicate rows, GPS/node coordinate bounds, integer IDs, arc endpoints, zero-length arcs, weak connectivity, route references, and directed route transitions.
- `data_cleaning.py` — writes a cleaned copy, deduplicating numeric GPS observations while preserving the first original text record, node/arc IDs, and route labels.
- `trip_inventory.csv` — per-trip audit of the source data.
- `gps_gap_report.csv` — per-interval gap class for all non-regular timestamp steps; source observations remain untouched.
- `route_issues.csv` — specific malformed, invalid-reference, and non-contiguous route/arc rows for review.
- `cleaned_dataset/` — cleaned `train/` and `student_test/` trip files plus `cleaning_log.csv`. Running the cleaner writes a detailed `gps_gap_report.csv` into its output folder.
- `cleaned_dataset_audit.csv` — audit of the cleaned copy.
- `eda.ipynb` — dataset summaries, plots, trajectory visualization, and a Folium map.

## Run from this folder

Extract the supplied ZIP so the data root contains `train/` and `student_test/`, then run:

```powershell
python dataset_audit.py --data-root "path/to/Student folder" --output trip_inventory.csv
python data_cleaning.py --data-root "path/to/Student folder" --output-root cleaned_dataset
```

The cleaning command expects the output folder to be new or empty. Install notebook dependencies with `python -m pip install -r requirements.txt`, then open `eda.ipynb` in Jupyter and run its cells.

## Cleaning policy and findings

The cleaner normalizes numeric GPS values for duplicate detection (for example, `12.5 45.2 100` and `12.500 45.2 100.0` are the same observation), while preserving the first record's original text in the output. It removes malformed/non-finite rows, out-of-range coordinates, and numeric duplicate GPS records. It preserves GPS row order and reports non-increasing timestamps instead of silently sorting a trajectory. It copies `.nodes`, `.arcs`, and `.route` without changing them because node row positions are node IDs, arc row positions are arc IDs, and route entries refer to arc IDs. Duplicate node coordinates and duplicate directed arcs are audited but not deduplicated because the dataset does not establish that they are accidental.

Timestamp intervals are classified as regular (0-2s), irregular (>2-10s), large (>10-60s), extreme (>60s), or non-increasing. Both audit and cleaner write per-interval reports. Every observation is preserved across a gap; no points are interpolated. Downstream code must explicitly handle the gap class when deriving speed, direction, alignment, or transitions.

The audit validates all route entries as integer arc IDs, preserves row positions while checking references, and reports non-integer values, out-of-range IDs, references to malformed arcs, invalid node endpoints, self-loops, duplicate directed arcs, zero-length road segments, and directed route-transition relationships. A discontinuity is classified (reverse-direction match, other shared endpoint, or no shared endpoint) and written to `route_issues.csv`. The cleaner does not alter supplied reference routes; the team must review flagged cases before using them as training labels.

For the supplied files, the audit found no malformed/non-integer arc or route rows, invalid GPS or node coordinates, invalid arc endpoints, or route references outside the arc table. It found 100 zero-length arc geometries and 1,775 duplicate directed arc pairs in training, plus 6 zero-length geometries and 128 duplicate directed arc pairs in student test. There are 13 training route transitions and 1 student-test transition for review: 13 transitions have no shared endpoint and 1 shares an endpoint in a non-continuing orientation. The gap report classifies 2,937 training intervals and 138 student-test intervals above 2 seconds; this includes 69 and 1 intervals over 60 seconds, respectively. These counts are flags for investigation, not automatic deletions.

This is Member 1A's data auditing and cleaning stage. Candidate generation, observation-to-route training labels, feature-level validation, and model-ready candidate tables belong to the separate Member 1B stage; this folder does not claim to produce those outputs.

## Partition handling

This package contains only the supplied training and student-testing partitions. The student-testing partition is for the assignment's development-stage evaluation, not model fitting. Evaluator-testing data is not present here and must remain separate when provided.

