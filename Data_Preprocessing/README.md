# Member 1A: Dataset analysis and cleaning

This folder contains the dataset audit, conservative cleaning script, trip inventory, EDA notebook, and cleaned copies of the `train` and `student_test` partitions.

## Folder contents

- `dataset_audit.py` — standard-library audit for file structure, malformed rows, exact duplicates, GPS bounds, timestamp gaps/order, road-arc endpoints, weak connectivity, isolated nodes, and route arc IDs.
- `data_cleaning.py` — writes a cleaned copy while preserving node/arc row IDs and route labels.
- `trip_inventory.csv` — per-trip audit of the source data.
- `cleaned_dataset/` — cleaned `train/` and `student_test/` trip files plus `cleaning_log.csv`.
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

The cleaner removes malformed/non-finite GPS rows, out-of-range coordinates, and exact duplicate GPS records. It preserves GPS row order and reports non-increasing timestamps instead of silently sorting a trajectory. It copies `.nodes`, `.arcs`, and `.route` without changing them because node row positions are node IDs, arc row positions are arc IDs, and route entries refer to arc IDs. Duplicate node coordinates and duplicate directed arcs are therefore audited but not deduplicated.

The supplied data had no malformed rows, invalid GPS coordinates, non-increasing timestamp pairs, out-of-range arc endpoints, route IDs outside the arc table, or isolated nodes. Exact duplicate GPS rows were also absent, so the cleaned GPS files retain all 208,417 training and 19,160 student-testing observations. The audit found 89 duplicate node rows and 1,775 duplicate arc rows in training, and 3 duplicate node rows and 128 duplicate arc rows in student test; these remain unchanged to preserve IDs.

There are long GPS gaps and a small number of route transitions that do not directly join according to directed arc endpoints. The inventory records these for the team to review. The cleaner does not interpolate gaps, drop observations because of long gaps, or repair reference routes.

## Partition handling

This package contains only the supplied training and student-testing partitions. The student-testing partition is for the assignment's development-stage evaluation, not model fitting. Evaluator-testing data is not present here and must remain separate when provided.
