# Member 1B: Candidate Generation and Feature Engineering

Creates candidate road arcs for each GPS observation and computes ML features.

## Input
Use Member 1A's cleaned dataset:
\`Data_Preprocessing/cleaned_dataset/{train,student_test}/{trip_id}/{trip_id}.{track,nodes,arcs,route}\`

## Install
\`\`\`bash
python -m pip install numpy pandas scipy
\`\`\`

## Run from repository root
\`\`\`bash
python Member_1B/candidate_generation.py --data-root Data_Preprocessing/cleaned_dataset --output-dir Member_1B/outputs
\`\`\`

Optional parameters:
\`\`\`bash
--radius-m 60 --max-candidates 8
\`\`\`

## Outputs
- \`outputs/train_candidates.csv\`: candidate-level features and training labels.
- \`outputs/student_test_candidates.csv\`: candidate-level features; labels are -1 and not used for fitting.
- Per-partition candidate summaries.

## Feature definitions
- Distance to finite directed road segment in metres (local equirectangular projection).
- GPS step distance and speed (m/s), time delta, GPS heading and road heading (degrees).
- Circular heading difference, projection fraction along the segment, candidate rank/count.
- Directed arc endpoint IDs for later HMM transition modelling.

## Label safety
Training labels are assigned only after candidates are generated. The reference route is never used to select candidates or compute inference features. The labeler aligns route arc IDs in non-decreasing route order and leaves observations unmatched where no plausible candidate aligns. Review unmatched/ambiguous points before supervised training.

Do not use \`.route\` from student-test or evaluator-test to create predictions or features.
