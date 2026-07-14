# Week6_Unibo_Dataset_Validation_Final_Package_v2

This package contains the final Week 6 Unibo dataset validation deliverables.

## Contents

- Unified ground-truth CSV/JSON/schema
- Scanpoint frame index and nested viewer JSON
- Dataset statistics and split protocol outputs
- Visualization interface demo files
- Detection, marker, feature extractor, embedding, and segmentation feature tables
- Segmentation visual QC outputs
- Task compliance and quality-hardening audits
- Shared Excel task tracker
- Final report and executive summary notes

## Key counts

- Manual labels: 432
- Scanpoint frames: 72
- Detector bboxes: 540
- Detector-backed embedding rows: 540
- Embedding dimension: 896
- Segmentation feature rows: 540
- Recommended split: 288 train / 72 validation / 72 test labels

## Quality status

- Task-sheet compliance v2: all items done.
- Quality hardening v2: no FAIL or WARN items remain.
- Segmentation visual QC: accepted for preliminary feature extraction with notes.
- Learned embeddings: detector-backed, not random/untrained.

## Scope note

The package does not include raw videos. It includes derived frame-level outputs, feature tables, visual QC artifacts, and documentation.
