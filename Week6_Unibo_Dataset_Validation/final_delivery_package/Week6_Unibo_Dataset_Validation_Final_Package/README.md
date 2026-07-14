# Week 6 Unibo Dataset Validation Final Package

Generated: 2026-07-04T20:20:46

## Contents

- `00_report_notes/`: executive summary, report outline, audit and split notes.
- `01_ground_truth/`: final recommended GT table and scanpoint frame/label alignment tables.
- `02_statistics_split_qc/`: dataset statistics, recommended split v2, final audit, detector QC summaries.
- `03_feature_extractors/`: YOLOv8-s bbox features, conservative subset, colour-marker features, feature extractor comparison.
- `04_visualizations/`: marker-bbox demo slideshow and bbox warning QC contact sheets.
- `MANIFEST.csv`: package file list with copy status and checksums.
- `CHECKSUMS_SHA256.txt`: SHA256 checksums for copied files.

## Key result summary

- Final manual labels: 432
- Scanpoint frames: 72
- Manual labels per frame: 6
- YOLOv8-s primary bboxes: 540
- Medium/high marker candidates: 196
- Recommended split v2: train 288, val 72, test 72 labels
- Final post-fix audit: 0 failed checks

## Important limitations

- Labels come from one Excel sheet/day/camera-pen context.
- 15:00-19:00 c-token video mappings are candidate recovered mappings and should remain medium confidence.
- Bboxes are automatic detector outputs, not manual bbox annotations.
- Bbox-to-colour identity is candidate-level only; red markers are ambiguous and no_color cannot be marker-detected.
- Rare behaviour classes remain limited.

## Missing files

No missing files in selected final package.
