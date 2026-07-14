# Week 6 Unibo Dataset Validation — Executive Summary v2

## Final status

The Week 6 Unibo dataset validation work is task-sheet complete and quality-hardened.

## Main outputs

- 432 manual behaviour labels extracted into unified GT CSV/JSON.
- 72 scanpoint frames extracted.
- 540 YOLOv8-s detector bboxes generated and QC-flagged.
- Recommended no-leakage split v2 created: 288 train / 72 validation / 72 test labels.
- Visualization interface created with Streamlit and static HTML viewer.
- Feature extractor outputs created for bbox geometry, group-spatial context, ROI proxies, crop descriptors, marker evidence, detector-backed learned embeddings, and segmentation-derived features.
- Detector-backed learned crop embeddings were generated for all 540 crops with 896-dimensional vectors.
- Preliminary bbox-guided segmentation features were generated and visually QC-reviewed.
- Shared Excel/task tracker was generated and verified as readable.
- Final task-sheet compliance audit v2 reports all 14 items done.
- Quality hardening audit v2 reports 24 PASS and no WARN/FAIL items.

## Important interpretation

The outputs are suitable for experimental behaviour-classification dataset preparation and feature comparison. Automatic bboxes, colour-marker matches, and segmentation masks are feature-extraction artifacts, not manually validated ground-truth annotations. This distinction is intentionally preserved throughout the deliverables.
