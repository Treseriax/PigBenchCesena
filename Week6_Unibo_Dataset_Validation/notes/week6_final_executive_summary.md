# Week 6 Final Executive Summary

The Week 6 Unibo dataset validation pipeline successfully produced a unified and auditable dataset workflow. The final recommended ground-truth table contains 432 manual scan-sampling behaviour labels across 72 extracted scanpoint frames. Each scanpoint frame has six manual colour-ID labels. YOLOv8-s detection produced 540 pig bounding boxes across all 72 frames, and the primary detection table was preserved while QC flags and conservative subsets were added. Crop-based colour-marker analysis produced 196 medium/high candidate bbox-to-colour associations, which are useful for visual QC but not final identity-resolved ground truth. The recommended split v2 uses video-hour units to avoid leakage and contains 288 train labels, 72 validation labels, and 72 test labels. Final post-fix consistency audit passed with 0 failed checks. The dataset is ready for reporting, visualization demo, and future feature-extractor experiments.

## Main limitations

- Labels come from one Excel sheet/day/camera-pen context.
- 15:00-19:00 c-token video mappings are candidate recovered mappings and should remain marked as medium confidence.
- Bboxes are automatic detector outputs, not manual bbox annotations.
- Bbox-to-colour identity is candidate-level only; red markers are ambiguous and no_color cannot be marker-detected.
- Rare behaviour classes remain limited and require careful handling in future classification experiments.
