# Week 6 Segmentation Manual Visual QC Review

## Purpose

This note records the human visual QC outcome for the preliminary segmentation baseline. The high-risk contact sheet was prioritized because it contains the automatically risk-ranked hardest frames.

## Verdict

- High-risk sheet score: `4/5`
- Verdict: `accepted_for_preliminary_feature_extraction_with_notes`
- Fix required before final package: `False`

## Observations

Most red segmentation contours roughly follow pig bodies even in the high-risk sample. Some contours include pen bars, floor/background regions, or partial/fragmented pig body regions under occlusion. The quality is acceptable for preliminary segmentation-derived feature extraction.

## Outputs

- Manual review results: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_segmentation_visual_qc_manual_review_results.csv`
- Frame-level review results: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_segmentation_visual_qc_frame_level_review_results.csv`
- Sheet-level verdict: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_segmentation_visual_qc_sheet_level_verdict.csv`
- Final review summary: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_segmentation_visual_qc_final_review_summary.csv`

## Interpretation

The segmentation baseline is accepted for preliminary shape, posture, contact, foreground, and ROI-related feature extraction. It should not be described as manually annotated segmentation ground truth.
