# Week 6 Segmentation Visual QC Package

## Purpose

This step creates a visual quality-control package for the preliminary segmentation baseline. It does not automatically declare segmentation masks correct; instead, it prepares risk-ranked overlays for manual review.

## Outputs

- Full frame QC index: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_segmentation_visual_qc_frame_index.csv`
- Manual review sample: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_segmentation_visual_qc_sample_for_manual_review.csv`
- Manual review template: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_segmentation_visual_qc_manual_review_template.csv`
- High-risk contact sheet: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/segmentation_visual_qc/segmentation_visual_qc_high_risk_contact_sheet.jpg`
- Manual review sample contact sheet: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/segmentation_visual_qc/segmentation_visual_qc_manual_review_sample_contact_sheet.jpg`
- Representative contact sheet: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/segmentation_visual_qc/segmentation_visual_qc_representative_contact_sheet.jpg`
- Static HTML reviewer: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/visual_label_check/segmentation_visual_qc/segmentation_visual_qc_static_reviewer.html`
- Summary: `/home/oyavuz/PigBench/Week6_Unibo_Dataset_Validation/outputs/dataset_statistics/week6_segmentation_visual_qc_summary.csv`

## Summary

| metric                     |   value | interpretation                                                            |
|:---------------------------|--------:|:--------------------------------------------------------------------------|
| total_segmentation_frames  |      72 | All scanpoint frames with segmentation overlays.                          |
| overlay_exists_count       |      72 | Frames with overlay image available.                                      |
| high_risk_frame_count      |      55 | Frames with visual QC risk score >= 3.                                    |
| manual_review_sample_count |      33 | Frames selected for manual visual review.                                 |
| manual_review_required     |    True | Human visual confirmation is needed before claiming segmentation quality. |

## What needs human review

Open the contact sheets or the static HTML reviewer and inspect whether the red segmentation contours roughly follow pig bodies. Use the manual review template to record obvious failures, acceptable masks, and frames that need correction.

## Interpretation

This package upgrades the segmentation baseline from pure automatic output to an inspectable QC artifact. Final segmentation quality should be reported based on the manual visual review outcome.
