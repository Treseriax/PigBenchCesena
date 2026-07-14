# Week 7 Final Colour Identity Lock v17 Fixed

## Purpose

This corrected v17 step locks the manually corrected colour identity table for behaviour-label fusion.

## Fix applied

The previous v17 script incorrectly merged v11 and v16 using exact box coordinates. This failed because v16 uses clamped/integer coordinates while v11 may preserve original coordinate values. The corrected version merges on stable keys only: `scan_frame_id` and `final_box_id`.

## Policy

- Valid colour labels are kept as usable identities.
- `not_visible` is kept as `identity_unknown_not_visible`.
- `uncertain` is kept as `identity_unknown_uncertain`.
- Duplicate colours, unassigned boxes, or failed v16 matches are hard issues.

## Summary

- Total boxes: `429`
- v16 matched boxes: `429`
- Usable colour identity boxes: `374`
- Not-visible unknown boxes: `54`
- Uncertain unknown boxes: `1`
- Unassigned boxes: `0`
- Frames with hard issue: `0`
- Frames with soft issue: `19`
- Ready for behaviour fusion: `True`

## Outputs

- Locked assignments: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/colour_identity/final_colour_identity_v17_fixed/week7_final_colour_identity_v17_fixed_locked_assignments.csv`
- Frame QA: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/colour_identity/final_colour_identity_v17_fixed/week7_final_colour_identity_v17_fixed_frame_qa.csv`
- Hard issues: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/colour_identity/final_colour_identity_v17_fixed/week7_final_colour_identity_v17_fixed_hard_issues.csv`
- Soft issues: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/colour_identity/final_colour_identity_v17_fixed/week7_final_colour_identity_v17_fixed_soft_issues.csv`
- Summary: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/colour_identity/final_colour_identity_v17_fixed/week7_final_colour_identity_v17_fixed_summary.csv`
