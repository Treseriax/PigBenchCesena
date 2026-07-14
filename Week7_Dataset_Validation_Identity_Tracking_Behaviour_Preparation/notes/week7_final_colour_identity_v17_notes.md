# Week 7 Final Colour Identity Lock v17

## Purpose

This step locks the manually corrected colour identity table for behaviour-label fusion.

## Policy

- Valid colour labels are kept as usable identities.
- `not_visible` is kept as `identity_unknown_not_visible`.
- `uncertain` is kept as `identity_unknown_uncertain`.
- Duplicate colours or unassigned boxes are hard issues.

## Summary

- Total boxes: `429`
- Usable colour identity boxes: `0`
- Not-visible unknown boxes: `0`
- Uncertain unknown boxes: `0`
- Unassigned boxes: `429`
- Frames with hard issue: `72`
- Frames with soft issue: `0`
- Ready for behaviour fusion: `False`

## Outputs

- Locked assignments: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/colour_identity/final_colour_identity_v17/week7_final_colour_identity_v17_locked_assignments.csv`
- Frame QA: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/colour_identity/final_colour_identity_v17/week7_final_colour_identity_v17_frame_qa.csv`
- Hard issues: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/colour_identity/final_colour_identity_v17/week7_final_colour_identity_v17_hard_issues.csv`
- Summary: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/colour_identity/final_colour_identity_v17/week7_final_colour_identity_v17_summary.csv`
