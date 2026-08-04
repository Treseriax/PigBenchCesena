# Week 8 Manual Visual Validation Protocol

## Purpose

This protocol validates the 72 annotated observation clips before behaviour classification. The goal is not to annotate new raw videos. The goal is to verify the existing pig-colour-behaviour associations and the hybrid tracking support layer.

## Data scope

- Raw video archive is larger than the validated dataset.
- Week 8 validation uses the 72 annotated scanframe clips.
- Each scanframe corresponds to a 10-second observation window.
- Anchor GT remains the primary reference.
- Hybrid tracking is a validation support layer:
  - stable_tracklet = primary temporal evidence
  - recall_fallback = visible but review-needed
  - missing = preserved, not hidden

## Review priority

1. High priority clips:
   - challenging tracking support
   - high missing ratio
   - high review-needed ratio

2. Medium priority clips:
   - limited tracking support
   - moderate missing/review ratio

3. Normal review clips:
   - usable with review

4. Strong spot-check clips:
   - strong tracking support
   - only start/middle/end check required

## What to check in the v52d visualizer

For each priority clip:

- Anchor GT alignment at the annotation frame
- Stable tracklet boxes remain on the correct pig
- Recall fallback boxes are not misleading
- Missing pigs are correctly not drawn
- Colour identity remains plausible
- Behaviour label is plausible for the visible pig
- No obvious false positive outside the pen
- No identity switch between pigs

## Required note types

Use the v52d interface note panel with one of:

- bbox_ok
- bbox_wrong
- identity_uncertain
- identity_switch
- behaviour_uncertain
- missing_pig
- false_positive
- other

## Recommended inspection times

For each reviewed clip, inspect approximately:

- 0.0 s
- 2.5 s
- 5.0 s
- 7.5 s
- 9.5 s

If the clip has a visible issue, pause exactly where it occurs and save a note.

## Outputs

- Review plan: `/home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v52e_manual_visual_validation_protocol/week8_v52e_manual_review_plan.csv`
- Checklist: `/home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v52e_manual_visual_validation_protocol/week8_v52e_visual_validation_checklist.csv`
- Strong spot-check plan: `/home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v52e_manual_visual_validation_protocol/week8_v52e_strong_clip_spotcheck_plan.csv`
- Priority order: `/home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v52e_manual_visual_validation_protocol/week8_v52e_priority_review_order.csv`
- Notes saved by interface: `/home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/validation/week8_v52d_visual_validation_notes.csv`

## Classification policy

Classification can proceed after this validation phase as a baseline/pilot experiment. The dataset is not claimed to support a production-level final classifier. Rare behaviour classes should remain limited/report-only unless additional annotation is later requested.
