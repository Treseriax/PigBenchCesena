# Week 7 Colour Rule Workspace v13

## Purpose

This step prepares pen/crate-specific colour rules before final colour identity assignment. The v12 marker candidates are visual evidence only; they are not final labels. Colour assignment must be constrained by the valid colour set for each annotated pen/crate.

## Main rule

Do not assign a colour globally from HSV evidence alone. For each scan frame, first verify the allowed colour set for that pen/crate, then match pig boxes to those allowed colours.

## Summary

- Frame rule rows: `72`
- Rules with six suggested colours from existing GT: `14`
- Rules needing manual confirmation: `58`
- Box marker/rule review rows: `429`

## Outputs

- Source audit: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/colour_identity/colour_rule_workspace_v13/week7_colour_rule_workspace_v13_source_audit.csv`
- Frame colour rule template: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/colour_identity/colour_rule_workspace_v13/week7_colour_rule_workspace_v13_frame_colour_rule_template.csv`
- Allowed colour dictionary: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/colour_identity/colour_rule_workspace_v13/week7_colour_rule_workspace_v13_allowed_colour_dictionary.csv`
- Box marker/rule review table: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/colour_identity/colour_rule_workspace_v13/week7_colour_rule_workspace_v13_box_marker_rule_review.csv`

## Next step

Manually verify or fill the allowed colours in the frame colour rule template. After that, run the colour identity assignment step.
