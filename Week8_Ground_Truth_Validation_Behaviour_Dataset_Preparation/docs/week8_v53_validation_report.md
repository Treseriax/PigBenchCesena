# Week 8 Validation Report

## Validation status

- v53 decision: gt_documentation_and_validation_report_completed
- Ready for v54 final delivery package: True

## Propagation and GT validation

The dataset uses a 10-second observation-window propagation policy. Behaviour labels are not treated as isolated single-frame labels; they apply to the observation interval.

## Hybrid tracking validation

The hybrid tracking layer was created to support visual validation of pig-colour-behaviour associations.

Global tracking-support metrics:

- Mean stable tracklet ratio: 0.7582
- Mean recall fallback ratio: 0.1385
- Mean missing ratio: 0.1033
- Mean draw-ok ratio: 0.8967
- Mean review-needed ratio: 0.2418

## Clip quality tiers

| quality_tier                     |   clip_count |   mean_draw_ok_ratio |   mean_missing_ratio |   mean_review_needed_ratio |
|:---------------------------------|-------------:|---------------------:|---------------------:|---------------------------:|
| strong_visual_tracking_support   |           33 |             0.97871  |            0.0212897 |                   0.114227 |
| usable_with_review               |           30 |             0.870632 |            0.129368  |                   0.310534 |
| limited_review_required          |            7 |             0.714395 |            0.285605  |                   0.431935 |
| challenging_low_tracking_support |            2 |             0.574333 |            0.425667  |                   0.649    |

## Manual validation

High-priority clips identified by automatic tracking QA were manually reviewed in the v52d visualizer.

- High-priority clips: 3
- High-priority reviewed: 3
- High-priority accepted: 3
- Manual issue found clips: 0
- Manual notes saved: 3

Manual validation status summary:

| manual_priority_level   | manual_status         | accepted_for_current_stage   |   clip_count |
|:------------------------|:----------------------|:-----------------------------|-------------:|
| high_priority           | manual_checked_ok     | True                         |            3 |
| low_spot_check          | pending_manual_review | False                        |           12 |
| medium_priority         | pending_manual_review | False                        |           15 |
| normal_review           | pending_manual_review | False                        |           21 |

## Important limitations

The tracking output is not claimed as fully automatic ground-truth replacement. It is a tracking-assisted validation layer.

The full raw video archive is not fully manually annotated. The validated dataset covers the 72 annotated scanframe clips.

Rare behaviour classes remain limited. Classification should initially be presented as a baseline/pilot experiment.

## Report-ready claim

A validated behaviour-dataset preparation pipeline was completed for 72 annotated observation clips. The pipeline includes GT schema documentation, behaviour label propagation, hybrid tracking-assisted visual validation, quality-tier assessment, manual review of high-priority clips, and report-ready dataset documentation.
