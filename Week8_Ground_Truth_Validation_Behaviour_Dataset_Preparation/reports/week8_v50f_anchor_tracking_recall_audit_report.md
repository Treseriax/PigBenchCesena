Week 8 v50f Anchor Tracking Recall Audit Report

Decision: anchor_tracking_recall_audit_completed_use_tracking_as_optional_layer
Anchor objects: 429
Filtered strong matches: 277
Filtered moderate matches: 24
Raw matches removed by ROI: 0
Weak filtered matches: 30
Detector/sampling misses: 98
Filtered anchor recall: 0.7016
Raw anchor recall with weak/ROI-removed: 0.7716
Low recall clips: 20
Review queue rows: 128
Hard issues: 0
Warnings: 2

Interpretation:
This audit checks whether every anchor GT pig has a nearby tracking box. Tracking can be integrated into the visualizer as an optional full-clip layer, but it must not be treated as complete ground truth or final pig identity.

Next:
Use v51 to build a tracking-integrated visualizer with clear layer labels: Anchor GT, ROI-filtered tracking, missed-anchor warnings, and manual review notes.
