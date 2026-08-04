Week 8 v49a BBox Coordinate System Audit Report

Decision:
v49a decision: bbox_coordinate_system_audit_completed
Clip count: 72
Object count: 429
Clips with bbox outside video: 0
Objects with bbox outside video: 0
Objects with invalid basic bbox: 0
Clips probable resolution mismatch: 0
Clips where bbox coordinates fit video: 72
Hard issue count: 0
Warning count: 0

Interpretation:
If many boxes are outside the video frame, this is not merely motion drift. It indicates a coordinate-system or resolution mismatch between the annotation boxes and the clip video display resolution, or incorrect source association.

Outputs:
Clip audit: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v49a_bbox_coordinate_system_audit/week8_v49a_clip_dimension_bbox_audit.csv
Object audit: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v49a_bbox_coordinate_system_audit/week8_v49a_object_bbox_bounds_audit.csv
Resolution diagnosis: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v49a_bbox_coordinate_system_audit/week8_v49a_resolution_diagnosis.csv
Preview overlay index: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v49a_bbox_coordinate_system_audit/week8_v49a_preview_overlay_index.csv
Preview overlays folder: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v49a_bbox_coordinate_system_audit/preview_overlays
Issues: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v49a_bbox_coordinate_system_audit/week8_v49a_issues.csv

Next step:
Inspect preview overlays. If scaled candidate overlays align better than as-is overlays, v49b should implement bbox coordinate scaling for the interface. If both fail, the source box/frame mapping must be reviewed before using tracking-refined boxes.
