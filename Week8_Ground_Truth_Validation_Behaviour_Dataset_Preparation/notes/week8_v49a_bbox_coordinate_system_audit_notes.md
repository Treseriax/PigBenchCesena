# Week 8 v49a BBox Coordinate System Audit

## Summary

- v49a decision: bbox_coordinate_system_audit_completed
- Clip count: 72
- Object count: 429
- Clips with bbox outside video: 0
- Objects with bbox outside video: 0
- Objects with invalid basic bbox: 0
- Clips probable resolution mismatch: 0
- Clips where bbox coordinates fit video: 72
- Hard issue count: 0
- Warning count: 0
- Ready for v49b bbox coordinate fix: True

## Interpretation

If boxes are outside the video frame, this is stronger than normal motion drift. It suggests a coordinate-system or resolution mismatch that must be fixed before trusting visual identity/behaviour review.

## Outputs

- Clip audit: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v49a_bbox_coordinate_system_audit/week8_v49a_clip_dimension_bbox_audit.csv
- Object audit: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v49a_bbox_coordinate_system_audit/week8_v49a_object_bbox_bounds_audit.csv
- Resolution diagnosis: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v49a_bbox_coordinate_system_audit/week8_v49a_resolution_diagnosis.csv
- Preview index: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v49a_bbox_coordinate_system_audit/week8_v49a_preview_overlay_index.csv
- Preview folder: /home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs/v49a_bbox_coordinate_system_audit/preview_overlays
