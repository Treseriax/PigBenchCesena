# Week 7 Manual Annotation Package v9

## Purpose

The automatic detector/ROI workflow still produced wrong boxes in many frames. Therefore, all 72 scanpoint frames are exported to a COCO-compatible manual annotation package.

## Output

- CVAT/COCO zip: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/manual_annotation_v9/week7_manual_annotation_v9_cvat_coco_import.zip`
- Candidate metadata: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/manual_annotation_v9/week7_manual_annotation_v9_candidate_metadata.csv`
- Frame map: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/manual_annotation_v9/week7_manual_annotation_v9_frame_image_map.csv`
- COCO JSON: `/home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/manual_annotation_v9/cvat_coco_import/annotations/instances_default.json`

## Manual rule

Keep only true GT-pen pigs, delete wrong-pen boxes, adjust inaccurate boxes, and draw missing true GT-pen pigs.
