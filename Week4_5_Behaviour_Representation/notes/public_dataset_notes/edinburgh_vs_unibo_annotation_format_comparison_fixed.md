# Edinburgh vs Unibo Annotation Format Comparison — Fixed Summary

## Why this fixed version was created

The first automatic comparison used a broad recursive key search. It over-counted Unibo segments because keys such as `segment_time_sec` were also matched, and it missed bounding boxes because Unibo stores them as `bbox_xyxy` and `bbox_xywh`. This fixed version uses the known Unibo JSON structure directly.

## Fixed comparison table

| aspect                                | Edinburgh manually annotated dataset                                                                      | Unibo corrected scan-window dataset                                                                                                            |
|:--------------------------------------|:----------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------|
| Data source                           | Public Edinburgh annotated.tar ground-truth package                                                       | Our Week 3 corrected scan-window outputs                                                                                                       |
| Main annotation file                  | output.json inside each annotated sequence folder                                                         | corrected_scan_window_behaviour_annotations.json                                                                                               |
| Top-level JSON structure              | dict with keys: videoFileName, fullVideoFilePath, stepSize, config, objects                               | dict with keys: dataset_name, source_video, method_note, annotation_policy, segments                                                           |
| Number of sequences / segments        | 12 manually ground-truthed sequences in annotated.tar                                                     | 6 corrected scan-window segments: scan_09_00, scan_09_10, scan_09_20, scan_09_30, scan_09_40, scan_09_50                                       |
| Object / track representation         | objects list; each object corresponds to one manually tracked pig; inspected sample has 8 objects         | 23234 frame-level track instances from YOLOv8-s + ByteTrack; 35 unique global track IDs                                                        |
| Frame range / temporal representation | Ground-truth frameNumber range 0–599; source page notes GT corresponds to every third raw video frame     | Frame range 1–500; segments aligned to Excel scan-window observations                                                                          |
| Bounding box format                   | bbox object with x, y, width, height                                                                      | bbox_xyxy and bbox_xywh stored for each track instance; bbox_xyxy count = 23234, bbox_xywh count = 23234                                       |
| Behaviour labels                      | standing, walk, investigating, sleep, sitting, drink, playwithtoy, lying                                  | BOX, out_of_view_or_box, STI, lying_sternal, NU, eating, AN, exploring_or_sniffing, PI, standing_inactive, IN, interaction, LAI, lying_lateral |
| Behaviour granularity                 | Manual frame descriptors; behaviour changes stored sparsely and propagated according to source-page notes | Excel ethogram labels are attached at scan-window/segment level, then linked to frame-level tracks                                             |
| Identity handling                     | Persistent object IDs in manually ground-truthed object list                                              | ByteTrack IDs are available; true pig identity is still conservative / not fully verified                                                      |
| Depth information                     | Colour video, depth video, masks, and calibration-related files are available                             | RGB/video-based tracking outputs; no depth information                                                                                         |
| Best use in Week 4                    | Public reference for annotation structure, bbox + track + behaviour representation                        | Primary dataset for our trajectory, ROI, and representation feature engineering                                                                |

## Fixed Unibo schema summary

| field                      | value                                                                  |
|:---------------------------|:-----------------------------------------------------------------------|
| top_level_type             | dict                                                                   |
| top_level_keys             | dataset_name, source_video, method_note, annotation_policy, segments   |
| segment_count              | 6                                                                      |
| segment_ids                | scan_09_00, scan_09_10, scan_09_20, scan_09_30, scan_09_40, scan_09_50 |
| frame_count                | 2450                                                                   |
| track_instance_count       | 23234                                                                  |
| unique_track_ids_global    | 35                                                                     |
| min_frame                  | 1                                                                      |
| max_frame                  | 500                                                                    |
| bbox_xyxy_count            | 23234                                                                  |
| bbox_xywh_count            | 23234                                                                  |
| identity_status_values     | unverified:23234                                                       |
| assigned_colour_values_top | None:23234                                                             |

## Track-level behaviour values

| track_behaviour_value   |   count |
|:------------------------|--------:|
| identity_not_verified   |   23234 |

## Segment-level available behaviour labels

| segment_available_label_value   |   count |
|:--------------------------------|--------:|
| BOX                             |      13 |
| out_of_view_or_box              |      13 |
| STI                             |       9 |
| lying_sternal                   |       9 |
| NU                              |       5 |
| eating                          |       5 |
| AN                              |       5 |
| exploring_or_sniffing           |       5 |
| PI                              |       2 |
| standing_inactive               |       2 |
| IN                              |       1 |
| interaction                     |       1 |
| LAI                             |       1 |
| lying_lateral                   |       1 |

## Interpretation

The Edinburgh dataset is cleaner as a public manual ground-truth reference because it contains persistent object IDs, sparse frame descriptors, bounding boxes, and behaviour labels in `output.json`. The Unibo dataset is more project-specific and has corrected scan-window alignment plus detector/tracker outputs, but its behaviour labels are coarser because they come from scan-window Excel observations rather than full manual frame-by-frame annotation.
