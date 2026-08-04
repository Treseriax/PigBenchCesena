# Week 8 Ground-Truth JSON Format Specification v44

## Purpose

This document defines the JSON annotation format for the Week 8 Unibo behaviour ground-truth dataset.

The dataset is designed to associate each pig with tracked pig, colour identity, and behaviour label across the annotated 10-second observation interval.

## Annotation protocol

The Week 8 protocol treats each behavioural observation as a 10-second annotated clip interval.

Behaviour labels apply to the full annotated clip interval.

Every frame belonging to the annotated interval inherits the corresponding pig-level behaviour label.

This propagation rule is used for visualization, validation, and future clip-based behaviour classification experiments.

## Main JSON hierarchy

dataset
  clips
    objects
    frames
      objects

## Required top-level fields

| level | field_name | type | required | source | description | example |
| --- | --- | --- | --- | --- | --- | --- |
| dataset | dataset_version | string | True | generated | Version identifier for the Week 8 ground-truth dataset. | week8_v1 |
| dataset | created_at | string_datetime | True | generated | Timestamp when the JSON annotation file was generated. | 2026-07-16T19:11:06 |
| dataset | annotation_protocol | object | True | Week 8 assignment rule | Defines that behaviour labels are propagated across the full annotated 10-second observation interval. | 10_second_observation_window |

## Clip-level fields

| level | field_name | type | required | source | description | example |
| --- | --- | --- | --- | --- | --- | --- |
| clip | scan_frame_id | string | True | clip_extraction_index_72.scan_frame_id | Unique scanpoint identifier used as the annotation anchor. | scanframe_0000 |
| clip | video_id | string | True | clip_extraction_index_72.video_id | Identifier of the source video. | TLC 1 -B1 0700-0800 |
| clip | source_video_path | string_path | True | clip_extraction_index_72.source_video_path | Path to the original Unibo video. | /work/pig/datasets/Unibo/TLC 1 -B1 0700-0800.mp4 |
| clip | clip_path | string_path | True | clip_extraction_index_72.clip_path | Path to the extracted 10-second clip. | /home/oyavuz/PigBench/Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs/clip_extraction_temporal_qa_v26/clips/scanframe_0000__TLC_1_-B1_0700-0800__centerf_000000__00000000ms_00010000ms.mp4 |
| clip | start_sec | float | True | clip_extraction_index_72.start_sec | Start time of the annotated observation interval within the source video. | 0.0 |
| clip | end_sec | float | True | clip_extraction_index_72.end_sec | End time of the annotated observation interval within the source video. | 10.0 |
| clip | duration_sec | float | True | clip_extraction_index_72.duration_sec | Duration of the annotated observation interval. | 10.0 |
| clip | fps_used | float | True | clip_extraction_index_72.fps_used | FPS used to convert timestamps and frame indices. | 24.9997 |

## Frame-level fields

| level | field_name | type | required | source | description | example |
| --- | --- | --- | --- | --- | --- | --- |
| frame | frame_index_in_clip | integer | True | generated from video frames | Frame index relative to the extracted 10-second clip. | 0 |
| frame | source_frame_index | integer | True | clip start frame + frame index | Estimated absolute frame index in the source video. | 0 |
| frame | timestamp_sec | float | True | start_sec + frame_index_in_clip / fps_used | Timestamp of the frame in source-video time. | 0.0 |

## Object-level fields

| level | field_name | type | required | source | description | example |
| --- | --- | --- | --- | --- | --- | --- |
| object | final_box_id | string | True | behaviour_fusion_box_level_429.final_box_id | Final corrected pig box identifier. | scanframe_0002_final_01 |
| object | bbox_xyxy | array[float] | True | behaviour_fusion_box_level_429.x1/y1/x2/y2 or propagated/tracked bbox | Bounding box in [x1, y1, x2, y2] pixel coordinates. | [x1, y1, x2, y2] |
| object | visual_marker_colour | string | True | behaviour_fusion_box_level_429.visual_marker_colour_v18c | Visual marker colour assigned to the pig. | red |
| object | behaviour_pig_id | string | True | behaviour_fusion_box_level_429.behaviour_pig_id_v18c | Pig identity used in the behaviour annotation file after colour crosswalk. | red_neck |
| object | behaviour_code | string | True | behaviour_fusion_box_level_429.behaviour_code | Behaviour code assigned to this pig in the annotated observation interval. | AN |
| object | behaviour_label | string | False | behaviour_fusion_box_level_429.behaviour_label | Human-readable behaviour label if available. In the current source it may be same as behaviour_code. | Annusano, grufolano - Sniffing - rooting |
| object | label_source | string | True | generated | Describes how the behaviour label was assigned to the frame/object. | propagated_from_10_second_observation_window |
| object | identity_status | string | True | behaviour_fusion_box_level_429.final_identity_status_v17 / arbitration output | Identity confidence/status such as accepted, candidate, unknown, not_visible, uncertain or review_required. | usable_colour_identity |
| object | validation_status | string | True | generated / validation interface | Manual validation status for the object annotation. | unchecked |
| object | validation_flags | array[string] | False | generated / validation interface | Issue flags such as colour_error, behaviour_error, identity_switch, missing_label, occlusion, uncertain. | [] |

## Behaviour codes

| behaviour_code | behaviour_label | box_level_count_429_source | training_ready_count_374_source | clip_level_positive_count_72_source | recommended_use | definition_note |
| --- | --- | --- | --- | --- | --- | --- |
| STI | STI | 151 | 151 | 60 | baseline_ready | Behaviour code from Unibo annotation. Full semantic definition should follow the project annotation guide / Excel legend. |
| LAI | LAI | 55 | 55 | 32 | baseline_ready | Behaviour code from Unibo annotation. Full semantic definition should follow the project annotation guide / Excel legend. |
| BOX | BOX | 44 | 44 | 25 | baseline_ready | Behaviour code from Unibo annotation. Full semantic definition should follow the project annotation guide / Excel legend. |
| AN | AN | 38 | 38 | 26 | baseline_ready | Behaviour code from Unibo annotation. Full semantic definition should follow the project annotation guide / Excel legend. |
| NU | NU | 23 | 23 | 4 | limited_use | Behaviour code from Unibo annotation. Full semantic definition should follow the project annotation guide / Excel legend. |
| PI | PI | 20 | 20 | 11 | limited_use | Behaviour code from Unibo annotation. Full semantic definition should follow the project annotation guide / Excel legend. |
| IN | IN | 18 | 18 | 13 | limited_use | Behaviour code from Unibo annotation. Full semantic definition should follow the project annotation guide / Excel legend. |
| SI | SI | 12 | 12 | 11 | limited_use | Behaviour code from Unibo annotation. Full semantic definition should follow the project annotation guide / Excel legend. |
| BE | BE | 5 | 5 | 4 | report_only_rare | Behaviour code from Unibo annotation. Full semantic definition should follow the project annotation guide / Excel legend. |
| DE | DE | 5 | 5 | 5 | report_only_rare | Behaviour code from Unibo annotation. Full semantic definition should follow the project annotation guide / Excel legend. |
| IA | IA | 3 | 3 | 2 | report_only_rare | Behaviour code from Unibo annotation. Full semantic definition should follow the project annotation guide / Excel legend. |

## Colour identities

| colour_identity | type | behaviour_pig_id_crosswalk | box_level_count_429_source | training_ready_count_374_source | annotation_rule |
| --- | --- | --- | --- | --- | --- |
| blue | valid_visual_marker_colour | blue | 66 | 66 | Valid colour-marker identity used for pig identity association. |
| green | valid_visual_marker_colour | green | 64 | 64 | Valid colour-marker identity used for pig identity association. |
| cyan | valid_visual_marker_colour | no_color | 65 | 65 | Valid colour-marker identity used for pig identity association. |
| red | valid_visual_marker_colour | red_neck | 61 | 61 | Valid colour-marker identity used for pig identity association. |
| pink | valid_visual_marker_colour | red_tail | 63 | 63 | Valid colour-marker identity used for pig identity association. |
| purple | valid_visual_marker_colour | purple | 55 | 55 | Valid colour-marker identity used for pig identity association. |
| unknown | identity_status_or_missing_value |  | 55 | 0 | Not a valid identity for training unless explicitly reviewed; preserved for validation and issue tracking. |
| not_visible | identity_status_or_missing_value |  | 54 | 0 | Not a valid identity for training unless explicitly reviewed; preserved for validation and issue tracking. |
| uncertain | identity_status_or_missing_value |  | 1 | 0 | Not a valid identity for training unless explicitly reviewed; preserved for validation and issue tracking. |
| unassigned | identity_status_or_missing_value |  | 0 | 0 | Not a valid identity for training unless explicitly reviewed; preserved for validation and issue tracking. |

## BBox convention

Bounding boxes use pixel coordinates in xyxy format:

bbox_xyxy = [x1, y1, x2, y2]

where x1, y1 is the top-left corner and x2, y2 is the bottom-right corner.

## Timestamp convention

For a frame inside a clip:

timestamp_sec = clip.start_sec + frame_index_in_clip / fps_used

The corresponding source-video frame index is estimated from the clip start frame and the frame index inside the clip.

## Validation status

All generated annotations start as unchecked.

The visualization interface may update this to accepted, rejected, or review_required.

## Claim scope

This schema defines the ground-truth validation dataset format. It is not a final behaviour classifier and does not claim production-grade tracking.
