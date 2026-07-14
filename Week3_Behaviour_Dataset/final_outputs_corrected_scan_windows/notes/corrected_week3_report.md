# Week 3 Corrected Report — Behaviour Dataset Construction and Visualization

## 1. Purpose of the correction

The first Week 3 version successfully built the behaviour annotation structure, but the original video sample did not exactly overlap with the Excel scan-sampling observation windows. To fix this limitation, a new manually concatenated screen-recording video was prepared. This video contains the relevant scan-window intervals at 09:00, 09:10, 09:20, 09:30, 09:40, and 09:50.

The corrected version therefore replaces the previous no-overlap situation with a segment-level alignment strategy. Each video segment is mapped to its corresponding Excel scan-sampling window.

## 2. Methodological note

The corrected video is not a continuous camera recording. It is a manually concatenated screen recording of selected observation windows. For this reason, the pipeline does not treat video elapsed time as continuous real camera time. Instead, each segment is manually mapped to its Excel observation interval.

This is the corrected time mapping used in the dataset:

- `scan_09_00` → `09:00:00–09:00:10`
- `scan_09_10` → `09:10:00–09:10:10`
- `scan_09_20` → `09:20:00–09:20:10`
- `scan_09_30` → `09:30:00–09:30:10`
- `scan_09_40` → `09:40:00–09:40:10`
- `scan_09_50` → `09:50:00–09:50:10`

## 3. Corrected processing pipeline

The corrected Week 3 pipeline follows these steps:

1. Check the clean scan-window video metadata and sample frames.
2. Create a manual segment-to-Excel-window mapping.
3. Extract each scan-window segment into a separate image sequence.
4. Run YOLOv8-s detection on each segment.
5. Run ByteTrack tracking on each segment.
6. Attach segment ID and Excel interval information to each track row.
7. Parse the Excel behaviour labels for each scan window.
8. Build the corrected JSON annotation dataset.
9. Generate visualization videos showing track IDs, segment windows, and behaviour candidates.
10. Package corrected CSV, JSON, figures, videos, scripts, and notes into a final output folder.

## 4. Corrected JSON dataset summary

The corrected JSON dataset contains 6 aligned scan-window segments, 2450 processed frames, and 23234 track instances.

| segment_id | excel_interval_start | excel_interval_end | frames | track_instances | verified_track_instances | unverified_track_instances |
| --- | --- | --- | --- | --- | --- | --- |
| scan_09_00 | 09:00:00 | 09:00:10 | 500 | 5139 | 0 | 5139 |
| scan_09_10 | 09:10:00 | 09:10:10 | 375 | 3294 | 0 | 3294 |
| scan_09_20 | 09:20:00 | 09:20:10 | 250 | 2682 | 0 | 2682 |
| scan_09_30 | 09:30:00 | 09:30:10 | 500 | 4968 | 0 | 4968 |
| scan_09_40 | 09:40:00 | 09:40:10 | 500 | 4481 | 0 | 4481 |
| scan_09_50 | 09:50:00 | 09:50:10 | 325 | 2670 | 0 | 2670 |

## 5. Tracking summary

ByteTrack was applied separately to each extracted scan-window sequence. The number of unique track IDs is higher than the number of pigs in some segments because of tracking fragmentation, partial visibility, neighbouring pen detections, and screen-recording artefacts.

| segment_id | frames | track_rows | unique_track_ids | excel_interval_start | excel_interval_end |
| --- | --- | --- | --- | --- | --- |
| scan_09_00 | 500 | 5139 | 33 | 09:00:00 | 09:00:10 |
| scan_09_10 | 375 | 3294 | 19 | 09:10:00 | 09:10:10 |
| scan_09_20 | 250 | 2682 | 19 | 09:20:00 | 09:20:10 |
| scan_09_30 | 500 | 4968 | 32 | 09:30:00 | 09:30:10 |
| scan_09_40 | 500 | 4481 | 15 | 09:40:00 | 09:40:10 |
| scan_09_50 | 325 | 2670 | 10 | 09:50:00 | 09:50:10 |

## 6. Behaviour label summary from Excel

The Excel annotation file provides scan-sampling behaviour labels for colour-coded pig identities. These labels were successfully attached at the segment level.

| segment_id | excel_interval_start | excel_interval_end | num_pigs | colours | behaviour_codes | behaviour_labels |
| --- | --- | --- | --- | --- | --- | --- |
| scan_09_00 | 09:00:00 | 09:00:10 | 6 | green, blue, purple, red neck, red tail, no color | NU, NU, NU, NU, AN, NU | eating, eating, eating, eating, exploring_or_sniffing, eating |
| scan_09_10 | 09:10:00 | 09:10:10 | 6 | green, blue, purple, red neck, red tail, no color | BOX, BOX, BOX, BOX, STI, BOX | out_of_view_or_box, out_of_view_or_box, out_of_view_or_box, out_of_view_or_box, lying_sternal, out_of_view_or_box |
| scan_09_20 | 09:20:00 | 09:20:10 | 6 | green, blue, purple, red neck, red tail, no color | BOX, BOX, PI, BOX, STI, AN | out_of_view_or_box, out_of_view_or_box, standing_inactive, out_of_view_or_box, lying_sternal, exploring_or_sniffing |
| scan_09_30 | 09:30:00 | 09:30:10 | 6 | green, blue, purple, red neck, red tail, no color | BOX, BOX, STI, BOX, STI, BOX | out_of_view_or_box, out_of_view_or_box, lying_sternal, out_of_view_or_box, lying_sternal, out_of_view_or_box |
| scan_09_40 | 09:40:00 | 09:40:10 | 6 | green, blue, purple, red neck, red tail, no color | AN, BOX, STI, AN, IN, AN | exploring_or_sniffing, out_of_view_or_box, lying_sternal, exploring_or_sniffing, interaction, exploring_or_sniffing |
| scan_09_50 | 09:50:00 | 09:50:10 | 6 | green, blue, purple, red neck, red tail, no color | LAI, STI, PI, STI, STI, STI | lying_lateral, lying_sternal, standing_inactive, lying_sternal, lying_sternal, lying_sternal |

## 7. Identity mapping policy

The tracker produces numerical IDs such as `track_id=1`, `track_id=2`, and so on. However, the Excel file provides labels for colour-coded identities such as green, blue, purple, red neck, red tail, and no color. Therefore, a track-to-colour mapping step is needed before assigning a behaviour label to an individual track.

This corrected version keeps individual track identities unverified unless a human manually confirms the colour identity. This conservative approach prevents false behaviour assignment when colour markers are unclear or when track fragmentation occurs.

Current status:

- Segment-level behaviour labels are available.
- Track-level bounding boxes and track IDs are available.
- Track-to-colour identity mapping is prepared as a manual verification table.
- Individual track behaviour labels are not forced unless the identity mapping is manually verified.

## 8. Final corrected outputs

The corrected final output package contains:

- Corrected JSON annotation file
- Segment-level behaviour label JSON and CSV files
- Tracking CSV files with Excel window information
- Track ID summary and manual identity mapping template
- Track ID montage figures for all scan windows
- Corrected visualization videos for all scan windows
- Python scripts used in the corrected pipeline
- Summary notes and this corrected report

## 9. Remaining limitation and future work

The main previous limitation, no exact overlap between video and Excel scan windows, has been fixed. The remaining limitation is identity verification: individual track IDs still require manual confirmation before they can be safely linked to colour-coded pig identities. A future improvement would be to manually complete the track-to-colour mapping table for high-confidence tracks, or to train a colour-marker/identity recognition module.

## 10. Conclusion

The corrected Week 3 dataset now provides a stronger behaviour annotation foundation. It links video frames, detected/tracked pig instances, scan-window timing, and Excel behaviour labels in a structured JSON format. The dataset is intentionally conservative at the individual identity level to avoid incorrect behaviour labels, while still preserving all information needed for future manual verification and downstream behaviour analysis.

## 11. Semi-automatic colour-marker identity suggestion

An additional semi-automatic identity suggestion module was implemented after the corrected dataset was generated. The purpose of this module is to support the manual track-to-colour mapping step.

For each dominant track ID, bounding-box crops were sampled across multiple frames. A refined HSV-based marker-blob detection method was then applied to estimate whether a visible colour marker was present. The method produced colour candidates such as green, blue, purple, and red candidate, together with a confidence level and a recommended action.

The refined marker-based method is treated as a support tool, not as final ground truth. High-confidence green, blue, and purple candidates can be visually reviewed and then manually accepted. Red candidates still require manual red-neck/red-tail verification because both identities use red markers. Tracks with no reliable marker remain unverified.

The final JSON was intentionally kept conservative: behaviour labels are not forced onto individual track IDs unless identity mapping is visually verified. This avoids false behaviour assignment while still providing a practical path toward semi-automatic identity mapping.

Additional files added to the corrected final package:

- `refined_marker_mapping_candidates.csv`
- `refined_marker_candidates_review_needed.csv`
- `refined_marker_mapping_candidates.json`
- `refined_marker_candidate_crop_montage.jpg`
- `28_auto_colour_mapping_candidates.py`
- `29_refined_marker_blob_mapping.py`

## 11. Semi-automatic colour-marker identity suggestion

An additional semi-automatic identity suggestion module was implemented after the corrected dataset was generated. The purpose of this module is to support the manual track-to-colour mapping step.

For each dominant track ID, bounding-box crops were sampled across multiple frames. A refined HSV-based marker-blob detection method was then applied to estimate whether a visible colour marker was present. The method produced colour candidates such as green, blue, purple, and red candidate, together with a confidence level and a recommended action.

The refined marker-based method is treated as a support tool, not as final ground truth. High-confidence green, blue, and purple candidates can be visually reviewed and then manually accepted. Red candidates still require manual red-neck/red-tail verification because both identities use red markers. Tracks with no reliable marker remain unverified.

The final JSON was intentionally kept conservative: behaviour labels are not forced onto individual track IDs unless identity mapping is visually verified. This avoids false behaviour assignment while still providing a practical path toward semi-automatic identity mapping.

Additional files added to the corrected final package:

- `refined_marker_mapping_candidates.csv`
- `refined_marker_candidates_review_needed.csv`
- `refined_marker_mapping_candidates.json`
- `refined_marker_candidate_crop_montage.jpg`
- `28_auto_colour_mapping_candidates.py`
- `29_refined_marker_blob_mapping.py`

