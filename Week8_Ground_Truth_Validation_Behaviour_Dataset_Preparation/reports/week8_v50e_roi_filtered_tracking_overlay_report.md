Week 8 v50e ROI-filtered Tracking Overlay Report

Decision: roi_filtered_tracking_overlay_completed
Raw tracking rows: 26154
Kept tracking rows: 16952
Removed outside ROI rows: 9202
Removed ratio total: 0.3518
High ROI-removal clips: 26
Moderate ROI-removal clips: 42
Hard issues: 0
Warnings: 2

Interpretation:
ROI filtering reduces detections outside the annotated pen area. The filter is derived from video-level anchor GT boxes expanded by a conservative margin. This should be treated as a QA filter, not a final biological identity assignment.

Visualization update:
The new overlay draws anchor GT boxes thin, kept tracking boxes thick white, outside-ROI detections red, and the ROI boundary yellow.

Next:
Proceed to v51 tracking-integrated visualizer if no hard issues are present.
