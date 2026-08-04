Week 8 v48 Visual QA Anchor Mode Report

Decision:
v48 decision: visual_qa_anchor_mode_created
Clip count: 72
Existing clip paths: 72
Object count: 429
Hard issue count: 0
Warning count: 0

Finding:
The v47 interface is technically functional, but preliminary visual QA shows that scanpoint-anchor boxes are not sufficient for full-clip playback because pigs move during the 10-second interval. Some colour and behaviour labels may appear wrong when the box drifts to another pig.

Interpretation:
This is expected for scanpoint-anchor propagation. It does not invalidate the label propagation. It means visual validation must first use anchor-frame inspection and then move toward tracking-refined per-frame boxes.

v48 changes:
- Added bbox source warning.
- Added note path display.
- Added jump start, jump anchor, jump end controls.
- Added review mode field.
- Added better issue types.
- Added source info overlay.
- Recorded preliminary v47 finding.

Run:
cd ~/PigBench
python Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/interface/week8_visualizer_server_v48.py --port 8514

Open:
http://127.0.0.1:8514/
