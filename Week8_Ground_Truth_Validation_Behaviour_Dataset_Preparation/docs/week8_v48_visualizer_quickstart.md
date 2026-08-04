Week 8 v48 Behaviour Dataset Visualizer Quickstart

Purpose:
v48 is the visual QA bugfix version of the Week 8 validation interface.

Main changes over v47:
- Shows bbox source warning.
- Shows note CSV path in the UI.
- Adds jump start / jump annotation anchor / jump end controls.
- Adds better issue types for bbox and label-association review.
- Saves review mode, bbox source, and label source with each note.

Run:
cd ~/PigBench
python Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/interface/week8_visualizer_server_v48.py --port 8514

Open:
http://127.0.0.1:8514/

Port forwarding if needed:
ssh -L 8514:127.0.0.1:8514 oyavuz@137.204.72.3

Validation notes:
/home/oyavuz/PigBench/Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/validation/week8_v47_visual_validation_notes.csv

Important scope:
The current boxes are scanpoint-anchor boxes repeated across the interval. If boxes are correct at the anchor but drift during motion, this is expected and means tracking-refined per-frame boxes are needed.
