# Demo Workflow - Annotation Visualizer and Correction App

## Purpose

This demo shows the annotation visualizer and correction interface for the Unibo pig behaviour pipeline.

The app is used to inspect mapped videos, load exported JSON annotations, view annotation intervals, and save manual correction records without overwriting the original JSON files.

## Launch command

cd ~/PigBench
python -m streamlit run Full_Unibo_Behaviour_Pipeline/interface/v80d_annotation_gt_visualizer_app.py --server.port 8560 --server.address 0.0.0.0

Open in browser:

http://137.204.72.3:8560

## Demo steps

1. Open the Streamlit app.
2. Select a mapped Unibo video.
3. Confirm that the video loads.
4. Inspect the annotation intervals.
5. Open the editable correction table.
6. Modify or add a correction row.
7. Save the correction.
8. Confirm that the correction is exported separately.
9. State explicitly that original JSON files are not overwritten.

## What this demo proves

- The project has a working annotation visualizer.
- The app can load mapped videos and behaviour annotation JSON files.
- The user can inspect intervals and correction tables.
- Manual corrections are saved separately for auditability.
- The app supports the annotation-review workflow required for the project.

## Claim boundary

This app is an annotation inspection and correction-support tool. It is not a production labelling platform and does not claim final automatic identity tracking.
