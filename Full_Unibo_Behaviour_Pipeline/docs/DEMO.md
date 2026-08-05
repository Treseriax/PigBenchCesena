# DEMO - Unibo Pig Behaviour Pipeline

## 1. Annotation visualizer app

Launch from the repository root:

cd ~/PigBench
python -m streamlit run Full_Unibo_Behaviour_Pipeline/interface/v80d_annotation_gt_visualizer_app.py --server.port 8560 --server.address 0.0.0.0

Open in browser:

http://137.204.72.3:8560

## 2. App demo checklist

During the demo:

1. Open the Streamlit app.
2. Select a mapped Unibo video.
3. Confirm that the video loads.
4. Inspect the annotation intervals.
5. Open the editable correction table.
6. Modify or add a correction row.
7. Save the correction.
8. Confirm that the correction is exported separately.
9. Explain that original JSON files are not overwritten.

## 3. Overlay / visual evidence

Overlay and frame visual evidence is available at:

outputs/v84_app_demo_evidence/v84c_overlay_demo_board.html
outputs/v84_app_demo_evidence/v84c_overlay_demo_samples.csv
outputs/v84_app_demo_evidence/v84c_overlay_demo_samples/

These images support visual inspection, overlay review, and mapping evidence.

## 4. Main demo claims

The project demonstrates:

- 84-video mapping and JSON export workflow.
- Annotation visualizer and correction-support app.
- Manual correction export without overwriting original JSON files.
- Visual overlay/frame evidence for mapping and review.
- Reproducible split, modelling, and evaluation outputs.

## 5. Claim boundaries

Do not claim:

- Production-ready behaviour recognition.
- Final automatic identity tracking.
- Fully solved 11-class behaviour classification.
- Grouped classifier as replacement for the 11-class task.

Correct claim:

The app and overlay outputs support annotation inspection, correction workflow, and visual mapping review. The grouped classifier is supplemental analysis, and final automatic identity tracking remains future work.

## Standard claim-boundary wording

This project is not production-ready.

The project does not claim final automatic identity tracking.

The project does not claim a final pretrained VideoMAE model.

The grouped classifier does not replace the original 11-class behaviour classification task.

The 11-class task remains difficult and is documented with limitations.

