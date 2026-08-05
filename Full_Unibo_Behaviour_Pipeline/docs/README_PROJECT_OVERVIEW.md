# PigBench / Unibo Pig Behaviour Analysis Pipeline

## Project status

This repository contains a reproducible experimental pipeline for Unibo pig behaviour analysis.

The project includes dataset mapping, JSON annotation export, annotation visualisation, manual correction support, train/validation/test split definitions, baseline models, clip-level feature extraction, grouped behaviour analysis, app/demo documentation, and final claim-boundary reports.

This is not a production-ready behaviour recognition system. The final outputs should be interpreted as an academic/research pipeline with documented limitations.

## Main outputs

The main project folder is:

Full_Unibo_Behaviour_Pipeline/

Important folders:

- Full_Unibo_Behaviour_Pipeline/interface
- Full_Unibo_Behaviour_Pipeline/docs
- Full_Unibo_Behaviour_Pipeline/notes
- Full_Unibo_Behaviour_Pipeline/outputs/v80_final_project_completion
- Full_Unibo_Behaviour_Pipeline/outputs/v81_performance_improvement
- Full_Unibo_Behaviour_Pipeline/outputs/v83_splitB_grouped_eval
- Full_Unibo_Behaviour_Pipeline/outputs/v84_app_demo_evidence
- Full_Unibo_Behaviour_Pipeline/outputs/v85_videomae_pretrained_decision
- Full_Unibo_Behaviour_Pipeline/outputs/v86_final_documentation

## Dataset and annotation status

The pipeline maps the available Unibo videos and exports annotation JSON files.

Final dataset status:

- 84 mapped videos.
- 36 label-bearing videos.
- 48 metadata-only videos.
- 2768 behaviour annotation clips used for modelling.
- 11 original behaviour classes.

Important evidence:

- outputs/v80_final_project_completion/00_source_registry_and_mapping
- outputs/v80_final_project_completion/01_dataset_json_export
- outputs/v80_final_project_completion/03_splits

## Annotation visualizer and correction app

The Streamlit app is located at:

Full_Unibo_Behaviour_Pipeline/interface/v80d_annotation_gt_visualizer_app.py

Launch command:

cd ~/PigBench
python -m streamlit run Full_Unibo_Behaviour_Pipeline/interface/v80d_annotation_gt_visualizer_app.py --server.port 8560 --server.address 0.0.0.0

Open:

http://137.204.72.3:8560

The app supports video/annotation inspection and manual correction workflow. Corrections are handled as separate review artifacts. Original JSON files are not claimed to be overwritten.

Demo documentation:

- Full_Unibo_Behaviour_Pipeline/docs/DEMO.md
- Full_Unibo_Behaviour_Pipeline/docs/DEMO_APP_WORKFLOW.md
- Full_Unibo_Behaviour_Pipeline/docs/OVERLAY_EVIDENCE_DEMO.md

## Visual overlay evidence

The project includes visual overlay/frame evidence for mapping and review.

Important evidence:

- outputs/v84_app_demo_evidence/v84c_overlay_demo_board.html
- outputs/v84_app_demo_evidence/v84c_overlay_demo_samples.csv
- outputs/v84_app_demo_evidence/v84c_overlay_demo_samples

Claim boundary:

These images support visual inspection, overlay review, and mapping evidence. They do not prove final automatic identity tracking.

## Splits

The project includes two main split protocols.

Split A:

- Grouped video-level split.
- Used as the main development/evaluation split.

Split B:

- Cross-camera/pen split.
- Used to test harder generalisation.

Important evidence:

- outputs/v80_final_project_completion/03_splits/v80e_split_A_clip_split_assignments.csv
- outputs/v80_final_project_completion/03_splits/v80e_split_A_video_leakage_audit.csv
- outputs/v80_final_project_completion/03_splits/v80e_split_B_clip_split_assignments.csv
- outputs/v80_final_project_completion/03_splits/v80e_split_B_video_leakage_audit.csv
- outputs/v80_final_project_completion/03_splits/v80e_split_quality_decision_summary.csv

## Modelling summary

The project includes several modelling stages.

### Frame-proxy baseline

Frame-proxy baseline results are stored in:

outputs/v80_final_project_completion/04_frame_based_baseline

This is a baseline only. It is not final identity-aware behaviour recognition.

### Limited clip visual baseline

The best v80 experimental 11-class result was the limited clip visual feature baseline.

Result:

- Test macro F1: 0.110956
- Test accuracy: 0.136364

Important evidence:

outputs/v80_final_project_completion/05_clip_based_videomae/v80g5_limited_clip_visual_feature_baseline

### Full 2768-clip visual-temporal features

v81 extracted visual-temporal features for all 2768 clips.

Important evidence:

outputs/v81_performance_improvement/01_full_visual_temporal_features/v81b2_full_visual_temporal_features.csv

### v81 selected 11-class model

The v81 selected 11-class model improved accuracy but not macro F1.

Result:

- Test macro F1: 0.094654
- Test accuracy: 0.246951

Interpretation:

The 11-class task remains difficult because of class imbalance, rare labels, and overlapping visual behaviour patterns.

Important evidence:

- outputs/v81_performance_improvement/02_full_feature_models
- outputs/v81_performance_improvement/03_comparison_with_v80

## Grouped behaviour analysis

A supplemental grouped classifier was added because the original 11-class task is highly imbalanced and difficult.

Grouped task:

- STI and LAI are grouped as STI_LAI.
- All other behaviours are grouped as OTHER.

This grouped task is supplemental. It does not replace the original 11-class task.

### Split A grouped result

- Macro F1: 0.639560
- Accuracy: 0.652439

### Split B official validation-selected result

- Macro F1: 0.238434
- Accuracy: 0.313084

This single Split B result showed validation-to-test model-selection instability.

### Leave-camera/pen-out grouped CV

- Selected model: random forest
- Folds: 6
- Mean macro F1: 0.582280
- Mean accuracy: 0.648309

Interpretation:

The grouped classifier is more learnable than the original 11-class task, but split protocol matters. Leave-camera/pen-out CV gives stronger grouped generalisation evidence than a single validation/test split.

Important evidence:

- outputs/v83_splitB_grouped_eval/v83e_grouped_generalization_summary_report.md
- outputs/v83_splitB_grouped_eval/v83d_leave_camera_pen_out_grouped_cv_report.md
- outputs/v83_splitB_grouped_eval/v83c_splitB_grouped_interpretation_report.md

## VideoMAE decision

VideoMAE was investigated as a clip-level video-transformer direction.

Final decision:

- Torch, Transformers, and VideoMAE classes are importable.
- No local pretrained VideoMAE snapshot with weights was available during the final offline audit.
- A final pretrained VideoMAE model is not claimed.
- Tiny random VideoMAE remains feasibility/infrastructure evidence only.

Important evidence:

- Full_Unibo_Behaviour_Pipeline/docs/VIDEOMAE_DECISION.md
- outputs/v85_videomae_pretrained_decision/v85c_videomae_final_decision_report.md
- outputs/v80_final_project_completion/05_clip_based_videomae/v80g7_tiny_videomae_limited_training

## Main claim boundaries

Do not claim:

- Production-ready behaviour recognition.
- Fully solved 11-class classification.
- Final automatic identity tracking.
- Final pretrained VideoMAE training/evaluation.
- Grouped classifier as a replacement for the 11-class task.

Correct claim:

This project provides a reproducible experimental pipeline for pig behaviour analysis, including annotation mapping, JSON export, visual inspection/correction app, split definitions, baseline models, full visual-temporal feature extraction, grouped behaviour analysis, and documented limitations.

## How to inspect final documentation

Important final documents:

- README.md
- Full_Unibo_Behaviour_Pipeline/docs/DEMO.md
- Full_Unibo_Behaviour_Pipeline/docs/DEMO_APP_WORKFLOW.md
- Full_Unibo_Behaviour_Pipeline/docs/OVERLAY_EVIDENCE_DEMO.md
- Full_Unibo_Behaviour_Pipeline/docs/VIDEOMAE_DECISION.md
- Full_Unibo_Behaviour_Pipeline/docs/final_report.md
- Full_Unibo_Behaviour_Pipeline/docs/final_presentation_outline.md

## Current documentation stage

v86 final documentation is in progress.

Completed:

- Demo guide.
- App/overlay evidence summary.
- VideoMAE decision.
- Grouped generalisation summary.
- Root README.

Remaining after this README:

- Final report narrative.
- Final presentation outline.

## Standard claim-boundary wording

This project is not production-ready.

The project does not claim final automatic identity tracking.

The project does not claim a final pretrained VideoMAE model.

The grouped classifier does not replace the original 11-class behaviour classification task.

The 11-class task remains difficult and is documented with limitations.

