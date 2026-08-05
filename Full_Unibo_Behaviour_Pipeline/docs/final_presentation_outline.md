# Final Presentation Outline - Unibo Pig Behaviour Analysis Pipeline

## Presentation goal

Explain the project as a reproducible research pipeline: what was built, what was evaluated, what worked, what failed, and what must not be overclaimed.

## Recommended structure

### Slide 1: Project Overview

Main message: Reproducible experimental pipeline for Unibo pig behaviour analysis.

Content:
- Goal: map videos, inspect annotations, build baselines, evaluate behaviour recognition.
- Scope: academic/research pipeline, not production-ready system.
- Main outputs: mapping, JSON export, app, splits, models, grouped analysis, final reports.

Evidence: README.md; docs/final_report.md

Claim boundary: Do not claim production-ready behaviour recognition.

### Slide 2: Dataset and Annotation Status

Main message: The project mapped 84 videos, but only 36 are label-bearing.

Content:
- 84 mapped videos.
- 36 label-bearing videos.
- 48 metadata-only videos.
- 2768 labelled behaviour clips.
- 11 original behaviour classes.

Evidence: outputs/v80_final_project_completion/00_source_registry_and_mapping; outputs/v80_final_project_completion/01_dataset_json_export

Claim boundary: Do not imply all 84 videos were used for supervised model training.

### Slide 3: Annotation Visualizer and Correction Workflow

Main message: The project includes a demo-ready Streamlit app.

Content:
- Loads mapped videos and annotation data.
- Supports annotation interval inspection.
- Includes editable correction workflow.
- Manual corrections are separate artifacts.
- Original JSON overwrite is not claimed.

Evidence: interface/v80d_annotation_gt_visualizer_app.py; docs/DEMO.md; docs/DEMO_APP_WORKFLOW.md

Claim boundary: Correction-support app, not production labelling platform.

### Slide 4: Overlay and Visual Evidence

Main message: Visual overlay/frame evidence supports mapping and review.

Content:
- 1044 image candidates found.
- 168 overlay crop candidates.
- 87 full frame candidates.
- 549 annotated frame candidates.
- 36 demo sample images copied.

Evidence: outputs/v84_app_demo_evidence/v84c_overlay_demo_board.html; outputs/v84_app_demo_evidence/v84c_overlay_demo_samples.csv

Claim boundary: Visual evidence supports review/mapping only; final automatic identity tracking is not claimed.

### Slide 5: Split Protocols

Main message: Two split protocols were created to control leakage and test generalisation.

Content:
- Split A: grouped video-level split.
- Split B: cross-camera/pen split.
- Split A used as main development/evaluation split.
- Split B used as harder generalisation setting.
- Leakage audit files were generated.

Evidence: outputs/v80_final_project_completion/03_splits

Claim boundary: Split B validation has 10 classes while test has 11; this affects model selection.

### Slide 6: 11-Class Baseline Results

Main message: Fine-grained 11-class behaviour recognition remains difficult.

Content:
- v80 best 11-class macro F1: 0.110956.
- v80 best 11-class accuracy: 0.136364.
- v81 selected 11-class macro F1: 0.094654.
- v81 selected 11-class accuracy: 0.246951.
- Accuracy improved, but macro F1 did not.

Evidence: outputs/v80_final_project_completion/06_evaluation_comparison; outputs/v81_performance_improvement/02_full_feature_models

Claim boundary: Do not claim the 11-class task is solved.

### Slide 7: Error Analysis

Main message: Low 11-class macro F1 is explained by imbalance and visual ambiguity.

Content:
- Rare or weak classes: BE, SI, IA, DE, BOX, PI.
- Some classes had zero recall.
- Frequent confusions include STI, LAI, and PI.
- The result is a dataset/model limitation, not just a coding issue.

Evidence: outputs/v81_performance_improvement/03_comparison_with_v80/v81e1_class_imbalance_error_analysis_report.md

Claim boundary: Use this to justify limitations, not to hide weak 11-class results.

### Slide 8: Grouped Behaviour Analysis

Main message: Broader grouped behaviour classes are more learnable.

Content:
- Grouped task: STI + LAI = STI_LAI; all others = OTHER.
- Split A grouped macro F1: 0.639560.
- Split A grouped accuracy: 0.652439.
- Grouped classifier is supplemental.

Evidence: outputs/v81_performance_improvement/03_comparison_with_v80/v81f1_decision_summary.csv

Claim boundary: Grouped classifier does not replace the original 11-class task.

### Slide 9: Cross-Camera/Pen Generalisation

Main message: Grouped performance depends on split protocol.

Content:
- Split B official grouped macro F1: 0.238434.
- Split B official grouped accuracy: 0.313084.
- Single Split B result showed validation-to-test instability.
- Leave-camera/pen-out CV mean macro F1: 0.582280.
- Leave-camera/pen-out CV mean accuracy: 0.648309.

Evidence: outputs/v83_splitB_grouped_eval/v83e_grouped_generalization_summary_report.md

Claim boundary: Best-test exploratory model must not be claimed as selected final model.

### Slide 10: VideoMAE Decision

Main message: VideoMAE was investigated but kept as feasibility/future work.

Content:
- Torch, Transformers, and VideoMAE classes were available.
- No local pretrained VideoMAE snapshot with weights was available.
- Pretrained forward smoke could not be completed offline.
- Tiny random VideoMAE remains infrastructure evidence only.

Evidence: docs/VIDEOMAE_DECISION.md; outputs/v85_videomae_pretrained_decision/v85c_videomae_final_decision_report.md

Claim boundary: Do not claim final pretrained VideoMAE training or evaluation.

### Slide 11: Final Claim Boundaries

Main message: The project is successful because it is reproducible and honest about limitations.

Content:
- Do not claim production-ready behaviour recognition.
- Do not claim final automatic identity tracking.
- Do not claim fully solved 11-class classification.
- Do not claim final pretrained VideoMAE.
- Do not use grouped classifier as replacement for 11-class task.

Evidence: README.md; docs/final_report.md; docs/DEMO.md

Claim boundary: This slide is the defence against overclaiming.

### Slide 12: Conclusion and Future Work

Main message: The pipeline is a strong foundation for future behaviour analysis work.

Content:
- Delivered: mapping, JSON export, app, demo, splits, baselines, grouped analysis, reports.
- Main finding: fine-grained 11-class task is hard; grouped task is more learnable.
- Future work: final identity tracking, pretrained video models, more balanced labels, stronger temporal modelling.
- End with live demo or evidence walkthrough.

Evidence: docs/DEMO.md; docs/final_report.md; outputs/v83_splitB_grouped_eval; outputs/v84_app_demo_evidence

Claim boundary: Present as reproducible research pipeline and future-work foundation.

## Demo order

1. Show README.md as project entry point.
2. Open docs/DEMO.md and explain launch command.
3. Show Streamlit annotation visualizer if server/browser is available.
4. Show overlay demo board from outputs/v84_app_demo_evidence/v84c_overlay_demo_board.html.
5. Show final results from docs/final_report.md.
6. End with claim boundaries and future work.

## One-minute final summary

This project delivers a reproducible Unibo pig behaviour analysis pipeline with video mapping, JSON annotation export, annotation visualisation, correction support, split definitions, baselines, full visual-temporal features, grouped behaviour analysis, demo materials, and final limitation reports. The original 11-class task remains difficult, while supplemental grouped behaviour classification is more learnable and shows stronger generalisation under leave-camera/pen-out CV. Final automatic identity tracking and pretrained VideoMAE are documented as future work, not claimed as completed.
