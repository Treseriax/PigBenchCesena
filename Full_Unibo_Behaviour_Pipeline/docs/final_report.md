# Final Report - Unibo Pig Behaviour Analysis Pipeline

## 1. Project overview

This project builds a reproducible experimental pipeline for pig behaviour analysis using the Unibo video dataset.

The work focuses on mapping videos, exporting behaviour annotations, creating an annotation visualizer and correction-support workflow, defining leakage-aware train/validation/test splits, evaluating baseline behaviour classifiers, analysing model limitations, and documenting final claim boundaries.

The project should be interpreted as an academic/research pipeline. It is not presented as a production-ready behaviour recognition system.

## 2. Objectives

The main objectives were:

1. Organize and map the available Unibo videos.
2. Export annotation data into usable JSON and tabular formats.
3. Build an app for video and annotation inspection.
4. Support manual correction review without overwriting original JSON files.
5. Define reproducible train/validation/test splits.
6. Train and evaluate behaviour recognition baselines.
7. Investigate clip-level and video-transformer feasibility.
8. Analyse class imbalance, errors, and generalisation limitations.
9. Provide grouped behaviour analysis as a supplemental task.
10. Prepare a clean final handover with README, demo guide, report, and presentation outline.

## 3. Dataset and annotation status

The final mapped dataset contains:

- 84 mapped videos.
- 36 label-bearing videos.
- 48 metadata-only videos.
- 2768 behaviour annotation clips.
- 11 original behaviour classes.

The 84-video mapping is important because it shows broad coverage of the available video set. However, only 36 videos contain behaviour labels usable for model training and evaluation. The remaining 48 videos are metadata-only in this project context.

Important evidence paths:

- outputs/v80_final_project_completion/00_source_registry_and_mapping
- outputs/v80_final_project_completion/01_dataset_json_export
- outputs/v80_final_project_completion/03_splits

## 4. Annotation visualizer and correction workflow

The project includes a Streamlit annotation visualizer app:

Full_Unibo_Behaviour_Pipeline/interface/v80d_annotation_gt_visualizer_app.py

The app is documented in:

- docs/DEMO.md
- docs/DEMO_APP_WORKFLOW.md
- docs/OVERLAY_EVIDENCE_DEMO.md

The app supports:

- Loading mapped videos.
- Inspecting annotation intervals.
- Viewing annotation tables.
- Editing correction rows.
- Saving manual correction artifacts separately.
- Keeping original JSON files protected from overwrite claims.

The correct claim is that the app supports annotation inspection and correction review. It is not a production labelling platform.

## 5. Overlay and visual evidence

The project includes overlay and frame visual evidence for mapping and review.

v84 evidence summary:

- Image candidates: 1044.
- Overlay crop candidates: 168.
- Full frame candidates: 87.
- Annotated frame candidates: 549.
- Demo sample images copied: 36.

Important evidence paths:

- outputs/v84_app_demo_evidence/v84c_overlay_demo_board.html
- outputs/v84_app_demo_evidence/v84c_overlay_demo_samples.csv
- outputs/v84_app_demo_evidence/v84c_overlay_demo_samples

Claim boundary:

The overlay/frame images support visual inspection, overlay review, and mapping evidence. They do not prove final automatic identity tracking.

## 6. Split protocols

The project uses two main split protocols.

### 6.1 Split A - grouped video-level split

Split A is the main development and evaluation split. It groups clips at video level to reduce leakage risk between train, validation, and test.

Important evidence:

- outputs/v80_final_project_completion/03_splits/v80e_split_A_clip_split_assignments.csv
- outputs/v80_final_project_completion/03_splits/v80e_split_A_grouped_video_level.csv
- outputs/v80_final_project_completion/03_splits/v80e_split_A_video_leakage_audit.csv
- outputs/v80_final_project_completion/03_splits/v80e_split_A_summary.csv

### 6.2 Split B - cross-camera/pen split

Split B is a harder generalisation split based on camera/pen separation.

Important evidence:

- outputs/v80_final_project_completion/03_splits/v80e_split_B_clip_split_assignments.csv
- outputs/v80_final_project_completion/03_splits/v80e_split_B_cross_camera_pen_level.csv
- outputs/v80_final_project_completion/03_splits/v80e_split_B_video_leakage_audit.csv
- outputs/v80_final_project_completion/03_splits/v80e_split_B_summary.csv

Split B summary:

- Train: 2142 clips, 24 videos, 11 classes.
- Validation: 198 clips, 6 videos, 10 classes.
- Test: 428 clips, 6 videos, 11 classes.

The missing class in validation is important because it can affect model selection stability.

## 7. Frame-proxy baseline

A frame-proxy baseline was implemented as a simple reproducible baseline.

Important evidence:

- outputs/v80_final_project_completion/04_frame_based_baseline

Best frame-proxy results were low. This result is useful as a baseline, but it should not be overinterpreted as final behaviour recognition.

Claim boundary:

The frame-proxy baseline is not final identity-aware tracking and not a production model.

## 8. Clip-level visual baseline

A limited clip visual feature baseline was evaluated in v80.

Best v80 11-class experimental result:

- Model: limited clip visual feature baseline.
- Test macro F1: 0.110956.
- Test accuracy: 0.136364.

Important evidence:

- outputs/v80_final_project_completion/05_clip_based_videomae/v80g5_limited_clip_visual_feature_baseline
- outputs/v80_final_project_completion/06_evaluation_comparison/v80h_final_evaluation_summary.csv

Interpretation:

This result shows that the original 11-class behaviour classification task is difficult under the available visual features and labelled data.

## 9. Full visual-temporal feature extraction

v81 extracted visual-temporal features for all 2768 labelled clips.

Important evidence:

- outputs/v81_performance_improvement/01_full_visual_temporal_features/v81b2_full_visual_temporal_features.csv

The feature table includes clip metadata and visual-temporal summary features such as RGB statistics, brightness statistics, motion statistics, first-last frame difference, histograms, grid brightness, edge statistics, and temporal segment features.

This provided a full-feature baseline for the complete labelled clip set.

## 10. v81 11-class model results

The v81 selected 11-class model improved accuracy but did not improve macro F1.

Selected v81 11-class result:

- Test macro F1: 0.094654.
- Test accuracy: 0.246951.

Reference v80 best 11-class result:

- Test macro F1: 0.110956.
- Test accuracy: 0.136364.

Interpretation:

v81 improved accuracy but reduced macro F1. This means the model became better at predicting frequent classes but did not solve rare-class behaviour recognition.

Important evidence:

- outputs/v81_performance_improvement/02_full_feature_models
- outputs/v81_performance_improvement/03_comparison_with_v80

## 11. Error and class imbalance analysis

The 11-class task is strongly affected by class imbalance and visually overlapping behaviours.

Key issue classes:

- BE
- SI
- IA
- DE
- BOX
- PI

Observed problems include:

- Rare classes with few examples.
- Zero-recall classes.
- Confusions between STI, LAI, and PI.
- Poor fine-grained separation for behaviours with similar visual patterns.

Important evidence:

- outputs/v81_performance_improvement/03_comparison_with_v80/v81e1_class_imbalance_error_analysis_report.md
- outputs/v81_performance_improvement/03_comparison_with_v80/v81e2_next_strategy_decision.md

Interpretation:

The low 11-class macro F1 is not just a model issue. It is also caused by dataset imbalance, limited examples for rare classes, visual ambiguity, and split-level class coverage problems.

## 12. Supplemental grouped behaviour analysis

Because the 11-class task was very difficult, a supplemental grouped behaviour task was evaluated.

Grouped task:

- STI and LAI are grouped as STI_LAI.
- All other behaviours are grouped as OTHER.

This grouped task is not a replacement for the original 11-class task. It is a supplemental analysis to test whether broader behaviour categories are more learnable.

## 13. Split A grouped result

Split A grouped classifier result:

- Selected scheme: dominant_vs_other_2.
- Selected model: logistic regression.
- Test macro F1: 0.639560.
- Test accuracy: 0.652439.

Important evidence:

- outputs/v81_performance_improvement/03_comparison_with_v80/v81f1_decision_summary.csv

Interpretation:

Grouped behaviour classification is substantially easier than the original 11-class task under Split A.

## 14. Split B grouped result

Split B official validation-selected grouped result:

- Selected scheme: dominant_vs_other_2.
- Selected feature set: visual_only.
- Selected model: logistic regression.
- Validation macro F1: 0.598594.
- Test macro F1: 0.238434.
- Test accuracy: 0.313084.

Exploratory best-test model:

- Model: random forest.
- Test macro F1: 0.676086.
- Test accuracy: 0.764019.

Claim boundary:

The exploratory best-test model cannot be selected as the final model because selecting by test performance would be leakage. It is only used to diagnose validation-to-test model selection instability.

Important evidence:

- outputs/v83_splitB_grouped_eval/v83b_splitB_grouped_eval_report.md
- outputs/v83_splitB_grouped_eval/v83c_splitB_grouped_interpretation_report.md

## 15. Leave-camera/pen-out grouped cross-validation

A stronger grouped generalisation test was added with leave-camera/pen-out cross-validation.

Selected result:

- Task: dominant_vs_other_2 grouped classification.
- Feature set: visual_only.
- Selected model: random forest.
- Folds: 6.
- Mean macro F1: 0.582280.
- Min macro F1: 0.398176.
- Max macro F1: 0.718056.
- Mean accuracy: 0.648309.

Aggregate report:

- Aggregate macro F1: 0.615545.
- Aggregate accuracy: 0.639090.

Important evidence:

- outputs/v83_splitB_grouped_eval/v83d_leave_camera_pen_out_grouped_cv_report.md
- outputs/v83_splitB_grouped_eval/v83e_grouped_generalization_summary_report.md

Interpretation:

The grouped classifier is learnable, but evaluation depends on split protocol. Leave-camera/pen-out CV provides stronger evidence than a single Split B validation/test split.

## 16. VideoMAE decision

VideoMAE was investigated as a clip-level video-transformer direction.

Final v85 decision:

- Torch import available.
- Transformers import available.
- VideoMAE classes available.
- Local pretrained VideoMAE cache entries: 0.
- Candidate local pretrained snapshots with weights: 0.
- Pretrained forward smoke: not run.
- Final pretrained VideoMAE model claimed: False.

Important evidence:

- docs/VIDEOMAE_DECISION.md
- outputs/v85_videomae_pretrained_decision/v85c_videomae_final_decision_report.md
- outputs/v80_final_project_completion/05_clip_based_videomae/v80g7_tiny_videomae_limited_training

Final wording:

A tiny random VideoMAE experiment was used to validate the feasibility of the clip-level video-transformer pipeline. A pretrained VideoMAE model was not finalized because no local pretrained snapshot with weights was available during the final offline audit. Therefore, VideoMAE is reported as feasibility evidence and future work, not as the final selected model.

## 17. Final results table

| Area | Result | Interpretation |
|---|---:|---|
| v80 best 11-class macro F1 | 0.110956 | Best safe 11-class checkpoint |
| v80 best 11-class accuracy | 0.136364 | Low accuracy, difficult task |
| v81 selected 11-class macro F1 | 0.094654 | Macro F1 did not improve |
| v81 selected 11-class accuracy | 0.246951 | Accuracy improved on frequent classes |
| Split A grouped macro F1 | 0.639560 | Strong supplemental grouped result |
| Split A grouped accuracy | 0.652439 | Grouped task is more learnable |
| Split B official grouped macro F1 | 0.238434 | Single split model selection unstable |
| Split B official grouped accuracy | 0.313084 | Weak official Split B selected result |
| Leave-camera/pen-out grouped mean macro F1 | 0.582280 | Stronger grouped generalisation evidence |
| Leave-camera/pen-out grouped mean accuracy | 0.648309 | Acceptable supplemental grouped result |

## 18. Main limitations

The main limitations are:

1. The original 11-class task remains unsolved.
2. Rare classes have too few examples.
3. Some behaviours are visually overlapping.
4. Split B validation/test mismatch shows model selection instability.
5. Final automatic identity tracking is not claimed.
6. Pretrained VideoMAE was not finalized.
7. The grouped classifier is supplemental and does not replace the original 11-class task.
8. The project is not production-ready.

## 19. Correct final claims

The project can claim:

- A reproducible pipeline for Unibo pig behaviour analysis.
- 84-video mapping and JSON annotation export.
- A working annotation visualizer and correction-support app.
- Demo documentation and overlay/frame visual evidence.
- Leakage-aware split definitions.
- Frame and clip-level baseline experiments.
- Full 2768-clip visual-temporal feature extraction.
- 11-class model evaluation with honest limitations.
- Grouped behaviour analysis with stronger generalisation evidence.
- VideoMAE feasibility investigation.
- Clear claim boundaries and final handover documentation.

## 20. Claims that must not be made

Do not claim:

- Production-ready behaviour recognition.
- Fully solved 11-class behaviour classification.
- Final automatic identity tracking.
- Final pretrained VideoMAE model.
- Test-selected model as final model.
- Grouped classifier as replacement for 11-class behaviour recognition.

## 21. Conclusion

The final project successfully delivers a reproducible research pipeline for pig behaviour analysis.

The most important technical conclusion is that fine-grained 11-class behaviour recognition remains difficult under the available labels and features, mainly because of class imbalance, rare behaviours, and visual ambiguity. However, broader grouped behaviour classification is substantially more learnable and shows stronger generalisation when evaluated with leave-camera/pen-out cross-validation.

The project therefore provides a useful foundation for future work: improved identity tracking, stronger pretrained video models, more balanced labels, better temporal representations, and production-quality annotation tooling.
