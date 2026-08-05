# Final Handover Index

## Main status file

- Full_Unibo_Behaviour_Pipeline/FINAL_PROJECT_STATUS_V80_V81.md

This file summarizes the final v80 + v81 project status, claim boundaries, main metrics, and recommended presentation wording.

## v80 safe checkpoint

Package:

- Full_Unibo_Behaviour_Pipeline/outputs/v80_final_project_completion/07_github_handover/V80_Current_Final_Delivery_Package_LITE.zip

SHA256:

- 29ab47d3c5a61a2072e49f547c1f7e338717dd788ed17949c741eda40f486dec

Role:

- Safe reproducible final checkpoint
- Experimental 11-class baseline
- Annotation interface and mapping deliverables

## v81 performance improvement package

Package:

- Full_Unibo_Behaviour_Pipeline/outputs/v81_performance_improvement/04_v81_delivery_package/V81_Performance_Improvement_Delivery_Package_LITE.zip

Role:

- Full 2768-clip visual-temporal feature extraction
- 11-class model tuning and diagnostics
- Class imbalance/error analysis
- Supplemental grouped behaviour classifier
- Final comparison report

## Final result summary

v80 best 11-class:

- Macro F1: 0.110956
- Accuracy: 0.136364

v81 selected 11-class:

- Macro F1: 0.094654
- Accuracy: 0.246951

v81 supplemental grouped classifier:

- Macro F1: 0.639560
- Accuracy: 0.652439

## Claim boundary

The project demonstrates a reproducible experimental behaviour-analysis pipeline. It does not claim a production-ready behaviour classifier.

The grouped classifier is supplemental and diagnostic. It does not replace the original 11-class behaviour classification task.


## v81 package hash

The refreshed v81 ZIP SHA256 is stored externally in:

- Full_Unibo_Behaviour_Pipeline/outputs/v81_performance_improvement/04_v81_delivery_package/v81g3_refreshed_delivery_package_manifest.csv

The ZIP hash is not hard-coded inside the package contents to avoid self-referential hash inconsistency.
