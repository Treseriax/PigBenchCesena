# Final Project Status — v80 + v81

## Project scope

This project builds a reproducible behaviour-analysis pipeline for the Unibo pig videos.

The pipeline includes:

- Full 84-video mapping/status table
- JSON export for all mapped videos
- Annotation visualizer/correction interface
- Train/validation/test split definitions
- Frame-proxy baseline
- Limited clip-based visual feature baseline
- Tiny random VideoMAE feasibility experiment
- Full 2768-clip visual-temporal feature extraction
- 11-class experimental behaviour classifiers
- Class imbalance and error analysis
- Supplemental grouped behaviour classifier
- Delivery packages for v80 and v81

## v80 checkpoint

v80 is the safe final project checkpoint.

Main v80 result:

- Best experimental model: limited clip visual feature baseline
- 11-class macro F1: 0.110956
- 11-class accuracy: 0.136364

v80 claim boundary:

- No production classifier is claimed.
- No full pretrained VideoMAE classifier is claimed.
- No final colour-to-track identity assignment is claimed.
- v80 is a reproducible experimental checkpoint.

## v81 performance improvement branch

v81 extends v80 with full 2768-clip visual-temporal feature extraction and additional model analysis.

Main v81 selected 11-class result:

- Selected model: extra_trees_half
- Selection rule: best validation macro F1
- 11-class macro F1: 0.094654
- 11-class accuracy: 0.246951

Interpretation:

- v81 improves 11-class accuracy compared with v80.
- v81 does not improve 11-class macro F1 compared with v80.
- The main bottleneck is class imbalance, rare labels, and visually overlapping behaviours.

## Class imbalance finding

The selected v81 11-class model has zero recall for several classes:

- BE
- SI
- IA
- DE
- BOX

Low-F1 classes:

- BE
- SI
- IA
- DE
- BOX
- PI

Major confusions include:

- STI → LAI
- LAI → STI
- STI → PI
- LAI → PI

This explains why accuracy can improve while macro F1 remains limited.

## Supplemental grouped classifier

A grouped classifier was added as a diagnostic experiment.

Selected grouped result:

- Grouping scheme: dominant_vs_other_2
- Model: logistic regression
- Selection rule: best validation macro F1
- Test macro F1: 0.639560
- Test accuracy: 0.652439

Grouping:

- STI_LAI: STI, LAI
- OTHER: all remaining labels

Interpretation:

The grouped classifier shows that broader behaviour categories are more learnable under the current feature representation.

Claim boundary:

- The grouped classifier is supplemental and diagnostic.
- It does not replace the original 11-class classifier.
- It is not a production-ready behaviour classifier.

## Final claim boundaries

The project can claim:

- A reproducible end-to-end behaviour-analysis pipeline
- Full mapping and JSON export for the available Unibo videos
- A working annotation/correction interface
- Full 2768-clip feature extraction
- Experimental 11-class behaviour classification
- Evidence that 11-class recognition is limited by imbalance and label overlap
- A supplemental grouped classifier with stronger performance
- Clear documentation, metrics, limitations, and delivery packages

The project must not claim:

- Production-ready behaviour recognition
- Final pretrained VideoMAE performance
- Fully solved 11-class classification
- Final colour-to-track identity assignment
- Biological certainty for the grouped labels

## Delivery packages

v80 package:

- V80_Current_Final_Delivery_Package_LITE.zip
- SHA256: 29ab47d3c5a61a2072e49f547c1f7e338717dd788ed17949c741eda40f486dec

v81 package:

- V81_Performance_Improvement_Delivery_Package_LITE.zip
- SHA256 is stored in the external refreshed package manifest:
  - v81g3_refreshed_delivery_package_manifest.csv

Note: the v81 package SHA is not hard-coded inside this status file to avoid self-referential hash inconsistency after package refreshes.

## Recommended final presentation wording

The original 11-class behaviour classification task remains challenging due to rare classes, class imbalance, and visually overlapping behaviours. The selected v81 11-class model improves accuracy but not macro F1. Therefore, the 11-class model is reported as an experimental classifier with clear limitations.

A supplemental grouped behaviour classifier is reported separately. It achieves much stronger performance, suggesting that broader behaviour categories are more learnable with the current feature representation. This grouped classifier is diagnostic and does not replace the original 11-class task.
