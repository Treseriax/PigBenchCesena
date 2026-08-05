# v81d-1 Model Tuning Zoo

- Decision: model_tuning_zoo_completed
- Selection rule: best validation macro F1
- Selected model: extra_trees_half
- Selected val macro F1: 0.080776
- Selected test macro F1: 0.094654
- Selected test accuracy: 0.246951
- Exploratory best test model: logreg_C1
- Exploratory best test macro F1: 0.109981
- Hard issues: 0
- Warnings: 2

This stage compares several tuned classical models. The official selected model is chosen by validation macro F1. The best test model is reported only as exploratory to avoid test-set model selection leakage.
