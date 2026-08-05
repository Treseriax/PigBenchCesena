from pathlib import Path
from datetime import datetime
import pandas as pd

F = Path.home() / "PigBench" / "Full_Unibo_Behaviour_Pipeline"
V81 = F / "outputs/v81_performance_improvement"
INP = V81 / "03_comparison_with_v80"
OUT = INP
OUT.mkdir(parents=True, exist_ok=True)

per_class = pd.read_csv(INP / "v81e1_class_imbalance_per_class_analysis.csv").fillna("")
conf = pd.read_csv(INP / "v81e1_top_confusion_pairs.csv").fillna("")
d2 = pd.read_csv(V81 / "02_full_feature_models/v81d2_decision_summary.csv").fillna("")

for c in ["recall", "f1-score", "train", "val", "test", "total"]:
    if c in per_class.columns:
        per_class[c] = pd.to_numeric(per_class[c], errors="coerce").fillna(0)

zero_recall = per_class[per_class["recall"] == 0]["behaviour_label"].astype(str).tolist()
low_f1 = per_class[per_class["f1-score"] < 0.05]["behaviour_label"].astype(str).tolist()
val_missing = per_class[per_class["val"] == 0]["behaviour_label"].astype(str).tolist()
rare = per_class[(per_class["train"] < 50) | (per_class["test"] < 10)]["behaviour_label"].astype(str).tolist()

selected_macro = float(d2.iloc[0]["selected_test_macro_f1"])
selected_acc = float(d2.iloc[0]["selected_test_accuracy"])
v80_macro = float(d2.iloc[0]["v80_best_macro_f1"])
v80_acc = float(d2.iloc[0]["v80_best_accuracy"])
exploratory_model = str(d2.iloc[0]["exploratory_best_test_model"])
exploratory_macro = float(d2.iloc[0]["exploratory_best_test_macro_f1"])

strategy = pd.DataFrame([
    {
        "option": "continue_11_class_tuning_only",
        "decision": "not_primary_next_step",
        "reason": "The selected v81 model improved accuracy but not macro F1; five classes have zero recall.",
        "risk": "More model tuning may overfit the incomplete/imbalanced validation split.",
    },
    {
        "option": "select_best_test_model",
        "decision": "rejected_for_final_selection",
        "reason": f"The exploratory best-test model {exploratory_model} nearly matches v80 macro F1, but it was identified by looking at test results.",
        "risk": "Selecting it officially would introduce test-set model-selection leakage.",
    },
    {
        "option": "add_grouped_behaviour_classifier",
        "decision": "chosen_next_step",
        "reason": "The main failure mode is label granularity, rare classes, and visually overlapping behaviours; grouped labels can test whether broader behaviour categories are learnable.",
        "risk": "Grouped classifier is supplemental and must not replace the 11-class result.",
    },
    {
        "option": "report_11_class_limitations",
        "decision": "required",
        "reason": "The 11-class model remains important because the original labels are 11 behaviour categories.",
        "risk": "Claim boundaries must state that the classifier is experimental, not production-ready.",
    },
])

strategy.to_csv(OUT / "v81e2_strategy_options.csv", index=False)

issues = pd.DataFrame([
    {
        "item": "zero_recall_classes",
        "issue_type": "model_misses_multiple_classes",
        "severity": "warning",
        "detail": ";".join(zero_recall),
    },
    {
        "item": "validation_split",
        "issue_type": "missing_classes_in_validation",
        "severity": "warning",
        "detail": ";".join(val_missing),
    },
    {
        "item": "grouped_classifier",
        "issue_type": "supplemental_not_replacement",
        "severity": "info",
        "detail": "Grouped classifier should be reported as an additional diagnostic/experimental model, not as replacement for 11-class classifier.",
    },
])
issues.to_csv(OUT / "v81e2_issues.csv", index=False)

decision = pd.DataFrame([{
    "v81e2_decision": "proceed_to_v81f_grouped_behaviour_classifier",
    "selected_v81_11class_macro_f1": round(selected_macro, 6),
    "selected_v81_11class_accuracy": round(selected_acc, 6),
    "v80_best_macro_f1": round(v80_macro, 6),
    "v80_best_accuracy": round(v80_acc, 6),
    "zero_recall_classes": ";".join(zero_recall),
    "low_f1_classes": ";".join(low_f1),
    "rare_classes": ";".join(rare),
    "validation_missing_classes": ";".join(val_missing),
    "chosen_next_strategy": "supplemental_grouped_behaviour_classifier_plus_11class_limitations",
    "hard_issue_count": 0,
    "warning_count": 2,
    "ready_for_v81f_grouped_classifier": True,
    "claim_scope": "strategy_selection_no_model_training",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
decision.to_csv(OUT / "v81e2_decision_summary.csv", index=False)

report = OUT / "v81e2_strategy_selection_report.md"
report.write_text(
    "# v81e2 Strategy Selection\n\n"
    "## Current finding\n\n"
    "The v81 full-feature models improved accuracy but did not improve 11-class macro F1. "
    "The bottleneck is class-balanced recognition: several classes have zero recall and several classes have very low support.\n\n"
    "## Evidence\n\n"
    f"- Selected v81 11-class test macro F1: {selected_macro:.6f}\n"
    f"- Selected v81 11-class test accuracy: {selected_acc:.6f}\n"
    f"- v80 best macro F1: {v80_macro:.6f}\n"
    f"- Zero-recall classes: {', '.join(zero_recall)}\n"
    f"- Low-F1 classes: {', '.join(low_f1)}\n"
    f"- Validation-missing classes: {', '.join(val_missing)}\n\n"
    "## Decision\n\n"
    "Proceed to v81f with a supplemental grouped behaviour classifier.\n\n"
    "This is not a replacement for the original 11-class classifier. "
    "It is an additional experiment to test whether broader behaviour categories are more learnable under the current feature representation and dataset imbalance.\n\n"
    "## Claim boundary\n\n"
    "- Keep the 11-class classifier as the original-label result.\n"
    "- Report the grouped classifier separately as a supplemental/diagnostic model.\n"
    "- Do not select the exploratory best-test model as the official final model.\n"
    "- Do not claim production-ready behaviour recognition.\n",
    encoding="utf-8",
)

note = F / "notes/v81e2_strategy_selection_notes.md"
note.write_text(
    "# v81e2 Strategy Selection\n\n"
    f"- Decision: {decision.iloc[0]['v81e2_decision']}\n"
    f"- Selected v81 11-class macro F1: {selected_macro:.6f}\n"
    f"- Selected v81 11-class accuracy: {selected_acc:.6f}\n"
    f"- Zero-recall classes: {';'.join(zero_recall)}\n"
    f"- Low-F1 classes: {';'.join(low_f1)}\n"
    f"- Validation-missing classes: {';'.join(val_missing)}\n"
    f"- Chosen next strategy: supplemental grouped behaviour classifier\n"
    f"- Ready for v81f grouped classifier: True\n",
    encoding="utf-8",
)

print(decision.to_string(index=False))
print("=== strategy options ===")
print(strategy.to_string(index=False))
print("=== issues ===")
print(issues.to_string(index=False))
