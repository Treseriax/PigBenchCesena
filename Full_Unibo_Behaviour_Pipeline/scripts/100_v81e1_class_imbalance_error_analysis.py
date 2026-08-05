from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

F = Path.home() / "PigBench" / "Full_Unibo_Behaviour_Pipeline"
V81 = F / "outputs/v81_performance_improvement"
MODEL = V81 / "02_full_feature_models"
OUT = V81 / "03_comparison_with_v80"
OUT.mkdir(parents=True, exist_ok=True)

features = pd.read_csv(V81 / "01_full_visual_temporal_features/v81b2_full_visual_temporal_features.csv").fillna("")
pred = pd.read_csv(MODEL / "v81d1_selected_val_model_test_predictions.csv").fillna("")
per_class = pd.read_csv(MODEL / "v81d1_selected_val_model_test_per_class_report.csv").fillna("")
cm = pd.read_csv(MODEL / "v81d1_selected_val_model_test_confusion_matrix.csv", index_col=0).fillna(0)

classes = sorted(features["behaviour_label"].astype(str).unique())

dist = (
    features.groupby(["split_A", "behaviour_label"])
    .size()
    .reset_index(name="clips")
)

pivot = dist.pivot(index="behaviour_label", columns="split_A", values="clips").fillna(0).reset_index()
for col in ["train", "val", "test"]:
    if col not in pivot.columns:
        pivot[col] = 0

pivot["total"] = pivot["train"] + pivot["val"] + pivot["test"]
pivot["train_ratio"] = pivot["train"] / max(float(pivot["train"].sum()), 1.0)
pivot["test_ratio"] = pivot["test"] / max(float(pivot["test"].sum()), 1.0)

pc = per_class.rename(columns={"class": "behaviour_label"}).copy()
pc = pc[pc["behaviour_label"].astype(str).isin(classes)].copy()

for c in ["precision", "recall", "f1-score", "support"]:
    if c in pc.columns:
        pc[c] = pd.to_numeric(pc[c], errors="coerce")

analysis = pivot.merge(pc, on="behaviour_label", how="left")
analysis = analysis.sort_values(["f1-score", "test", "total"], ascending=[True, True, True])
analysis.to_csv(OUT / "v81e1_class_imbalance_per_class_analysis.csv", index=False)

# Confusion pairs.
pairs = []
for true_label in cm.index:
    for pred_label in cm.columns:
        val = int(cm.loc[true_label, pred_label])
        if true_label != pred_label and val > 0:
            pairs.append({
                "true_label": true_label,
                "pred_label": pred_label,
                "count": val,
            })

confusions = pd.DataFrame(pairs)
if len(confusions):
    confusions = confusions.sort_values("count", ascending=False)
else:
    confusions = pd.DataFrame(columns=["true_label", "pred_label", "count"])

confusions.to_csv(OUT / "v81e1_top_confusion_pairs.csv", index=False)

# Prediction distribution.
pred_dist = (
    pred.groupby(["true_label", "pred_label"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
pred_dist.to_csv(OUT / "v81e1_prediction_distribution.csv", index=False)

rare = analysis[(analysis["train"] < 50) | (analysis["test"] < 10)].copy()
zero_recall = analysis[pd.to_numeric(analysis["recall"], errors="coerce").fillna(0) == 0].copy()
low_f1 = analysis[pd.to_numeric(analysis["f1-score"], errors="coerce").fillna(0) < 0.05].copy()

issues = []

if len(rare) > 0:
    issues.append({
        "item": "rare_classes",
        "issue_type": "low_train_or_test_support",
        "severity": "warning",
        "detail": ";".join(rare["behaviour_label"].astype(str).tolist()),
    })

if len(zero_recall) > 0:
    issues.append({
        "item": "zero_recall_classes",
        "issue_type": "zero_recall_on_selected_test_model",
        "severity": "warning",
        "detail": ";".join(zero_recall["behaviour_label"].astype(str).tolist()),
    })

if len(low_f1) > 0:
    issues.append({
        "item": "low_f1_classes",
        "issue_type": "per_class_f1_below_0_05",
        "severity": "warning",
        "detail": ";".join(low_f1["behaviour_label"].astype(str).tolist()),
    })

if not issues:
    issues = [{
        "item": "none",
        "issue_type": "none",
        "severity": "info",
        "detail": "class imbalance/error analysis completed",
    }]

pd.DataFrame(issues).to_csv(OUT / "v81e1_issues.csv", index=False)

hard = sum(1 for x in issues if x["severity"] == "hard")
warning = sum(1 for x in issues if x["severity"] == "warning")

worst = analysis.head(5)
top_conf = confusions.head(10)

decision = pd.DataFrame([{
    "v81e1_decision": "class_imbalance_error_analysis_completed" if hard == 0 else "class_imbalance_error_analysis_has_blocking_issues",
    "classes": len(classes),
    "rare_class_count": len(rare),
    "zero_recall_class_count": len(zero_recall),
    "low_f1_class_count": len(low_f1),
    "top_low_f1_classes": ";".join(worst["behaviour_label"].astype(str).tolist()),
    "hard_issue_count": hard,
    "warning_count": warning,
    "ready_for_v81e2_strategy_selection": hard == 0,
    "claim_scope": "diagnostic_analysis_no_model_training",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

decision.to_csv(OUT / "v81e1_decision_summary.csv", index=False)

report = OUT / "v81e1_class_imbalance_error_analysis_report.md"
report.write_text(
    "# v81e1 Class Imbalance and Error Analysis\n\n"
    "## Scope\n\n"
    "This diagnostic report analyzes why the v81 selected model improved accuracy but not macro F1.\n\n"
    "## Key diagnostics\n\n"
    f"- Number of classes: {len(classes)}\n"
    f"- Rare class count: {len(rare)}\n"
    f"- Zero-recall class count: {len(zero_recall)}\n"
    f"- Low-F1 class count: {len(low_f1)}\n"
    f"- Top low-F1 classes: {', '.join(worst['behaviour_label'].astype(str).tolist())}\n\n"
    "## Interpretation\n\n"
    "Macro F1 remains low because performance is not balanced across all behaviour classes. "
    "Rare classes and visually overlapping behaviours reduce recall and per-class F1. "
    "Accuracy can improve while macro F1 stays low when the model becomes better at dominant classes but still misses rare classes.\n\n"
    "## Top confusion pairs\n\n"
    + (top_conf.to_markdown(index=False) if len(top_conf) else "No off-diagonal confusions found.")
    + "\n\n"
    "## Next step\n\n"
    "Proceed to v81e2 strategy selection. The next model should either simplify the label structure, "
    "focus on grouped behaviour categories, or explicitly address rare-class imbalance.\n",
    encoding="utf-8",
)

note = F / "notes/v81e1_class_imbalance_error_analysis_notes.md"
note.write_text(
    "# v81e1 Class Imbalance and Error Analysis\n\n"
    f"- Decision: {decision.iloc[0]['v81e1_decision']}\n"
    f"- Classes: {len(classes)}\n"
    f"- Rare class count: {len(rare)}\n"
    f"- Zero-recall class count: {len(zero_recall)}\n"
    f"- Low-F1 class count: {len(low_f1)}\n"
    f"- Top low-F1 classes: {decision.iloc[0]['top_low_f1_classes']}\n"
    f"- Hard issues: {hard}\n"
    f"- Warnings: {warning}\n"
    f"- Ready for v81e2 strategy selection: {hard == 0}\n",
    encoding="utf-8",
)

print(decision.to_string(index=False))
print("=== per-class analysis ===")
print(analysis.to_string(index=False))
print("=== top confusions ===")
print(confusions.head(15).to_string(index=False))
print("=== issues ===")
print(pd.DataFrame(issues).to_string(index=False))
