from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

F = Path.home() / "PigBench" / "Full_Unibo_Behaviour_Pipeline"
OUT = F / "outputs/v80_final_project_completion/05_clip_based_videomae/v80g5_limited_clip_visual_feature_baseline"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(OUT / "v80g5_limited_clip_visual_temporal_features.csv").fillna("")

features = [
    "mean_r", "mean_g", "mean_b",
    "std_r", "std_g", "std_b",
    "first_last_diff_mean",
    "temporal_diff_r", "temporal_diff_g", "temporal_diff_b",
    "brightness_mean", "brightness_std", "brightness_min", "brightness_max",
]

classes = sorted(df["behaviour_label"].astype(str).unique())

train = df[df["split"] == "train"].copy()
val = df[df["split"] == "val"].copy()
test = df[df["split"] == "test"].copy()

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    class_weight="balanced",
    random_state=42,
)
clf.fit(train[features], train["behaviour_label"].astype(str))

joblib.dump(clf, OUT / "v80g5_limited_clip_visual_feature_random_forest.joblib")

def eval_split(split_df, split_name):
    y_true = split_df["behaviour_label"].astype(str)
    y_pred = clf.predict(split_df[features])
    proba = clf.predict_proba(split_df[features])
    clf_classes = list(clf.classes_)

    pred = split_df[["clip_id", "split", "video_id", "video_filename", "behaviour_label"]].copy()
    pred["true_label"] = y_true.values
    pred["pred_label"] = y_pred
    pred["confidence"] = proba.max(axis=1)

    for i, c in enumerate(clf_classes):
        pred["proba_" + str(c)] = proba[:, i]

    pred.to_csv(OUT / f"v80g5_{split_name}_clip_predictions.csv", index=False)

    report = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=classes,
            output_dict=True,
            zero_division=0,
        )
    ).transpose().reset_index().rename(columns={"index": "class"})
    report.to_csv(OUT / f"v80g5_{split_name}_per_class_report.csv", index=False)

    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=classes),
        index=classes,
        columns=classes,
    )
    cm.to_csv(OUT / f"v80g5_{split_name}_confusion_matrix.csv")

    return {
        "split": split_name,
        "clips": len(split_df),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0),
    }

metrics = pd.DataFrame([
    eval_split(train, "train"),
    eval_split(val, "val"),
    eval_split(test, "test"),
])
metrics.to_csv(OUT / "v80g5_limited_clip_visual_feature_metrics_summary.csv", index=False)

test_row = metrics[metrics["split"] == "test"].iloc[0]

issues = []
if float(test_row["macro_f1"]) < 0.10:
    issues.append({
        "item": "limited_clip_visual_feature_baseline",
        "issue_type": "low_macro_f1_expected_limitation",
        "severity": "warning",
        "detail": "limited visual feature baseline is weak; this validates pipeline but is not final VideoMAE",
    })

if not issues:
    issues = [{"item": "none", "issue_type": "none", "severity": "info", "detail": "limited clip visual feature baseline completed"}]

pd.DataFrame(issues).to_csv(OUT / "v80g5_issues.csv", index=False)

hard = sum(1 for x in issues if x["severity"] == "hard")
warning = sum(1 for x in issues if x["severity"] == "warning")

decision = pd.DataFrame([{
    "v80g5_decision": "limited_clip_visual_feature_baseline_completed" if hard == 0 else "limited_clip_visual_feature_baseline_has_blocking_issues",
    "model_type": "random_forest_on_clip_visual_temporal_summary_features",
    "train_clips": len(train),
    "val_clips": len(val),
    "test_clips": len(test),
    "test_accuracy": round(float(test_row["accuracy"]), 6),
    "test_macro_f1": round(float(test_row["macro_f1"]), 6),
    "test_weighted_f1": round(float(test_row["weighted_f1"]), 6),
    "hard_issue_count": hard,
    "warning_count": warning,
    "ready_for_v80g6_videomae_feasibility": hard == 0,
    "claim_scope": "limited_clip_visual_feature_baseline_not_final_videomae_not_production_classifier",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
decision.to_csv(OUT / "v80g5_decision_summary.csv", index=False)

note = F / "notes/v80g5_limited_clip_visual_feature_baseline_notes.md"
note.write_text(
    "# v80g5 Limited Clip Visual-Temporal Feature Baseline\n\n"
    f"- Decision: {decision.iloc[0]['v80g5_decision']}\n"
    "- Model type: Random Forest on visual-temporal summary features\n"
    "- Input: 32 sampled RGB frames per 10-second clip\n"
    f"- Train clips: {len(train)}\n"
    f"- Val clips: {len(val)}\n"
    f"- Test clips: {len(test)}\n"
    f"- Test accuracy: {float(test_row['accuracy']):.6f}\n"
    f"- Test macro F1: {float(test_row['macro_f1']):.6f}\n"
    f"- Test weighted F1: {float(test_row['weighted_f1']):.6f}\n"
    f"- Hard issues: {hard}\n"
    f"- Warnings: {warning}\n\n"
    "This is a limited clip-based baseline used to validate the visual-temporal modelling pipeline. "
    "It is not final VideoMAE and not a production classifier.\n",
    encoding="utf-8",
)

print(decision.to_string(index=False))
print(metrics.to_string(index=False))
print(pd.DataFrame(issues).to_string(index=False))
