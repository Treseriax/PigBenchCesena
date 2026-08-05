from pathlib import Path
from datetime import datetime
import json
import joblib
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

F = Path.home() / "PigBench" / "Full_Unibo_Behaviour_Pipeline"
V81 = F / "outputs/v81_performance_improvement"
INP = V81 / "01_full_visual_temporal_features/v81b2_full_visual_temporal_features.csv"
OUT = V81 / "02_full_feature_models"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INP).fillna("")
df["split_A"] = df["split_A"].astype(str)
df["behaviour_label"] = df["behaviour_label"].astype(str)

meta_cols = {
    "clip_id", "split_A", "video_id", "video_filename",
    "tlc_camera", "room_pen", "identity_colour",
    "behaviour_label", "claim_scope"
}

feature_cols = []
for c in df.columns:
    if c in meta_cols:
        continue
    x = pd.to_numeric(df[c], errors="coerce")
    if x.notna().sum() > 0:
        df[c] = x
        feature_cols.append(c)

train = df[df["split_A"] == "train"].copy()
val = df[df["split_A"] == "val"].copy()
test = df[df["split_A"] == "test"].copy()

classes = sorted(train["behaviour_label"].unique())

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", ExtraTreesClassifier(
        n_estimators=800,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=1
    ))
])

pipe.fit(train[feature_cols], train["behaviour_label"])

joblib.dump(pipe, OUT / "v81c1_extratrees_visual_only.joblib")

with open(OUT / "v81c1_extratrees_visual_only_features.json", "w", encoding="utf-8") as f:
    json.dump({"feature_cols": feature_cols, "classes": classes}, f, indent=2)

def evaluate(split_df, split_name):
    y_true = split_df["behaviour_label"].astype(str)
    y_pred = pipe.predict(split_df[feature_cols])

    pred = split_df[["clip_id", "split_A", "video_id", "video_filename", "behaviour_label"]].copy()
    pred["true_label"] = y_true.values
    pred["pred_label"] = y_pred
    pred.to_csv(OUT / f"v81c1_extratrees_visual_only_{split_name}_predictions.csv", index=False)

    report = pd.DataFrame(
        classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    ).transpose().reset_index().rename(columns={"index": "class"})
    report.to_csv(OUT / f"v81c1_extratrees_visual_only_{split_name}_per_class_report.csv", index=False)

    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=classes), index=classes, columns=classes)
    cm.to_csv(OUT / f"v81c1_extratrees_visual_only_{split_name}_confusion_matrix.csv")

    return {
        "model_name": "extra_trees",
        "feature_set": "visual_only",
        "split": split_name,
        "clips": len(split_df),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0),
    }

metrics = pd.DataFrame([
    evaluate(train, "train"),
    evaluate(val, "val"),
    evaluate(test, "test"),
])
metrics.to_csv(OUT / "v81c1_extratrees_visual_only_metrics.csv", index=False)

test_row = metrics[metrics["split"] == "test"].iloc[0]

issues = []
if float(test_row["macro_f1"]) < 0.110956:
    issues.append({
        "item": "extra_trees_visual_only",
        "issue_type": "no_improvement_over_v80_best_macro_f1",
        "severity": "warning",
        "detail": "test macro F1 below v80 best limited baseline 0.110956",
    })

if not issues:
    issues = [{"item": "none", "issue_type": "none", "severity": "info", "detail": "ExtraTrees visual-only model completed"}]

pd.DataFrame(issues).to_csv(OUT / "v81c1_extratrees_visual_only_issues.csv", index=False)

hard = sum(1 for x in issues if x["severity"] == "hard")
warning = sum(1 for x in issues if x["severity"] == "warning")

decision = pd.DataFrame([{
    "v81c1a_decision": "extratrees_visual_only_completed",
    "model_name": "extra_trees",
    "feature_set": "visual_only",
    "train_clips": len(train),
    "val_clips": len(val),
    "test_clips": len(test),
    "feature_columns": len(feature_cols),
    "test_accuracy": round(float(test_row["accuracy"]), 6),
    "test_macro_f1": round(float(test_row["macro_f1"]), 6),
    "test_weighted_f1": round(float(test_row["weighted_f1"]), 6),
    "hard_issue_count": hard,
    "warning_count": warning,
    "ready_for_v81c1b_context_model": hard == 0,
    "claim_scope": "full_visual_feature_model_experimental_not_production_classifier",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
decision.to_csv(OUT / "v81c1a_decision_summary.csv", index=False)

note = F / "notes/v81c1a_extratrees_visual_only_notes.md"
note.write_text(
    "# v81c1-A ExtraTrees Visual-Only Split A\n\n"
    f"- Decision: {decision.iloc[0]['v81c1a_decision']}\n"
    f"- Train/Val/Test clips: {len(train)}/{len(val)}/{len(test)}\n"
    f"- Feature columns: {len(feature_cols)}\n"
    f"- Test accuracy: {float(test_row['accuracy']):.6f}\n"
    f"- Test macro F1: {float(test_row['macro_f1']):.6f}\n"
    f"- Test weighted F1: {float(test_row['weighted_f1']):.6f}\n"
    f"- Hard issues: {hard}\n"
    f"- Warnings: {warning}\n\n"
    "This is a full 2768-clip visual-only feature model on Split A. It is experimental and not a production classifier.\n",
    encoding="utf-8",
)

print(decision.to_string(index=False))
print("=== metrics ===")
print(metrics.to_string(index=False))
print("=== issues ===")
print(pd.DataFrame(issues).to_string(index=False))
