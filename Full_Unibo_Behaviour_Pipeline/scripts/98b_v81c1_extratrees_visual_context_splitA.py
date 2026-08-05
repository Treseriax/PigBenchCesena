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

context_cols = ["tlc_camera", "room_pen", "identity_colour"]

meta_cols = {
    "clip_id", "split_A", "video_id", "video_filename",
    "behaviour_label", "claim_scope",
    "tlc_camera", "room_pen", "identity_colour",
}

numeric_cols = []
for c in df.columns:
    if c in meta_cols:
        continue
    x = pd.to_numeric(df[c], errors="coerce")
    if x.notna().sum() > 0:
        df[c] = x
        numeric_cols.append(c)

context = pd.get_dummies(df[context_cols].astype(str), prefix=context_cols)
X_all = pd.concat([df[numeric_cols], context], axis=1)

train_idx = df["split_A"] == "train"
val_idx = df["split_A"] == "val"
test_idx = df["split_A"] == "test"

train = df[train_idx].copy()
val = df[val_idx].copy()
test = df[test_idx].copy()

X_train = X_all[train_idx].copy()
X_val = X_all[val_idx].copy()
X_test = X_all[test_idx].copy()

y_train = train["behaviour_label"].astype(str)
y_val = val["behaviour_label"].astype(str)
y_test = test["behaviour_label"].astype(str)

classes = sorted(y_train.unique().tolist())

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", ExtraTreesClassifier(
        n_estimators=900,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=1
    ))
])

pipe.fit(X_train, y_train)

joblib.dump(pipe, OUT / "v81c1_extratrees_visual_context.joblib")

with open(OUT / "v81c1_extratrees_visual_context_features.json", "w", encoding="utf-8") as f:
    json.dump({
        "numeric_visual_features": numeric_cols,
        "context_features": context_cols,
        "expanded_feature_count": int(X_all.shape[1]),
        "classes": classes,
        "claim_boundary": "visual_plus_context_may_use_camera_pen_identity_metadata",
    }, f, indent=2)

def evaluate(split_df, X, y_true, split_name):
    y_pred = pipe.predict(X)

    pred = split_df[["clip_id", "split_A", "video_id", "video_filename", "tlc_camera", "room_pen", "identity_colour", "behaviour_label"]].copy()
    pred["true_label"] = y_true.values
    pred["pred_label"] = y_pred
    pred.to_csv(OUT / f"v81c1_extratrees_visual_context_{split_name}_predictions.csv", index=False)

    report = pd.DataFrame(
        classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    ).transpose().reset_index().rename(columns={"index": "class"})
    report.to_csv(OUT / f"v81c1_extratrees_visual_context_{split_name}_per_class_report.csv", index=False)

    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=classes), index=classes, columns=classes)
    cm.to_csv(OUT / f"v81c1_extratrees_visual_context_{split_name}_confusion_matrix.csv")

    return {
        "model_name": "extra_trees",
        "feature_set": "visual_plus_context",
        "split": split_name,
        "clips": len(split_df),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0),
    }

metrics = pd.DataFrame([
    evaluate(train, X_train, y_train, "train"),
    evaluate(val, X_val, y_val, "val"),
    evaluate(test, X_test, y_test, "test"),
])
metrics.to_csv(OUT / "v81c1_extratrees_visual_context_metrics.csv", index=False)

test_row = metrics[metrics["split"] == "test"].iloc[0]

issues = []
if float(test_row["macro_f1"]) < 0.110956:
    issues.append({
        "item": "extra_trees_visual_context",
        "issue_type": "no_improvement_over_v80_best_macro_f1",
        "severity": "warning",
        "detail": "test macro F1 below v80 best limited baseline 0.110956",
    })

if not issues:
    issues = [{"item": "none", "issue_type": "none", "severity": "info", "detail": "ExtraTrees visual+context model completed"}]

pd.DataFrame(issues).to_csv(OUT / "v81c1_extratrees_visual_context_issues.csv", index=False)

hard = sum(1 for x in issues if x["severity"] == "hard")
warning = sum(1 for x in issues if x["severity"] == "warning")

decision = pd.DataFrame([{
    "v81c1b_decision": "extratrees_visual_context_completed",
    "model_name": "extra_trees",
    "feature_set": "visual_plus_context",
    "train_clips": len(train),
    "val_clips": len(val),
    "test_clips": len(test),
    "numeric_feature_columns": len(numeric_cols),
    "expanded_feature_columns": int(X_all.shape[1]),
    "test_accuracy": round(float(test_row["accuracy"]), 6),
    "test_macro_f1": round(float(test_row["macro_f1"]), 6),
    "test_weighted_f1": round(float(test_row["weighted_f1"]), 6),
    "hard_issue_count": hard,
    "warning_count": warning,
    "ready_for_v81c1c_model_comparison": hard == 0,
    "claim_scope": "full_visual_plus_context_model_experimental_context_features_report_separately",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
decision.to_csv(OUT / "v81c1b_decision_summary.csv", index=False)

note = F / "notes/v81c1b_extratrees_visual_context_notes.md"
note.write_text(
    "# v81c1-B ExtraTrees Visual+Context Split A\n\n"
    f"- Decision: {decision.iloc[0]['v81c1b_decision']}\n"
    f"- Train/Val/Test clips: {len(train)}/{len(val)}/{len(test)}\n"
    f"- Numeric visual feature columns: {len(numeric_cols)}\n"
    f"- Expanded feature columns: {int(X_all.shape[1])}\n"
    f"- Test accuracy: {float(test_row['accuracy']):.6f}\n"
    f"- Test macro F1: {float(test_row['macro_f1']):.6f}\n"
    f"- Test weighted F1: {float(test_row['weighted_f1']):.6f}\n"
    f"- Hard issues: {hard}\n"
    f"- Warnings: {warning}\n\n"
    "This model uses visual-temporal features plus context metadata: tlc_camera, room_pen, and identity_colour. "
    "It must be reported separately from visual-only models because context features may encode dataset-specific information.\n",
    encoding="utf-8",
)

print(decision.to_string(index=False))
print("=== metrics ===")
print(metrics.to_string(index=False))
print("=== issues ===")
print(pd.DataFrame(issues).to_string(index=False))
