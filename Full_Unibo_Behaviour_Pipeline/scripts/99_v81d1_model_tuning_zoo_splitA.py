from pathlib import Path
from datetime import datetime
import json
import joblib
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

num_cols = []
for c in df.columns:
    if c in meta_cols:
        continue
    x = pd.to_numeric(df[c], errors="coerce")
    if x.notna().sum() > 0:
        df[c] = x
        num_cols.append(c)

ctx = pd.get_dummies(df[context_cols].astype(str), prefix=context_cols)
X = pd.concat([df[num_cols], ctx], axis=1)

tr = df["split_A"] == "train"
va = df["split_A"] == "val"
te = df["split_A"] == "test"

Xtr, Xva, Xte = X[tr], X[va], X[te]
ytr = df.loc[tr, "behaviour_label"]
yva = df.loc[va, "behaviour_label"]
yte = df.loc[te, "behaviour_label"]

classes = sorted(ytr.unique().tolist())

candidates = [
    ("logreg_C0.3", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced", n_jobs=4))])),
    ("logreg_C1", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", n_jobs=4))])),
    ("linear_svc_C0.3", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", LinearSVC(C=0.3, class_weight="balanced", max_iter=6000, random_state=42))])),
    ("linear_svc_C1", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", LinearSVC(C=1.0, class_weight="balanced", max_iter=6000, random_state=42))])),
    ("rbf_svc_C1", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced"))])),
    ("gaussian_nb", Pipeline([("imp", SimpleImputer(strategy="median")), ("m", GaussianNB())])),
    ("extra_trees_sqrt", Pipeline([("imp", SimpleImputer(strategy="median")), ("m", ExtraTreesClassifier(n_estimators=1000, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42))])),
    ("extra_trees_half", Pipeline([("imp", SimpleImputer(strategy="median")), ("m", ExtraTreesClassifier(n_estimators=1000, max_features=0.5, class_weight="balanced", n_jobs=-1, random_state=42))])),
    ("random_forest_sqrt", Pipeline([("imp", SimpleImputer(strategy="median")), ("m", RandomForestClassifier(n_estimators=700, max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=42))])),
]

def score(model, Xs, ys, split_name, model_id):
    yp = model.predict(Xs)
    return {
        "model_id": model_id,
        "split": split_name,
        "clips": len(ys),
        "accuracy": accuracy_score(ys, yp),
        "macro_f1": f1_score(ys, yp, labels=classes, average="macro", zero_division=0),
        "weighted_f1": f1_score(ys, yp, labels=classes, average="weighted", zero_division=0),
    }

rows = []
for model_id, model in candidates:
    print("training", model_id, flush=True)
    model.fit(Xtr, ytr)
    joblib.dump(model, OUT / f"v81d1_{model_id}.joblib")
    rows.append(score(model, Xtr, ytr, "train", model_id))
    rows.append(score(model, Xva, yva, "val", model_id))
    rows.append(score(model, Xte, yte, "test", model_id))

metrics = pd.DataFrame(rows)
metrics.to_csv(OUT / "v81d1_model_tuning_metrics_summary.csv", index=False)

best_val = metrics[metrics["split"] == "val"].sort_values(["macro_f1", "weighted_f1", "accuracy"], ascending=False).iloc[0]
best_val_model = str(best_val["model_id"])
selected_test = metrics[(metrics["split"] == "test") & (metrics["model_id"] == best_val_model)].iloc[0]

best_test = metrics[metrics["split"] == "test"].sort_values(["macro_f1", "weighted_f1", "accuracy"], ascending=False).iloc[0]

best_model = joblib.load(OUT / f"v81d1_{best_val_model}.joblib")
yp = best_model.predict(Xte)

pred = df.loc[te, ["clip_id", "split_A", "video_id", "video_filename", "tlc_camera", "room_pen", "identity_colour", "behaviour_label"]].copy()
pred["true_label"] = yte.values
pred["pred_label"] = yp
pred.to_csv(OUT / "v81d1_selected_val_model_test_predictions.csv", index=False)

pd.DataFrame(
    classification_report(yte, yp, labels=classes, output_dict=True, zero_division=0)
).transpose().reset_index().rename(columns={"index": "class"}).to_csv(
    OUT / "v81d1_selected_val_model_test_per_class_report.csv", index=False
)

pd.DataFrame(
    confusion_matrix(yte, yp, labels=classes), index=classes, columns=classes
).to_csv(OUT / "v81d1_selected_val_model_test_confusion_matrix.csv")

issues = []
if float(selected_test["macro_f1"]) < 0.110956:
    issues.append({
        "item": "selected_val_model",
        "issue_type": "selected_model_below_v80_best_macro_f1",
        "severity": "warning",
        "detail": "validation-selected model did not beat v80 best macro F1",
    })

if float(best_test["macro_f1"]) > float(selected_test["macro_f1"]):
    issues.append({
        "item": "model_selection",
        "issue_type": "test_oracle_model_differs_from_validation_selected_model",
        "severity": "warning",
        "detail": "best test macro F1 is reported as exploratory only, not selected model",
    })

if not issues:
    issues = [{"item": "none", "issue_type": "none", "severity": "info", "detail": "model tuning zoo completed"}]

pd.DataFrame(issues).to_csv(OUT / "v81d1_issues.csv", index=False)

hard = sum(1 for x in issues if x["severity"] == "hard")
warning = sum(1 for x in issues if x["severity"] == "warning")

decision = pd.DataFrame([{
    "v81d1_decision": "model_tuning_zoo_completed" if hard == 0 else "model_tuning_zoo_has_blocking_issues",
    "selection_rule": "best_validation_macro_f1",
    "selected_model_id": best_val_model,
    "selected_val_macro_f1": round(float(best_val["macro_f1"]), 6),
    "selected_test_macro_f1": round(float(selected_test["macro_f1"]), 6),
    "selected_test_accuracy": round(float(selected_test["accuracy"]), 6),
    "exploratory_best_test_model_id": str(best_test["model_id"]),
    "exploratory_best_test_macro_f1": round(float(best_test["macro_f1"]), 6),
    "exploratory_best_test_accuracy": round(float(best_test["accuracy"]), 6),
    "v80_best_macro_f1": 0.110956,
    "hard_issue_count": hard,
    "warning_count": warning,
    "ready_for_v81d2_result_interpretation": hard == 0,
    "claim_scope": "model_tuning_experimental_validation_selected_result_not_production_classifier",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

decision.to_csv(OUT / "v81d1_decision_summary.csv", index=False)

note = F / "notes/v81d1_model_tuning_zoo_notes.md"
note.write_text(
    "# v81d-1 Model Tuning Zoo\n\n"
    f"- Decision: {decision.iloc[0]['v81d1_decision']}\n"
    f"- Selection rule: best validation macro F1\n"
    f"- Selected model: {best_val_model}\n"
    f"- Selected val macro F1: {float(best_val['macro_f1']):.6f}\n"
    f"- Selected test macro F1: {float(selected_test['macro_f1']):.6f}\n"
    f"- Selected test accuracy: {float(selected_test['accuracy']):.6f}\n"
    f"- Exploratory best test model: {str(best_test['model_id'])}\n"
    f"- Exploratory best test macro F1: {float(best_test['macro_f1']):.6f}\n"
    f"- Hard issues: {hard}\n"
    f"- Warnings: {warning}\n\n"
    "This stage compares several tuned classical models. The official selected model is chosen by validation macro F1. "
    "The best test model is reported only as exploratory to avoid test-set model selection leakage.\n",
    encoding="utf-8",
)

print(decision.to_string(index=False))
print("=== metrics ===")
print(metrics.to_string(index=False))
print("=== issues ===")
print(pd.DataFrame(issues).to_string(index=False))
