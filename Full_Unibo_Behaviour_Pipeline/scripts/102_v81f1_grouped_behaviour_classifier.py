from pathlib import Path
from datetime import datetime
import json
import joblib
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

F = Path.home() / "PigBench" / "Full_Unibo_Behaviour_Pipeline"
V81 = F / "outputs/v81_performance_improvement"
INP = V81 / "01_full_visual_temporal_features/v81b2_full_visual_temporal_features.csv"
OUT = V81 / "03_comparison_with_v80"
MODEL_OUT = V81 / "02_full_feature_models"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INP).fillna("")
df["split_A"] = df["split_A"].astype(str)
df["behaviour_label"] = df["behaviour_label"].astype(str)

def group_label(label, scheme):
    label = str(label)
    if scheme == "dominant_vs_other_2":
        if label in ["STI", "LAI"]:
            return "STI_LAI"
        return "OTHER"
    if scheme == "coarse_support_3":
        if label in ["STI", "LAI"]:
            return "STI_LAI"
        if label in ["AN", "BOX", "IN", "NU", "PI"]:
            return "COMMON_OTHER"
        if label in ["BE", "DE", "IA", "SI"]:
            return "RARE_OTHER"
        return "UNMAPPED"
    raise ValueError(scheme)

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

candidates = [
    ("extra_trees", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("m", ExtraTreesClassifier(
            n_estimators=900,
            max_features=0.5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])),
    ("random_forest", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("m", RandomForestClassifier(
            n_estimators=700,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ))
    ])),
    ("logreg", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("m", LogisticRegression(
            C=1.0,
            max_iter=3000,
            class_weight="balanced",
            n_jobs=4
        ))
    ])),
]

all_metrics = []
all_group_counts = []

for scheme in ["dominant_vs_other_2", "coarse_support_3"]:
    y = df["behaviour_label"].apply(lambda z: group_label(z, scheme)).astype(str)
    ytr, yva, yte = y[tr], y[va], y[te]
    classes = sorted(ytr.unique().tolist())

    counts = (
        pd.DataFrame({"split_A": df["split_A"], "group_label": y})
        .groupby(["split_A", "group_label"])
        .size()
        .reset_index(name="clips")
    )
    counts["scheme"] = scheme
    all_group_counts.append(counts)

    for model_id, model in candidates:
        run_id = f"{scheme}_{model_id}"
        print("training", run_id, flush=True)

        model.fit(Xtr, ytr)
        joblib.dump(model, MODEL_OUT / f"v81f1_{run_id}.joblib")

        for split_name, Xs, ys, mask in [
            ("train", Xtr, ytr, tr),
            ("val", Xva, yva, va),
            ("test", Xte, yte, te),
        ]:
            yp = model.predict(Xs)

            all_metrics.append({
                "scheme": scheme,
                "model_id": model_id,
                "split": split_name,
                "clips": len(ys),
                "accuracy": accuracy_score(ys, yp),
                "macro_f1": f1_score(ys, yp, labels=classes, average="macro", zero_division=0),
                "weighted_f1": f1_score(ys, yp, labels=classes, average="weighted", zero_division=0),
            })

            if split_name == "test":
                pred = df.loc[mask, [
                    "clip_id", "split_A", "video_id", "video_filename",
                    "tlc_camera", "room_pen", "identity_colour", "behaviour_label"
                ]].copy()
                pred["group_scheme"] = scheme
                pred["true_group"] = ys.values
                pred["pred_group"] = yp
                pred.to_csv(OUT / f"v81f1_{run_id}_test_predictions.csv", index=False)

                report = pd.DataFrame(
                    classification_report(ys, yp, labels=classes, output_dict=True, zero_division=0)
                ).transpose().reset_index().rename(columns={"index": "class"})
                report.to_csv(OUT / f"v81f1_{run_id}_test_per_group_report.csv", index=False)

                cm = pd.DataFrame(confusion_matrix(ys, yp, labels=classes), index=classes, columns=classes)
                cm.to_csv(OUT / f"v81f1_{run_id}_test_confusion_matrix.csv")

metrics = pd.DataFrame(all_metrics)
metrics.to_csv(OUT / "v81f1_grouped_classifier_metrics_summary.csv", index=False)

group_counts = pd.concat(all_group_counts, ignore_index=True)
group_counts.to_csv(OUT / "v81f1_grouped_label_counts.csv", index=False)

val = metrics[metrics["split"] == "val"].copy()
best_val = val.sort_values(["macro_f1", "weighted_f1", "accuracy"], ascending=False).iloc[0]

best_scheme = str(best_val["scheme"])
best_model = str(best_val["model_id"])

best_test = metrics[
    (metrics["split"] == "test")
    & (metrics["scheme"] == best_scheme)
    & (metrics["model_id"] == best_model)
].iloc[0]

issues = []
if float(best_test["macro_f1"]) < 0.50:
    issues.append({
        "item": "grouped_classifier",
        "issue_type": "grouped_macro_f1_below_0_50",
        "severity": "warning",
        "detail": "grouped classifier is still limited; report as diagnostic only",
    })

if not issues:
    issues = [{
        "item": "none",
        "issue_type": "none",
        "severity": "info",
        "detail": "grouped classifier completed",
    }]

pd.DataFrame(issues).to_csv(OUT / "v81f1_issues.csv", index=False)

hard = sum(1 for x in issues if x["severity"] == "hard")
warning = sum(1 for x in issues if x["severity"] == "warning")

decision = pd.DataFrame([{
    "v81f1_decision": "grouped_classifier_completed" if hard == 0 else "grouped_classifier_has_blocking_issues",
    "selection_rule": "best_validation_macro_f1",
    "selected_group_scheme": best_scheme,
    "selected_model_id": best_model,
    "selected_val_macro_f1": round(float(best_val["macro_f1"]), 6),
    "selected_val_accuracy": round(float(best_val["accuracy"]), 6),
    "selected_test_macro_f1": round(float(best_test["macro_f1"]), 6),
    "selected_test_accuracy": round(float(best_test["accuracy"]), 6),
    "selected_test_weighted_f1": round(float(best_test["weighted_f1"]), 6),
    "hard_issue_count": hard,
    "warning_count": warning,
    "ready_for_v81g_final_comparison_package": hard == 0,
    "claim_scope": "supplemental_grouped_classifier_diagnostic_not_replacement_for_11class",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
decision.to_csv(OUT / "v81f1_decision_summary.csv", index=False)

report = OUT / "v81f1_grouped_classifier_report.md"
report.write_text(
    "# v81f1 Supplemental Grouped Behaviour Classifier\n\n"
    "## Scope\n\n"
    "This model is a supplemental diagnostic classifier. It does not replace the original 11-class behaviour classifier.\n\n"
    "## Grouping schemes\n\n"
    "### dominant_vs_other_2\n"
    "- STI_LAI: STI, LAI\n"
    "- OTHER: all remaining labels\n\n"
    "### coarse_support_3\n"
    "- STI_LAI: STI, LAI\n"
    "- COMMON_OTHER: AN, BOX, IN, NU, PI\n"
    "- RARE_OTHER: BE, DE, IA, SI\n\n"
    "These groups are confusion/support-based diagnostic groups, not definitive biological behaviour categories.\n\n"
    "## Selected result\n\n"
    f"- Selection rule: best validation macro F1\n"
    f"- Selected scheme: {best_scheme}\n"
    f"- Selected model: {best_model}\n"
    f"- Selected validation macro F1: {float(best_val['macro_f1']):.6f}\n"
    f"- Selected test macro F1: {float(best_test['macro_f1']):.6f}\n"
    f"- Selected test accuracy: {float(best_test['accuracy']):.6f}\n\n"
    "## Interpretation\n\n"
    "The grouped classifier tests whether broader behaviour groups are easier to learn than the original 11-class label space. "
    "It should be reported separately from the 11-class classifier and used as evidence about label granularity and dataset imbalance.\n",
    encoding="utf-8",
)

note = F / "notes/v81f1_grouped_classifier_notes.md"
note.write_text(
    "# v81f1 Supplemental Grouped Behaviour Classifier\n\n"
    f"- Decision: {decision.iloc[0]['v81f1_decision']}\n"
    f"- Selected group scheme: {best_scheme}\n"
    f"- Selected model: {best_model}\n"
    f"- Selected val macro F1: {float(best_val['macro_f1']):.6f}\n"
    f"- Selected test macro F1: {float(best_test['macro_f1']):.6f}\n"
    f"- Selected test accuracy: {float(best_test['accuracy']):.6f}\n"
    f"- Hard issues: {hard}\n"
    f"- Warnings: {warning}\n"
    f"- Ready for v81g final comparison/package: {hard == 0}\n",
    encoding="utf-8",
)

with open(OUT / "v81f1_grouping_definition.json", "w", encoding="utf-8") as f:
    json.dump({
        "dominant_vs_other_2": {
            "STI_LAI": ["STI", "LAI"],
            "OTHER": ["AN", "BE", "BOX", "DE", "IA", "IN", "NU", "PI", "SI"]
        },
        "coarse_support_3": {
            "STI_LAI": ["STI", "LAI"],
            "COMMON_OTHER": ["AN", "BOX", "IN", "NU", "PI"],
            "RARE_OTHER": ["BE", "DE", "IA", "SI"]
        },
        "claim_boundary": "diagnostic grouping based on confusion/support analysis; not a replacement for 11-class labels"
    }, f, indent=2)

print(decision.to_string(index=False))
print("=== metrics ===")
print(metrics.to_string(index=False))
print("=== group counts ===")
print(group_counts.to_string(index=False))
print("=== issues ===")
print(pd.DataFrame(issues).to_string(index=False))
