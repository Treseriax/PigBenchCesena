from pathlib import Path
from datetime import datetime
import csv
import json
import math
import pandas as pd
import numpy as np


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V36B = W7 / "outputs" / "week7_full_crop_lightweight_feature_extraction_v36b"
CONFIG_IN = V36B / "week7_v36b_recommended_v37_baseline_classifier_config.json"

OUT_ROOT = W7 / "outputs" / "week7_crop_baseline_classifier_dryrun_v37"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_PREDICTIONS = OUT_ROOT / "week7_v37_crop_baseline_predictions.csv"
OUT_METRICS = OUT_ROOT / "week7_v37_crop_baseline_metrics.csv"
OUT_PER_CLASS = OUT_ROOT / "week7_v37_crop_baseline_per_class_metrics.csv"
OUT_CONFUSION = OUT_ROOT / "week7_v37_crop_baseline_confusion_matrices.csv"
OUT_MODEL_SUMMARY = OUT_ROOT / "week7_v37_crop_baseline_model_summary.csv"
OUT_DATA_AUDIT = OUT_ROOT / "week7_v37_crop_baseline_data_audit.csv"
OUT_REPORT = OUT_ROOT / "week7_v37_crop_baseline_classifier_dryrun_report.md"
OUT_DECISION = OUT_ROOT / "week7_v37_crop_baseline_classifier_dryrun_decision_summary.csv"
OUT_ISSUES = OUT_ROOT / "week7_v37_crop_baseline_classifier_dryrun_issues.csv"
OUT_V38_CONFIG = OUT_ROOT / "week7_v37_recommended_v38_clip_feature_preparation_config.json"
OUT_NOTE = W7 / "notes" / "week7_v37_crop_baseline_classifier_dryrun_notes.md"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def clean(v):
    if pd.isna(v):
        return ""
    return str(v).strip()


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def read_df(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def md_table(rows, columns):
    if not rows:
        return "_No rows available._"

    out = []
    out.append("| " + " | ".join(columns) + " |")
    out.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for r in rows:
        vals = []
        for c in columns:
            x = str(r.get(c, ""))
            x = x.replace("\n", " ").replace("|", "/")
            vals.append(x)
        out.append("| " + " | ".join(vals) + " |")

    return "\n".join(out)


def majority_train_classifier_fit(y_train):
    values, counts = np.unique(y_train, return_counts=True)
    return values[np.argmax(counts)]


def nearest_centroid_fit(X_train, y_train, classes):
    centroids = {}
    for c in classes:
        Xc = X_train[y_train == c]
        if len(Xc):
            centroids[c] = Xc.mean(axis=0)
    return centroids


def nearest_centroid_predict(X, centroids, classes):
    preds = []
    for row in X:
        best_c = None
        best_d = None
        for c in classes:
            if c not in centroids:
                continue
            d = float(np.linalg.norm(row - centroids[c]))
            if best_d is None or d < best_d:
                best_d = d
                best_c = c
        preds.append(best_c if best_c is not None else classes[0])
    return np.array(preds)


def compute_metrics(y_true, y_pred, classes):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    total = len(y_true)
    accuracy = float(np.mean(y_true == y_pred)) if total else 0.0

    per_class = []

    for c in classes:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        tn = int(((y_true != c) & (y_pred != c)).sum())
        support = int((y_true == c).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        per_class.append({
            "class": c,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    macro_precision = float(np.mean([r["precision"] for r in per_class])) if per_class else 0.0
    macro_recall = float(np.mean([r["recall"] for r in per_class])) if per_class else 0.0
    macro_f1 = float(np.mean([r["f1"] for r in per_class])) if per_class else 0.0

    weighted_f1 = 0.0
    total_support = sum(r["support"] for r in per_class)
    if total_support:
        weighted_f1 = sum(r["f1"] * r["support"] for r in per_class) / total_support

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }


def confusion_matrix_rows(y_true, y_pred, classes, model_name, split_name):
    rows = []
    for true_c in classes:
        for pred_c in classes:
            rows.append({
                "model_name": model_name,
                "split": split_name,
                "true_label": true_c,
                "predicted_label": pred_c,
                "count": int(((y_true == true_c) & (y_pred == pred_c)).sum()),
            })
    return rows


issues = []

config = read_json(CONFIG_IN)

if not config:
    issues.append({
        "item": "v37_config",
        "issue_type": "hard_missing_or_unreadable_config",
        "issue_detail": str(CONFIG_IN),
    })

feature_matrix_path = Path(config.get("feature_matrix_standardized", "")) if config else Path("")
metadata_path = Path(config.get("metadata", "")) if config else Path("")
feature_cols = config.get("feature_columns", []) if config else []
label_column = config.get("label_column", "behaviour_code") if config else "behaviour_code"
split_column = config.get("split_column", "split") if config else "split"
eligible_classes = config.get("eligible_classes", []) if config else []

X_df = read_df(feature_matrix_path)
meta = read_df(metadata_path)

if len(X_df) == 0:
    issues.append({
        "item": "feature_matrix_standardized",
        "issue_type": "hard_missing_or_empty_input",
        "issue_detail": str(feature_matrix_path),
    })

if len(meta) == 0:
    issues.append({
        "item": "metadata",
        "issue_type": "hard_missing_or_empty_input",
        "issue_detail": str(metadata_path),
    })

if "v36b_feature_row_id" not in X_df.columns or "v36b_feature_row_id" not in meta.columns:
    issues.append({
        "item": "v36b_feature_row_id",
        "issue_type": "hard_missing_join_key",
        "issue_detail": "Both matrix and metadata must contain v36b_feature_row_id.",
    })

for c in feature_cols:
    if c not in X_df.columns:
        issues.append({
            "item": c,
            "issue_type": "hard_missing_feature_column",
            "issue_detail": f"Feature column `{c}` missing from matrix.",
        })

for c in [label_column, split_column]:
    if c not in meta.columns:
        issues.append({
            "item": c,
            "issue_type": "hard_missing_metadata_column",
            "issue_detail": f"Metadata column `{c}` missing.",
        })

if len(issues):
    issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
    safe_to_csv(issues_df, OUT_ISSUES)
    print("Hard input issues:")
    print(issues_df.to_string(index=False))
    raise SystemExit(1)

data = meta.merge(X_df, on="v36b_feature_row_id", how="inner")
data[label_column] = data[label_column].fillna("").astype(str).str.strip()
data[split_column] = data[split_column].fillna("").astype(str).str.strip()

data = data[data[label_column].isin(eligible_classes)].copy()
data = data[data[split_column].isin(["train", "val", "test"])].copy()
data = data.reset_index(drop=True)

for c in feature_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce")

nan_cells = int(data[feature_cols].isna().sum().sum())

if nan_cells > 0:
    issues.append({
        "item": "feature_matrix",
        "issue_type": "hard_nan_values_found",
        "issue_detail": f"{nan_cells} NaN values found.",
    })

classes = sorted(data[label_column].unique().tolist())

train = data[data[split_column] == "train"].copy()
val = data[data[split_column] == "val"].copy()
test = data[data[split_column] == "test"].copy()

split_data = {
    "train": train,
    "val": val,
    "test": test,
}

for split_name, df_split in split_data.items():
    if len(df_split) == 0:
        issues.append({
            "item": split_name,
            "issue_type": "hard_empty_split",
            "issue_detail": f"{split_name} split is empty.",
        })

for c in classes:
    if (train[label_column] == c).sum() == 0:
        issues.append({
            "item": c,
            "issue_type": "hard_class_missing_from_train",
            "issue_detail": f"Class {c} has no train examples.",
        })

if len([x for x in issues if str(x["issue_type"]).startswith("hard_")]):
    issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
    safe_to_csv(issues_df, OUT_ISSUES)
    print("Hard data issues:")
    print(issues_df.to_string(index=False))
    raise SystemExit(1)

X_train = train[feature_cols].to_numpy(dtype=float)
y_train = train[label_column].to_numpy()

majority_class = majority_train_classifier_fit(y_train)
centroids = nearest_centroid_fit(X_train, y_train, classes)

model_summaries = [
    {
        "model_name": "majority_train_class",
        "model_type": "dummy_baseline",
        "description": f"Always predicts the majority train class: {majority_class}",
        "uses_features": False,
    },
    {
        "model_name": "nearest_centroid_lightweight_features",
        "model_type": "nearest_centroid",
        "description": "Class centroid classifier using train-standardized lightweight crop features.",
        "uses_features": True,
    },
]

prediction_rows = []
metrics_rows = []
per_class_rows = []
confusion_rows = []

models = ["majority_train_class", "nearest_centroid_lightweight_features"]

for split_name, df_split in split_data.items():
    X = df_split[feature_cols].to_numpy(dtype=float)
    y = df_split[label_column].to_numpy()

    preds_by_model = {
        "majority_train_class": np.array([majority_class] * len(df_split)),
        "nearest_centroid_lightweight_features": nearest_centroid_predict(X, centroids, classes),
    }

    for model_name in models:
        pred = preds_by_model[model_name]
        m = compute_metrics(y, pred, classes)

        metrics_rows.append({
            "model_name": model_name,
            "split": split_name,
            "n_rows": int(len(df_split)),
            "accuracy": round(float(m["accuracy"]), 4),
            "macro_precision": round(float(m["macro_precision"]), 4),
            "macro_recall": round(float(m["macro_recall"]), 4),
            "macro_f1": round(float(m["macro_f1"]), 4),
            "weighted_f1": round(float(m["weighted_f1"]), 4),
        })

        for pc in m["per_class"]:
            row = {
                "model_name": model_name,
                "split": split_name,
                "class": pc["class"],
                "support": pc["support"],
                "tp": pc["tp"],
                "fp": pc["fp"],
                "fn": pc["fn"],
                "tn": pc["tn"],
                "precision": round(float(pc["precision"]), 4),
                "recall": round(float(pc["recall"]), 4),
                "f1": round(float(pc["f1"]), 4),
            }
            per_class_rows.append(row)

        confusion_rows.extend(confusion_matrix_rows(y, pred, classes, model_name, split_name))

        for i, (_, r) in enumerate(df_split.iterrows()):
            prediction_rows.append({
                "model_name": model_name,
                "split": split_name,
                "v36b_feature_row_id": clean(r.get("v36b_feature_row_id", "")),
                "scan_frame_id": clean(r.get("scan_frame_id", "")),
                "true_label": clean(r.get(label_column, "")),
                "predicted_label": clean(pred[i]),
                "correct": bool(clean(r.get(label_column, "")) == clean(pred[i])),
                "resolved_crop_path": clean(r.get("resolved_crop_path", "")),
            })

predictions = pd.DataFrame(prediction_rows)
metrics = pd.DataFrame(metrics_rows)
per_class = pd.DataFrame(per_class_rows)
confusion = pd.DataFrame(confusion_rows)
model_summary = pd.DataFrame(model_summaries)

safe_to_csv(predictions, OUT_PREDICTIONS)
safe_to_csv(metrics, OUT_METRICS)
safe_to_csv(per_class, OUT_PER_CLASS)
safe_to_csv(confusion, OUT_CONFUSION)
safe_to_csv(model_summary, OUT_MODEL_SUMMARY)

# Data audit.
audit_rows = []

for split_name, df_split in split_data.items():
    row = {
        "split": split_name,
        "rows": int(len(df_split)),
        "classes_present": " | ".join(sorted(df_split[label_column].unique().tolist())),
    }
    for c in classes:
        row[f"count__{c}"] = int((df_split[label_column] == c).sum())
    audit_rows.append(row)

data_audit = pd.DataFrame(audit_rows)
safe_to_csv(data_audit, OUT_DATA_AUDIT)

# Decision logic:
# This is a dry-run, so pass is based on pipeline validity, not metric quality.
test_nearest = metrics[
    (metrics["model_name"] == "nearest_centroid_lightweight_features")
    & (metrics["split"] == "test")
]

test_macro_f1 = float(test_nearest["macro_f1"].iloc[0]) if len(test_nearest) else 0.0
test_accuracy = float(test_nearest["accuracy"].iloc[0]) if len(test_nearest) else 0.0

warnings = []

if test_macro_f1 < 0.25:
    warnings.append({
        "item": "test_macro_f1",
        "issue_type": "warning_low_test_macro_f1",
        "issue_detail": f"Nearest centroid test macro-F1 is low: {test_macro_f1:.4f}. This is expected to be interpreted only as baseline sanity check.",
    })

if len(test) < 25:
    warnings.append({
        "item": "test_split_size",
        "issue_type": "warning_small_test_split",
        "issue_detail": f"Test split has only {len(test)} rows. Metrics are unstable.",
    })

hard_issues = []

if len(predictions) == 0:
    hard_issues.append({
        "item": "predictions",
        "issue_type": "hard_no_predictions_created",
        "issue_detail": "No predictions were created.",
    })

if len(metrics) == 0:
    hard_issues.append({
        "item": "metrics",
        "issue_type": "hard_no_metrics_created",
        "issue_detail": "No metrics were created.",
    })

all_issues = hard_issues + warnings
issues_df = pd.DataFrame(all_issues, columns=["item", "issue_type", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)

ready_for_v38 = len(hard_issues) == 0

v38_config = {
    "stage": "v37_crop_baseline_classifier_dryrun",
    "baseline_result_scope": "sanity_check_only_not_final_behaviour_classifier",
    "predictions": str(OUT_PREDICTIONS),
    "metrics": str(OUT_METRICS),
    "per_class_metrics": str(OUT_PER_CLASS),
    "confusion_matrices": str(OUT_CONFUSION),
    "recommended_next_stage": "clip_level_feature_preparation_or_temporal_representation",
    "reason": "Crop-only lightweight features are insufficient for final behaviour modelling; next step should use clip-level temporal context.",
    "ready_for_v38_clip_feature_preparation": bool(ready_for_v38),
}

OUT_V38_CONFIG.write_text(json.dumps(v38_config, indent=2))

decision = pd.DataFrame([{
    "v37_decision": "crop_baseline_classifier_dryrun_completed" if ready_for_v38 else "crop_baseline_classifier_dryrun_issues_found",
    "feature_rows_used": int(len(data)),
    "train_rows": int(len(train)),
    "val_rows": int(len(val)),
    "test_rows": int(len(test)),
    "class_count": int(len(classes)),
    "classes": " | ".join(classes),
    "models_evaluated": " | ".join(models),
    "nearest_centroid_test_accuracy": round(test_accuracy, 4),
    "nearest_centroid_test_macro_f1": round(test_macro_f1, 4),
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "issue_count": int(len(issues_df)),
    "claim_scope": "baseline_sanity_check_only_not_final_behaviour_classifier",
    "ready_for_v38_clip_feature_preparation": bool(ready_for_v38),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

metrics_table = md_table(
    metrics.to_dict("records"),
    ["model_name", "split", "n_rows", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"],
)

audit_table = md_table(
    data_audit.to_dict("records"),
    ["split", "rows", "classes_present"] + [f"count__{c}" for c in classes],
)

report = f"""# Week 7 v37 Crop Baseline Classifier Dry-Run Report

## Purpose

This step runs a minimal classifier dry-run using the v36b standardized lightweight crop feature matrix.

This is not a final behaviour classifier. It is a baseline sanity check to verify that the feature matrix, labels, splits, predictions and metrics pipeline works end-to-end.

## Inputs

- Standardized feature matrix: `{feature_matrix_path}`
- Metadata: `{metadata_path}`
- Config: `{CONFIG_IN}`

## Dataset audit

{audit_table}

## Models evaluated

1. Majority train-class dummy baseline.
2. Nearest-centroid classifier using lightweight crop features.

## Metrics

{metrics_table}

## Interpretation

The crop-only baseline is expected to be limited because many behaviours require temporal context, posture change, motion and interaction information. Therefore the result should not be reported as a final model performance claim.

The useful result of v37 is that the baseline experiment pipeline works and produces predictions, metrics and confusion matrices.

## Recommended next step

Proceed to v38: clip-level feature preparation or temporal representation. This is more appropriate for behaviour modelling because the annotations and behaviours are temporal and multi-pig.
"""

OUT_REPORT.write_text(report)

OUT_NOTE.write_text(
    "# Week 7 v37 Crop Baseline Classifier Dry-Run\n\n"
    "## Summary\n\n"
    f"- Feature rows used: `{len(data)}`\n"
    f"- Train rows: `{len(train)}`\n"
    f"- Val rows: `{len(val)}`\n"
    f"- Test rows: `{len(test)}`\n"
    f"- Classes: `{', '.join(classes)}`\n"
    f"- Models evaluated: `{', '.join(models)}`\n"
    f"- Nearest centroid test accuracy: `{test_accuracy:.4f}`\n"
    f"- Nearest centroid test macro-F1: `{test_macro_f1:.4f}`\n"
    f"- Hard issue count: `{len(hard_issues)}`\n"
    f"- Warning count: `{len(warnings)}`\n"
    f"- Ready for v38 clip feature preparation: `{ready_for_v38}`\n\n"
    "## Important interpretation\n\n"
    "This is a crop-only lightweight baseline sanity check, not a final behaviour classifier.\n\n"
    "## Outputs\n\n"
    f"- Predictions: `{OUT_PREDICTIONS}`\n"
    f"- Metrics: `{OUT_METRICS}`\n"
    f"- Per-class metrics: `{OUT_PER_CLASS}`\n"
    f"- Confusion matrices: `{OUT_CONFUSION}`\n"
    f"- Model summary: `{OUT_MODEL_SUMMARY}`\n"
    f"- Data audit: `{OUT_DATA_AUDIT}`\n"
    f"- Report: `{OUT_REPORT}`\n"
    f"- Decision: `{OUT_DECISION}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
    f"- v38 config: `{OUT_V38_CONFIG}`\n"
)

print("Saved:")
print(OUT_PREDICTIONS)
print(OUT_METRICS)
print(OUT_PER_CLASS)
print(OUT_CONFUSION)
print(OUT_MODEL_SUMMARY)
print(OUT_DATA_AUDIT)
print(OUT_REPORT)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_V38_CONFIG)
print(OUT_NOTE)

print()
print("=== v37 decision ===")
print(decision.to_string(index=False))

print()
print("=== v37 issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
