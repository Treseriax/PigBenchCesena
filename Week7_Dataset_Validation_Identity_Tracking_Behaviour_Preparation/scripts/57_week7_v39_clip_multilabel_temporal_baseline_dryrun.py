from pathlib import Path
from datetime import datetime
import csv
import json
import pandas as pd
import numpy as np


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V38 = W7 / "outputs" / "week7_clip_temporal_feature_preparation_v38"
CONFIG_IN = V38 / "week7_v38_recommended_v39_clip_temporal_baseline_config.json"

OUT_ROOT = W7 / "outputs" / "week7_clip_multilabel_temporal_baseline_dryrun_v39"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_DATA_AUDIT = OUT_ROOT / "week7_v39_clip_multilabel_data_audit.csv"
OUT_PREDICTIONS = OUT_ROOT / "week7_v39_clip_multilabel_predictions.csv"
OUT_METRICS = OUT_ROOT / "week7_v39_clip_multilabel_metrics.csv"
OUT_PER_LABEL = OUT_ROOT / "week7_v39_clip_multilabel_per_label_metrics.csv"
OUT_LABEL_CONFUSION = OUT_ROOT / "week7_v39_clip_multilabel_label_confusion.csv"
OUT_MODEL_SUMMARY = OUT_ROOT / "week7_v39_clip_multilabel_model_summary.csv"
OUT_REPORT = OUT_ROOT / "week7_v39_clip_multilabel_temporal_baseline_report.md"
OUT_DECISION = OUT_ROOT / "week7_v39_clip_multilabel_temporal_baseline_decision_summary.csv"
OUT_ISSUES = OUT_ROOT / "week7_v39_clip_multilabel_temporal_baseline_issues.csv"
OUT_V40_CONFIG = OUT_ROOT / "week7_v39_recommended_v40_behaviour_temporal_evidence_package_config.json"
OUT_NOTE = W7 / "notes" / "week7_v39_clip_multilabel_temporal_baseline_notes.md"


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


def read_df(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


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


def labels_to_set(row, label_cols):
    labs = []
    for c in label_cols:
        if int(row.get(c, 0)) == 1:
            labs.append(c.replace("label__", ""))
    return " | ".join(labs)


def pred_to_set(pred_row, label_names):
    labs = []
    for value, label in zip(pred_row, label_names):
        if int(value) == 1:
            labs.append(label)
    return " | ".join(labs)


def fit_label_prior(Y_train):
    prevalence = Y_train.mean(axis=0)
    pred = (prevalence >= 0.5).astype(int)
    return prevalence, pred


def predict_label_prior(n_rows, prior_pred):
    return np.tile(prior_pred.reshape(1, -1), (n_rows, 1))


def fit_centroid_ovr(X_train, Y_train):
    models = []

    for j in range(Y_train.shape[1]):
        y = Y_train[:, j]
        pos = X_train[y == 1]
        neg = X_train[y == 0]

        entry = {
            "label_index": j,
            "pos_count": int(len(pos)),
            "neg_count": int(len(neg)),
            "has_pos_centroid": bool(len(pos) > 0),
            "has_neg_centroid": bool(len(neg) > 0),
            "pos_centroid": pos.mean(axis=0) if len(pos) else None,
            "neg_centroid": neg.mean(axis=0) if len(neg) else None,
        }

        models.append(entry)

    return models


def predict_centroid_ovr(X, models):
    preds = np.zeros((X.shape[0], len(models)), dtype=int)

    for j, m in enumerate(models):
        pos = m["pos_centroid"]
        neg = m["neg_centroid"]

        if pos is None and neg is None:
            preds[:, j] = 0
        elif pos is None:
            preds[:, j] = 0
        elif neg is None:
            preds[:, j] = 1
        else:
            d_pos = np.linalg.norm(X - pos.reshape(1, -1), axis=1)
            d_neg = np.linalg.norm(X - neg.reshape(1, -1), axis=1)
            preds[:, j] = (d_pos <= d_neg).astype(int)

    return preds


def multilabel_metrics(Y_true, Y_pred, label_names):
    Y_true = Y_true.astype(int)
    Y_pred = Y_pred.astype(int)

    n_samples = Y_true.shape[0]
    n_labels = Y_true.shape[1]

    tp = ((Y_true == 1) & (Y_pred == 1)).sum(axis=0)
    fp = ((Y_true == 0) & (Y_pred == 1)).sum(axis=0)
    fn = ((Y_true == 1) & (Y_pred == 0)).sum(axis=0)
    tn = ((Y_true == 0) & (Y_pred == 0)).sum(axis=0)

    per_label = []

    for i, label in enumerate(label_names):
        precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) else 0.0
        recall = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        per_label.append({
            "label": label,
            "support": int(Y_true[:, i].sum()),
            "predicted_positive": int(Y_pred[:, i].sum()),
            "tp": int(tp[i]),
            "fp": int(fp[i]),
            "fn": int(fn[i]),
            "tn": int(tn[i]),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        })

    micro_tp = int(tp.sum())
    micro_fp = int(fp.sum())
    micro_fn = int(fn.sum())

    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0.0

    macro_precision = float(np.mean([x["precision"] for x in per_label])) if per_label else 0.0
    macro_recall = float(np.mean([x["recall"] for x in per_label])) if per_label else 0.0
    macro_f1 = float(np.mean([x["f1"] for x in per_label])) if per_label else 0.0

    subset_accuracy = float(np.mean(np.all(Y_true == Y_pred, axis=1))) if n_samples else 0.0
    hamming_loss = float(np.mean(Y_true != Y_pred)) if n_samples and n_labels else 0.0

    true_cardinality = float(Y_true.sum(axis=1).mean()) if n_samples else 0.0
    pred_cardinality = float(Y_pred.sum(axis=1).mean()) if n_samples else 0.0

    sample_f1s = []
    for yt, yp in zip(Y_true, Y_pred):
        inter = int(((yt == 1) & (yp == 1)).sum())
        denom = int(yt.sum() + yp.sum())
        if denom == 0:
            sample_f1s.append(1.0)
        else:
            sample_f1s.append(2 * inter / denom)

    sample_f1 = float(np.mean(sample_f1s)) if sample_f1s else 0.0

    return {
        "subset_accuracy": subset_accuracy,
        "hamming_loss": hamming_loss,
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "sample_f1": float(sample_f1),
        "true_label_cardinality": true_cardinality,
        "predicted_label_cardinality": pred_cardinality,
        "per_label": per_label,
    }


issues = []

config = read_json(CONFIG_IN)

if not config:
    issues.append({
        "item": "v38_v39_config",
        "issue_type": "hard_missing_or_unreadable_config",
        "issue_detail": str(CONFIG_IN),
    })

feature_matrix_path = Path(config.get("feature_matrix_standardized", "")) if config else Path("")
targets_path = Path(config.get("targets", "")) if config else Path("")
clip_features_path = Path(config.get("clip_features", "")) if config else Path("")
feature_cols = config.get("feature_columns", []) if config else []
label_cols = config.get("label_columns", []) if config else []
id_col = config.get("id_column", "v38_clip_row_id") if config else "v38_clip_row_id"
split_col = config.get("split_column", "split") if config else "split"

X_df = read_df(feature_matrix_path)
targets = read_df(targets_path)
clip_features = read_df(clip_features_path)

if len(X_df) == 0:
    issues.append({
        "item": "feature_matrix_standardized",
        "issue_type": "hard_missing_or_empty_input",
        "issue_detail": str(feature_matrix_path),
    })

if len(targets) == 0:
    issues.append({
        "item": "targets",
        "issue_type": "hard_missing_or_empty_input",
        "issue_detail": str(targets_path),
    })

for df_name, df in [("feature_matrix", X_df), ("targets", targets)]:
    if id_col not in df.columns:
        issues.append({
            "item": f"{df_name}.{id_col}",
            "issue_type": "hard_missing_id_column",
            "issue_detail": f"{id_col} missing from {df_name}.",
        })

for c in feature_cols:
    if c not in X_df.columns:
        issues.append({
            "item": c,
            "issue_type": "hard_missing_feature_column",
            "issue_detail": f"Feature column `{c}` missing.",
        })

for c in label_cols:
    if c not in targets.columns:
        issues.append({
            "item": c,
            "issue_type": "hard_missing_label_column",
            "issue_detail": f"Label column `{c}` missing.",
        })

if split_col not in targets.columns:
    issues.append({
        "item": split_col,
        "issue_type": "hard_missing_split_column",
        "issue_detail": f"Split column `{split_col}` missing from targets.",
    })

if len(issues):
    issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
    safe_to_csv(issues_df, OUT_ISSUES)
    print("Hard input issues:")
    print(issues_df.to_string(index=False))
    raise SystemExit(1)

data = targets.merge(X_df, on=id_col, how="inner")
data[split_col] = data[split_col].fillna("").astype(str).str.strip()

for c in label_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce").fillna(0).astype(int)

for c in feature_cols:
    data[c] = pd.to_numeric(data[c], errors="coerce")

model_ready = data[data[split_col].isin(["train", "val", "test"])].copy()
model_ready = model_ready.reset_index(drop=True)

nan_feature_cells = int(model_ready[feature_cols].isna().sum().sum())

if nan_feature_cells > 0:
    issues.append({
        "item": "feature_matrix",
        "issue_type": "hard_nan_feature_values",
        "issue_detail": f"{nan_feature_cells} NaN cells found.",
    })

train = model_ready[model_ready[split_col] == "train"].copy()
val = model_ready[model_ready[split_col] == "val"].copy()
test = model_ready[model_ready[split_col] == "test"].copy()

for split_name, df_split in [("train", train), ("val", val), ("test", test)]:
    if len(df_split) == 0:
        issues.append({
            "item": split_name,
            "issue_type": "hard_empty_split",
            "issue_detail": f"{split_name} split is empty.",
        })

if len([x for x in issues if str(x["issue_type"]).startswith("hard_")]):
    issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
    safe_to_csv(issues_df, OUT_ISSUES)
    print("Hard data issues:")
    print(issues_df.to_string(index=False))
    raise SystemExit(1)

label_names = [c.replace("label__", "") for c in label_cols]

X_train = train[feature_cols].to_numpy(dtype=float)
Y_train = train[label_cols].to_numpy(dtype=int)

prior_prevalence, prior_pred = fit_label_prior(Y_train)
centroid_models = fit_centroid_ovr(X_train, Y_train)

model_rows = [
    {
        "model_name": "label_prior_majority_multilabel",
        "model_type": "dummy_multilabel",
        "description": "Predicts each label as positive if its train prevalence is >= 0.5.",
        "uses_features": False,
    },
    {
        "model_name": "one_vs_rest_nearest_centroid_multilabel",
        "model_type": "one_vs_rest_nearest_centroid",
        "description": "For each label, compares distance to positive and negative train centroids using standardized temporal features.",
        "uses_features": True,
    },
]

model_summary = pd.DataFrame(model_rows)
safe_to_csv(model_summary, OUT_MODEL_SUMMARY)

prediction_rows = []
metric_rows = []
per_label_rows = []
confusion_rows = []

split_map = {
    "train": train,
    "val": val,
    "test": test,
}

model_names = [
    "label_prior_majority_multilabel",
    "one_vs_rest_nearest_centroid_multilabel",
]

for split_name, df_split in split_map.items():
    X = df_split[feature_cols].to_numpy(dtype=float)
    Y = df_split[label_cols].to_numpy(dtype=int)

    preds = {
        "label_prior_majority_multilabel": predict_label_prior(len(df_split), prior_pred),
        "one_vs_rest_nearest_centroid_multilabel": predict_centroid_ovr(X, centroid_models),
    }

    for model_name in model_names:
        P = preds[model_name].astype(int)
        m = multilabel_metrics(Y, P, label_names)

        metric_rows.append({
            "model_name": model_name,
            "split": split_name,
            "n_clips": int(len(df_split)),
            "subset_accuracy": round(m["subset_accuracy"], 4),
            "hamming_loss": round(m["hamming_loss"], 4),
            "micro_precision": round(m["micro_precision"], 4),
            "micro_recall": round(m["micro_recall"], 4),
            "micro_f1": round(m["micro_f1"], 4),
            "macro_precision": round(m["macro_precision"], 4),
            "macro_recall": round(m["macro_recall"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "sample_f1": round(m["sample_f1"], 4),
            "true_label_cardinality": round(m["true_label_cardinality"], 4),
            "predicted_label_cardinality": round(m["predicted_label_cardinality"], 4),
        })

        for label_metric in m["per_label"]:
            row = dict(label_metric)
            row["model_name"] = model_name
            row["split"] = split_name
            row["precision"] = round(float(row["precision"]), 4)
            row["recall"] = round(float(row["recall"]), 4)
            row["f1"] = round(float(row["f1"]), 4)
            per_label_rows.append(row)

            confusion_rows.append({
                "model_name": model_name,
                "split": split_name,
                "label": label_metric["label"],
                "tp": label_metric["tp"],
                "fp": label_metric["fp"],
                "fn": label_metric["fn"],
                "tn": label_metric["tn"],
            })

        for i, (_, r) in enumerate(df_split.iterrows()):
            pred_row = P[i, :]
            true_set = labels_to_set(r, label_cols)
            pred_set = pred_to_set(pred_row, label_names)

            out = {
                "model_name": model_name,
                "split": split_name,
                "v38_clip_row_id": clean(r.get(id_col, "")),
                "scan_frame_id": clean(r.get("scan_frame_id", "")),
                "true_label_set": true_set,
                "predicted_label_set": pred_set,
                "exact_match": bool(true_set == pred_set),
            }

            for c in label_cols:
                label = c.replace("label__", "")
                out[f"true__{label}"] = int(r.get(c, 0))

            for j, label in enumerate(label_names):
                out[f"pred__{label}"] = int(pred_row[j])

            prediction_rows.append(out)

predictions = pd.DataFrame(prediction_rows)
metrics = pd.DataFrame(metric_rows)
per_label = pd.DataFrame(per_label_rows)
confusion = pd.DataFrame(confusion_rows)

safe_to_csv(predictions, OUT_PREDICTIONS)
safe_to_csv(metrics, OUT_METRICS)
safe_to_csv(per_label, OUT_PER_LABEL)
safe_to_csv(confusion, OUT_LABEL_CONFUSION)

# Data audit.
audit_rows = []

for split_name, df_split in split_map.items():
    row = {
        "split": split_name,
        "clip_rows": int(len(df_split)),
        "mean_label_cardinality": round(float(df_split[label_cols].sum(axis=1).mean()), 4),
    }

    for c in label_cols:
        row[c] = int(df_split[c].sum())

    audit_rows.append(row)

data_audit = pd.DataFrame(audit_rows)
safe_to_csv(data_audit, OUT_DATA_AUDIT)

# Decision logic.
test_centroid = metrics[
    (metrics["model_name"] == "one_vs_rest_nearest_centroid_multilabel")
    & (metrics["split"] == "test")
]

test_micro_f1 = float(test_centroid["micro_f1"].iloc[0]) if len(test_centroid) else 0.0
test_macro_f1 = float(test_centroid["macro_f1"].iloc[0]) if len(test_centroid) else 0.0
test_sample_f1 = float(test_centroid["sample_f1"].iloc[0]) if len(test_centroid) else 0.0
test_hamming_loss = float(test_centroid["hamming_loss"].iloc[0]) if len(test_centroid) else 1.0

warnings = []

if len(test) < 15:
    warnings.append({
        "item": "test_clip_count",
        "issue_type": "warning_small_test_clip_count",
        "issue_detail": f"Test split has only {len(test)} clips. Metrics are unstable.",
    })

rare_train_labels = []

for c in label_cols:
    train_pos = int(train[c].sum())
    if train_pos < 5:
        rare_train_labels.append(c.replace("label__", ""))

if rare_train_labels:
    warnings.append({
        "item": "rare_train_labels",
        "issue_type": "warning_rare_labels_in_train",
        "issue_detail": "Labels with fewer than 5 train positives: " + " | ".join(rare_train_labels),
    })

if test_macro_f1 < 0.25:
    warnings.append({
        "item": "test_macro_f1",
        "issue_type": "warning_low_test_macro_f1",
        "issue_detail": f"Nearest-centroid multi-label test macro-F1 is low: {test_macro_f1:.4f}. Interpret as baseline sanity check only.",
    })

hard_issues = []

if len(predictions) == 0:
    hard_issues.append({
        "item": "predictions",
        "issue_type": "hard_no_predictions_created",
        "issue_detail": "No prediction rows were created.",
    })

if len(metrics) == 0:
    hard_issues.append({
        "item": "metrics",
        "issue_type": "hard_no_metrics_created",
        "issue_detail": "No metric rows were created.",
    })

all_issues = hard_issues + warnings
issues_df = pd.DataFrame(all_issues, columns=["item", "issue_type", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)

ready_for_v40 = len(hard_issues) == 0

v40_config = {
    "stage": "v39_clip_multilabel_temporal_baseline_dryrun",
    "claim_scope": "clip-level multi-label temporal baseline sanity check only; not final behaviour classifier",
    "data_audit": str(OUT_DATA_AUDIT),
    "predictions": str(OUT_PREDICTIONS),
    "metrics": str(OUT_METRICS),
    "per_label_metrics": str(OUT_PER_LABEL),
    "label_confusion": str(OUT_LABEL_CONFUSION),
    "report": str(OUT_REPORT),
    "recommended_next_stage": "behaviour_temporal_evidence_package_and_report_ready_summary",
    "reason": "Package crop-only baseline evidence, clip-level temporal baseline evidence, limitations, and next-step recommendations.",
    "ready_for_v40_evidence_package": bool(ready_for_v40),
}

OUT_V40_CONFIG.write_text(json.dumps(v40_config, indent=2))

decision = pd.DataFrame([{
    "v39_decision": "clip_multilabel_temporal_baseline_dryrun_completed" if ready_for_v40 else "clip_multilabel_temporal_baseline_dryrun_issues_found",
    "model_ready_clip_rows": int(len(model_ready)),
    "train_clips": int(len(train)),
    "val_clips": int(len(val)),
    "test_clips": int(len(test)),
    "label_count": int(len(label_cols)),
    "feature_count": int(len(feature_cols)),
    "models_evaluated": " | ".join(model_names),
    "centroid_test_micro_f1": round(test_micro_f1, 4),
    "centroid_test_macro_f1": round(test_macro_f1, 4),
    "centroid_test_sample_f1": round(test_sample_f1, 4),
    "centroid_test_hamming_loss": round(test_hamming_loss, 4),
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "issue_count": int(len(issues_df)),
    "claim_scope": "clip_multilabel_temporal_baseline_sanity_check_only_not_final_behaviour_classifier",
    "ready_for_v40_evidence_package": bool(ready_for_v40),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

metrics_table = md_table(
    metrics.to_dict("records"),
    [
        "model_name",
        "split",
        "n_clips",
        "subset_accuracy",
        "hamming_loss",
        "micro_f1",
        "macro_f1",
        "sample_f1",
        "true_label_cardinality",
        "predicted_label_cardinality",
    ],
)

audit_table = md_table(
    data_audit.to_dict("records"),
    ["split", "clip_rows", "mean_label_cardinality"] + label_cols,
)

report = f"""# Week 7 v39 Clip-Level Multi-Label Temporal Baseline Dry-Run Report

## Purpose

This step runs a minimal multi-label baseline dry-run using the v38 clip-level temporal feature matrix.

This is not a final behaviour classifier. It is a sanity check to verify that clip-level temporal features, multi-label targets, train/val/test splits, predictions and metrics work end-to-end.

## Inputs

- Standardized clip feature matrix: `{feature_matrix_path}`
- Multi-label targets: `{targets_path}`
- Config: `{CONFIG_IN}`

## Dataset audit

{audit_table}

## Models evaluated

1. Label-prior majority multi-label baseline.
2. One-vs-rest nearest-centroid multi-label baseline using standardized temporal features.

## Metrics

{metrics_table}

## Interpretation

The result should be interpreted only as a lightweight temporal baseline sanity check. The main value is that the clip-level multi-label pipeline now works and produces auditable predictions, per-label metrics and label-level confusion counts.

The dataset remains small and label-imbalanced, so test metrics are unstable. Rare labels should not be overclaimed.

## Recommended next step

Proceed to v40: create a behaviour temporal evidence package that summarizes crop-only baseline evidence, clip-level temporal baseline evidence, limitations and next-step recommendations.
"""

OUT_REPORT.write_text(report)

OUT_NOTE.write_text(
    "# Week 7 v39 Clip Multi-Label Temporal Baseline Dry-Run\n\n"
    "## Summary\n\n"
    f"- Model-ready clip rows: `{len(model_ready)}`\n"
    f"- Train clips: `{len(train)}`\n"
    f"- Val clips: `{len(val)}`\n"
    f"- Test clips: `{len(test)}`\n"
    f"- Label count: `{len(label_cols)}`\n"
    f"- Feature count: `{len(feature_cols)}`\n"
    f"- Models evaluated: `{', '.join(model_names)}`\n"
    f"- Centroid test micro-F1: `{test_micro_f1:.4f}`\n"
    f"- Centroid test macro-F1: `{test_macro_f1:.4f}`\n"
    f"- Centroid test sample-F1: `{test_sample_f1:.4f}`\n"
    f"- Centroid test hamming loss: `{test_hamming_loss:.4f}`\n"
    f"- Hard issue count: `{len(hard_issues)}`\n"
    f"- Warning count: `{len(warnings)}`\n"
    f"- Ready for v40 evidence package: `{ready_for_v40}`\n\n"
    "## Important interpretation\n\n"
    "This is a clip-level multi-label temporal baseline sanity check, not a final behaviour classifier.\n\n"
    "## Outputs\n\n"
    f"- Data audit: `{OUT_DATA_AUDIT}`\n"
    f"- Predictions: `{OUT_PREDICTIONS}`\n"
    f"- Metrics: `{OUT_METRICS}`\n"
    f"- Per-label metrics: `{OUT_PER_LABEL}`\n"
    f"- Label confusion: `{OUT_LABEL_CONFUSION}`\n"
    f"- Model summary: `{OUT_MODEL_SUMMARY}`\n"
    f"- Report: `{OUT_REPORT}`\n"
    f"- Decision: `{OUT_DECISION}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
    f"- v40 config: `{OUT_V40_CONFIG}`\n"
)

print("Saved:")
print(OUT_DATA_AUDIT)
print(OUT_PREDICTIONS)
print(OUT_METRICS)
print(OUT_PER_LABEL)
print(OUT_LABEL_CONFUSION)
print(OUT_MODEL_SUMMARY)
print(OUT_REPORT)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_V40_CONFIG)
print(OUT_NOTE)

print()
print("=== v39 decision ===")
print(decision.to_string(index=False))

print()
print("=== v39 issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
