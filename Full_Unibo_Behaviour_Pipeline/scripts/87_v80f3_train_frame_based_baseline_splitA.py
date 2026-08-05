from pathlib import Path
from datetime import datetime
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


F = Path.home() / "PigBench" / "Full_Unibo_Behaviour_Pipeline"
O = F / "outputs/v80_final_project_completion/04_frame_based_baseline"
O.mkdir(parents=True, exist_ok=True)

data_path = O / "v80f2_split_A_frame_proxy_dataset.csv"
df = pd.read_csv(data_path).fillna("")

# This is a frame-proxy baseline: it uses ROI-filtered candidate bbox/track features,
# not raw RGB image pixels.
numeric_features = [
    "score",
    "bbox_cx_norm",
    "bbox_cy_norm",
    "bbox_w_norm",
    "bbox_h_norm",
    "bbox_area_norm",
    "bbox_aspect",
    "time_norm_in_clip",
    "candidate_rank_proxy",
]

categorical_features = [
    "tlc_camera",
    "room_pen",
    "identity_colour",
]

target_col = "behaviour_label"

for c in numeric_features:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=numeric_features + [target_col]).copy()

train = df[df["split"] == "train"].copy()
val = df[df["split"] == "val"].copy()
test = df[df["split"] == "test"].copy()

classes = sorted(train[target_col].astype(str).unique().tolist())

X_train = train[numeric_features + categorical_features]
y_train = train[target_col].astype(str)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

model = LogisticRegression(
    max_iter=500,
    class_weight="balanced",
    multi_class="auto",
    n_jobs=4,
)

pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", model),
    ]
)

pipe.fit(X_train, y_train)

joblib.dump(pipe, O / "v80f3_splitA_frame_proxy_logistic_baseline.joblib")

def frame_predictions(split_df, split_name):
    X = split_df[numeric_features + categorical_features]
    y_true = split_df[target_col].astype(str).values
    y_pred = pipe.predict(X)
    proba = pipe.predict_proba(X)
    class_names = pipe.classes_

    pred = split_df[
        [
            "split",
            "clip_id",
            "video_id",
            "video_filename",
            "tlc_camera",
            "room_pen",
            "identity_colour",
            "behaviour_label",
            "time_sec",
            "frame_index",
            "candidate_tracklet_id",
            "candidate_rank_proxy",
        ]
    ].copy()

    pred["frame_true_label"] = y_true
    pred["frame_pred_label"] = y_pred
    pred["frame_pred_confidence"] = proba.max(axis=1)

    for i, cls in enumerate(class_names):
        pred["proba_" + str(cls)] = proba[:, i]

    pred.to_csv(O / f"v80f3_splitA_{split_name}_frame_predictions.csv", index=False)

    report = classification_report(
        y_true,
        y_pred,
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "class"})
    per_class.to_csv(O / f"v80f3_splitA_{split_name}_frame_per_class_report.csv", index=False)

    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=class_names),
        index=class_names,
        columns=class_names,
    )
    cm.to_csv(O / f"v80f3_splitA_{split_name}_frame_confusion_matrix.csv")

    summary = {
        "split": split_name,
        "level": "frame",
        "rows": len(split_df),
        "clips": split_df["clip_id"].nunique(),
        "videos": split_df["video_id"].nunique(),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    return pred, summary, class_names

def clip_majority_vote(frame_pred):
    rows = []

    for clip_id, g in frame_pred.groupby("clip_id"):
        true_label = str(g["frame_true_label"].iloc[0])
        counts = g["frame_pred_label"].value_counts()
        pred_label = str(counts.index[0])
        confidence = float(counts.iloc[0] / len(g))

        rows.append(
            {
                "clip_id": clip_id,
                "video_id": g["video_id"].iloc[0],
                "video_filename": g["video_filename"].iloc[0],
                "tlc_camera": g["tlc_camera"].iloc[0],
                "room_pen": g["room_pen"].iloc[0],
                "true_label": true_label,
                "pred_label": pred_label,
                "aggregation": "majority_vote",
                "clip_confidence": confidence,
                "frame_rows": len(g),
            }
        )

    return pd.DataFrame(rows)

def clip_average_probability(frame_pred, class_names):
    proba_cols = ["proba_" + str(c) for c in class_names]
    rows = []

    for clip_id, g in frame_pred.groupby("clip_id"):
        true_label = str(g["frame_true_label"].iloc[0])
        mean_proba = g[proba_cols].mean(axis=0)
        best_col = str(mean_proba.idxmax())
        pred_label = best_col.replace("proba_", "", 1)
        confidence = float(mean_proba.max())

        row = {
            "clip_id": clip_id,
            "video_id": g["video_id"].iloc[0],
            "video_filename": g["video_filename"].iloc[0],
            "tlc_camera": g["tlc_camera"].iloc[0],
            "room_pen": g["room_pen"].iloc[0],
            "true_label": true_label,
            "pred_label": pred_label,
            "aggregation": "average_probability",
            "clip_confidence": confidence,
            "frame_rows": len(g),
        }

        for c in class_names:
            row["avg_proba_" + str(c)] = float(mean_proba["proba_" + str(c)])

        rows.append(row)

    return pd.DataFrame(rows)

def evaluate_clip_predictions(clip_df, split_name, aggregation, class_names):
    y_true = clip_df["true_label"].astype(str).values
    y_pred = clip_df["pred_label"].astype(str).values

    clip_df.to_csv(O / f"v80f3_splitA_{split_name}_clip_predictions_{aggregation}.csv", index=False)

    report = classification_report(
        y_true,
        y_pred,
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )
    per_class = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "class"})
    per_class.to_csv(O / f"v80f3_splitA_{split_name}_clip_per_class_report_{aggregation}.csv", index=False)

    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=class_names),
        index=class_names,
        columns=class_names,
    )
    cm.to_csv(O / f"v80f3_splitA_{split_name}_clip_confusion_matrix_{aggregation}.csv")

    return {
        "split": split_name,
        "level": "clip",
        "aggregation": aggregation,
        "rows": len(clip_df),
        "videos": clip_df["video_id"].nunique(),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

all_summaries = []

for split_name, split_df in [("val", val), ("test", test)]:
    frame_pred, frame_summary, class_names = frame_predictions(split_df, split_name)
    all_summaries.append(frame_summary)

    maj = clip_majority_vote(frame_pred)
    avg = clip_average_probability(frame_pred, class_names)

    all_summaries.append(evaluate_clip_predictions(maj, split_name, "majority_vote", class_names))
    all_summaries.append(evaluate_clip_predictions(avg, split_name, "average_probability", class_names))

metrics = pd.DataFrame(all_summaries)
metrics.to_csv(O / "v80f3_splitA_frame_based_baseline_metrics_summary.csv", index=False)

# Train distribution is saved for reporting.
train_dist = (
    train.groupby("behaviour_label", dropna=False)
    .agg(frame_rows=("behaviour_label", "count"), clips=("clip_id", "nunique"), videos=("video_id", "nunique"))
    .reset_index()
)
train_dist.to_csv(O / "v80f3_splitA_train_distribution.csv", index=False)

issues = []

if len(train) == 0 or len(val) == 0 or len(test) == 0:
    issues.append(
        {
            "item": "split_data",
            "issue_type": "empty_train_val_or_test",
            "severity": "hard",
            "detail": f"train={len(train)}, val={len(val)}, test={len(test)}",
        }
    )

if metrics["macro_f1"].isna().any():
    issues.append(
        {
            "item": "metrics",
            "issue_type": "nan_macro_f1",
            "severity": "hard",
            "detail": "NaN macro F1 detected",
        }
    )

if not issues:
    issues = [
        {
            "item": "none",
            "issue_type": "none",
            "severity": "info",
            "detail": "frame-based baseline completed",
        }
    ]

pd.DataFrame(issues).to_csv(O / "v80f3_issues.csv", index=False)
hard = sum(1 for x in issues if x["severity"] == "hard")

best_test = metrics[(metrics["split"] == "test") & (metrics["level"] == "clip")].copy()
if len(best_test) > 0:
    best_test = best_test.sort_values(["macro_f1", "weighted_f1"], ascending=False).iloc[0]
    best_test_aggregation = str(best_test.get("aggregation", ""))
    best_test_macro_f1 = float(best_test["macro_f1"])
    best_test_accuracy = float(best_test["accuracy"])
else:
    best_test_aggregation = ""
    best_test_macro_f1 = 0.0
    best_test_accuracy = 0.0

decision = pd.DataFrame(
    [
        {
            "v80f3_decision": "frame_based_baseline_splitA_completed"
            if hard == 0
            else "frame_based_baseline_splitA_has_blocking_issues",
            "train_frame_rows": len(train),
            "val_frame_rows": len(val),
            "test_frame_rows": len(test),
            "train_clips": train["clip_id"].nunique(),
            "val_clips": val["clip_id"].nunique(),
            "test_clips": test["clip_id"].nunique(),
            "model_type": "logistic_regression_on_roi_bbox_frame_proxy_features",
            "best_test_clip_aggregation": best_test_aggregation,
            "best_test_clip_macro_f1": round(best_test_macro_f1, 6),
            "best_test_clip_accuracy": round(best_test_accuracy, 6),
            "hard_issue_count": hard,
            "ready_for_v80f4_splitB_generalization_eval": hard == 0,
            "claim_scope": "frame_proxy_baseline_not_rgb_cnn_not_production_classifier",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
    ]
)

decision.to_csv(O / "v80f3_decision_summary.csv", index=False)

note = F / "notes/v80f3_frame_based_baseline_splitA_notes.md"
note.write_text(
    "# v80f-3 Frame-based Baseline on Split A\n\n"
    f"- Decision: {decision.iloc[0]['v80f3_decision']}\n"
    f"- Model type: logistic regression on ROI bbox/frame-proxy features\n"
    f"- Train frame rows: {len(train)}\n"
    f"- Val frame rows: {len(val)}\n"
    f"- Test frame rows: {len(test)}\n"
    f"- Best test clip aggregation: {best_test_aggregation}\n"
    f"- Best test clip macro F1: {best_test_macro_f1:.6f}\n"
    f"- Best test clip accuracy: {best_test_accuracy:.6f}\n"
    f"- Hard issues: {hard}\n"
    f"- Ready for v80f-4 Split B generalization eval: {hard == 0}\n\n"
    "This is a frame-proxy baseline, not an RGB image CNN and not a production classifier. "
    "Clip-level results are produced using majority voting and average probability aggregation.\n",
    encoding="utf-8",
)

print(decision.to_string(index=False))
print("=== metrics ===")
print(metrics.to_string(index=False))
print("=== issues ===")
print(pd.DataFrame(issues).to_string(index=False))
