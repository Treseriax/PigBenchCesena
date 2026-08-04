from pathlib import Path
from datetime import datetime
import csv
import json
import hashlib
import zipfile
from collections import Counter
import numpy as np
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V69B = W8 / "outputs" / "v69b_feature_baseline"
PKG69B = V69B / "Week8_Feature_Baseline"

FEATURES = PKG69B / "week8_v69b_feature_vectors.csv"
BEST = PKG69B / "week8_v69b_best_model_summary.csv"
V69B_DECISION = V69B / "week8_v69b_decision_summary.csv"

OUT = W8 / "outputs" / "v69c_error_analysis"
PKG = OUT / "Week8_Feature_Baseline_Error_Analysis"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_PRED = PKG / "week8_v69c_best_model_predictions.csv"
OUT_ERRORS = PKG / "week8_v69c_best_model_errors.csv"
OUT_CORRECT = PKG / "week8_v69c_best_model_correct_predictions.csv"
OUT_CONF = PKG / "week8_v69c_top_confusions.csv"
OUT_CLASS = PKG / "week8_v69c_per_class_error_summary.csv"
OUT_SCAN = PKG / "week8_v69c_scanframe_error_summary.csv"
OUT_VIDEO = PKG / "week8_v69c_video_error_summary.csv"
OUT_QA = PKG / "week8_v69c_error_analysis_quality_checks.csv"
OUT_README = PKG / "README_Week8_Feature_Baseline_Error_Analysis.md"
OUT_MANIFEST = PKG / "week8_v69c_error_analysis_manifest.json"

OUT_DECISION = OUT / "week8_v69c_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v69c_issues.csv"
OUT_ZIP = OUT / "Week8_Feature_Baseline_Error_Analysis.zip"
OUT_SHA256 = OUT / "Week8_Feature_Baseline_Error_Analysis.sha256"
OUT_NOTE = NOTES / "week8_v69c_error_analysis_notes.md"
OUT_REPORT = REPORTS / "week8_v69c_error_analysis_report.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


def read_csv_clean(path):
    df = pd.read_csv(path).fillna("")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(clean)
    return df


def bool_true(x):
    return str(x).strip().lower() == "true"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def standardize_train(X_train, X_eval):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return (X_train - mu) / sd, (X_eval - mu) / sd


def pred_centroid(X_train, y_train, X_eval, classes):
    centroids = []
    for c in classes:
        centroids.append(X_train[y_train == c].mean(axis=0))
    centroids = np.vstack(centroids)
    d = ((X_eval[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    pred_idx = d.argmin(axis=1)
    conf = 1.0 / (1.0 + d.min(axis=1))
    return np.array([classes[i] for i in pred_idx]), conf


def pred_knn3(X_train, y_train, X_eval, classes):
    preds = []
    confs = []
    for x in X_eval:
        d = ((X_train - x) ** 2).sum(axis=1)
        idx = np.argsort(d)[:3]
        votes = Counter(y_train[idx])
        best = sorted(votes.items(), key=lambda kv: (-kv[1], classes.index(kv[0])))[0][0]
        preds.append(best)
        confs.append(votes[best] / 3.0)
    return np.array(preds), np.array(confs)


issues = []

for p in [FEATURES, BEST, V69B_DECISION]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required v69b input missing.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v69c_decision": "error_analysis_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v70a_improved_baseline": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


v69b_decision = read_csv_clean(V69B_DECISION)
if len(v69b_decision) == 0 or not bool_true(v69b_decision.iloc[0].get("ready_for_v69c_error_analysis", "")):
    issues.append({
        "item": str(V69B_DECISION),
        "issue_type": "hard_v69b_not_ready",
        "issue_detail": "v69b must be ready before v69c.",
        "severity": "hard",
    })

features = read_csv_clean(FEATURES)
best = read_csv_clean(BEST)

feature_cols = [c for c in features.columns if c.startswith("f")]
classes = sorted(features["behaviour_code"].unique().tolist())

pred_rows = []

for _, b in best.iterrows():
    policy = clean(b["split_policy"])
    crop_type = clean(b["crop_type"])
    model_type = clean(b["model_type"])

    split_col = "current_split" if policy == "current_recommended_split" else "group_aware_split"

    f = features[features["crop_type"] == crop_type].copy()
    train = f[f[split_col] == "train"].copy()
    test = f[f[split_col] == "test"].copy()

    X_train_raw = train[feature_cols].to_numpy(dtype=np.float32)
    y_train = train["behaviour_code"].to_numpy()
    X_test_raw = test[feature_cols].to_numpy(dtype=np.float32)
    y_test = test["behaviour_code"].to_numpy()

    X_train, X_test = standardize_train(X_train_raw, X_test_raw)

    if model_type == "nearest_centroid":
        y_pred, conf = pred_centroid(X_train, y_train, X_test, classes)
    else:
        y_pred, conf = pred_knn3(X_train, y_train, X_test, classes)

    for i, (_, r) in enumerate(test.iterrows()):
        actual = clean(r["behaviour_code"])
        predicted = clean(y_pred[i])
        pred_rows.append({
            "split_policy": policy,
            "crop_type": crop_type,
            "model_type": model_type,
            "eval_split": "test",
            "canonical_gt_object_id": clean(r["canonical_gt_object_id"]),
            "scan_frame_id": clean(r["scan_frame_id"]),
            "video_id": clean(r["video_id"]),
            "actual": actual,
            "predicted": predicted,
            "correct": actual == predicted,
            "confidence_like_score": float(conf[i]),
            "crop_path": clean(r["crop_path"]),
        })

pred = pd.DataFrame(pred_rows)
errors = pred[pred["correct"] == False].copy()
correct = pred[pred["correct"] == True].copy()

safe_to_csv(pred, OUT_PRED)
safe_to_csv(errors, OUT_ERRORS)
safe_to_csv(correct, OUT_CORRECT)

conf = (
    errors.groupby(["split_policy", "crop_type", "model_type", "actual", "predicted"])
    .size()
    .reset_index(name="error_count")
    .sort_values(["split_policy", "error_count"], ascending=[True, False])
)
safe_to_csv(conf, OUT_CONF)

class_summary = (
    pred.groupby(["split_policy", "crop_type", "model_type", "actual"])
    .agg(
        support=("actual", "size"),
        correct=("correct", "sum"),
        mean_confidence=("confidence_like_score", "mean"),
    )
    .reset_index()
)
class_summary["errors"] = class_summary["support"] - class_summary["correct"]
class_summary["recall"] = class_summary["correct"] / class_summary["support"].replace(0, np.nan)
class_summary = class_summary.sort_values(["split_policy", "recall", "support"])
safe_to_csv(class_summary, OUT_CLASS)

scan_summary = (
    pred.groupby(["split_policy", "scan_frame_id"])
    .agg(
        test_objects=("actual", "size"),
        errors=("correct", lambda s: int((~s).sum())),
        correct=("correct", "sum"),
    )
    .reset_index()
)
scan_summary["error_rate"] = scan_summary["errors"] / scan_summary["test_objects"].replace(0, np.nan)
scan_summary = scan_summary.sort_values(["split_policy", "error_rate", "errors"], ascending=[True, False, False])
safe_to_csv(scan_summary, OUT_SCAN)

video_summary = (
    pred.groupby(["split_policy", "video_id"])
    .agg(
        test_objects=("actual", "size"),
        errors=("correct", lambda s: int((~s).sum())),
        correct=("correct", "sum"),
    )
    .reset_index()
)
video_summary["error_rate"] = video_summary["errors"] / video_summary["test_objects"].replace(0, np.nan)
video_summary = video_summary.sort_values(["split_policy", "error_rate", "errors"], ascending=[True, False, False])
safe_to_csv(video_summary, OUT_VIDEO)

qa_rows = []


def add_qa(name, expected, actual, passed, severity, detail):
    qa_rows.append({
        "check_name": name,
        "expected": str(expected),
        "actual": str(actual),
        "passed": bool(passed),
        "severity": severity,
        "detail": detail,
    })


current_pred = pred[pred["split_policy"] == "current_recommended_split"]
group_pred = pred[pred["split_policy"] == "selected_group_aware_split"]

add_qa("prediction_rows_total", 150, len(pred), len(pred) == 150, "hard", "Expected 78 current test + 72 group-aware test predictions.")
add_qa("current_test_predictions", 78, len(current_pred), len(current_pred) == 78, "hard", "Current split should have 78 test predictions.")
add_qa("group_test_predictions", 72, len(group_pred), len(group_pred) == 72, "hard", "Group-aware split should have 72 test predictions.")
add_qa("error_rows_nonempty", ">0", len(errors), len(errors) > 0, "info", "Errors should exist for analysis because model is weak.")
add_qa("top_confusions_nonempty", ">0", len(conf), len(conf) > 0, "hard", "Top confusion table should be created.")
add_qa("class_summary_rows", ">0", len(class_summary), len(class_summary) > 0, "hard", "Per-class error summary should exist.")
add_qa("scanframe_summary_rows", ">0", len(scan_summary), len(scan_summary) > 0, "hard", "Scanframe error summary should exist.")

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_QA)

hard_quality_failures = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v69c_quality_checks",
        "issue_type": "hard_error_analysis_quality_check_failed",
        "issue_detail": f"{hard_quality_failures} hard checks failed.",
        "severity": "hard",
    })

issues.append({
    "item": "model_strength",
    "issue_type": "info_baseline_model_is_weak",
    "issue_detail": "Feature baseline improves over sanity baseline but remains weak; use for error analysis only.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

top_conf_text = conf.head(10).to_string(index=False)
weak_classes_text = class_summary.head(12).to_string(index=False)

manifest = {
    "version": "week8_v69c_error_analysis",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "prediction_rows": int(len(pred)),
    "error_rows": int(len(errors)),
    "correct_rows": int(len(correct)),
    "top_confusions_preview": conf.head(10).to_dict(orient="records"),
    "claim_boundary": "error analysis for simple feature baseline only",
}

OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

OUT_README.write_text(
    "# Week8 v69c Error Analysis\n\n"
    "This package analyzes errors from the best v69b simple feature baselines.\n\n"
    "It is not a production classifier evaluation.\n\n"
    "Main files:\n"
    "- week8_v69c_best_model_predictions.csv\n"
    "- week8_v69c_best_model_errors.csv\n"
    "- week8_v69c_top_confusions.csv\n"
    "- week8_v69c_per_class_error_summary.csv\n"
)

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

zip_hash = sha256_file(OUT_ZIP)
OUT_SHA256.write_text(f"{zip_hash}  {OUT_ZIP.name}\n")

decision = pd.DataFrame([{
    "v69c_decision": "error_analysis_completed" if hard_issue_count == 0 else "error_analysis_has_blocking_issues",
    "prediction_rows": int(len(pred)),
    "error_rows": int(len(errors)),
    "correct_rows": int(len(correct)),
    "current_errors": int((current_pred["correct"] == False).sum()),
    "group_errors": int((group_pred["correct"] == False).sum()),
    "top_confusion_count": int(len(conf)),
    "hard_quality_failures": hard_quality_failures,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "issue_count": int(len(issues_df)),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "ready_for_v70a_improved_baseline": bool(hard_issue_count == 0),
    "claim_scope": "feature_baseline_error_analysis_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v69c Error Analysis\n\n"
    f"- v69c decision: {decision.iloc[0]['v69c_decision']}\n"
    f"- Prediction rows: {len(pred)}\n"
    f"- Error rows: {len(errors)}\n"
    f"- Correct rows: {len(correct)}\n"
    f"- Current split errors: {decision.iloc[0]['current_errors']}\n"
    f"- Group-aware split errors: {decision.iloc[0]['group_errors']}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Ready for v70a improved baseline: {bool(hard_issue_count == 0)}\n\n"
    "Top confusions:\n\n"
    f"{top_conf_text}\n\n"
    "Weakest class summaries:\n\n"
    f"{weak_classes_text}\n"
)

OUT_REPORT.write_text(
    "# Week 8 v69c Error Analysis Report\n\n"
    f"Decision: {decision.iloc[0]['v69c_decision']}\n\n"
    f"Prediction rows: {len(pred)}\n\n"
    f"Error rows: {len(errors)}\n\n"
    f"ZIP: {OUT_ZIP}\n\n"
    f"SHA256: {zip_hash}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v69c",
    "task_name": "Feature baseline error analysis",
    "status": "PASS" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": str(PKG69B),
    "output_summary": str(PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Run v70a improved baseline if desired." if hard_issue_count == 0 else "Fix v69c hard issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("=== v69c decision ===")
print(decision.to_string(index=False))
print("\n=== top confusions ===")
print(conf.head(12).to_string(index=False))
print("\n=== issues ===")
print(issues_df.to_string(index=False) if len(issues_df) else "No issues found.")
