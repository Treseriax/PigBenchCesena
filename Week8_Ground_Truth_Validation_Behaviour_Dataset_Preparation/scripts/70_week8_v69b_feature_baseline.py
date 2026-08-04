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

V68D = W8 / "outputs" / "v68d_group_aware_split_policy"
V69A = W8 / "outputs" / "v69a_baseline_sanity"

META = V68D / "Week8_GroupAware_Split_Policy" / "week8_v68d_group_aware_crop_metadata.csv"
V68D_DECISION = V68D / "week8_v68d_decision_summary.csv"
V69A_DECISION = V69A / "week8_v69a_decision_summary.csv"
V69A_METRICS = V69A / "Week8_Baseline_Sanity" / "week8_v69a_baseline_sanity_metrics.csv"

OUT = W8 / "outputs" / "v69b_feature_baseline"
PKG = OUT / "Week8_Feature_Baseline"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_FEATURES = PKG / "week8_v69b_feature_vectors.csv"
OUT_FAILED = PKG / "week8_v69b_failed_feature_rows.csv"
OUT_METRICS = PKG / "week8_v69b_feature_baseline_metrics.csv"
OUT_PER_CLASS = PKG / "week8_v69b_feature_baseline_per_class_metrics.csv"
OUT_CONFUSION = PKG / "week8_v69b_feature_baseline_confusion_matrix_long.csv"
OUT_BEST = PKG / "week8_v69b_best_model_summary.csv"
OUT_QA = PKG / "week8_v69b_feature_baseline_quality_checks.csv"
OUT_README = PKG / "README_Week8_Feature_Baseline.md"
OUT_MANIFEST = PKG / "week8_v69b_feature_baseline_manifest.json"

OUT_DECISION = OUT / "week8_v69b_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v69b_issues.csv"
OUT_ZIP = OUT / "Week8_Feature_Baseline.zip"
OUT_SHA256 = OUT / "Week8_Feature_Baseline.sha256"
OUT_NOTE = NOTES / "week8_v69b_feature_baseline_notes.md"
OUT_REPORT = REPORTS / "week8_v69b_feature_baseline_report.md"
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


def image_feature(path):
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        return None, "image_read_failed"

    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    feats = []

    feats.extend(rgb.reshape(-1, 3).mean(axis=0).tolist())
    feats.extend(rgb.reshape(-1, 3).std(axis=0).tolist())

    for ch in range(3):
        hist, _ = np.histogram(rgb[:, :, ch], bins=16, range=(0, 1), density=False)
        hist = hist.astype(np.float32)
        hist = hist / max(hist.sum(), 1.0)
        feats.extend(hist.tolist())

    hsv_norm = np.zeros_like(hsv)
    hsv_norm[:, :, 0] = hsv[:, :, 0] / 179.0
    hsv_norm[:, :, 1] = hsv[:, :, 1] / 255.0
    hsv_norm[:, :, 2] = hsv[:, :, 2] / 255.0

    for ch in range(3):
        hist, _ = np.histogram(hsv_norm[:, :, ch], bins=16, range=(0, 1), density=False)
        hist = hist.astype(np.float32)
        hist = hist / max(hist.sum(), 1.0)
        feats.extend(hist.tolist())

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    bins = np.linspace(0, 2 * np.pi, 9)
    ghist, _ = np.histogram(ang, bins=bins, weights=mag)
    ghist = ghist.astype(np.float32)
    ghist = ghist / max(ghist.sum(), 1e-6)
    feats.extend(ghist.tolist())

    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    feats.extend(small.flatten().tolist())

    return np.array(feats, dtype=np.float32), "ok"


def standardize_train(X_train, X_eval):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return (X_train - mu) / sd, (X_eval - mu) / sd


def predict_nearest_centroid(X_train, y_train, X_eval, classes):
    centroids = []
    for c in classes:
        centroids.append(X_train[y_train == c].mean(axis=0))
    centroids = np.vstack(centroids)
    d = ((X_eval[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return np.array([classes[i] for i in d.argmin(axis=1)])


def predict_knn(X_train, y_train, X_eval, classes, k=3):
    preds = []
    for x in X_eval:
        d = ((X_train - x) ** 2).sum(axis=1)
        idx = np.argsort(d)[:k]
        votes = Counter(y_train[idx])
        best = sorted(votes.items(), key=lambda kv: (-kv[1], classes.index(kv[0])))[0][0]
        preds.append(best)
    return np.array(preds)


def per_class_metrics(y_true, y_pred, classes):
    rows = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for c in classes:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        tn = int(((y_true != c) & (y_pred != c)).sum())
        support = int((y_true == c).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        rows.append({
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

    return pd.DataFrame(rows)


def aggregate_metrics(y_true, y_pred, classes):
    pc = per_class_metrics(y_true, y_pred, classes)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    macro_precision = float(pc["precision"].mean())
    macro_recall = float(pc["recall"].mean())
    macro_f1 = float(pc["f1"].mean())
    weighted_f1 = float((pc["f1"] * pc["support"]).sum() / max(pc["support"].sum(), 1))
    return {
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": macro_recall,
    }, pc


def confusion_long(y_true, y_pred, classes):
    rows = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    for a in classes:
        for p in classes:
            rows.append({
                "actual": a,
                "predicted": p,
                "count": int(((y_true == a) & (y_pred == p)).sum()),
            })
    return pd.DataFrame(rows)


def get_sanity_metric(sanity, policy, split, baseline, metric):
    r = sanity[
        (sanity["split_policy"] == policy)
        & (sanity["eval_split"] == split)
        & (sanity["baseline_type"] == baseline)
    ]
    if len(r) == 0:
        return np.nan
    return float(r.iloc[0][metric])


issues = []

for p in [META, V68D_DECISION, V69A_DECISION, V69A_METRICS]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required input missing for v69b feature baseline.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v69b_decision": "feature_baseline_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v69c_error_analysis": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


v68d_decision = read_csv_clean(V68D_DECISION)
v69a_decision = read_csv_clean(V69A_DECISION)

if len(v68d_decision) == 0 or not bool_true(v68d_decision.iloc[0].get("ready_for_v69a_baseline_sanity", "")):
    issues.append({
        "item": str(V68D_DECISION),
        "issue_type": "hard_v68d_not_ready",
        "issue_detail": "v68d must be ready before v69b.",
        "severity": "hard",
    })

if len(v69a_decision) == 0 or not bool_true(v69a_decision.iloc[0].get("ready_for_v69b_feature_baseline", "")):
    issues.append({
        "item": str(V69A_DECISION),
        "issue_type": "hard_v69a_not_ready",
        "issue_detail": "v69a must be ready before v69b.",
        "severity": "hard",
    })

try:
    import cv2
except Exception as e:
    issues.append({
        "item": "opencv",
        "issue_type": "hard_opencv_import_failed",
        "issue_detail": str(e),
        "severity": "hard",
    })

if any(i["severity"] == "hard" for i in issues):
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v69b_decision": "feature_baseline_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v69c_error_analysis": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    print(issues_df.to_string(index=False))
    raise SystemExit(1)


meta = read_csv_clean(META)
sanity = read_csv_clean(V69A_METRICS)

crop_specs = [
    ("tight", "tight_crop_path"),
    ("context10", "context10_crop_path"),
]

feature_rows = []
failed_rows = []
cache = {}

for _, r in meta.iterrows():
    for crop_type, col in crop_specs:
        path = Path(clean(r[col]))
        key = str(path)

        if key in cache:
            feat, status = cache[key]
        else:
            if not path.exists():
                feat, status = None, "file_missing"
            else:
                feat, status = image_feature(path)
            cache[key] = (feat, status)

        base = {
            "canonical_gt_object_id": clean(r["canonical_gt_object_id"]),
            "scan_frame_id": clean(r["scan_frame_id"]),
            "video_id": clean(r["video_id"]),
            "behaviour_code": clean(r["behaviour_code"]),
            "current_split": clean(r.get("current_split", r.get("recommended_split", ""))),
            "group_aware_split": clean(r["group_aware_split"]),
            "crop_type": crop_type,
            "crop_path": str(path),
        }

        if feat is None:
            failed_rows.append({**base, "failure_type": status})
            continue

        feat_dict = {f"f{i:03d}": float(v) for i, v in enumerate(feat)}
        feature_rows.append({**base, **feat_dict})

features = pd.DataFrame(feature_rows)
failed = pd.DataFrame(failed_rows)

safe_to_csv(features, OUT_FEATURES)
safe_to_csv(failed, OUT_FAILED)

feature_cols = [c for c in features.columns if c.startswith("f")]
classes = sorted(meta["behaviour_code"].unique().tolist())

policies = [
    ("current_recommended_split", "current_split"),
    ("selected_group_aware_split", "group_aware_split"),
]

models = ["nearest_centroid", "knn3"]

metrics_rows = []
per_class_frames = []
confusion_frames = []

for crop_type in ["tight", "context10"]:
    feat_crop = features[features["crop_type"] == crop_type].copy()

    for policy_name, split_col in policies:
        train = feat_crop[feat_crop[split_col] == "train"].copy()

        X_train_raw = train[feature_cols].to_numpy(dtype=np.float32)
        y_train = train["behaviour_code"].to_numpy()

        for eval_split in ["train", "val", "test"]:
            ev = feat_crop[feat_crop[split_col] == eval_split].copy()

            X_eval_raw = ev[feature_cols].to_numpy(dtype=np.float32)
            y_eval = ev["behaviour_code"].to_numpy()

            X_train, X_eval = standardize_train(X_train_raw, X_eval_raw)

            for model_name in models:
                if model_name == "nearest_centroid":
                    pred = predict_nearest_centroid(X_train, y_train, X_eval, classes)
                else:
                    pred = predict_knn(X_train, y_train, X_eval, classes, k=3)

                agg, pc = aggregate_metrics(y_eval, pred, classes)

                majority_macro = get_sanity_metric(
                    sanity,
                    policy_name,
                    eval_split,
                    "majority_train_class",
                    "macro_f1",
                )
                random_macro = get_sanity_metric(
                    sanity,
                    policy_name,
                    eval_split,
                    "train_prior_random_mean",
                    "macro_f1",
                )

                metrics_rows.append({
                    "split_policy": policy_name,
                    "crop_type": crop_type,
                    "model_type": model_name,
                    "eval_split": eval_split,
                    "n_train": int(len(train)),
                    "n_eval": int(len(ev)),
                    "n_classes": int(len(classes)),
                    **agg,
                    "sanity_majority_macro_f1": majority_macro,
                    "sanity_train_prior_random_macro_f1": random_macro,
                    "delta_macro_f1_vs_majority": agg["macro_f1"] - majority_macro if not np.isnan(majority_macro) else "",
                    "delta_macro_f1_vs_random": agg["macro_f1"] - random_macro if not np.isnan(random_macro) else "",
                    "beats_majority_macro_f1": bool(agg["macro_f1"] > majority_macro) if not np.isnan(majority_macro) else "",
                    "beats_random_macro_f1": bool(agg["macro_f1"] > random_macro) if not np.isnan(random_macro) else "",
                })

                pc.insert(0, "eval_split", eval_split)
                pc.insert(0, "model_type", model_name)
                pc.insert(0, "crop_type", crop_type)
                pc.insert(0, "split_policy", policy_name)
                per_class_frames.append(pc)

                cm = confusion_long(y_eval, pred, classes)
                cm.insert(0, "eval_split", eval_split)
                cm.insert(0, "model_type", model_name)
                cm.insert(0, "crop_type", crop_type)
                cm.insert(0, "split_policy", policy_name)
                confusion_frames.append(cm)

metrics = pd.DataFrame(metrics_rows)
per_class = pd.concat(per_class_frames, ignore_index=True)
confusion = pd.concat(confusion_frames, ignore_index=True)

safe_to_csv(metrics, OUT_METRICS)
safe_to_csv(per_class, OUT_PER_CLASS)
safe_to_csv(confusion, OUT_CONFUSION)

test_metrics = metrics[metrics["eval_split"] == "test"].copy()
best = (
    test_metrics.sort_values(["split_policy", "macro_f1", "balanced_accuracy", "accuracy"], ascending=[True, False, False, False])
    .groupby("split_policy")
    .head(1)
    .reset_index(drop=True)
)
safe_to_csv(best, OUT_BEST)

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


add_qa("metadata_rows", 372, len(meta), len(meta) == 372, "hard", "Input metadata should contain 372 rows.")
add_qa("feature_rows", 744, len(features), len(features) == 744, "hard", "Two crop types should produce 744 feature rows.")
add_qa("failed_feature_rows", 0, len(failed), len(failed) == 0, "hard", "No image feature extraction should fail.")
add_qa("feature_dimension_positive", ">0", len(feature_cols), len(feature_cols) > 0, "hard", "Feature vectors should contain numeric dimensions.")
add_qa("metrics_rows", 24, len(metrics), len(metrics) == 24, "hard", "Expected 2 policies x 2 crops x 2 models x 3 eval splits = 24 metric rows.")
add_qa("per_class_rows_created", ">0", len(per_class), len(per_class) > 0, "hard", "Per-class metrics should exist.")
add_qa("confusion_rows_created", ">0", len(confusion), len(confusion) > 0, "hard", "Confusion matrices should exist.")

for _, r in best.iterrows():
    add_qa(
        f"{r['split_policy']}_test_beats_majority_macro_f1",
        True,
        r["beats_majority_macro_f1"],
        bool(r["beats_majority_macro_f1"]),
        "warning",
        "Best test feature baseline should beat majority macro-F1; warning only because this is an early baseline.",
    )
    add_qa(
        f"{r['split_policy']}_test_beats_random_macro_f1",
        True,
        r["beats_random_macro_f1"],
        bool(r["beats_random_macro_f1"]),
        "warning",
        "Best test feature baseline should beat train-prior random macro-F1; warning only because this is an early baseline.",
    )

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_QA)

hard_quality_failures = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())
warning_quality_failures = int(((qa["severity"] == "warning") & (~qa["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v69b_quality_checks",
        "issue_type": "hard_feature_baseline_quality_check_failed",
        "issue_detail": f"{hard_quality_failures} hard checks failed.",
        "severity": "hard",
    })

if warning_quality_failures:
    issues.append({
        "item": "v69b_baseline_strength",
        "issue_type": "warning_feature_baseline_did_not_beat_all_sanity_baselines",
        "issue_detail": f"{warning_quality_failures} sanity-beating warning checks failed.",
        "severity": "warning",
    })

issues.append({
    "item": "claim_scope",
    "issue_type": "info_feature_baseline_only",
    "issue_detail": "v69b is a simple feature baseline, not a production classifier.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

manifest = {
    "version": "week8_v69b_feature_baseline",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "input_rows": int(len(meta)),
    "feature_rows": int(len(features)),
    "feature_dimension": int(len(feature_cols)),
    "crop_types": ["tight", "context10"],
    "split_policies": ["current_recommended_split", "selected_group_aware_split"],
    "models": models,
    "best_test_models": best.to_dict(orient="records"),
    "claim_boundary": "simple image feature baseline only; no production classifier claim",
}

OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

readme = f"""# Week8 v69b Feature Baseline

## Purpose

This stage trains simple image-feature baselines on strict-gold crop data.

## Inputs

- 372 strict-gold objects
- tight crops and context10 crops
- current split and group-aware split

## Models

- nearest centroid
- 3-nearest-neighbour

## Claim Boundary

This is a simple image-based baseline, not a production classifier.

## Best Test Models

{best.to_string(index=False)}
"""

OUT_README.write_text(readme)

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

zip_hash = sha256_file(OUT_ZIP)
OUT_SHA256.write_text(f"{zip_hash}  {OUT_ZIP.name}\n")

current_best = best[best["split_policy"] == "current_recommended_split"].iloc[0]
group_best = best[best["split_policy"] == "selected_group_aware_split"].iloc[0]

decision = pd.DataFrame([{
    "v69b_decision": "feature_baseline_completed" if hard_issue_count == 0 else "feature_baseline_has_blocking_issues",
    "metadata_rows": int(len(meta)),
    "feature_rows": int(len(features)),
    "failed_feature_rows": int(len(failed)),
    "feature_dimension": int(len(feature_cols)),
    "current_best_crop_type": current_best["crop_type"],
    "current_best_model": current_best["model_type"],
    "current_best_test_accuracy": current_best["accuracy"],
    "current_best_test_macro_f1": current_best["macro_f1"],
    "current_best_test_balanced_accuracy": current_best["balanced_accuracy"],
    "group_best_crop_type": group_best["crop_type"],
    "group_best_model": group_best["model_type"],
    "group_best_test_accuracy": group_best["accuracy"],
    "group_best_test_macro_f1": group_best["macro_f1"],
    "group_best_test_balanced_accuracy": group_best["balanced_accuracy"],
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_quality_failures": hard_quality_failures,
    "warning_quality_failures": warning_quality_failures,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v69c_error_analysis": bool(hard_issue_count == 0),
    "claim_scope": "simple_feature_baseline_only_no_production_classifier_claim",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v69b Feature Baseline\n\n"
    f"- v69b decision: {decision.iloc[0]['v69b_decision']}\n"
    f"- Feature rows: {len(features)}\n"
    f"- Failed feature rows: {len(failed)}\n"
    f"- Feature dimension: {len(feature_cols)}\n"
    f"- Current best: {current_best['crop_type']} / {current_best['model_type']} / test macro-F1={current_best['macro_f1']:.4f} / acc={current_best['accuracy']:.4f}\n"
    f"- Group-aware best: {group_best['crop_type']} / {group_best['model_type']} / test macro-F1={group_best['macro_f1']:.4f} / acc={group_best['accuracy']:.4f}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Warnings: {warning_count}\n"
    f"- Ready for v69c error analysis: {bool(hard_issue_count == 0)}\n\n"
    "This is a simple image feature baseline only. It is not a production classifier.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v69b Feature Baseline Report\n\n"
    f"Decision: {decision.iloc[0]['v69b_decision']}\n\n"
    f"Current best test macro-F1: {current_best['macro_f1']}\n\n"
    f"Group-aware best test macro-F1: {group_best['macro_f1']}\n\n"
    f"ZIP: {OUT_ZIP}\n\n"
    f"SHA256: {zip_hash}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v69b",
    "task_name": "Simple image feature baseline",
    "status": "PASS_WITH_WARNINGS" if hard_issue_count == 0 and warning_count > 0 else ("PASS" if hard_issue_count == 0 else "NEEDS_FIX"),
    "input_summary": str(META),
    "output_summary": str(PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Run v69c error analysis and confusion inspection." if hard_issue_count == 0 else "Fix v69b feature baseline hard issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_METRICS)
print(OUT_PER_CLASS)
print(OUT_CONFUSION)
print(OUT_BEST)
print(OUT_QA)
print(OUT_ZIP)
print(OUT_SHA256)
print(OUT_DECISION)
print(OUT_NOTE)

print()
print("=== v69b decision ===")
print(decision.to_string(index=False))

print()
print("=== best test models ===")
print(best.to_string(index=False))

print()
print("=== QA ===")
print(qa.to_string(index=False))

print()
print("=== issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
