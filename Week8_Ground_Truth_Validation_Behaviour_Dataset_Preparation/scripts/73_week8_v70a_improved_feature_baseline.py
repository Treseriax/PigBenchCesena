from pathlib import Path
from datetime import datetime
from collections import Counter
import csv, json, zipfile, hashlib
import numpy as np
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V69B = W8 / "outputs" / "v69b_feature_baseline"
V69C = W8 / "outputs" / "v69c_error_analysis"
V69D = W8 / "outputs" / "v69d_error_analysis_interface"

FEATURES = V69B / "Week8_Feature_Baseline" / "week8_v69b_feature_vectors.csv"
BEST69B = V69B / "Week8_Feature_Baseline" / "week8_v69b_best_model_summary.csv"
V69B_DECISION = V69B / "week8_v69b_decision_summary.csv"
V69C_DECISION = V69C / "week8_v69c_decision_summary.csv"
V69D_DECISION = V69D / "week8_v69d_decision_summary.csv"

OUT = W8 / "outputs" / "v70a_improved_feature_baseline"
PKG = OUT / "Week8_Improved_Feature_Baseline"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_ALL = PKG / "week8_v70a_all_config_metrics.csv"
OUT_SELECTED = PKG / "week8_v70a_selected_config_metrics.csv"
OUT_BEST = PKG / "week8_v70a_best_test_summary.csv"
OUT_PRED = PKG / "week8_v70a_selected_test_predictions.csv"
OUT_PER_CLASS = PKG / "week8_v70a_selected_test_per_class_metrics.csv"
OUT_CONF = PKG / "week8_v70a_selected_test_confusion_matrix_long.csv"
OUT_COMPARE = PKG / "week8_v70a_vs_v69b_comparison.csv"
OUT_QA = PKG / "week8_v70a_quality_checks.csv"
OUT_README = PKG / "README_Week8_Improved_Feature_Baseline.md"
OUT_MANIFEST = PKG / "week8_v70a_manifest.json"

OUT_DECISION = OUT / "week8_v70a_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v70a_issues.csv"
OUT_ZIP = OUT / "Week8_Improved_Feature_Baseline.zip"
OUT_SHA = OUT / "Week8_Improved_Feature_Baseline.sha256"
OUT_NOTE = NOTES / "week8_v70a_improved_feature_baseline_notes.md"
OUT_REPORT = REPORTS / "week8_v70a_improved_feature_baseline_report.md"
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


def read_csv(path):
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


def standardize(Xtr, Xev):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return (Xtr - mu) / sd, (Xev - mu) / sd, mu, sd


def pca_fit_transform(Xtr, Xev, k):
    max_k = min(Xtr.shape[0] - 1, Xtr.shape[1])
    if k == "full" or int(k) >= max_k:
        return Xtr, Xev, "full"

    k = int(k)
    k = max(1, min(k, max_k))
    U, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
    comp = Vt[:k]
    return Xtr @ comp.T, Xev @ comp.T, k


def fit_ridge(X, y, classes, lam):
    n, d = X.shape
    Xb = np.concatenate([np.ones((n, 1), dtype=np.float32), X], axis=1)

    class_counts = Counter(y)
    weights = np.array([n / (len(classes) * class_counts[label]) for label in y], dtype=np.float32)
    sw = np.sqrt(weights)[:, None]

    Y = np.zeros((n, len(classes)), dtype=np.float32)
    idx = {c: i for i, c in enumerate(classes)}
    for i, label in enumerate(y):
        Y[i, idx[label]] = 1.0

    Xw = Xb * sw
    Yw = Y * sw

    reg = np.eye(Xb.shape[1], dtype=np.float32) * float(lam)
    reg[0, 0] = 0.0

    A = Xw.T @ Xw + reg
    B = Xw.T @ Yw

    try:
        W = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(A) @ B

    return W


def predict_ridge(X, W, classes):
    Xb = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float32), X], axis=1)
    scores = Xb @ W
    pred_idx = scores.argmax(axis=1)
    conf = scores.max(axis=1) - np.partition(scores, -2, axis=1)[:, -2]
    return np.array([classes[i] for i in pred_idx]), conf


def per_class(y_true, y_pred, classes):
    rows = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for c in classes:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        tn = int(((y_true != c) & (y_pred != c)).sum())
        support = int((y_true == c).sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

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


def metrics(y_true, y_pred, classes):
    pc = per_class(y_true, y_pred, classes)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    macro_f1 = float(pc["f1"].mean())
    macro_precision = float(pc["precision"].mean())
    macro_recall = float(pc["recall"].mean())
    weighted_f1 = float((pc["f1"] * pc["support"]).sum() / max(pc["support"].sum(), 1))
    return {
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": macro_recall,
    }, pc


def confusion(y_true, y_pred, classes):
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


issues = []

for p in [FEATURES, BEST69B, V69B_DECISION, V69C_DECISION, V69D_DECISION]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required input missing for v70a.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v70a_decision": "improved_feature_baseline_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v70b_error_interface": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)

d69b = read_csv(V69B_DECISION)
d69c = read_csv(V69C_DECISION)
d69d = read_csv(V69D_DECISION)

if not bool_true(d69b.iloc[0].get("ready_for_v69c_error_analysis", "")):
    issues.append({"item": "v69b", "issue_type": "hard_v69b_not_ready", "issue_detail": "v69b not ready.", "severity": "hard"})
if not bool_true(d69c.iloc[0].get("ready_for_v70a_improved_baseline", "")):
    issues.append({"item": "v69c", "issue_type": "hard_v69c_not_ready", "issue_detail": "v69c not ready.", "severity": "hard"})
if not bool_true(d69d.iloc[0].get("ready_for_v70a_improved_baseline", "")):
    issues.append({"item": "v69d", "issue_type": "hard_v69d_not_ready", "issue_detail": "v69d not ready.", "severity": "hard"})

features = read_csv(FEATURES)
best69b = read_csv(BEST69B)

feature_cols = [c for c in features.columns if c.startswith("f")]
classes = sorted(features["behaviour_code"].unique().tolist())

if len(features) != 744:
    issues.append({"item": "feature_rows", "issue_type": "hard_unexpected_feature_rows", "issue_detail": f"Expected 744, got {len(features)}.", "severity": "hard"})
if len(feature_cols) == 0:
    issues.append({"item": "feature_columns", "issue_type": "hard_no_feature_columns", "issue_detail": "No feature columns found.", "severity": "hard"})

if any(i["severity"] == "hard" for i in issues):
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v70a_decision": "improved_feature_baseline_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v70b_error_interface": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    print(issues_df.to_string(index=False))
    raise SystemExit(1)


policies = [
    ("current_recommended_split", "current_split"),
    ("selected_group_aware_split", "group_aware_split"),
]

crop_types = ["tight", "context10"]
pca_dims = [16, 32, 64, 128, "full"]
lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]

all_metric_rows = []
selected_metric_rows = []
prediction_rows = []
per_class_frames = []
conf_frames = []

for policy_name, split_col in policies:
    for crop_type in crop_types:
        f = features[features["crop_type"] == crop_type].copy()

        train = f[f[split_col] == "train"].copy()
        val = f[f[split_col] == "val"].copy()
        test = f[f[split_col] == "test"].copy()

        X_train_raw = train[feature_cols].to_numpy(dtype=np.float32)
        y_train = train["behaviour_code"].to_numpy()

        split_data = {
            "train": train,
            "val": val,
            "test": test,
        }

        config_records = []

        for dim in pca_dims:
            for lam in lambdas:
                # Use val to select config.
                val_df = split_data["val"]
                X_val_raw = val_df[feature_cols].to_numpy(dtype=np.float32)
                y_val = val_df["behaviour_code"].to_numpy()

                Xtr_std, Xval_std, _, _ = standardize(X_train_raw, X_val_raw)
                Xtr_pca, Xval_pca, used_dim = pca_fit_transform(Xtr_std, Xval_std, dim)

                W = fit_ridge(Xtr_pca, y_train, classes, lam)
                pred_val, _ = predict_ridge(Xval_pca, W, classes)
                m_val, _ = metrics(y_val, pred_val, classes)

                config_records.append({
                    "policy": policy_name,
                    "crop_type": crop_type,
                    "pca_dim": used_dim,
                    "lambda": lam,
                    "val_macro_f1": m_val["macro_f1"],
                    "val_balanced_accuracy": m_val["balanced_accuracy"],
                    "val_accuracy": m_val["accuracy"],
                })

        configs = pd.DataFrame(config_records)
        best_config = configs.sort_values(
            ["val_macro_f1", "val_balanced_accuracy", "val_accuracy"],
            ascending=[False, False, False],
        ).iloc[0].to_dict()

        for dim in pca_dims:
            for lam in lambdas:
                for eval_split, eval_df in split_data.items():
                    X_eval_raw = eval_df[feature_cols].to_numpy(dtype=np.float32)
                    y_eval = eval_df["behaviour_code"].to_numpy()

                    Xtr_std, Xev_std, _, _ = standardize(X_train_raw, X_eval_raw)
                    Xtr_pca, Xev_pca, used_dim = pca_fit_transform(Xtr_std, Xev_std, dim)

                    W = fit_ridge(Xtr_pca, y_train, classes, lam)
                    pred, conf_score = predict_ridge(Xev_pca, W, classes)
                    m, pc = metrics(y_eval, pred, classes)

                    row = {
                        "split_policy": policy_name,
                        "crop_type": crop_type,
                        "model_type": "class_balanced_ridge",
                        "pca_dim": used_dim,
                        "lambda": lam,
                        "eval_split": eval_split,
                        "n_train": len(train),
                        "n_eval": len(eval_df),
                        "n_classes": len(classes),
                        **m,
                        "selected_by_val": bool(str(used_dim) == str(best_config["pca_dim"]) and float(lam) == float(best_config["lambda"])),
                    }
                    all_metric_rows.append(row)

                    if row["selected_by_val"]:
                        selected_metric_rows.append(row)

                        if eval_split == "test":
                            pc.insert(0, "eval_split", eval_split)
                            pc.insert(0, "lambda", lam)
                            pc.insert(0, "pca_dim", used_dim)
                            pc.insert(0, "model_type", "class_balanced_ridge")
                            pc.insert(0, "crop_type", crop_type)
                            pc.insert(0, "split_policy", policy_name)
                            per_class_frames.append(pc)

                            cm = confusion(y_eval, pred, classes)
                            cm.insert(0, "eval_split", eval_split)
                            cm.insert(0, "lambda", lam)
                            cm.insert(0, "pca_dim", used_dim)
                            cm.insert(0, "model_type", "class_balanced_ridge")
                            cm.insert(0, "crop_type", crop_type)
                            cm.insert(0, "split_policy", policy_name)
                            conf_frames.append(cm)

                            for i, (_, rr) in enumerate(eval_df.iterrows()):
                                prediction_rows.append({
                                    "split_policy": policy_name,
                                    "crop_type": crop_type,
                                    "model_type": "class_balanced_ridge",
                                    "pca_dim": used_dim,
                                    "lambda": lam,
                                    "eval_split": "test",
                                    "canonical_gt_object_id": clean(rr["canonical_gt_object_id"]),
                                    "scan_frame_id": clean(rr["scan_frame_id"]),
                                    "video_id": clean(rr["video_id"]),
                                    "actual": clean(y_eval[i]),
                                    "predicted": clean(pred[i]),
                                    "correct": bool(y_eval[i] == pred[i]),
                                    "confidence_like_score": float(conf_score[i]),
                                    "crop_path": clean(rr["crop_path"]),
                                })

all_metrics = pd.DataFrame(all_metric_rows)
selected_metrics = pd.DataFrame(selected_metric_rows)
predictions = pd.DataFrame(prediction_rows)
per_class_df = pd.concat(per_class_frames, ignore_index=True)
conf_df = pd.concat(conf_frames, ignore_index=True)

safe_to_csv(all_metrics, OUT_ALL)
safe_to_csv(selected_metrics, OUT_SELECTED)
safe_to_csv(predictions, OUT_PRED)
safe_to_csv(per_class_df, OUT_PER_CLASS)
safe_to_csv(conf_df, OUT_CONF)

best_test = (
    selected_metrics[selected_metrics["eval_split"] == "test"]
    .sort_values(["split_policy", "macro_f1", "balanced_accuracy", "accuracy"], ascending=[True, False, False, False])
    .groupby("split_policy")
    .head(1)
    .reset_index(drop=True)
)
safe_to_csv(best_test, OUT_BEST)

compare_rows = []
for _, r in best_test.iterrows():
    policy = clean(r["split_policy"])
    old = best69b[best69b["split_policy"] == policy]
    old_macro = float(old.iloc[0]["macro_f1"]) if len(old) else np.nan
    old_acc = float(old.iloc[0]["accuracy"]) if len(old) else np.nan

    compare_rows.append({
        "split_policy": policy,
        "v69b_best_macro_f1": old_macro,
        "v70a_best_macro_f1": float(r["macro_f1"]),
        "delta_macro_f1": float(r["macro_f1"]) - old_macro,
        "v69b_best_accuracy": old_acc,
        "v70a_best_accuracy": float(r["accuracy"]),
        "delta_accuracy": float(r["accuracy"]) - old_acc,
        "v70a_crop_type": clean(r["crop_type"]),
        "v70a_pca_dim": clean(r["pca_dim"]),
        "v70a_lambda": clean(r["lambda"]),
        "v70a_beats_v69b_macro_f1": bool(float(r["macro_f1"]) > old_macro),
    })

compare = pd.DataFrame(compare_rows)
safe_to_csv(compare, OUT_COMPARE)

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

add_qa("feature_rows", 744, len(features), len(features) == 744, "hard", "Feature rows should be 744.")
add_qa("all_metric_rows", 300, len(all_metrics), len(all_metrics) == 300, "hard", "2 policies x 2 crops x 5 dims x 5 lambdas x 3 splits = 300.")
add_qa("selected_metric_rows", 12, len(selected_metrics), len(selected_metrics) == 12, "hard", "2 policies x 2 crops x 3 splits selected metrics.")
add_qa("test_prediction_rows", 300, len(predictions), len(predictions) == 300, "hard", "Selected test predictions: current 78*2 crops + group 72*2 crops = 300.")
add_qa("per_class_rows", 44, len(per_class_df), len(per_class_df) == 44, "hard", "2 policies x 2 crops x 11 classes.")
add_qa("confusion_rows", 484, len(conf_df), len(conf_df) == 484, "hard", "2 policies x 2 crops x 11 x 11 confusion rows.")

for _, r in compare.iterrows():
    add_qa(
        f"{r['split_policy']}_beats_v69b_macro_f1",
        True,
        r["v70a_beats_v69b_macro_f1"],
        bool(r["v70a_beats_v69b_macro_f1"]),
        "warning",
        "Improved baseline should ideally beat v69b macro-F1.",
    )

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_QA)

hard_quality_failures = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())
warning_quality_failures = int(((qa["severity"] == "warning") & (~qa["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v70a_quality_checks",
        "issue_type": "hard_v70a_quality_check_failed",
        "issue_detail": f"{hard_quality_failures} hard checks failed.",
        "severity": "hard",
    })

if warning_quality_failures:
    issues.append({
        "item": "v70a_model_strength",
        "issue_type": "warning_v70a_did_not_beat_v69b_for_all_policies",
        "issue_detail": f"{warning_quality_failures} macro-F1 improvement warnings.",
        "severity": "warning",
    })

issues.append({
    "item": "claim_scope",
    "issue_type": "info_improved_baseline_only",
    "issue_detail": "v70a is an improved classical baseline, not a production classifier.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

manifest = {
    "version": "week8_v70a_improved_feature_baseline",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "model": "class_balanced_ridge_with_optional_pca",
    "selection": "best validation macro_f1, tie balanced_accuracy and accuracy",
    "features_source": str(FEATURES),
    "best_test": best_test.to_dict(orient="records"),
    "comparison_vs_v69b": compare.to_dict(orient="records"),
    "claim_boundary": "improved classical baseline only; no production classifier claim",
}
OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

OUT_README.write_text(
    "# Week8 v70a Improved Feature Baseline\n\n"
    "Class-balanced ridge classifier with optional PCA.\n\n"
    "Config is selected by validation macro-F1.\n\n"
    "This is not a production classifier.\n\n"
    "Best test summary:\n\n"
    + best_test.to_string(index=False)
    + "\n\nComparison vs v69b:\n\n"
    + compare.to_string(index=False)
    + "\n"
)

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

zip_hash = sha256_file(OUT_ZIP)
OUT_SHA.write_text(f"{zip_hash}  {OUT_ZIP.name}\n")

cur = best_test[best_test["split_policy"] == "current_recommended_split"].iloc[0]
grp = best_test[best_test["split_policy"] == "selected_group_aware_split"].iloc[0]

decision = pd.DataFrame([{
    "v70a_decision": "improved_feature_baseline_completed" if hard_issue_count == 0 else "improved_feature_baseline_has_blocking_issues",
    "feature_rows": int(len(features)),
    "feature_dimension": int(len(feature_cols)),
    "current_best_crop_type": clean(cur["crop_type"]),
    "current_best_pca_dim": clean(cur["pca_dim"]),
    "current_best_lambda": clean(cur["lambda"]),
    "current_best_test_accuracy": float(cur["accuracy"]),
    "current_best_test_macro_f1": float(cur["macro_f1"]),
    "current_best_balanced_accuracy": float(cur["balanced_accuracy"]),
    "group_best_crop_type": clean(grp["crop_type"]),
    "group_best_pca_dim": clean(grp["pca_dim"]),
    "group_best_lambda": clean(grp["lambda"]),
    "group_best_test_accuracy": float(grp["accuracy"]),
    "group_best_test_macro_f1": float(grp["macro_f1"]),
    "group_best_balanced_accuracy": float(grp["balanced_accuracy"]),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_quality_failures": hard_quality_failures,
    "warning_quality_failures": warning_quality_failures,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v70b_error_interface": bool(hard_issue_count == 0),
    "claim_scope": "improved_classical_feature_baseline_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v70a Improved Feature Baseline\n\n"
    f"- v70a decision: {decision.iloc[0]['v70a_decision']}\n"
    f"- Current best: {cur['crop_type']} / PCA={cur['pca_dim']} / lambda={cur['lambda']} / test macro-F1={float(cur['macro_f1']):.4f} / acc={float(cur['accuracy']):.4f}\n"
    f"- Group-aware best: {grp['crop_type']} / PCA={grp['pca_dim']} / lambda={grp['lambda']} / test macro-F1={float(grp['macro_f1']):.4f} / acc={float(grp['accuracy']):.4f}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Warnings: {warning_count}\n"
    f"- Ready for v70b error interface: {bool(hard_issue_count == 0)}\n\n"
    "This is an improved classical feature baseline only. No production classifier claim.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v70a Improved Feature Baseline Report\n\n"
    f"Decision: {decision.iloc[0]['v70a_decision']}\n\n"
    f"Current best macro-F1: {float(cur['macro_f1'])}\n\n"
    f"Group-aware best macro-F1: {float(grp['macro_f1'])}\n\n"
    f"ZIP: {OUT_ZIP}\n\n"
    f"SHA256: {zip_hash}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v70a",
    "task_name": "Improved classical feature baseline",
    "status": "PASS_WITH_WARNINGS" if hard_issue_count == 0 and warning_count > 0 else ("PASS" if hard_issue_count == 0 else "NEEDS_FIX"),
    "input_summary": str(FEATURES),
    "output_summary": str(PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Create v70b error interface or package v70 results." if hard_issue_count == 0 else "Fix v70a hard issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("=== v70a decision ===")
print(decision.to_string(index=False))
print("\n=== best test summary ===")
print(best_test.to_string(index=False))
print("\n=== comparison vs v69b ===")
print(compare.to_string(index=False))
print("\n=== issues ===")
print(issues_df.to_string(index=False) if len(issues_df) else "No issues found.")
