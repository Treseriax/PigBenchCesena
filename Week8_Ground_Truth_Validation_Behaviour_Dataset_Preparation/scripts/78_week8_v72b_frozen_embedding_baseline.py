from pathlib import Path
from datetime import datetime
from collections import Counter
import csv, json, zipfile, hashlib
import numpy as np
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V69B = W8 / "outputs" / "v69b_feature_baseline"
V72A = W8 / "outputs" / "v72a_frozen_detector_embedding_extraction"

EMB = V72A / "Week8_Frozen_Detector_Embeddings" / "week8_v72a_frozen_detector_embeddings.csv"
V72A_DECISION = V72A / "week8_v72a_decision_summary.csv"
V69B_BEST = V69B / "Week8_Feature_Baseline" / "week8_v69b_best_model_summary.csv"

OUT = W8 / "outputs" / "v72b_frozen_embedding_baseline"
PKG = OUT / "Week8_Frozen_Embedding_Baseline"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_ALL = PKG / "week8_v72b_all_embedding_baseline_metrics.csv"
OUT_SELECTED = PKG / "week8_v72b_selected_config_metrics.csv"
OUT_BEST = PKG / "week8_v72b_best_test_summary.csv"
OUT_PRED = PKG / "week8_v72b_selected_test_predictions.csv"
OUT_PER_CLASS = PKG / "week8_v72b_selected_test_per_class_metrics.csv"
OUT_CONF = PKG / "week8_v72b_selected_test_confusion_matrix_long.csv"
OUT_COMPARE = PKG / "week8_v72b_vs_v69b_champion_comparison.csv"
OUT_QA = PKG / "week8_v72b_quality_checks.csv"
OUT_README = PKG / "README_Week8_Frozen_Embedding_Baseline.md"
OUT_MANIFEST = PKG / "week8_v72b_manifest.json"

OUT_DECISION = OUT / "week8_v72b_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v72b_issues.csv"
OUT_ZIP = OUT / "Week8_Frozen_Embedding_Baseline.zip"
OUT_SHA = OUT / "Week8_Frozen_Embedding_Baseline.sha256"
OUT_NOTE = NOTES / "week8_v72b_frozen_embedding_baseline_notes.md"
OUT_REPORT = REPORTS / "week8_v72b_frozen_embedding_baseline_report.md"
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


def standardize(X_train, X_eval):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return (X_train - mu) / sd, (X_eval - mu) / sd


def l2_normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < 1e-8] = 1.0
    return X / n


def pca_fit_transform(X_train, X_eval, k):
    max_k = min(X_train.shape[0] - 1, X_train.shape[1])

    if str(k) == "full" or int(k) >= max_k:
        return X_train, X_eval, "full"

    k = max(1, min(int(k), max_k))
    _, _, vt = np.linalg.svd(X_train, full_matrices=False)
    comp = vt[:k]

    return X_train @ comp.T, X_eval @ comp.T, k


def per_class_metrics(y_true, y_pred, classes):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rows = []

    for c in classes:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        tn = int(((y_true != c) & (y_pred != c)).sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = int((y_true == c).sum())

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


def compute_metrics(y_true, y_pred, classes):
    pc = per_class_metrics(y_true, y_pred, classes)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    macro_precision = float(pc["precision"].mean())
    macro_recall = float(pc["recall"].mean())
    macro_f1 = float(pc["f1"].mean())
    weighted_f1 = float((pc["f1"] * pc["support"]).sum() / max(pc["support"].sum(), 1))

    return {
        "accuracy": accuracy,
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


def predict_nearest_centroid(X_train, y_train, X_eval, classes, metric):
    if metric == "cosine":
        X_train = l2_normalize(X_train)
        X_eval = l2_normalize(X_eval)

    cents = []

    for c in classes:
        xc = X_train[y_train == c]
        if len(xc) == 0:
            cents.append(np.zeros((X_train.shape[1],), dtype=np.float32))
        else:
            cent = xc.mean(axis=0)
            cents.append(cent)

    cents = np.vstack(cents)

    if metric == "cosine":
        cents = l2_normalize(cents)
        scores = X_eval @ cents.T
        idx = scores.argmax(axis=1)
        conf = scores.max(axis=1)
    else:
        d = ((X_eval[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
        idx = d.argmin(axis=1)
        conf = 1.0 / (1.0 + d.min(axis=1))

    return np.array([classes[i] for i in idx]), conf


def predict_knn(X_train, y_train, X_eval, classes, k, metric):
    if metric == "cosine":
        X_train = l2_normalize(X_train)
        X_eval = l2_normalize(X_eval)
        score = X_eval @ X_train.T
        order = np.argsort(-score, axis=1)
    else:
        d = ((X_eval[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2)
        order = np.argsort(d, axis=1)

    preds = []
    confs = []

    for i in range(X_eval.shape[0]):
        idx = order[i, :k]
        votes = Counter(y_train[idx])
        best = sorted(votes.items(), key=lambda kv: (-kv[1], classes.index(kv[0])))[0][0]
        preds.append(best)
        confs.append(votes[best] / k)

    return np.array(preds), np.array(confs)


def fit_ridge(X_train, y_train, classes, lam):
    n, d = X_train.shape
    Xb = np.concatenate([np.ones((n, 1), dtype=np.float32), X_train], axis=1)

    counts = Counter(y_train)
    weights = np.array([n / (len(classes) * counts[y]) for y in y_train], dtype=np.float32)

    Y = np.zeros((n, len(classes)), dtype=np.float32)
    idx = {c: i for i, c in enumerate(classes)}
    for i, y in enumerate(y_train):
        Y[i, idx[y]] = 1.0

    sw = np.sqrt(weights)[:, None]
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


def predict_ridge(X_eval, W, classes):
    Xb = np.concatenate([np.ones((X_eval.shape[0], 1), dtype=np.float32), X_eval], axis=1)
    scores = Xb @ W

    idx = scores.argmax(axis=1)
    if scores.shape[1] > 1:
        conf = scores.max(axis=1) - np.partition(scores, -2, axis=1)[:, -2]
    else:
        conf = scores.max(axis=1)

    return np.array([classes[i] for i in idx]), conf


issues = []

for p in [EMB, V72A_DECISION, V69B_BEST]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required input missing for v72b.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v72b_decision": "frozen_embedding_baseline_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v72c_error_analysis": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


v72a = read_csv_clean(V72A_DECISION)
emb = read_csv_clean(EMB)
best69b = read_csv_clean(V69B_BEST)

if len(v72a) == 0 or not bool_true(v72a.iloc[0].get("ready_for_v72b_embedding_baseline", "")):
    issues.append({
        "item": str(V72A_DECISION),
        "issue_type": "hard_v72a_not_ready",
        "issue_detail": "v72a must be ready before v72b.",
        "severity": "hard",
    })

emb_cols = [c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]
classes = sorted(emb["behaviour_code"].unique().tolist())

if len(emb) != 744:
    issues.append({
        "item": "embedding_rows",
        "issue_type": "hard_unexpected_embedding_rows",
        "issue_detail": f"Expected 744, found {len(emb)}.",
        "severity": "hard",
    })

if len(emb_cols) <= 0:
    issues.append({
        "item": "embedding_dim",
        "issue_type": "hard_no_embedding_columns",
        "issue_detail": "No embedding columns found.",
        "severity": "hard",
    })

if any(i["severity"] == "hard" for i in issues):
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v72b_decision": "frozen_embedding_baseline_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v72c_error_analysis": False,
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

metric_models = [
    ("nearest_centroid", "cosine", None),
    ("nearest_centroid", "euclidean", None),
    ("knn", "cosine", 1),
    ("knn", "cosine", 3),
    ("knn", "euclidean", 1),
    ("knn", "euclidean", 3),
]

ridge_dims = [32, 64, 128, 256, "full"]
ridge_lambdas = [0.1, 1.0, 10.0, 100.0]

all_rows = []
selected_rows = []
pred_rows = []
per_class_frames = []
conf_frames = []

for policy_name, split_col in policies:
    for crop_type in crop_types:
        data = emb[emb["crop_type"] == crop_type].copy()

        train = data[data[split_col] == "train"].copy()
        val = data[data[split_col] == "val"].copy()
        test = data[data[split_col] == "test"].copy()

        X_train_raw = train[emb_cols].to_numpy(dtype=np.float32)
        y_train = train["behaviour_code"].to_numpy()

        split_data = {
            "train": train,
            "val": val,
            "test": test,
        }

        configs = []

        for model_type, metric, k in metric_models:
            X_train_std, X_val_std = standardize(X_train_raw, val[emb_cols].to_numpy(dtype=np.float32))
            y_val = val["behaviour_code"].to_numpy()

            if model_type == "nearest_centroid":
                y_pred, _ = predict_nearest_centroid(X_train_std, y_train, X_val_std, classes, metric)
                config_name = f"{model_type}_{metric}"
            else:
                y_pred, _ = predict_knn(X_train_std, y_train, X_val_std, classes, k, metric)
                config_name = f"{model_type}{k}_{metric}"

            m, _ = compute_metrics(y_val, y_pred, classes)
            configs.append({
                "config_name": config_name,
                "model_family": model_type,
                "metric": metric,
                "k": k if k is not None else "",
                "pca_dim": "",
                "lambda": "",
                "val_macro_f1": m["macro_f1"],
                "val_balanced_accuracy": m["balanced_accuracy"],
                "val_accuracy": m["accuracy"],
            })

        for dim in ridge_dims:
            for lam in ridge_lambdas:
                X_train_std, X_val_std = standardize(X_train_raw, val[emb_cols].to_numpy(dtype=np.float32))
                Xtr, Xv, used_dim = pca_fit_transform(X_train_std, X_val_std, dim)

                W = fit_ridge(Xtr, y_train, classes, lam)
                y_pred, _ = predict_ridge(Xv, W, classes)

                m, _ = compute_metrics(val["behaviour_code"].to_numpy(), y_pred, classes)
                configs.append({
                    "config_name": f"ridge_pca{used_dim}_lam{lam}",
                    "model_family": "class_balanced_ridge",
                    "metric": "",
                    "k": "",
                    "pca_dim": used_dim,
                    "lambda": lam,
                    "val_macro_f1": m["macro_f1"],
                    "val_balanced_accuracy": m["balanced_accuracy"],
                    "val_accuracy": m["accuracy"],
                })

        cfg = pd.DataFrame(configs).sort_values(
            ["val_macro_f1", "val_balanced_accuracy", "val_accuracy"],
            ascending=[False, False, False],
        ).iloc[0].to_dict()

        for eval_split, eval_df in split_data.items():
            X_train_std, X_eval_std = standardize(X_train_raw, eval_df[emb_cols].to_numpy(dtype=np.float32))
            y_eval = eval_df["behaviour_code"].to_numpy()

            if cfg["model_family"] == "nearest_centroid":
                y_pred, conf = predict_nearest_centroid(
                    X_train_std, y_train, X_eval_std, classes, clean(cfg["metric"])
                )
            elif cfg["model_family"] == "knn":
                y_pred, conf = predict_knn(
                    X_train_std, y_train, X_eval_std, classes, int(cfg["k"]), clean(cfg["metric"])
                )
            else:
                Xtr, Xev, used_dim = pca_fit_transform(X_train_std, X_eval_std, cfg["pca_dim"])
                W = fit_ridge(Xtr, y_train, classes, float(cfg["lambda"]))
                y_pred, conf = predict_ridge(Xev, W, classes)

            m, pc = compute_metrics(y_eval, y_pred, classes)

            metric_row = {
                "split_policy": policy_name,
                "crop_type": crop_type,
                "selected_config_name": clean(cfg["config_name"]),
                "model_family": clean(cfg["model_family"]),
                "metric": clean(cfg["metric"]),
                "k": clean(cfg["k"]),
                "pca_dim": clean(cfg["pca_dim"]),
                "lambda": clean(cfg["lambda"]),
                "eval_split": eval_split,
                "n_train": len(train),
                "n_eval": len(eval_df),
                "n_classes": len(classes),
                **m,
                "selection_rule": "validation_macro_f1_then_balanced_accuracy_then_accuracy",
            }

            selected_rows.append(metric_row)
            all_rows.append(metric_row)

            if eval_split == "test":
                pc.insert(0, "eval_split", eval_split)
                pc.insert(0, "selected_config_name", clean(cfg["config_name"]))
                pc.insert(0, "crop_type", crop_type)
                pc.insert(0, "split_policy", policy_name)
                per_class_frames.append(pc)

                cm = confusion_long(y_eval, y_pred, classes)
                cm.insert(0, "eval_split", eval_split)
                cm.insert(0, "selected_config_name", clean(cfg["config_name"]))
                cm.insert(0, "crop_type", crop_type)
                cm.insert(0, "split_policy", policy_name)
                conf_frames.append(cm)

                for i, (_, rr) in enumerate(eval_df.iterrows()):
                    pred_rows.append({
                        "split_policy": policy_name,
                        "crop_type": crop_type,
                        "selected_config_name": clean(cfg["config_name"]),
                        "model_family": clean(cfg["model_family"]),
                        "metric": clean(cfg["metric"]),
                        "k": clean(cfg["k"]),
                        "pca_dim": clean(cfg["pca_dim"]),
                        "lambda": clean(cfg["lambda"]),
                        "eval_split": "test",
                        "canonical_gt_object_id": clean(rr["canonical_gt_object_id"]),
                        "scan_frame_id": clean(rr["scan_frame_id"]),
                        "video_id": clean(rr["video_id"]),
                        "actual": clean(y_eval[i]),
                        "predicted": clean(y_pred[i]),
                        "correct": bool(y_eval[i] == y_pred[i]),
                        "confidence_like_score": float(conf[i]),
                        "crop_path": clean(rr["crop_path"]),
                    })

all_metrics = pd.DataFrame(all_rows)
selected_metrics = pd.DataFrame(selected_rows)
predictions = pd.DataFrame(pred_rows)
per_class = pd.concat(per_class_frames, ignore_index=True)
confusion = pd.concat(conf_frames, ignore_index=True)

safe_to_csv(all_metrics, OUT_ALL)
safe_to_csv(selected_metrics, OUT_SELECTED)
safe_to_csv(predictions, OUT_PRED)
safe_to_csv(per_class, OUT_PER_CLASS)
safe_to_csv(confusion, OUT_CONF)

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

    old_macro = float(old.iloc[0]["macro_f1"]) if len(old) else 0.0
    old_acc = float(old.iloc[0]["accuracy"]) if len(old) else 0.0

    compare_rows.append({
        "split_policy": policy,
        "v69b_champion_macro_f1": old_macro,
        "v72b_best_macro_f1": float(r["macro_f1"]),
        "delta_macro_f1": float(r["macro_f1"]) - old_macro,
        "v69b_champion_accuracy": old_acc,
        "v72b_best_accuracy": float(r["accuracy"]),
        "delta_accuracy": float(r["accuracy"]) - old_acc,
        "v72b_best_crop_type": clean(r["crop_type"]),
        "v72b_best_config": clean(r["selected_config_name"]),
        "v72b_beats_v69b_macro_f1": bool(float(r["macro_f1"]) > old_macro),
        "v72b_beats_v69b_accuracy": bool(float(r["accuracy"]) > old_acc),
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

add_qa("embedding_rows", 744, len(emb), len(emb) == 744, "hard", "Embedding rows should be 744.")
add_qa("embedding_dim", ">0", len(emb_cols), len(emb_cols) > 0, "hard", "Embedding dimension should be positive.")
add_qa("selected_metric_rows", 12, len(selected_metrics), len(selected_metrics) == 12, "hard", "2 policies x 2 crops x 3 eval splits.")
add_qa("test_prediction_rows", 300, len(predictions), len(predictions) == 300, "hard", "Current test 78*2 crops + group test 72*2 crops.")
add_qa("per_class_rows", 44, len(per_class), len(per_class) == 44, "hard", "2 policies x 2 crops x 11 classes.")
add_qa("confusion_rows", 484, len(confusion), len(confusion) == 484, "hard", "2 policies x 2 crops x 11x11 confusion rows.")

for _, r in compare.iterrows():
    add_qa(
        f"{r['split_policy']}_beats_v69b_macro_f1",
        True,
        r["v72b_beats_v69b_macro_f1"],
        bool(r["v72b_beats_v69b_macro_f1"]),
        "warning",
        "Frozen detector embedding baseline should ideally beat v69b macro-F1.",
    )

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_QA)

hard_quality_failures = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())
warning_quality_failures = int(((qa["severity"] == "warning") & (~qa["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v72b_quality_checks",
        "issue_type": "hard_embedding_baseline_quality_failed",
        "issue_detail": f"{hard_quality_failures} hard checks failed.",
        "severity": "hard",
    })

if warning_quality_failures:
    issues.append({
        "item": "v72b_model_strength",
        "issue_type": "warning_embedding_baseline_did_not_beat_v69b_for_all_policies",
        "issue_detail": f"{warning_quality_failures} macro-F1 improvement warnings.",
        "severity": "warning",
    })

issues.append({
    "item": "claim_scope",
    "issue_type": "info_frozen_embedding_baseline_only",
    "issue_detail": "v72b is a frozen embedding baseline only, not a production classifier.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

manifest = {
    "version": "week8_v72b_frozen_embedding_baseline",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "embedding_rows": int(len(emb)),
    "embedding_dim": int(len(emb_cols)),
    "best_test": best_test.to_dict(orient="records"),
    "comparison_vs_v69b": compare.to_dict(orient="records"),
    "claim_boundary": "frozen embedding baseline only; no production classifier claim",
}
OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

OUT_README.write_text(
    "# Week8 v72b Frozen Embedding Baseline\n\n"
    "Frozen YOLOv8 detector embeddings from v72a were used for baseline classification.\n\n"
    "Configurations were selected using validation macro-F1.\n\n"
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
    "v72b_decision": "frozen_embedding_baseline_completed" if hard_issue_count == 0 else "frozen_embedding_baseline_has_blocking_issues",
    "embedding_rows": int(len(emb)),
    "embedding_dim": int(len(emb_cols)),
    "current_best_crop_type": clean(cur["crop_type"]),
    "current_best_config": clean(cur["selected_config_name"]),
    "current_best_test_accuracy": float(cur["accuracy"]),
    "current_best_test_macro_f1": float(cur["macro_f1"]),
    "current_best_balanced_accuracy": float(cur["balanced_accuracy"]),
    "group_best_crop_type": clean(grp["crop_type"]),
    "group_best_config": clean(grp["selected_config_name"]),
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
    "ready_for_v72c_error_analysis": bool(hard_issue_count == 0),
    "claim_scope": "frozen_detector_embedding_baseline_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v72b Frozen Embedding Baseline\n\n"
    f"- v72b decision: {decision.iloc[0]['v72b_decision']}\n"
    f"- Embedding rows: {len(emb)}\n"
    f"- Embedding dimension: {len(emb_cols)}\n"
    f"- Current best: {cur['crop_type']} / {cur['selected_config_name']} / test macro-F1={float(cur['macro_f1']):.4f} / acc={float(cur['accuracy']):.4f}\n"
    f"- Group-aware best: {grp['crop_type']} / {grp['selected_config_name']} / test macro-F1={float(grp['macro_f1']):.4f} / acc={float(grp['accuracy']):.4f}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Warnings: {warning_count}\n"
    f"- Ready for v72c error analysis: {bool(hard_issue_count == 0)}\n\n"
    "v72b is a frozen detector embedding baseline only. No production classifier claim.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v72b Frozen Embedding Baseline Report\n\n"
    f"Decision: {decision.iloc[0]['v72b_decision']}\n\n"
    f"Current best macro-F1: {float(cur['macro_f1'])}\n\n"
    f"Group-aware best macro-F1: {float(grp['macro_f1'])}\n\n"
    f"ZIP: {OUT_ZIP}\n\n"
    f"SHA256: {zip_hash}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v72b",
    "task_name": "Frozen detector embedding baseline",
    "status": "PASS_WITH_WARNINGS" if hard_issue_count == 0 and warning_count > 0 else ("PASS" if hard_issue_count == 0 else "NEEDS_FIX"),
    "input_summary": str(EMB),
    "output_summary": str(PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Run v72c embedding error analysis/interface." if hard_issue_count == 0 else "Fix v72b baseline.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("=== v72b decision ===")
print(decision.to_string(index=False))
print("\n=== best test summary ===")
print(best_test.to_string(index=False))
print("\n=== comparison vs v69b ===")
print(compare.to_string(index=False))
print("\n=== issues ===")
print(issues_df.to_string(index=False) if len(issues_df) else "No issues found.")
