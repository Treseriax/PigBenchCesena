from pathlib import Path
from datetime import datetime
import csv
import json
import hashlib
import zipfile
import pandas as pd
import numpy as np


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V68A = W8 / "outputs" / "v68a_strict_gold_anchor_crop_materialization"
V68D = W8 / "outputs" / "v68d_group_aware_split_policy"

CURRENT_METADATA = V68A / "Week8_StrictGold_AnchorFrame_Crop_Dataset" / "metadata" / "week8_v68a_strict_gold_anchor_crop_metadata.csv"
GROUP_METADATA = V68D / "Week8_GroupAware_Split_Policy" / "week8_v68d_group_aware_crop_metadata.csv"
V68D_DECISION = V68D / "week8_v68d_decision_summary.csv"

OUT = W8 / "outputs" / "v69a_baseline_sanity"
PKG = OUT / "Week8_Baseline_Sanity"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_METRICS = PKG / "week8_v69a_baseline_sanity_metrics.csv"
OUT_PER_CLASS = PKG / "week8_v69a_baseline_sanity_per_class_metrics.csv"
OUT_CONFUSION = PKG / "week8_v69a_baseline_sanity_confusion_matrix_long.csv"
OUT_SPLIT_SUMMARY = PKG / "week8_v69a_baseline_split_summary.csv"
OUT_CLASS_DIST = PKG / "week8_v69a_baseline_class_distribution.csv"
OUT_RECOMMENDATIONS = PKG / "week8_v69a_baseline_recommendations.csv"
OUT_QA = PKG / "week8_v69a_baseline_sanity_quality_checks.csv"
OUT_MANIFEST = PKG / "week8_v69a_baseline_sanity_manifest.json"
OUT_README = PKG / "README_Week8_Baseline_Sanity.md"

OUT_DECISION = OUT / "week8_v69a_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v69a_issues.csv"
OUT_ZIP = OUT / "Week8_Baseline_Sanity.zip"
OUT_SHA256 = OUT / "Week8_Baseline_Sanity.sha256"
OUT_NOTE = NOTES / "week8_v69a_baseline_sanity_notes.md"
OUT_REPORT = REPORTS / "week8_v69a_baseline_sanity_report.md"
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


def compute_per_class_metrics(y_true, y_pred, classes):
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
    pc = compute_per_class_metrics(y_true, y_pred, classes)

    accuracy = float((np.array(y_true) == np.array(y_pred)).mean()) if len(y_true) else 0.0
    macro_precision = float(pc["precision"].mean()) if len(pc) else 0.0
    macro_recall = float(pc["recall"].mean()) if len(pc) else 0.0
    macro_f1 = float(pc["f1"].mean()) if len(pc) else 0.0

    total_support = pc["support"].sum()
    weighted_f1 = float((pc["f1"] * pc["support"]).sum() / total_support) if total_support else 0.0
    balanced_accuracy = macro_recall

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_accuracy,
    }, pc


def confusion_long(y_true, y_pred, classes):
    rows = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for actual in classes:
        for predicted in classes:
            rows.append({
                "actual": actual,
                "predicted": predicted,
                "count": int(((y_true == actual) & (y_pred == predicted)).sum()),
            })

    return pd.DataFrame(rows)


def evaluate_policy(df, split_col, policy_name, rng_seed=42, random_repeats=50):
    classes = sorted(df["behaviour_code"].unique().tolist())

    train = df[df[split_col] == "train"].copy()
    val = df[df[split_col] == "val"].copy()
    test = df[df[split_col] == "test"].copy()

    train_counts = train["behaviour_code"].value_counts()
    majority_class = train_counts.idxmax()
    train_prior = train_counts.reindex(classes, fill_value=0).astype(float)
    train_prior = train_prior / train_prior.sum()

    metrics_rows = []
    per_class_rows = []
    confusion_rows = []

    eval_sets = {
        "train": train,
        "val": val,
        "test": test,
    }

    for eval_split, eval_df in eval_sets.items():
        y_true = eval_df["behaviour_code"].to_numpy()

        # Majority baseline.
        y_pred_majority = np.array([majority_class] * len(eval_df))
        agg, pc = aggregate_metrics(y_true, y_pred_majority, classes)
        metrics_rows.append({
            "split_policy": policy_name,
            "eval_split": eval_split,
            "baseline_type": "majority_train_class",
            "majority_class": majority_class,
            "random_seed": "",
            "repeat_count": "",
            "n_samples": len(eval_df),
            **agg,
        })

        pc.insert(0, "baseline_type", "majority_train_class")
        pc.insert(0, "eval_split", eval_split)
        pc.insert(0, "split_policy", policy_name)
        per_class_rows.append(pc)

        cm = confusion_long(y_true, y_pred_majority, classes)
        cm.insert(0, "baseline_type", "majority_train_class")
        cm.insert(0, "eval_split", eval_split)
        cm.insert(0, "split_policy", policy_name)
        confusion_rows.append(cm)

        # Train-prior random baseline, repeated.
        rng = np.random.default_rng(rng_seed)
        repeat_metrics = []
        repeat_pc_frames = []

        for rep in range(random_repeats):
            y_pred_random = rng.choice(classes, size=len(eval_df), replace=True, p=train_prior.values)
            agg_r, pc_r = aggregate_metrics(y_true, y_pred_random, classes)
            agg_r["repeat"] = rep
            repeat_metrics.append(agg_r)

            pc_r["repeat"] = rep
            repeat_pc_frames.append(pc_r)

        rep_df = pd.DataFrame(repeat_metrics)

        mean_row = {
            "split_policy": policy_name,
            "eval_split": eval_split,
            "baseline_type": "train_prior_random_mean",
            "majority_class": majority_class,
            "random_seed": rng_seed,
            "repeat_count": random_repeats,
            "n_samples": len(eval_df),
        }

        std_row = {
            "split_policy": policy_name,
            "eval_split": eval_split,
            "baseline_type": "train_prior_random_std",
            "majority_class": majority_class,
            "random_seed": rng_seed,
            "repeat_count": random_repeats,
            "n_samples": len(eval_df),
        }

        for m in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "balanced_accuracy"]:
            mean_row[m] = float(rep_df[m].mean())
            std_row[m] = float(rep_df[m].std(ddof=0))

        metrics_rows.append(mean_row)
        metrics_rows.append(std_row)

        pc_all = pd.concat(repeat_pc_frames, ignore_index=True)
        pc_mean = pc_all.groupby("class", as_index=False)[["support", "tp", "fp", "fn", "tn", "precision", "recall", "f1"]].mean()
        pc_mean.insert(0, "baseline_type", "train_prior_random_mean")
        pc_mean.insert(0, "eval_split", eval_split)
        pc_mean.insert(0, "split_policy", policy_name)
        per_class_rows.append(pc_mean)

        pc_std = pc_all.groupby("class", as_index=False)[["support", "tp", "fp", "fn", "tn", "precision", "recall", "f1"]].std(ddof=0).fillna(0)
        pc_std.insert(0, "baseline_type", "train_prior_random_std")
        pc_std.insert(0, "eval_split", eval_split)
        pc_std.insert(0, "split_policy", policy_name)
        per_class_rows.append(pc_std)

    split_summary = []
    for sname, sdf in eval_sets.items():
        split_summary.append({
            "split_policy": policy_name,
            "split": sname,
            "object_count": len(sdf),
            "scanframe_count": sdf["scan_frame_id"].nunique(),
            "video_count": sdf["video_id"].nunique(),
            "class_count": sdf["behaviour_code"].nunique(),
            "classes": ";".join(sorted(sdf["behaviour_code"].unique().tolist())),
        })

    class_dist = (
        df.groupby([split_col, "behaviour_code"])
        .size()
        .reset_index(name="object_count")
        .rename(columns={split_col: "split"})
    )
    class_dist.insert(0, "split_policy", policy_name)

    return {
        "metrics": pd.DataFrame(metrics_rows),
        "per_class": pd.concat(per_class_rows, ignore_index=True) if per_class_rows else pd.DataFrame(),
        "confusion": pd.concat(confusion_rows, ignore_index=True) if confusion_rows else pd.DataFrame(),
        "split_summary": pd.DataFrame(split_summary),
        "class_dist": class_dist,
    }


issues = []

for p in [CURRENT_METADATA, GROUP_METADATA, V68D_DECISION]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required input missing for v69a baseline sanity.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v69a_decision": "baseline_sanity_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v69b_feature_baseline": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


v68d_decision = read_csv_clean(V68D_DECISION)
if len(v68d_decision) == 0 or not bool_true(v68d_decision.iloc[0].get("ready_for_v69a_baseline_sanity", "")):
    issues.append({
        "item": str(V68D_DECISION),
        "issue_type": "hard_v68d_not_ready",
        "issue_detail": "v68d must be ready before v69a.",
        "severity": "hard",
    })

current = read_csv_clean(CURRENT_METADATA)
group = read_csv_clean(GROUP_METADATA)

if len(current) != 372:
    issues.append({
        "item": "current_metadata_rows",
        "issue_type": "hard_current_metadata_row_count_unexpected",
        "issue_detail": f"Expected 372 rows, found {len(current)}.",
        "severity": "hard",
    })

if len(group) != 372:
    issues.append({
        "item": "group_metadata_rows",
        "issue_type": "hard_group_metadata_row_count_unexpected",
        "issue_detail": f"Expected 372 rows, found {len(group)}.",
        "severity": "hard",
    })

if "recommended_split" not in current.columns:
    issues.append({
        "item": "recommended_split",
        "issue_type": "hard_missing_current_split_column",
        "issue_detail": "Current metadata missing recommended_split.",
        "severity": "hard",
    })

if "group_aware_split" not in group.columns:
    issues.append({
        "item": "group_aware_split",
        "issue_type": "hard_missing_group_split_column",
        "issue_detail": "Group metadata missing group_aware_split.",
        "severity": "hard",
    })

if any(i["severity"] == "hard" for i in issues):
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v69a_decision": "baseline_sanity_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v69b_feature_baseline": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    print(issues_df.to_string(index=False))
    raise SystemExit(1)


current_eval = evaluate_policy(current, "recommended_split", "current_recommended_split")
group_eval = evaluate_policy(group, "group_aware_split", "selected_group_aware_split")

metrics = pd.concat([current_eval["metrics"], group_eval["metrics"]], ignore_index=True)
per_class = pd.concat([current_eval["per_class"], group_eval["per_class"]], ignore_index=True)
confusion = pd.concat([current_eval["confusion"], group_eval["confusion"]], ignore_index=True)
split_summary = pd.concat([current_eval["split_summary"], group_eval["split_summary"]], ignore_index=True)
class_dist = pd.concat([current_eval["class_dist"], group_eval["class_dist"]], ignore_index=True)

safe_to_csv(metrics, OUT_METRICS)
safe_to_csv(per_class, OUT_PER_CLASS)
safe_to_csv(confusion, OUT_CONFUSION)
safe_to_csv(split_summary, OUT_SPLIT_SUMMARY)
safe_to_csv(class_dist, OUT_CLASS_DIST)

# Extract key metrics.
def get_metric(policy, split, baseline, metric):
    row = metrics[
        (metrics["split_policy"] == policy)
        & (metrics["eval_split"] == split)
        & (metrics["baseline_type"] == baseline)
    ]
    if len(row) == 0:
        return ""
    return float(row.iloc[0][metric])


current_test_majority_acc = get_metric("current_recommended_split", "test", "majority_train_class", "accuracy")
current_test_majority_macro_f1 = get_metric("current_recommended_split", "test", "majority_train_class", "macro_f1")
group_test_majority_acc = get_metric("selected_group_aware_split", "test", "majority_train_class", "accuracy")
group_test_majority_macro_f1 = get_metric("selected_group_aware_split", "test", "majority_train_class", "macro_f1")

current_test_random_macro_f1 = get_metric("current_recommended_split", "test", "train_prior_random_mean", "macro_f1")
group_test_random_macro_f1 = get_metric("selected_group_aware_split", "test", "train_prior_random_mean", "macro_f1")

recommendations = pd.DataFrame([
    {
        "topic": "baseline_interpretation",
        "recommendation": "Any learned model must beat majority and train-prior random baselines, especially macro-F1.",
        "severity": "required",
    },
    {
        "topic": "primary_metric",
        "recommendation": "Use macro-F1, balanced accuracy, and per-class recall; accuracy alone is misleading because STI is dominant.",
        "severity": "required",
    },
    {
        "topic": "current_split",
        "recommendation": "Current split can be used for continuity/exploratory results but source-video overlap must be reported.",
        "severity": "warning",
    },
    {
        "topic": "group_aware_split",
        "recommendation": "Group-aware split should be used as conservative sensitivity/generalization check.",
        "severity": "recommended",
    },
    {
        "topic": "rare_classes",
        "recommendation": "Do not claim reliable standalone metrics for BE, DE, IA due very small support.",
        "severity": "required_limitation",
    },
    {
        "topic": "next_step",
        "recommendation": "Proceed to v69b feature baseline using strict-gold crops only.",
        "severity": "ready",
    },
])
safe_to_csv(recommendations, OUT_RECOMMENDATIONS)

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


add_qa("current_rows", 372, len(current), len(current) == 372, "hard", "Current split metadata should have 372 rows.")
add_qa("group_rows", 372, len(group), len(group) == 372, "hard", "Group-aware split metadata should have 372 rows.")
add_qa("current_split_count", 3, current["recommended_split"].nunique(), current["recommended_split"].nunique() == 3, "hard", "Current split should include train/val/test.")
add_qa("group_split_count", 3, group["group_aware_split"].nunique(), group["group_aware_split"].nunique() == 3, "hard", "Group-aware split should include train/val/test.")
add_qa("current_class_count", 11, current["behaviour_code"].nunique(), current["behaviour_code"].nunique() == 11, "hard", "Current metadata should have 11 classes.")
add_qa("group_class_count", 11, group["behaviour_code"].nunique(), group["behaviour_code"].nunique() == 11, "hard", "Group metadata should have 11 classes.")
add_qa("metrics_rows_created", ">0", len(metrics), len(metrics) > 0, "hard", "Metrics table should be created.")
add_qa("per_class_rows_created", ">0", len(per_class), len(per_class) > 0, "hard", "Per-class metrics table should be created.")
add_qa("confusion_rows_created", ">0", len(confusion), len(confusion) > 0, "hard", "Confusion matrix table should be created.")
add_qa("current_test_majority_macro_f1_recorded", "nonempty", current_test_majority_macro_f1, current_test_majority_macro_f1 != "", "hard", "Current test majority macro-F1 should be recorded.")
add_qa("group_test_majority_macro_f1_recorded", "nonempty", group_test_majority_macro_f1, group_test_majority_macro_f1 != "", "hard", "Group-aware test majority macro-F1 should be recorded.")

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_QA)

hard_quality_failures = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v69a_quality_checks",
        "issue_type": "hard_baseline_sanity_quality_check_failed",
        "issue_detail": f"{hard_quality_failures} hard baseline sanity checks failed.",
        "severity": "hard",
    })

# Informational limitations.
issues.append({
    "item": "current_split",
    "issue_type": "info_current_split_is_exploratory_due_source_video_overlap",
    "issue_detail": "Use current split for continuity/exploratory baseline only.",
    "severity": "info",
})

issues.append({
    "item": "rare_classes",
    "issue_type": "info_rare_class_limitation",
    "issue_detail": "BE, DE, IA have very small support and should not receive strong standalone claims.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

manifest = {
    "version": "week8_v69a_baseline_sanity",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "baseline_types": ["majority_train_class", "train_prior_random_mean", "train_prior_random_std"],
    "random_seed": 42,
    "random_repeats": 50,
    "current_split": {
        "test_majority_accuracy": current_test_majority_acc,
        "test_majority_macro_f1": current_test_majority_macro_f1,
        "test_random_macro_f1_mean": current_test_random_macro_f1,
    },
    "group_aware_split": {
        "test_majority_accuracy": group_test_majority_acc,
        "test_majority_macro_f1": group_test_majority_macro_f1,
        "test_random_macro_f1_mean": group_test_random_macro_f1,
    },
    "claim_boundary": "sanity baselines only; no image model trained in v69a",
}

OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

readme = f"""# Week8 v69a Baseline Sanity

## Purpose

This package provides non-image sanity baselines before any learned crop classifier is trained.

## Baselines

- Majority train class baseline
- Train-prior random baseline, 50 repeats, seed 42

## Current Split Test Sanity

- Majority accuracy: {current_test_majority_acc}
- Majority macro-F1: {current_test_majority_macro_f1}
- Train-prior random macro-F1 mean: {current_test_random_macro_f1}

## Group-Aware Split Test Sanity

- Majority accuracy: {group_test_majority_acc}
- Majority macro-F1: {group_test_majority_macro_f1}
- Train-prior random macro-F1 mean: {group_test_random_macro_f1}

## Usage

A learned model should beat these baselines, especially on macro-F1 and balanced accuracy.

Accuracy alone is not enough because STI is dominant.

## Claim Boundary

No image model is trained in v69a. These are only sanity baselines.
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

decision = pd.DataFrame([{
    "v69a_decision": "baseline_sanity_completed" if hard_issue_count == 0 else "baseline_sanity_has_blocking_issues",
    "current_rows": int(len(current)),
    "group_rows": int(len(group)),
    "behaviour_classes": int(current["behaviour_code"].nunique()),
    "current_test_majority_accuracy": current_test_majority_acc,
    "current_test_majority_macro_f1": current_test_majority_macro_f1,
    "current_test_train_prior_random_macro_f1_mean": current_test_random_macro_f1,
    "group_test_majority_accuracy": group_test_majority_acc,
    "group_test_majority_macro_f1": group_test_majority_macro_f1,
    "group_test_train_prior_random_macro_f1_mean": group_test_random_macro_f1,
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_quality_failures": hard_quality_failures,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v69b_feature_baseline": bool(hard_issue_count == 0),
    "claim_scope": "sanity_baselines_only_no_image_model_trained",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v69a Baseline Sanity\n\n"
    f"- v69a decision: {decision.iloc[0]['v69a_decision']}\n"
    f"- Behaviour classes: {current['behaviour_code'].nunique()}\n"
    f"- Current test majority accuracy: {current_test_majority_acc:.4f}\n"
    f"- Current test majority macro-F1: {current_test_majority_macro_f1:.4f}\n"
    f"- Current test random macro-F1 mean: {current_test_random_macro_f1:.4f}\n"
    f"- Group-aware test majority accuracy: {group_test_majority_acc:.4f}\n"
    f"- Group-aware test majority macro-F1: {group_test_majority_macro_f1:.4f}\n"
    f"- Group-aware test random macro-F1 mean: {group_test_random_macro_f1:.4f}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Ready for v69b feature baseline: {bool(hard_issue_count == 0)}\n\n"
    "No image model is trained here. These are sanity baselines only.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v69a Baseline Sanity Report\n\n"
    f"Decision: {decision.iloc[0]['v69a_decision']}\n\n"
    f"Current test majority macro-F1: {current_test_majority_macro_f1}\n\n"
    f"Group-aware test majority macro-F1: {group_test_majority_macro_f1}\n\n"
    f"ZIP: {OUT_ZIP}\n\n"
    f"SHA256: {zip_hash}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v69a",
    "task_name": "Baseline sanity checks",
    "status": "PASS" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": f"{CURRENT_METADATA}; {GROUP_METADATA}",
    "output_summary": str(PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Run v69b feature baseline using strict-gold crops." if hard_issue_count == 0 else "Fix v69a sanity baseline issues.",
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
print(OUT_SPLIT_SUMMARY)
print(OUT_CLASS_DIST)
print(OUT_RECOMMENDATIONS)
print(OUT_QA)
print(OUT_ZIP)
print(OUT_SHA256)
print(OUT_DECISION)
print(OUT_NOTE)

print()
print("=== v69a decision ===")
print(decision.to_string(index=False))

print()
print("=== metrics ===")
print(metrics.to_string(index=False))

print()
print("=== recommendations ===")
print(recommendations.to_string(index=False))

print()
print("=== QA ===")
print(qa.to_string(index=False))

print()
print("=== issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
