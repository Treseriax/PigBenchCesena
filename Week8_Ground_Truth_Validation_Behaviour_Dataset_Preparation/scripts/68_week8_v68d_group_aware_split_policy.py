from pathlib import Path
from datetime import datetime
import csv
import json
import hashlib
import zipfile
import itertools
import pandas as pd
import numpy as np


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V68A = W8 / "outputs" / "v68a_strict_gold_anchor_crop_materialization"
V68B = W8 / "outputs" / "v68b_crop_qa_gallery"
V68C = W8 / "outputs" / "v68c_split_leakage_audit"

DATASET = V68A / "Week8_StrictGold_AnchorFrame_Crop_Dataset"
METADATA = DATASET / "metadata" / "week8_v68a_strict_gold_anchor_crop_metadata.csv"

V68B_DECISION = V68B / "week8_v68b_decision_summary.csv"
V68C_DECISION = V68C / "week8_v68c_decision_summary.csv"

OUT = W8 / "outputs" / "v68d_group_aware_split_policy"
PKG = OUT / "Week8_GroupAware_Split_Policy"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_VIDEO_GROUPS = PKG / "week8_v68d_source_video_group_statistics.csv"
OUT_CANDIDATES = PKG / "week8_v68d_top_group_aware_split_candidates.csv"
OUT_ASSIGNMENT = PKG / "week8_v68d_selected_group_aware_video_assignment.csv"
OUT_METADATA = PKG / "week8_v68d_group_aware_crop_metadata.csv"
OUT_SPLIT_COUNTS = PKG / "week8_v68d_group_aware_split_counts.csv"
OUT_CLASS_COVERAGE = PKG / "week8_v68d_group_aware_class_coverage.csv"
OUT_COMPARISON = PKG / "week8_v68d_current_vs_group_aware_split_comparison.csv"
OUT_POLICY = PKG / "week8_v68d_split_policy_recommendations.csv"
OUT_QA = PKG / "week8_v68d_group_aware_split_quality_checks.csv"
OUT_MANIFEST = PKG / "week8_v68d_group_aware_split_manifest.json"
OUT_README = PKG / "README_Week8_GroupAware_Split_Policy.md"

OUT_DECISION = OUT / "week8_v68d_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v68d_issues.csv"
OUT_ZIP = OUT / "Week8_GroupAware_Split_Policy.zip"
OUT_SHA256 = OUT / "Week8_GroupAware_Split_Policy.sha256"
OUT_NOTE = NOTES / "week8_v68d_group_aware_split_policy_notes.md"
OUT_REPORT = REPORTS / "week8_v68d_group_aware_split_policy_report.md"
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


issues = []

for p in [METADATA, V68B_DECISION, V68C_DECISION]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required input missing for v68d group-aware split policy.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v68d_decision": "group_aware_split_policy_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v69a_baseline_sanity": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


v68b_decision = read_csv_clean(V68B_DECISION)
v68c_decision = read_csv_clean(V68C_DECISION)

if len(v68b_decision) == 0 or not bool_true(v68b_decision.iloc[0].get("ready_for_v68c_split_leakage_audit", "")):
    issues.append({
        "item": str(V68B_DECISION),
        "issue_type": "hard_v68b_not_ready",
        "issue_detail": "v68b must be ready before v68d.",
        "severity": "hard",
    })

if len(v68c_decision) == 0 or not bool_true(v68c_decision.iloc[0].get("ready_for_v69a_baseline_sanity", "")):
    issues.append({
        "item": str(V68C_DECISION),
        "issue_type": "hard_v68c_not_ready",
        "issue_detail": "v68c must be ready before v68d.",
        "severity": "hard",
    })

metadata = read_csv_clean(METADATA)

required_cols = [
    "canonical_gt_object_id",
    "scan_frame_id",
    "video_id",
    "behaviour_code",
    "recommended_split",
    "tight_crop_path",
    "context10_crop_path",
]

for c in required_cols:
    if c not in metadata.columns:
        issues.append({
            "item": c,
            "issue_type": "hard_missing_required_metadata_column",
            "issue_detail": "Required metadata column missing.",
            "severity": "hard",
        })

if any(i["severity"] == "hard" for i in issues):
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v68d_decision": "group_aware_split_policy_blocked",
        "metadata_rows": int(len(metadata)),
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v69a_baseline_sanity": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    print(issues_df.to_string(index=False))
    raise SystemExit(1)


splits = ["train", "val", "test"]
videos = sorted(metadata["video_id"].unique().tolist())
classes = sorted(metadata["behaviour_code"].unique().tolist())

video_class = (
    metadata.groupby(["video_id", "behaviour_code"])
    .size()
    .unstack(fill_value=0)
    .reindex(index=videos, columns=classes, fill_value=0)
    .astype(int)
)

video_total = video_class.sum(axis=1).astype(int)
video_scanframes = metadata.groupby("video_id")["scan_frame_id"].nunique().reindex(videos).fillna(0).astype(int)

video_stats = video_class.copy()
video_stats.insert(0, "scanframe_count", video_scanframes)
video_stats.insert(0, "object_count", video_total)
video_stats = video_stats.reset_index()

safe_to_csv(video_stats, OUT_VIDEO_GROUPS)

current_split_counts = metadata.groupby("recommended_split").size().to_dict()
target_counts = {
    "train": int(current_split_counts.get("train", round(len(metadata) * 0.68))),
    "val": int(current_split_counts.get("val", round(len(metadata) * 0.10))),
    "test": int(current_split_counts.get("test", round(len(metadata) * 0.21))),
}

major_classes = sorted(metadata.groupby("behaviour_code").size()[lambda s: s >= 10].index.tolist())
rare_classes = sorted(metadata.groupby("behaviour_code").size()[lambda s: s < 10].index.tolist())

class_index = {c: i for i, c in enumerate(classes)}
major_idx = [class_index[c] for c in major_classes]
rare_idx = [class_index[c] for c in rare_classes]

video_arrays = video_class.values.astype(int)
video_object_counts = video_total.values.astype(int)

best_candidates = []

def class_missing_names(counts):
    return [classes[i] for i, v in enumerate(counts) if int(v) == 0]


def major_missing_names(counts):
    return [classes[i] for i in major_idx if int(counts[i]) == 0]


def rare_missing_names(counts):
    return [classes[i] for i in rare_idx if int(counts[i]) == 0]


counter = 0

for assign_tuple in itertools.product(splits, repeat=len(videos)):
    if len(set(assign_tuple)) < 3:
        continue

    split_class_counts = {s: np.zeros(len(classes), dtype=int) for s in splits}
    split_object_counts = {s: 0 for s in splits}
    split_group_counts = {s: 0 for s in splits}

    for i, split in enumerate(assign_tuple):
        split_class_counts[split] += video_arrays[i]
        split_object_counts[split] += int(video_object_counts[i])
        split_group_counts[split] += 1

    if min(split_object_counts.values()) <= 0:
        continue

    train_missing = class_missing_names(split_class_counts["train"])
    val_missing = class_missing_names(split_class_counts["val"])
    test_missing = class_missing_names(split_class_counts["test"])

    val_major_missing = major_missing_names(split_class_counts["val"])
    test_major_missing = major_missing_names(split_class_counts["test"])

    val_rare_missing = rare_missing_names(split_class_counts["val"])
    test_rare_missing = rare_missing_names(split_class_counts["test"])

    size_penalty = 0.0
    for s in splits:
        t = max(target_counts[s], 1)
        size_penalty += ((split_object_counts[s] - t) / t) ** 2

    # Prefer roughly 8/2/2 videos, but not as strongly as class coverage.
    group_target = {"train": 8, "val": 2, "test": 2}
    group_penalty = sum((split_group_counts[s] - group_target[s]) ** 2 for s in splits)

    score = 0.0
    score += 100000.0 * len(train_missing)
    score += 10000.0 * (len(val_major_missing) + len(test_major_missing))
    score += 1000.0 * (len(val_missing) + len(test_missing))
    score += 150.0 * (len(val_rare_missing) + len(test_rare_missing))
    score += 100.0 * size_penalty
    score += 10.0 * group_penalty

    candidate = {
        "candidate_rank": 0,
        "score": float(score),
        "train_objects": split_object_counts["train"],
        "val_objects": split_object_counts["val"],
        "test_objects": split_object_counts["test"],
        "train_video_count": split_group_counts["train"],
        "val_video_count": split_group_counts["val"],
        "test_video_count": split_group_counts["test"],
        "train_missing_classes": ";".join(train_missing),
        "val_missing_classes": ";".join(val_missing),
        "test_missing_classes": ";".join(test_missing),
        "val_major_missing_classes": ";".join(val_major_missing),
        "test_major_missing_classes": ";".join(test_major_missing),
        "val_rare_missing_classes": ";".join(val_rare_missing),
        "test_rare_missing_classes": ";".join(test_rare_missing),
        "assignment": json.dumps({videos[i]: assign_tuple[i] for i in range(len(videos))}, ensure_ascii=False),
    }

    best_candidates.append(candidate)
    counter += 1

    if len(best_candidates) > 2000:
        best_candidates = sorted(best_candidates, key=lambda x: x["score"])[:200]

best_candidates = sorted(best_candidates, key=lambda x: x["score"])[:100]

if not best_candidates:
    issues.append({
        "item": "group_aware_search",
        "issue_type": "hard_no_group_aware_candidate_found",
        "issue_detail": "No group-aware split candidate found.",
        "severity": "hard",
    })

    candidates_df = pd.DataFrame()
    safe_to_csv(candidates_df, OUT_CANDIDATES)
else:
    for i, c in enumerate(best_candidates, start=1):
        c["candidate_rank"] = i

    candidates_df = pd.DataFrame(best_candidates)
    safe_to_csv(candidates_df, OUT_CANDIDATES)

selected = best_candidates[0] if best_candidates else None

if selected is None:
    selected_assignment = {}
else:
    selected_assignment = json.loads(selected["assignment"])

assignment_rows = []
for video_id in videos:
    row = {
        "video_id": video_id,
        "group_aware_split": selected_assignment.get(video_id, ""),
        "object_count": int(video_total.loc[video_id]),
        "scanframe_count": int(video_scanframes.loc[video_id]),
    }
    for c in classes:
        row[f"class_{c}"] = int(video_class.loc[video_id, c])
    assignment_rows.append(row)

assignment_df = pd.DataFrame(assignment_rows)
safe_to_csv(assignment_df, OUT_ASSIGNMENT)

group_metadata = metadata.copy()
group_metadata["group_aware_split"] = group_metadata["video_id"].map(selected_assignment)
group_metadata["current_split"] = group_metadata["recommended_split"]
group_metadata["split_policy_note"] = "group_aware_split_has_no_source_video_overlap"
safe_to_csv(group_metadata, OUT_METADATA)

group_split_counts = (
    group_metadata.groupby("group_aware_split")
    .size()
    .reset_index(name="object_count")
)
group_split_counts = group_split_counts.merge(
    group_metadata.groupby("group_aware_split")["scan_frame_id"].nunique().reset_index(name="scanframe_count"),
    on="group_aware_split",
    how="left",
)
group_split_counts = group_split_counts.merge(
    group_metadata.groupby("group_aware_split")["video_id"].nunique().reset_index(name="source_video_count"),
    on="group_aware_split",
    how="left",
)
group_split_counts = group_split_counts.merge(
    group_metadata.groupby("group_aware_split")["behaviour_code"].nunique().reset_index(name="behaviour_class_count"),
    on="group_aware_split",
    how="left",
)
safe_to_csv(group_split_counts, OUT_SPLIT_COUNTS)

current_class = (
    metadata.groupby(["behaviour_code", "recommended_split"])
    .size()
    .unstack(fill_value=0)
    .reindex(index=classes, columns=splits, fill_value=0)
    .astype(int)
)

group_class = (
    group_metadata.groupby(["behaviour_code", "group_aware_split"])
    .size()
    .unstack(fill_value=0)
    .reindex(index=classes, columns=splits, fill_value=0)
    .astype(int)
)

coverage_rows = []
for c in classes:
    row = {
        "behaviour_code": c,
        "total": int(metadata[metadata["behaviour_code"] == c].shape[0]),
        "is_rare_lt10": bool(metadata[metadata["behaviour_code"] == c].shape[0] < 10),
        "current_train": int(current_class.loc[c, "train"]),
        "current_val": int(current_class.loc[c, "val"]),
        "current_test": int(current_class.loc[c, "test"]),
        "group_train": int(group_class.loc[c, "train"]),
        "group_val": int(group_class.loc[c, "val"]),
        "group_test": int(group_class.loc[c, "test"]),
    }

    row["group_present_in_train"] = row["group_train"] > 0
    row["group_present_in_val"] = row["group_val"] > 0
    row["group_present_in_test"] = row["group_test"] > 0
    row["group_present_in_all_splits"] = row["group_present_in_train"] and row["group_present_in_val"] and row["group_present_in_test"]
    coverage_rows.append(row)

coverage_df = pd.DataFrame(coverage_rows)
safe_to_csv(coverage_df, OUT_CLASS_COVERAGE)

current_video_split = (
    metadata.groupby("video_id")["recommended_split"]
    .agg(lambda s: ";".join(sorted(set(s))))
    .reset_index()
)
current_video_split["split_count"] = current_video_split["recommended_split"].map(lambda s: len([x for x in s.split(";") if x]))
current_video_overlap = int((current_video_split["split_count"] > 1).sum())

group_video_split = (
    group_metadata.groupby("video_id")["group_aware_split"]
    .agg(lambda s: ";".join(sorted(set(s))))
    .reset_index()
)
group_video_split["split_count"] = group_video_split["group_aware_split"].map(lambda s: len([x for x in s.split(";") if x]))
group_video_overlap = int((group_video_split["split_count"] > 1).sum())

comparison = pd.DataFrame([
    {
        "policy": "current_recommended_split",
        "source_video_overlap_count": current_video_overlap,
        "train_objects": int(current_split_counts.get("train", 0)),
        "val_objects": int(current_split_counts.get("val", 0)),
        "test_objects": int(current_split_counts.get("test", 0)),
        "classes_missing_from_val": int((current_class["val"] == 0).sum()),
        "classes_missing_from_test": int((current_class["test"] == 0).sum()),
        "claim_scope": "exploratory_object_level_baseline_with_source_video_overlap_warning",
    },
    {
        "policy": "selected_group_aware_split",
        "source_video_overlap_count": group_video_overlap,
        "train_objects": int(group_split_counts.loc[group_split_counts["group_aware_split"] == "train", "object_count"].sum()),
        "val_objects": int(group_split_counts.loc[group_split_counts["group_aware_split"] == "val", "object_count"].sum()),
        "test_objects": int(group_split_counts.loc[group_split_counts["group_aware_split"] == "test", "object_count"].sum()),
        "classes_missing_from_val": int((group_class["val"] == 0).sum()),
        "classes_missing_from_test": int((group_class["test"] == 0).sum()),
        "claim_scope": "conservative_source_video_group_aware_baseline_with_class_coverage_limitations",
    },
])
safe_to_csv(comparison, OUT_COMPARISON)

group_train_missing = coverage_df[coverage_df["group_train"] == 0]["behaviour_code"].tolist()
group_val_missing = coverage_df[coverage_df["group_val"] == 0]["behaviour_code"].tolist()
group_test_missing = coverage_df[coverage_df["group_test"] == 0]["behaviour_code"].tolist()

major_set = set(major_classes)
group_val_major_missing = [c for c in group_val_missing if c in major_set]
group_test_major_missing = [c for c in group_test_missing if c in major_set]

policy_rows = [
    {
        "policy_item": "current_split",
        "recommendation": "Keep as continuity/exploratory baseline split. Report source-video overlap limitation.",
        "status": "usable_with_warning",
    },
    {
        "policy_item": "group_aware_split",
        "recommendation": "Use as conservative source-video-aware evaluation split. Report missing-class limitations.",
        "status": "usable_with_warning" if len(group_train_missing) == 0 else "blocked_if_training_all_classes_required",
    },
    {
        "policy_item": "baseline_reporting",
        "recommendation": "Report both current split and group-aware split separately. Do not merge metrics.",
        "status": "recommended",
    },
    {
        "policy_item": "rare_classes",
        "recommendation": "Do not claim reliable standalone performance for rare classes under 10 examples.",
        "status": "required_limitation",
    },
    {
        "policy_item": "primary_next_step",
        "recommendation": "Run v69a sanity baselines on current split first, and include group-aware split as an additional conservative audit if feasible.",
        "status": "ready" if len(group_train_missing) == 0 else "needs_review",
    },
]
policy_df = pd.DataFrame(policy_rows)
safe_to_csv(policy_df, OUT_POLICY)

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


add_qa("metadata_rows", 372, len(metadata), len(metadata) == 372, "hard", "Input crop metadata should contain 372 rows.")
add_qa("source_video_count", 12, len(videos), len(videos) == 12, "hard", "Expected 12 source video groups.")
add_qa("group_assignment_complete", len(videos), int(assignment_df["group_aware_split"].astype(str).str.strip().ne("").sum()), int(assignment_df["group_aware_split"].astype(str).str.strip().ne("").sum()) == len(videos), "hard", "Every source video should receive a group-aware split.")
add_qa("group_metadata_rows", len(metadata), len(group_metadata), len(group_metadata) == len(metadata), "hard", "Group-aware metadata should preserve all rows.")
add_qa("group_source_video_overlap", 0, group_video_overlap, group_video_overlap == 0, "hard", "Group-aware split must have zero source-video overlap.")
add_qa("group_train_not_empty", ">0", int(group_split_counts.loc[group_split_counts["group_aware_split"] == "train", "object_count"].sum()), int(group_split_counts.loc[group_split_counts["group_aware_split"] == "train", "object_count"].sum()) > 0, "hard", "Group train split should not be empty.")
add_qa("group_val_not_empty", ">0", int(group_split_counts.loc[group_split_counts["group_aware_split"] == "val", "object_count"].sum()), int(group_split_counts.loc[group_split_counts["group_aware_split"] == "val", "object_count"].sum()) > 0, "hard", "Group val split should not be empty.")
add_qa("group_test_not_empty", ">0", int(group_split_counts.loc[group_split_counts["group_aware_split"] == "test", "object_count"].sum()), int(group_split_counts.loc[group_split_counts["group_aware_split"] == "test", "object_count"].sum()) > 0, "hard", "Group test split should not be empty.")
add_qa("group_train_contains_all_classes", 0, len(group_train_missing), len(group_train_missing) == 0, "hard", "Training split should contain all behaviour classes for multiclass baseline.")
add_qa("current_source_video_overlap_warning", 0, current_video_overlap, current_video_overlap == 0, "warning", "Current split has source-video overlap and should be reported as exploratory.")
add_qa("group_val_major_missing", 0, len(group_val_major_missing), len(group_val_major_missing) == 0, "warning", "Major classes missing from group-aware val limit validation metrics.")
add_qa("group_test_major_missing", 0, len(group_test_major_missing), len(group_test_major_missing) == 0, "warning", "Major classes missing from group-aware test limit test metrics.")
add_qa("group_val_all_class_coverage", 0, len(group_val_missing), len(group_val_missing) == 0, "info", "Some classes may be missing from group-aware val due small data.")
add_qa("group_test_all_class_coverage", 0, len(group_test_missing), len(group_test_missing) == 0, "info", "Some classes may be missing from group-aware test due small data.")

qa_df = pd.DataFrame(qa_rows)
safe_to_csv(qa_df, OUT_QA)

hard_quality_failures = int(((qa_df["severity"] == "hard") & (~qa_df["passed"])).sum())
warning_quality_failures = int(((qa_df["severity"] == "warning") & (~qa_df["passed"])).sum())
info_quality_findings = int(((qa_df["severity"] == "info") & (~qa_df["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v68d_quality_checks",
        "issue_type": "hard_group_aware_split_policy_failed",
        "issue_detail": f"{hard_quality_failures} hard quality checks failed.",
        "severity": "hard",
    })

if current_video_overlap:
    issues.append({
        "item": "current_split_source_video_overlap",
        "issue_type": "warning_current_split_has_source_video_overlap",
        "issue_detail": f"Current split has {current_video_overlap} source videos appearing in multiple splits.",
        "severity": "warning",
    })

if group_val_major_missing:
    issues.append({
        "item": "group_aware_val_major_coverage",
        "issue_type": "warning_group_val_missing_major_classes",
        "issue_detail": ";".join(group_val_major_missing),
        "severity": "warning",
    })

if group_test_major_missing:
    issues.append({
        "item": "group_aware_test_major_coverage",
        "issue_type": "warning_group_test_missing_major_classes",
        "issue_detail": ";".join(group_test_major_missing),
        "severity": "warning",
    })

for c in group_val_missing:
    if c not in major_set:
        issues.append({
            "item": c,
            "issue_type": "info_group_val_missing_rare_or_low_class",
            "issue_detail": "Class missing from group-aware val.",
            "severity": "info",
        })

for c in group_test_missing:
    if c not in major_set:
        issues.append({
            "item": c,
            "issue_type": "info_group_test_missing_rare_or_low_class",
            "issue_detail": "Class missing from group-aware test.",
            "severity": "info",
        })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

manifest = {
    "version": "week8_v68d_group_aware_split_policy",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "source_metadata": str(METADATA),
    "source_video_count": len(videos),
    "classes": classes,
    "major_classes": major_classes,
    "rare_classes": rare_classes,
    "selected_candidate": selected,
    "current_split": {
        "source_video_overlap_count": current_video_overlap,
        "train_objects": int(current_split_counts.get("train", 0)),
        "val_objects": int(current_split_counts.get("val", 0)),
        "test_objects": int(current_split_counts.get("test", 0)),
    },
    "group_aware_split": {
        "source_video_overlap_count": group_video_overlap,
        "train_missing_classes": group_train_missing,
        "val_missing_classes": group_val_missing,
        "test_missing_classes": group_test_missing,
    },
    "claim_boundary": "current split exploratory; group-aware split conservative but may have class coverage limitations",
}

OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

readme = f"""# Week8 Group-Aware Split Policy

## Purpose

This package audits the source-video overlap warning found in v68c and creates a selected group-aware split candidate.

## Current Split

- Source-video overlap count: {current_video_overlap}
- Train objects: {int(current_split_counts.get("train", 0))}
- Val objects: {int(current_split_counts.get("val", 0))}
- Test objects: {int(current_split_counts.get("test", 0))}

The current split is usable for exploratory object-level baseline metrics but must be reported with source-video overlap limitation.

## Selected Group-Aware Split

- Source-video overlap count: {group_video_overlap}
- Train missing classes: {';'.join(group_train_missing) if group_train_missing else 'none'}
- Val missing classes: {';'.join(group_val_missing) if group_val_missing else 'none'}
- Test missing classes: {';'.join(group_test_missing) if group_test_missing else 'none'}

The group-aware split is more conservative for source-video generalization, but class coverage limitations must be reported.

## Recommendation

Run baseline sanity on the current split for continuity. Also keep the group-aware split for conservative evaluation or sensitivity analysis.

Do not merge current-split and group-aware metrics into a single claim.
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
    "v68d_decision": "group_aware_split_policy_completed" if hard_issue_count == 0 else "group_aware_split_policy_has_blocking_issues",
    "metadata_rows": int(len(metadata)),
    "source_video_count": int(len(videos)),
    "behaviour_classes": int(len(classes)),
    "major_class_count": int(len(major_classes)),
    "rare_class_count": int(len(rare_classes)),
    "current_source_video_overlap_count": int(current_video_overlap),
    "group_source_video_overlap_count": int(group_video_overlap),
    "group_train_missing_classes": ";".join(group_train_missing),
    "group_val_missing_classes": ";".join(group_val_missing),
    "group_test_missing_classes": ";".join(group_test_missing),
    "selected_candidate_score": float(selected["score"]) if selected else "",
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_quality_failures": hard_quality_failures,
    "warning_quality_failures": warning_quality_failures,
    "info_quality_findings": info_quality_findings,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v69a_baseline_sanity": bool(hard_issue_count == 0),
    "recommended_v69a_policy": "run_current_split_sanity_and_keep_group_aware_as_conservative_split",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v68d Group-Aware Split Policy\n\n"
    f"- v68d decision: {decision.iloc[0]['v68d_decision']}\n"
    f"- Metadata rows: {len(metadata)}\n"
    f"- Source videos: {len(videos)}\n"
    f"- Behaviour classes: {len(classes)}\n"
    f"- Current source-video overlap count: {current_video_overlap}\n"
    f"- Group-aware source-video overlap count: {group_video_overlap}\n"
    f"- Group train missing classes: {';'.join(group_train_missing) if group_train_missing else 'none'}\n"
    f"- Group val missing classes: {';'.join(group_val_missing) if group_val_missing else 'none'}\n"
    f"- Group test missing classes: {';'.join(group_test_missing) if group_test_missing else 'none'}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Warnings: {warning_count}\n"
    f"- Ready for v69a baseline sanity: {bool(hard_issue_count == 0)}\n\n"
    "Recommendation: run current split sanity baseline for continuity and keep group-aware split as conservative sensitivity/generalization check.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v68d Group-Aware Split Policy Report\n\n"
    f"Decision: {decision.iloc[0]['v68d_decision']}\n\n"
    f"Current source-video overlap: {current_video_overlap}\n\n"
    f"Group-aware source-video overlap: {group_video_overlap}\n\n"
    f"ZIP: {OUT_ZIP}\n\n"
    f"SHA256: {zip_hash}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v68d",
    "task_name": "Group-aware split policy",
    "status": "PASS_WITH_WARNINGS" if hard_issue_count == 0 and warning_count > 0 else ("PASS" if hard_issue_count == 0 else "NEEDS_FIX"),
    "input_summary": str(METADATA),
    "output_summary": str(PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Run v69a baseline sanity on current split and preserve group-aware split for conservative evaluation." if hard_issue_count == 0 else "Fix group-aware split hard issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_VIDEO_GROUPS)
print(OUT_CANDIDATES)
print(OUT_ASSIGNMENT)
print(OUT_METADATA)
print(OUT_SPLIT_COUNTS)
print(OUT_CLASS_COVERAGE)
print(OUT_COMPARISON)
print(OUT_POLICY)
print(OUT_QA)
print(OUT_ZIP)
print(OUT_SHA256)
print(OUT_DECISION)
print(OUT_NOTE)

print()
print("=== v68d decision ===")
print(decision.to_string(index=False))

print()
print("=== selected assignment ===")
print(assignment_df.to_string(index=False))

print()
print("=== split comparison ===")
print(comparison.to_string(index=False))

print()
print("=== group-aware class coverage ===")
print(coverage_df.to_string(index=False))

print()
print("=== QA ===")
print(qa_df.to_string(index=False))

print()
print("=== issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
