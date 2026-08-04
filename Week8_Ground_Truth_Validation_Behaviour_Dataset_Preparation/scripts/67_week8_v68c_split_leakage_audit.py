from pathlib import Path
from datetime import datetime
import csv
import json
import hashlib
import zipfile
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V68A = W8 / "outputs" / "v68a_strict_gold_anchor_crop_materialization"
V68B = W8 / "outputs" / "v68b_crop_qa_gallery"

DATASET = V68A / "Week8_StrictGold_AnchorFrame_Crop_Dataset"
METADATA = DATASET / "metadata" / "week8_v68a_strict_gold_anchor_crop_metadata.csv"
V68A_DECISION = V68A / "week8_v68a_decision_summary.csv"
V68B_DECISION = V68B / "week8_v68b_decision_summary.csv"

OUT = W8 / "outputs" / "v68c_split_leakage_audit"
AUDIT = OUT / "Week8_StrictGold_Split_Leakage_Audit"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, AUDIT, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_QA = AUDIT / "week8_v68c_split_leakage_quality_checks.csv"
OUT_OBJECT_SPLIT = AUDIT / "week8_v68c_object_split_summary.csv"
OUT_SCANFRAME_SPLIT = AUDIT / "week8_v68c_scanframe_split_summary.csv"
OUT_CLIP_SPLIT = AUDIT / "week8_v68c_clip_path_split_summary.csv"
OUT_VIDEO_SPLIT = AUDIT / "week8_v68c_source_video_split_overlap.csv"
OUT_CLASS_COVERAGE = AUDIT / "week8_v68c_class_split_coverage.csv"
OUT_SPLIT_COUNTS = AUDIT / "week8_v68c_split_counts.csv"
OUT_POLICY = AUDIT / "week8_v68c_split_policy_recommendations.csv"
OUT_MANIFEST = AUDIT / "week8_v68c_split_leakage_audit_manifest.json"
OUT_README = AUDIT / "README_Week8_StrictGold_Split_Leakage_Audit.md"

OUT_DECISION = OUT / "week8_v68c_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v68c_issues.csv"
OUT_ZIP = OUT / "Week8_StrictGold_Split_Leakage_Audit.zip"
OUT_SHA256 = OUT / "Week8_StrictGold_Split_Leakage_Audit.sha256"
OUT_NOTE = NOTES / "week8_v68c_split_leakage_audit_notes.md"
OUT_REPORT = REPORTS / "week8_v68c_split_leakage_audit_report.md"
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

for p in [METADATA, V68A_DECISION, V68B_DECISION]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required v68a/v68b output missing for split leakage audit.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v68c_decision": "split_leakage_audit_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v69a_baseline_sanity": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


v68a_decision = read_csv_clean(V68A_DECISION)
v68b_decision = read_csv_clean(V68B_DECISION)

if len(v68a_decision) == 0 or not bool_true(v68a_decision.iloc[0].get("ready_for_v68b_crop_qa_gallery", "")):
    issues.append({
        "item": str(V68A_DECISION),
        "issue_type": "hard_v68a_not_ready",
        "issue_detail": "v68a must be completed before v68c split leakage audit.",
        "severity": "hard",
    })

if len(v68b_decision) == 0 or not bool_true(v68b_decision.iloc[0].get("ready_for_v68c_split_leakage_audit", "")):
    issues.append({
        "item": str(V68B_DECISION),
        "issue_type": "hard_v68b_not_ready",
        "issue_detail": "v68b must be completed before v68c split leakage audit.",
        "severity": "hard",
    })

metadata = read_csv_clean(METADATA)

required_columns = [
    "canonical_gt_object_id",
    "scan_frame_id",
    "video_id",
    "clip_path",
    "anchor_frame_path",
    "behaviour_code",
    "recommended_split",
    "tight_crop_path",
    "context10_crop_path",
    "tracking_used",
]

for c in required_columns:
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
        "v68c_decision": "split_leakage_audit_blocked",
        "metadata_rows": int(len(metadata)),
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v69a_baseline_sanity": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    print(issues_df.to_string(index=False))
    raise SystemExit(1)


# Core summaries.
split_order = ["train", "val", "test"]

object_split = (
    metadata.groupby("canonical_gt_object_id")["recommended_split"]
    .agg(lambda s: ";".join(sorted(set(s))))
    .reset_index()
)
object_split["split_count"] = object_split["recommended_split"].map(lambda s: len([x for x in s.split(";") if x]))

scanframe_split = (
    metadata.groupby("scan_frame_id")["recommended_split"]
    .agg(lambda s: ";".join(sorted(set(s))))
    .reset_index()
)
scanframe_split["split_count"] = scanframe_split["recommended_split"].map(lambda s: len([x for x in s.split(";") if x]))
scanframe_split = scanframe_split.merge(
    metadata.groupby("scan_frame_id").size().reset_index(name="object_count"),
    on="scan_frame_id",
    how="left",
)

clip_split = (
    metadata.groupby("clip_path")["recommended_split"]
    .agg(lambda s: ";".join(sorted(set(s))))
    .reset_index()
)
clip_split["split_count"] = clip_split["recommended_split"].map(lambda s: len([x for x in s.split(";") if x]))
clip_split = clip_split.merge(
    metadata.groupby("clip_path").size().reset_index(name="object_count"),
    on="clip_path",
    how="left",
)

video_split = (
    metadata.groupby("video_id")["recommended_split"]
    .agg(lambda s: ";".join(sorted(set(s))))
    .reset_index()
)
video_split["split_count"] = video_split["recommended_split"].map(lambda s: len([x for x in s.split(";") if x]))
video_split = video_split.merge(
    metadata.groupby("video_id").size().reset_index(name="object_count"),
    on="video_id",
    how="left",
)
video_split = video_split.merge(
    metadata.groupby("video_id")["scan_frame_id"].nunique().reset_index(name="scanframe_count"),
    on="video_id",
    how="left",
)

class_split = (
    metadata.groupby(["behaviour_code", "recommended_split"])
    .size()
    .reset_index(name="object_count")
)

pivot = class_split.pivot(index="behaviour_code", columns="recommended_split", values="object_count").fillna(0).astype(int).reset_index()
for s in split_order:
    if s not in pivot.columns:
        pivot[s] = 0

pivot["total"] = pivot[split_order].sum(axis=1)
pivot["present_in_train"] = pivot["train"] > 0
pivot["present_in_val"] = pivot["val"] > 0
pivot["present_in_test"] = pivot["test"] > 0
pivot["present_in_all_splits"] = pivot["present_in_train"] & pivot["present_in_val"] & pivot["present_in_test"]
pivot["rare_class_lt10_total"] = pivot["total"] < 10
pivot["missing_split_count"] = 3 - pivot[["present_in_train", "present_in_val", "present_in_test"]].sum(axis=1)

split_counts = (
    metadata.groupby("recommended_split")
    .size()
    .reset_index(name="object_count")
)

split_counts = split_counts.merge(
    metadata.groupby("recommended_split")["scan_frame_id"].nunique().reset_index(name="scanframe_count"),
    on="recommended_split",
    how="left",
)
split_counts = split_counts.merge(
    metadata.groupby("recommended_split")["video_id"].nunique().reset_index(name="source_video_count"),
    on="recommended_split",
    how="left",
)
split_counts = split_counts.merge(
    metadata.groupby("recommended_split")["behaviour_code"].nunique().reset_index(name="behaviour_class_count"),
    on="recommended_split",
    how="left",
)

safe_to_csv(object_split, OUT_OBJECT_SPLIT)
safe_to_csv(scanframe_split, OUT_SCANFRAME_SPLIT)
safe_to_csv(clip_split, OUT_CLIP_SPLIT)
safe_to_csv(video_split, OUT_VIDEO_SPLIT)
safe_to_csv(pivot, OUT_CLASS_COVERAGE)
safe_to_csv(split_counts, OUT_SPLIT_COUNTS)

# QA checks.
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


object_leak_count = int((object_split["split_count"] > 1).sum())
scanframe_leak_count = int((scanframe_split["split_count"] > 1).sum())
clip_leak_count = int((clip_split["split_count"] > 1).sum())
video_overlap_count = int((video_split["split_count"] > 1).sum())

unknown_split_count = int((~metadata["recommended_split"].isin(split_order)).sum())
tracking_used_count = int((metadata["tracking_used"].astype(str).str.lower() != "no").sum())
duplicate_object_rows = int(len(metadata) - metadata["canonical_gt_object_id"].nunique())
duplicate_tight_paths = int(len(metadata) - metadata["tight_crop_path"].nunique())
duplicate_context_paths = int(len(metadata) - metadata["context10_crop_path"].nunique())

missing_val_classes = int((pivot["val"] == 0).sum())
missing_test_classes = int((pivot["test"] == 0).sum())
rare_class_count = int((pivot["rare_class_lt10_total"] == True).sum())
not_all_split_class_count = int((pivot["present_in_all_splits"] == False).sum())

train_count = int(split_counts.loc[split_counts["recommended_split"] == "train", "object_count"].sum())
val_count = int(split_counts.loc[split_counts["recommended_split"] == "val", "object_count"].sum())
test_count = int(split_counts.loc[split_counts["recommended_split"] == "test", "object_count"].sum())

add_qa("metadata_rows", 372, len(metadata), len(metadata) == 372, "hard", "Split audit input should have 372 crop rows.")
add_qa("object_id_unique", 0, duplicate_object_rows, duplicate_object_rows == 0, "hard", "Each canonical GT object should appear once.")
add_qa("object_id_split_leakage", 0, object_leak_count, object_leak_count == 0, "hard", "Same object ID must not appear in multiple splits.")
add_qa("scanframe_split_leakage", 0, scanframe_leak_count, scanframe_leak_count == 0, "hard", "Same scanframe must not appear in multiple splits.")
add_qa("clip_path_split_leakage", 0, clip_leak_count, clip_leak_count == 0, "hard", "Same extracted clip path must not appear in multiple splits.")
add_qa("train_count", 255, train_count, train_count == 255, "hard", "Train object count should match v67c/v68a count.")
add_qa("val_count", 39, val_count, val_count == 39, "hard", "Val object count should match v67c/v68a count.")
add_qa("test_count", 78, test_count, test_count == 78, "hard", "Test object count should match v67c/v68a count.")
add_qa("train_val_test_only", 0, unknown_split_count, unknown_split_count == 0, "hard", "All rows should belong to train/val/test.")
add_qa("tracking_not_used", 0, tracking_used_count, tracking_used_count == 0, "hard", "Tracking must not be used for crop labels.")
add_qa("tight_crop_paths_unique", 0, duplicate_tight_paths, duplicate_tight_paths == 0, "hard", "Tight crop paths should be unique.")
add_qa("context_crop_paths_unique", 0, duplicate_context_paths, duplicate_context_paths == 0, "hard", "Context crop paths should be unique.")

add_qa("source_video_overlap_across_splits", 0, video_overlap_count, video_overlap_count == 0, "warning", "Source-video overlap can inflate metrics; if present, report as limitation.")
add_qa("classes_missing_from_val", 0, missing_val_classes, missing_val_classes == 0, "warning", "Classes missing from validation prevent per-class validation metrics.")
add_qa("classes_missing_from_test", 0, missing_test_classes, missing_test_classes == 0, "warning", "Classes missing from test prevent per-class test metrics.")
add_qa("rare_classes_lt10_total", 0, rare_class_count, rare_class_count == 0, "info", "Rare classes should be documented; this is not a blocker.")
add_qa("classes_not_present_in_all_splits", 0, not_all_split_class_count, not_all_split_class_count == 0, "info", "Not all classes are present in all splits due small/imbalanced data.")

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_QA)

# Issues from QA.
hard_quality_failures = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())
warning_quality_failures = int(((qa["severity"] == "warning") & (~qa["passed"])).sum())
info_quality_findings = int(((qa["severity"] == "info") & (~qa["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v68c_quality_checks",
        "issue_type": "hard_split_leakage_audit_failed",
        "issue_detail": f"{hard_quality_failures} hard split leakage checks failed.",
        "severity": "hard",
    })

if video_overlap_count:
    overlap_videos = video_split[video_split["split_count"] > 1]["video_id"].tolist()
    issues.append({
        "item": "source_video_overlap_across_splits",
        "issue_type": "warning_source_video_group_leakage_risk",
        "issue_detail": f"{video_overlap_count} source videos appear in multiple splits: {';'.join(overlap_videos)}",
        "severity": "warning",
    })

if missing_val_classes:
    missing_val = pivot[pivot["val"] == 0]["behaviour_code"].tolist()
    issues.append({
        "item": "validation_class_coverage",
        "issue_type": "warning_classes_missing_from_val",
        "issue_detail": f"Classes missing from val: {';'.join(missing_val)}",
        "severity": "warning",
    })

if missing_test_classes:
    missing_test = pivot[pivot["test"] == 0]["behaviour_code"].tolist()
    issues.append({
        "item": "test_class_coverage",
        "issue_type": "warning_classes_missing_from_test",
        "issue_detail": f"Classes missing from test: {';'.join(missing_test)}",
        "severity": "warning",
    })

for _, r in pivot[pivot["rare_class_lt10_total"] == True].iterrows():
    issues.append({
        "item": r["behaviour_code"],
        "issue_type": "info_rare_class_split_limitation",
        "issue_detail": f"Total strict rows={r['total']}. Train={r['train']}, val={r['val']}, test={r['test']}.",
        "severity": "info",
    })

policy_rows = [
    {
        "topic": "object_id_leakage",
        "status": "pass" if object_leak_count == 0 else "fail",
        "recommendation": "Do not proceed if same canonical object appears in multiple splits.",
    },
    {
        "topic": "scanframe_leakage",
        "status": "pass" if scanframe_leak_count == 0 else "fail",
        "recommendation": "Do not proceed if same scanframe appears in multiple splits.",
    },
    {
        "topic": "source_video_overlap",
        "status": "warning" if video_overlap_count else "pass",
        "recommendation": "If source-video overlap exists, report baseline metrics as clip/object-level exploratory metrics. Consider optional group-by-video split later.",
    },
    {
        "topic": "class_coverage",
        "status": "warning" if missing_val_classes or missing_test_classes else "pass",
        "recommendation": "Report macro-F1 carefully. Missing classes in val/test mean per-class metrics are not reliable for those classes.",
    },
    {
        "topic": "rare_classes",
        "status": "info" if rare_class_count else "pass",
        "recommendation": "Do not claim reliable standalone performance for IA, BE, DE or any class below 10 examples.",
    },
    {
        "topic": "next_step",
        "status": "ready" if hard_quality_failures == 0 else "blocked",
        "recommendation": "Proceed to v69a baseline sanity only if hard split leakage checks are zero.",
    },
]

policy = pd.DataFrame(policy_rows)
safe_to_csv(policy, OUT_POLICY)

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

manifest = {
    "audit_version": "week8_v68c_split_leakage_audit",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "metadata_rows": int(len(metadata)),
    "train_objects": train_count,
    "val_objects": val_count,
    "test_objects": test_count,
    "object_id_split_leakage": object_leak_count,
    "scanframe_split_leakage": scanframe_leak_count,
    "clip_path_split_leakage": clip_leak_count,
    "source_video_overlap_count": video_overlap_count,
    "missing_val_classes": missing_val_classes,
    "missing_test_classes": missing_test_classes,
    "rare_class_count": rare_class_count,
    "claim_boundary": "baseline metrics are exploratory if source-video overlap or rare class limitations exist",
}

OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

readme = f"""# Week8 Strict-Gold Split Leakage Audit

## Summary

This audit checks leakage and split limitations for the v68a strict-gold anchor-frame crop dataset.

## Hard Leakage Checks

- Object ID leakage across splits: {object_leak_count}
- Scanframe leakage across splits: {scanframe_leak_count}
- Clip path leakage across splits: {clip_leak_count}

## Split Counts

- Train objects: {train_count}
- Val objects: {val_count}
- Test objects: {test_count}

## Warnings

- Source videos appearing in multiple splits: {video_overlap_count}
- Classes missing from validation: {missing_val_classes}
- Classes missing from test: {missing_test_classes}
- Rare classes below 10 total strict examples: {rare_class_count}

## Usage

If hard leakage checks are zero, the dataset can proceed to baseline sanity checks. However, source-video overlap and rare classes must be reported as limitations.
"""

OUT_README.write_text(readme)

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(AUDIT.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

zip_hash = sha256_file(OUT_ZIP)
OUT_SHA256.write_text(f"{zip_hash}  {OUT_ZIP.name}\n")

decision = pd.DataFrame([{
    "v68c_decision": "split_leakage_audit_completed" if hard_issue_count == 0 else "split_leakage_audit_has_blocking_issues",
    "metadata_rows": int(len(metadata)),
    "train_objects": train_count,
    "val_objects": val_count,
    "test_objects": test_count,
    "behaviour_classes": int(metadata["behaviour_code"].nunique()),
    "object_id_split_leakage": object_leak_count,
    "scanframe_split_leakage": scanframe_leak_count,
    "clip_path_split_leakage": clip_leak_count,
    "source_video_overlap_count": video_overlap_count,
    "classes_missing_from_val": missing_val_classes,
    "classes_missing_from_test": missing_test_classes,
    "rare_class_count": rare_class_count,
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
    "baseline_claim_scope": "exploratory_baseline_with_documented_split_limitations",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v68c Split Leakage Audit\n\n"
    f"- v68c decision: {decision.iloc[0]['v68c_decision']}\n"
    f"- Metadata rows: {len(metadata)}\n"
    f"- Train objects: {train_count}\n"
    f"- Val objects: {val_count}\n"
    f"- Test objects: {test_count}\n"
    f"- Object ID split leakage: {object_leak_count}\n"
    f"- Scanframe split leakage: {scanframe_leak_count}\n"
    f"- Clip path split leakage: {clip_leak_count}\n"
    f"- Source-video overlap count: {video_overlap_count}\n"
    f"- Classes missing from val: {missing_val_classes}\n"
    f"- Classes missing from test: {missing_test_classes}\n"
    f"- Rare class count: {rare_class_count}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Warnings: {warning_count}\n"
    f"- Ready for v69a baseline sanity: {bool(hard_issue_count == 0)}\n\n"
    "If source-video overlap exists, baseline metrics should be reported as exploratory clip/object-level metrics, not as production generalization evidence.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v68c Split Leakage Audit Report\n\n"
    f"Decision: {decision.iloc[0]['v68c_decision']}\n\n"
    f"Object leakage: {object_leak_count}\n\n"
    f"Scanframe leakage: {scanframe_leak_count}\n\n"
    f"Clip path leakage: {clip_leak_count}\n\n"
    f"Source-video overlap: {video_overlap_count}\n\n"
    f"ZIP: {OUT_ZIP}\n\n"
    f"SHA256: {zip_hash}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v68c",
    "task_name": "Strict-gold split leakage audit",
    "status": "PASS_WITH_WARNINGS" if hard_issue_count == 0 and warning_count > 0 else ("PASS" if hard_issue_count == 0 else "NEEDS_FIX"),
    "input_summary": str(METADATA),
    "output_summary": str(AUDIT),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Run v69a baseline sanity checks." if hard_issue_count == 0 else "Fix split leakage hard issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_QA)
print(OUT_OBJECT_SPLIT)
print(OUT_SCANFRAME_SPLIT)
print(OUT_CLIP_SPLIT)
print(OUT_VIDEO_SPLIT)
print(OUT_CLASS_COVERAGE)
print(OUT_SPLIT_COUNTS)
print(OUT_POLICY)
print(OUT_ZIP)
print(OUT_SHA256)
print(OUT_DECISION)
print(OUT_NOTE)

print()
print("=== v68c decision ===")
print(decision.to_string(index=False))

print()
print("=== QA ===")
print(qa.to_string(index=False))

print()
print("=== split counts ===")
print(split_counts.to_string(index=False))

print()
print("=== class coverage ===")
print(pivot.to_string(index=False))

print()
print("=== issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
