from pathlib import Path
from datetime import datetime
import csv
import json
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

SOLID = W8 / "outputs" / "v65b_gt_v2_solid_foundation_snapshot" / "Week8_GT_v2_Solid_Foundation"

GT_ALL = SOLID / "week8_gt_v2_solid_all_reviewed_objects.csv"
GT_STRICT = SOLID / "week8_gt_v2_solid_strict_gold_objects_for_classification.csv"
GT_CAUTION = SOLID / "week8_gt_v2_solid_caution_objects_for_analysis.csv"
GT_NONUSABLE = SOLID / "week8_gt_v2_solid_nonusable_fix_or_excluded_objects.csv"
GT_SCANFRAME = SOLID / "week8_gt_v2_solid_scanframe_quality_summary.csv"
GT_MANIFEST = SOLID / "week8_gt_v2_solid_manifest.json"

V45_CLIP_OBJECTS = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_object_propagated_annotations.csv"
V65A_SPLIT = W8 / "outputs" / "v65a_classification_readiness_audit" / "week8_v65a_recommended_scanframe_level_split.csv"

OUT = W8 / "outputs" / "v66a_final_gt_v2_label_propagation"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_OBJECTS = OUT / "week8_v66a_final_gt_v2_object_table.csv"
OUT_FRAME_OBJECTS = OUT / "week8_v66a_final_gt_v2_frame_object_labels.csv"
OUT_STRICT_FRAME_OBJECTS = OUT / "week8_v66a_strict_gold_frame_object_labels_for_classification.csv"
OUT_CAUTION_FRAME_OBJECTS = OUT / "week8_v66a_caution_frame_object_labels_for_analysis.csv"
OUT_NONUSABLE_FRAME_OBJECTS = OUT / "week8_v66a_nonusable_frame_object_labels.csv"
OUT_SCANFRAME_SUMMARY = OUT / "week8_v66a_scanframe_propagation_summary.csv"
OUT_BEHAVIOUR_CATEGORY = OUT / "week8_v66a_behaviour_category_distribution.csv"
OUT_QA = OUT / "week8_v66a_propagation_quality_checks.csv"
OUT_DECISION = OUT / "week8_v66a_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v66a_issues.csv"
OUT_NOTE = NOTES / "week8_v66a_final_gt_v2_label_propagation_notes.md"
OUT_REPORT = REPORTS / "week8_v66a_final_gt_v2_label_propagation_report.md"
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


def to_int(x, default=0):
    try:
        return int(float(clean(x)))
    except Exception:
        return default


def to_float(x, default=0.0):
    try:
        return float(clean(x))
    except Exception:
        return default


issues = []

required = [GT_ALL, GT_STRICT, GT_CAUTION, GT_NONUSABLE, GT_SCANFRAME, GT_MANIFEST, V45_CLIP_OBJECTS]

for p in required:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required input missing for final GT v2 label propagation.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v66a_decision": "final_gt_v2_label_propagation_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v66b_final_gt_visualizer": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


all_gt = pd.read_csv(GT_ALL).fillna("")
strict = pd.read_csv(GT_STRICT).fillna("")
caution = pd.read_csv(GT_CAUTION).fillna("")
nonusable = pd.read_csv(GT_NONUSABLE).fillna("")
clip_objects = pd.read_csv(V45_CLIP_OBJECTS).fillna("")

for df in [all_gt, strict, caution, nonusable, clip_objects]:
    for c in df.columns:
        df[c] = df[c].map(clean) if df[c].dtype == object else df[c]

strict_keys = set(zip(strict["scan_frame_id"], strict["canonical_colour_label_norm"]))
caution_keys = set(zip(caution["scan_frame_id"], caution["canonical_colour_label_norm"]))
nonusable_keys = set(zip(nonusable["scan_frame_id"], nonusable["canonical_colour_label_norm"]))

def category(row):
    key = (row["scan_frame_id"], row["canonical_colour_label_norm"])
    if key in strict_keys:
        return "strict_gold_classification"
    if key in caution_keys:
        return "caution_analysis"
    if key in nonusable_keys:
        return "nonusable_fix_or_excluded"
    return "unknown_category"

all_gt["final_gt_v2_category_v66a"] = all_gt.apply(category, axis=1)

all_gt["classification_gate_v66a"] = all_gt["final_gt_v2_category_v66a"].map({
    "strict_gold_classification": "use_for_classification",
    "caution_analysis": "analysis_only_with_caution",
    "nonusable_fix_or_excluded": "do_not_use_for_classification",
    "unknown_category": "do_not_use_for_classification",
})

# Optional split mapping from v65a.
if V65A_SPLIT.exists():
    split_df = pd.read_csv(V65A_SPLIT).fillna("")
    for c in split_df.columns:
        split_df[c] = split_df[c].map(clean) if split_df[c].dtype == object else split_df[c]

    if "canonical_gt_object_id" in split_df.columns and "recommended_split" in split_df.columns:
        split_map = split_df[["canonical_gt_object_id", "recommended_split"]].drop_duplicates()
        all_gt = all_gt.merge(split_map, on="canonical_gt_object_id", how="left")
    else:
        all_gt["recommended_split"] = ""
else:
    all_gt["recommended_split"] = ""

all_gt["recommended_split"] = all_gt["recommended_split"].fillna("")
all_gt.loc[all_gt["final_gt_v2_category_v66a"] != "strict_gold_classification", "recommended_split"] = ""

# Clip metadata from old v45 clip object propagation.
meta_candidates = [
    "scan_frame_id",
    "video_id",
    "clip_path",
    "source_video_path",
    "start_sec",
    "end_sec",
    "duration_sec",
    "fps_used",
    "generated_frame_count",
]

meta_cols = [c for c in meta_candidates if c in clip_objects.columns]
clip_meta = clip_objects[meta_cols].drop_duplicates("scan_frame_id").copy()

for c in ["generated_frame_count"]:
    if c in clip_meta.columns:
        clip_meta[c] = clip_meta[c].map(lambda x: to_int(x, 0))

for c in ["start_sec", "end_sec", "duration_sec", "fps_used"]:
    if c in clip_meta.columns:
        clip_meta[c] = clip_meta[c].map(lambda x: to_float(x, 0.0))

all_gt = all_gt.merge(
    clip_meta,
    on=["scan_frame_id", "video_id"] if "video_id" in clip_meta.columns else ["scan_frame_id"],
    how="left",
    suffixes=("", "_clipmeta"),
)

# Required manual bbox columns.
bbox_cols = ["manual_bbox_x1", "manual_bbox_y1", "manual_bbox_x2", "manual_bbox_y2"]

for c in bbox_cols:
    if c in all_gt.columns:
        all_gt[c] = pd.to_numeric(all_gt[c], errors="coerce")

def bbox_valid(row):
    try:
        x1 = float(row["manual_bbox_x1"])
        y1 = float(row["manual_bbox_y1"])
        x2 = float(row["manual_bbox_x2"])
        y2 = float(row["manual_bbox_y2"])
        return x2 > x1 and y2 > y1
    except Exception:
        return False

all_gt["has_valid_manual_bbox"] = all_gt.apply(bbox_valid, axis=1)

def bbox_propagation_status(row):
    if not row["has_valid_manual_bbox"]:
        return "no_valid_manual_bbox_to_propagate"
    if row["manual_bbox_status"] == "bbox_ok":
        return "propagated_static_manual_anchor_bbox"
    if row["manual_bbox_status"] in ["bbox_needs_manual_redraw", "bbox_wrong"]:
        return "bbox_present_but_marked_fix_required"
    return "bbox_present_non_strict_status"

all_gt["bbox_propagation_status"] = all_gt.apply(bbox_propagation_status, axis=1)

safe_to_csv(all_gt, OUT_OBJECTS)

# Frame-object propagation.
frame_rows = []

for _, obj in all_gt.iterrows():
    scan = obj["scan_frame_id"]
    frame_count = to_int(obj.get("generated_frame_count", ""), 0)

    if frame_count <= 0:
        issues.append({
            "item": scan,
            "issue_type": "warning_missing_generated_frame_count",
            "issue_detail": "No generated_frame_count found; frame-object labels could not be produced for this object.",
            "severity": "warning",
        })
        continue

    fps = to_float(obj.get("fps_used", ""), 0.0)
    if fps <= 0:
        fps = 25.0

    start_sec = to_float(obj.get("start_sec", ""), 0.0)

    for frame_idx in range(frame_count):
        frame_time_in_clip_sec = frame_idx / fps
        source_video_time_sec = start_sec + frame_time_in_clip_sec

        frame_rows.append({
            "final_gt_v2_version": "v66a_final_gt_v2_label_propagation",
            "canonical_gt_object_id": obj["canonical_gt_object_id"],
            "scan_frame_id": scan,
            "video_id": obj["video_id"],
            "frame_idx_in_clip": frame_idx,
            "frame_time_in_clip_sec": round(frame_time_in_clip_sec, 6),
            "source_video_time_sec": round(source_video_time_sec, 6),
            "clip_path": obj.get("clip_path", ""),
            "source_video_path": obj.get("source_video_path", ""),
            "canonical_colour_label_norm": obj["canonical_colour_label_norm"],
            "canonical_colour_label_raw": obj.get("canonical_colour_label_raw", ""),
            "behaviour_code": obj["behaviour_code"],
            "behaviour_raw": obj.get("behaviour_raw", ""),
            "manual_bbox_x1": obj.get("manual_bbox_x1", ""),
            "manual_bbox_y1": obj.get("manual_bbox_y1", ""),
            "manual_bbox_x2": obj.get("manual_bbox_x2", ""),
            "manual_bbox_y2": obj.get("manual_bbox_y2", ""),
            "manual_bbox_status": obj.get("manual_bbox_status", ""),
            "manual_identity_status": obj.get("manual_identity_status", ""),
            "manual_gt_v2_status": obj.get("manual_gt_v2_status", ""),
            "manual_classification_use": obj.get("manual_classification_use", ""),
            "manual_assigned_candidate_box_id": obj.get("manual_assigned_candidate_box_id", ""),
            "final_gt_v2_category": obj["final_gt_v2_category_v66a"],
            "classification_gate": obj["classification_gate_v66a"],
            "recommended_split": obj.get("recommended_split", ""),
            "bbox_propagation_status": obj["bbox_propagation_status"],
            "label_propagation_rule": "behaviour_label_inherited_across_10s_observation_clip_from_canonical_excel_gt_v2",
            "bbox_rule": "manual_anchor_bbox_repeated_across_clip_for_dataset_labeling_not_tracking_claim",
        })

frame_df = pd.DataFrame(frame_rows)
safe_to_csv(frame_df, OUT_FRAME_OBJECTS)

strict_frame = frame_df[frame_df["final_gt_v2_category"] == "strict_gold_classification"].copy()
caution_frame = frame_df[frame_df["final_gt_v2_category"] == "caution_analysis"].copy()
nonusable_frame = frame_df[frame_df["final_gt_v2_category"] == "nonusable_fix_or_excluded"].copy()

safe_to_csv(strict_frame, OUT_STRICT_FRAME_OBJECTS)
safe_to_csv(caution_frame, OUT_CAUTION_FRAME_OBJECTS)
safe_to_csv(nonusable_frame, OUT_NONUSABLE_FRAME_OBJECTS)

scan_rows = []
for scan, g in all_gt.groupby("scan_frame_id"):
    gf = frame_df[frame_df["scan_frame_id"] == scan]
    strict_obj = g[g["final_gt_v2_category_v66a"] == "strict_gold_classification"]
    caution_obj = g[g["final_gt_v2_category_v66a"] == "caution_analysis"]
    nonusable_obj = g[g["final_gt_v2_category_v66a"] == "nonusable_fix_or_excluded"]

    frame_count = int(g["generated_frame_count"].dropna().astype(str).map(lambda x: to_int(x, 0)).max()) if "generated_frame_count" in g.columns else 0

    scan_rows.append({
        "scan_frame_id": scan,
        "video_id": g["video_id"].iloc[0],
        "clip_path": g["clip_path"].iloc[0] if "clip_path" in g.columns else "",
        "generated_frame_count": frame_count,
        "object_count_total": int(len(g)),
        "strict_gold_object_count": int(len(strict_obj)),
        "caution_object_count": int(len(caution_obj)),
        "nonusable_object_count": int(len(nonusable_obj)),
        "frame_object_rows": int(len(gf)),
        "strict_gold_frame_object_rows": int((gf["final_gt_v2_category"] == "strict_gold_classification").sum()) if len(gf) else 0,
        "caution_frame_object_rows": int((gf["final_gt_v2_category"] == "caution_analysis").sum()) if len(gf) else 0,
        "nonusable_frame_object_rows": int((gf["final_gt_v2_category"] == "nonusable_fix_or_excluded").sum()) if len(gf) else 0,
        "behaviour_codes": ";".join(g["behaviour_code"].astype(str).tolist()),
        "strict_gold_behaviour_codes": ";".join(strict_obj["behaviour_code"].astype(str).tolist()),
        "scanframe_propagation_status": "propagated" if len(gf) > 0 else "not_propagated",
    })

scan_summary = pd.DataFrame(scan_rows).sort_values("scan_frame_id")
safe_to_csv(scan_summary, OUT_SCANFRAME_SUMMARY)

behaviour_category = (
    all_gt.groupby(["final_gt_v2_category_v66a", "behaviour_code"])
    .size()
    .reset_index(name="object_count")
    .sort_values(["final_gt_v2_category_v66a", "behaviour_code"])
)
safe_to_csv(behaviour_category, OUT_BEHAVIOUR_CATEGORY)

qa_rows = []

def add_check(name, expected, actual, passed, severity, detail):
    qa_rows.append({
        "check_name": name,
        "expected": str(expected),
        "actual": str(actual),
        "passed": bool(passed),
        "severity": severity,
        "detail": detail,
    })

add_check("object_rows", 432, len(all_gt), len(all_gt) == 432, "hard", "Final object table should preserve all 432 reviewed objects.")
add_check("scanframe_count", 72, all_gt["scan_frame_id"].nunique(), all_gt["scan_frame_id"].nunique() == 72, "hard", "All 72 scanframes should be present.")
expected_strict_count = len(strict_keys)
expected_caution_count = len(caution_keys)
expected_nonusable_count = len(nonusable_keys)

add_check("strict_object_rows_dynamic", expected_strict_count, (all_gt["final_gt_v2_category_v66a"] == "strict_gold_classification").sum(), (all_gt["final_gt_v2_category_v66a"] == "strict_gold_classification").sum() == expected_strict_count, "hard", "Strict object count should match current final GT strict subset.")
add_check("caution_object_rows_dynamic", expected_caution_count, (all_gt["final_gt_v2_category_v66a"] == "caution_analysis").sum(), (all_gt["final_gt_v2_category_v66a"] == "caution_analysis").sum() == expected_caution_count, "hard", "Caution object count should match current final GT caution subset.")
add_check("nonusable_object_rows_dynamic", expected_nonusable_count, (all_gt["final_gt_v2_category_v66a"] == "nonusable_fix_or_excluded").sum(), (all_gt["final_gt_v2_category_v66a"] == "nonusable_fix_or_excluded").sum() == expected_nonusable_count, "hard", "Nonusable object count should match current final GT nonusable subset.")
add_check("frame_object_rows_created", ">0", len(frame_df), len(frame_df) > 0, "hard", "Frame-object propagated labels should be created.")
add_check("strict_frame_object_rows_created", ">0", len(strict_frame), len(strict_frame) > 0, "hard", "Strict frame-object labels should be created.")
add_check("all_scanframes_propagated", 72, int((scan_summary["scanframe_propagation_status"] == "propagated").sum()), int((scan_summary["scanframe_propagation_status"] == "propagated").sum()) == 72, "hard", "Every scanframe should have propagated frame-object labels.")

if "recommended_split" in strict_frame.columns and len(strict_frame):
    split_leak = strict_frame.groupby("scan_frame_id")["recommended_split"].nunique().reset_index()
    split_leak = split_leak[split_leak["recommended_split"] > 1]
    add_check("no_same_scanframe_split_leakage", 0, len(split_leak), len(split_leak) == 0, "hard", "Strict rows from same scanframe should not appear in multiple splits.")

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_QA)

hard_quality_failures = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())
warning_quality_failures = int(((qa["severity"] == "warning") & (~qa["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "propagation_quality_checks",
        "issue_type": "hard_propagation_quality_check_failed",
        "issue_detail": f"{hard_quality_failures} hard propagation quality checks failed.",
        "severity": "hard",
    })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0

decision = pd.DataFrame([{
    "v66a_decision": "final_gt_v2_label_propagation_completed" if hard_issue_count == 0 else "final_gt_v2_label_propagation_has_blocking_issues",
    "object_rows": int(len(all_gt)),
    "scanframe_count": int(all_gt["scan_frame_id"].nunique()),
    "frame_object_rows": int(len(frame_df)),
    "strict_gold_object_rows": int((all_gt["final_gt_v2_category_v66a"] == "strict_gold_classification").sum()),
    "strict_gold_frame_object_rows": int(len(strict_frame)),
    "caution_object_rows": int((all_gt["final_gt_v2_category_v66a"] == "caution_analysis").sum()),
    "caution_frame_object_rows": int(len(caution_frame)),
    "nonusable_object_rows": int((all_gt["final_gt_v2_category_v66a"] == "nonusable_fix_or_excluded").sum()),
    "nonusable_frame_object_rows": int(len(nonusable_frame)),
    "hard_quality_failures": hard_quality_failures,
    "warning_quality_failures": warning_quality_failures,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "ready_for_v66b_final_gt_visualizer": bool(hard_issue_count == 0),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v66a Final GT v2 Label Propagation\n\n"
    f"- v66a decision: {decision.iloc[0]['v66a_decision']}\n"
    f"- Object rows: {len(all_gt)}\n"
    f"- Scanframes: {all_gt['scan_frame_id'].nunique()}\n"
    f"- Frame-object rows: {len(frame_df)}\n"
    f"- Strict gold objects: {int((all_gt['final_gt_v2_category_v66a'] == 'strict_gold_classification').sum())}\n"
    f"- Strict gold frame-object rows: {len(strict_frame)}\n"
    f"- Caution objects: {int((all_gt['final_gt_v2_category_v66a'] == 'caution_analysis').sum())}\n"
    f"- Nonusable/fix/excluded objects: {int((all_gt['final_gt_v2_category_v66a'] == 'nonusable_fix_or_excluded').sum())}\n"
    f"- Hard quality failures: {hard_quality_failures}\n"
    f"- Ready for v66b final GT visualizer: {bool(hard_issue_count == 0)}\n\n"
    "Important: bbox coordinates are propagated as static manual anchor boxes across each 10-second clip for label representation. This is not a tracking-quality claim.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v66a Final GT v2 Label Propagation Report\n\n"
    f"Decision: {decision.iloc[0]['v66a_decision']}\n\n"
    f"Object table: `{OUT_OBJECTS}`\n\n"
    f"Frame-object labels: `{OUT_FRAME_OBJECTS}`\n\n"
    f"Strict classification frame-object labels: `{OUT_STRICT_FRAME_OBJECTS}`\n\n"
    "BBox propagation rule: manual anchor boxes are repeated across the clip for dataset label representation. Tracking remains a helper layer and will be reattached separately.\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v66a",
    "task_name": "Final GT v2 label propagation",
    "status": "PASS" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": str(GT_ALL),
    "output_summary": str(OUT),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Build final GT v2 visualizer using propagated labels and final categories.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_OBJECTS)
print(OUT_FRAME_OBJECTS)
print(OUT_STRICT_FRAME_OBJECTS)
print(OUT_CAUTION_FRAME_OBJECTS)
print(OUT_NONUSABLE_FRAME_OBJECTS)
print(OUT_SCANFRAME_SUMMARY)
print(OUT_BEHAVIOUR_CATEGORY)
print(OUT_QA)
print(OUT_DECISION)
print(OUT_NOTE)

print()
print("=== v66a decision ===")
print(decision.to_string(index=False))

print()
print("=== QA ===")
print(qa.to_string(index=False))

print()
print("=== scanframe summary head ===")
print(scan_summary.head(20).to_string(index=False))

print()
print("=== behaviour/category distribution ===")
print(behaviour_category.to_string(index=False))

print()
print("=== issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
