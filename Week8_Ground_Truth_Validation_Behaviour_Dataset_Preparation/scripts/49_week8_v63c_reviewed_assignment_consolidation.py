from pathlib import Path
from datetime import datetime
import csv
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

ASSIGNMENTS = W8 / "validation" / "week8_v63b_manual_gt_v2_assignments.csv"

OUT = W8 / "outputs" / "v63c_reviewed_assignment_consolidation"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_REVIEWED = OUT / "week8_v63c_reviewed_object_assignments.csv"
OUT_USABLE = OUT / "week8_v63c_gold_usable_object_gt.csv"
OUT_PENDING = OUT / "week8_v63c_pending_or_fix_required_object_gt.csv"
OUT_SCANFRAME = OUT / "week8_v63c_scanframe_assignment_summary.csv"
OUT_COUNTS = OUT / "week8_v63c_assignment_counts.csv"
OUT_DECISION = OUT / "week8_v63c_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v63c_issues.csv"
OUT_NOTE = NOTES / "week8_v63c_reviewed_assignment_consolidation_notes.md"
OUT_REPORT = REPORTS / "week8_v63c_reviewed_assignment_consolidation_report.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def clean_str(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


issues = []

if not ASSIGNMENTS.exists():
    issues.append({
        "item": str(ASSIGNMENTS),
        "issue_type": "hard_missing_assignments_csv",
        "issue_detail": "v63b manual assignment CSV is missing.",
        "severity": "hard",
    })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v63c_decision": "reviewed_assignment_consolidation_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v64_manual_gt_v2_continuation": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


df = pd.read_csv(ASSIGNMENTS).fillna("")

for c in df.columns:
    df[c] = df[c].map(clean_str)

reviewed = df[df["manual_gt_v2_status"].astype(str).str.strip() != ""].copy()

reviewed["object_gt_v2_decision"] = "pending_or_unclear"

gold_mask = (
    (reviewed["manual_gt_v2_status"] == "gold_usable")
    & (reviewed["manual_classification_use"] == "use_for_classification")
    & (reviewed["manual_bbox_status"] == "bbox_ok")
    & (reviewed["manual_identity_status"] == "identity_confirmed")
    & (reviewed["manual_assigned_candidate_box_id"].astype(str).str.strip() != "")
)

reviewed.loc[gold_mask, "object_gt_v2_decision"] = "gold_usable_for_classification"

fix_mask = reviewed["manual_gt_v2_status"].isin(["fix_required", "red_exclude"])
reviewed.loc[fix_mask, "object_gt_v2_decision"] = "fix_or_exclude"

pending_mask = reviewed["manual_gt_v2_status"].isin(["unknown_pending_review", ""])
reviewed.loc[pending_mask, "object_gt_v2_decision"] = "pending_review"

usable = reviewed[reviewed["object_gt_v2_decision"] == "gold_usable_for_classification"].copy()
pending = reviewed[reviewed["object_gt_v2_decision"] != "gold_usable_for_classification"].copy()

safe_to_csv(reviewed, OUT_REVIEWED)
safe_to_csv(usable, OUT_USABLE)
safe_to_csv(pending, OUT_PENDING)

scan_rows = []
for scan, g in df.groupby("scan_frame_id"):
    reviewed_g = g[g["manual_gt_v2_status"].astype(str).str.strip() != ""].copy()
    usable_g = reviewed_g[
        (reviewed_g["manual_gt_v2_status"] == "gold_usable")
        & (reviewed_g["manual_classification_use"] == "use_for_classification")
        & (reviewed_g["manual_bbox_status"] == "bbox_ok")
        & (reviewed_g["manual_identity_status"] == "identity_confirmed")
        & (reviewed_g["manual_assigned_candidate_box_id"].astype(str).str.strip() != "")
    ]

    pending_g = reviewed_g[~reviewed_g.index.isin(usable_g.index)]

    if len(reviewed_g) == 0:
        scan_decision = "not_reviewed"
        classification_scope = "none"
    elif len(reviewed_g) < 6:
        scan_decision = "partially_reviewed"
        classification_scope = "object_level_only"
    elif len(usable_g) == 6:
        scan_decision = "full_scanframe_gold"
        classification_scope = "full_scanframe_usable"
    elif len(usable_g) > 0:
        scan_decision = "mixed_object_level_gt"
        classification_scope = "object_level_only"
    else:
        scan_decision = "no_usable_objects"
        classification_scope = "exclude"

    scan_rows.append({
        "scan_frame_id": scan,
        "video_id": g["video_id"].iloc[0] if "video_id" in g.columns else "",
        "total_expected_objects": int(len(g)),
        "reviewed_objects": int(len(reviewed_g)),
        "gold_usable_objects": int(len(usable_g)),
        "pending_or_fix_objects": int(len(pending_g)),
        "reviewed_colours": ";".join(reviewed_g["canonical_colour_label_norm"].tolist()),
        "gold_usable_colours": ";".join(usable_g["canonical_colour_label_norm"].tolist()),
        "pending_or_fix_colours": ";".join(pending_g["canonical_colour_label_norm"].tolist()),
        "scanframe_gt_v2_decision": scan_decision,
        "classification_scope": classification_scope,
    })

scan_summary = pd.DataFrame(scan_rows).sort_values("scan_frame_id")
safe_to_csv(scan_summary, OUT_SCANFRAME)

counts_rows = []

for col in [
    "manual_gt_v2_status",
    "manual_classification_use",
    "manual_bbox_status",
    "manual_identity_status",
    "object_gt_v2_decision",
]:
    vc = reviewed[col].value_counts().reset_index()
    vc.columns = ["category", "count"]
    vc["summary_type"] = col
    counts_rows.append(vc)

scan_vc = scan_summary["scanframe_gt_v2_decision"].value_counts().reset_index()
scan_vc.columns = ["category", "count"]
scan_vc["summary_type"] = "scanframe_gt_v2_decision"
counts_rows.append(scan_vc)

counts = pd.concat(counts_rows, ignore_index=True)[["summary_type", "category", "count"]]
safe_to_csv(counts, OUT_COUNTS)

# Issues are not failures; they document remaining work.
if len(reviewed) == 0:
    issues.append({
        "item": "assignments",
        "issue_type": "hard_no_reviewed_assignments",
        "issue_detail": "No reviewed assignment rows found.",
        "severity": "hard",
    })

if len(pending):
    issues.append({
        "item": "assignments",
        "issue_type": "info_pending_or_fix_required_objects",
        "issue_detail": f"{len(pending)} reviewed objects are pending/fix/excluded, not classification usable.",
        "severity": "info",
    })

if len(usable) == 0:
    issues.append({
        "item": "usable_objects",
        "issue_type": "hard_no_gold_usable_objects",
        "issue_detail": "No gold usable objects found.",
        "severity": "hard",
    })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

full_gold_scanframes = int((scan_summary["scanframe_gt_v2_decision"] == "full_scanframe_gold").sum())
mixed_scanframes = int((scan_summary["scanframe_gt_v2_decision"] == "mixed_object_level_gt").sum())
not_reviewed_scanframes = int((scan_summary["scanframe_gt_v2_decision"] == "not_reviewed").sum())

decision = pd.DataFrame([{
    "v63c_decision": "reviewed_assignment_consolidation_completed" if hard_issue_count == 0 else "reviewed_assignment_consolidation_blocked",
    "total_assignment_rows": int(len(df)),
    "reviewed_rows": int(len(reviewed)),
    "pending_rows": int(len(df) - len(reviewed)),
    "reviewed_scanframes": int(reviewed["scan_frame_id"].nunique()) if len(reviewed) else 0,
    "gold_usable_object_count": int(len(usable)),
    "pending_or_fix_object_count": int(len(pending)),
    "full_gold_scanframe_count": full_gold_scanframes,
    "mixed_object_level_scanframe_count": mixed_scanframes,
    "not_reviewed_scanframe_count": not_reviewed_scanframes,
    "gold_usable_output": str(OUT_USABLE),
    "pending_or_fix_output": str(OUT_PENDING),
    "hard_issue_count": hard_issue_count,
    "info_count": info_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v64_manual_gt_v2_continuation": bool(hard_issue_count == 0),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v63c Reviewed Assignment Consolidation\n\n"
    f"- v63c decision: {decision.iloc[0]['v63c_decision']}\n"
    f"- Total assignment rows: {len(df)}\n"
    f"- Reviewed rows: {len(reviewed)}\n"
    f"- Pending rows: {len(df) - len(reviewed)}\n"
    f"- Reviewed scanframes: {reviewed['scan_frame_id'].nunique() if len(reviewed) else 0}\n"
    f"- Gold usable objects: {len(usable)}\n"
    f"- Pending/fix objects: {len(pending)}\n"
    f"- Full gold scanframes: {full_gold_scanframes}\n"
    f"- Mixed object-level scanframes: {mixed_scanframes}\n"
    f"- Not reviewed scanframes: {not_reviewed_scanframes}\n"
    f"- Ready for v64 manual GT v2 continuation: {bool(hard_issue_count == 0)}\n\n"
    "Interpretation: classification must use object-level gold rows only unless a full scanframe has six gold usable assignments.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v63c Reviewed Assignment Consolidation Report\n\n"
    f"Decision: {decision.iloc[0]['v63c_decision']}\n\n"
    f"Gold usable object GT: `{OUT_USABLE}`\n\n"
    f"Pending/fix object GT: `{OUT_PENDING}`\n\n"
    f"Scanframe summary: `{OUT_SCANFRAME}`\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v63c",
    "task_name": "Reviewed assignment consolidation",
    "status": "PASS" if hard_issue_count == 0 else "BLOCKED",
    "input_summary": str(ASSIGNMENTS),
    "output_summary": str(OUT),
    "hard_issues": hard_issue_count,
    "warnings": 0,
    "next_action": "Continue manual GT v2 assignment for more P1/P2 scanframes or export object-level gold subset.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_REVIEWED)
print(OUT_USABLE)
print(OUT_PENDING)
print(OUT_SCANFRAME)
print(OUT_COUNTS)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_NOTE)

print()
print("=== v63c decision ===")
print(decision.to_string(index=False))

print()
print("=== counts ===")
print(counts.to_string(index=False))

print()
print("=== scanframe summary for reviewed scanframes ===")
print(scan_summary[scan_summary["scanframe_gt_v2_decision"] != "not_reviewed"].to_string(index=False))

print()
print("=== issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
