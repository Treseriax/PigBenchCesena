from pathlib import Path
from datetime import datetime
import csv
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

ASSIGNMENTS = W8 / "validation" / "week8_v63b_manual_gt_v2_assignments.csv"

OUT = W8 / "outputs" / "v64b_full_manual_gt_v2_audit"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_STRICT_GOLD = OUT / "week8_v64b_strict_gold_usable_object_gt.csv"
OUT_CAUTION = OUT / "week8_v64b_caution_usable_object_gt.csv"
OUT_NONUSABLE = OUT / "week8_v64b_nonusable_or_fix_required_object_gt.csv"
OUT_SCANFRAME = OUT / "week8_v64b_scanframe_quality_summary.csv"
OUT_INCONSISTENCIES = OUT / "week8_v64b_assignment_inconsistencies.csv"
OUT_COUNTS = OUT / "week8_v64b_counts.csv"
OUT_DECISION = OUT / "week8_v64b_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v64b_issues.csv"
OUT_NOTE = NOTES / "week8_v64b_full_manual_gt_v2_audit_notes.md"
OUT_REPORT = REPORTS / "week8_v64b_full_manual_gt_v2_audit_report.md"
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
        "issue_type": "hard_missing_assignments",
        "issue_detail": "Manual GT v2 assignment CSV is missing.",
        "severity": "hard",
    })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v64b_decision": "full_manual_gt_v2_audit_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v64c_final_gt_v2_export": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


df = pd.read_csv(ASSIGNMENTS).fillna("")
for c in df.columns:
    df[c] = df[c].map(clean_str)

df["is_reviewed"] = df["manual_gt_v2_status"] != ""

strict_gold_mask = (
    (df["manual_gt_v2_status"] == "gold_usable")
    & (df["manual_classification_use"] == "use_for_classification")
    & (df["manual_bbox_status"] == "bbox_ok")
    & (df["manual_identity_status"] == "identity_confirmed")
    & (df["manual_assigned_candidate_box_id"] != "")
)

caution_mask = (
    (df["manual_gt_v2_status"] == "gold_usable")
    & (df["manual_classification_use"] == "use_for_classification_with_caution")
    & (df["manual_bbox_status"] == "bbox_ok")
    & (df["manual_identity_status"] == "identity_confirmed")
    & (df["manual_assigned_candidate_box_id"] != "")
)

df["strict_gold_usable_for_classification"] = strict_gold_mask
df["caution_usable_for_analysis"] = caution_mask

strict_gold = df[strict_gold_mask].copy()
caution = df[caution_mask].copy()
nonusable = df[~strict_gold_mask & ~caution_mask].copy()

safe_to_csv(strict_gold, OUT_STRICT_GOLD)
safe_to_csv(caution, OUT_CAUTION)
safe_to_csv(nonusable, OUT_NONUSABLE)

inconsistency_rows = []

def add_inconsistency(mask, issue_type, severity, detail):
    sub = df[mask].copy()
    for _, r in sub.iterrows():
        d = r.to_dict()
        d["audit_issue_type"] = issue_type
        d["audit_severity"] = severity
        d["audit_detail"] = detail
        inconsistency_rows.append(d)

add_inconsistency(
    (df["manual_gt_v2_status"].isin(["fix_required", "red_exclude", "unknown_pending_review"]))
    & (df["manual_classification_use"] == "use_for_classification"),
    "non_gold_marked_use_for_classification",
    "hard",
    "fix_required/red_exclude/unknown_pending_review should not be marked use_for_classification.",
)

add_inconsistency(
    (df["manual_gt_v2_status"] == "gold_usable")
    & (df["manual_classification_use"] == "use_for_classification")
    & (df["manual_bbox_status"] != "bbox_ok"),
    "gold_classification_but_bbox_not_ok",
    "hard",
    "Gold classification row must have bbox_ok.",
)

add_inconsistency(
    (df["manual_gt_v2_status"] == "gold_usable")
    & (df["manual_classification_use"] == "use_for_classification")
    & (df["manual_identity_status"] != "identity_confirmed"),
    "gold_classification_but_identity_not_confirmed",
    "hard",
    "Gold classification row must have identity_confirmed.",
)

add_inconsistency(
    (df["manual_gt_v2_status"] == "gold_usable")
    & (df["manual_classification_use"] == "use_for_classification")
    & (df["manual_assigned_candidate_box_id"] == ""),
    "gold_classification_missing_candidate_box",
    "hard",
    "Gold classification row must have assigned candidate box.",
)

add_inconsistency(
    (df["manual_gt_v2_status"] == "gold_usable")
    & (df["manual_classification_use"] == "use_for_classification_with_caution"),
    "gold_usable_but_caution_classification",
    "info",
    "This row is valid as caution usable, but excluded from strict gold classification subset.",
)

inconsistencies = pd.DataFrame(inconsistency_rows)
safe_to_csv(inconsistencies, OUT_INCONSISTENCIES)

scan_rows = []
for scan, g in df.groupby("scan_frame_id"):
    sg = g[g["strict_gold_usable_for_classification"]]
    cg = g[g["caution_usable_for_analysis"]]
    ng = g[~g["strict_gold_usable_for_classification"] & ~g["caution_usable_for_analysis"]]

    if len(sg) == 6:
        scan_decision = "full_strict_gold"
        classification_scope = "full_scanframe_classification"
    elif len(sg) + len(cg) == 6:
        scan_decision = "full_usable_with_caution"
        classification_scope = "full_scanframe_with_caution"
    elif len(sg) > 0:
        scan_decision = "mixed_object_level_strict_gold"
        classification_scope = "object_level_classification_only"
    elif len(cg) > 0:
        scan_decision = "mixed_object_level_caution_only"
        classification_scope = "caution_analysis_only"
    else:
        scan_decision = "no_classification_usable_object"
        classification_scope = "exclude_scanframe"

    scan_rows.append({
        "scan_frame_id": scan,
        "video_id": g["video_id"].iloc[0],
        "total_objects": int(len(g)),
        "strict_gold_objects": int(len(sg)),
        "caution_objects": int(len(cg)),
        "nonusable_or_fix_objects": int(len(ng)),
        "strict_gold_colours": ";".join(sg["canonical_colour_label_norm"].tolist()),
        "caution_colours": ";".join(cg["canonical_colour_label_norm"].tolist()),
        "nonusable_or_fix_colours": ";".join(ng["canonical_colour_label_norm"].tolist()),
        "scanframe_quality_decision": scan_decision,
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
]:
    vc = df[col].value_counts().reset_index()
    vc.columns = ["category", "count"]
    vc["summary_type"] = col
    counts_rows.append(vc)

scan_vc = scan_summary["scanframe_quality_decision"].value_counts().reset_index()
scan_vc.columns = ["category", "count"]
scan_vc["summary_type"] = "scanframe_quality_decision"
counts_rows.append(scan_vc)

extra = pd.DataFrame([
    {"summary_type": "strict_gold_object_count", "category": "strict_gold", "count": len(strict_gold)},
    {"summary_type": "caution_object_count", "category": "caution", "count": len(caution)},
    {"summary_type": "nonusable_or_fix_object_count", "category": "nonusable_or_fix", "count": len(nonusable)},
    {"summary_type": "review_completion", "category": "reviewed_rows", "count": int(df["is_reviewed"].sum())},
    {"summary_type": "review_completion", "category": "pending_rows", "count": int((~df["is_reviewed"]).sum())},
])
counts_rows.append(extra)

counts = pd.concat(counts_rows, ignore_index=True)[["summary_type", "category", "count"]]
safe_to_csv(counts, OUT_COUNTS)

hard_inconsistency_count = int((inconsistencies["audit_severity"] == "hard").sum()) if len(inconsistencies) else 0
info_inconsistency_count = int((inconsistencies["audit_severity"] == "info").sum()) if len(inconsistencies) else 0

if hard_inconsistency_count > 0:
    issues.append({
        "item": "manual_assignments",
        "issue_type": "hard_assignment_consistency_errors",
        "issue_detail": f"{hard_inconsistency_count} hard consistency errors found in assignment CSV.",
        "severity": "hard",
    })

if int(df["is_reviewed"].sum()) != 432:
    issues.append({
        "item": "manual_assignments",
        "issue_type": "hard_not_all_rows_reviewed",
        "issue_detail": f"Reviewed rows: {int(df['is_reviewed'].sum())}; expected 432.",
        "severity": "hard",
    })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0

decision = pd.DataFrame([{
    "v64b_decision": "full_manual_gt_v2_audit_passed" if hard_issue_count == 0 else "full_manual_gt_v2_audit_needs_consistency_fix",
    "total_rows": int(len(df)),
    "reviewed_rows": int(df["is_reviewed"].sum()),
    "pending_rows": int((~df["is_reviewed"]).sum()),
    "strict_gold_usable_objects": int(len(strict_gold)),
    "caution_usable_objects": int(len(caution)),
    "nonusable_or_fix_objects": int(len(nonusable)),
    "full_strict_gold_scanframes": int((scan_summary["scanframe_quality_decision"] == "full_strict_gold").sum()),
    "full_usable_with_caution_scanframes": int((scan_summary["scanframe_quality_decision"] == "full_usable_with_caution").sum()),
    "mixed_object_level_scanframes": int(scan_summary["scanframe_quality_decision"].str.contains("mixed").sum()),
    "no_classification_usable_scanframes": int((scan_summary["scanframe_quality_decision"] == "no_classification_usable_object").sum()),
    "hard_inconsistency_count": hard_inconsistency_count,
    "info_inconsistency_count": info_inconsistency_count,
    "hard_issue_count": hard_issue_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v64c_final_gt_v2_export": bool(hard_issue_count == 0),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v64b Full Manual GT v2 Audit\n\n"
    f"- v64b decision: {decision.iloc[0]['v64b_decision']}\n"
    f"- Total rows: {len(df)}\n"
    f"- Reviewed rows: {int(df['is_reviewed'].sum())}\n"
    f"- Pending rows: {int((~df['is_reviewed']).sum())}\n"
    f"- Strict gold usable objects: {len(strict_gold)}\n"
    f"- Caution usable objects: {len(caution)}\n"
    f"- Nonusable/fix objects: {len(nonusable)}\n"
    f"- Full strict-gold scanframes: {int((scan_summary['scanframe_quality_decision'] == 'full_strict_gold').sum())}\n"
    f"- Mixed object-level scanframes: {int(scan_summary['scanframe_quality_decision'].str.contains('mixed').sum())}\n"
    f"- No classification-usable scanframes: {int((scan_summary['scanframe_quality_decision'] == 'no_classification_usable_object').sum())}\n"
    f"- Hard inconsistencies: {hard_inconsistency_count}\n"
    f"- Ready for v64c final GT v2 export: {bool(hard_issue_count == 0)}\n\n"
    "Strict classification GT requires: gold_usable + use_for_classification + bbox_ok + identity_confirmed + assigned candidate box.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v64b Full Manual GT v2 Audit Report\n\n"
    f"Decision: {decision.iloc[0]['v64b_decision']}\n\n"
    f"Strict gold object GT: `{OUT_STRICT_GOLD}`\n\n"
    f"Caution object GT: `{OUT_CAUTION}`\n\n"
    f"Nonusable/fix object GT: `{OUT_NONUSABLE}`\n\n"
    f"Inconsistencies: `{OUT_INCONSISTENCIES}`\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v64b",
    "task_name": "Full manual GT v2 audit",
    "status": "PASS" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": str(ASSIGNMENTS),
    "output_summary": str(OUT),
    "hard_issues": hard_issue_count,
    "warnings": 0,
    "next_action": "Fix hard inconsistencies if any, then export final GT v2.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_STRICT_GOLD)
print(OUT_CAUTION)
print(OUT_NONUSABLE)
print(OUT_SCANFRAME)
print(OUT_INCONSISTENCIES)
print(OUT_COUNTS)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_NOTE)

print()
print("=== v64b decision ===")
print(decision.to_string(index=False))

print()
print("=== inconsistencies ===")
if len(inconsistencies):
    cols = [
        "scan_frame_id",
        "canonical_colour_label_norm",
        "behaviour_code",
        "manual_gt_v2_status",
        "manual_classification_use",
        "manual_bbox_status",
        "manual_identity_status",
        "manual_assigned_candidate_box_id",
        "manual_reviewer_note",
        "audit_issue_type",
        "audit_severity",
    ]
    print(inconsistencies[cols].to_string(index=False))
else:
    print("No inconsistencies found.")

print()
print("=== scanframe quality counts ===")
print(scan_summary["scanframe_quality_decision"].value_counts().to_string())

print()
print("=== issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
