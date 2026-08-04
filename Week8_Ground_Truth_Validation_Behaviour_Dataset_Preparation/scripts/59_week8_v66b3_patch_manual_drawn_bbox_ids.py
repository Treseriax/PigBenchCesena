from pathlib import Path
from datetime import datetime
import shutil
import csv
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

ASSIGNMENTS = W8 / "validation" / "week8_v63b_manual_gt_v2_assignments.csv"

OUT = W8 / "outputs" / "v66b3_patch_manual_drawn_bbox_ids"
OUT.mkdir(parents=True, exist_ok=True)

PATCH_LOG = OUT / "week8_v66b3_patch_manual_drawn_bbox_ids_log.csv"
DECISION = OUT / "week8_v66b3_decision_summary.csv"
ISSUES = OUT / "week8_v66b3_issues.csv"
NOTE = W8 / "notes" / "week8_v66b3_patch_manual_drawn_bbox_ids_notes.md"


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def clean(x):
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
        "issue_detail": "Assignment CSV missing.",
        "severity": "hard",
    })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, ISSUES)
    raise SystemExit(1)


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = ASSIGNMENTS.with_name(f"{ASSIGNMENTS.stem}_backup_before_v66b3_{timestamp}.csv")
shutil.copy2(ASSIGNMENTS, backup)

df = pd.read_csv(ASSIGNMENTS).fillna("")
for c in df.columns:
    df[c] = df[c].map(clean)

mask = (
    (df["manual_gt_v2_status"] == "gold_usable")
    & (df["manual_classification_use"] == "use_for_classification")
    & (df["manual_bbox_status"] == "bbox_ok")
    & (df["manual_identity_status"] == "identity_confirmed")
    & (df["manual_assigned_candidate_box_id"] == "")
    & (df["manual_reviewer_note"].str.contains("v66b2", case=False, na=False))
)

patch_rows = []

for idx, r in df[mask].iterrows():
    obj_id = r["canonical_gt_object_id"]
    new_id = f"{obj_id}__manual_bbox_v66b2"

    df.at[idx, "manual_assigned_candidate_box_id"] = new_id

    if "bbox_source_of_truth" in df.columns:
        df.at[idx, "bbox_source_of_truth"] = "manual_drawn_bbox_v66b2"

    if "identity_assignment_source_of_truth" in df.columns:
        df.at[idx, "identity_assignment_source_of_truth"] = "manual_gt_v2_review_v66b2"

    df.at[idx, "assignment_updated_at"] = datetime.now().isoformat(timespec="seconds")

    patch_rows.append({
        "canonical_gt_object_id": obj_id,
        "scan_frame_id": r["scan_frame_id"],
        "canonical_colour_label_norm": r["canonical_colour_label_norm"],
        "old_manual_assigned_candidate_box_id": "",
        "new_manual_assigned_candidate_box_id": new_id,
        "patch_status": "patched_manual_drawn_bbox_id",
    })

safe_to_csv(df, ASSIGNMENTS)

patch_log = pd.DataFrame(patch_rows)
safe_to_csv(patch_log, PATCH_LOG)

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, ISSUES)

decision = pd.DataFrame([{
    "v66b3_decision": "manual_drawn_bbox_ids_patched",
    "assignment_csv": str(ASSIGNMENTS),
    "backup_csv": str(backup),
    "patched_rows": int(len(patch_log)),
    "patch_log": str(PATCH_LOG),
    "hard_issue_count": 0,
    "ready_to_rerun_v64b_audit": True,
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, DECISION)

NOTE.write_text(
    "# Week 8 v66b3 Patch Manual Drawn BBox IDs\n\n"
    f"- v66b3 decision: {decision.iloc[0]['v66b3_decision']}\n"
    f"- Patched rows: {len(patch_log)}\n"
    f"- Backup: {backup}\n"
    f"- Patch log: {PATCH_LOG}\n\n"
    "Rows manually drawn in v66b2 now receive synthetic manual bbox IDs, because they are not detector candidate boxes.\n"
)

print("=== v66b3 decision ===")
print(decision.to_string(index=False))
print()
print("=== patch log ===")
print(patch_log.to_string(index=False))
