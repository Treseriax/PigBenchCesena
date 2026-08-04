from pathlib import Path
from datetime import datetime
import shutil
import pandas as pd
import csv

ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

ASSIGNMENTS = W8 / "validation" / "week8_v63b_manual_gt_v2_assignments.csv"
INCONSISTENCIES = W8 / "outputs" / "v64b_full_manual_gt_v2_audit" / "week8_v64b_assignment_inconsistencies.csv"

OUT = W8 / "outputs" / "v64b2_assignment_consistency_fix"
OUT.mkdir(parents=True, exist_ok=True)

PATCH_LOG = OUT / "week8_v64b2_assignment_consistency_fix_log.csv"
DECISION = OUT / "week8_v64b2_decision_summary.csv"
NOTE = W8 / "notes" / "week8_v64b2_assignment_consistency_fix_notes.md"


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


if not ASSIGNMENTS.exists():
    raise FileNotFoundError(ASSIGNMENTS)

if not INCONSISTENCIES.exists():
    raise FileNotFoundError(INCONSISTENCIES)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = ASSIGNMENTS.with_name(f"{ASSIGNMENTS.stem}_backup_before_v64b2_{timestamp}.csv")
shutil.copy2(ASSIGNMENTS, backup)

df = pd.read_csv(ASSIGNMENTS).fillna("")
inc = pd.read_csv(INCONSISTENCIES).fillna("")

for c in df.columns:
    df[c] = df[c].map(clean)

for c in inc.columns:
    inc[c] = inc[c].map(clean)

hard = inc[
    (inc["audit_severity"] == "hard")
    & (inc["audit_issue_type"] == "non_gold_marked_use_for_classification")
].copy()

patch_rows = []

for _, r in hard.iterrows():
    obj_id = clean(r.get("canonical_gt_object_id"))
    scan = clean(r.get("scan_frame_id"))
    colour = clean(r.get("canonical_colour_label_norm"))

    mask = (
        (df["scan_frame_id"] == scan)
        & (df["canonical_colour_label_norm"] == colour)
    )

    if not mask.any():
        patch_rows.append({
            "canonical_gt_object_id": obj_id,
            "scan_frame_id": scan,
            "canonical_colour_label_norm": colour,
            "old_manual_classification_use": "",
            "new_manual_classification_use": "",
            "patch_status": "not_found",
        })
        continue

    idx = df.index[mask][0]
    old_use = df.at[idx, "manual_classification_use"]
    status = df.at[idx, "manual_gt_v2_status"]

    if status == "fix_required":
        new_use = "pending_review"
    elif status == "red_exclude":
        new_use = "exclude_from_classification"
    elif status == "unknown_pending_review":
        new_use = "pending_review"
    else:
        new_use = old_use

    df.at[idx, "manual_classification_use"] = new_use
    df.at[idx, "assignment_updated_at"] = datetime.now().isoformat(timespec="seconds")

    patch_rows.append({
        "canonical_gt_object_id": obj_id,
        "scan_frame_id": scan,
        "canonical_colour_label_norm": colour,
        "manual_gt_v2_status": status,
        "old_manual_classification_use": old_use,
        "new_manual_classification_use": new_use,
        "patch_status": "patched",
    })

safe_to_csv(df, ASSIGNMENTS)

patch_log = pd.DataFrame(patch_rows)
safe_to_csv(patch_log, PATCH_LOG)

decision = pd.DataFrame([{
    "v64b2_decision": "assignment_consistency_fix_completed",
    "assignment_csv": str(ASSIGNMENTS),
    "backup_csv": str(backup),
    "hard_inconsistency_rows_input": int(len(hard)),
    "patched_rows": int((patch_log["patch_status"] == "patched").sum()) if len(patch_log) else 0,
    "patch_log": str(PATCH_LOG),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, DECISION)

NOTE.write_text(
    "# Week 8 v64b2 Assignment Consistency Fix\n\n"
    f"- v64b2 decision: {decision.iloc[0]['v64b2_decision']}\n"
    f"- Backup created: {backup}\n"
    f"- Patched rows: {decision.iloc[0]['patched_rows']}\n"
    f"- Patch log: {PATCH_LOG}\n\n"
    "Rule applied: fix_required and unknown_pending_review rows cannot be use_for_classification. They are set to pending_review. red_exclude rows are set to exclude_from_classification.\n"
)

print("=== v64b2 decision ===")
print(decision.to_string(index=False))
print()
print("=== patch log ===")
print(patch_log.to_string(index=False))
