from pathlib import Path
from datetime import datetime
import csv
import pandas as pd


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V16_ROOT = W7 / "outputs" / "colour_identity" / "manual_colour_identity_v16"
V16_ASSIGN = V16_ROOT / "week7_manual_colour_identity_v16_final_assignments.csv"
V16_FRAME_SUMMARY = V16_ROOT / "week7_manual_colour_identity_v16_frame_summary.csv"

FINAL_BOXES = W7 / "outputs" / "final_corrected_gt_pen_boxes_v11" / "week7_final_corrected_gt_pen_boxes_v11_for_colour_matching.csv"

OUT_ROOT = W7 / "outputs" / "colour_identity" / "final_colour_identity_v17"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_LOCKED = OUT_ROOT / "week7_final_colour_identity_v17_locked_assignments.csv"
OUT_FRAME_QA = OUT_ROOT / "week7_final_colour_identity_v17_frame_qa.csv"
OUT_HARD_ISSUES = OUT_ROOT / "week7_final_colour_identity_v17_hard_issues.csv"
OUT_SUMMARY = OUT_ROOT / "week7_final_colour_identity_v17_summary.csv"
OUT_NOTE = W7 / "notes" / "week7_final_colour_identity_v17_notes.md"

VALID = ["blue", "green", "cyan", "red", "pink", "purple"]


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


assign = pd.read_csv(V16_ASSIGN)
boxes = pd.read_csv(FINAL_BOXES)

# Merge with final box metadata.
merged = boxes.merge(
    assign,
    on=["scan_frame_id", "final_box_id", "x1", "y1", "x2", "y2"],
    how="left",
    suffixes=("", "_v16")
)

def final_status(row):
    colour = str(row.get("final_colour_v16", "")).strip().lower()
    manual_status = str(row.get("manual_status_v16", "")).strip().lower()
    manual_sel = str(row.get("manual_selection_v16", "")).strip().lower()

    if colour in VALID:
        return "usable_colour_identity"

    if manual_status == "not_visible" or manual_sel == "not_visible":
        return "identity_unknown_not_visible"

    if manual_status == "uncertain" or manual_sel == "uncertain":
        return "identity_unknown_uncertain"

    return "identity_unknown_unassigned"


def final_colour(row):
    colour = str(row.get("final_colour_v16", "")).strip().lower()
    return colour if colour in VALID else ""


merged["final_colour_identity_v17"] = merged.apply(final_colour, axis=1)
merged["final_identity_status_v17"] = merged.apply(final_status, axis=1)
merged["locked_for_behaviour_fusion_v17"] = True
merged["locked_at_v17"] = datetime.now().isoformat(timespec="seconds")

# Frame-level QA.
frame_rows = []
hard_rows = []

for sid, g in merged.groupby("scan_frame_id", sort=True):
    valid_colours = [c for c in g["final_colour_identity_v17"].tolist() if c in VALID]
    counts = pd.Series(valid_colours).value_counts()
    duplicates = [c for c, n in counts.items() if n > 1]
    missing = sorted(set(VALID) - set(valid_colours))

    not_visible = int((g["final_identity_status_v17"] == "identity_unknown_not_visible").sum())
    uncertain = int((g["final_identity_status_v17"] == "identity_unknown_uncertain").sum())
    unassigned = int((g["final_identity_status_v17"] == "identity_unknown_unassigned").sum())
    usable = int((g["final_identity_status_v17"] == "usable_colour_identity").sum())

    hard_issue = bool(len(duplicates) > 0 or unassigned > 0)

    frame_rows.append({
        "scan_frame_id": sid,
        "box_count": int(len(g)),
        "usable_colour_identity_count": usable,
        "not_visible_count": not_visible,
        "uncertain_count": uncertain,
        "unassigned_count": unassigned,
        "duplicate_colours": " | ".join(duplicates),
        "missing_colours": " | ".join(missing),
        "hard_issue": hard_issue,
        "soft_issue": bool(uncertain > 0 or not_visible > 0),
    })

    if len(duplicates) > 0:
        hard_rows.append({
            "scan_frame_id": sid,
            "issue_type": "duplicate_colour",
            "issue_detail": " | ".join(duplicates),
        })

    if unassigned > 0:
        hard_rows.append({
            "scan_frame_id": sid,
            "issue_type": "unassigned_identity",
            "issue_detail": f"{unassigned} unassigned boxes",
        })

frame_qa = pd.DataFrame(frame_rows)
hard_issues = pd.DataFrame(hard_rows)

summary = pd.DataFrame([{
    "total_boxes": int(len(merged)),
    "usable_colour_identity_boxes": int((merged["final_identity_status_v17"] == "usable_colour_identity").sum()),
    "identity_unknown_not_visible_boxes": int((merged["final_identity_status_v17"] == "identity_unknown_not_visible").sum()),
    "identity_unknown_uncertain_boxes": int((merged["final_identity_status_v17"] == "identity_unknown_uncertain").sum()),
    "identity_unknown_unassigned_boxes": int((merged["final_identity_status_v17"] == "identity_unknown_unassigned").sum()),
    "frames_total": int(frame_qa["scan_frame_id"].nunique()),
    "frames_with_hard_issue": int(frame_qa["hard_issue"].sum()),
    "frames_with_soft_issue": int(frame_qa["soft_issue"].sum()),
    "hard_issue_count": int(len(hard_issues)),
    "ready_for_behaviour_fusion": bool(len(hard_issues) == 0),
}])

safe_to_csv(merged, OUT_LOCKED)
safe_to_csv(frame_qa, OUT_FRAME_QA)
safe_to_csv(hard_issues, OUT_HARD_ISSUES)
safe_to_csv(summary, OUT_SUMMARY)

ready = bool(summary.iloc[0]["ready_for_behaviour_fusion"])

OUT_NOTE.write_text(
    "# Week 7 Final Colour Identity Lock v17\n\n"
    "## Purpose\n\n"
    "This step locks the manually corrected colour identity table for behaviour-label fusion.\n\n"
    "## Policy\n\n"
    "- Valid colour labels are kept as usable identities.\n"
    "- `not_visible` is kept as `identity_unknown_not_visible`.\n"
    "- `uncertain` is kept as `identity_unknown_uncertain`.\n"
    "- Duplicate colours or unassigned boxes are hard issues.\n\n"
    "## Summary\n\n"
    f"- Total boxes: `{int(summary.iloc[0]['total_boxes'])}`\n"
    f"- Usable colour identity boxes: `{int(summary.iloc[0]['usable_colour_identity_boxes'])}`\n"
    f"- Not-visible unknown boxes: `{int(summary.iloc[0]['identity_unknown_not_visible_boxes'])}`\n"
    f"- Uncertain unknown boxes: `{int(summary.iloc[0]['identity_unknown_uncertain_boxes'])}`\n"
    f"- Unassigned boxes: `{int(summary.iloc[0]['identity_unknown_unassigned_boxes'])}`\n"
    f"- Frames with hard issue: `{int(summary.iloc[0]['frames_with_hard_issue'])}`\n"
    f"- Frames with soft issue: `{int(summary.iloc[0]['frames_with_soft_issue'])}`\n"
    f"- Ready for behaviour fusion: `{ready}`\n\n"
    "## Outputs\n\n"
    f"- Locked assignments: `{OUT_LOCKED}`\n"
    f"- Frame QA: `{OUT_FRAME_QA}`\n"
    f"- Hard issues: `{OUT_HARD_ISSUES}`\n"
    f"- Summary: `{OUT_SUMMARY}`\n"
)

print("Saved:")
print(OUT_LOCKED)
print(OUT_FRAME_QA)
print(OUT_HARD_ISSUES)
print(OUT_SUMMARY)
print(OUT_NOTE)

print()
print("=== v17 final colour identity lock summary ===")
print(summary.to_string(index=False))

if ready:
    print()
    print("READY_FOR_BEHAVIOUR_FUSION=True")
else:
    print()
    print("READY_FOR_BEHAVIOUR_FUSION=False")
    print("Hard issues:")
    print(hard_issues.to_string(index=False))
