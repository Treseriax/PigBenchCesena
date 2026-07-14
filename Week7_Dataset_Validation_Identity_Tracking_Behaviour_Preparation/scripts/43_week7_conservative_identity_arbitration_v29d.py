from pathlib import Path
from datetime import datetime
import csv
import pandas as pd
import numpy as np


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V29B_TRACKLETS = W7 / "outputs" / "colour_constrained_tracklet_linking_v29b" / "week7_colour_constrained_tracklet_linking_v29b_tracklet_identity_candidates.csv"
V29B_CLIP_SUMMARY = W7 / "outputs" / "colour_constrained_tracklet_linking_v29b" / "week7_colour_constrained_tracklet_linking_v29b_clip_summary.csv"

V29C_TRACKLETS = W7 / "outputs" / "full_tracklet_colour_evidence_v29c" / "week7_full_tracklet_colour_evidence_v29c_tracklet_colour_evidence.csv"
V29C_CLIP_SUMMARY = W7 / "outputs" / "full_tracklet_colour_evidence_v29c" / "week7_full_tracklet_colour_evidence_v29c_clip_summary.csv"
V29C_MERGE = W7 / "outputs" / "full_tracklet_colour_evidence_v29c" / "week7_full_tracklet_colour_evidence_v29c_same_colour_merge_candidates.csv"

V28D_CLIP_SUMMARY = W7 / "outputs" / "dense_polygon_filtered_tracking_v28d" / "week7_dense_polygon_filtered_tracking_v28d_clip_summary.csv"

OUT_ROOT = W7 / "outputs" / "conservative_identity_arbitration_v29d"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_ARBITRATION = OUT_ROOT / "week7_conservative_identity_arbitration_v29d_tracklet_arbitration.csv"
OUT_CLIP_SUMMARY = OUT_ROOT / "week7_conservative_identity_arbitration_v29d_clip_summary.csv"
OUT_MERGE = OUT_ROOT / "week7_conservative_identity_arbitration_v29d_conservative_merge_candidates.csv"
OUT_REVIEW = OUT_ROOT / "week7_conservative_identity_arbitration_v29d_review_queue.csv"
OUT_DECISION = OUT_ROOT / "week7_conservative_identity_arbitration_v29d_decision_summary.csv"
OUT_REPORT = OUT_ROOT / "week7_conservative_identity_arbitration_v29d_report.md"
OUT_LIMITATIONS = OUT_ROOT / "week7_conservative_identity_arbitration_v29d_limitations.md"
OUT_ISSUES = OUT_ROOT / "week7_conservative_identity_arbitration_v29d_issues.csv"
OUT_README = OUT_ROOT / "README_conservative_identity_arbitration_v29d.md"
OUT_NOTE = W7 / "notes" / "week7_conservative_identity_arbitration_v29d_notes.md"

VALID_COLOURS = {"blue", "green", "cyan", "red", "pink", "purple"}
COLOUR_TO_BEHAVIOUR_ID = {
    "blue": "blue",
    "green": "green",
    "cyan": "no_color",
    "red": "red_neck",
    "pink": "red_tail",
    "purple": "purple",
}


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def clean(v):
    if pd.isna(v):
        return ""
    return str(v).strip()


def norm_colour(v):
    c = clean(v).lower()
    if c in VALID_COLOURS:
        return c
    return ""


def to_float(v, default=np.nan):
    try:
        x = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def has_high_medium(status):
    s = clean(status).lower()
    return ("high" in s) or ("medium" in s)


def has_high(status):
    return "high" in clean(status).lower()


def has_low(status):
    return "low" in clean(status).lower()


def arbitrate(row):
    b_colour = norm_colour(row.get("v29b_assigned_visual_colour", ""))
    c_colour = norm_colour(row.get("v29c_assigned_visual_colour", ""))

    b_status = clean(row.get("v29b_assignment_status", ""))
    c_status = clean(row.get("v29c_assignment_status", ""))

    b_behaviour = clean(row.get("v29b_assigned_behaviour_pig_id", ""))
    c_behaviour = clean(row.get("v29c_assigned_behaviour_pig_id", ""))

    b_has_valid = bool(b_colour)
    c_has_valid = bool(c_colour)

    c_hm = has_high_medium(c_status)
    c_low = has_low(c_status)

    # Rule 1: v29b valid GT-overlap assignment is the conservative primary identity.
    if b_has_valid:
        if c_has_valid and c_colour == b_colour and c_hm:
            return {
                "conservative_identity_accepted": True,
                "accepted_visual_colour": b_colour,
                "accepted_behaviour_pig_id": b_behaviour or COLOUR_TO_BEHAVIOUR_ID.get(b_colour, ""),
                "candidate_visual_colour": "",
                "candidate_behaviour_pig_id": "",
                "arbitration_status": "accepted_v29b_confirmed_by_v29c",
                "arbitration_confidence": "high",
                "review_required": False,
                "review_reason": "",
                "recommended_use": "use_as_conservative_identity",
            }

        if c_has_valid and c_colour != b_colour:
            return {
                "conservative_identity_accepted": True,
                "accepted_visual_colour": b_colour,
                "accepted_behaviour_pig_id": b_behaviour or COLOUR_TO_BEHAVIOUR_ID.get(b_colour, ""),
                "candidate_visual_colour": c_colour,
                "candidate_behaviour_pig_id": c_behaviour or COLOUR_TO_BEHAVIOUR_ID.get(c_colour, ""),
                "arbitration_status": "accepted_keep_v29b_conflict_with_v29c",
                "arbitration_confidence": "medium_review",
                "review_required": True,
                "review_reason": "v29b GT-overlap assignment conflicts with v29c HSV full-tracklet evidence; keep v29b but review.",
                "recommended_use": "use_v29b_identity_but_flag_for_review",
            }

        return {
            "conservative_identity_accepted": True,
            "accepted_visual_colour": b_colour,
            "accepted_behaviour_pig_id": b_behaviour or COLOUR_TO_BEHAVIOUR_ID.get(b_colour, ""),
            "candidate_visual_colour": "",
            "candidate_behaviour_pig_id": "",
            "arbitration_status": "accepted_from_v29b_only",
            "arbitration_confidence": "high_anchor",
            "review_required": False,
            "review_reason": "",
            "recommended_use": "use_as_conservative_identity",
        }

    # Rule 2: no valid v29b identity, but v29c has high/medium evidence.
    # This becomes candidate, not final accepted identity.
    if (not b_has_valid) and c_has_valid and c_hm:
        return {
            "conservative_identity_accepted": False,
            "accepted_visual_colour": "",
            "accepted_behaviour_pig_id": "",
            "candidate_visual_colour": c_colour,
            "candidate_behaviour_pig_id": c_behaviour or COLOUR_TO_BEHAVIOUR_ID.get(c_colour, ""),
            "arbitration_status": "recovered_candidate_from_v29c_needs_review",
            "arbitration_confidence": "candidate",
            "review_required": True,
            "review_reason": "v29b unmatched/unknown; v29c suggests high/medium full-tracklet colour evidence.",
            "recommended_use": "candidate_only_not_final_without_review",
        }

    # Rule 3: no v29b identity and only low v29c evidence.
    if (not b_has_valid) and c_has_valid and c_low:
        return {
            "conservative_identity_accepted": False,
            "accepted_visual_colour": "",
            "accepted_behaviour_pig_id": "",
            "candidate_visual_colour": c_colour,
            "candidate_behaviour_pig_id": c_behaviour or COLOUR_TO_BEHAVIOUR_ID.get(c_colour, ""),
            "arbitration_status": "low_confidence_v29c_candidate_review",
            "arbitration_confidence": "low",
            "review_required": True,
            "review_reason": "Only low HSV colour evidence is available.",
            "recommended_use": "do_not_use_as_final_identity",
        }

    return {
        "conservative_identity_accepted": False,
        "accepted_visual_colour": "",
        "accepted_behaviour_pig_id": "",
        "candidate_visual_colour": "",
        "candidate_behaviour_pig_id": "",
        "arbitration_status": "unknown_no_reliable_identity",
        "arbitration_confidence": "unknown",
        "review_required": True,
        "review_reason": "No valid v29b identity and no reliable v29c recovery.",
        "recommended_use": "unknown_do_not_force_identity",
    }


issues = []

required = [V29B_TRACKLETS, V29B_CLIP_SUMMARY, V29C_TRACKLETS, V29C_CLIP_SUMMARY, V28D_CLIP_SUMMARY]
for p in required:
    if not p.exists():
        issues.append({"issue_type": "missing_required_file", "issue_detail": str(p)})

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    print(issues_df.to_string(index=False))
    raise SystemExit(1)

v29b = pd.read_csv(V29B_TRACKLETS)
v29b_clip = pd.read_csv(V29B_CLIP_SUMMARY)
v29c = pd.read_csv(V29C_TRACKLETS)
v29c_clip = pd.read_csv(V29C_CLIP_SUMMARY)
v28d_clip = pd.read_csv(V28D_CLIP_SUMMARY)

for df in [v29b, v29b_clip, v29c, v29c_clip, v28d_clip]:
    if "scan_frame_id" in df.columns:
        df["scan_frame_id"] = df["scan_frame_id"].fillna("").astype(str).str.strip()
    if "clip_id" in df.columns:
        df["clip_id"] = df["clip_id"].fillna("").astype(str).str.strip()
    if "dense_track_id" in df.columns:
        df["dense_track_id"] = pd.to_numeric(df["dense_track_id"], errors="coerce").astype("Int64")

b = v29b.rename(columns={
    "assigned_visual_colour": "v29b_assigned_visual_colour",
    "assigned_behaviour_pig_id": "v29b_assigned_behaviour_pig_id",
    "assigned_behaviour_code_at_center": "v29b_behaviour_code_at_center",
    "center_match_iou_max": "v29b_center_match_iou_max",
    "assignment_status": "v29b_assignment_status",
    "tracklet_frames_seen": "v29b_tracklet_frames_seen",
})

b_cols = [
    "clip_id", "scan_frame_id", "dense_track_id",
    "v29b_tracklet_frames_seen",
    "tracklet_frame_min", "tracklet_frame_max",
    "tracklet_score_mean",
    "v29b_assigned_visual_colour",
    "v29b_assigned_behaviour_pig_id",
    "v29b_behaviour_code_at_center",
    "v29b_center_match_iou_max",
    "center_match_candidate_count",
    "v29b_assignment_status",
]
b = b[[c for c in b_cols if c in b.columns]].copy()

c = v29c.rename(columns={
    "v29c_assigned_visual_colour": "v29c_assigned_visual_colour",
    "v29c_assigned_behaviour_pig_id": "v29c_assigned_behaviour_pig_id",
    "v29c_assignment_status": "v29c_assignment_status",
})

c_cols = [
    "clip_id", "scan_frame_id", "dense_track_id",
    "tracklet_frames_seen",
    "non_unknown_colour_frames",
    "best_colour_evidence_frames",
    "v29c_assigned_visual_colour",
    "v29c_assigned_behaviour_pig_id",
    "v29c_assignment_status",
    "v29c_best_score_sum",
    "v29c_second_colour",
    "v29c_second_score_sum",
    "v29c_margin",
    "v29b_v29c_comparison_status",
]
c = c[[cc for cc in c_cols if cc in c.columns]].copy()
c = c.rename(columns={"tracklet_frames_seen": "v29c_tracklet_frames_seen"})

arb = b.merge(c, on=["clip_id", "scan_frame_id", "dense_track_id"], how="outer")

# Fill missing text.
for col in arb.columns:
    if arb[col].dtype == object:
        arb[col] = arb[col].fillna("")

# Apply arbitration.
arb_rows = []
for _, row in arb.iterrows():
    base = row.to_dict()
    decision = arbitrate(row)
    base.update(decision)
    arb_rows.append(base)

arb_df = pd.DataFrame(arb_rows)

# Useful flags.
arb_df["v29b_valid_colour"] = arb_df["v29b_assigned_visual_colour"].apply(lambda x: norm_colour(x) != "")
arb_df["v29c_valid_colour"] = arb_df["v29c_assigned_visual_colour"].apply(lambda x: norm_colour(x) != "")
arb_df["v29b_v29c_agree_on_colour"] = (
    arb_df["v29b_assigned_visual_colour"].apply(norm_colour)
    == arb_df["v29c_assigned_visual_colour"].apply(norm_colour)
) & arb_df["v29b_valid_colour"] & arb_df["v29c_valid_colour"]

safe_to_csv(arb_df, OUT_ARBITRATION)

# Review queue.
review_df = arb_df[
    (arb_df["review_required"] == True)
    | (arb_df["arbitration_status"].astype(str).str.contains("conflict|candidate|unknown|low", regex=True))
].copy()

review_order = {
    "accepted_keep_v29b_conflict_with_v29c": 1,
    "recovered_candidate_from_v29c_needs_review": 2,
    "low_confidence_v29c_candidate_review": 3,
    "unknown_no_reliable_identity": 4,
}
review_df["review_priority"] = review_df["arbitration_status"].map(review_order).fillna(9).astype(int)
review_df = review_df.sort_values(["review_priority", "clip_id", "dense_track_id"])

safe_to_csv(review_df, OUT_REVIEW)

# Conservative merge candidates: only accepted conservative identities.
accepted = arb_df[
    (arb_df["conservative_identity_accepted"] == True)
    & (arb_df["accepted_visual_colour"].isin(VALID_COLOURS))
].copy()

merge_rows = []
for (clip_id, colour), cg in accepted.groupby(["clip_id", "accepted_visual_colour"], sort=True):
    tids = sorted(cg["dense_track_id"].dropna().astype(int).tolist())
    if len(tids) <= 1:
        continue

    sid = clean(cg["scan_frame_id"].iloc[0])
    behaviour_id = COLOUR_TO_BEHAVIOUR_ID.get(colour, "")
    status_counts = cg["arbitration_status"].value_counts().to_dict()

    merge_rows.append({
        "clip_id": clip_id,
        "scan_frame_id": sid,
        "accepted_visual_colour": colour,
        "accepted_behaviour_pig_id": behaviour_id,
        "candidate_track_ids": " | ".join(map(str, tids)),
        "candidate_tracklet_count": len(tids),
        "arbitration_status_counts": " | ".join(f"{k}:{v}" for k, v in status_counts.items()),
        "merge_status": "conservative_same_identity_merge_candidate",
        "review_required": bool(cg["review_required"].any()),
        "reason": "Multiple conservatively accepted tracklets share the same visual colour/behaviour identity in the same clip.",
    })

merge_df = pd.DataFrame(merge_rows)
safe_to_csv(merge_df, OUT_MERGE)

# Clip summary.
clip_rows = []
for clip_id, cg in arb_df.groupby("clip_id", sort=True):
    sid = clean(cg["scan_frame_id"].iloc[0])
    accepted_cg = cg[cg["conservative_identity_accepted"] == True]
    confirmed = cg[cg["arbitration_status"] == "accepted_v29b_confirmed_by_v29c"]
    conflict = cg[cg["arbitration_status"] == "accepted_keep_v29b_conflict_with_v29c"]
    recovered = cg[cg["arbitration_status"] == "recovered_candidate_from_v29c_needs_review"]
    low = cg[cg["arbitration_status"] == "low_confidence_v29c_candidate_review"]
    unknown = cg[cg["arbitration_status"] == "unknown_no_reliable_identity"]

    accepted_colours = sorted(set(accepted_cg["accepted_visual_colour"].dropna().astype(str)) - {""})
    candidate_colours = sorted(set(cg["candidate_visual_colour"].dropna().astype(str)) - {""})

    old = v28d_clip[v28d_clip["clip_id"].astype(str).str.contains(clip_id.replace("__dense_polygon", ""), regex=False)]
    expected = ""
    raw_tracklets = int(cg["dense_track_id"].nunique())

    if len(old):
        expected = old.iloc[0].get("expected_target_pig_rows", "")

    clip_merge = merge_df[merge_df["clip_id"] == clip_id]

    if len(conflict) > 0:
        status = "accepted_with_conflicts_review_needed"
    elif len(recovered) > 0 or len(low) > 0:
        status = "accepted_plus_candidates_review_needed"
    else:
        status = "clean_conservative_identity_set"

    clip_rows.append({
        "clip_id": clip_id,
        "scan_frame_id": sid,
        "expected_target_pig_rows": expected,
        "raw_tracklets": raw_tracklets,
        "conservative_accepted_tracklets": int(len(accepted_cg)),
        "accepted_confirmed_by_v29c": int(len(confirmed)),
        "accepted_keep_v29b_conflict_with_v29c": int(len(conflict)),
        "recovered_candidates_from_v29c": int(len(recovered)),
        "low_confidence_v29c_candidates": int(len(low)),
        "unknown_no_reliable_identity": int(len(unknown)),
        "unique_conservative_colours": int(len(accepted_colours)),
        "conservative_colours": " | ".join(accepted_colours),
        "candidate_colours": " | ".join(candidate_colours),
        "conservative_merge_candidate_count": int(len(clip_merge)),
        "conservative_merge_candidate_colours": " | ".join(sorted(set(clip_merge["accepted_visual_colour"].astype(str)))) if len(clip_merge) else "",
        "clip_arbitration_status": status,
    })

clip_df = pd.DataFrame(clip_rows)
safe_to_csv(clip_df, OUT_CLIP_SUMMARY)

issues_df = pd.DataFrame(issues, columns=["issue_type", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)

# Decision summary.
total_tracklets = int(arb_df["dense_track_id"].nunique()) if len(arb_df) else 0
accepted_count = int((arb_df["conservative_identity_accepted"] == True).sum()) if len(arb_df) else 0
confirmed_count = int((arb_df["arbitration_status"] == "accepted_v29b_confirmed_by_v29c").sum()) if len(arb_df) else 0
conflict_count = int((arb_df["arbitration_status"] == "accepted_keep_v29b_conflict_with_v29c").sum()) if len(arb_df) else 0
recovered_count = int((arb_df["arbitration_status"] == "recovered_candidate_from_v29c_needs_review").sum()) if len(arb_df) else 0
low_count = int((arb_df["arbitration_status"] == "low_confidence_v29c_candidate_review").sum()) if len(arb_df) else 0
unknown_count = int((arb_df["arbitration_status"] == "unknown_no_reliable_identity").sum()) if len(arb_df) else 0
review_count = int((arb_df["review_required"] == True).sum()) if len(arb_df) else 0

decision = pd.DataFrame([{
    "v29d_decision": "conservative_identity_arbitration_completed",
    "total_tracklets": total_tracklets,
    "conservative_accepted_tracklets": accepted_count,
    "accepted_confirmed_by_v29c": confirmed_count,
    "accepted_keep_v29b_conflict_with_v29c": conflict_count,
    "recovered_candidates_from_v29c": recovered_count,
    "low_confidence_v29c_candidates": low_count,
    "unknown_no_reliable_identity": unknown_count,
    "review_queue_tracklets": review_count,
    "conservative_merge_candidates": int(len(merge_df)),
    "clips_evaluated": int(len(clip_df)),
    "issue_count": int(len(issues_df)),
    "primary_identity_source": "v29b_center_gt_overlap",
    "secondary_identity_source": "v29c_full_tracklet_hsv_colour_evidence",
    "v29c_overwrites_v29b": False,
    "ready_for_v29e_final_identity_linking_report": bool(len(arb_df) > 0 and len(issues_df) == 0),
    "recommended_next_step": "visual review of conflict/recovered candidates, then final identity-linking report and package",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

# Reports.
report = f"""# Week 7 v29d Conservative Identity Arbitration Report

## Purpose

This step combines v29b and v29c safely.

- v29b is treated as the primary identity source because it uses center-frame overlap with final GT colour/behaviour boxes.
- v29c is treated as secondary support evidence because it uses HSV full-tracklet colour evidence and produced conflicts.

## Main rule

v29c does **not** overwrite v29b.

## Summary

- Total tracklets: `{total_tracklets}`
- Conservative accepted tracklets: `{accepted_count}`
- Accepted and confirmed by v29c: `{confirmed_count}`
- Accepted from v29b but conflict with v29c: `{conflict_count}`
- Recovered candidates from v29c: `{recovered_count}`
- Low-confidence v29c candidates: `{low_count}`
- Unknown / no reliable identity: `{unknown_count}`
- Review queue tracklets: `{review_count}`
- Conservative merge candidates: `{len(merge_df)}`

## Interpretation

Simple IoU track IDs are not final identities. The conservative identity layer uses v29b as the anchor and v29c only as supporting evidence.

Accepted identities can be used as conservative tracklet-to-pig identity candidates. Recovered candidates and conflicts must be reviewed before becoming final.

## Next step

v29e should create the final identity-linking report and package, including:

1. accepted conservative identities,
2. review queue,
3. conservative merge candidates,
4. explicit limitations,
5. recommended next engineering steps.
"""

OUT_REPORT.write_text(report)

OUT_LIMITATIONS.write_text(
    "# Week 7 v29d Limitations\n\n"
    "1. This step produces conservative identity candidates, not final production tracking.\n"
    "2. v29b is trusted over v29c because v29c HSV evidence produced conflicts.\n"
    "3. v29c recovered candidates are useful but require visual review.\n"
    "4. Same-colour merge candidates are not automatically merged into final identities.\n"
    "5. Unknown identities are intentionally preserved instead of being forced into a colour class.\n"
)

OUT_README.write_text(
    "# Week 7 Conservative Identity Arbitration v29d\n\n"
    "## Purpose\n\n"
    "Safely combine v29b GT-overlap identity linking with v29c full-tracklet HSV colour evidence.\n\n"
    "## Outputs\n\n"
    "- `week7_conservative_identity_arbitration_v29d_tracklet_arbitration.csv`\n"
    "- `week7_conservative_identity_arbitration_v29d_clip_summary.csv`\n"
    "- `week7_conservative_identity_arbitration_v29d_conservative_merge_candidates.csv`\n"
    "- `week7_conservative_identity_arbitration_v29d_review_queue.csv`\n"
    "- `week7_conservative_identity_arbitration_v29d_decision_summary.csv`\n"
    "- `week7_conservative_identity_arbitration_v29d_report.md`\n"
)

OUT_NOTE.write_text(
    "# Week 7 Conservative Identity Arbitration v29d\n\n"
    "## Purpose\n\n"
    "This step conservatively arbitrates between v29b GT-overlap identity linking and v29c full-tracklet HSV colour evidence.\n\n"
    "## Summary\n\n"
    f"- Total tracklets: `{total_tracklets}`\n"
    f"- Conservative accepted tracklets: `{accepted_count}`\n"
    f"- Accepted confirmed by v29c: `{confirmed_count}`\n"
    f"- Accepted keep v29b despite v29c conflict: `{conflict_count}`\n"
    f"- Recovered candidates from v29c: `{recovered_count}`\n"
    f"- Low-confidence v29c candidates: `{low_count}`\n"
    f"- Unknown identities: `{unknown_count}`\n"
    f"- Review queue tracklets: `{review_count}`\n"
    f"- Conservative merge candidates: `{len(merge_df)}`\n"
    f"- Issue count: `{len(issues_df)}`\n"
    f"- Ready for v29e final identity-linking report: `{bool(decision.iloc[0]['ready_for_v29e_final_identity_linking_report'])}`\n\n"
    "## Outputs\n\n"
    f"- Decision summary: `{OUT_DECISION}`\n"
    f"- Tracklet arbitration: `{OUT_ARBITRATION}`\n"
    f"- Clip summary: `{OUT_CLIP_SUMMARY}`\n"
    f"- Merge candidates: `{OUT_MERGE}`\n"
    f"- Review queue: `{OUT_REVIEW}`\n"
    f"- Report: `{OUT_REPORT}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
)

print("Saved:")
print(OUT_ARBITRATION)
print(OUT_CLIP_SUMMARY)
print(OUT_MERGE)
print(OUT_REVIEW)
print(OUT_DECISION)
print(OUT_REPORT)
print(OUT_LIMITATIONS)
print(OUT_ISSUES)
print(OUT_README)
print(OUT_NOTE)

print()
print("=== v29d decision ===")
print(decision.to_string(index=False))

print()
print("=== v29d clip summary ===")
print(clip_df.to_string(index=False))

print()
print("=== v29d merge candidates ===")
if len(merge_df):
    print(merge_df.to_string(index=False))
else:
    print("No conservative merge candidates.")

print()
print("=== v29d issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
