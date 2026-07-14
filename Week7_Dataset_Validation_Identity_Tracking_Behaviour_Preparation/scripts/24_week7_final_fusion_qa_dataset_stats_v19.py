from pathlib import Path
from datetime import datetime
import csv
import pandas as pd


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V18C_ROOT = W7 / "outputs" / "behaviour_label_fusion_v18c_verified_crosswalk"

FUSED_READY = V18C_ROOT / "week7_behaviour_label_fusion_v18c_fused_pig_colour_behaviour.csv"
BOX_LEVEL = V18C_ROOT / "week7_behaviour_label_fusion_v18c_box_level_dataset.csv"
V18C_SUMMARY = V18C_ROOT / "week7_behaviour_label_fusion_v18c_summary.csv"
V18C_FRAME_SUMMARY = V18C_ROOT / "week7_behaviour_label_fusion_v18c_frame_summary.csv"
V18C_STATUS_DIST = V18C_ROOT / "week7_behaviour_label_fusion_v18c_status_distribution.csv"

OUT_ROOT = W7 / "outputs" / "final_fusion_qa_dataset_stats_v19"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_TRAINING_LOCK = OUT_ROOT / "week7_final_training_ready_dataset_v19_locked.csv"
OUT_SUMMARY = OUT_ROOT / "week7_final_fusion_qa_v19_summary.csv"
OUT_BEHAVIOUR_DIST = OUT_ROOT / "week7_final_fusion_qa_v19_behaviour_distribution.csv"
OUT_BEHAVIOUR_IMBALANCE = OUT_ROOT / "week7_final_fusion_qa_v19_behaviour_imbalance_report.csv"
OUT_VISUAL_COLOUR_DIST = OUT_ROOT / "week7_final_fusion_qa_v19_visual_colour_distribution.csv"
OUT_BEHAVIOUR_PIG_ID_DIST = OUT_ROOT / "week7_final_fusion_qa_v19_behaviour_pig_id_distribution.csv"
OUT_VIDEO_DIST = OUT_ROOT / "week7_final_fusion_qa_v19_video_distribution.csv"
OUT_FRAME_COVERAGE = OUT_ROOT / "week7_final_fusion_qa_v19_frame_coverage.csv"
OUT_EXCLUDED_ROWS = OUT_ROOT / "week7_final_fusion_qa_v19_excluded_rows.csv"
OUT_HARD_ISSUES = OUT_ROOT / "week7_final_fusion_qa_v19_hard_issues.csv"
OUT_SOFT_ISSUES = OUT_ROOT / "week7_final_fusion_qa_v19_soft_issues.csv"
OUT_NOTE = W7 / "notes" / "week7_final_fusion_qa_dataset_stats_v19_notes.md"

VALID_VISUAL = ["blue", "green", "cyan", "red", "pink", "purple"]
VALID_BEHAVIOUR_IDS = ["blue", "green", "purple", "red_neck", "red_tail", "no_color"]


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def col_or_empty(df, col):
    if col in df.columns:
        return df[col].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index)


def bool_col(df, col):
    if col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    s = df[col]
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


ready = pd.read_csv(FUSED_READY)
box = pd.read_csv(BOX_LEVEL)
v18c_summary = pd.read_csv(V18C_SUMMARY)
frame_summary = pd.read_csv(V18C_FRAME_SUMMARY)

# Normalize essential columns.
for df in [ready, box]:
    for c in ["scan_frame_id", "final_box_id", "visual_marker_colour_v18c", "behaviour_pig_id_v18c", "behaviour_code", "behaviour_label"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()

# Try to find useful video/timestamp columns.
video_col = None
for c in ["video_id", "video_id_label", "video_id_identity"]:
    if c in ready.columns:
        video_col = c
        break

timestamp_col = None
for c in ["timestamp", "timestamp_label", "timestamp_identity"]:
    if c in ready.columns:
        timestamp_col = c
        break

# Locked clean training dataset.
ready["v19_dataset_role"] = "training_ready_behaviour_fusion"
ready["v19_locked_at"] = datetime.now().isoformat(timespec="seconds")
safe_to_csv(ready, OUT_TRAINING_LOCK)

# Hard QA checks.
hard_rows = []

# 1. Duplicate final boxes in training-ready.
dup_box = (
    ready.groupby(["scan_frame_id", "final_box_id"])
    .size()
    .reset_index(name="n")
)
dup_box = dup_box[dup_box["n"] > 1]
for _, r in dup_box.iterrows():
    hard_rows.append({
        "issue_type": "duplicate_training_ready_box",
        "scan_frame_id": r["scan_frame_id"],
        "final_box_id": r["final_box_id"],
        "issue_detail": f"{int(r['n'])} rows for same scan_frame_id/final_box_id",
    })

# 2. Missing behaviour code / label.
missing_beh = ready[
    col_or_empty(ready, "behaviour_code").eq("") |
    col_or_empty(ready, "behaviour_label").eq("")
]
for _, r in missing_beh.iterrows():
    hard_rows.append({
        "issue_type": "missing_behaviour_label",
        "scan_frame_id": r.get("scan_frame_id", ""),
        "final_box_id": r.get("final_box_id", ""),
        "issue_detail": "Missing behaviour_code or behaviour_label in training-ready row",
    })

# 3. Invalid visual colours.
invalid_visual = ready[~ready["visual_marker_colour_v18c"].isin(VALID_VISUAL)]
for _, r in invalid_visual.iterrows():
    hard_rows.append({
        "issue_type": "invalid_visual_marker_colour",
        "scan_frame_id": r.get("scan_frame_id", ""),
        "final_box_id": r.get("final_box_id", ""),
        "issue_detail": str(r.get("visual_marker_colour_v18c", "")),
    })

# 4. Invalid behaviour pig IDs.
invalid_pigid = ready[~ready["behaviour_pig_id_v18c"].isin(VALID_BEHAVIOUR_IDS)]
for _, r in invalid_pigid.iterrows():
    hard_rows.append({
        "issue_type": "invalid_behaviour_pig_id",
        "scan_frame_id": r.get("scan_frame_id", ""),
        "final_box_id": r.get("final_box_id", ""),
        "issue_detail": str(r.get("behaviour_pig_id_v18c", "")),
    })

# 5. Duplicate behaviour pig id within same frame in ready set.
dup_pig_frame = (
    ready.groupby(["scan_frame_id", "behaviour_pig_id_v18c"])
    .size()
    .reset_index(name="n")
)
dup_pig_frame = dup_pig_frame[dup_pig_frame["n"] > 1]
for _, r in dup_pig_frame.iterrows():
    hard_rows.append({
        "issue_type": "duplicate_behaviour_pig_id_within_frame",
        "scan_frame_id": r["scan_frame_id"],
        "final_box_id": "",
        "issue_detail": f"{r['behaviour_pig_id_v18c']} appears {int(r['n'])} times in training-ready rows",
    })

hard_issues = pd.DataFrame(hard_rows, columns=["issue_type", "scan_frame_id", "final_box_id", "issue_detail"])

# Soft issues: excluded rows from box-level.
ready_flag = bool_col(box, "ready_for_behaviour_model_training")
excluded = box[~ready_flag].copy()

excluded["v19_exclusion_reason"] = "unknown"
if "final_identity_status_v17" in excluded.columns:
    excluded.loc[excluded["final_identity_status_v17"].astype(str).eq("identity_unknown_not_visible"), "v19_exclusion_reason"] = "identity_unknown_not_visible"
    excluded.loc[excluded["final_identity_status_v17"].astype(str).eq("identity_unknown_uncertain"), "v19_exclusion_reason"] = "identity_unknown_uncertain"
    excluded.loc[excluded["final_identity_status_v17"].astype(str).eq("identity_unknown_unassigned"), "v19_exclusion_reason"] = "identity_unknown_unassigned"

if "behaviour_match_status_v18c" in excluded.columns:
    no_match_mask = excluded["behaviour_match_status_v18c"].astype(str).str.contains("no_behaviour_label", na=False)
    excluded.loc[no_match_mask, "v19_exclusion_reason"] = "no_behaviour_label_match"

safe_to_csv(excluded, OUT_EXCLUDED_ROWS)

soft_issues = (
    excluded.groupby("v19_exclusion_reason")
    .size()
    .reset_index(name="excluded_box_count")
    .sort_values("excluded_box_count", ascending=False)
)
safe_to_csv(soft_issues, OUT_SOFT_ISSUES)

# Distribution reports.
behaviour_dist = (
    ready.groupby(["behaviour_code", "behaviour_label"], dropna=False)
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
behaviour_dist["percentage"] = (behaviour_dist["count"] / len(ready) * 100).round(2)
safe_to_csv(behaviour_dist, OUT_BEHAVIOUR_DIST)

# Imbalance report.
max_count = int(behaviour_dist["count"].max()) if len(behaviour_dist) else 0
min_count = int(behaviour_dist["count"].min()) if len(behaviour_dist) else 0
imbalance_rows = []
for _, r in behaviour_dist.iterrows():
    count = int(r["count"])
    imbalance_rows.append({
        "behaviour_code": r["behaviour_code"],
        "behaviour_label": r["behaviour_label"],
        "count": count,
        "percentage": r["percentage"],
        "relative_to_majority": round(count / max_count, 4) if max_count else 0,
        "scarcity_flag": (
            "very_low" if count < 5 else
            "low" if count < 15 else
            "medium" if count < 30 else
            "ok"
        ),
    })
behaviour_imbalance = pd.DataFrame(imbalance_rows)
safe_to_csv(behaviour_imbalance, OUT_BEHAVIOUR_IMBALANCE)

visual_colour_dist = (
    ready.groupby("visual_marker_colour_v18c", dropna=False)
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
visual_colour_dist["percentage"] = (visual_colour_dist["count"] / len(ready) * 100).round(2)
safe_to_csv(visual_colour_dist, OUT_VISUAL_COLOUR_DIST)

behaviour_pig_id_dist = (
    ready.groupby("behaviour_pig_id_v18c", dropna=False)
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)
behaviour_pig_id_dist["percentage"] = (behaviour_pig_id_dist["count"] / len(ready) * 100).round(2)
safe_to_csv(behaviour_pig_id_dist, OUT_BEHAVIOUR_PIG_ID_DIST)

# Video distribution if available.
if video_col:
    video_dist = (
        ready.groupby(video_col, dropna=False)
        .size()
        .reset_index(name="training_ready_rows")
        .sort_values("training_ready_rows", ascending=False)
    )
else:
    video_dist = pd.DataFrame([{
        "warning": "No video_id-like column found in fused ready dataset",
        "training_ready_rows": len(ready),
    }])
safe_to_csv(video_dist, OUT_VIDEO_DIST)

# Frame coverage from v18c frame summary.
frame_cov = frame_summary.copy()
for c in ["ready_rows_for_training", "box_count", "unknown_identity_boxes", "behaviour_label_rows"]:
    if c in frame_cov.columns:
        frame_cov[c] = pd.to_numeric(frame_cov[c], errors="coerce").fillna(0).astype(int)

if "ready_rows_for_training" in frame_cov.columns:
    frame_cov["frame_training_coverage_status"] = frame_cov["ready_rows_for_training"].apply(
        lambda n: "full_6" if n >= 6 else "partial" if n > 0 else "none"
    )
safe_to_csv(frame_cov, OUT_FRAME_COVERAGE)

# Summary.
frames_total = int(frame_cov["scan_frame_id"].nunique()) if "scan_frame_id" in frame_cov.columns else 0
frames_with_training = int((frame_cov["ready_rows_for_training"] > 0).sum()) if "ready_rows_for_training" in frame_cov.columns else 0
frames_full_6 = int((frame_cov["ready_rows_for_training"] >= 6).sum()) if "ready_rows_for_training" in frame_cov.columns else 0
frames_partial = int(((frame_cov["ready_rows_for_training"] > 0) & (frame_cov["ready_rows_for_training"] < 6)).sum()) if "ready_rows_for_training" in frame_cov.columns else 0
frames_none = int((frame_cov["ready_rows_for_training"] == 0).sum()) if "ready_rows_for_training" in frame_cov.columns else 0

summary = pd.DataFrame([{
    "v18c_source_ready_dataset": str(FUSED_READY),
    "v18c_source_box_level_dataset": str(BOX_LEVEL),
    "training_ready_rows": int(len(ready)),
    "box_level_rows": int(len(box)),
    "excluded_rows": int(len(excluded)),
    "hard_issue_count": int(len(hard_issues)),
    "soft_issue_category_count": int(len(soft_issues)),
    "behaviour_class_count": int(behaviour_dist["behaviour_code"].nunique()) if len(behaviour_dist) else 0,
    "majority_behaviour_count": max_count,
    "minority_behaviour_count": min_count,
    "imbalance_ratio_majority_to_minority": round(max_count / min_count, 4) if min_count else "",
    "frames_total": frames_total,
    "frames_with_training_rows": frames_with_training,
    "frames_full_6_training_rows": frames_full_6,
    "frames_partial_training_rows": frames_partial,
    "frames_without_training_rows": frames_none,
    "ready_for_split_design": bool(len(hard_issues) == 0 and len(ready) > 0),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
safe_to_csv(summary, OUT_SUMMARY)
safe_to_csv(hard_issues, OUT_HARD_ISSUES)

ready_for_split = bool(summary.iloc[0]["ready_for_split_design"])

OUT_NOTE.write_text(
    "# Week 7 Final Fusion QA + Dataset Statistics v19\n\n"
    "## Purpose\n\n"
    "This step audits the v18c fused dataset and locks a clean training-ready table for later split design, tracking, and clip-level representation.\n\n"
    "## Inputs\n\n"
    f"- v18c fused ready dataset: `{FUSED_READY}`\n"
    f"- v18c box-level dataset: `{BOX_LEVEL}`\n"
    f"- v18c summary: `{V18C_SUMMARY}`\n\n"
    "## QA policy\n\n"
    "Hard issues include duplicate training-ready boxes, missing behaviour labels, invalid visual colours, invalid behaviour pig IDs, "
    "or duplicate behaviour pig IDs within the same frame. Soft issues are expected exclusions such as `not_visible` or `uncertain` identity.\n\n"
    "## Summary\n\n"
    f"- Training-ready rows: `{int(summary.iloc[0]['training_ready_rows'])}`\n"
    f"- Box-level rows: `{int(summary.iloc[0]['box_level_rows'])}`\n"
    f"- Excluded rows: `{int(summary.iloc[0]['excluded_rows'])}`\n"
    f"- Hard issue count: `{int(summary.iloc[0]['hard_issue_count'])}`\n"
    f"- Behaviour class count: `{int(summary.iloc[0]['behaviour_class_count'])}`\n"
    f"- Majority behaviour count: `{int(summary.iloc[0]['majority_behaviour_count'])}`\n"
    f"- Minority behaviour count: `{int(summary.iloc[0]['minority_behaviour_count'])}`\n"
    f"- Frames with training rows: `{int(summary.iloc[0]['frames_with_training_rows'])}` / `{int(summary.iloc[0]['frames_total'])}`\n"
    f"- Ready for split design: `{ready_for_split}`\n\n"
    "## Outputs\n\n"
    f"- Locked training-ready dataset: `{OUT_TRAINING_LOCK}`\n"
    f"- Summary: `{OUT_SUMMARY}`\n"
    f"- Behaviour distribution: `{OUT_BEHAVIOUR_DIST}`\n"
    f"- Behaviour imbalance report: `{OUT_BEHAVIOUR_IMBALANCE}`\n"
    f"- Visual colour distribution: `{OUT_VISUAL_COLOUR_DIST}`\n"
    f"- Behaviour pig ID distribution: `{OUT_BEHAVIOUR_PIG_ID_DIST}`\n"
    f"- Video distribution: `{OUT_VIDEO_DIST}`\n"
    f"- Frame coverage: `{OUT_FRAME_COVERAGE}`\n"
    f"- Excluded rows: `{OUT_EXCLUDED_ROWS}`\n"
    f"- Hard issues: `{OUT_HARD_ISSUES}`\n"
    f"- Soft issues: `{OUT_SOFT_ISSUES}`\n"
)

print("Saved:")
print(OUT_TRAINING_LOCK)
print(OUT_SUMMARY)
print(OUT_BEHAVIOUR_DIST)
print(OUT_BEHAVIOUR_IMBALANCE)
print(OUT_VISUAL_COLOUR_DIST)
print(OUT_BEHAVIOUR_PIG_ID_DIST)
print(OUT_VIDEO_DIST)
print(OUT_FRAME_COVERAGE)
print(OUT_EXCLUDED_ROWS)
print(OUT_HARD_ISSUES)
print(OUT_SOFT_ISSUES)
print(OUT_NOTE)

print()
print("=== v19 final fusion QA summary ===")
print(summary.to_string(index=False))

print()
print("=== v19 behaviour distribution ===")
print(behaviour_dist.to_string(index=False))

print()
print("=== v19 visual colour distribution ===")
print(visual_colour_dist.to_string(index=False))

print()
print("=== v19 hard issues ===")
if len(hard_issues):
    print(hard_issues.to_string(index=False))
else:
    print("No hard issues found.")
