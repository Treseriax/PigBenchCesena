from pathlib import Path
from datetime import datetime
import csv
import re

import pandas as pd


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

LOCKED_IDENTITY = W7 / "outputs" / "colour_identity" / "final_colour_identity_v17_fixed" / "week7_final_colour_identity_v17_fixed_locked_assignments.csv"

# Main expected W6 behaviour label file.
LABELS_LONG = W6 / "outputs" / "unified_ground_truth" / "week6_scanpoint_frame_labels_long.csv"

OUT_ROOT = W7 / "outputs" / "behaviour_label_fusion_v18"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_FUSED = OUT_ROOT / "week7_behaviour_label_fusion_v18_fused_pig_colour_behaviour.csv"
OUT_BOX_LEVEL = OUT_ROOT / "week7_behaviour_label_fusion_v18_box_level_dataset.csv"
OUT_UNMATCHED_BOXES = OUT_ROOT / "week7_behaviour_label_fusion_v18_unmatched_identity_boxes.csv"
OUT_UNMATCHED_LABELS = OUT_ROOT / "week7_behaviour_label_fusion_v18_unmatched_behaviour_labels.csv"
OUT_FRAME_SUMMARY = OUT_ROOT / "week7_behaviour_label_fusion_v18_frame_summary.csv"
OUT_BEHAVIOUR_DIST = OUT_ROOT / "week7_behaviour_label_fusion_v18_behaviour_distribution.csv"
OUT_COLOUR_STATUS_DIST = OUT_ROOT / "week7_behaviour_label_fusion_v18_colour_status_distribution.csv"
OUT_SUMMARY = OUT_ROOT / "week7_behaviour_label_fusion_v18_summary.csv"
OUT_NOTE = W7 / "notes" / "week7_behaviour_label_fusion_v18_notes.md"

VALID = ["blue", "green", "cyan", "red", "pink", "purple"]


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def norm_colour(v):
    if pd.isna(v):
        return ""

    s = str(v).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")

    if not s or s in ["nan", "none", "null"]:
        return ""

    mapping = {
        "blue": "blue",
        "blu": "blue",
        "b": "blue",

        "green": "green",
        "verde": "green",
        "g": "green",

        "cyan": "cyan",
        "ciano": "cyan",
        "azzurro": "cyan",
        "light_blue": "cyan",
        "lightblue": "cyan",
        "sky_blue": "cyan",

        "red": "red",
        "rosso": "red",
        "r": "red",

        "pink": "pink",
        "rosa": "pink",
        "p": "pink",

        "purple": "purple",
        "violet": "purple",
        "viola": "purple",
        "lilla": "purple",
        "lilac": "purple",
        "v": "purple",
    }

    return mapping.get(s, s)


def find_label_file():
    if LABELS_LONG.exists():
        return LABELS_LONG

    candidates = list((W6 / "outputs").rglob("*labels*long*.csv"))
    if candidates:
        return candidates[0]

    candidates = list((W6 / "outputs" / "unified_ground_truth").rglob("*.csv"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError("Could not find Week6 long behaviour label CSV.")


label_path = find_label_file()

identity = pd.read_csv(LOCKED_IDENTITY)
labels = pd.read_csv(label_path)

# Normalize keys.
identity["scan_frame_id"] = identity["scan_frame_id"].astype(str).str.strip()
identity["final_box_id"] = identity["final_box_id"].astype(str).str.strip()
labels["scan_frame_id"] = labels["scan_frame_id"].astype(str).str.strip()

# Normalize identity colour.
identity["colour_key"] = identity["final_colour_identity_v17"].apply(norm_colour)

# Detect behaviour label colour column.
colour_cols = [c for c in labels.columns if c.lower() in ["colour_id", "color_id", "colour", "color", "colour_raw", "color_raw"]]
if "colour_id" in labels.columns:
    label_colour_col = "colour_id"
elif "colour_raw" in labels.columns:
    label_colour_col = "colour_raw"
elif colour_cols:
    label_colour_col = colour_cols[0]
else:
    raise ValueError(f"No colour column found in behaviour labels. Columns: {list(labels.columns)}")

labels["colour_key"] = labels[label_colour_col].apply(norm_colour)

# Keep only valid colour-label rows for fusion.
labels_valid = labels[labels["colour_key"].isin(VALID)].copy()

# Deduplicate if needed.
label_dupes = (
    labels_valid
    .groupby(["scan_frame_id", "colour_key"])
    .size()
    .reset_index(name="n")
)
dup_label_keys = label_dupes[label_dupes["n"] > 1]

if len(dup_label_keys):
    # Keep first but record in summary. Better than crashing.
    labels_valid = labels_valid.drop_duplicates(["scan_frame_id", "colour_key"], keep="first").copy()

# Fuse known colour identity boxes to behaviour label.
known_identity = identity[identity["colour_key"].isin(VALID)].copy()
unknown_identity = identity[~identity["colour_key"].isin(VALID)].copy()

fused_known = known_identity.merge(
    labels_valid,
    on=["scan_frame_id", "colour_key"],
    how="left",
    suffixes=("_identity", "_label"),
    indicator=True,
)

fused_known["behaviour_match_status"] = fused_known["_merge"].map({
    "both": "matched_by_scan_frame_and_colour",
    "left_only": "no_behaviour_label_for_colour",
    "right_only": "unexpected_right_only",
}).astype(str)

fused_known.drop(columns=["_merge"], inplace=True)

# Unknown identity boxes remain in box-level dataset, but behaviour cannot be safely assigned.
unknown_rows = unknown_identity.copy()
unknown_rows["behaviour_match_status"] = unknown_rows["final_identity_status_v17"].map({
    "identity_unknown_not_visible": "identity_unknown_not_visible_no_safe_behaviour_match",
    "identity_unknown_uncertain": "identity_unknown_uncertain_no_safe_behaviour_match",
    "identity_unknown_unassigned": "identity_unknown_unassigned_no_safe_behaviour_match",
}).fillna("identity_unknown_no_safe_behaviour_match")

# Add missing label columns to unknown rows so concat is clean.
for c in labels_valid.columns:
    if c not in unknown_rows.columns:
        unknown_rows[c] = ""

# Add missing identity columns to fused_known if any.
for c in unknown_rows.columns:
    if c not in fused_known.columns:
        fused_known[c] = ""

# Main box-level dataset: one row per corrected box.
box_level = pd.concat(
    [
        fused_known[unknown_rows.columns],
        unknown_rows,
    ],
    ignore_index=True,
)

# For convenience, create clear final behaviour columns.
if "behaviour_code" not in box_level.columns:
    box_level["behaviour_code"] = ""
if "behaviour_label" not in box_level.columns:
    box_level["behaviour_label"] = ""

box_level["has_usable_colour_identity"] = box_level["final_colour_identity_v17"].apply(lambda x: norm_colour(x) in VALID)
box_level["has_matched_behaviour_label"] = box_level["behaviour_match_status"].astype(str).eq("matched_by_scan_frame_and_colour")
box_level["ready_for_behaviour_model_training"] = (
    box_level["has_usable_colour_identity"]
    & box_level["has_matched_behaviour_label"]
)

# Fused training-ready subset.
fused_ready = box_level[box_level["ready_for_behaviour_model_training"] == True].copy()

# Unmatched boxes.
unmatched_boxes = box_level[box_level["has_matched_behaviour_label"] == False].copy()

# Unmatched labels: behaviour labels whose colour has no usable identity box.
matched_label_keys = set(
    fused_ready["scan_frame_id"].astype(str) + "||" + fused_ready["colour_key"].astype(str)
)

labels_valid["label_key"] = labels_valid["scan_frame_id"].astype(str) + "||" + labels_valid["colour_key"].astype(str)
unmatched_labels = labels_valid[~labels_valid["label_key"].isin(matched_label_keys)].copy()

# Frame summary.
frame_rows = []

for sid, g in box_level.groupby("scan_frame_id", sort=True):
    label_g = labels_valid[labels_valid["scan_frame_id"] == sid]
    frame_rows.append({
        "scan_frame_id": sid,
        "box_count": int(len(g)),
        "usable_colour_identity_boxes": int(g["has_usable_colour_identity"].sum()),
        "matched_behaviour_boxes": int(g["has_matched_behaviour_label"].sum()),
        "unknown_identity_boxes": int((~g["has_usable_colour_identity"]).sum()),
        "behaviour_label_rows": int(len(label_g)),
        "unmatched_behaviour_labels": int(len(label_g) - g["has_matched_behaviour_label"].sum()),
        "ready_rows_for_training": int(g["ready_for_behaviour_model_training"].sum()),
    })

frame_summary = pd.DataFrame(frame_rows)

behaviour_dist = (
    fused_ready
    .groupby(["behaviour_code", "behaviour_label"], dropna=False)
    .size()
    .reset_index(name="fused_ready_count")
    .sort_values("fused_ready_count", ascending=False)
)

colour_status_dist = (
    box_level
    .groupby(["final_identity_status_v17", "final_colour_identity_v17"], dropna=False)
    .size()
    .reset_index(name="box_count")
    .sort_values("box_count", ascending=False)
)

summary = pd.DataFrame([{
    "identity_source_file": str(LOCKED_IDENTITY),
    "behaviour_label_source_file": str(label_path),
    "label_colour_column_used": label_colour_col,
    "total_corrected_boxes": int(len(identity)),
    "box_level_rows": int(len(box_level)),
    "usable_colour_identity_boxes": int(box_level["has_usable_colour_identity"].sum()),
    "unknown_identity_boxes": int((~box_level["has_usable_colour_identity"]).sum()),
    "matched_behaviour_boxes": int(box_level["has_matched_behaviour_label"].sum()),
    "ready_for_behaviour_model_training_rows": int(box_level["ready_for_behaviour_model_training"].sum()),
    "unmatched_identity_boxes": int(len(unmatched_boxes)),
    "unmatched_behaviour_labels": int(len(unmatched_labels)),
    "duplicate_label_keys_detected": int(len(dup_label_keys)),
    "frames_total": int(frame_summary["scan_frame_id"].nunique()),
    "frames_with_at_least_one_training_row": int((frame_summary["ready_rows_for_training"] > 0).sum()),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(fused_ready, OUT_FUSED)
safe_to_csv(box_level, OUT_BOX_LEVEL)
safe_to_csv(unmatched_boxes, OUT_UNMATCHED_BOXES)
safe_to_csv(unmatched_labels, OUT_UNMATCHED_LABELS)
safe_to_csv(frame_summary, OUT_FRAME_SUMMARY)
safe_to_csv(behaviour_dist, OUT_BEHAVIOUR_DIST)
safe_to_csv(colour_status_dist, OUT_COLOUR_STATUS_DIST)
safe_to_csv(summary, OUT_SUMMARY)

OUT_NOTE.write_text(
    "# Week 7 Behaviour Label Fusion v18\n\n"
    "## Purpose\n\n"
    "This step fuses final corrected pig boxes, final locked colour identity, and original behaviour labels into the main "
    "`Pig → Colour Marker → Behaviour Label` dataset.\n\n"
    "## Matching rule\n\n"
    "Known colour identities are matched to behaviour labels by `scan_frame_id + canonical colour`. "
    "`not_visible` and `uncertain` identities are retained in the box-level dataset but are not used as safe behaviour matches.\n\n"
    "## Summary\n\n"
    f"- Total corrected boxes: `{int(summary.iloc[0]['total_corrected_boxes'])}`\n"
    f"- Usable colour identity boxes: `{int(summary.iloc[0]['usable_colour_identity_boxes'])}`\n"
    f"- Unknown identity boxes: `{int(summary.iloc[0]['unknown_identity_boxes'])}`\n"
    f"- Matched behaviour boxes: `{int(summary.iloc[0]['matched_behaviour_boxes'])}`\n"
    f"- Ready training rows: `{int(summary.iloc[0]['ready_for_behaviour_model_training_rows'])}`\n"
    f"- Unmatched behaviour labels: `{int(summary.iloc[0]['unmatched_behaviour_labels'])}`\n"
    f"- Duplicate label keys detected: `{int(summary.iloc[0]['duplicate_label_keys_detected'])}`\n\n"
    "## Outputs\n\n"
    f"- Fused ready dataset: `{OUT_FUSED}`\n"
    f"- Box-level dataset: `{OUT_BOX_LEVEL}`\n"
    f"- Unmatched identity boxes: `{OUT_UNMATCHED_BOXES}`\n"
    f"- Unmatched behaviour labels: `{OUT_UNMATCHED_LABELS}`\n"
    f"- Frame summary: `{OUT_FRAME_SUMMARY}`\n"
    f"- Behaviour distribution: `{OUT_BEHAVIOUR_DIST}`\n"
    f"- Colour/status distribution: `{OUT_COLOUR_STATUS_DIST}`\n"
    f"- Summary: `{OUT_SUMMARY}`\n"
)

print("Saved:")
print(OUT_FUSED)
print(OUT_BOX_LEVEL)
print(OUT_UNMATCHED_BOXES)
print(OUT_UNMATCHED_LABELS)
print(OUT_FRAME_SUMMARY)
print(OUT_BEHAVIOUR_DIST)
print(OUT_COLOUR_STATUS_DIST)
print(OUT_SUMMARY)
print(OUT_NOTE)

print()
print("=== v18 behaviour fusion summary ===")
print(summary.to_string(index=False))

print()
print("=== behaviour distribution ===")
print(behaviour_dist.to_string(index=False))
