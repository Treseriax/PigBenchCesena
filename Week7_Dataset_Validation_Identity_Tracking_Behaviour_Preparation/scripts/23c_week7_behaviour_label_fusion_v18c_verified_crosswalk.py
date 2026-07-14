from pathlib import Path
from datetime import datetime
import csv
import re
import pandas as pd


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

LOCKED_IDENTITY = W7 / "outputs" / "colour_identity" / "final_colour_identity_v17_fixed" / "week7_final_colour_identity_v17_fixed_locked_assignments.csv"
LABELS_LONG = W6 / "outputs" / "unified_ground_truth" / "week6_scanpoint_frame_labels_long.csv"

OUT_ROOT = W7 / "outputs" / "behaviour_label_fusion_v18c_verified_crosswalk"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_CROSSWALK = OUT_ROOT / "week7_behaviour_label_fusion_v18c_visual_to_behaviour_pig_id_crosswalk.csv"
OUT_FUSED = OUT_ROOT / "week7_behaviour_label_fusion_v18c_fused_pig_colour_behaviour.csv"
OUT_BOX_LEVEL = OUT_ROOT / "week7_behaviour_label_fusion_v18c_box_level_dataset.csv"
OUT_UNMATCHED_BOXES = OUT_ROOT / "week7_behaviour_label_fusion_v18c_unmatched_identity_boxes.csv"
OUT_UNMATCHED_LABELS = OUT_ROOT / "week7_behaviour_label_fusion_v18c_unmatched_behaviour_labels.csv"
OUT_FRAME_SUMMARY = OUT_ROOT / "week7_behaviour_label_fusion_v18c_frame_summary.csv"
OUT_BEHAVIOUR_DIST = OUT_ROOT / "week7_behaviour_label_fusion_v18c_behaviour_distribution.csv"
OUT_STATUS_DIST = OUT_ROOT / "week7_behaviour_label_fusion_v18c_status_distribution.csv"
OUT_SUMMARY = OUT_ROOT / "week7_behaviour_label_fusion_v18c_summary.csv"
OUT_NOTE = W7 / "notes" / "week7_behaviour_label_fusion_v18c_verified_crosswalk_notes.md"

VALID_VISUAL = ["blue", "green", "cyan", "red", "pink", "purple"]
VALID_BEHAVIOUR_IDS = ["blue", "green", "purple", "red_neck", "red_tail", "no_color"]

# Verified by user:
# pink -> red_tail
# red  -> red_neck / red_head
# cyan -> no_color
CROSSWALK = {
    "blue": "blue",
    "green": "green",
    "purple": "purple",
    "red": "red_neck",
    "pink": "red_tail",
    "cyan": "no_color",
}


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def norm_text(v):
    if pd.isna(v):
        return ""
    return re.sub(r"[^a-z0-9]+", "_", str(v).strip().lower()).strip("_")


def norm_visual_colour(v):
    s = norm_text(v)

    mapping = {
        "blu": "blue",
        "blue": "blue",
        "verde": "green",
        "green": "green",
        "cyan": "cyan",
        "ciano": "cyan",
        "azzurro": "cyan",
        "light_blue": "cyan",
        "rosso": "red",
        "red": "red",
        "rosa": "pink",
        "pink": "pink",
        "viola": "purple",
        "violet": "purple",
        "purple": "purple",
    }

    return mapping.get(s, s if s in VALID_VISUAL else "")


def behaviour_pig_id_from_label_row(row):
    """
    Preserve annotation-specific pig IDs:
    blue, green, purple, red_neck, red_tail, no_color.
    Do NOT collapse red_neck/red_tail into red.
    """
    vals = []

    for c in ["colour_id", "pig_id", "colour_raw", "color_id", "color_raw"]:
        if c in row.index:
            vals.append(str(row.get(c, "")))

    text = " ".join(vals).lower()

    if "red_neck" in text or "red neck" in text or "rosso testa" in text or "red_head" in text or "red head" in text:
        return "red_neck"

    if "red_tail" in text or "red tail" in text or "rosso coda" in text:
        return "red_tail"

    if "no_color" in text or "no color" in text or "# - no color" in text:
        return "no_color"

    if re.search(r"\bblue\b|\bblu\b", text):
        return "blue"

    if re.search(r"\bgreen\b|\bverde\b", text):
        return "green"

    if re.search(r"\bpurple\b|\bviola\b|\bviolet\b", text):
        return "purple"

    return ""


identity = pd.read_csv(LOCKED_IDENTITY)
labels = pd.read_csv(LABELS_LONG)

identity["scan_frame_id"] = identity["scan_frame_id"].astype(str).str.strip()
identity["final_box_id"] = identity["final_box_id"].astype(str).str.strip()

labels["scan_frame_id"] = labels["scan_frame_id"].astype(str).str.strip()

# Keep visual colour and behaviour pig ID as separate fields.
identity["visual_marker_colour_v18c"] = identity["final_colour_identity_v17"].apply(norm_visual_colour)
identity["behaviour_pig_id_v18c"] = identity["visual_marker_colour_v18c"].map(CROSSWALK).fillna("")

labels["behaviour_pig_id_v18c"] = labels.apply(behaviour_pig_id_from_label_row, axis=1)

crosswalk_df = pd.DataFrame([
    {
        "visual_marker_colour": k,
        "behaviour_pig_id": v,
        "manual_crosswalk_status": "verified_by_user",
        "notes": (
            "blue/green/purple direct; red maps to red_neck/red_head; "
            "pink maps to red_tail; cyan maps to no_color"
        )
    }
    for k, v in CROSSWALK.items()
])
safe_to_csv(crosswalk_df, OUT_CROSSWALK)

labels_valid = labels[labels["behaviour_pig_id_v18c"].isin(VALID_BEHAVIOUR_IDS)].copy()
labels_invalid = labels[~labels["behaviour_pig_id_v18c"].isin(VALID_BEHAVIOUR_IDS)].copy()

# There should be exactly one label per frame per behaviour pig ID.
dup_keys = (
    labels_valid
    .groupby(["scan_frame_id", "behaviour_pig_id_v18c"])
    .size()
    .reset_index(name="n")
)
dup_keys = dup_keys[dup_keys["n"] > 1].copy()

if len(dup_keys):
    labels_valid = labels_valid.drop_duplicates(["scan_frame_id", "behaviour_pig_id_v18c"], keep="first").copy()

known_identity = identity[identity["behaviour_pig_id_v18c"].isin(VALID_BEHAVIOUR_IDS)].copy()
unknown_identity = identity[~identity["behaviour_pig_id_v18c"].isin(VALID_BEHAVIOUR_IDS)].copy()

fused_known = known_identity.merge(
    labels_valid,
    on=["scan_frame_id", "behaviour_pig_id_v18c"],
    how="left",
    suffixes=("_identity", "_label"),
    indicator=True,
)

fused_known["behaviour_match_status_v18c"] = fused_known["_merge"].map({
    "both": "matched_by_scan_frame_and_behaviour_pig_id_crosswalk",
    "left_only": "no_behaviour_label_for_crosswalk_id",
    "right_only": "unexpected_right_only",
}).astype(str)

fused_known.drop(columns=["_merge"], inplace=True)

unknown_rows = unknown_identity.copy()
unknown_rows["behaviour_match_status_v18c"] = unknown_rows["final_identity_status_v17"].map({
    "identity_unknown_not_visible": "identity_unknown_not_visible_no_safe_behaviour_match",
    "identity_unknown_uncertain": "identity_unknown_uncertain_no_safe_behaviour_match",
    "identity_unknown_unassigned": "identity_unknown_unassigned_no_safe_behaviour_match",
}).fillna("identity_unknown_no_safe_behaviour_match")

# Align columns for concat.
for c in labels_valid.columns:
    if c not in unknown_rows.columns:
        unknown_rows[c] = ""

for c in unknown_rows.columns:
    if c not in fused_known.columns:
        fused_known[c] = ""

box_level = pd.concat(
    [fused_known[unknown_rows.columns], unknown_rows],
    ignore_index=True,
)

for c in ["behaviour_code", "behaviour_label"]:
    if c not in box_level.columns:
        box_level[c] = ""

box_level["has_usable_visual_colour_identity"] = box_level["visual_marker_colour_v18c"].isin(VALID_VISUAL)
box_level["has_behaviour_pig_id_crosswalk"] = box_level["behaviour_pig_id_v18c"].isin(VALID_BEHAVIOUR_IDS)
box_level["has_matched_behaviour_label"] = box_level["behaviour_match_status_v18c"].astype(str).eq(
    "matched_by_scan_frame_and_behaviour_pig_id_crosswalk"
)
box_level["ready_for_behaviour_model_training"] = (
    box_level["has_usable_visual_colour_identity"]
    & box_level["has_behaviour_pig_id_crosswalk"]
    & box_level["has_matched_behaviour_label"]
)

fused_ready = box_level[box_level["ready_for_behaviour_model_training"] == True].copy()
unmatched_boxes = box_level[box_level["has_matched_behaviour_label"] == False].copy()

matched_label_keys = set(
    fused_ready["scan_frame_id"].astype(str) + "||" + fused_ready["behaviour_pig_id_v18c"].astype(str)
)

labels_valid["label_key_v18c"] = labels_valid["scan_frame_id"].astype(str) + "||" + labels_valid["behaviour_pig_id_v18c"].astype(str)
unmatched_labels = labels_valid[~labels_valid["label_key_v18c"].isin(matched_label_keys)].copy()

frame_rows = []

for sid, g in box_level.groupby("scan_frame_id", sort=True):
    label_g = labels_valid[labels_valid["scan_frame_id"] == sid]

    frame_rows.append({
        "scan_frame_id": sid,
        "box_count": int(len(g)),
        "usable_visual_colour_identity_boxes": int(g["has_usable_visual_colour_identity"].sum()),
        "crosswalked_behaviour_pig_id_boxes": int(g["has_behaviour_pig_id_crosswalk"].sum()),
        "matched_behaviour_boxes": int(g["has_matched_behaviour_label"].sum()),
        "unknown_identity_boxes": int((~g["has_usable_visual_colour_identity"]).sum()),
        "behaviour_label_rows": int(len(label_g)),
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

status_dist = (
    box_level
    .groupby(["final_identity_status_v17", "visual_marker_colour_v18c", "behaviour_pig_id_v18c"], dropna=False)
    .size()
    .reset_index(name="box_count")
    .sort_values("box_count", ascending=False)
)

summary = pd.DataFrame([{
    "identity_source_file": str(LOCKED_IDENTITY),
    "behaviour_label_source_file": str(LABELS_LONG),
    "total_corrected_boxes": int(len(identity)),
    "box_level_rows": int(len(box_level)),
    "usable_visual_colour_identity_boxes": int(box_level["has_usable_visual_colour_identity"].sum()),
    "unknown_identity_boxes": int((~box_level["has_usable_visual_colour_identity"]).sum()),
    "total_behaviour_label_rows": int(len(labels)),
    "valid_behaviour_pig_id_label_rows": int(len(labels_valid)),
    "invalid_behaviour_pig_id_label_rows": int(len(labels_invalid)),
    "matched_behaviour_boxes": int(box_level["has_matched_behaviour_label"].sum()),
    "ready_for_behaviour_model_training_rows": int(box_level["ready_for_behaviour_model_training"].sum()),
    "unmatched_identity_boxes": int(len(unmatched_boxes)),
    "unmatched_behaviour_labels": int(len(unmatched_labels)),
    "duplicate_label_keys_detected": int(len(dup_keys)),
    "frames_total": int(frame_summary["scan_frame_id"].nunique()),
    "frames_with_at_least_one_training_row": int((frame_summary["ready_rows_for_training"] > 0).sum()),
    "crosswalk_policy": "blue->blue; green->green; purple->purple; red->red_neck; pink->red_tail; cyan->no_color",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(fused_ready, OUT_FUSED)
safe_to_csv(box_level, OUT_BOX_LEVEL)
safe_to_csv(unmatched_boxes, OUT_UNMATCHED_BOXES)
safe_to_csv(unmatched_labels, OUT_UNMATCHED_LABELS)
safe_to_csv(frame_summary, OUT_FRAME_SUMMARY)
safe_to_csv(behaviour_dist, OUT_BEHAVIOUR_DIST)
safe_to_csv(status_dist, OUT_STATUS_DIST)
safe_to_csv(summary, OUT_SUMMARY)

OUT_NOTE.write_text(
    "# Week 7 Behaviour Label Fusion v18c Verified Crosswalk\n\n"
    "## Purpose\n\n"
    "This step fuses final visual marker colours with behaviour labels using a manually verified crosswalk from visual marker colour to annotation-specific behaviour pig ID.\n\n"
    "## Crosswalk\n\n"
    "- blue → blue\n"
    "- green → green\n"
    "- purple → purple\n"
    "- red → red_neck / red_head\n"
    "- pink → red_tail\n"
    "- cyan → no_color\n\n"
    "## Important\n\n"
    "The visual marker colour and the behaviour annotation pig ID are stored separately. "
    "This preserves the visual evidence layer while allowing correct behaviour-label fusion.\n\n"
    "## Summary\n\n"
    f"- Total corrected boxes: `{int(summary.iloc[0]['total_corrected_boxes'])}`\n"
    f"- Usable visual colour identity boxes: `{int(summary.iloc[0]['usable_visual_colour_identity_boxes'])}`\n"
    f"- Unknown identity boxes: `{int(summary.iloc[0]['unknown_identity_boxes'])}`\n"
    f"- Total behaviour label rows: `{int(summary.iloc[0]['total_behaviour_label_rows'])}`\n"
    f"- Valid behaviour pig ID label rows: `{int(summary.iloc[0]['valid_behaviour_pig_id_label_rows'])}`\n"
    f"- Matched behaviour boxes: `{int(summary.iloc[0]['matched_behaviour_boxes'])}`\n"
    f"- Ready training rows: `{int(summary.iloc[0]['ready_for_behaviour_model_training_rows'])}`\n"
    f"- Unmatched behaviour labels: `{int(summary.iloc[0]['unmatched_behaviour_labels'])}`\n"
    f"- Duplicate label keys detected: `{int(summary.iloc[0]['duplicate_label_keys_detected'])}`\n\n"
    "## Outputs\n\n"
    f"- Crosswalk: `{OUT_CROSSWALK}`\n"
    f"- Fused ready dataset: `{OUT_FUSED}`\n"
    f"- Box-level dataset: `{OUT_BOX_LEVEL}`\n"
    f"- Unmatched identity boxes: `{OUT_UNMATCHED_BOXES}`\n"
    f"- Unmatched behaviour labels: `{OUT_UNMATCHED_LABELS}`\n"
    f"- Frame summary: `{OUT_FRAME_SUMMARY}`\n"
    f"- Behaviour distribution: `{OUT_BEHAVIOUR_DIST}`\n"
    f"- Status distribution: `{OUT_STATUS_DIST}`\n"
    f"- Summary: `{OUT_SUMMARY}`\n"
)

print("Saved:")
print(OUT_CROSSWALK)
print(OUT_FUSED)
print(OUT_BOX_LEVEL)
print(OUT_UNMATCHED_BOXES)
print(OUT_UNMATCHED_LABELS)
print(OUT_FRAME_SUMMARY)
print(OUT_BEHAVIOUR_DIST)
print(OUT_STATUS_DIST)
print(OUT_SUMMARY)
print(OUT_NOTE)

print()
print("=== v18c verified crosswalk behaviour fusion summary ===")
print(summary.to_string(index=False))

print()
print("=== v18c behaviour distribution ===")
print(behaviour_dist.to_string(index=False))
