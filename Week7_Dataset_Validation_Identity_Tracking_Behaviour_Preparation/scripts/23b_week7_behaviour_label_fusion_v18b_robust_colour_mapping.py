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

OUT_ROOT = W7 / "outputs" / "behaviour_label_fusion_v18b"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_LABEL_AUDIT = OUT_ROOT / "week7_behaviour_label_fusion_v18b_label_colour_audit.csv"
OUT_FUSED = OUT_ROOT / "week7_behaviour_label_fusion_v18b_fused_pig_colour_behaviour.csv"
OUT_BOX_LEVEL = OUT_ROOT / "week7_behaviour_label_fusion_v18b_box_level_dataset.csv"
OUT_UNMATCHED_BOXES = OUT_ROOT / "week7_behaviour_label_fusion_v18b_unmatched_identity_boxes.csv"
OUT_UNMATCHED_LABELS = OUT_ROOT / "week7_behaviour_label_fusion_v18b_unmatched_behaviour_labels.csv"
OUT_FRAME_SUMMARY = OUT_ROOT / "week7_behaviour_label_fusion_v18b_frame_summary.csv"
OUT_BEHAVIOUR_DIST = OUT_ROOT / "week7_behaviour_label_fusion_v18b_behaviour_distribution.csv"
OUT_SUMMARY = OUT_ROOT / "week7_behaviour_label_fusion_v18b_summary.csv"
OUT_NOTE = W7 / "notes" / "week7_behaviour_label_fusion_v18b_notes.md"

VALID = ["blue", "green", "cyan", "red", "pink", "purple"]


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def text_norm(v):
    if pd.isna(v):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(v).strip().lower()).strip()


def robust_colour_from_values(*values):
    """
    Robustly infer canonical colour from several possible label fields.
    Priority is based on explicit colour words.
    """
    text = " ".join(text_norm(v) for v in values if not pd.isna(v))
    padded = f" {text} "

    # Important: check cyan/light-blue before blue.
    if any(k in padded for k in [" cyan ", " ciano ", " azzurro ", " celeste ", " light blue ", " lightblue ", " sky blue "]):
        return "cyan"

    if any(k in padded for k in [" blue ", " blu "]):
        return "blue"

    if any(k in padded for k in [" green ", " verde "]):
        return "green"

    if any(k in padded for k in [" red ", " rosso ", " rossa "]):
        return "red"

    if any(k in padded for k in [" pink ", " rosa "]):
        return "pink"

    if any(k in padded for k in [" purple ", " viola ", " violet ", " lilla ", " lilac "]):
        return "purple"

    # Fallback exact compact tokens.
    compact = text.replace(" ", "_")
    exact = {
        "b": "blue",
        "g": "green",
        "r": "red",
        "p": "pink",
        "v": "purple",
    }
    return exact.get(compact, "")


identity = pd.read_csv(LOCKED_IDENTITY)
labels = pd.read_csv(LABELS_LONG)

identity["scan_frame_id"] = identity["scan_frame_id"].astype(str).str.strip()
identity["final_box_id"] = identity["final_box_id"].astype(str).str.strip()
identity["colour_key"] = identity["final_colour_identity_v17"].apply(lambda x: robust_colour_from_values(x))

labels["scan_frame_id"] = labels["scan_frame_id"].astype(str).str.strip()

possible_colour_cols = [c for c in ["colour_id", "colour_raw", "pig_id", "color_id", "color_raw"] if c in labels.columns]

labels["colour_key"] = labels.apply(
    lambda r: robust_colour_from_values(*[r.get(c, "") for c in possible_colour_cols]),
    axis=1,
)

# Audit all source colour values.
audit_rows = []
for c in possible_colour_cols:
    vc = labels[c].fillna("").astype(str).value_counts().reset_index()
    vc.columns = ["raw_value", "count"]
    for _, r in vc.iterrows():
        audit_rows.append({
            "source_column": c,
            "raw_value": r["raw_value"],
            "count": int(r["count"]),
            "canonical_from_this_value": robust_colour_from_values(r["raw_value"]),
        })

audit = pd.DataFrame(audit_rows)
safe_to_csv(audit, OUT_LABEL_AUDIT)

labels_valid = labels[labels["colour_key"].isin(VALID)].copy()
labels_invalid = labels[~labels["colour_key"].isin(VALID)].copy()

# Detect duplicates.
dupes = (
    labels_valid
    .groupby(["scan_frame_id", "colour_key"])
    .size()
    .reset_index(name="n")
)
dup_keys = dupes[dupes["n"] > 1].copy()

if len(dup_keys):
    labels_valid = labels_valid.drop_duplicates(["scan_frame_id", "colour_key"], keep="first").copy()

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

unknown_rows = unknown_identity.copy()
unknown_rows["behaviour_match_status"] = unknown_rows["final_identity_status_v17"].map({
    "identity_unknown_not_visible": "identity_unknown_not_visible_no_safe_behaviour_match",
    "identity_unknown_uncertain": "identity_unknown_uncertain_no_safe_behaviour_match",
    "identity_unknown_unassigned": "identity_unknown_unassigned_no_safe_behaviour_match",
}).fillna("identity_unknown_no_safe_behaviour_match")

# Align columns.
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

box_level["has_usable_colour_identity"] = box_level["final_colour_identity_v17"].apply(
    lambda x: robust_colour_from_values(x) in VALID
)
box_level["has_matched_behaviour_label"] = box_level["behaviour_match_status"].astype(str).eq("matched_by_scan_frame_and_colour")
box_level["ready_for_behaviour_model_training"] = (
    box_level["has_usable_colour_identity"]
    & box_level["has_matched_behaviour_label"]
)

fused_ready = box_level[box_level["ready_for_behaviour_model_training"] == True].copy()
unmatched_boxes = box_level[box_level["has_matched_behaviour_label"] == False].copy()

matched_label_keys = set(
    fused_ready["scan_frame_id"].astype(str) + "||" + fused_ready["colour_key"].astype(str)
)

labels_valid["label_key"] = labels_valid["scan_frame_id"].astype(str) + "||" + labels_valid["colour_key"].astype(str)
unmatched_labels = labels_valid[~labels_valid["label_key"].isin(matched_label_keys)].copy()

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

summary = pd.DataFrame([{
    "identity_source_file": str(LOCKED_IDENTITY),
    "behaviour_label_source_file": str(LABELS_LONG),
    "colour_columns_used": " | ".join(possible_colour_cols),
    "total_corrected_boxes": int(len(identity)),
    "box_level_rows": int(len(box_level)),
    "usable_colour_identity_boxes": int(box_level["has_usable_colour_identity"].sum()),
    "unknown_identity_boxes": int((~box_level["has_usable_colour_identity"]).sum()),
    "total_behaviour_label_rows": int(len(labels)),
    "valid_colour_behaviour_label_rows": int(len(labels_valid)),
    "invalid_colour_behaviour_label_rows": int(len(labels_invalid)),
    "matched_behaviour_boxes": int(box_level["has_matched_behaviour_label"].sum()),
    "ready_for_behaviour_model_training_rows": int(box_level["ready_for_behaviour_model_training"].sum()),
    "unmatched_identity_boxes": int(len(unmatched_boxes)),
    "unmatched_behaviour_labels": int(len(unmatched_labels)),
    "duplicate_label_keys_detected": int(len(dup_keys)),
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
safe_to_csv(summary, OUT_SUMMARY)

OUT_NOTE.write_text(
    "# Week 7 Behaviour Label Fusion v18b\n\n"
    "## Purpose\n\n"
    "This corrected fusion step uses robust colour-name normalization across `colour_id`, `colour_raw`, and `pig_id` before matching behaviour labels to final colour identities.\n\n"
    "## Summary\n\n"
    f"- Total corrected boxes: `{int(summary.iloc[0]['total_corrected_boxes'])}`\n"
    f"- Usable colour identity boxes: `{int(summary.iloc[0]['usable_colour_identity_boxes'])}`\n"
    f"- Unknown identity boxes: `{int(summary.iloc[0]['unknown_identity_boxes'])}`\n"
    f"- Total behaviour label rows: `{int(summary.iloc[0]['total_behaviour_label_rows'])}`\n"
    f"- Valid-colour behaviour label rows: `{int(summary.iloc[0]['valid_colour_behaviour_label_rows'])}`\n"
    f"- Matched behaviour boxes: `{int(summary.iloc[0]['matched_behaviour_boxes'])}`\n"
    f"- Ready training rows: `{int(summary.iloc[0]['ready_for_behaviour_model_training_rows'])}`\n"
    f"- Unmatched behaviour labels: `{int(summary.iloc[0]['unmatched_behaviour_labels'])}`\n"
    f"- Duplicate label keys detected: `{int(summary.iloc[0]['duplicate_label_keys_detected'])}`\n\n"
    "## Outputs\n\n"
    f"- Label colour audit: `{OUT_LABEL_AUDIT}`\n"
    f"- Fused ready dataset: `{OUT_FUSED}`\n"
    f"- Box-level dataset: `{OUT_BOX_LEVEL}`\n"
    f"- Unmatched identity boxes: `{OUT_UNMATCHED_BOXES}`\n"
    f"- Unmatched behaviour labels: `{OUT_UNMATCHED_LABELS}`\n"
    f"- Frame summary: `{OUT_FRAME_SUMMARY}`\n"
    f"- Behaviour distribution: `{OUT_BEHAVIOUR_DIST}`\n"
    f"- Summary: `{OUT_SUMMARY}`\n"
)

print("Saved:")
print(OUT_LABEL_AUDIT)
print(OUT_FUSED)
print(OUT_BOX_LEVEL)
print(OUT_UNMATCHED_BOXES)
print(OUT_UNMATCHED_LABELS)
print(OUT_FRAME_SUMMARY)
print(OUT_BEHAVIOUR_DIST)
print(OUT_SUMMARY)
print(OUT_NOTE)

print()
print("=== v18b robust behaviour fusion summary ===")
print(summary.to_string(index=False))

print()
print("=== v18b behaviour distribution ===")
print(behaviour_dist.to_string(index=False))
