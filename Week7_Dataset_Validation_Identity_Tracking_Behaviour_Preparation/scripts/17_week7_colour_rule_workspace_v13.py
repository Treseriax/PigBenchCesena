from pathlib import Path
from datetime import datetime
import csv
import re

import pandas as pd


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

FINAL_BOXES_PATH = W7 / "outputs" / "final_corrected_gt_pen_boxes_v11" / "week7_final_corrected_gt_pen_boxes_v11_for_colour_matching.csv"
FRAME_INDEX_PATH = W6 / "outputs" / "unified_ground_truth" / "week6_scanpoint_frame_index.csv"

V12_ROOT = W7 / "outputs" / "colour_identity" / "colour_marker_evidence_v12"
V12_BOX_FEATURES = V12_ROOT / "week7_colour_marker_evidence_v12_per_box_features.csv"
V12_MARKER_CANDIDATES = V12_ROOT / "week7_colour_marker_evidence_v12_marker_candidates.csv"
V12_FRAME_SUMMARY = V12_ROOT / "week7_colour_marker_evidence_v12_frame_summary.csv"

OUT_ROOT = W7 / "outputs" / "colour_identity" / "colour_rule_workspace_v13"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_INFERRED_SOURCE_AUDIT = OUT_ROOT / "week7_colour_rule_workspace_v13_source_audit.csv"
OUT_FRAME_RULE_TEMPLATE = OUT_ROOT / "week7_colour_rule_workspace_v13_frame_colour_rule_template.csv"
OUT_BOX_RULE_REVIEW = OUT_ROOT / "week7_colour_rule_workspace_v13_box_marker_rule_review.csv"
OUT_ALLOWED_COLOUR_DICTIONARY = OUT_ROOT / "week7_colour_rule_workspace_v13_allowed_colour_dictionary.csv"
OUT_NOTE = W7 / "notes" / "week7_colour_rule_workspace_v13_notes.md"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def normalize_colour(value):
    if pd.isna(value):
        return ""

    s = str(value).strip().lower()

    if not s or s in ["nan", "none", "null", "-"]:
        return ""

    replacements = {
        "blu": "blue",
        "azzurro": "blue",
        "verde": "green",
        "giallo": "yellow",
        "rosso": "red",
        "arancione": "orange",
        "viola": "purple",
        "rosa": "pink",
        "bianco": "white",
        "nero": "black",
        "white_or_light": "white",
        "low_saturation": "",
        "dark": "",
    }

    s = replacements.get(s, s)
    s = re.sub(r"[^a-z0-9_]+", "_", s).strip("_")

    return s


def likely_colour_column(col):
    c = col.lower()
    return (
        "colour" in c
        or "color" in c
        or "marker" in c
        or c in ["red", "blue", "green", "yellow", "orange", "pink", "purple", "white", "black"]
    )


def likely_frame_column(col):
    c = col.lower()
    return (
        "scan_frame_id" in c
        or "frame_id" in c
        or c == "frame"
        or "scanframe" in c
    )


def find_candidate_gt_colour_sources():
    rows = []

    search_roots = [
        W6 / "outputs",
        W7 / "outputs",
    ]

    for root in search_roots:
        if not root.exists():
            continue

        for p in root.rglob("*.csv"):
            try:
                df = pd.read_csv(p, nrows=25)
            except Exception:
                continue

            cols = list(df.columns)

            colour_cols = [c for c in cols if likely_colour_column(c)]
            frame_cols = [c for c in cols if likely_frame_column(c)]

            score = len(colour_cols) * 3 + len(frame_cols) * 2

            # Prefer sources that are not our v12/v13 generated evidence unless useful.
            if "colour_marker_evidence_v12" in str(p):
                score -= 2
            if "colour_rule_workspace_v13" in str(p):
                score -= 10

            if score > 0:
                rows.append({
                    "path": str(p),
                    "score": score,
                    "n_columns": len(cols),
                    "frame_like_columns": " | ".join(frame_cols),
                    "colour_like_columns": " | ".join(colour_cols),
                    "columns_preview": " | ".join(cols[:30]),
                })

    return pd.DataFrame(rows).sort_values("score", ascending=False)


def infer_colours_from_sources(scan_frames):
    """
    Conservative inference:
    - If a source has scan_frame_id and colour-like columns, collect normalized colour values.
    - This creates suggested colours only, not final rules.
    """
    frame_to_colours = {sid: set() for sid in scan_frames}
    source_hits = []

    audit = find_candidate_gt_colour_sources()

    for _, src in audit.head(20).iterrows():
        p = Path(src["path"])

        try:
            df = pd.read_csv(p)
        except Exception:
            continue

        cols = list(df.columns)
        frame_cols = [c for c in cols if likely_frame_column(c)]
        colour_cols = [c for c in cols if likely_colour_column(c)]

        if not frame_cols or not colour_cols:
            continue

        # Prefer exact scan_frame_id.
        if "scan_frame_id" in df.columns:
            frame_col = "scan_frame_id"
        else:
            frame_col = frame_cols[0]

        local_hits = 0

        for _, r in df.iterrows():
            sid = str(r.get(frame_col, "")).strip()

            if sid not in frame_to_colours:
                continue

            for cc in colour_cols:
                val = normalize_colour(r.get(cc, ""))

                if val in ["red", "blue", "green", "yellow", "orange", "pink", "purple", "white", "black", "cyan"]:
                    frame_to_colours[sid].add(val)
                    local_hits += 1

        if local_hits > 0:
            source_hits.append({
                "path": str(p),
                "frame_col_used": frame_col,
                "colour_cols_used": " | ".join(colour_cols),
                "local_colour_hits": local_hits,
            })

    return frame_to_colours, pd.DataFrame(source_hits)


# ---------------------------------------------------------------------
# Load data.
# ---------------------------------------------------------------------
frames = pd.read_csv(FRAME_INDEX_PATH)
final_boxes = pd.read_csv(FINAL_BOXES_PATH)
box_features = pd.read_csv(V12_BOX_FEATURES)
marker_candidates = pd.read_csv(V12_MARKER_CANDIDATES) if V12_MARKER_CANDIDATES.exists() else pd.DataFrame()
frame_summary_v12 = pd.read_csv(V12_FRAME_SUMMARY)

scan_frames = (
    final_boxes["scan_frame_id"]
    .astype(str)
    .drop_duplicates()
    .sort_values()
    .tolist()
)

# Source audit and conservative colour inference.
source_audit = find_candidate_gt_colour_sources()
safe_to_csv(source_audit, OUT_INFERRED_SOURCE_AUDIT)

frame_to_colours, source_hits = infer_colours_from_sources(scan_frames)

# Build frame rule template.
frame_meta_cols = ["scan_frame_id"]

for c in frames.columns:
    lc = c.lower()
    if any(k in lc for k in ["video", "crate", "pen", "session", "camera", "timestamp", "time", "frame"]):
        if c not in frame_meta_cols:
            frame_meta_cols.append(c)

frame_meta = frames[frame_meta_cols].drop_duplicates("scan_frame_id").copy()

rule_rows = []

for sid in scan_frames:
    meta = frame_meta[frame_meta["scan_frame_id"].astype(str) == sid]

    row = {"scan_frame_id": sid}

    if len(meta):
        for c in frame_meta.columns:
            if c != "scan_frame_id":
                row[c] = meta.iloc[0][c]
    else:
        for c in frame_meta.columns:
            if c != "scan_frame_id":
                row[c] = ""

    suggested = sorted(frame_to_colours.get(sid, set()))

    row["suggested_colours_from_existing_gt"] = " | ".join(suggested)
    row["allowed_colour_1"] = suggested[0] if len(suggested) > 0 else ""
    row["allowed_colour_2"] = suggested[1] if len(suggested) > 1 else ""
    row["allowed_colour_3"] = suggested[2] if len(suggested) > 2 else ""
    row["allowed_colour_4"] = suggested[3] if len(suggested) > 3 else ""
    row["allowed_colour_5"] = suggested[4] if len(suggested) > 4 else ""
    row["allowed_colour_6"] = suggested[5] if len(suggested) > 5 else ""

    row["rule_status"] = "needs_manual_confirmation" if len(suggested) != 6 else "suggested_from_existing_gt_verify"
    row["rule_confidence"] = "low" if len(suggested) == 0 else "medium"
    row["manual_notes"] = ""

    rule_rows.append(row)

rule_template = pd.DataFrame(rule_rows)
safe_to_csv(rule_template, OUT_FRAME_RULE_TEMPLATE)

# Allowed colour dictionary.
colour_dict = pd.DataFrame([
    {"canonical_colour": "red", "aliases": "rosso | red", "notes": ""},
    {"canonical_colour": "blue", "aliases": "blu | azzurro | blue", "notes": ""},
    {"canonical_colour": "green", "aliases": "verde | green", "notes": ""},
    {"canonical_colour": "yellow", "aliases": "giallo | yellow", "notes": ""},
    {"canonical_colour": "orange", "aliases": "arancione | orange", "notes": ""},
    {"canonical_colour": "pink", "aliases": "rosa | pink", "notes": ""},
    {"canonical_colour": "purple", "aliases": "viola | purple", "notes": ""},
    {"canonical_colour": "white", "aliases": "bianco | white | white_or_light", "notes": "Usually weak evidence under low saturation."},
    {"canonical_colour": "black", "aliases": "nero | black | dark", "notes": "Usually difficult under shadows."},
    {"canonical_colour": "cyan", "aliases": "cyan", "notes": "Use only if explicitly part of pen rule."},
])
safe_to_csv(colour_dict, OUT_ALLOWED_COLOUR_DICTIONARY)

# Box + marker rule review table.
top_marker = (
    marker_candidates
    .sort_values(["scan_frame_id", "final_box_id", "candidate_rank_in_box"])
    .groupby(["scan_frame_id", "final_box_id"], as_index=False)
    .first()
    if len(marker_candidates)
    else pd.DataFrame(columns=["scan_frame_id", "final_box_id"])
)

review = box_features.merge(
    top_marker[[
        c for c in top_marker.columns
        if c in [
            "scan_frame_id",
            "final_box_id",
            "rough_colour_label",
            "candidate_score",
            "mean_h",
            "mean_s",
            "mean_v",
            "candidate_area_fraction_of_crop",
        ]
    ]],
    on=["scan_frame_id", "final_box_id"],
    how="left",
    suffixes=("", "_top_marker"),
)

allowed_cols = [
    "scan_frame_id",
    "allowed_colour_1",
    "allowed_colour_2",
    "allowed_colour_3",
    "allowed_colour_4",
    "allowed_colour_5",
    "allowed_colour_6",
    "rule_status",
    "rule_confidence",
]

review = review.merge(rule_template[allowed_cols], on="scan_frame_id", how="left")

def allowed_list(row):
    out = []
    for c in [
        "allowed_colour_1",
        "allowed_colour_2",
        "allowed_colour_3",
        "allowed_colour_4",
        "allowed_colour_5",
        "allowed_colour_6",
    ]:
        val = normalize_colour(row.get(c, ""))
        if val:
            out.append(val)
    return out


def marker_rule_flag(row):
    allowed = allowed_list(row)
    marker = normalize_colour(row.get("rough_colour_label", ""))

    if not marker:
        return "no_marker_candidate"

    if len(allowed) == 0:
        return "no_rule_defined_yet"

    if marker in allowed:
        return "top_marker_allowed_by_rule"

    return "top_marker_outside_allowed_rule"


review["allowed_colour_set"] = review.apply(lambda r: " | ".join(allowed_list(r)), axis=1)
review["marker_rule_flag"] = review.apply(marker_rule_flag, axis=1)
review["manual_colour_assignment"] = ""
review["manual_colour_assignment_status"] = "pending_rule_confirmation"
review["manual_notes"] = ""

safe_to_csv(review, OUT_BOX_RULE_REVIEW)

# Notes.
n_rules = len(rule_template)
n_suggested_six = int((rule_template["rule_status"] == "suggested_from_existing_gt_verify").sum())
n_need_confirm = int((rule_template["rule_status"] == "needs_manual_confirmation").sum())
n_review_rows = len(review)

OUT_NOTE.write_text(
    "# Week 7 Colour Rule Workspace v13\n\n"
    "## Purpose\n\n"
    "This step prepares pen/crate-specific colour rules before final colour identity assignment. "
    "The v12 marker candidates are visual evidence only; they are not final labels. "
    "Colour assignment must be constrained by the valid colour set for each annotated pen/crate.\n\n"
    "## Main rule\n\n"
    "Do not assign a colour globally from HSV evidence alone. "
    "For each scan frame, first verify the allowed colour set for that pen/crate, then match pig boxes to those allowed colours.\n\n"
    "## Summary\n\n"
    f"- Frame rule rows: `{n_rules}`\n"
    f"- Rules with six suggested colours from existing GT: `{n_suggested_six}`\n"
    f"- Rules needing manual confirmation: `{n_need_confirm}`\n"
    f"- Box marker/rule review rows: `{n_review_rows}`\n\n"
    "## Outputs\n\n"
    f"- Source audit: `{OUT_INFERRED_SOURCE_AUDIT}`\n"
    f"- Frame colour rule template: `{OUT_FRAME_RULE_TEMPLATE}`\n"
    f"- Allowed colour dictionary: `{OUT_ALLOWED_COLOUR_DICTIONARY}`\n"
    f"- Box marker/rule review table: `{OUT_BOX_RULE_REVIEW}`\n\n"
    "## Next step\n\n"
    "Manually verify or fill the allowed colours in the frame colour rule template. "
    "After that, run the colour identity assignment step.\n"
)

print("Saved:")
print(OUT_INFERRED_SOURCE_AUDIT)
print(OUT_FRAME_RULE_TEMPLATE)
print(OUT_ALLOWED_COLOUR_DICTIONARY)
print(OUT_BOX_RULE_REVIEW)
print(OUT_NOTE)

print()
print("=== colour rule workspace v13 summary ===")
print({
    "frame_rule_rows": n_rules,
    "rules_with_six_suggested_colours": n_suggested_six,
    "rules_needing_manual_confirmation": n_need_confirm,
    "box_marker_rule_review_rows": n_review_rows,
    "source_hits": len(source_hits),
})
