from pathlib import Path
from datetime import datetime
from itertools import permutations
import csv
import math

import cv2
import numpy as np
import pandas as pd


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

FINAL_BOXES_PATH = W7 / "outputs" / "final_corrected_gt_pen_boxes_v11" / "week7_final_corrected_gt_pen_boxes_v11_for_colour_matching.csv"
VIDEO_RULES_PATH = W7 / "outputs" / "colour_identity" / "colour_rule_workspace_v13" / "week7_colour_rule_workspace_v13b_video_level_rule_template.csv"

V12_ROOT = W7 / "outputs" / "colour_identity" / "colour_marker_evidence_v12"
V12_MARKERS = V12_ROOT / "week7_colour_marker_evidence_v12_marker_candidates.csv"
V12_BOX_FEATURES = V12_ROOT / "week7_colour_marker_evidence_v12_per_box_features.csv"

IMG_DIR = W7 / "outputs" / "manual_annotation_v9" / "cvat_coco_import" / "images"

OUT_ROOT = W7 / "outputs" / "colour_identity" / "colour_identity_assignment_v14"
OUT_VIS = OUT_ROOT / "frame_overlays"

OUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT_VIS.mkdir(parents=True, exist_ok=True)

OUT_FRAME_RULES = OUT_ROOT / "week7_colour_identity_assignment_v14_frame_rules_propagated.csv"
OUT_RULE_VALIDATION = OUT_ROOT / "week7_colour_identity_assignment_v14_rule_validation.csv"
OUT_EVIDENCE_MATRIX = OUT_ROOT / "week7_colour_identity_assignment_v14_colour_evidence_matrix.csv"
OUT_ASSIGNMENT = OUT_ROOT / "week7_colour_identity_assignment_v14_auto_assignments.csv"
OUT_REVIEW = OUT_ROOT / "week7_colour_identity_assignment_v14_review_recommended.csv"
OUT_FRAME_SUMMARY = OUT_ROOT / "week7_colour_identity_assignment_v14_frame_summary.csv"
OUT_CONTACT = OUT_ROOT / "week7_colour_identity_assignment_v14_contact_sheet.jpg"
OUT_HTML = OUT_ROOT / "week7_colour_identity_assignment_v14_static_review.html"
OUT_NOTE = W7 / "notes" / "week7_colour_identity_assignment_v14_notes.md"


CANONICAL = ["blue", "green", "cyan", "red", "pink", "purple"]
INVALID = ["yellow", "orange"]

COLOUR_BGR = {
    "blue": (255, 0, 0),
    "green": (0, 190, 0),
    "cyan": (255, 255, 0),
    "red": (0, 0, 255),
    "pink": (203, 120, 255),
    "purple": (180, 0, 180),
    "unknown": (160, 160, 160),
}


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
    repl = {
        "white_or_light": "",
        "low_saturation": "",
        "dark": "",
        "blu": "blue",
        "verde": "green",
        "rosso": "red",
        "rosa": "pink",
        "viola": "purple",
        "giallo": "yellow",
        "arancione": "orange",
    }
    return repl.get(s, s)


def parse_scanframes(s):
    if pd.isna(s):
        return []
    return [x.strip() for x in str(s).split("|") if x.strip()]


def make_contact_sheet(paths, out_path):
    if not paths:
        return False

    sample = paths
    if len(sample) > 16:
        idx = np.linspace(0, len(sample) - 1, 16).round().astype(int)
        sample = [sample[i] for i in idx]

    thumbs = []
    tw, th = 380, 250

    for p in sample:
        img = cv2.imread(str(p))
        if img is None:
            continue
        thumbs.append(cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA))

    if not thumbs:
        return False

    cols = 4
    rows = math.ceil(len(thumbs) / cols)
    sheet = np.full((rows * th, cols * tw, 3), 255, dtype=np.uint8)

    for i, im in enumerate(thumbs):
        r = i // cols
        c = i % cols
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = im

    cv2.imwrite(str(out_path), sheet)
    return True


# ---------------------------------------------------------------------
# Load inputs.
# ---------------------------------------------------------------------
boxes = pd.read_csv(FINAL_BOXES_PATH)
video_rules = pd.read_csv(VIDEO_RULES_PATH)
markers = pd.read_csv(V12_MARKERS) if V12_MARKERS.exists() else pd.DataFrame()
box_features = pd.read_csv(V12_BOX_FEATURES) if V12_BOX_FEATURES.exists() else pd.DataFrame()

for c in ["x1", "y1", "x2", "y2"]:
    boxes[c] = pd.to_numeric(boxes[c], errors="coerce")

boxes = boxes.dropna(subset=["x1", "y1", "x2", "y2"]).copy()

# ---------------------------------------------------------------------
# Propagate video-level rules to frame-level rules and validate.
# ---------------------------------------------------------------------
frame_rule_rows = []
validation_rows = []

for _, r in video_rules.iterrows():
    video_id = str(r["video_id"])
    scanframes = parse_scanframes(r["scan_frame_ids"])
    colours = [norm_colour(r.get(f"video_rule_colour_{i}", "")) for i in range(1, 7)]
    colours = [c for c in colours if c]

    unique_colours = sorted(set(colours))
    missing_canonical = sorted(set(CANONICAL) - set(unique_colours))
    invalid_colours = sorted(set(unique_colours) - set(CANONICAL))

    valid = (
        str(r.get("manual_rule_status", "")).strip().lower() == "confirmed"
        and len(colours) == 6
        and len(unique_colours) == 6
        and set(unique_colours) == set(CANONICAL)
    )

    validation_rows.append({
        "video_id": video_id,
        "manual_rule_status": r.get("manual_rule_status", ""),
        "colours": " | ".join(colours),
        "unique_colour_count": len(unique_colours),
        "missing_canonical_colours": " | ".join(missing_canonical),
        "invalid_colours": " | ".join(invalid_colours),
        "rule_valid_for_v14": valid,
    })

    for sid in scanframes:
        frame_rule_rows.append({
            "scan_frame_id": sid,
            "video_id": video_id,
            "allowed_colour_set": " | ".join(CANONICAL),
            "rule_status_v14": "confirmed_global_valid_marker_set",
            "rule_note_v14": "Allowed: blue, green, cyan, red, pink, purple. Orange/yellow ignored.",
        })

frame_rules = pd.DataFrame(frame_rule_rows)
rule_validation = pd.DataFrame(validation_rows)

safe_to_csv(frame_rules, OUT_FRAME_RULES)
safe_to_csv(rule_validation, OUT_RULE_VALIDATION)

# ---------------------------------------------------------------------
# Build evidence matrix per final box and canonical colour.
# ---------------------------------------------------------------------
if len(markers):
    markers["rough_colour_label_norm"] = markers["rough_colour_label"].apply(norm_colour)
    markers["candidate_score"] = pd.to_numeric(markers["candidate_score"], errors="coerce").fillna(0.0)
else:
    markers = pd.DataFrame(columns=["scan_frame_id", "final_box_id", "rough_colour_label_norm", "candidate_score"])

evidence_rows = []

for _, b in boxes.iterrows():
    sid = str(b["scan_frame_id"])
    bid = str(b["final_box_id"])
    g = markers[
        (markers["scan_frame_id"].astype(str) == sid)
        & (markers["final_box_id"].astype(str) == bid)
    ].copy()

    invalid_hits = g[g["rough_colour_label_norm"].isin(INVALID)].copy()
    all_valid = g[g["rough_colour_label_norm"].isin(CANONICAL)].copy()

    for colour in CANONICAL:
        cg = all_valid[all_valid["rough_colour_label_norm"] == colour]

        if len(cg):
            score = float(cg["candidate_score"].max())
            count = int(len(cg))
        else:
            score = 0.0
            count = 0

        evidence_rows.append({
            "scan_frame_id": sid,
            "final_box_id": bid,
            "candidate_colour": colour,
            "evidence_score": score,
            "candidate_count_for_colour": count,
            "invalid_or_ignored_marker_colours_seen": " | ".join(sorted(set(invalid_hits["rough_colour_label_norm"].tolist()))),
            "invalid_or_ignored_marker_count": int(len(invalid_hits)),
        })

evidence = pd.DataFrame(evidence_rows)
safe_to_csv(evidence, OUT_EVIDENCE_MATRIX)

# ---------------------------------------------------------------------
# Assign one unique allowed colour per box within each frame.
# ---------------------------------------------------------------------
assignment_rows = []
frame_summary_rows = []
overlay_paths = []

for sid, g_boxes in boxes.groupby("scan_frame_id", sort=True):
    sid = str(sid)
    g_boxes = g_boxes.sort_values("final_box_id").copy()
    box_ids = g_boxes["final_box_id"].astype(str).tolist()
    n = len(box_ids)

    score_matrix = []

    for bid in box_ids:
        row_scores = []
        eg = evidence[
            (evidence["scan_frame_id"].astype(str) == sid)
            & (evidence["final_box_id"].astype(str) == bid)
        ]

        for colour in CANONICAL:
            val = eg.loc[eg["candidate_colour"] == colour, "evidence_score"]
            row_scores.append(float(val.iloc[0]) if len(val) else 0.0)

        score_matrix.append(row_scores)

    score_matrix = np.array(score_matrix, dtype=float)

    best_perm = None
    best_total = -1.0

    # If fewer than 6 visible boxes, assign a unique subset of colours.
    for perm in permutations(range(len(CANONICAL)), n):
        total = 0.0
        for i, cidx in enumerate(perm):
            total += score_matrix[i, cidx]

        if total > best_total:
            best_total = total
            best_perm = perm

    assigned_colours = [CANONICAL[i] for i in best_perm] if best_perm is not None else ["unknown"] * n

    confidence_counts = {"high": 0, "medium": 0, "low_conflict": 0, "low_no_marker": 0}
    review_count = 0

    for i, (_, b) in enumerate(g_boxes.iterrows()):
        bid = str(b["final_box_id"])
        assigned = assigned_colours[i]
        assigned_idx = CANONICAL.index(assigned)
        scores = score_matrix[i, :]

        order = np.argsort(scores)[::-1]
        top_idx = int(order[0])
        second_idx = int(order[1]) if len(order) > 1 else top_idx

        top_colour = CANONICAL[top_idx]
        top_score = float(scores[top_idx])
        second_score = float(scores[second_idx])
        assigned_score = float(scores[assigned_idx])

        if assigned_score <= 0:
            confidence = "low_no_marker"
        elif assigned != top_colour:
            confidence = "low_conflict"
        elif second_score <= 0 or (top_score / max(second_score, 1e-9)) >= 1.35:
            confidence = "high"
        else:
            confidence = "medium"

        confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
        manual_review_recommended = confidence not in ["high"]

        if manual_review_recommended:
            review_count += 1

        eg_bid = evidence[
            (evidence["scan_frame_id"].astype(str) == sid)
            & (evidence["final_box_id"].astype(str) == bid)
        ]

        invalid_seen = ""
        invalid_count = 0
        if len(eg_bid):
            invalid_seen = str(eg_bid["invalid_or_ignored_marker_colours_seen"].iloc[0])
            invalid_count = int(eg_bid["invalid_or_ignored_marker_count"].iloc[0])

        assignment_rows.append({
            "scan_frame_id": sid,
            "final_box_id": bid,
            "assigned_colour_v14": assigned,
            "assignment_method": "rule_constrained_unique_assignment_from_marker_evidence",
            "assignment_confidence_v14": confidence,
            "manual_review_recommended": manual_review_recommended,
            "assigned_colour_evidence_score": assigned_score,
            "top_evidence_colour": top_colour,
            "top_evidence_score": top_score,
            "second_evidence_colour": CANONICAL[second_idx],
            "second_evidence_score": second_score,
            "ignored_invalid_marker_colours": invalid_seen,
            "ignored_invalid_marker_count": invalid_count,
            "allowed_colour_set": " | ".join(CANONICAL),
            "x1": float(b["x1"]),
            "y1": float(b["y1"]),
            "x2": float(b["x2"]),
            "y2": float(b["y2"]),
        })

    missing_colours = sorted(set(CANONICAL) - set(assigned_colours))

    frame_summary_rows.append({
        "scan_frame_id": sid,
        "final_box_count": n,
        "assigned_colour_count": len(assigned_colours),
        "unique_assigned_colour_count": len(set(assigned_colours)),
        "missing_colours_due_to_fewer_visible_boxes": " | ".join(missing_colours),
        "total_assignment_evidence_score": best_total,
        "high_confidence_assignments": confidence_counts.get("high", 0),
        "medium_confidence_assignments": confidence_counts.get("medium", 0),
        "low_conflict_assignments": confidence_counts.get("low_conflict", 0),
        "low_no_marker_assignments": confidence_counts.get("low_no_marker", 0),
        "manual_review_recommended_boxes": review_count,
        "frame_review_recommended": review_count > 0 or n < 6,
    })

assignments = pd.DataFrame(assignment_rows)
frame_summary = pd.DataFrame(frame_summary_rows)

safe_to_csv(assignments, OUT_ASSIGNMENT)
safe_to_csv(assignments[assignments["manual_review_recommended"] == True].copy(), OUT_REVIEW)
safe_to_csv(frame_summary, OUT_FRAME_SUMMARY)

# ---------------------------------------------------------------------
# Visual overlays.
# ---------------------------------------------------------------------
for sid, g in assignments.groupby("scan_frame_id", sort=True):
    img_path = IMG_DIR / f"{sid}.jpg"

    if not img_path.exists():
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    overlay = img.copy()

    for _, r in g.iterrows():
        colour = str(r["assigned_colour_v14"])
        bgr = COLOUR_BGR.get(colour, COLOUR_BGR["unknown"])

        x1 = int(round(float(r["x1"])))
        y1 = int(round(float(r["y1"])))
        x2 = int(round(float(r["x2"])))
        y2 = int(round(float(r["y2"])))

        thickness = 3 if r["assignment_confidence_v14"] in ["high", "medium"] else 2

        cv2.rectangle(overlay, (x1, y1), (x2, y2), bgr, thickness)

        label = f"{colour}"
        if r["assignment_confidence_v14"].startswith("low"):
            label += " ?"

        cv2.putText(
            overlay,
            label,
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            bgr,
            2,
            cv2.LINE_AA,
        )

    fs = frame_summary[frame_summary["scan_frame_id"] == sid].iloc[0]
    title = (
        f"{sid} | boxes={int(fs['final_box_count'])} | "
        f"review={int(fs['manual_review_recommended_boxes'])}"
    )

    cv2.putText(
        overlay,
        title,
        (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    out_path = OUT_VIS / f"{sid}_colour_assignment_v14.jpg"
    cv2.imwrite(str(out_path), overlay)
    overlay_paths.append(out_path)

contact_ok = make_contact_sheet(overlay_paths, OUT_CONTACT)

# HTML.
cards = []
for _, fs in frame_summary.sort_values("scan_frame_id").iterrows():
    sid = str(fs["scan_frame_id"])
    src = f"frame_overlays/{sid}_colour_assignment_v14.jpg"
    cls = "review" if bool(fs["frame_review_recommended"]) else "ok"

    cards.append(
        f"""
        <div class="frame-card {cls}">
          <h3>{sid}</h3>
          <p>
            boxes={fs['final_box_count']} |
            high={fs['high_confidence_assignments']} |
            medium={fs['medium_confidence_assignments']} |
            low_conflict={fs['low_conflict_assignments']} |
            low_no_marker={fs['low_no_marker_assignments']} |
            review_boxes={fs['manual_review_recommended_boxes']} |
            missing_colours={fs['missing_colours_due_to_fewer_visible_boxes']}
          </p>
          <img src="{src}">
        </div>
        """
    )

html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Week 7 Colour Identity Assignment v14</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      background: #f7f7f7;
    }}
    .legend {{
      background: white;
      padding: 14px;
      border-left: 6px solid #222;
      border-radius: 8px;
      margin-bottom: 20px;
    }}
    .frame-card {{
      background: white;
      margin: 18px 0;
      padding: 12px;
      border-radius: 8px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.15);
    }}
    .ok {{ border-left: 8px solid green; }}
    .review {{ border-left: 8px solid darkorange; }}
    img {{
      max-width: 100%;
      border: 1px solid #ddd;
      display: block;
    }}
  </style>
</head>
<body>
  <h1>Week 7 Colour Identity Assignment v14</h1>
  <div class="legend">
    <p>Allowed marker colours: blue, green, cyan, red, pink, purple.</p>
    <p>Orange/yellow evidence is ignored as invalid marker evidence.</p>
    <p>Labels with ? indicate low-confidence assignments that should be reviewed.</p>
  </div>
  {''.join(cards)}
</body>
</html>
"""
OUT_HTML.write_text(html)

# Notes.
total_boxes = len(assignments)
review_boxes = int(assignments["manual_review_recommended"].sum())
high = int((assignments["assignment_confidence_v14"] == "high").sum())
medium = int((assignments["assignment_confidence_v14"] == "medium").sum())
low_conflict = int((assignments["assignment_confidence_v14"] == "low_conflict").sum())
low_no_marker = int((assignments["assignment_confidence_v14"] == "low_no_marker").sum())
review_frames = int(frame_summary["frame_review_recommended"].sum())

OUT_NOTE.write_text(
    "# Week 7 Colour Identity Assignment v14\n\n"
    "## Purpose\n\n"
    "This step assigns colour identities to manually corrected GT-pen pig boxes using rule-constrained marker evidence. "
    "The valid marker colours were manually confirmed as blue, green, cyan, red, pink, and purple. "
    "Orange and yellow are ignored as invalid/lighting evidence.\n\n"
    "## Important\n\n"
    "This is an automatic first-pass colour assignment, not the final manually verified identity table. "
    "Low-confidence assignments should be reviewed before behaviour-label fusion.\n\n"
    "## Summary\n\n"
    f"- Total assigned boxes: `{total_boxes}`\n"
    f"- High confidence: `{high}`\n"
    f"- Medium confidence: `{medium}`\n"
    f"- Low conflict: `{low_conflict}`\n"
    f"- Low no-marker: `{low_no_marker}`\n"
    f"- Boxes recommended for review: `{review_boxes}`\n"
    f"- Frames recommended for review: `{review_frames}`\n\n"
    "## Outputs\n\n"
    f"- Frame rules propagated: `{OUT_FRAME_RULES}`\n"
    f"- Rule validation: `{OUT_RULE_VALIDATION}`\n"
    f"- Evidence matrix: `{OUT_EVIDENCE_MATRIX}`\n"
    f"- Automatic assignments: `{OUT_ASSIGNMENT}`\n"
    f"- Review-recommended boxes: `{OUT_REVIEW}`\n"
    f"- Frame summary: `{OUT_FRAME_SUMMARY}`\n"
    f"- Static review HTML: `{OUT_HTML}`\n"
)

print("Saved:")
print(OUT_FRAME_RULES)
print(OUT_RULE_VALIDATION)
print(OUT_EVIDENCE_MATRIX)
print(OUT_ASSIGNMENT)
print(OUT_REVIEW)
print(OUT_FRAME_SUMMARY)
print(OUT_CONTACT)
print(OUT_HTML)
print(OUT_NOTE)

print()
print("=== colour identity assignment v14 summary ===")
print({
    "total_assigned_boxes": total_boxes,
    "high_confidence": high,
    "medium_confidence": medium,
    "low_conflict": low_conflict,
    "low_no_marker": low_no_marker,
    "boxes_recommended_for_review": review_boxes,
    "frames_recommended_for_review": review_frames,
    "contact_sheet_generated": contact_ok,
})
