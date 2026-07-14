from pathlib import Path
from datetime import datetime
import csv
import math

import cv2
import numpy as np
import pandas as pd


ROOT = Path.home() / "PigBench"

W6 = ROOT / "Week6_Unibo_Dataset_Validation"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

FRAME_INDEX_PATH = W6 / "outputs" / "unified_ground_truth" / "week6_scanpoint_frame_index.csv"
V5_SELECTION_PATH = W7 / "outputs" / "roi_and_crate_mapping" / "week7_gt_pen_detection_selection_v5_confidence_review.csv"
V5_FRAME_SUMMARY_PATH = W7 / "outputs" / "roi_and_crate_mapping" / "week7_gt_pen_detection_selection_v5_frame_summary.csv"

OUT_ROI = W7 / "outputs" / "roi_and_crate_mapping"
OUT_VIS = W7 / "outputs" / "visualizations"
OUT_STATS = W7 / "outputs" / "dataset_statistics"
OUT_NOTES = W7 / "notes"

for p in [OUT_ROI, OUT_VIS, OUT_STATS, OUT_NOTES]:
    p.mkdir(parents=True, exist_ok=True)

REVIEW_DIR = OUT_VIS / "gt_pen_selection_v6_final_safe_review_frames"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)

OUT_ALL = OUT_ROI / "week7_gt_pen_detection_selection_v6_final_safe_layer.csv"
OUT_STRICT = OUT_ROI / "week7_gt_pen_detection_selection_v6_strict_auto_safe_for_colour_matching.csv"
OUT_REVIEW = OUT_ROI / "week7_gt_pen_detection_selection_v6_manual_review_required.csv"
OUT_FRAME_SUMMARY = OUT_ROI / "week7_gt_pen_detection_selection_v6_frame_summary.csv"
OUT_MANUAL_TEMPLATE = OUT_ROI / "week7_gt_pen_detection_selection_v6_manual_override_template.csv"

OUT_CONTACT_ALL = OUT_VIS / "week7_gt_pen_selection_v6_all_frames_contact_sheet.jpg"
OUT_CONTACT_REVIEW = OUT_VIS / "week7_gt_pen_selection_v6_review_required_contact_sheet.jpg"
OUT_HTML = OUT_VIS / "week7_gt_pen_selection_v6_static_review.html"

OUT_NOTE = OUT_NOTES / "week7_gt_pen_selection_v6_final_safe_layer_notes.md"
TASK_TRACKER_PATH = OUT_STATS / "week7_master_task_tracker.csv"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def to_bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def resolve_path(value):
    if pd.isna(value):
        return None

    value = str(value).strip()
    if not value:
        return None

    candidates = [
        Path(value),
        W6 / value,
        W7 / value,
        ROOT / value,
        Path.home() / value,
    ]

    for c in candidates:
        if c.exists():
            return c

    return None


def first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def make_frame_review_images(frames, df):
    image_col = first_col(frames, ["frame_image_path", "image_path", "frame_path", "scanpoint_frame_path"])

    if image_col is None:
        return pd.DataFrame()

    frame_list = frames.drop_duplicates("scan_frame_id").sort_values("scan_frame_id").copy()
    rows = []

    for _, fr in frame_list.iterrows():
        sid = str(fr["scan_frame_id"])
        img_path = resolve_path(fr[image_col])

        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        overlay = img.copy()
        frame_dets = df[df["scan_frame_id"].astype(str) == sid].copy()

        for _, d in frame_dets.iterrows():
            x1 = int(round(float(d["x1"])))
            y1 = int(round(float(d["y1"])))
            x2 = int(round(float(d["x2"])))
            y2 = int(round(float(d["y2"])))

            cls = str(d["v6_selection_class"])

            if cls == "strict_auto_safe":
                color = (0, 220, 0)
                thickness = 3
                prefix = "SAFE"
            elif cls == "selected_review_required":
                color = (0, 165, 255)
                thickness = 3
                prefix = "REVIEW"
            elif cls == "ignored_high_risk_review":
                color = (0, 0, 255)
                thickness = 2
                prefix = "RISK"
            else:
                color = (135, 135, 135)
                thickness = 1
                prefix = "IGN"

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

            det_id = d.get("det_id", "")
            conf = float(d.get("v6_final_safe_confidence", 0))
            label = f"{prefix} d{det_id} {conf:.2f}"

            cv2.putText(
                overlay,
                label,
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

        strict_count = int((frame_dets["v6_selection_class"] == "strict_auto_safe").sum())
        review_count = int(frame_dets["v6_manual_review_required"].sum())

        title = f"{sid} | strict_safe={strict_count} | review_required={review_count}"

        cv2.putText(
            overlay,
            title,
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out_path = REVIEW_DIR / f"{sid}_gt_pen_selection_v6_review.jpg"
        cv2.imwrite(str(out_path), overlay)

        rows.append({
            "scan_frame_id": sid,
            "review_image_path": str(out_path),
            "strict_auto_safe_count": strict_count,
            "manual_review_required_count": review_count,
            "frame_needs_manual_review_v6": bool(
                review_count > 0
                or strict_count < 6
            ),
        })

    return pd.DataFrame(rows)


def make_contact_sheet(review_index, output_path, only_review=False):
    df = review_index.copy()

    if only_review:
        df = df[df["frame_needs_manual_review_v6"] == True].copy()

    if len(df) == 0:
        return False

    if len(df) > 16:
        idx = np.linspace(0, len(df) - 1, 16).round().astype(int)
        df = df.iloc[idx]

    thumbs = []
    thumb_w, thumb_h = 380, 250

    for _, r in df.iterrows():
        p = Path(r["review_image_path"])

        if not p.exists():
            continue

        img = cv2.imread(str(p))
        if img is None:
            continue

        thumbs.append(cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA))

    if not thumbs:
        return False

    cols = 4
    rows = math.ceil(len(thumbs) / cols)
    sheet = np.full((rows * thumb_h, cols * thumb_w, 3), 255, dtype=np.uint8)

    for i, img in enumerate(thumbs):
        r = i // cols
        c = i % cols
        y0 = r * thumb_h
        x0 = c * thumb_w
        sheet[y0:y0 + thumb_h, x0:x0 + thumb_w] = img

    cv2.imwrite(str(output_path), sheet)
    return True


def make_html(review_index, out_path):
    cards = []

    for _, r in review_index.sort_values("scan_frame_id").iterrows():
        img_path = Path(str(r["review_image_path"]))
        src = "gt_pen_selection_v6_final_safe_review_frames/" + img_path.name

        status_class = "review" if bool(r["frame_needs_manual_review_v6"]) else "safe"

        cards.append(
            f"""
            <div class="frame-card {status_class}">
              <h3>{r['scan_frame_id']}</h3>
              <p>
                strict_auto_safe={r['strict_auto_safe_count']} |
                manual_review_required={r['manual_review_required_count']} |
                frame_needs_manual_review={r['frame_needs_manual_review_v6']}
              </p>
              <img src="{src}" alt="{r['scan_frame_id']}">
            </div>
            """
        )

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Week 7 GT Pen Selection v6 Final-Safe Review</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      background: #f7f7f7;
    }}
    .legend {{
      background: white;
      padding: 14px;
      border-radius: 8px;
      margin-bottom: 20px;
      border-left: 6px solid #222;
    }}
    .frame-card {{
      background: white;
      margin: 18px 0;
      padding: 12px;
      border-radius: 8px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.15);
    }}
    .frame-card.safe {{
      border-left: 8px solid green;
    }}
    .frame-card.review {{
      border-left: 8px solid darkorange;
    }}
    img {{
      max-width: 100%;
      border: 1px solid #ddd;
      display: block;
    }}
    .safe-text {{
      color: green;
      font-weight: bold;
    }}
    .review-text {{
      color: darkorange;
      font-weight: bold;
    }}
    .risk-text {{
      color: red;
      font-weight: bold;
    }}
  </style>
</head>
<body>
  <h1>Week 7 Ground Truth Pen Selection v6 Final-Safe Review</h1>

  <div class="legend">
    <p><span class="safe-text">SAFE</span>: strict automatic safe detection for colour matching.</p>
    <p><span class="review-text">REVIEW</span>: selected detection, but excluded from strict automatic matching until human confirmation.</p>
    <p><span class="risk-text">RISK</span>: ignored high-risk detection that should be checked before any manual add.</p>
    <p><b>Rule:</b> colour matching should use only strict_auto_safe detections unless a manual override confirms additional detections.</p>
  </div>

  {''.join(cards)}
</body>
</html>
"""
    out_path.write_text(html)


# ---------------------------------------------------------------------
# Load inputs.
# ---------------------------------------------------------------------
frames = pd.read_csv(FRAME_INDEX_PATH)
sel = pd.read_csv(V5_SELECTION_PATH)
v5_frame_summary = pd.read_csv(V5_FRAME_SUMMARY_PATH)

for c in ["x1", "y1", "x2", "y2"]:
    sel[c] = pd.to_numeric(sel[c], errors="coerce")

sel = sel.dropna(subset=["x1", "y1", "x2", "y2"]).copy()

sel["selected_for_gt_pen_matching_v4"] = to_bool_series(sel["selected_for_gt_pen_matching_v4"])
sel["recommended_for_colour_matching_v5"] = to_bool_series(sel["recommended_for_colour_matching_v5"])
sel["recommended_for_strict_colour_matching_v5"] = to_bool_series(sel["recommended_for_strict_colour_matching_v5"])

sel["roi_overlap_num"] = pd.to_numeric(sel.get("roi_overlap_num", 0), errors="coerce").fillna(0.0)
sel["detector_score_num"] = pd.to_numeric(sel.get("detector_score_num", sel.get("score", 0.5)), errors="coerce").fillna(0.5)
sel["relative_y_rank"] = pd.to_numeric(sel.get("relative_y_rank", 0.5), errors="coerce").fillna(0.5)
sel["selected_confidence_score_v5"] = pd.to_numeric(sel.get("selected_confidence_score_v5", 0.0), errors="coerce").fillna(0.0)

# v6 confidence.
sel["v6_final_safe_confidence"] = (
    0.50 * sel["roi_overlap_num"]
    + 0.30 * sel["detector_score_num"]
    + 0.20 * (1.0 - sel["relative_y_rank"])
)

# Strict rule:
# We deliberately prefer false negatives over wrong-pen false positives.
strict_auto_safe = (
    (sel["gt_pen_selection_status_v5"] == "clean_selected")
    & (sel["selected_for_gt_pen_matching_v4"])
    & (sel["roi_overlap_num"] >= 0.70)
    & (sel["detector_score_num"] >= 0.35)
    & (
        (sel["relative_y_rank"] <= 0.85)
        | (sel["roi_overlap_num"] >= 0.90)
    )
    & (sel["v6_final_safe_confidence"] >= 0.68)
)

selected_review_required = (
    sel["selected_for_gt_pen_matching_v4"]
    & (~strict_auto_safe)
)

ignored_high_risk_review = (
    (~sel["selected_for_gt_pen_matching_v4"])
    & (sel["gt_pen_selection_status_v5"] == "ignored_high_risk")
)

sel["strict_auto_safe_for_colour_matching_v6"] = strict_auto_safe
sel["v6_manual_review_required"] = selected_review_required | ignored_high_risk_review

sel["v6_selection_class"] = "ignored"

sel.loc[strict_auto_safe, "v6_selection_class"] = "strict_auto_safe"
sel.loc[selected_review_required, "v6_selection_class"] = "selected_review_required"
sel.loc[ignored_high_risk_review, "v6_selection_class"] = "ignored_high_risk_review"

sel["v6_default_action"] = "auto_ignore"

sel.loc[strict_auto_safe, "v6_default_action"] = "auto_keep_for_strict_colour_matching"
sel.loc[selected_review_required, "v6_default_action"] = "needs_manual_review_before_colour_matching"
sel.loc[ignored_high_risk_review, "v6_default_action"] = "inspect_before_possible_manual_add"

safe_to_csv(sel, OUT_ALL)

strict_df = sel[sel["strict_auto_safe_for_colour_matching_v6"]].copy()
review_df = sel[sel["v6_manual_review_required"]].copy()

safe_to_csv(strict_df, OUT_STRICT)
safe_to_csv(review_df, OUT_REVIEW)

# Frame summary.
frame_rows = []

for sid, g in sel.groupby("scan_frame_id", sort=True):
    total = len(g)
    strict_count = int(g["strict_auto_safe_for_colour_matching_v6"].sum())
    selected_review_count = int((g["v6_selection_class"] == "selected_review_required").sum())
    ignored_high_risk_count = int((g["v6_selection_class"] == "ignored_high_risk_review").sum())
    selected_v4_count = int(g["selected_for_gt_pen_matching_v4"].sum())

    needs_review = (
        strict_count < 6
        or selected_review_count > 0
        or ignored_high_risk_count > 0
    )

    reasons = []
    if strict_count < 6:
        reasons.append("strict_auto_safe_less_than_6")
    if selected_review_count > 0:
        reasons.append("selected_review_required_present")
    if ignored_high_risk_count > 0:
        reasons.append("ignored_high_risk_review_present")

    frame_rows.append({
        "scan_frame_id": sid,
        "total_detections": total,
        "selected_v4_count": selected_v4_count,
        "strict_auto_safe_count_v6": strict_count,
        "selected_review_required_count_v6": selected_review_count,
        "ignored_high_risk_review_count_v6": ignored_high_risk_count,
        "frame_needs_manual_review_v6": needs_review,
        "reason": "; ".join(reasons),
    })

frame_summary = pd.DataFrame(frame_rows)
safe_to_csv(frame_summary, OUT_FRAME_SUMMARY)

# Manual override template: include all rows that need review, plus all rows from frames with review needed.
review_frame_ids = set(frame_summary.loc[frame_summary["frame_needs_manual_review_v6"], "scan_frame_id"].astype(str))
template = sel[sel["scan_frame_id"].astype(str).isin(review_frame_ids)].copy()

template["manual_override_action"] = np.where(
    template["strict_auto_safe_for_colour_matching_v6"],
    "auto_keep_no_action_needed",
    np.where(
        template["v6_selection_class"] == "selected_review_required",
        "choose_keep_or_reject",
        np.where(
            template["v6_selection_class"] == "ignored_high_risk_review",
            "choose_add_or_ignore",
            "auto_ignore_no_action_needed",
        ),
    ),
)

template["manual_final_use_for_colour_matching"] = template["strict_auto_safe_for_colour_matching_v6"]
template["manual_reviewer_notes"] = ""

safe_to_csv(template, OUT_MANUAL_TEMPLATE)

# Visual outputs.
review_index = make_frame_review_images(frames, sel)
safe_to_csv(review_index, OUT_VIS / "week7_gt_pen_selection_v6_review_image_index.csv")

contact_all_ok = make_contact_sheet(review_index, OUT_CONTACT_ALL, only_review=False)
contact_review_ok = make_contact_sheet(review_index, OUT_CONTACT_REVIEW, only_review=True)
make_html(review_index, OUT_HTML)

# Update task tracker.
if TASK_TRACKER_PATH.exists():
    tracker = pd.read_csv(TASK_TRACKER_PATH)
    mask = tracker["stage"].astype(str) == "2"

    if mask.any():
        tracker.loc[mask, "status"] = "gt_pen_selection_v6_final_safe_layer_generated"

    safe_to_csv(tracker, TASK_TRACKER_PATH)

# Notes.
total = len(sel)
strict_total = int(sel["strict_auto_safe_for_colour_matching_v6"].sum())
review_total = int((sel["v6_selection_class"] == "selected_review_required").sum())
risk_total = int((sel["v6_selection_class"] == "ignored_high_risk_review").sum())
ignored_total = int((sel["v6_selection_class"] == "ignored").sum())
frames_review = int(frame_summary["frame_needs_manual_review_v6"].sum())

with open(OUT_NOTE, "w") as f:
    f.write("# Week 7 GT Pen Selection v6 Final-Safe Layer\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This step creates a final-safe automatic subset for colour matching and separates all uncertain detections into a manual review layer. "
        "The purpose is to avoid silently using wrong-pen detections for behaviour-label association.\n\n"
    )

    f.write("## Main rule\n\n")
    f.write("Use only `strict_auto_safe_for_colour_matching_v6 = True` for automatic colour matching.\n\n")

    f.write("## Summary\n\n")
    f.write(f"- Total detections: `{total}`\n")
    f.write(f"- Strict automatic safe detections: `{strict_total}`\n")
    f.write(f"- Selected detections requiring review: `{review_total}`\n")
    f.write(f"- Ignored high-risk detections requiring review: `{risk_total}`\n")
    f.write(f"- Ignored detections: `{ignored_total}`\n")
    f.write(f"- Frames requiring manual review: `{frames_review}` out of `{frame_summary['scan_frame_id'].nunique()}`\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The strict automatic subset may contain fewer than six detections per frame. "
        "This is intentional: it is safer to leave a pig slot unresolved than to assign a wrong-pen detection to a behaviour label. "
        "Manual overrides can later add or reject detections using the manual override template.\n\n"
    )

    f.write("## Outputs\n\n")
    for p in [
        OUT_ALL,
        OUT_STRICT,
        OUT_REVIEW,
        OUT_FRAME_SUMMARY,
        OUT_MANUAL_TEMPLATE,
        OUT_CONTACT_ALL,
        OUT_CONTACT_REVIEW,
        OUT_HTML,
    ]:
        f.write(f"- `{p}`\n")

print("Saved:")
print(OUT_ALL)
print(OUT_STRICT)
print(OUT_REVIEW)
print(OUT_FRAME_SUMMARY)
print(OUT_MANUAL_TEMPLATE)
print(OUT_CONTACT_ALL)
print(OUT_CONTACT_REVIEW)
print(OUT_HTML)
print(OUT_NOTE)

print()
print("=== v6 summary ===")
print({
    "total_detections": total,
    "strict_auto_safe": strict_total,
    "selected_review_required": review_total,
    "ignored_high_risk_review": risk_total,
    "ignored": ignored_total,
    "frames_requiring_manual_review": frames_review,
    "contact_all_ok": contact_all_ok,
    "contact_review_ok": contact_review_ok,
})
