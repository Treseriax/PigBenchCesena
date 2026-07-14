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
V4_SELECTION_PATH = W7 / "outputs" / "roi_and_crate_mapping" / "week7_gt_pen_detection_selection_v4.csv"

OUT_ROI = W7 / "outputs" / "roi_and_crate_mapping"
OUT_VIS = W7 / "outputs" / "visualizations"
OUT_STATS = W7 / "outputs" / "dataset_statistics"
OUT_NOTES = W7 / "notes"

for p in [OUT_ROI, OUT_VIS, OUT_STATS, OUT_NOTES]:
    p.mkdir(parents=True, exist_ok=True)

REVIEW_DIR = OUT_VIS / "gt_pen_selection_v5_review_frames"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)

OUT_SELECTION = OUT_ROI / "week7_gt_pen_detection_selection_v5_confidence_review.csv"
OUT_FRAME_SUMMARY = OUT_ROI / "week7_gt_pen_detection_selection_v5_frame_summary.csv"
OUT_RISKY_DETECTIONS = OUT_ROI / "week7_gt_pen_detection_selection_v5_risky_detections.csv"
OUT_REVIEW_TEMPLATE = OUT_ROI / "week7_gt_pen_detection_selection_v5_manual_override_template.csv"

OUT_CONTACT_ALL = OUT_VIS / "week7_gt_pen_selection_v5_all_review_contact_sheet.jpg"
OUT_CONTACT_RISKY = OUT_VIS / "week7_gt_pen_selection_v5_risky_frames_contact_sheet.jpg"
OUT_HTML = OUT_VIS / "week7_gt_pen_selection_v5_static_review.html"

OUT_NOTE = OUT_NOTES / "week7_gt_pen_selection_v5_confidence_review_notes.md"
TASK_TRACKER_PATH = OUT_STATS / "week7_master_task_tracker.csv"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


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


def make_frame_review_images(frames, selection):
    image_col = first_col(frames, ["frame_image_path", "image_path", "frame_path", "scanpoint_frame_path"])

    if image_col is None:
        return pd.DataFrame()

    rows = []

    frame_list = frames.drop_duplicates("scan_frame_id").sort_values("scan_frame_id").copy()

    for _, fr in frame_list.iterrows():
        sid = str(fr["scan_frame_id"])
        img_path = resolve_path(fr[image_col])

        if img_path is None:
            continue

        img = cv2.imread(str(img_path))

        if img is None:
            continue

        overlay = img.copy()
        frame_dets = selection[selection["scan_frame_id"].astype(str) == sid].copy()

        for _, d in frame_dets.iterrows():
            x1 = int(round(float(d["x1"])))
            y1 = int(round(float(d["y1"])))
            x2 = int(round(float(d["x2"])))
            y2 = int(round(float(d["y2"])))

            cls = str(d["gt_pen_selection_status_v5"])

            if cls == "clean_selected":
                color = (0, 220, 0)
                thickness = 3
                prefix = "CLEAN"
            elif cls == "selected_needs_review":
                color = (0, 165, 255)
                thickness = 3
                prefix = "REVIEW"
            elif cls == "ignored_high_risk":
                color = (0, 0, 255)
                thickness = 2
                prefix = "IGN_RISK"
            else:
                color = (130, 130, 130)
                thickness = 1
                prefix = "IGN"

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

            label = f"{prefix} d{d.get('det_id', '')}"
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

        clean_count = int((frame_dets["gt_pen_selection_status_v5"] == "clean_selected").sum())
        review_count = int((frame_dets["gt_pen_selection_status_v5"] == "selected_needs_review").sum())
        selected_count = int(frame_dets["selected_for_gt_pen_matching_v4"].sum())

        title = f"{sid} | selected={selected_count} | clean={clean_count} | review={review_count}"

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

        out_path = REVIEW_DIR / f"{sid}_gt_pen_selection_v5_review.jpg"
        cv2.imwrite(str(out_path), overlay)

        rows.append({
            "scan_frame_id": sid,
            "review_image_path": str(out_path),
            "selected_count": selected_count,
            "clean_selected_count": clean_count,
            "selected_needs_review_count": review_count,
        })

    return pd.DataFrame(rows)


def make_contact_sheet(review_index, output_path, only_risky=False):
    df = review_index.copy()

    if only_risky:
        df = df[df["frame_needs_review"] == True].copy()

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

        resized = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        thumbs.append(resized)

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
    rows = []

    for _, r in review_index.sort_values("scan_frame_id").iterrows():
        img_path = Path(r["review_image_path"])

        try:
            rel_img = img_path.relative_to(W7)
        except Exception:
            rel_img = img_path

        rows.append(
            f"""
            <div class="frame-card">
              <h3>{r['scan_frame_id']}</h3>
              <p>
                selected={r['selected_count']} |
                clean={r['clean_selected_count']} |
                review={r['selected_needs_review_count']} |
                risky={r['frame_needs_review']}
              </p>
              <img src="../{rel_img}" />
            </div>
            """
        )

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Week 7 GT Pen Selection v5 Review</title>
      <style>
        body {{
          font-family: Arial, sans-serif;
          margin: 24px;
          background: #f7f7f7;
        }}
        .legend {{
          background: white;
          padding: 12px;
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
        img {{
          max-width: 100%;
          border: 1px solid #ddd;
        }}
        .clean {{ color: green; font-weight: bold; }}
        .review {{ color: darkorange; font-weight: bold; }}
        .risk {{ color: red; font-weight: bold; }}
      </style>
    </head>
    <body>
      <h1>Week 7 Ground Truth Pen Selection v5 Review</h1>
      <div class="legend">
        <p><span class="clean">CLEAN</span>: selected and high-confidence.</p>
        <p><span class="review">REVIEW</span>: selected, but should be visually checked.</p>
        <p><span class="risk">IGN_RISK</span>: ignored but suspicious, usually because it may overlap the candidate region.</p>
        <p>The aim is not to force six detections blindly. The aim is to avoid using wrong-pen detections for colour and behaviour matching.</p>
      </div>
      {''.join(rows)}
    </body>
    </html>
    """

    out_path.write_text(html)


# ---------------------------------------------------------------------
# Load data.
# ---------------------------------------------------------------------
frames = pd.read_csv(FRAME_INDEX_PATH)
sel = pd.read_csv(V4_SELECTION_PATH)

for c in ["x1", "y1", "x2", "y2"]:
    sel[c] = pd.to_numeric(sel[c], errors="coerce")

sel = sel.dropna(subset=["x1", "y1", "x2", "y2"]).copy()

if "roi_overlap_num" not in sel.columns:
    sel["roi_overlap_num"] = pd.to_numeric(
        sel.get("selected_roi_mask_overlap_fraction_v3", 0),
        errors="coerce",
    ).fillna(0.0)

sel["roi_overlap_num"] = pd.to_numeric(sel["roi_overlap_num"], errors="coerce").fillna(0.0)
sel["detector_score_num"] = pd.to_numeric(sel.get("detector_score_num", sel.get("score", 0.5)), errors="coerce").fillna(0.5)
sel["relative_y_rank"] = pd.to_numeric(sel.get("relative_y_rank", 0.5), errors="coerce").fillna(0.5)

# Confidence rules.
# We do NOT want to hide uncertainty. Selected detections with low ROI overlap or very low detector score are flagged.
sel["selected_for_gt_pen_matching_v4"] = sel["selected_for_gt_pen_matching_v4"].astype(bool)

sel["selected_confidence_score_v5"] = (
    0.55 * sel["roi_overlap_num"]
    + 0.30 * sel["detector_score_num"]
    + 0.15 * (1.0 - sel["relative_y_rank"])
)

clean_selected = (
    sel["selected_for_gt_pen_matching_v4"]
    & (sel["roi_overlap_num"] >= 0.55)
    & (sel["detector_score_num"] >= 0.35)
)

selected_needs_review = (
    sel["selected_for_gt_pen_matching_v4"]
    & (~clean_selected)
)

ignored_high_risk = (
    (~sel["selected_for_gt_pen_matching_v4"])
    & (
        (sel["roi_overlap_num"] >= 0.55)
        | (sel["selected_confidence_score_v5"] >= 0.55)
    )
)

sel["gt_pen_selection_status_v5"] = "ignored"

sel.loc[clean_selected, "gt_pen_selection_status_v5"] = "clean_selected"
sel.loc[selected_needs_review, "gt_pen_selection_status_v5"] = "selected_needs_review"
sel.loc[ignored_high_risk, "gt_pen_selection_status_v5"] = "ignored_high_risk"

sel["recommended_for_colour_matching_v5"] = sel["gt_pen_selection_status_v5"].isin([
    "clean_selected",
    "selected_needs_review",
])

sel["recommended_for_strict_colour_matching_v5"] = sel["gt_pen_selection_status_v5"].eq("clean_selected")

safe_to_csv(sel, OUT_SELECTION)

# Frame summary.
frame_rows = []

for sid, g in sel.groupby("scan_frame_id", sort=True):
    selected = int(g["selected_for_gt_pen_matching_v4"].sum())
    clean = int((g["gt_pen_selection_status_v5"] == "clean_selected").sum())
    review = int((g["gt_pen_selection_status_v5"] == "selected_needs_review").sum())
    ignored_risk = int((g["gt_pen_selection_status_v5"] == "ignored_high_risk").sum())

    frame_needs_review = (
        review > 0
        or ignored_risk > 0
        or selected < 6
        or clean < 5
    )

    frame_rows.append({
        "scan_frame_id": sid,
        "total_detections": len(g),
        "selected_count_v4": selected,
        "clean_selected_count_v5": clean,
        "selected_needs_review_count_v5": review,
        "ignored_high_risk_count_v5": ignored_risk,
        "frame_needs_review": frame_needs_review,
        "reason": "; ".join([
            "selected_less_than_6" if selected < 6 else "",
            "selected_needs_review_present" if review > 0 else "",
            "ignored_high_risk_present" if ignored_risk > 0 else "",
            "clean_selected_less_than_5" if clean < 5 else "",
        ]).strip("; "),
    })

frame_summary = pd.DataFrame(frame_rows)
safe_to_csv(frame_summary, OUT_FRAME_SUMMARY)

risky = sel[sel["gt_pen_selection_status_v5"].isin(["selected_needs_review", "ignored_high_risk"])].copy()
safe_to_csv(risky, OUT_RISKY_DETECTIONS)

# Review images.
review_index = make_frame_review_images(frames, sel)
review_index = review_index.merge(
    frame_summary[["scan_frame_id", "frame_needs_review"]],
    on="scan_frame_id",
    how="left",
)
safe_to_csv(review_index, OUT_REVIEW_TEMPLATE)

contact_all_ok = make_contact_sheet(review_index, OUT_CONTACT_ALL, only_risky=False)
contact_risky_ok = make_contact_sheet(review_index, OUT_CONTACT_RISKY, only_risky=True)
make_html(review_index, OUT_HTML)

# Update task tracker.
if TASK_TRACKER_PATH.exists():
    tracker = pd.read_csv(TASK_TRACKER_PATH)
    mask = tracker["stage"].astype(str) == "2"

    if mask.any():
        tracker.loc[mask, "status"] = "gt_pen_selection_v5_review_package_generated"

    safe_to_csv(tracker, TASK_TRACKER_PATH)

# Notes.
total = len(sel)
clean_total = int((sel["gt_pen_selection_status_v5"] == "clean_selected").sum())
review_total = int((sel["gt_pen_selection_status_v5"] == "selected_needs_review").sum())
ignored_total = int((sel["gt_pen_selection_status_v5"] == "ignored").sum())
ignored_risk_total = int((sel["gt_pen_selection_status_v5"] == "ignored_high_risk").sum())
frames_needing_review = int(frame_summary["frame_needs_review"].sum())

with open(OUT_NOTE, "w") as f:
    f.write("# Week 7 GT Pen Selection v5 Confidence Review Package\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This step does not blindly finalize the v4 automatic selection. "
        "Instead, it separates clean selected detections from selected detections that need review and ignored detections that remain suspicious.\n\n"
    )

    f.write("## Summary\n\n")
    f.write(f"- Total detections: `{total}`\n")
    f.write(f"- Clean selected detections: `{clean_total}`\n")
    f.write(f"- Selected detections needing review: `{review_total}`\n")
    f.write(f"- Ignored detections: `{ignored_total}`\n")
    f.write(f"- Ignored high-risk detections: `{ignored_risk_total}`\n")
    f.write(f"- Frames needing review: `{frames_needing_review}` out of `{frame_summary['scan_frame_id'].nunique()}`\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "For strict colour matching, use only `clean_selected` detections. "
        "For exploratory matching, `selected_needs_review` can be included but should be visually checked. "
        "Wrong-pen detections should not be silently used for behaviour-label association.\n\n"
    )

    f.write("## Visual review files\n\n")
    f.write(f"- All review contact sheet: `{OUT_CONTACT_ALL}`\n")
    f.write(f"- Risky frames contact sheet: `{OUT_CONTACT_RISKY}`\n")
    f.write(f"- Static review HTML: `{OUT_HTML}`\n")
    f.write(f"- Individual frame review directory: `{REVIEW_DIR}`\n\n")

    f.write("## Outputs\n\n")
    for p in [OUT_SELECTION, OUT_FRAME_SUMMARY, OUT_RISKY_DETECTIONS, OUT_REVIEW_TEMPLATE, OUT_CONTACT_ALL, OUT_CONTACT_RISKY, OUT_HTML]:
        f.write(f"- `{p}`\n")

print("Saved:")
print(OUT_SELECTION)
print(OUT_FRAME_SUMMARY)
print(OUT_RISKY_DETECTIONS)
print(OUT_REVIEW_TEMPLATE)
print(OUT_CONTACT_ALL)
print(OUT_CONTACT_RISKY)
print(OUT_HTML)
print(OUT_NOTE)

print()
print("=== v5 summary ===")
print({
    "total_detections": total,
    "clean_selected": clean_total,
    "selected_needs_review": review_total,
    "ignored": ignored_total,
    "ignored_high_risk": ignored_risk_total,
    "frames_needing_review": frames_needing_review,
    "contact_all_ok": contact_all_ok,
    "contact_risky_ok": contact_risky_ok,
})
