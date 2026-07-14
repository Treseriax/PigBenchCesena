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

V6_ALL_PATH = W7 / "outputs" / "roi_and_crate_mapping" / "week7_gt_pen_detection_selection_v6_final_safe_layer.csv"
V6_FRAME_SUMMARY_PATH = W7 / "outputs" / "roi_and_crate_mapping" / "week7_gt_pen_detection_selection_v6_frame_summary.csv"

LOW_DET_PATH = W7 / "outputs" / "roi_and_crate_mapping" / "week7_detector_recall_expansion_low_threshold_detections.csv"

OUT_ROI = W7 / "outputs" / "roi_and_crate_mapping"
OUT_VIS = W7 / "outputs" / "visualizations"
OUT_STATS = W7 / "outputs" / "dataset_statistics"
OUT_NOTES = W7 / "notes"

for p in [OUT_ROI, OUT_VIS, OUT_STATS, OUT_NOTES]:
    p.mkdir(parents=True, exist_ok=True)

REVIEW_DIR = OUT_VIS / "gt_pen_candidate_pool_v7_review_frames"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)

OUT_POOL = OUT_ROI / "week7_gt_pen_candidate_pool_v7_corrected.csv"
OUT_STRICT = OUT_ROI / "week7_gt_pen_candidate_pool_v7_strict_auto_safe.csv"
OUT_REVIEW = OUT_ROI / "week7_gt_pen_candidate_pool_v7_manual_review_required.csv"
OUT_NEW_PRIORITY = OUT_ROI / "week7_gt_pen_candidate_pool_v7_priority_new_recall_candidates.csv"
OUT_FRAME_SUMMARY = OUT_ROI / "week7_gt_pen_candidate_pool_v7_frame_summary.csv"
OUT_MANUAL_TEMPLATE = OUT_ROI / "week7_gt_pen_candidate_pool_v7_manual_override_template.csv"

OUT_CONTACT_ALL = OUT_VIS / "week7_gt_pen_candidate_pool_v7_all_frames_contact_sheet.jpg"
OUT_CONTACT_REVIEW = OUT_VIS / "week7_gt_pen_candidate_pool_v7_review_required_contact_sheet.jpg"
OUT_HTML = OUT_VIS / "week7_gt_pen_candidate_pool_v7_static_review.html"

OUT_NOTE = OUT_NOTES / "week7_gt_pen_candidate_pool_v7_corrected_notes.md"

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


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))

    return inter / float(area_a + area_b - inter)


def dedupe_new_against_pool(new_df, existing_df, iou_threshold=0.50):
    keep_rows = []

    for _, n in new_df.iterrows():
        sid = str(n["scan_frame_id"])
        nbox = [float(n["x1"]), float(n["y1"]), float(n["x2"]), float(n["y2"])]

        existing_frame = existing_df[existing_df["scan_frame_id"].astype(str) == sid]

        best_iou = 0.0

        for _, e in existing_frame.iterrows():
            ebox = [float(e["x1"]), float(e["y1"]), float(e["x2"]), float(e["y2"])]
            best_iou = max(best_iou, bbox_iou(nbox, ebox))

        row = n.to_dict()
        row["max_iou_with_v6_pool"] = best_iou
        row["keep_as_new_v7_candidate"] = best_iou < iou_threshold
        keep_rows.append(row)

    return pd.DataFrame(keep_rows)


def make_review_images(frames, pool):
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
        g = pool[pool["scan_frame_id"].astype(str) == sid].copy()

        # Draw ignored faint first.
        draw_order = {
            "existing_ignored": 0,
            "existing_ignored_high_risk_review": 1,
            "existing_selected_review_required": 2,
            "new_low_threshold_recall_candidate": 3,
            "strict_auto_safe": 4,
        }

        g["draw_order"] = g["v7_candidate_class"].map(draw_order).fillna(0)
        g = g.sort_values("draw_order")

        for _, d in g.iterrows():
            x1 = int(round(float(d["x1"])))
            y1 = int(round(float(d["y1"])))
            x2 = int(round(float(d["x2"])))
            y2 = int(round(float(d["y2"])))

            cls = str(d["v7_candidate_class"])

            if cls == "strict_auto_safe":
                color = (0, 220, 0)
                thickness = 3
                prefix = "SAFE"
            elif cls == "existing_selected_review_required":
                color = (0, 165, 255)
                thickness = 2
                prefix = "REV"
            elif cls == "existing_ignored_high_risk_review":
                color = (0, 0, 255)
                thickness = 2
                prefix = "RISK"
            elif cls == "new_low_threshold_recall_candidate":
                color = (255, 180, 0)
                thickness = 2
                prefix = "NEW"
            else:
                color = (130, 130, 130)
                thickness = 1
                prefix = "IGN"

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

            if cls != "existing_ignored":
                score = float(d.get("candidate_score", 0))
                cid = str(d.get("v7_candidate_id", ""))[-6:]
                label = f"{prefix} {cid} {score:.2f}"

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

        strict_count = int((g["v7_candidate_class"] == "strict_auto_safe").sum())
        new_count = int((g["v7_candidate_class"] == "new_low_threshold_recall_candidate").sum())
        review_count = int(g["v7_manual_review_required"].sum())

        title = f"{sid} | strict={strict_count} | new={new_count} | review={review_count}"

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

        out_path = REVIEW_DIR / f"{sid}_candidate_pool_v7_review.jpg"
        cv2.imwrite(str(out_path), overlay)

        rows.append({
            "scan_frame_id": sid,
            "review_image_path": str(out_path),
            "strict_auto_safe_count_v7": strict_count,
            "new_recall_candidate_count_v7": new_count,
            "manual_review_required_count_v7": review_count,
            "frame_needs_manual_review_v7": bool(review_count > 0 or strict_count < 6),
        })

    return pd.DataFrame(rows)


def make_contact_sheet(review_index, output_path, only_review=False):
    df = review_index.copy()

    if only_review:
        df = df[df["frame_needs_manual_review_v7"] == True].copy()

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
        src = "gt_pen_candidate_pool_v7_review_frames/" + img_path.name

        cls = "review" if bool(r["frame_needs_manual_review_v7"]) else "safe"

        cards.append(
            f"""
            <div class="frame-card {cls}">
              <h3>{r['scan_frame_id']}</h3>
              <p>
                strict_auto_safe={r['strict_auto_safe_count_v7']} |
                new_recall_candidates={r['new_recall_candidate_count_v7']} |
                manual_review_required={r['manual_review_required_count_v7']} |
                needs_review={r['frame_needs_manual_review_v7']}
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
  <title>Week 7 GT Pen Candidate Pool v7 Review</title>
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
    .green {{ color: green; font-weight: bold; }}
    .orange {{ color: darkorange; font-weight: bold; }}
    .red {{ color: red; font-weight: bold; }}
    .blue {{ color: royalblue; font-weight: bold; }}
  </style>
</head>
<body>
  <h1>Week 7 Ground Truth Pen Candidate Pool v7 Review</h1>

  <div class="legend">
    <p><span class="green">SAFE</span>: strict automatic safe detection from v6.</p>
    <p><span class="orange">REV</span>: existing selected detection requiring manual review.</p>
    <p><span class="red">RISK</span>: ignored high-risk detection requiring review.</p>
    <p><span class="blue">NEW</span>: low-threshold recall candidate. Never accept automatically.</p>
    <p><b>Rule:</b> final colour matching can use SAFE automatically. REV/RISK/NEW require manual confirmation.</p>
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
v6 = pd.read_csv(V6_ALL_PATH)
v6_frame = pd.read_csv(V6_FRAME_SUMMARY_PATH)
low = pd.read_csv(LOW_DET_PATH)

for df in [v6, low]:
    for c in ["x1", "y1", "x2", "y2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

v6 = v6.dropna(subset=["x1", "y1", "x2", "y2"]).copy()
low = low.dropna(subset=["x1", "y1", "x2", "y2"]).copy()

v6["strict_auto_safe_for_colour_matching_v6"] = to_bool_series(v6["strict_auto_safe_for_colour_matching_v6"])
v6["v6_manual_review_required"] = to_bool_series(v6["v6_manual_review_required"])

# ---------------------------------------------------------------------
# Existing v6 rows into v7 pool.
# ---------------------------------------------------------------------
existing_rows = []

for _, r in v6.iterrows():
    cls = str(r.get("v6_selection_class", "ignored"))

    if bool(r["strict_auto_safe_for_colour_matching_v6"]):
        v7_class = "strict_auto_safe"
        review_required = False
        default_action = "auto_keep_for_colour_matching"
    elif cls == "selected_review_required":
        v7_class = "existing_selected_review_required"
        review_required = True
        default_action = "manual_keep_or_reject"
    elif cls == "ignored_high_risk_review":
        v7_class = "existing_ignored_high_risk_review"
        review_required = True
        default_action = "manual_add_or_ignore"
    else:
        v7_class = "existing_ignored"
        review_required = False
        default_action = "auto_ignore"

    candidate_id = f"{str(r['scan_frame_id'])}_old_det_{str(r.get('det_id', r.name))}"

    existing_rows.append({
        "scan_frame_id": r["scan_frame_id"],
        "v7_candidate_id": candidate_id,
        "candidate_source": "existing_week6_detection_v6",
        "source_det_id": r.get("det_id", ""),
        "source_pred_id": "",
        "x1": r["x1"],
        "y1": r["y1"],
        "x2": r["x2"],
        "y2": r["y2"],
        "candidate_score": float(r.get("v6_final_safe_confidence", r.get("selected_confidence_score_v5", 0))),
        "detector_score": float(r.get("detector_score_num", r.get("score", 0))),
        "v7_candidate_class": v7_class,
        "v7_manual_review_required": review_required,
        "v7_default_action": default_action,
        "notes": "",
    })

existing_pool = pd.DataFrame(existing_rows)

# ---------------------------------------------------------------------
# Low-threshold new candidates.
# ---------------------------------------------------------------------
low["score"] = pd.to_numeric(low["score"], errors="coerce").fillna(0.0)

if "is_new_candidate_vs_week6" in low.columns:
    low["is_new_candidate_vs_week6"] = to_bool_series(low["is_new_candidate_vs_week6"])
else:
    low["is_new_candidate_vs_week6"] = True

# Include new candidates from score >= 0.10 for manual pool.
new_raw = low[
    (low["is_new_candidate_vs_week6"] == True)
    & (low["score"] >= 0.10)
].copy()

new_deduped = dedupe_new_against_pool(new_raw, existing_pool, iou_threshold=0.50)
new_deduped = new_deduped[new_deduped["keep_as_new_v7_candidate"] == True].copy()

# Add v6 frame context.
v6_frame_small = v6_frame[[
    "scan_frame_id",
    "strict_auto_safe_count_v6",
    "frame_needs_manual_review_v6",
]].copy()

new_deduped = new_deduped.merge(v6_frame_small, on="scan_frame_id", how="left")

new_deduped["strict_auto_safe_count_v6"] = pd.to_numeric(
    new_deduped["strict_auto_safe_count_v6"],
    errors="coerce",
).fillna(0).astype(int)

# Priority new candidates:
# - Score >= 0.15 and frame has fewer than six strict safe detections, or
# - Score >= 0.20 regardless, because it is a stronger low-th candidate.
new_deduped["priority_new_recall_candidate_v7"] = (
    (
        (new_deduped["score"] >= 0.15)
        & (new_deduped["strict_auto_safe_count_v6"] < 6)
    )
    | (new_deduped["score"] >= 0.20)
)

new_rows = []

for _, r in new_deduped.iterrows():
    candidate_id = f"{str(r['scan_frame_id'])}_new_pred_{str(r.get('pred_id', r.name))}"

    # New candidates are never strict safe.
    # They are only recall candidates for manual check.
    new_rows.append({
        "scan_frame_id": r["scan_frame_id"],
        "v7_candidate_id": candidate_id,
        "candidate_source": "low_threshold_detector_new_candidate",
        "source_det_id": "",
        "source_pred_id": r.get("pred_id", ""),
        "x1": r["x1"],
        "y1": r["y1"],
        "x2": r["x2"],
        "y2": r["y2"],
        "candidate_score": float(r["score"]),
        "detector_score": float(r["score"]),
        "v7_candidate_class": "new_low_threshold_recall_candidate",
        "v7_manual_review_required": True,
        "v7_default_action": "manual_add_or_ignore_new_recall_candidate",
        "priority_new_recall_candidate_v7": bool(r["priority_new_recall_candidate_v7"]),
        "max_iou_with_v6_pool": float(r.get("max_iou_with_v6_pool", 0)),
        "notes": "New low-threshold candidate. Must be visually checked before use.",
    })

new_pool = pd.DataFrame(new_rows)

# Ensure columns exist on existing pool.
existing_pool["priority_new_recall_candidate_v7"] = False
existing_pool["max_iou_with_v6_pool"] = ""

pool = pd.concat([existing_pool, new_pool], ignore_index=True, sort=False)

safe_to_csv(pool, OUT_POOL)

strict_df = pool[pool["v7_candidate_class"] == "strict_auto_safe"].copy()
review_df = pool[pool["v7_manual_review_required"] == True].copy()
priority_new_df = pool[
    (pool["v7_candidate_class"] == "new_low_threshold_recall_candidate")
    & (pool["priority_new_recall_candidate_v7"] == True)
].copy()

safe_to_csv(strict_df, OUT_STRICT)
safe_to_csv(review_df, OUT_REVIEW)
safe_to_csv(priority_new_df, OUT_NEW_PRIORITY)

# ---------------------------------------------------------------------
# Frame summary.
# ---------------------------------------------------------------------
frame_rows = []

for sid, g in pool.groupby("scan_frame_id", sort=True):
    strict_count = int((g["v7_candidate_class"] == "strict_auto_safe").sum())
    existing_review_count = int((g["v7_candidate_class"] == "existing_selected_review_required").sum())
    risk_count = int((g["v7_candidate_class"] == "existing_ignored_high_risk_review").sum())
    new_count = int((g["v7_candidate_class"] == "new_low_threshold_recall_candidate").sum())
    priority_new_count = int(
        (
            (g["v7_candidate_class"] == "new_low_threshold_recall_candidate")
            & (g["priority_new_recall_candidate_v7"] == True)
        ).sum()
    )

    manual_review_count = int(g["v7_manual_review_required"].sum())

    missing_strict_slots = max(0, 6 - strict_count)
    missing_after_priority_new = max(0, 6 - (strict_count + priority_new_count))

    needs_review = (
        manual_review_count > 0
        or strict_count < 6
        or priority_new_count > 0
    )

    frame_rows.append({
        "scan_frame_id": sid,
        "strict_auto_safe_count_v7": strict_count,
        "existing_selected_review_required_count_v7": existing_review_count,
        "existing_ignored_high_risk_review_count_v7": risk_count,
        "new_low_threshold_recall_candidate_count_v7": new_count,
        "priority_new_recall_candidate_count_v7": priority_new_count,
        "manual_review_required_count_v7": manual_review_count,
        "missing_slots_after_strict_auto_safe": missing_strict_slots,
        "missing_slots_after_strict_plus_priority_new": missing_after_priority_new,
        "frame_needs_manual_review_v7": needs_review,
    })

frame_summary = pd.DataFrame(frame_rows)
safe_to_csv(frame_summary, OUT_FRAME_SUMMARY)

# Manual override template.
review_frame_ids = set(frame_summary.loc[frame_summary["frame_needs_manual_review_v7"], "scan_frame_id"].astype(str))

manual_template = pool[pool["scan_frame_id"].astype(str).isin(review_frame_ids)].copy()

manual_template["manual_final_use_for_colour_matching"] = manual_template["v7_candidate_class"].eq("strict_auto_safe")
manual_template["manual_override_action"] = manual_template["v7_default_action"]
manual_template["manual_reviewer_notes"] = ""

safe_to_csv(manual_template, OUT_MANUAL_TEMPLATE)

# Visuals.
review_index = make_review_images(frames, pool)
safe_to_csv(review_index, OUT_VIS / "week7_gt_pen_candidate_pool_v7_review_image_index.csv")

contact_all_ok = make_contact_sheet(review_index, OUT_CONTACT_ALL, only_review=False)
contact_review_ok = make_contact_sheet(review_index, OUT_CONTACT_REVIEW, only_review=True)
make_html(review_index, OUT_HTML)

# Update tracker.
if TASK_TRACKER_PATH.exists():
    tracker = pd.read_csv(TASK_TRACKER_PATH)
    mask = tracker["stage"].astype(str) == "2"

    if mask.any():
        tracker.loc[mask, "status"] = "gt_pen_candidate_pool_v7_corrected_generated"

    safe_to_csv(tracker, TASK_TRACKER_PATH)

# Notes.
total_pool = len(pool)
strict_total = len(strict_df)
review_total = len(review_df)
new_total = int((pool["v7_candidate_class"] == "new_low_threshold_recall_candidate").sum())
priority_new_total = len(priority_new_df)
frames_review = int(frame_summary["frame_needs_manual_review_v7"].sum())
frames_with_missing_after_priority = int((frame_summary["missing_slots_after_strict_plus_priority_new"] > 0).sum())

with open(OUT_NOTE, "w") as f:
    f.write("# Week 7 GT Pen Candidate Pool v7 Corrected\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This step merges v6 strict/review detections with low-threshold detector recall candidates. "
        "New low-threshold detections are not accepted automatically; they are added as manual recall candidates only.\n\n"
    )

    f.write("## Key rule\n\n")
    f.write(
        "Only `v7_candidate_class = strict_auto_safe` can be used automatically for colour matching. "
        "`existing_selected_review_required`, `existing_ignored_high_risk_review`, and `new_low_threshold_recall_candidate` require manual confirmation.\n\n"
    )

    f.write("## Summary\n\n")
    f.write(f"- Total candidate pool rows: `{total_pool}`\n")
    f.write(f"- Strict automatic safe detections: `{strict_total}`\n")
    f.write(f"- Manual review required candidates: `{review_total}`\n")
    f.write(f"- New low-threshold recall candidates included: `{new_total}`\n")
    f.write(f"- Priority new recall candidates: `{priority_new_total}`\n")
    f.write(f"- Frames requiring manual review: `{frames_review}` out of `{frame_summary['scan_frame_id'].nunique()}`\n")
    f.write(f"- Frames still missing slots after strict + priority new candidates: `{frames_with_missing_after_priority}`\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "Low-threshold detection expands recall but introduces false positives. "
        "Therefore v7 is a corrected candidate pool, not a final automatic label-matching table. "
        "The next step is manual confirmation of review candidates or generation of missing manual boxes where the detector still fails.\n\n"
    )

    f.write("## Outputs\n\n")
    for p in [
        OUT_POOL,
        OUT_STRICT,
        OUT_REVIEW,
        OUT_NEW_PRIORITY,
        OUT_FRAME_SUMMARY,
        OUT_MANUAL_TEMPLATE,
        OUT_CONTACT_ALL,
        OUT_CONTACT_REVIEW,
        OUT_HTML,
    ]:
        f.write(f"- `{p}`\n")

print("Saved:")
print(OUT_POOL)
print(OUT_STRICT)
print(OUT_REVIEW)
print(OUT_NEW_PRIORITY)
print(OUT_FRAME_SUMMARY)
print(OUT_MANUAL_TEMPLATE)
print(OUT_CONTACT_ALL)
print(OUT_CONTACT_REVIEW)
print(OUT_HTML)
print(OUT_NOTE)

print()
print("=== v7 summary ===")
print({
    "total_candidate_pool_rows": total_pool,
    "strict_auto_safe": strict_total,
    "manual_review_required": review_total,
    "new_low_threshold_candidates": new_total,
    "priority_new_recall_candidates": priority_new_total,
    "frames_requiring_manual_review": frames_review,
    "frames_missing_after_strict_plus_priority_new": frames_with_missing_after_priority,
    "contact_all_ok": contact_all_ok,
    "contact_review_ok": contact_review_ok,
})
