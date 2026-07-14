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

V7_POOL_PATH = W7 / "outputs" / "roi_and_crate_mapping" / "week7_gt_pen_candidate_pool_v7_corrected.csv"
V7_FRAME_SUMMARY_PATH = W7 / "outputs" / "roi_and_crate_mapping" / "week7_gt_pen_candidate_pool_v7_frame_summary.csv"

OUT_ROI = W7 / "outputs" / "roi_and_crate_mapping"
OUT_VIS = W7 / "outputs" / "visualizations"
OUT_STATS = W7 / "outputs" / "dataset_statistics"
OUT_NOTES = W7 / "notes"

for p in [OUT_ROI, OUT_VIS, OUT_STATS, OUT_NOTES]:
    p.mkdir(parents=True, exist_ok=True)

REVIEW_DIR = OUT_VIS / "manual_correction_v8_review_frames"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)

OUT_DECISION_TEMPLATE = OUT_ROI / "week7_manual_correction_v8_candidate_decisions_template.csv"
OUT_MANUAL_ADD_TEMPLATE = OUT_ROI / "week7_manual_correction_v8_manual_add_boxes_template.csv"
OUT_FRAME_CHECKLIST = OUT_ROI / "week7_manual_correction_v8_frame_checklist.csv"
OUT_FINAL_EMPTY = OUT_ROI / "week7_manual_correction_v8_final_corrected_boxes_EMPTY_TEMPLATE.csv"

OUT_REVIEW_INDEX = OUT_VIS / "week7_manual_correction_v8_review_image_index.csv"
OUT_CONTACT_ALL = OUT_VIS / "week7_manual_correction_v8_all_frames_contact_sheet.jpg"
OUT_CONTACT_PRIORITY = OUT_VIS / "week7_manual_correction_v8_priority_review_contact_sheet.jpg"
OUT_HTML = OUT_VIS / "week7_manual_correction_v8_static_review.html"

OUT_NOTE = OUT_NOTES / "week7_manual_correction_v8_workspace_notes.md"

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


def short_id(value):
    value = str(value)
    value = value.replace("scanframe_", "sf")
    value = value.replace("_old_det_", "_od")
    value = value.replace("_new_pred_", "_np")
    return value[-18:]


def draw_review_images(frames, pool):
    image_col = first_col(frames, ["frame_image_path", "image_path", "frame_path", "scanpoint_frame_path"])

    if image_col is None:
        raise RuntimeError("No frame image path column found.")

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

        draw_order_map = {
            "existing_ignored": 0,
            "existing_ignored_high_risk_review": 1,
            "existing_selected_review_required": 2,
            "new_low_threshold_recall_candidate": 3,
            "strict_auto_safe": 4,
        }

        g["draw_order"] = g["v7_candidate_class"].map(draw_order_map).fillna(0)
        g = g.sort_values(["draw_order", "candidate_score"], ascending=[True, True])

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
                color = (120, 120, 120)
                thickness = 1
                prefix = "IGN"

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

            if cls != "existing_ignored":
                sid_short = short_id(d["v7_candidate_id"])
                score = float(d.get("candidate_score", 0.0))

                label = f"{prefix}:{sid_short}:{score:.2f}"

                cv2.putText(
                    overlay,
                    label,
                    (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        strict_count = int((g["v7_candidate_class"] == "strict_auto_safe").sum())
        review_count = int((g["v7_manual_review_required"] == True).sum())
        new_count = int((g["v7_candidate_class"] == "new_low_threshold_recall_candidate").sum())
        priority_new_count = int(
            (
                (g["v7_candidate_class"] == "new_low_threshold_recall_candidate")
                & (g["priority_new_recall_candidate_v7"] == True)
            ).sum()
        )

        title = (
            f"{sid} | SAFE={strict_count} | REVIEW={review_count} | "
            f"NEW={new_count} | PRIORITY_NEW={priority_new_count}"
        )

        cv2.putText(
            overlay,
            title,
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out_path = REVIEW_DIR / f"{sid}_manual_correction_v8_review.jpg"
        cv2.imwrite(str(out_path), overlay)

        rows.append({
            "scan_frame_id": sid,
            "review_image_path": str(out_path),
            "strict_auto_safe_count": strict_count,
            "manual_review_candidate_count": review_count,
            "new_candidate_count": new_count,
            "priority_new_candidate_count": priority_new_count,
            "frame_priority_review": bool(strict_count < 6 or review_count > 0 or priority_new_count > 0),
        })

    return pd.DataFrame(rows)


def make_contact_sheet(review_index, output_path, only_priority=False):
    df = review_index.copy()

    if only_priority:
        df = df[df["frame_priority_review"] == True].copy()

    if len(df) == 0:
        return False

    if len(df) > 16:
        idx = np.linspace(0, len(df) - 1, 16).round().astype(int)
        df = df.iloc[idx]

    thumbs = []
    thumb_w, thumb_h = 380, 250

    for _, r in df.iterrows():
        p = Path(str(r["review_image_path"]))

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
        src = "manual_correction_v8_review_frames/" + img_path.name

        cls = "priority" if bool(r["frame_priority_review"]) else "ok"

        cards.append(
            f"""
            <div class="frame-card {cls}">
              <h3>{r['scan_frame_id']}</h3>
              <p>
                SAFE={r['strict_auto_safe_count']} |
                REVIEW={r['manual_review_candidate_count']} |
                NEW={r['new_candidate_count']} |
                PRIORITY_NEW={r['priority_new_candidate_count']} |
                priority_review={r['frame_priority_review']}
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
  <title>Week 7 Manual Correction Workspace v8</title>
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
    .frame-card.priority {{
      border-left: 8px solid darkorange;
    }}
    .frame-card.ok {{
      border-left: 8px solid green;
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
    code {{
      background: #eee;
      padding: 2px 4px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <h1>Week 7 Manual Correction Workspace v8</h1>

  <div class="legend">
    <p><span class="green">SAFE</span>: auto-safe candidate, still reject manually if it is outside the GT pen.</p>
    <p><span class="orange">REV</span>: existing candidate requiring manual keep/reject.</p>
    <p><span class="red">RISK</span>: ignored high-risk candidate; can be manually added only if it is actually a GT-pen pig.</p>
    <p><span class="blue">NEW</span>: low-threshold recall candidate; never automatic, manual add only.</p>
    <p><b>Manual decision values:</b> <code>keep</code>, <code>reject_wrong_pen</code>, <code>reject_duplicate</code>, <code>reject_false_positive</code>, <code>uncertain</code>.</p>
    <p>If the true GT-pen pig is missing entirely, fill the manual add boxes template with x1, y1, x2, y2.</p>
  </div>

  {''.join(cards)}
</body>
</html>
"""
    out_path.write_text(html)


# ---------------------------------------------------------------------
# Load data.
# ---------------------------------------------------------------------
frames = pd.read_csv(FRAME_INDEX_PATH)
pool = pd.read_csv(V7_POOL_PATH)
frame_summary_v7 = pd.read_csv(V7_FRAME_SUMMARY_PATH)

for c in ["x1", "y1", "x2", "y2", "candidate_score", "detector_score"]:
    if c in pool.columns:
        pool[c] = pd.to_numeric(pool[c], errors="coerce")

pool = pool.dropna(subset=["x1", "y1", "x2", "y2"]).copy()

pool["v7_manual_review_required"] = to_bool_series(pool["v7_manual_review_required"])

if "priority_new_recall_candidate_v7" in pool.columns:
    pool["priority_new_recall_candidate_v7"] = to_bool_series(pool["priority_new_recall_candidate_v7"])
else:
    pool["priority_new_recall_candidate_v7"] = False

# ---------------------------------------------------------------------
# Candidate decision template.
# ---------------------------------------------------------------------
decision = pool.copy()

decision["candidate_short_id"] = decision["v7_candidate_id"].apply(short_id)

decision["manual_decision"] = np.where(
    decision["v7_candidate_class"].eq("strict_auto_safe"),
    "keep_if_visually_inside_gt_pen",
    "pending_review",
)

decision["allowed_manual_decisions"] = "keep | reject_wrong_pen | reject_duplicate | reject_false_positive | uncertain"
decision["final_use_for_colour_matching_after_manual_review"] = decision["v7_candidate_class"].eq("strict_auto_safe")
decision["manual_reviewer_notes"] = ""

# Sort most important first within each frame.
class_rank = {
    "strict_auto_safe": 0,
    "existing_selected_review_required": 1,
    "new_low_threshold_recall_candidate": 2,
    "existing_ignored_high_risk_review": 3,
    "existing_ignored": 4,
}

decision["class_rank"] = decision["v7_candidate_class"].map(class_rank).fillna(99)
decision = decision.sort_values(["scan_frame_id", "class_rank", "candidate_score"], ascending=[True, True, False])

safe_to_csv(decision, OUT_DECISION_TEMPLATE)

# ---------------------------------------------------------------------
# Manual add boxes template.
# ---------------------------------------------------------------------
add_rows = []

for _, r in frame_summary_v7.sort_values("scan_frame_id").iterrows():
    sid = str(r["scan_frame_id"])

    missing_after_strict = int(r.get("missing_slots_after_strict_auto_safe", 0))
    missing_after_priority = int(r.get("missing_slots_after_strict_plus_priority_new", 0))

    # At least one optional row for frames with any review;
    # more rows if missing slots remain.
    add_slot_count = max(1 if bool(str(r.get("frame_needs_manual_review_v7", "")).lower() == "true") else 0, missing_after_priority)

    for i in range(add_slot_count):
        add_rows.append({
            "scan_frame_id": sid,
            "manual_add_box_id": f"{sid}_manual_add_{i+1}",
            "x1": "",
            "y1": "",
            "x2": "",
            "y2": "",
            "manual_add_reason": "missing_true_gt_pen_pig_not_detected",
            "final_use_for_colour_matching_after_manual_review": "",
            "manual_reviewer_notes": "",
        })

manual_add = pd.DataFrame(add_rows)
safe_to_csv(manual_add, OUT_MANUAL_ADD_TEMPLATE)

# ---------------------------------------------------------------------
# Frame checklist.
# ---------------------------------------------------------------------
checklist = frame_summary_v7.copy()

checklist["review_priority_level"] = np.where(
    checklist["missing_slots_after_strict_plus_priority_new"] > 0,
    "high_missing_boxes",
    np.where(
        checklist["manual_review_required_count_v7"] > 0,
        "medium_candidate_decisions",
        "low_clean_frame",
    ),
)

checklist["manual_review_instruction"] = (
    "1) Keep only GT-pen pigs. "
    "2) Reject wrong-pen/duplicate/false-positive candidates. "
    "3) Add manual bbox if a GT-pen pig is missing. "
    "4) Aim for six final pigs per scanpoint when visible."
)

safe_to_csv(checklist, OUT_FRAME_CHECKLIST)

# Empty final corrected boxes template.
final_empty_cols = [
    "scan_frame_id",
    "final_box_id",
    "source_candidate_id",
    "source_type",
    "x1",
    "y1",
    "x2",
    "y2",
    "final_use_for_colour_matching",
    "final_correction_status",
    "manual_reviewer_notes",
]

safe_to_csv(pd.DataFrame(columns=final_empty_cols), OUT_FINAL_EMPTY)

# Visuals.
review_index = draw_review_images(frames, pool)
safe_to_csv(review_index, OUT_REVIEW_INDEX)

contact_all_ok = make_contact_sheet(review_index, OUT_CONTACT_ALL, only_priority=False)
contact_priority_ok = make_contact_sheet(review_index, OUT_CONTACT_PRIORITY, only_priority=True)
make_html(review_index, OUT_HTML)

# Update task tracker.
if TASK_TRACKER_PATH.exists():
    tracker = pd.read_csv(TASK_TRACKER_PATH)

    mask = tracker["stage"].astype(str) == "2"

    if mask.any():
        tracker.loc[mask, "status"] = "manual_correction_workspace_v8_generated"

    safe_to_csv(tracker, TASK_TRACKER_PATH)

# Notes.
total_candidates = len(decision)
manual_review_candidates = int(decision["v7_manual_review_required"].sum())
strict_candidates = int(decision["v7_candidate_class"].eq("strict_auto_safe").sum())
new_candidates = int(decision["v7_candidate_class"].eq("new_low_threshold_recall_candidate").sum())
manual_add_rows = len(manual_add)
high_priority_frames = int(checklist["review_priority_level"].eq("high_missing_boxes").sum())

with open(OUT_NOTE, "w") as f:
    f.write("# Week 7 Manual Correction Workspace v8\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "The automatic ROI/detection pipeline was not reliable enough to finalize colour/behaviour matching. "
        "Some wrong-pen pigs were still marked as safe, while some correct GT-pen pigs were missed by the detector. "
        "Therefore, this workspace supports a human-in-the-loop correction step.\n\n"
    )

    f.write("## Correction rule\n\n")
    f.write(
        "Final colour matching must use the manually corrected GT-pen box set, not raw detector output. "
        "A candidate is kept only if visual review confirms that it belongs to the annotated GT pen. "
        "If a true GT-pen pig is missing, a manual bbox should be added.\n\n"
    )

    f.write("## Summary\n\n")
    f.write(f"- Candidate decision rows: `{total_candidates}`\n")
    f.write(f"- Strict auto-safe candidates to verify: `{strict_candidates}`\n")
    f.write(f"- Manual review candidates: `{manual_review_candidates}`\n")
    f.write(f"- New low-threshold recall candidates: `{new_candidates}`\n")
    f.write(f"- Manual add rows prepared: `{manual_add_rows}`\n")
    f.write(f"- High-priority frames with missing slots: `{high_priority_frames}`\n\n")

    f.write("## Manual workflow\n\n")
    f.write("1. Open the static HTML review page.\n")
    f.write("2. For each frame, inspect whether each candidate belongs to the annotated GT pen.\n")
    f.write("3. Fill `week7_manual_correction_v8_candidate_decisions_template.csv`.\n")
    f.write("4. If a correct GT-pen pig is missing, fill `week7_manual_correction_v8_manual_add_boxes_template.csv`.\n")
    f.write("5. After manual edits, run the next script to compile final corrected boxes.\n\n")

    f.write("## Outputs\n\n")
    for p in [
        OUT_DECISION_TEMPLATE,
        OUT_MANUAL_ADD_TEMPLATE,
        OUT_FRAME_CHECKLIST,
        OUT_FINAL_EMPTY,
        OUT_REVIEW_INDEX,
        OUT_CONTACT_ALL,
        OUT_CONTACT_PRIORITY,
        OUT_HTML,
    ]:
        f.write(f"- `{p}`\n")

print("Saved:")
print(OUT_DECISION_TEMPLATE)
print(OUT_MANUAL_ADD_TEMPLATE)
print(OUT_FRAME_CHECKLIST)
print(OUT_FINAL_EMPTY)
print(OUT_REVIEW_INDEX)
print(OUT_CONTACT_ALL)
print(OUT_CONTACT_PRIORITY)
print(OUT_HTML)
print(OUT_NOTE)

print()
print("=== v8 workspace summary ===")
print({
    "candidate_decision_rows": total_candidates,
    "strict_candidates_to_verify": strict_candidates,
    "manual_review_candidates": manual_review_candidates,
    "new_low_threshold_recall_candidates": new_candidates,
    "manual_add_rows_prepared": manual_add_rows,
    "high_priority_frames_with_missing_slots": high_priority_frames,
    "contact_all_ok": contact_all_ok,
    "contact_priority_ok": contact_priority_ok,
})
