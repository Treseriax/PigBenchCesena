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

EDITOR_ROOT = W7 / "outputs" / "manual_annotation_v10_editor"
CORRECTED_BOXES_PATH = EDITOR_ROOT / "week7_manual_box_editor_v10_corrected_boxes.csv"
ALL_DECISIONS_PATH = EDITOR_ROOT / "week7_manual_box_editor_v10_all_box_decisions.csv"
FRAME_SUMMARY_PATH = EDITOR_ROOT / "week7_manual_box_editor_v10_frame_summary.csv"

IMG_DIR = W7 / "outputs" / "manual_annotation_v9" / "cvat_coco_import" / "images"

OUT_ROOT = W7 / "outputs" / "final_corrected_gt_pen_boxes_v11"
OUT_VIS = OUT_ROOT / "qa_overlays"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT_VIS.mkdir(parents=True, exist_ok=True)

OUT_FINAL = OUT_ROOT / "week7_final_corrected_gt_pen_boxes_v11_for_colour_matching.csv"
OUT_FRAME_QA = OUT_ROOT / "week7_final_corrected_gt_pen_boxes_v11_frame_qa_summary.csv"
OUT_DECISION_AUDIT = OUT_ROOT / "week7_final_corrected_gt_pen_boxes_v11_decision_audit.csv"
OUT_CONTACT_ALL = OUT_ROOT / "week7_final_corrected_gt_pen_boxes_v11_all_frames_contact_sheet.jpg"
OUT_CONTACT_INCOMPLETE = OUT_ROOT / "week7_final_corrected_gt_pen_boxes_v11_incomplete_frames_contact_sheet.jpg"
OUT_HTML = OUT_ROOT / "week7_final_corrected_gt_pen_boxes_v11_static_qa.html"
OUT_NOTE = W7 / "notes" / "week7_final_corrected_gt_pen_boxes_v11_notes.md"

TASK_TRACKER_PATH = W7 / "outputs" / "dataset_statistics" / "week7_master_task_tracker.csv"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def make_overlay_images(frames, corrected, frame_summary):
    rows = []

    for _, fr in frames.drop_duplicates("scan_frame_id").sort_values("scan_frame_id").iterrows():
        sid = str(fr["scan_frame_id"])
        img_path = IMG_DIR / f"{sid}.jpg"

        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        overlay = img.copy()
        g = corrected[corrected["scan_frame_id"].astype(str) == sid].copy()
        fs = frame_summary[frame_summary["scan_frame_id"].astype(str) == sid]

        if len(fs):
            kept = int(fs["kept_boxes"].iloc[0])
            missing = int(fs["missing_vs_expected"].iloc[0])
            complete = str(fs["frame_complete_candidate"].iloc[0]).lower() == "true"
        else:
            kept = len(g)
            missing = max(0, 6 - kept)
            complete = kept == 6

        for i, (_, b) in enumerate(g.iterrows(), start=1):
            x1 = int(round(float(b["x1"])))
            y1 = int(round(float(b["y1"])))
            x2 = int(round(float(b["x2"])))
            y2 = int(round(float(b["y2"])))

            color = (0, 220, 0)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)

            label = f"FINAL {i}"
            cv2.putText(
                overlay,
                label,
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
                cv2.LINE_AA,
            )

        title_color = (255, 255, 255)
        title = f"{sid} | final boxes={kept} | missing={missing} | complete={complete}"

        cv2.putText(
            overlay,
            title,
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            title_color,
            2,
            cv2.LINE_AA,
        )

        if not complete:
            cv2.putText(
                overlay,
                "NOTE: fewer than 6 visible/kept pigs",
                (8, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

        out_path = OUT_VIS / f"{sid}_final_corrected_v11_overlay.jpg"
        cv2.imwrite(str(out_path), overlay)

        rows.append({
            "scan_frame_id": sid,
            "overlay_path": str(out_path),
            "kept_boxes": kept,
            "missing_vs_expected": missing,
            "frame_complete_candidate": complete,
        })

    return pd.DataFrame(rows)


def make_contact_sheet(index_df, output_path, only_incomplete=False):
    df = index_df.copy()

    if only_incomplete:
        df = df[df["frame_complete_candidate"] == False].copy()

    if len(df) == 0:
        return False

    if len(df) > 16:
        idx = np.linspace(0, len(df) - 1, 16).round().astype(int)
        df = df.iloc[idx]

    thumbs = []
    thumb_w, thumb_h = 380, 250

    for _, r in df.iterrows():
        p = Path(str(r["overlay_path"]))
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


def make_html(index_df, out_path):
    cards = []

    for _, r in index_df.sort_values("scan_frame_id").iterrows():
        img_path = Path(str(r["overlay_path"]))
        src = "qa_overlays/" + img_path.name

        cls = "incomplete" if not bool(r["frame_complete_candidate"]) else "complete"

        cards.append(
            f"""
            <div class="frame-card {cls}">
              <h3>{r['scan_frame_id']}</h3>
              <p>
                final boxes={r['kept_boxes']} |
                missing_vs_expected={r['missing_vs_expected']} |
                complete={r['frame_complete_candidate']}
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
  <title>Week 7 Final Corrected GT-Pen Boxes v11 QA</title>
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
    .complete {{
      border-left: 8px solid green;
    }}
    .incomplete {{
      border-left: 8px solid darkorange;
    }}
    img {{
      max-width: 100%;
      border: 1px solid #ddd;
      display: block;
    }}
  </style>
</head>
<body>
  <h1>Week 7 Final Corrected GT-Pen Boxes v11 QA</h1>
  <div class="legend">
    <p>Green boxes are the final manually corrected GT-pen pig boxes used for colour matching.</p>
    <p>Incomplete frames are allowed only when fewer than six true GT-pen pigs are visible.</p>
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
corrected = pd.read_csv(CORRECTED_BOXES_PATH)
all_decisions = pd.read_csv(ALL_DECISIONS_PATH)
frame_summary = pd.read_csv(FRAME_SUMMARY_PATH)

for c in ["x1", "y1", "x2", "y2", "width", "height"]:
    corrected[c] = pd.to_numeric(corrected[c], errors="coerce")

corrected = corrected.dropna(subset=["x1", "y1", "x2", "y2"]).copy()

# Add final metadata.
corrected = corrected.sort_values(["scan_frame_id", "final_box_id"]).copy()
corrected["final_verified_gt_pen_box_v11"] = True
corrected["ready_for_colour_matching"] = True
corrected["correction_source"] = "manual_box_editor_v10"
corrected["finalized_at"] = datetime.now().isoformat(timespec="seconds")

safe_to_csv(corrected, OUT_FINAL)

# Decision audit.
decision_counts = (
    all_decisions
    .groupby(["status", "candidate_class"], dropna=False)
    .size()
    .reset_index(name="count")
    .sort_values(["status", "candidate_class"])
)

safe_to_csv(decision_counts, OUT_DECISION_AUDIT)

# Frame QA.
qa_rows = []

for _, r in frame_summary.iterrows():
    sid = str(r["scan_frame_id"])
    kept = int(r["kept_boxes"])
    expected = int(r["expected_pigs"])
    missing = int(r["missing_vs_expected"])
    complete = str(r["frame_complete_candidate"]).lower() == "true"

    note = ""
    if missing > 0:
        note = "accepted_fewer_than_six_only_if_not_visible_or_not_reliably_annotatable"

    qa_rows.append({
        "scan_frame_id": sid,
        "expected_pigs": expected,
        "final_corrected_box_count": kept,
        "missing_vs_expected": missing,
        "extra_vs_expected": int(r["extra_vs_expected"]),
        "pending_boxes": int(r["pending_boxes"]),
        "rejected_boxes": int(r["rejected_boxes"]),
        "deleted_boxes": int(r["deleted_boxes"]),
        "frame_complete_candidate": complete,
        "qa_note": note,
    })

frame_qa = pd.DataFrame(qa_rows)
safe_to_csv(frame_qa, OUT_FRAME_QA)

# Visual QA.
overlay_index = make_overlay_images(frames, corrected, frame_summary)
safe_to_csv(overlay_index, OUT_ROOT / "week7_final_corrected_gt_pen_boxes_v11_overlay_index.csv")

contact_all_ok = make_contact_sheet(overlay_index, OUT_CONTACT_ALL, only_incomplete=False)
contact_incomplete_ok = make_contact_sheet(overlay_index, OUT_CONTACT_INCOMPLETE, only_incomplete=True)
make_html(overlay_index, OUT_HTML)

# Update tracker.
if TASK_TRACKER_PATH.exists():
    tracker = pd.read_csv(TASK_TRACKER_PATH)
    mask = tracker["stage"].astype(str) == "2"
    if mask.any():
        tracker.loc[mask, "status"] = "final_corrected_gt_pen_boxes_v11_completed"
    safe_to_csv(tracker, TASK_TRACKER_PATH)

# Notes.
total_boxes = len(corrected)
total_frames = frame_qa["scan_frame_id"].nunique()
complete_frames = int(frame_qa["frame_complete_candidate"].sum())
incomplete_frames = total_frames - complete_frames
missing_total = int(frame_qa["missing_vs_expected"].sum())
incomplete_ids = frame_qa.loc[frame_qa["missing_vs_expected"] > 0, "scan_frame_id"].tolist()

OUT_NOTE.write_text(
    "# Week 7 Final Corrected GT-Pen Boxes v11\n\n"
    "## Purpose\n\n"
    "This step validates the manually corrected GT-pen pig boxes from the custom v10 editor. "
    "These boxes replace the unreliable automatic ROI/detector selection for downstream colour matching.\n\n"
    "## Summary\n\n"
    f"- Total scanpoint frames: `{total_frames}`\n"
    f"- Complete frames with six kept boxes: `{complete_frames}`\n"
    f"- Incomplete frames: `{incomplete_frames}`\n"
    f"- Total final corrected boxes: `{total_boxes}`\n"
    f"- Total missing vs six-per-frame expectation: `{missing_total}`\n"
    f"- Incomplete frame IDs: `{', '.join(incomplete_ids)}`\n\n"
    "## Interpretation\n\n"
    "Frames with fewer than six boxes are accepted only when the missing pig is not visible or not reliably annotatable. "
    "It is safer to keep five true GT-pen boxes than to add a wrong-pen or hallucinated box.\n\n"
    "## Main output for colour matching\n\n"
    f"- `{OUT_FINAL}`\n\n"
    "## QA outputs\n\n"
    f"- Frame QA summary: `{OUT_FRAME_QA}`\n"
    f"- Decision audit: `{OUT_DECISION_AUDIT}`\n"
    f"- All frame contact sheet: `{OUT_CONTACT_ALL}`\n"
    f"- Incomplete frame contact sheet: `{OUT_CONTACT_INCOMPLETE}`\n"
    f"- Static QA HTML: `{OUT_HTML}`\n"
)

print("Saved:")
print(OUT_FINAL)
print(OUT_FRAME_QA)
print(OUT_DECISION_AUDIT)
print(OUT_CONTACT_ALL)
print(OUT_CONTACT_INCOMPLETE)
print(OUT_HTML)
print(OUT_NOTE)

print()
print("=== final corrected boxes v11 summary ===")
print({
    "total_frames": total_frames,
    "complete_frames": complete_frames,
    "incomplete_frames": incomplete_frames,
    "total_final_corrected_boxes": total_boxes,
    "missing_total": missing_total,
    "incomplete_frame_ids": incomplete_ids,
    "contact_all_ok": contact_all_ok,
    "contact_incomplete_ok": contact_incomplete_ok,
})
