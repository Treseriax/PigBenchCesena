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
ADAPTIVE_ASSIGNMENT_PATH = W7 / "outputs" / "roi_and_crate_mapping" / "week7_bbox_roi_assignment_adaptive_upper_pen_v3.csv"
FRAME_SELECTION_PATH = W7 / "outputs" / "roi_and_crate_mapping" / "week7_adaptive_upper_pen_roi_v3_frame_selection.csv"

OUT_ROI = W7 / "outputs" / "roi_and_crate_mapping"
OUT_VIS = W7 / "outputs" / "visualizations"
OUT_STATS = W7 / "outputs" / "dataset_statistics"
OUT_NOTES = W7 / "notes"

for p in [OUT_ROI, OUT_VIS, OUT_STATS, OUT_NOTES]:
    p.mkdir(parents=True, exist_ok=True)

OUT_SELECTION = OUT_ROI / "week7_gt_pen_detection_selection_v4.csv"
OUT_FRAME_SUMMARY = OUT_ROI / "week7_gt_pen_detection_selection_v4_frame_summary.csv"
OUT_REVIEW = OUT_ROI / "week7_gt_pen_detection_selection_v4_manual_review_template.csv"
OUT_CONTACT = OUT_VIS / "week7_gt_pen_detection_selection_v4_contact_sheet.jpg"
OUT_PROBLEM_CONTACT = OUT_VIS / "week7_gt_pen_detection_selection_v4_problem_frames_contact_sheet.jpg"
OUT_NOTE = OUT_NOTES / "week7_gt_pen_detection_selection_v4_notes.md"

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


def nms_like_select(df, score_col, max_keep=6, iou_threshold=0.55):
    selected_idx = []

    df = df.sort_values(score_col, ascending=False).copy()

    for idx, row in df.iterrows():
        box = [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])]

        duplicate = False

        for sidx in selected_idx:
            srow = df.loc[sidx]
            sbox = [float(srow["x1"]), float(srow["y1"]), float(srow["x2"]), float(srow["y2"])]

            if bbox_iou(box, sbox) >= iou_threshold:
                duplicate = True
                break

        if not duplicate:
            selected_idx.append(idx)

        if len(selected_idx) >= max_keep:
            break

    return selected_idx


def make_contact_sheet(frames, selection, output_path, only_problem=False):
    image_col = first_col(frames, ["frame_image_path", "image_path", "frame_path", "scanpoint_frame_path"])

    if image_col is None:
        return False

    frame_list = frames.copy()

    if "scan_frame_id" in frame_list.columns:
        frame_list = frame_list.drop_duplicates("scan_frame_id").sort_values("scan_frame_id")

    if only_problem:
        problem_sids = set(
            selection.groupby("scan_frame_id")["selected_for_gt_pen_matching_v4"].sum()
            .loc[lambda s: (s < 5) | (s > 7)]
            .index.astype(str)
        )

        frame_list = frame_list[frame_list["scan_frame_id"].astype(str).isin(problem_sids)]

        if len(frame_list) == 0:
            return False

    if len(frame_list) > 16:
        idx = np.linspace(0, len(frame_list) - 1, 16).round().astype(int)
        frame_list = frame_list.iloc[idx]

    thumbs = []
    thumb_w, thumb_h = 380, 250

    for _, fr in frame_list.iterrows():
        sid = str(fr.get("scan_frame_id", ""))

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

            selected = bool(d["selected_for_gt_pen_matching_v4"])

            if selected:
                color = (0, 220, 0)
                thickness = 3
                prefix = "GT"
            else:
                color = (130, 130, 130)
                thickness = 1
                prefix = "IGN"

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

            label = f"{prefix} det {d.get('det_id', '')}"
            cv2.putText(
                overlay,
                label,
                (x1, max(17, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

        selected_count = int(frame_dets["selected_for_gt_pen_matching_v4"].sum()) if len(frame_dets) else 0

        cv2.putText(
            overlay,
            f"{sid} | selected GT pen detections: {selected_count}",
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        thumbs.append(cv2.resize(overlay, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA))

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


# ---------------------------------------------------------------------
# Load data.
# ---------------------------------------------------------------------
frames = pd.read_csv(FRAME_INDEX_PATH)
assignment = pd.read_csv(ADAPTIVE_ASSIGNMENT_PATH)

# Normalize numeric columns.
for c in ["x1", "y1", "x2", "y2"]:
    assignment[c] = pd.to_numeric(assignment[c], errors="coerce")

assignment = assignment.dropna(subset=["x1", "y1", "x2", "y2"]).copy()

if "score" in assignment.columns:
    assignment["detector_score_num"] = pd.to_numeric(assignment["score"], errors="coerce").fillna(0.0)
else:
    assignment["detector_score_num"] = 0.5

if "selected_roi_mask_overlap_fraction_v3" in assignment.columns:
    assignment["roi_overlap_num"] = pd.to_numeric(
        assignment["selected_roi_mask_overlap_fraction_v3"],
        errors="coerce",
    ).fillna(0.0)
else:
    assignment["roi_overlap_num"] = 0.0

assignment["bbox_center_x"] = (assignment["x1"] + assignment["x2"]) / 2.0
assignment["bbox_center_y"] = (assignment["y1"] + assignment["y2"]) / 2.0
assignment["bbox_area"] = (assignment["x2"] - assignment["x1"]).clip(lower=1) * (assignment["y2"] - assignment["y1"]).clip(lower=1)

# The ground-truth labels have six colour identities per scanpoint.
TARGET_PIGS_PER_FRAME = 6

result_rows = []
frame_rows = []

for sid, group in assignment.groupby("scan_frame_id", sort=True):
    g = group.copy()

    # Compute robust within-frame position ranks.
    y_min = g["bbox_center_y"].min()
    y_max = g["bbox_center_y"].max()
    y_range = max(1.0, y_max - y_min)

    g["relative_y_rank"] = (g["bbox_center_y"] - y_min) / y_range

    # Penalize lower detections, but do not hard-code right/left.
    # Higher score means better GT-pen candidate.
    g["gt_pen_candidate_score_v4"] = (
        2.20 * g["roi_overlap_num"]
        + 1.00 * g["detector_score_num"]
        - 0.90 * g["relative_y_rank"]
    )

    # Strong prefilter: keep plausible upper-pen / ROI detections.
    plausible = g[
        (g["roi_overlap_num"] >= 0.45)
        | (g["relative_y_rank"] <= 0.65)
    ].copy()

    if len(plausible) < TARGET_PIGS_PER_FRAME:
        plausible = g.copy()

    selected_idx = nms_like_select(
        plausible,
        score_col="gt_pen_candidate_score_v4",
        max_keep=TARGET_PIGS_PER_FRAME,
        iou_threshold=0.55,
    )

    # If after duplicate filtering fewer than 6 remain, fill with best remaining plausible detections.
    if len(selected_idx) < min(TARGET_PIGS_PER_FRAME, len(plausible)):
        remaining = plausible.drop(index=selected_idx, errors="ignore").sort_values(
            "gt_pen_candidate_score_v4",
            ascending=False,
        )

        for idx in remaining.index:
            if idx not in selected_idx:
                selected_idx.append(idx)

            if len(selected_idx) >= min(TARGET_PIGS_PER_FRAME, len(plausible)):
                break

    g["selected_for_gt_pen_matching_v4"] = g.index.isin(selected_idx)

    g["gt_pen_selection_reason_v4"] = np.where(
        g["selected_for_gt_pen_matching_v4"],
        "selected_top_candidate_after_roi_overlap_position_score_and_duplicate_filter",
        "ignored_not_selected_for_gt_pen_matching",
    )

    selected_count = int(g["selected_for_gt_pen_matching_v4"].sum())

    frame_rows.append({
        "scan_frame_id": sid,
        "total_detections": len(g),
        "selected_for_gt_pen_matching_v4_count": selected_count,
        "ignored_detections_v4_count": len(g) - selected_count,
        "target_pigs_per_frame": TARGET_PIGS_PER_FRAME,
        "selection_count_delta_from_target": selected_count - TARGET_PIGS_PER_FRAME,
        "mean_selected_roi_overlap": float(g.loc[g["selected_for_gt_pen_matching_v4"], "roi_overlap_num"].mean()) if selected_count else 0,
        "mean_ignored_roi_overlap": float(g.loc[~g["selected_for_gt_pen_matching_v4"], "roi_overlap_num"].mean()) if len(g) - selected_count else 0,
    })

    result_rows.append(g)

selection = pd.concat(result_rows, ignore_index=True)
frame_summary = pd.DataFrame(frame_rows)

safe_to_csv(selection, OUT_SELECTION)
safe_to_csv(frame_summary, OUT_FRAME_SUMMARY)

review = frame_summary.copy()
review["human_review_status"] = "pending"
review["reviewer_notes"] = ""
review["recommended_next_action"] = "Inspect contact sheet. If wrong-pen detections remain selected, mark frame for manual override."
safe_to_csv(review, OUT_REVIEW)

contact_ok = make_contact_sheet(frames, selection, OUT_CONTACT, only_problem=False)
problem_ok = make_contact_sheet(frames, selection, OUT_PROBLEM_CONTACT, only_problem=True)

# Update tracker.
if TASK_TRACKER_PATH.exists():
    tracker = pd.read_csv(TASK_TRACKER_PATH)
    mask = tracker["stage"].astype(str) == "2"

    if mask.any():
        tracker.loc[mask, "status"] = "gt_pen_detection_selection_v4_generated_needs_visual_confirmation"

    safe_to_csv(tracker, TASK_TRACKER_PATH)

# Notes.
selected_total = int(selection["selected_for_gt_pen_matching_v4"].sum())
ignored_total = int((~selection["selected_for_gt_pen_matching_v4"]).sum())

frames_exact_6 = int((frame_summary["selected_for_gt_pen_matching_v4_count"] == 6).sum())
frames_under_6 = int((frame_summary["selected_for_gt_pen_matching_v4_count"] < 6).sum())
frames_over_6 = int((frame_summary["selected_for_gt_pen_matching_v4_count"] > 6).sum())

with open(OUT_NOTE, "w") as f:
    f.write("# Week 7 Ground Truth Pen Detection Selection v4\n\n")

    f.write("## Reason for v4\n\n")
    f.write(
        "Adaptive ROI v3 improved the Region of Interest but still selected some detections from the wrong pen. "
        "The problem is not only geometric Region of Interest selection; the task needs frame-level selection of detections that correspond to the six manually annotated pigs.\n\n"
    )

    f.write("## Method\n\n")
    f.write(
        "For each scanpoint frame, detections are scored using Segment Anything Model mask overlap with the adaptive upper-pen Region of Interest, detector confidence, and relative vertical position. "
        "A duplicate filtering step similar to non-maximum suppression is used, and up to six detections are selected for ground-truth colour and behaviour matching.\n\n"
    )

    f.write("## Summary\n\n")
    f.write(f"- Total detections: `{len(selection)}`\n")
    f.write(f"- Selected for ground-truth pen matching: `{selected_total}`\n")
    f.write(f"- Ignored detections: `{ignored_total}`\n")
    f.write(f"- Frames with exactly six selected detections: `{frames_exact_6}`\n")
    f.write(f"- Frames with fewer than six selected detections: `{frames_under_6}`\n")
    f.write(f"- Frames with more than six selected detections: `{frames_over_6}`\n\n")

    f.write("## Important interpretation\n\n")
    f.write(
        "This file should be used for colour and behaviour matching instead of raw Region of Interest membership alone. "
        "The selected detections are the best automatic estimate of which detector boxes belong to the annotated ground-truth pen. "
        "Manual visual review is still required before treating this as final.\n\n"
    )

    f.write("## Outputs\n\n")
    for p in [OUT_SELECTION, OUT_FRAME_SUMMARY, OUT_REVIEW, OUT_CONTACT, OUT_PROBLEM_CONTACT]:
        f.write(f"- `{p}`\n")

print("Saved:")
print(OUT_SELECTION)
print(OUT_FRAME_SUMMARY)
print(OUT_REVIEW)
print(OUT_CONTACT)
print(OUT_PROBLEM_CONTACT)
print(OUT_NOTE)

print()
print("=== Selection summary ===")
print({
    "total_detections": len(selection),
    "selected_for_gt_pen_matching": selected_total,
    "ignored_detections": ignored_total,
    "frames_exact_6": frames_exact_6,
    "frames_under_6": frames_under_6,
    "frames_over_6": frames_over_6,
    "contact_sheet_generated": contact_ok,
    "problem_contact_sheet_generated": problem_ok,
})
