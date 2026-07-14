from pathlib import Path
from datetime import datetime
import csv
import json
import math

import cv2
import numpy as np
import pandas as pd


ROOT = Path.home() / "PigBench"

W6 = ROOT / "Week6_Unibo_Dataset_Validation"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

SEMANTIC_ROI_DEF = W7 / "outputs" / "roi_and_crate_mapping" / "week7_semantic_gt_pen_roi_v1_definitions.csv"
SEMANTIC_ASSIGNMENT = W7 / "outputs" / "roi_and_crate_mapping" / "week7_bbox_roi_assignment_semantic_gt_pen_v1.csv"
FRAME_INDEX_PATH = W6 / "outputs" / "unified_ground_truth" / "week6_scanpoint_frame_index.csv"

OUT_ROI = W7 / "outputs" / "roi_and_crate_mapping"
OUT_VIS = W7 / "outputs" / "visualizations"
OUT_STATS = W7 / "outputs" / "dataset_statistics"
OUT_NOTES = W7 / "notes"

for p in [OUT_ROI, OUT_VIS, OUT_STATS, OUT_NOTES]:
    p.mkdir(parents=True, exist_ok=True)

FINAL_ROI_CSV = OUT_ROI / "week7_final_gt_pen_roi_definition.csv"
FINAL_ROI_JSON = OUT_ROI / "week7_final_gt_pen_roi_definition.json"
FINAL_ASSIGNMENT = OUT_ROI / "week7_bbox_roi_assignment_final.csv"
FINAL_FRAME_SUMMARY = OUT_ROI / "week7_roi_frame_summary_final.csv"
FINAL_CONTACT = OUT_VIS / "week7_final_gt_pen_roi_overlay_contact_sheet.jpg"
FINAL_NOTE = OUT_NOTES / "week7_roi_final_decision_notes.md"
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


def point_in_polygon(x, y, points):
    contour = np.array(points, dtype=np.int32)
    return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0


def draw_polygon(img, points, color=(0, 220, 255), thickness=3):
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def make_contact_sheet(frames, assignment, points, output_path):
    image_col = first_col(frames, ["frame_image_path", "image_path", "frame_path", "scanpoint_frame_path"])
    if image_col is None:
        return False

    sample = frames.copy()

    if "scan_frame_id" in sample.columns:
        sample = sample.drop_duplicates("scan_frame_id").sort_values("scan_frame_id")

    if len(sample) > 16:
        idx = np.linspace(0, len(sample) - 1, 16).round().astype(int)
        sample = sample.iloc[idx]

    thumbs = []
    thumb_w, thumb_h = 380, 250

    for _, fr in sample.iterrows():
        sid = str(fr.get("scan_frame_id", ""))
        img_path = resolve_path(fr[image_col])

        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        overlay = img.copy()

        draw_polygon(overlay, points, color=(0, 220, 255), thickness=3)

        frame_dets = assignment[assignment["scan_frame_id"].astype(str) == sid].copy()

        for _, d in frame_dets.iterrows():
            x1 = int(round(float(d["x1"])))
            y1 = int(round(float(d["y1"])))
            x2 = int(round(float(d["x2"])))
            y2 = int(round(float(d["y2"])))

            inside = bool(d["inside_final_gt_pen_roi"])

            color = (0, 190, 0) if inside else (135, 135, 135)
            thickness = 2 if inside else 1

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

            label = f"{'in' if inside else 'out'} det {d.get('det_id', '')}"

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

        cv2.putText(
            overlay,
            f"{sid} | final_gt_pen_roi",
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        resized = cv2.resize(overlay, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
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


# ---------------------------------------------------------------------
# Load semantic ROI candidates.
# ---------------------------------------------------------------------
roi_defs = pd.read_csv(SEMANTIC_ROI_DEF)
assignment = pd.read_csv(SEMANTIC_ASSIGNMENT)
frames = pd.read_csv(FRAME_INDEX_PATH)

selected = roi_defs[roi_defs["roi_name"] == "broad_semantic_gt_pen_roi_v1"].copy()

if len(selected) != 1:
    raise RuntimeError("Could not find broad_semantic_gt_pen_roi_v1 in semantic ROI definitions.")

selected = selected.iloc[0].to_dict()
points = json.loads(selected["points_json"])

final_roi = pd.DataFrame([
    {
        "roi_name": "final_gt_pen_roi_v1",
        "roi_type": "polygon",
        "points_json": json.dumps(points),
        "frame_width": selected["frame_width"],
        "frame_height": selected["frame_height"],
        "source": "human visual confirmation from broad semantic ground truth pen ROI",
        "decision": "final_selected",
        "reason": "The broad candidate matches the upper/right annotated ground truth pen and retains pigs visible lower in the same pen. The strict candidate excludes too many valid pigs.",
        "area_pixels": selected["area_pixels"],
        "area_fraction_of_frame": selected["area_fraction_of_frame"],
        "finalized_at": datetime.now().isoformat(timespec="seconds"),
    }
])

safe_to_csv(final_roi, FINAL_ROI_CSV)

with open(FINAL_ROI_JSON, "w") as f:
    json.dump(final_roi.to_dict(orient="records"), f, indent=2)

# ---------------------------------------------------------------------
# Final assignment.
# ---------------------------------------------------------------------
assignment["inside_final_gt_pen_roi"] = assignment.apply(
    lambda r: point_in_polygon(
        (float(r["x1"]) + float(r["x2"])) / 2.0,
        (float(r["y1"]) + float(r["y2"])) / 2.0,
        points,
    ),
    axis=1,
)

assignment["roi_use_for_gt_matching"] = np.where(
    assignment["inside_final_gt_pen_roi"],
    "use_for_ground_truth_colour_and_behaviour_matching",
    "ignore_for_ground_truth_matching_outside_or_adjacent_pen",
)

safe_to_csv(assignment, FINAL_ASSIGNMENT)

# ---------------------------------------------------------------------
# Frame summary.
# ---------------------------------------------------------------------
frame_rows = []

for sid, group in assignment.groupby("scan_frame_id"):
    inside = int(group["inside_final_gt_pen_roi"].sum())
    outside = int((~group["inside_final_gt_pen_roi"]).sum())

    frame_rows.append({
        "scan_frame_id": sid,
        "detections": len(group),
        "inside_final_gt_pen_roi_count": inside,
        "outside_final_gt_pen_roi_count": outside,
        "inside_final_gt_pen_roi_fraction": inside / len(group) if len(group) else np.nan,
    })

frame_summary = pd.DataFrame(frame_rows)
safe_to_csv(frame_summary, FINAL_FRAME_SUMMARY)

contact_ok = make_contact_sheet(frames, assignment, points, FINAL_CONTACT)

# ---------------------------------------------------------------------
# Update task tracker.
# ---------------------------------------------------------------------
if TASK_TRACKER_PATH.exists():
    tracker = pd.read_csv(TASK_TRACKER_PATH)
    mask = tracker["stage"].astype(str) == "2"

    if mask.any():
        tracker.loc[mask, "status"] = "done_final_roi_v1_selected"

    safe_to_csv(tracker, TASK_TRACKER_PATH)

# ---------------------------------------------------------------------
# Notes.
# ---------------------------------------------------------------------
total = len(assignment)
inside = int(assignment["inside_final_gt_pen_roi"].sum())
outside = int((~assignment["inside_final_gt_pen_roi"]).sum())

with open(FINAL_NOTE, "w") as f:
    f.write("# Week 7 Final Ground Truth Pen Region of Interest Decision\n\n")

    f.write("## Decision\n\n")
    f.write("The final Region of Interest is selected as `final_gt_pen_roi_v1`, based on the previous `broad_semantic_gt_pen_roi_v1` candidate.\n\n")

    f.write("## Reason\n\n")
    f.write(
        "Visual inspection showed that the annotated ground truth pen corresponds to the upper/right pen where the visible annotated pigs are located. "
        "The strict upper candidate was rejected because it excluded too many valid pigs from the same pen. "
        "The broad candidate better preserves all pigs visible in the ground truth pen while still reducing influence from adjacent or outside areas.\n\n"
    )

    f.write("## Detection assignment summary\n\n")
    f.write(f"- Total detections: `{total}`\n")
    f.write(f"- Inside final Region of Interest: `{inside}`\n")
    f.write(f"- Outside final Region of Interest: `{outside}`\n")
    f.write(f"- Inside fraction: `{inside / total if total else 0:.4f}`\n\n")

    f.write("## Important interpretation\n\n")
    f.write(
        "Detections outside the final Region of Interest should not be used for ground-truth colour or behaviour matching unless manually justified. "
        "They may correspond to adjacent pens, partial animals, or detections not associated with the annotated ground truth pen.\n\n"
    )

    f.write("## Outputs\n\n")
    for p in [FINAL_ROI_CSV, FINAL_ROI_JSON, FINAL_ASSIGNMENT, FINAL_FRAME_SUMMARY, FINAL_CONTACT]:
        f.write(f"- `{p}`\n")

print("Saved:")
print(FINAL_ROI_CSV)
print(FINAL_ROI_JSON)
print(FINAL_ASSIGNMENT)
print(FINAL_FRAME_SUMMARY)
print(FINAL_CONTACT)
print(FINAL_NOTE)

print()
print("=== Final ROI ===")
print(final_roi.to_string(index=False))

print()
print("=== Final assignment summary ===")
print({
    "total_detections": total,
    "inside_final_gt_pen_roi": inside,
    "outside_final_gt_pen_roi": outside,
    "inside_fraction": inside / total if total else 0,
    "contact_sheet_generated": contact_ok,
})
