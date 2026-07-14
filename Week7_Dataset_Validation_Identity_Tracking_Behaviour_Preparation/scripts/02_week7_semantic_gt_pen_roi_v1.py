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

FRAME_INDEX_PATH = W6 / "outputs" / "unified_ground_truth" / "week6_scanpoint_frame_index.csv"
DETS_PATH = W6 / "outputs" / "feature_extractors" / "week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv"

OUT_ROI = W7 / "outputs" / "roi_and_crate_mapping"
OUT_VIS = W7 / "outputs" / "visualizations"
OUT_STATS = W7 / "outputs" / "dataset_statistics"
OUT_NOTES = W7 / "notes"

for p in [OUT_ROI, OUT_VIS, OUT_STATS, OUT_NOTES]:
    p.mkdir(parents=True, exist_ok=True)


ROI_DEF_CSV = OUT_ROI / "week7_semantic_gt_pen_roi_v1_definitions.csv"
ROI_DEF_JSON = OUT_ROI / "week7_semantic_gt_pen_roi_v1_definitions.json"
BBOX_ASSIGNMENT = OUT_ROI / "week7_bbox_roi_assignment_semantic_gt_pen_v1.csv"
FRAME_SUMMARY = OUT_ROI / "week7_roi_frame_summary_semantic_gt_pen_v1.csv"
MANUAL_REVIEW = OUT_ROI / "week7_semantic_roi_v1_manual_review_template.csv"

CONTACT_BROAD = OUT_VIS / "week7_semantic_gt_pen_roi_v1_broad_contact_sheet.jpg"
CONTACT_STRICT = OUT_VIS / "week7_semantic_gt_pen_roi_v1_strict_contact_sheet.jpg"
CONTACT_COMPARE = OUT_VIS / "week7_semantic_gt_pen_roi_v1_comparison_contact_sheet.jpg"

NOTE_PATH = OUT_NOTES / "week7_semantic_gt_pen_roi_v1_notes.md"
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


def draw_polygon(img, points, color, thickness=3):
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


def roi_area(points):
    contour = np.array(points, dtype=np.int32)
    return float(cv2.contourArea(contour))


def get_image_shape(frames, image_col):
    for v in frames[image_col]:
        p = resolve_path(v)
        if p is None:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        return w, h
    raise RuntimeError("Could not read any frame image.")


def make_contact_sheet(frames, dets, roi_points, roi_name, output_path, colour_inside=(0, 190, 0), colour_outside=(140, 140, 140)):
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

        # ROI polygon in yellow.
        draw_polygon(overlay, roi_points, (0, 220, 255), 3)

        frame_dets = dets[dets["scan_frame_id"].astype(str) == sid].copy()

        for _, d in frame_dets.iterrows():
            x1 = int(round(float(d["x1"])))
            y1 = int(round(float(d["y1"])))
            x2 = int(round(float(d["x2"])))
            y2 = int(round(float(d["y2"])))

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            inside = point_in_polygon(cx, cy, roi_points)
            color = colour_inside if inside else colour_outside
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
            f"{sid} | {roi_name}",
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


def make_compare_contact_sheet(broad_path, strict_path, out_path):
    b = cv2.imread(str(broad_path)) if Path(broad_path).exists() else None
    s = cv2.imread(str(strict_path)) if Path(strict_path).exists() else None

    if b is None or s is None:
        return False

    # Resize to same width and stack vertically.
    width = min(b.shape[1], s.shape[1])
    b = cv2.resize(b, (width, int(b.shape[0] * width / b.shape[1])), interpolation=cv2.INTER_AREA)
    s = cv2.resize(s, (width, int(s.shape[0] * width / s.shape[1])), interpolation=cv2.INTER_AREA)

    title_h = 50
    canvas = np.full((b.shape[0] + s.shape[0] + title_h * 2, width, 3), 255, dtype=np.uint8)

    cv2.putText(canvas, "Broad semantic ground truth pen Region of Interest", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
    canvas[title_h:title_h + b.shape[0], :, :] = b

    y2 = title_h + b.shape[0]
    cv2.putText(canvas, "Strict upper ground truth pen Region of Interest", (20, y2 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
    canvas[y2 + title_h:y2 + title_h + s.shape[0], :, :] = s

    cv2.imwrite(str(out_path), canvas)
    return True


# ---------------------------------------------------------------------
# Load data.
# ---------------------------------------------------------------------
frames = pd.read_csv(FRAME_INDEX_PATH)
dets = pd.read_csv(DETS_PATH)

if "det_id" not in dets.columns:
    dets["det_id"] = dets.groupby("scan_frame_id").cumcount()

frame_cols = ["scan_frame_id"]

for c in [
    "frame_image_path",
    "image_path",
    "frame_path",
    "scanpoint_frame_path",
    "video_id",
    "timestamp",
    "frame_index",
    "timestamp_sec_in_video",
    "video_match_status",
]:
    if c in frames.columns:
        frame_cols.append(c)

frame_cols = list(dict.fromkeys(frame_cols))

dets = dets.merge(
    frames[frame_cols],
    on="scan_frame_id",
    how="left",
    suffixes=("", "_from_frame_index"),
)

if "frame_image_path" not in dets.columns and "frame_image_path_from_frame_index" in dets.columns:
    dets["frame_image_path"] = dets["frame_image_path_from_frame_index"]

x1_col = first_col(dets, ["x1", "bbox_x1", "xmin", "left"])
y1_col = first_col(dets, ["y1", "bbox_y1", "ymin", "top"])
x2_col = first_col(dets, ["x2", "bbox_x2", "xmax", "right"])
y2_col = first_col(dets, ["y2", "bbox_y2", "ymax", "bottom"])

if not all([x1_col, y1_col, x2_col, y2_col]):
    raise RuntimeError(f"Could not find bbox columns. Columns: {list(dets.columns)}")

dets["x1"] = pd.to_numeric(dets[x1_col], errors="coerce")
dets["y1"] = pd.to_numeric(dets[y1_col], errors="coerce")
dets["x2"] = pd.to_numeric(dets[x2_col], errors="coerce")
dets["y2"] = pd.to_numeric(dets[y2_col], errors="coerce")
dets = dets.dropna(subset=["x1", "y1", "x2", "y2"]).copy()

image_col = first_col(frames, ["frame_image_path", "image_path", "frame_path", "scanpoint_frame_path"])
if image_col is None:
    raise RuntimeError("Frame index has no image path column.")

frame_w, frame_h = get_image_shape(frames, image_col)

# ---------------------------------------------------------------------
# Semantic ROI candidates.
# Coordinates are intentionally saved as v1 and require visual review.
# User confirmation: the annotated Region of Interest is the upper/right pen where the visible pigs are located.
# ---------------------------------------------------------------------
broad_points = [
    [115, 0],
    [704, 0],
    [704, 576],
    [85, 576],
    [85, 420],
    [135, 315],
    [135, 0],
]

strict_points = [
    [140, 0],
    [704, 0],
    [704, 355],
    [170, 355],
    [170, 0],
]

# Clip points to frame size.
def clipped(points):
    out = []
    for x, y in points:
        out.append([
            int(max(0, min(frame_w, x))),
            int(max(0, min(frame_h, y))),
        ])
    return out

broad_points = clipped(broad_points)
strict_points = clipped(strict_points)

roi_defs = pd.DataFrame([
    {
        "roi_name": "broad_semantic_gt_pen_roi_v1",
        "roi_type": "polygon",
        "points_json": json.dumps(broad_points),
        "frame_width": frame_w,
        "frame_height": frame_h,
        "source": "human semantic confirmation from contact sheet",
        "interpretation": "Recommended initial ROI because it covers the upper/right annotated pen while retaining pigs visible lower in the same pen.",
        "human_confirmation_status": "confirmed_semantic_region_needs_coordinate_review",
        "area_pixels": roi_area(broad_points),
        "area_fraction_of_frame": roi_area(broad_points) / float(frame_w * frame_h),
    },
    {
        "roi_name": "strict_upper_gt_pen_roi_v1",
        "roi_type": "polygon",
        "points_json": json.dumps(strict_points),
        "frame_width": frame_w,
        "frame_height": frame_h,
        "source": "strict upper/right candidate",
        "interpretation": "More aggressive ROI for testing; may exclude pigs if the annotated pen extends downward.",
        "human_confirmation_status": "candidate_for_comparison",
        "area_pixels": roi_area(strict_points),
        "area_fraction_of_frame": roi_area(strict_points) / float(frame_w * frame_h),
    },
])

safe_to_csv(roi_defs, ROI_DEF_CSV)

with open(ROI_DEF_JSON, "w") as f:
    json.dump(roi_defs.to_dict(orient="records"), f, indent=2)

# ---------------------------------------------------------------------
# Assign bbox centres to both ROI candidates.
# ---------------------------------------------------------------------
assignment = dets.copy()
assignment["bbox_center_x"] = (assignment["x1"] + assignment["x2"]) / 2.0
assignment["bbox_center_y"] = (assignment["y1"] + assignment["y2"]) / 2.0

for roi_name, points in [
    ("broad_semantic_gt_pen_roi_v1", broad_points),
    ("strict_upper_gt_pen_roi_v1", strict_points),
]:
    assignment[f"inside_{roi_name}"] = assignment.apply(
        lambda r: point_in_polygon(r["bbox_center_x"], r["bbox_center_y"], points),
        axis=1,
    )

assignment["recommended_inside_gt_pen_roi_v1"] = assignment["inside_broad_semantic_gt_pen_roi_v1"]
assignment["recommended_roi_decision_v1"] = np.where(
    assignment["recommended_inside_gt_pen_roi_v1"],
    "inside_annotated_gt_pen",
    "outside_or_adjacent_pen_ignore_for_gt_matching",
)

safe_to_csv(assignment, BBOX_ASSIGNMENT)

# ---------------------------------------------------------------------
# Frame summary.
# ---------------------------------------------------------------------
frame_rows = []

for sid, group in assignment.groupby("scan_frame_id"):
    row = {
        "scan_frame_id": sid,
        "detections": len(group),
    }

    for roi_name in ["broad_semantic_gt_pen_roi_v1", "strict_upper_gt_pen_roi_v1"]:
        col = f"inside_{roi_name}"
        row[f"{roi_name}_inside_count"] = int(group[col].sum())
        row[f"{roi_name}_outside_count"] = int((~group[col]).sum())
        row[f"{roi_name}_inside_fraction"] = float(group[col].mean()) if len(group) else np.nan

    frame_rows.append(row)

frame_summary = pd.DataFrame(frame_rows)
safe_to_csv(frame_summary, FRAME_SUMMARY)

# ---------------------------------------------------------------------
# Manual review template.
# ---------------------------------------------------------------------
manual = roi_defs.copy()
manual["human_decision"] = [
    "recommended_pending_final_visual_confirmation",
    "comparison_candidate_pending_review",
]
manual["adjusted_points_json"] = manual["points_json"]
manual["reviewer_notes"] = ""
manual["recommended_next_action"] = (
    "Inspect broad and strict contact sheets. If broad ROI matches the upper/right annotated pen, mark it final. "
    "If coordinates need adjustment, edit adjusted_points_json."
)

safe_to_csv(manual, MANUAL_REVIEW)

# ---------------------------------------------------------------------
# Contact sheets.
# ---------------------------------------------------------------------
broad_ok = make_contact_sheet(
    frames,
    assignment,
    broad_points,
    "broad_semantic_gt_pen_roi_v1",
    CONTACT_BROAD,
)

strict_ok = make_contact_sheet(
    frames,
    assignment,
    strict_points,
    "strict_upper_gt_pen_roi_v1",
    CONTACT_STRICT,
)

compare_ok = make_compare_contact_sheet(CONTACT_BROAD, CONTACT_STRICT, CONTACT_COMPARE)

# ---------------------------------------------------------------------
# Update task tracker stage 2.
# ---------------------------------------------------------------------
if TASK_TRACKER_PATH.exists():
    tracker = pd.read_csv(TASK_TRACKER_PATH)

    mask = tracker["stage"].astype(str) == "2"

    if mask.any():
        tracker.loc[mask, "status"] = "semantic_roi_v1_generated_needs_final_coordinate_confirmation"

    safe_to_csv(tracker, TASK_TRACKER_PATH)

# ---------------------------------------------------------------------
# Notes.
# ---------------------------------------------------------------------
inside_summary = []

for roi_name in ["broad_semantic_gt_pen_roi_v1", "strict_upper_gt_pen_roi_v1"]:
    col = f"inside_{roi_name}"

    inside_summary.append({
        "roi_name": roi_name,
        "inside_count": int(assignment[col].sum()),
        "outside_count": int((~assignment[col]).sum()),
        "inside_fraction": float(assignment[col].mean()),
    })

inside_summary_df = pd.DataFrame(inside_summary)

with open(NOTE_PATH, "w") as f:
    f.write("# Week 7 Semantic Ground Truth Pen Region of Interest v1\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This step records the annotated Region of Interest as a semantic ground truth pen decision. "
        "The user confirmed that the Region of Interest should correspond to the upper/right pen where the visible annotated pigs are located. "
        "Two coordinate candidates are saved for review: a broad recommended candidate and a stricter upper candidate.\n\n"
    )

    f.write("## Region of Interest definitions\n\n")
    f.write(roi_defs.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Detection assignment summary\n\n")
    f.write(inside_summary_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Visual review files\n\n")
    f.write(f"- Broad candidate contact sheet: `{CONTACT_BROAD}`\n")
    f.write(f"- Strict candidate contact sheet: `{CONTACT_STRICT}`\n")
    f.write(f"- Comparison contact sheet: `{CONTACT_COMPARE}`\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The broad semantic Region of Interest is the recommended initial candidate because it keeps the upper/right pen while reducing influence from adjacent or outside areas. "
        "However, the coordinates should still be visually confirmed before colour identity matching and ground truth association are finalized.\n\n"
    )

    f.write("## Outputs\n\n")
    for p in [
        ROI_DEF_CSV,
        ROI_DEF_JSON,
        BBOX_ASSIGNMENT,
        FRAME_SUMMARY,
        MANUAL_REVIEW,
        CONTACT_BROAD,
        CONTACT_STRICT,
        CONTACT_COMPARE,
    ]:
        f.write(f"- `{p}`\n")

print("Saved:")
print(ROI_DEF_CSV)
print(ROI_DEF_JSON)
print(BBOX_ASSIGNMENT)
print(FRAME_SUMMARY)
print(MANUAL_REVIEW)
print(CONTACT_BROAD)
print(CONTACT_STRICT)
print(CONTACT_COMPARE)
print(NOTE_PATH)

print()
print("=== Semantic ROI definitions ===")
print(roi_defs.to_string(index=False))

print()
print("=== Detection inside/outside summary ===")
print(inside_summary_df.to_string(index=False))

print()
print("=== Contact sheets generated ===")
print({"broad": broad_ok, "strict": strict_ok, "comparison": compare_ok})
