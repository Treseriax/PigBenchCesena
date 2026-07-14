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
SAM_FEATURES_PATH = W6 / "outputs" / "feature_extractors" / "week6_sam_box_prompt_segmentation_features.csv"

OUT_ROI = W7 / "outputs" / "roi_and_crate_mapping"
OUT_VIS = W7 / "outputs" / "visualizations"
OUT_STATS = W7 / "outputs" / "dataset_statistics"
OUT_NOTES = W7 / "notes"

for p in [OUT_ROI, OUT_VIS, OUT_STATS, OUT_NOTES]:
    p.mkdir(parents=True, exist_ok=True)

OUT_CANDIDATES = OUT_ROI / "week7_adaptive_upper_pen_roi_v3_candidate_definitions.csv"
OUT_CANDIDATES_JSON = OUT_ROI / "week7_adaptive_upper_pen_roi_v3_candidate_definitions.json"
OUT_OVERLAP = OUT_ROI / "week7_adaptive_upper_pen_roi_v3_mask_overlap_per_detection.csv"
OUT_FRAME_SELECTION = OUT_ROI / "week7_adaptive_upper_pen_roi_v3_frame_selection.csv"
OUT_ASSIGNMENT = OUT_ROI / "week7_bbox_roi_assignment_adaptive_upper_pen_v3.csv"
OUT_FRAME_SUMMARY = OUT_ROI / "week7_roi_frame_summary_adaptive_upper_pen_v3.csv"
OUT_REVIEW_TEMPLATE = OUT_ROI / "week7_adaptive_upper_pen_roi_v3_manual_review_template.csv"

CONTACT_DIR = OUT_VIS / "roi_v3_adaptive_upper_pen"
CONTACT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CONTACT = OUT_VIS / "week7_adaptive_upper_pen_roi_v3_selected_contact_sheet.jpg"
OUT_COMBINED = OUT_VIS / "week7_adaptive_upper_pen_roi_v3_candidate_review_sheet.jpg"
OUT_NOTE = OUT_NOTES / "week7_adaptive_upper_pen_roi_v3_mask_overlap_notes.md"

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


def polygon_mask(shape_hw, points):
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 1)
    return mask


def polygon_area(points):
    return float(cv2.contourArea(np.array(points, dtype=np.int32)))


def draw_polygon(img, points, color=(0, 220, 255), thickness=3):
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


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
    raise RuntimeError("Could not read image shape.")


def load_sam_mask(mask_path):
    p = resolve_path(mask_path)
    if p is None:
        return None

    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None

    return (m > 0).astype(np.uint8)


def make_candidate_contact_sheet(frames, assignment, candidate_name, points, output_path):
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

    inside_col = f"inside_{candidate_name}"

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

            inside = bool(d[inside_col])
            color = (0, 190, 0) if inside else (135, 135, 135)
            thickness = 2 if inside else 1

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

        cv2.putText(
            overlay,
            f"{sid} | {candidate_name}",
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
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


def make_selected_contact_sheet(frames, assignment, candidate_points, output_path):
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

        frame_dets = assignment[assignment["scan_frame_id"].astype(str) == sid].copy()

        if len(frame_dets):
            selected_name = str(frame_dets["selected_roi_candidate_v3"].iloc[0])
            points = candidate_points[selected_name]
            draw_polygon(overlay, points, color=(0, 220, 255), thickness=3)
        else:
            selected_name = "none"

        for _, d in frame_dets.iterrows():
            x1 = int(round(float(d["x1"])))
            y1 = int(round(float(d["y1"])))
            x2 = int(round(float(d["x2"])))
            y2 = int(round(float(d["y2"])))

            inside = bool(d["inside_adaptive_upper_pen_roi_v3"])
            color = (0, 190, 0) if inside else (135, 135, 135)
            thickness = 2 if inside else 1

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

            label = f"{'in' if inside else 'out'} {d.get('det_id', '')}"
            cv2.putText(
                overlay,
                label,
                (x1, max(17, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            overlay,
            f"{sid} | selected: {selected_name}",
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
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


def make_combined_sheet(contact_rows, output_path):
    imgs = []

    for _, row in contact_rows.iterrows():
        name = row["roi_name"]
        p = Path(row["contact_sheet_path"])

        if not p.exists():
            continue

        img = cv2.imread(str(p))
        if img is None:
            continue

        title_h = 42
        canvas = np.full((img.shape[0] + title_h, img.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(canvas, name, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 0, 0), 2)
        canvas[title_h:, :, :] = img
        imgs.append(canvas)

    if not imgs:
        return False

    width = min(i.shape[1] for i in imgs)
    resized = []

    for img in imgs:
        h = int(img.shape[0] * width / img.shape[1])
        resized.append(cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA))

    combined = np.vstack(resized)
    cv2.imwrite(str(output_path), combined)
    return True


# ---------------------------------------------------------------------
# Load tables.
# ---------------------------------------------------------------------
frames = pd.read_csv(FRAME_INDEX_PATH)
dets = pd.read_csv(DETS_PATH)
sam = pd.read_csv(SAM_FEATURES_PATH)

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

if "row_index" not in dets.columns:
    dets["row_index"] = np.arange(len(dets))

# Join SAM mask paths.
join_cols = []
for c in ["scan_frame_id", "det_id"]:
    if c in dets.columns and c in sam.columns:
        join_cols.append(c)

if len(join_cols) < 2:
    raise RuntimeError("Cannot join detections and Segment Anything Model features by scan_frame_id and det_id.")

sam_cols = join_cols + [c for c in ["mask_path", "sam_predicted_iou_score", "mask_area_pixels"] if c in sam.columns]
dets = dets.merge(sam[sam_cols], on=join_cols, how="left")

image_col = first_col(frames, ["frame_image_path", "image_path", "frame_path", "scanpoint_frame_path"])
if image_col is None:
    raise RuntimeError("No image path column in frame index.")

frame_w, frame_h = get_image_shape(frames, image_col)

# ---------------------------------------------------------------------
# ROI v3 candidate family.
# These are not right-only; they represent the upper annotated pen area with different lower boundaries.
# ---------------------------------------------------------------------
candidate_points = {
    "upper_pen_y360_v3": [[0, 0], [704, 0], [704, 360], [0, 360]],
    "upper_pen_y390_v3": [[0, 0], [704, 0], [704, 390], [0, 390]],
    "upper_pen_y420_v3": [[0, 0], [704, 0], [704, 420], [0, 420]],
    "upper_pen_fence_diagonal_y390_v3": [[0, 0], [704, 0], [704, 395], [175, 395], [130, 335], [0, 335]],
    "upper_pen_fence_diagonal_y420_v3": [[0, 0], [704, 0], [704, 420], [185, 420], [135, 350], [0, 350]],
    "upper_pen_fence_diagonal_y450_v3": [[0, 0], [704, 0], [704, 450], [215, 450], [155, 370], [0, 370]],
}

# Clip to actual frame size.
for name, points in list(candidate_points.items()):
    clipped = []
    for x, y in points:
        clipped.append([
            int(max(0, min(frame_w, x))),
            int(max(0, min(frame_h, y))),
        ])
    candidate_points[name] = clipped

# Candidate definitions.
defs = []
for name, points in candidate_points.items():
    area = polygon_area(points)
    defs.append({
        "roi_name": name,
        "roi_type": "polygon",
        "points_json": json.dumps(points),
        "frame_width": frame_w,
        "frame_height": frame_h,
        "area_pixels": area,
        "area_fraction_of_frame": area / float(frame_w * frame_h),
        "source": "adaptive upper-pen candidate; not fixed top-right",
        "interpretation": "Candidate for upper annotated pen. Detection assignment uses Segment Anything Model mask-overlap with this ROI.",
    })

defs_df = pd.DataFrame(defs)
safe_to_csv(defs_df, OUT_CANDIDATES)

with open(OUT_CANDIDATES_JSON, "w") as f:
    json.dump(defs_df.to_dict(orient="records"), f, indent=2)

# ---------------------------------------------------------------------
# Compute mask overlap per detection and ROI candidate.
# ---------------------------------------------------------------------
roi_masks = {}
for name, points in candidate_points.items():
    roi_masks[name] = polygon_mask((frame_h, frame_w), points)

records = []

mask_cache = {}

for _, row in dets.iterrows():
    sid = row["scan_frame_id"]
    det_id = row["det_id"]

    mask_path = row.get("mask_path", "")
    sam_mask = None

    if isinstance(mask_path, str) and mask_path:
        if mask_path in mask_cache:
            sam_mask = mask_cache[mask_path]
        else:
            sam_mask = load_sam_mask(mask_path)
            mask_cache[mask_path] = sam_mask

    if sam_mask is None:
        # Fallback to bbox rectangle mask if SAM mask is unavailable.
        sam_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        x1 = int(max(0, min(frame_w - 1, round(float(row["x1"])))))
        y1 = int(max(0, min(frame_h - 1, round(float(row["y1"])))))
        x2 = int(max(0, min(frame_w, round(float(row["x2"])))))
        y2 = int(max(0, min(frame_h, round(float(row["y2"])))))
        sam_mask[y1:y2, x1:x2] = 1
        mask_source = "bbox_fallback"
    else:
        mask_source = "sam_mask"

    mask_area = int(sam_mask.sum())

    cx = (float(row["x1"]) + float(row["x2"])) / 2.0
    cy = (float(row["y1"]) + float(row["y2"])) / 2.0

    rec = {
        "scan_frame_id": sid,
        "det_id": det_id,
        "row_index": row.get("row_index", ""),
        "x1": row["x1"],
        "y1": row["y1"],
        "x2": row["x2"],
        "y2": row["y2"],
        "bbox_center_x": cx,
        "bbox_center_y": cy,
        "mask_source": mask_source,
        "mask_area_pixels_for_overlap": mask_area,
    }

    for name, points in candidate_points.items():
        overlap = int((sam_mask & roi_masks[name]).sum())
        frac = overlap / mask_area if mask_area > 0 else 0.0
        center_inside = point_in_polygon(cx, cy, points)

        rec[f"{name}_mask_overlap_pixels"] = overlap
        rec[f"{name}_mask_overlap_fraction"] = frac
        rec[f"{name}_center_inside"] = center_inside

    records.append(rec)

overlap_df = pd.DataFrame(records)
safe_to_csv(overlap_df, OUT_OVERLAP)

# ---------------------------------------------------------------------
# Candidate-level assignment using overlap threshold.
# ---------------------------------------------------------------------
assignment = dets.copy()

# Add overlap columns.
key_cols = ["scan_frame_id", "det_id"]
assignment = assignment.merge(
    overlap_df,
    on=key_cols,
    how="left",
    suffixes=("", "_overlap"),
)

MASK_OVERLAP_THRESHOLD = 0.55

for name in candidate_points:
    assignment[f"inside_{name}"] = (
        pd.to_numeric(assignment[f"{name}_mask_overlap_fraction"], errors="coerce").fillna(0) >= MASK_OVERLAP_THRESHOLD
    )

# ---------------------------------------------------------------------
# Select best candidate per frame.
# Score balances:
# - near six pigs per frame,
# - not too many over/under counts,
# - avoid overly wide ROIs.
# ---------------------------------------------------------------------
frame_selection_rows = []

for sid, group in assignment.groupby("scan_frame_id"):
    candidate_rows = []

    for name, points in candidate_points.items():
        col = f"inside_{name}"
        inside_count = int(group[col].sum())
        outside_count = int((~group[col]).sum())

        # Pen labels are six per scanpoint; detector count can be imperfect.
        target = 6

        # Penalize over-selection more than slight under-selection because lower-pen leakage is harmful.
        over = max(0, inside_count - target)
        under = max(0, target - inside_count)

        area_frac = polygon_area(points) / float(frame_w * frame_h)

        score = (
            12.0 * over
            + 7.0 * under
            + 3.0 * abs(inside_count - target)
            + 5.0 * area_frac
        )

        candidate_rows.append({
            "scan_frame_id": sid,
            "roi_name": name,
            "inside_count": inside_count,
            "outside_count": outside_count,
            "area_fraction_of_frame": area_frac,
            "frame_candidate_score_lower_is_better": score,
        })

    cdf = pd.DataFrame(candidate_rows).sort_values("frame_candidate_score_lower_is_better")
    best = cdf.iloc[0].to_dict()
    frame_selection_rows.append(best)

frame_selection = pd.DataFrame(frame_selection_rows)
safe_to_csv(frame_selection, OUT_FRAME_SELECTION)

# Add selected ROI back to assignment.
selected_map = dict(zip(frame_selection["scan_frame_id"], frame_selection["roi_name"]))
assignment["selected_roi_candidate_v3"] = assignment["scan_frame_id"].map(selected_map)

assignment["selected_roi_mask_overlap_fraction_v3"] = assignment.apply(
    lambda r: r[f"{r['selected_roi_candidate_v3']}_mask_overlap_fraction"],
    axis=1,
)

assignment["selected_roi_center_inside_v3"] = assignment.apply(
    lambda r: r[f"{r['selected_roi_candidate_v3']}_center_inside"],
    axis=1,
)

assignment["inside_adaptive_upper_pen_roi_v3"] = (
    pd.to_numeric(assignment["selected_roi_mask_overlap_fraction_v3"], errors="coerce").fillna(0) >= MASK_OVERLAP_THRESHOLD
)

assignment["roi_use_for_gt_matching_v3"] = np.where(
    assignment["inside_adaptive_upper_pen_roi_v3"],
    "use_for_ground_truth_colour_and_behaviour_matching",
    "ignore_for_ground_truth_matching_outside_or_adjacent_pen",
)

safe_to_csv(assignment, OUT_ASSIGNMENT)

# ---------------------------------------------------------------------
# Frame summary.
# ---------------------------------------------------------------------
frame_rows = []

for sid, group in assignment.groupby("scan_frame_id"):
    inside = int(group["inside_adaptive_upper_pen_roi_v3"].sum())
    outside = int((~group["inside_adaptive_upper_pen_roi_v3"]).sum())

    frame_rows.append({
        "scan_frame_id": sid,
        "selected_roi_candidate_v3": str(group["selected_roi_candidate_v3"].iloc[0]),
        "detections": len(group),
        "inside_adaptive_upper_pen_roi_v3_count": inside,
        "outside_adaptive_upper_pen_roi_v3_count": outside,
        "inside_adaptive_upper_pen_roi_v3_fraction": inside / len(group) if len(group) else np.nan,
        "mean_selected_mask_overlap_fraction": float(pd.to_numeric(group["selected_roi_mask_overlap_fraction_v3"], errors="coerce").mean()),
    })

frame_summary = pd.DataFrame(frame_rows)
safe_to_csv(frame_summary, OUT_FRAME_SUMMARY)

# ---------------------------------------------------------------------
# Contact sheets.
# ---------------------------------------------------------------------
contact_rows = []
for name, points in candidate_points.items():
    p = CONTACT_DIR / f"{name}_contact_sheet.jpg"
    ok = make_candidate_contact_sheet(frames, assignment, name, points, p)
    contact_rows.append({
        "roi_name": name,
        "contact_sheet_path": str(p),
        "generated": ok,
    })

contact_df = pd.DataFrame(contact_rows)
CONTACT_INDEX = OUT_VIS / "week7_adaptive_upper_pen_roi_v3_contact_sheet_index.csv"
safe_to_csv(contact_df, CONTACT_INDEX)

combined_ok = make_combined_sheet(contact_df, OUT_COMBINED)
selected_ok = make_selected_contact_sheet(frames, assignment, candidate_points, OUT_CONTACT)

# ---------------------------------------------------------------------
# Review template.
# ---------------------------------------------------------------------
review = frame_selection.copy()
review["human_review_decision"] = "pending"
review["reviewer_notes"] = ""
review["recommended_next_action"] = (
    "Inspect selected contact sheet. If adaptive selection wrongly includes lower-pen pigs or excludes valid upper-pen pigs, override selected_roi_candidate_v3 for that frame."
)

safe_to_csv(review, OUT_REVIEW_TEMPLATE)

# ---------------------------------------------------------------------
# Update task tracker.
# ---------------------------------------------------------------------
if TASK_TRACKER_PATH.exists():
    tracker = pd.read_csv(TASK_TRACKER_PATH)
    mask = tracker["stage"].astype(str) == "2"

    if mask.any():
        tracker.loc[mask, "status"] = "adaptive_roi_v3_generated_needs_visual_confirmation"

    safe_to_csv(tracker, TASK_TRACKER_PATH)

# ---------------------------------------------------------------------
# Notes.
# ---------------------------------------------------------------------
candidate_global_summary = []
for name in candidate_points:
    col = f"inside_{name}"
    inside = int(assignment[col].sum())
    outside = int((~assignment[col]).sum())
    candidate_global_summary.append({
        "roi_name": name,
        "inside_count": inside,
        "outside_count": outside,
        "inside_fraction": inside / len(assignment),
        "mean_inside_per_frame": float(assignment.groupby("scan_frame_id")[col].sum().mean()),
    })

candidate_global_summary = pd.DataFrame(candidate_global_summary).sort_values("mean_inside_per_frame")

selected_inside = int(assignment["inside_adaptive_upper_pen_roi_v3"].sum())
selected_outside = int((~assignment["inside_adaptive_upper_pen_roi_v3"]).sum())

with open(OUT_NOTE, "w") as f:
    f.write("# Week 7 Adaptive Upper-Pen Region of Interest v3 with Segment Anything Model Mask Overlap\n\n")

    f.write("## Reason for v3\n\n")
    f.write(
        "The ground truth pen is not always located at a fixed top-right screen position. "
        "The correct generalization is to focus on the upper annotated pen region and to use mask-overlap rather than only bounding-box centre checks. "
        "This avoids including lower adjacent-pen pigs when their bounding boxes cross candidate boundaries.\n\n"
    )

    f.write("## Method\n\n")
    f.write(
        "Multiple upper-pen Region of Interest candidates were generated. "
        "For each detection, the Segment Anything Model mask was intersected with each candidate Region of Interest. "
        "A detection is treated as inside a candidate when at least 55 percent of its mask area lies inside the candidate. "
        "For each scanpoint frame, a candidate is selected using a score that favours about six inside pigs while penalizing over-selection.\n\n"
    )

    f.write("## Candidate global summary\n\n")
    f.write(candidate_global_summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Adaptive selected summary\n\n")
    f.write(f"- Total detections: `{len(assignment)}`\n")
    f.write(f"- Inside selected adaptive Region of Interest: `{selected_inside}`\n")
    f.write(f"- Outside selected adaptive Region of Interest: `{selected_outside}`\n")
    f.write(f"- Mean inside per frame: `{frame_summary['inside_adaptive_upper_pen_roi_v3_count'].mean():.3f}`\n")
    f.write(f"- Median inside per frame: `{frame_summary['inside_adaptive_upper_pen_roi_v3_count'].median():.3f}`\n\n")

    f.write("## Visual review files\n\n")
    f.write(f"- Selected adaptive contact sheet: `{OUT_CONTACT}`\n")
    f.write(f"- Candidate review sheet: `{OUT_COMBINED}`\n")
    f.write(f"- Candidate contact index: `{CONTACT_INDEX}`\n\n")

    f.write("## Important interpretation\n\n")
    f.write(
        "This v3 output should be visually reviewed before it is finalized. "
        "If it performs better than fixed v1/v2 polygons, it should supersede previous Region of Interest decisions.\n\n"
    )

    f.write("## Outputs\n\n")
    for p in [
        OUT_CANDIDATES,
        OUT_CANDIDATES_JSON,
        OUT_OVERLAP,
        OUT_FRAME_SELECTION,
        OUT_ASSIGNMENT,
        OUT_FRAME_SUMMARY,
        OUT_REVIEW_TEMPLATE,
        OUT_CONTACT,
        OUT_COMBINED,
    ]:
        f.write(f"- `{p}`\n")

print("Saved:")
print(OUT_CANDIDATES)
print(OUT_CANDIDATES_JSON)
print(OUT_OVERLAP)
print(OUT_FRAME_SELECTION)
print(OUT_ASSIGNMENT)
print(OUT_FRAME_SUMMARY)
print(OUT_REVIEW_TEMPLATE)
print(OUT_CONTACT)
print(OUT_COMBINED)
print(OUT_NOTE)

print()
print("=== Candidate global summary ===")
print(candidate_global_summary.to_string(index=False))

print()
print("=== Adaptive selected summary ===")
print({
    "total_detections": len(assignment),
    "inside_selected_adaptive_roi": selected_inside,
    "outside_selected_adaptive_roi": selected_outside,
    "mean_inside_per_frame": float(frame_summary["inside_adaptive_upper_pen_roi_v3_count"].mean()),
    "median_inside_per_frame": float(frame_summary["inside_adaptive_upper_pen_roi_v3_count"].median()),
    "selected_contact_sheet_generated": selected_ok,
    "combined_candidate_sheet_generated": combined_ok,
})
