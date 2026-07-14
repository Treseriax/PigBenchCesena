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

ROI_V2_DEFS = OUT_ROI / "week7_refined_gt_pen_roi_v2_candidate_definitions.csv"
ROI_V2_DEFS_JSON = OUT_ROI / "week7_refined_gt_pen_roi_v2_candidate_definitions.json"
ROI_V2_ASSIGNMENT = OUT_ROI / "week7_bbox_roi_assignment_refined_candidates_v2.csv"
ROI_V2_FRAME_SUMMARY = OUT_ROI / "week7_roi_frame_summary_refined_candidates_v2.csv"
ROI_V2_RANKING = OUT_ROI / "week7_refined_gt_pen_roi_v2_candidate_ranking.csv"
ROI_V2_REVIEW_TEMPLATE = OUT_ROI / "week7_refined_gt_pen_roi_v2_manual_review_template.csv"

CONTACT_DIR = OUT_VIS / "roi_v2_candidates"
CONTACT_DIR.mkdir(parents=True, exist_ok=True)

NOTE_PATH = OUT_NOTES / "week7_refined_gt_pen_roi_v2_candidate_notes.md"
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


def make_contact_sheet(frames, assignment, points, roi_name, output_path):
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

    inside_col = f"inside_{roi_name}"

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
            0.55,
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


def make_combined_sheet(contact_paths, output_path):
    imgs = []

    for name, p in contact_paths:
        p = Path(p)
        if not p.exists():
            continue

        img = cv2.imread(str(p))

        if img is None:
            continue

        title_h = 44
        canvas = np.full((img.shape[0] + title_h, img.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(canvas, name, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
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
# Candidate polygons.
# Each candidate attempts to include the upper/right annotated pen while excluding lower adjacent pen.
# ---------------------------------------------------------------------
candidate_points = {
    "strict_upper_gt_pen_roi_v1_reference": [
        [140, 0],
        [704, 0],
        [704, 355],
        [170, 355],
        [170, 0],
    ],
    "balanced_upper_pen_roi_v2": [
        [125, 0],
        [704, 0],
        [704, 430],
        [185, 430],
        [165, 365],
        [125, 325],
        [125, 0],
    ],
    "fence_line_upper_pen_roi_v2": [
        [120, 0],
        [704, 0],
        [704, 400],
        [165, 400],
        [150, 340],
        [120, 310],
        [120, 0],
    ],
    "conservative_main_pen_roi_v2": [
        [135, 0],
        [704, 0],
        [704, 385],
        [190, 385],
        [170, 335],
        [135, 315],
        [135, 0],
    ],
    "wide_top_right_pen_roi_v2": [
        [110, 0],
        [704, 0],
        [704, 455],
        [220, 455],
        [170, 375],
        [120, 325],
        [110, 0],
    ],
}

# Clip points to frame dimensions.
for name, pts in list(candidate_points.items()):
    clipped = []
    for x, y in pts:
        clipped.append([
            int(max(0, min(frame_w, x))),
            int(max(0, min(frame_h, y))),
        ])
    candidate_points[name] = clipped

# ---------------------------------------------------------------------
# Candidate definitions and assignments.
# ---------------------------------------------------------------------
defs = []

assignment = dets.copy()
assignment["bbox_center_x"] = (assignment["x1"] + assignment["x2"]) / 2.0
assignment["bbox_center_y"] = (assignment["y1"] + assignment["y2"]) / 2.0

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
        "source": "refined human-guided candidate after broad v1 included too many lower-pen pigs",
        "interpretation": "Candidate for annotated upper/right ground truth pen. Needs visual review.",
    })

    assignment[f"inside_{name}"] = assignment.apply(
        lambda r, pts=points: point_in_polygon(r["bbox_center_x"], r["bbox_center_y"], pts),
        axis=1,
    )

defs_df = pd.DataFrame(defs)
safe_to_csv(defs_df, ROI_V2_DEFS)

with open(ROI_V2_DEFS_JSON, "w") as f:
    json.dump(defs_df.to_dict(orient="records"), f, indent=2)

safe_to_csv(assignment, ROI_V2_ASSIGNMENT)

# ---------------------------------------------------------------------
# Frame summaries and candidate ranking.
# ---------------------------------------------------------------------
frame_rows = []

for sid, group in assignment.groupby("scan_frame_id"):
    row = {
        "scan_frame_id": sid,
        "detections": len(group),
    }

    for name in candidate_points:
        col = f"inside_{name}"
        row[f"{name}_inside_count"] = int(group[col].sum())
        row[f"{name}_outside_count"] = int((~group[col]).sum())
        row[f"{name}_inside_fraction"] = float(group[col].mean()) if len(group) else np.nan

    frame_rows.append(row)

frame_summary = pd.DataFrame(frame_rows)
safe_to_csv(frame_summary, ROI_V2_FRAME_SUMMARY)

target_total_inside = 432
target_mean_per_frame = 6.0

ranking_rows = []

for name in candidate_points:
    col = f"inside_{name}"
    counts = assignment.groupby("scan_frame_id")[col].sum()

    inside = int(assignment[col].sum())
    outside = int((~assignment[col]).sum())
    mean_per_frame = float(counts.mean())
    median_per_frame = float(counts.median())
    min_per_frame = int(counts.min())
    max_per_frame = int(counts.max())

    frames_under_4 = int((counts < 4).sum())
    frames_over_8 = int((counts > 8).sum())

    score = (
        abs(inside - target_total_inside)
        + 20 * abs(mean_per_frame - target_mean_per_frame)
        + 5 * frames_under_4
        + 5 * frames_over_8
    )

    ranking_rows.append({
        "roi_name": name,
        "inside_count": inside,
        "outside_count": outside,
        "inside_fraction": inside / len(assignment),
        "mean_inside_per_frame": mean_per_frame,
        "median_inside_per_frame": median_per_frame,
        "min_inside_per_frame": min_per_frame,
        "max_inside_per_frame": max_per_frame,
        "frames_under_4_inside": frames_under_4,
        "frames_over_8_inside": frames_over_8,
        "target_total_inside": target_total_inside,
        "target_mean_per_frame": target_mean_per_frame,
        "heuristic_score_lower_is_better": score,
        "needs_visual_review": True,
    })

ranking = pd.DataFrame(ranking_rows).sort_values("heuristic_score_lower_is_better")
safe_to_csv(ranking, ROI_V2_RANKING)

review = ranking.copy()
review["human_decision"] = "pending"
review["reviewer_notes"] = ""
review["recommended_next_action"] = (
    "Inspect contact sheet. Choose the candidate that keeps the annotated upper/right pen and excludes lower adjacent-pen pigs."
)
safe_to_csv(review, ROI_V2_REVIEW_TEMPLATE)

# ---------------------------------------------------------------------
# Contact sheets.
# ---------------------------------------------------------------------
contact_rows = []

for name, points in candidate_points.items():
    out_path = CONTACT_DIR / f"{name}_contact_sheet.jpg"
    ok = make_contact_sheet(frames, assignment, points, name, out_path)

    contact_rows.append({
        "roi_name": name,
        "contact_sheet_path": str(out_path),
        "generated": ok,
    })

contact_df = pd.DataFrame(contact_rows)
CONTACT_INDEX = OUT_VIS / "week7_refined_gt_pen_roi_v2_contact_sheet_index.csv"
safe_to_csv(contact_df, CONTACT_INDEX)

combined_path = OUT_VIS / "week7_refined_gt_pen_roi_v2_combined_review_sheet.jpg"
combined_ok = make_combined_sheet(
    [(r["roi_name"], r["contact_sheet_path"]) for _, r in contact_df.iterrows()],
    combined_path,
)

# ---------------------------------------------------------------------
# Update task tracker.
# ---------------------------------------------------------------------
if TASK_TRACKER_PATH.exists():
    tracker = pd.read_csv(TASK_TRACKER_PATH)
    mask = tracker["stage"].astype(str) == "2"

    if mask.any():
        tracker.loc[mask, "status"] = "roi_v2_candidates_generated_after_broad_roi_rejected"

    safe_to_csv(tracker, TASK_TRACKER_PATH)

# ---------------------------------------------------------------------
# Notes.
# ---------------------------------------------------------------------
with open(NOTE_PATH, "w") as f:
    f.write("# Week 7 Refined Ground Truth Pen Region of Interest v2 Candidates\n\n")

    f.write("## Reason for refinement\n\n")
    f.write(
        "The previous broad semantic Region of Interest included too many pigs from the lower adjacent pen. "
        "This is not acceptable for ground-truth colour or behaviour matching because the Week 7 objective is to focus on the annotated ground truth pen only.\n\n"
    )

    f.write("## Candidate ranking\n\n")
    f.write(ranking.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Candidate definitions\n\n")
    f.write(defs_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Visual review files\n\n")
    f.write(contact_df.to_markdown(index=False))
    f.write("\n\n")

    f.write(f"- Combined review sheet: `{combined_path}`\n")
    f.write(f"- Combined review sheet generated: `{combined_ok}`\n\n")

    f.write("## Required decision\n\n")
    f.write(
        "Select the candidate that best includes the annotated upper/right pen while excluding the lower adjacent pen. "
        "The heuristic ranking is useful, but visual review has priority.\n\n"
    )

    f.write("## Outputs\n\n")
    for p in [
        ROI_V2_DEFS,
        ROI_V2_DEFS_JSON,
        ROI_V2_ASSIGNMENT,
        ROI_V2_FRAME_SUMMARY,
        ROI_V2_RANKING,
        ROI_V2_REVIEW_TEMPLATE,
        CONTACT_INDEX,
        combined_path,
    ]:
        f.write(f"- `{p}`\n")

print("Saved:")
print(ROI_V2_DEFS)
print(ROI_V2_DEFS_JSON)
print(ROI_V2_ASSIGNMENT)
print(ROI_V2_FRAME_SUMMARY)
print(ROI_V2_RANKING)
print(ROI_V2_REVIEW_TEMPLATE)
print(CONTACT_INDEX)
print(combined_path)
print(NOTE_PATH)

print()
print("=== ROI v2 ranking ===")
print(ranking.to_string(index=False))

print()
print("=== Contact sheets ===")
print(contact_df.to_string(index=False))

print()
print("Combined sheet generated:", combined_ok)
