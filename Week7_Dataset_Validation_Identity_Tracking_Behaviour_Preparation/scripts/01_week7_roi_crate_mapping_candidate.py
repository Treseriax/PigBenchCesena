from pathlib import Path
from datetime import datetime
import csv
import json
import re
import math

import cv2
import numpy as np
import pandas as pd


ROOT = Path.home() / "PigBench"

W6 = ROOT / "Week6_Unibo_Dataset_Validation"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

GT_PATH = W6 / "outputs" / "unified_ground_truth" / "week6_unified_ground_truth_v2_with_recommended_split_v2.csv"
FRAME_INDEX_PATH = W6 / "outputs" / "unified_ground_truth" / "week6_scanpoint_frame_index.csv"
DETS_PATH = W6 / "outputs" / "feature_extractors" / "week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv"

OUT_ROI = W7 / "outputs" / "roi_and_crate_mapping"
OUT_STATS = W7 / "outputs" / "dataset_statistics"
OUT_VIS = W7 / "outputs" / "visualizations"
OUT_NOTES = W7 / "notes"

for p in [OUT_ROI, OUT_STATS, OUT_VIS, OUT_NOTES]:
    p.mkdir(parents=True, exist_ok=True)


ROI_DEFINITIONS_CSV = OUT_ROI / "week7_roi_candidate_definitions.csv"
ROI_DEFINITIONS_JSON = OUT_ROI / "week7_roi_candidate_definitions.json"
BBOX_ASSIGNMENT_CSV = OUT_ROI / "week7_bbox_roi_assignment_candidate.csv"
FRAME_SUMMARY_CSV = OUT_ROI / "week7_roi_frame_summary_candidate.csv"
MANUAL_REVIEW_TEMPLATE_CSV = OUT_ROI / "week7_roi_manual_review_template.csv"
CRATE_VIDEO_SUMMARY_CSV = OUT_STATS / "week7_crate_pen_video_initial_summary.csv"
CONTACT_SHEET_PATH = OUT_VIS / "week7_roi_candidate_overlay_contact_sheet.jpg"
NOTE_PATH = OUT_NOTES / "week7_roi_crate_mapping_candidate_notes.md"
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


def extract_crate_token(text):
    text = str(text)

    patterns = [
        r"\bB\d+\b",
        r"_B\d+_",
        r"LC\d*_B\d+",
        r"TLC\d*",
        r"c\d{4}",
    ]

    found = []

    for pat in patterns:
        for m in re.findall(pat, text):
            found.append(m.strip("_"))

    return "|".join(sorted(set(found))) if found else "unknown"


def extract_hour_token(text):
    text = str(text)

    # Common patterns in the generated scanpoint image names.
    m = re.search(r"20\d{6}_(\d{2})(\d{2})(\d{2})", text)
    if m:
        return m.group(1)

    m = re.search(r"_(\d{2})(\d{2})(\d{2})", text)
    if m:
        return m.group(1)

    return "unknown"


def get_image_dimensions(path):
    path = resolve_path(path)

    if path is None:
        return None

    img = cv2.imread(str(path))

    if img is None:
        return None

    h, w = img.shape[:2]
    return w, h


def clip_roi(x1, y1, x2, y2, width, height):
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(1, min(width, round(x2))))
    y2 = int(max(1, min(height, round(y2))))

    if x2 <= x1:
        x2 = min(width, x1 + 1)

    if y2 <= y1:
        y2 = min(height, y1 + 1)

    return x1, y1, x2, y2


def bbox_center_inside(row, roi):
    cx = (float(row["x1"]) + float(row["x2"])) / 2.0
    cy = (float(row["y1"]) + float(row["y2"])) / 2.0

    return (
        cx >= roi["x1"]
        and cx <= roi["x2"]
        and cy >= roi["y1"]
        and cy <= roi["y2"]
    )


def draw_contact_sheet(frame_index, dets, roi_defs, output_path):
    sample = frame_index.copy()

    if "scan_frame_id" in sample.columns:
        sample = sample.drop_duplicates("scan_frame_id")

    sample = sample.sort_values("scan_frame_id") if "scan_frame_id" in sample.columns else sample

    if len(sample) > 16:
        idx = np.linspace(0, len(sample) - 1, 16).round().astype(int)
        sample = sample.iloc[idx]

    thumbs = []
    thumb_w, thumb_h = 380, 250

    image_col = first_col(sample, [
        "frame_image_path",
        "image_path",
        "frame_path",
        "scanpoint_frame_path",
    ])

    if image_col is None:
        return False

    for _, fr in sample.iterrows():
        img_path = resolve_path(fr[image_col])

        if img_path is None:
            continue

        img = cv2.imread(str(img_path))

        if img is None:
            continue

        h, w = img.shape[:2]

        sid = str(fr.get("scan_frame_id", ""))

        frame_dets = dets[dets["scan_frame_id"].astype(str) == sid].copy()

        overlay = img.copy()

        # Draw full frame ROI in blue and detection-extent ROI in yellow.
        for _, roi in roi_defs.iterrows():
            color = (255, 80, 0) if roi["roi_candidate_name"] == "global_full_frame_roi" else (0, 220, 255)
            thickness = 3 if roi["roi_candidate_name"] == "global_detection_extent_roi" else 2
            cv2.rectangle(
                overlay,
                (int(roi["x1"]), int(roi["y1"])),
                (int(roi["x2"]), int(roi["y2"])),
                color,
                thickness,
            )

        # Draw detections.
        for _, d in frame_dets.iterrows():
            x1 = int(round(float(d["x1"])))
            y1 = int(round(float(d["y1"])))
            x2 = int(round(float(d["x2"])))
            y2 = int(round(float(d["y2"])))

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 190, 0), 2)

            label = f"det {d.get('det_id', '')}"
            cv2.putText(
                overlay,
                label,
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 190, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            overlay,
            sid,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
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
# Load data.
# ---------------------------------------------------------------------
gt = pd.read_csv(GT_PATH)
frames = pd.read_csv(FRAME_INDEX_PATH)
dets = pd.read_csv(DETS_PATH)

if "scan_frame_id" not in dets.columns:
    raise RuntimeError("Detection table does not contain scan_frame_id.")

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

# If frame_image_path exists only after suffixing, normalize it.
if "frame_image_path" not in dets.columns and "frame_image_path_from_frame_index" in dets.columns:
    dets["frame_image_path"] = dets["frame_image_path_from_frame_index"]

x1_col = first_col(dets, ["x1", "bbox_x1", "xmin", "left"])
y1_col = first_col(dets, ["y1", "bbox_y1", "ymin", "top"])
x2_col = first_col(dets, ["x2", "bbox_x2", "xmax", "right"])
y2_col = first_col(dets, ["y2", "bbox_y2", "ymax", "bottom"])

if not all([x1_col, y1_col, x2_col, y2_col]):
    raise RuntimeError(f"Could not find bbox columns. Columns: {list(dets.columns)}")

# Normalize bbox column names.
dets["x1"] = pd.to_numeric(dets[x1_col], errors="coerce")
dets["y1"] = pd.to_numeric(dets[y1_col], errors="coerce")
dets["x2"] = pd.to_numeric(dets[x2_col], errors="coerce")
dets["y2"] = pd.to_numeric(dets[y2_col], errors="coerce")

dets = dets.dropna(subset=["x1", "y1", "x2", "y2"]).copy()

# Determine image dimensions from frame index.
image_col = first_col(frames, [
    "frame_image_path",
    "image_path",
    "frame_path",
    "scanpoint_frame_path",
])

dims = []

if image_col:
    for v in frames[image_col].head(72):
        d = get_image_dimensions(v)
        if d:
            dims.append(d)

if not dims:
    raise RuntimeError("Could not determine scanpoint frame image dimensions.")

dim_df = pd.DataFrame(dims, columns=["width", "height"])
width = int(dim_df["width"].mode().iloc[0])
height = int(dim_df["height"].mode().iloc[0])

# Use high-confidence detections for the detection extent candidate if scores exist.
extent_source = dets.copy()

if "score" in extent_source.columns:
    score_num = pd.to_numeric(extent_source["score"], errors="coerce")
    high_conf = extent_source[score_num >= 0.50].copy()

    if len(high_conf) >= 100:
        extent_source = high_conf

# Detection extent candidate with margin.
q_x1 = extent_source["x1"].quantile(0.01)
q_y1 = extent_source["y1"].quantile(0.01)
q_x2 = extent_source["x2"].quantile(0.99)
q_y2 = extent_source["y2"].quantile(0.99)

margin_x = 0.08 * width
margin_y = 0.08 * height

det_x1, det_y1, det_x2, det_y2 = clip_roi(
    q_x1 - margin_x,
    q_y1 - margin_y,
    q_x2 + margin_x,
    q_y2 + margin_y,
    width,
    height,
)

full_x1, full_y1, full_x2, full_y2 = 0, 0, width, height

roi_defs = pd.DataFrame([
    {
        "roi_candidate_name": "global_full_frame_roi",
        "roi_type": "rectangle",
        "x1": full_x1,
        "y1": full_y1,
        "x2": full_x2,
        "y2": full_y2,
        "frame_width": width,
        "frame_height": height,
        "source": "full frame default candidate",
        "interpretation": "Use only if the camera view corresponds to the annotated pen or crate.",
        "requires_human_confirmation": True,
    },
    {
        "roi_candidate_name": "global_detection_extent_roi",
        "roi_type": "rectangle",
        "x1": det_x1,
        "y1": det_y1,
        "x2": det_x2,
        "y2": det_y2,
        "frame_width": width,
        "frame_height": height,
        "source": "global detector extent with margin",
        "interpretation": "Useful as an animal-activity Region of Interest candidate, but it may not cover the full annotated pen.",
        "requires_human_confirmation": True,
    },
])

roi_defs["area_pixels"] = (roi_defs["x2"] - roi_defs["x1"]) * (roi_defs["y2"] - roi_defs["y1"])
roi_defs["area_fraction_of_frame"] = roi_defs["area_pixels"] / float(width * height)

safe_to_csv(roi_defs, ROI_DEFINITIONS_CSV)

with open(ROI_DEFINITIONS_JSON, "w") as f:
    json.dump(roi_defs.to_dict(orient="records"), f, indent=2)

# Bbox assignments for each candidate.
assignment = dets.copy()
assignment["bbox_center_x"] = (assignment["x1"] + assignment["x2"]) / 2.0
assignment["bbox_center_y"] = (assignment["y1"] + assignment["y2"]) / 2.0

for _, roi in roi_defs.iterrows():
    name = roi["roi_candidate_name"]
    assignment[f"inside_{name}"] = assignment.apply(lambda r: bbox_center_inside(r, roi), axis=1)

safe_to_csv(assignment, BBOX_ASSIGNMENT_CSV)

# Frame summary.
summary_rows = []

for sid, group in assignment.groupby("scan_frame_id"):
    row = {
        "scan_frame_id": sid,
        "detections": len(group),
    }

    for _, roi in roi_defs.iterrows():
        name = roi["roi_candidate_name"]
        col = f"inside_{name}"
        row[f"{name}_inside_count"] = int(group[col].sum())
        row[f"{name}_outside_count"] = int((~group[col]).sum())
        row[f"{name}_inside_fraction"] = float(group[col].mean()) if len(group) else np.nan

    summary_rows.append(row)

frame_summary = pd.DataFrame(summary_rows)
safe_to_csv(frame_summary, FRAME_SUMMARY_CSV)

# Crate / pen / video initial summary.
meta = frames.copy()

if "video_id" not in meta.columns:
    meta["video_id"] = "unknown"

image_text = ""

for c in ["frame_image_path", "image_path", "frame_path", "scanpoint_frame_path", "video_id"]:
    if c in meta.columns:
        image_text = meta[c].astype(str)
        break

meta["initial_crate_or_camera_token"] = image_text.apply(extract_crate_token)
meta["initial_hour_token"] = image_text.apply(extract_hour_token)

crate_summary = meta.groupby("initial_crate_or_camera_token").agg(
    scanpoint_frames=("scan_frame_id", "nunique"),
    videos=("video_id", "nunique"),
).reset_index()

hour_summary = meta.groupby("initial_hour_token").agg(
    scanpoint_frames=("scan_frame_id", "nunique"),
    videos=("video_id", "nunique"),
).reset_index()

crate_summary["summary_type"] = "crate_or_camera_token"
hour_summary = hour_summary.rename(columns={"initial_hour_token": "initial_crate_or_camera_token"})
hour_summary["summary_type"] = "hour_token"

crate_video_summary = pd.concat([crate_summary, hour_summary], ignore_index=True)
safe_to_csv(crate_video_summary, CRATE_VIDEO_SUMMARY_CSV)

# Manual review template.
manual_template = roi_defs.copy()
manual_template["human_decision"] = "pending"
manual_template["adjusted_x1"] = manual_template["x1"]
manual_template["adjusted_y1"] = manual_template["y1"]
manual_template["adjusted_x2"] = manual_template["x2"]
manual_template["adjusted_y2"] = manual_template["y2"]
manual_template["reviewer_notes"] = ""
manual_template["recommended_next_action"] = (
    "Inspect week7_roi_candidate_overlay_contact_sheet.jpg and decide whether full-frame ROI, detection-extent ROI, or a manually adjusted ROI best matches the annotated pen."
)

safe_to_csv(manual_template, MANUAL_REVIEW_TEMPLATE_CSV)

# Contact sheet.
contact_ok = draw_contact_sheet(frames, assignment, roi_defs, CONTACT_SHEET_PATH)

# Update master task tracker.
if TASK_TRACKER_PATH.exists():
    tracker = pd.read_csv(TASK_TRACKER_PATH)

    mask = tracker["stage"].astype(str) == "2"

    if mask.any():
        tracker.loc[mask, "status"] = "candidate_generated_needs_human_review"

    safe_to_csv(tracker, TASK_TRACKER_PATH)

# Notes.
inside_summary = []

for _, roi in roi_defs.iterrows():
    name = roi["roi_candidate_name"]
    col = f"inside_{name}"

    inside_summary.append({
        "roi_candidate_name": name,
        "inside_count": int(assignment[col].sum()),
        "outside_count": int((~assignment[col]).sum()),
        "inside_fraction": float(assignment[col].mean()),
    })

inside_summary_df = pd.DataFrame(inside_summary)

with open(NOTE_PATH, "w") as f:
    f.write("# Week 7 Region of Interest and Crate Mapping Candidate\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This step creates initial Region of Interest candidates for the annotated pen or crate. "
        "The candidates are not automatically treated as final because the annotated Region of Interest is a semantic ground-truth decision, not only a pixel-level inference.\n\n"
    )

    f.write("## Region of Interest candidates\n\n")
    f.write(roi_defs.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Detection assignment summary\n\n")
    f.write(inside_summary_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Crate / camera / hour initial summary\n\n")
    f.write(crate_video_summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Contact sheet\n\n")
    f.write(f"- `{CONTACT_SHEET_PATH}`\n")
    f.write(f"- Contact sheet generated: `{contact_ok}`\n\n")

    f.write("## Required review\n\n")
    f.write(
        "Please inspect the contact sheet and decide which Region of Interest candidate best corresponds to the annotated pen. "
        "If neither candidate is correct, manually adjust the coordinates in the manual review template.\n\n"
    )

    f.write("## Outputs\n\n")
    for p in [
        ROI_DEFINITIONS_CSV,
        ROI_DEFINITIONS_JSON,
        BBOX_ASSIGNMENT_CSV,
        FRAME_SUMMARY_CSV,
        MANUAL_REVIEW_TEMPLATE_CSV,
        CRATE_VIDEO_SUMMARY_CSV,
        CONTACT_SHEET_PATH,
    ]:
        f.write(f"- `{p}`\n")

print("Saved:")
print(ROI_DEFINITIONS_CSV)
print(ROI_DEFINITIONS_JSON)
print(BBOX_ASSIGNMENT_CSV)
print(FRAME_SUMMARY_CSV)
print(MANUAL_REVIEW_TEMPLATE_CSV)
print(CRATE_VIDEO_SUMMARY_CSV)
print(CONTACT_SHEET_PATH)
print(NOTE_PATH)

print()
print("=== ROI candidates ===")
print(roi_defs.to_string(index=False))

print()
print("=== Detection assignment summary ===")
print(inside_summary_df.to_string(index=False))

print()
print("=== Contact sheet generated ===")
print(contact_ok)
