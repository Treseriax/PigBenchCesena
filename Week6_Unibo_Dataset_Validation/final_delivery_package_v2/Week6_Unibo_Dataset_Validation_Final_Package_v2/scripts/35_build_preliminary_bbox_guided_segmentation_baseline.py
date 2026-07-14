from pathlib import Path
import csv
import json
import math
import shutil
import subprocess
import importlib.util
import pandas as pd
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"

FEAT = W6 / "outputs/feature_extractors"
GT = W6 / "outputs/unified_ground_truth"
STATS = W6 / "outputs/dataset_statistics"
VIS = W6 / "outputs/visual_label_check"
NOTES = W6 / "notes"

DETS_PATH = FEAT / "week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv"
FRAME_INDEX_PATH = GT / "week6_scanpoint_frame_index.csv"

SEG_VIS_DIR = VIS / "preliminary_segmentation_baseline"
SEG_VIS_DIR.mkdir(parents=True, exist_ok=True)
FEAT.mkdir(parents=True, exist_ok=True)
STATS.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


def rel(path):
    try:
        return str(Path(path).relative_to(W6))
    except Exception:
        return str(path)


def package_available(name):
    return importlib.util.find_spec(name) is not None


def command_available(cmd):
    return shutil.which(cmd) is not None


def read_csv(path):
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def get_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def resolve_image_path(p):
    if pd.isna(p):
        return None

    p = str(p)

    if not p:
        return None

    path = Path(p)

    if path.exists():
        return path

    candidate = W6 / p
    if candidate.exists():
        return candidate

    candidate = ROOT / p
    if candidate.exists():
        return candidate

    return None


def make_contact_sheet(image_paths, out_path, thumb_w=360, max_cols=4):
    if not CV2_AVAILABLE or len(image_paths) == 0:
        return False

    thumbs = []

    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue

        h, w = img.shape[:2]
        scale = thumb_w / max(w, 1)
        thumb_h = int(h * scale)
        thumb = cv2.resize(img, (thumb_w, thumb_h))
        thumbs.append(thumb)

    if not thumbs:
        return False

    cols = min(max_cols, len(thumbs))
    rows = int(math.ceil(len(thumbs) / cols))
    max_h = max(t.shape[0] for t in thumbs)

    sheet = np.full((rows * max_h, cols * thumb_w, 3), 255, dtype=np.uint8)

    for i, thumb in enumerate(thumbs):
        r = i // cols
        c = i % cols
        y = r * max_h
        x = c * thumb_w
        sheet[y:y+thumb.shape[0], x:x+thumb.shape[1]] = thumb

    cv2.imwrite(str(out_path), sheet)
    return True


# ------------------------------------------------------------
# 1) Segmentation resource audit
# ------------------------------------------------------------
audit_rows = [
    {
        "resource": "opencv_cv2",
        "available": CV2_AVAILABLE,
        "interpretation": "Required for bbox-guided GrabCut/Otsu segmentation baseline.",
    },
    {
        "resource": "ultralytics",
        "available": package_available("ultralytics"),
        "interpretation": "Would be useful for YOLO-seg if installed and segmentation weights are available.",
    },
    {
        "resource": "segment_anything",
        "available": package_available("segment_anything"),
        "interpretation": "Would be useful for SAM baseline if installed and weights are available.",
    },
    {
        "resource": "sam2",
        "available": package_available("sam2"),
        "interpretation": "Would be useful for SAM2 baseline if installed and weights are available.",
    },
    {
        "resource": "detectron2",
        "available": package_available("detectron2"),
        "interpretation": "Would be useful for Mask R-CNN style instance segmentation if installed.",
    },
    {
        "resource": "ffmpeg",
        "available": command_available("ffmpeg"),
        "interpretation": "Useful for video export but not required for static segmentation overlays.",
    },
]

audit = pd.DataFrame(audit_rows)
audit_path = STATS / "week6_segmentation_resource_audit.csv"
safe_to_csv(audit, audit_path)

if not CV2_AVAILABLE:
    raise RuntimeError("cv2 is required for the preliminary segmentation baseline.")

dets = read_csv(DETS_PATH)
frames = read_csv(FRAME_INDEX_PATH)

if dets.empty:
    raise FileNotFoundError(DETS_PATH)

scan_col = get_col(dets, ["scan_frame_id", "frame_id"])
x1_col = get_col(dets, ["x1", "bbox_x1"])
y1_col = get_col(dets, ["y1", "bbox_y1"])
x2_col = get_col(dets, ["x2", "bbox_x2"])
y2_col = get_col(dets, ["y2", "bbox_y2"])
score_col = get_col(dets, ["score", "confidence"])

if not all([scan_col, x1_col, y1_col, x2_col, y2_col]):
    raise ValueError(f"Detection table missing bbox columns. Columns={list(dets.columns)}")

# Merge frame image paths.
if len(frames) and "scan_frame_id" in frames.columns:
    keep_cols = ["scan_frame_id"]
    for c in ["frame_image_path", "video_id", "timestamp", "timestamp_sec_in_video", "frame_index", "video_match_status"]:
        if c in frames.columns:
            keep_cols.append(c)

    keep_cols = list(dict.fromkeys(keep_cols))
    dets = dets.merge(frames[keep_cols], left_on=scan_col, right_on="scan_frame_id", how="left", suffixes=("", "_frame"))

# ------------------------------------------------------------
# 2) Per-bbox segmentation baseline
# ------------------------------------------------------------
seg_rows = []
overlay_images = []

# We generate one overlay per frame.
frame_overlay_cache = {}

for sid, g in dets.groupby(scan_col):
    frame_image_path = None

    if "frame_image_path" in g.columns:
        frame_image_path = resolve_image_path(g["frame_image_path"].iloc[0])

    if frame_image_path is None:
        for _, r in g.iterrows():
            seg_rows.append({
                "scan_frame_id": sid,
                "det_id": r.get("det_id", r.get("detection_id", "")),
                "segmentation_status": "missing_frame_image",
            })
        continue

    img = cv2.imread(str(frame_image_path))

    if img is None:
        for _, r in g.iterrows():
            seg_rows.append({
                "scan_frame_id": sid,
                "det_id": r.get("det_id", r.get("detection_id", "")),
                "segmentation_status": "image_read_failed",
            })
        continue

    h_img, w_img = img.shape[:2]
    overlay = img.copy()

    for _, r in g.iterrows():
        det_id = r.get("det_id", r.get("detection_id", ""))

        x1 = int(round(safe_float(r[x1_col], 0)))
        y1 = int(round(safe_float(r[y1_col], 0)))
        x2 = int(round(safe_float(r[x2_col], 0)))
        y2 = int(round(safe_float(r[y2_col], 0)))

        x1 = max(0, min(w_img - 1, x1))
        y1 = max(0, min(h_img - 1, y1))
        x2 = max(0, min(w_img, x2))
        y2 = max(0, min(h_img, y2))

        bbox_w = max(0, x2 - x1)
        bbox_h = max(0, y2 - y1)
        bbox_area = bbox_w * bbox_h

        base = {
            "scan_frame_id": sid,
            "det_id": det_id,
            "video_id": r.get("video_id", ""),
            "timestamp": r.get("timestamp", ""),
            "score": safe_float(r[score_col]) if score_col else np.nan,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "bbox_width": bbox_w,
            "bbox_height": bbox_h,
            "bbox_area": bbox_area,
            "segmentation_method": "bbox_guided_grabcut_with_otsu_fallback",
            "segmentation_status": "not_run",
            "mask_area_pixels": np.nan,
            "mask_area_fraction_of_bbox": np.nan,
            "mask_bbox_x1": np.nan,
            "mask_bbox_y1": np.nan,
            "mask_bbox_x2": np.nan,
            "mask_bbox_y2": np.nan,
            "mask_width": np.nan,
            "mask_height": np.nan,
            "mask_aspect_ratio": np.nan,
            "largest_contour_area": np.nan,
            "contour_perimeter": np.nan,
            "shape_extent": np.nan,
            "shape_solidity": np.nan,
            "orientation_angle_deg": np.nan,
            "mean_foreground_r": np.nan,
            "mean_foreground_g": np.nan,
            "mean_foreground_b": np.nan,
            "note": "Preliminary automatic mask from detector bbox crop; not manual segmentation GT.",
        }

        if bbox_w < 12 or bbox_h < 12:
            base["segmentation_status"] = "bbox_too_small"
            seg_rows.append(base)
            continue

        crop = img[y1:y2, x1:x2].copy()

        if crop.size == 0:
            base["segmentation_status"] = "empty_crop"
            seg_rows.append(base)
            continue

        crop_h, crop_w = crop.shape[:2]

        mask_fg = None
        status = "grabcut_success"

        try:
            gc_mask = np.zeros((crop_h, crop_w), np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            margin_x = max(2, int(crop_w * 0.06))
            margin_y = max(2, int(crop_h * 0.06))
            rect_w = max(2, crop_w - 2 * margin_x)
            rect_h = max(2, crop_h - 2 * margin_y)

            rect = (margin_x, margin_y, rect_w, rect_h)

            cv2.grabCut(crop, gc_mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)

            mask_fg = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                255,
                0,
            ).astype("uint8")

            # If GrabCut returns too little or too much foreground, fallback.
            frac = (mask_fg > 0).mean()

            if frac < 0.05 or frac > 0.95:
                status = "otsu_fallback_after_grabcut"
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Choose foreground polarity that is more central.
                center_patch = otsu[crop_h//4:3*crop_h//4, crop_w//4:3*crop_w//4]

                if center_patch.mean() < 127:
                    otsu = 255 - otsu

                mask_fg = otsu

        except Exception as e:
            status = f"otsu_fallback_exception_{type(e).__name__}"
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            center_patch = otsu[crop_h//4:3*crop_h//4, crop_w//4:3*crop_w//4]

            if center_patch.mean() < 127:
                otsu = 255 - otsu

            mask_fg = otsu

        # Morphological cleanup.
        kernel = np.ones((3, 3), np.uint8)
        mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            base["segmentation_status"] = "no_contour"
            seg_rows.append(base)
            continue

        largest = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(largest))
        perimeter = float(cv2.arcLength(largest, True))

        # Keep only largest connected component for features.
        clean_mask = np.zeros_like(mask_fg)
        cv2.drawContours(clean_mask, [largest], -1, 255, thickness=-1)

        mask_area = int((clean_mask > 0).sum())
        mx, my, mw, mh = cv2.boundingRect(largest)

        hull = cv2.convexHull(largest)
        hull_area = float(cv2.contourArea(hull))
        solidity = contour_area / hull_area if hull_area > 0 else np.nan
        extent = contour_area / float(mw * mh) if mw > 0 and mh > 0 else np.nan

        angle = np.nan

        if len(largest) >= 5:
            try:
                (_, _), (_, _), angle = cv2.fitEllipse(largest)
            except Exception:
                angle = np.nan

        fg_pixels = crop[clean_mask > 0]

        if len(fg_pixels):
            mean_bgr = fg_pixels.mean(axis=0)
            mean_b = float(mean_bgr[0])
            mean_g = float(mean_bgr[1])
            mean_r = float(mean_bgr[2])
        else:
            mean_b = mean_g = mean_r = np.nan

        # Overlay contour in original image coordinates.
        shifted_contour = largest + np.array([[[x1, y1]]])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.drawContours(overlay, [shifted_contour], -1, (0, 0, 255), 2)
        cv2.putText(
            overlay,
            str(det_id),
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        base.update({
            "segmentation_status": status,
            "mask_area_pixels": mask_area,
            "mask_area_fraction_of_bbox": mask_area / bbox_area if bbox_area > 0 else np.nan,
            "mask_bbox_x1": x1 + mx,
            "mask_bbox_y1": y1 + my,
            "mask_bbox_x2": x1 + mx + mw,
            "mask_bbox_y2": y1 + my + mh,
            "mask_width": mw,
            "mask_height": mh,
            "mask_aspect_ratio": mw / mh if mh > 0 else np.nan,
            "largest_contour_area": contour_area,
            "contour_perimeter": perimeter,
            "shape_extent": extent,
            "shape_solidity": solidity,
            "orientation_angle_deg": angle,
            "mean_foreground_r": mean_r,
            "mean_foreground_g": mean_g,
            "mean_foreground_b": mean_b,
        })

        seg_rows.append(base)

    # Save overlay.
    out_img = SEG_VIS_DIR / f"{sid}_preliminary_segmentation_overlay.jpg"
    cv2.imwrite(str(out_img), overlay)
    overlay_images.append(out_img)

seg = pd.DataFrame(seg_rows)

seg_features_path = FEAT / "week6_preliminary_bbox_guided_segmentation_features.csv"
safe_to_csv(seg, seg_features_path)

seg_json_path = FEAT / "week6_preliminary_bbox_guided_segmentation_features.json"

with open(seg_json_path, "w") as f:
    json.dump(
        {
            "dataset": "Week6_Unibo_Dataset_Validation",
            "version": "preliminary_bbox_guided_segmentation_baseline_v1",
            "method": "bbox_guided_grabcut_with_otsu_fallback",
            "record_count": len(seg),
            "important_note": "Automatic preliminary masks from detector bbox crops; not manual segmentation ground truth.",
            "records": seg.replace({np.nan: None}).to_dict(orient="records"),
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

# ------------------------------------------------------------
# 3) Frame summary
# ------------------------------------------------------------
numeric_cols = [
    "mask_area_fraction_of_bbox",
    "mask_aspect_ratio",
    "shape_extent",
    "shape_solidity",
    "orientation_angle_deg",
]

agg = {
    "det_id": "count",
    "segmentation_status": lambda s: ";".join(sorted(set(str(x) for x in s))),
}

for c in numeric_cols:
    if c in seg.columns:
        agg[c] = "mean"

frame_summary = (
    seg.groupby("scan_frame_id", dropna=False)
    .agg(agg)
    .reset_index()
    .rename(columns={
        "det_id": "segmented_bbox_count",
        "mask_area_fraction_of_bbox": "mean_mask_area_fraction_of_bbox",
        "mask_aspect_ratio": "mean_mask_aspect_ratio",
        "shape_extent": "mean_shape_extent",
        "shape_solidity": "mean_shape_solidity",
        "orientation_angle_deg": "mean_orientation_angle_deg",
    })
)

frame_summary_path = FEAT / "week6_preliminary_segmentation_frame_summary.csv"
safe_to_csv(frame_summary, frame_summary_path)

# ------------------------------------------------------------
# 4) Quality summary
# ------------------------------------------------------------
status_summary = (
    seg.groupby("segmentation_status")
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

success_statuses = [
    "grabcut_success",
    "otsu_fallback_after_grabcut",
]

success_count = int(seg["segmentation_status"].isin(success_statuses).sum())
total_count = int(len(seg))

quality_rows = [
    {
        "metric": "total_segmentation_records",
        "value": total_count,
        "interpretation": "One preliminary segmentation attempt per detector bbox.",
    },
    {
        "metric": "successful_or_fallback_masks",
        "value": success_count,
        "interpretation": "Masks with usable foreground contour from GrabCut or Otsu fallback.",
    },
    {
        "metric": "success_rate",
        "value": success_count / total_count if total_count else np.nan,
        "interpretation": "Approximate technical mask generation rate; not accuracy against manual segmentation.",
    },
    {
        "metric": "overlay_images",
        "value": len(overlay_images),
        "interpretation": "One preliminary segmentation overlay per scanpoint frame.",
    },
    {
        "metric": "manual_segmentation_gt_available",
        "value": False,
        "interpretation": "No manual segmentation GT is available in this pipeline.",
    },
]

quality = pd.DataFrame(quality_rows)
quality_path = STATS / "week6_preliminary_segmentation_quality_summary.csv"
safe_to_csv(quality, quality_path)

status_path = STATS / "week6_preliminary_segmentation_status_summary.csv"
safe_to_csv(status_summary, status_path)

# Contact sheet.
contact_sheet_path = VIS / "week6_preliminary_segmentation_contact_sheet.jpg"

# Use first 24 overlays for a manageable contact sheet.
make_contact_sheet(overlay_images[:24], contact_sheet_path)

# ------------------------------------------------------------
# 5) Notes
# ------------------------------------------------------------
note_path = NOTES / "week6_preliminary_segmentation_baseline_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Preliminary Segmentation Baseline\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This step addresses the segmentation requirement with a preliminary bbox-guided segmentation baseline. "
        "The method uses detector bbox crops and applies GrabCut with an Otsu fallback to estimate foreground masks.\n\n"
    )

    f.write("## Important scope note\n\n")
    f.write(
        "These masks are automatic preliminary segmentation outputs, not manual segmentation ground truth. "
        "They are intended for early shape/posture/contact/ROI feature exploration and visual QC.\n\n"
    )

    f.write("## Segmentation resource audit\n\n")
    f.write(audit.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Outputs\n\n")
    f.write(f"- Segmentation feature CSV: `{seg_features_path}`\n")
    f.write(f"- Segmentation feature JSON: `{seg_json_path}`\n")
    f.write(f"- Per-frame segmentation summary: `{frame_summary_path}`\n")
    f.write(f"- Quality summary: `{quality_path}`\n")
    f.write(f"- Status summary: `{status_path}`\n")
    f.write(f"- Overlay directory: `{SEG_VIS_DIR}`\n")
    f.write(f"- Contact sheet: `{contact_sheet_path}`\n\n")

    f.write("## Quality summary\n\n")
    f.write(quality.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Segmentation status summary\n\n")
    f.write(status_summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The segmentation baseline provides mask area, mask aspect ratio, contour perimeter, shape extent, solidity, orientation, and foreground colour statistics. "
        "These features may improve shape/posture/contact estimates compared with bbox-only features, but they require visual/manual validation before being treated as reliable labels.\n"
    )

print("Saved:")
print(audit_path)
print(seg_features_path)
print(seg_json_path)
print(frame_summary_path)
print(quality_path)
print(status_path)
print(SEG_VIS_DIR)
print(contact_sheet_path)
print(note_path)

print()
print("=== Segmentation quality summary ===")
print(quality.to_string(index=False))

print()
print("=== Segmentation status summary ===")
print(status_summary.to_string(index=False))

print()
print("overlay images:", len(overlay_images))
