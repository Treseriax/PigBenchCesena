from pathlib import Path
import csv
import math
import json
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
NOTES = W6 / "notes"

DETS_PATH = FEAT / "week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv"
FRAME_INDEX_PATH = GT / "week6_scanpoint_frame_index.csv"
MARKER_PATH = FEAT / "week6_crop_colour_marker_features.csv"
ASSIGN_PATH = FEAT / "week6_candidate_bbox_to_colour_assignments.csv"

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


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    denom = area_a + area_b - inter

    if denom <= 0:
        return 0.0

    return inter / denom


def resolve_image_path(p):
    if pd.isna(p):
        return None

    p = str(p)

    if not p:
        return None

    path = Path(p)

    if path.exists():
        return path

    # Some tables may store relative paths.
    candidate = W6 / p
    if candidate.exists():
        return candidate

    candidate = ROOT / p
    if candidate.exists():
        return candidate

    return None


dets = read_csv(DETS_PATH)
frames = read_csv(FRAME_INDEX_PATH)
markers = read_csv(MARKER_PATH)
assignments = read_csv(ASSIGN_PATH)

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

# Merge image paths from frame index.
if len(frames) and "scan_frame_id" in frames.columns:
    keep_cols = ["scan_frame_id"]
    for c in ["frame_image_path", "video_id", "timestamp", "timestamp_sec_in_video", "frame_index", "video_match_status"]:
        if c in frames.columns:
            keep_cols.append(c)

    keep_cols = list(dict.fromkeys(keep_cols))
    dets = dets.merge(frames[keep_cols], left_on=scan_col, right_on="scan_frame_id", how="left", suffixes=("", "_frame"))

# Image dimensions per frame.
frame_dims = {}

if CV2_AVAILABLE and "frame_image_path" in dets.columns:
    for sid, grp in dets.groupby(scan_col):
        img_path = resolve_image_path(grp["frame_image_path"].iloc[0])
        if img_path is not None:
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                frame_dims[sid] = (w, h)

# ------------------------------------------------------------
# 1) BBox geometry features
# ------------------------------------------------------------
geom_rows = []

for _, r in dets.iterrows():
    sid = r[scan_col]
    x1 = safe_float(r[x1_col])
    y1 = safe_float(r[y1_col])
    x2 = safe_float(r[x2_col])
    y2 = safe_float(r[y2_col])

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    cx = x1 + width / 2
    cy = y1 + height / 2
    aspect = width / height if height > 0 else np.nan

    img_w, img_h = frame_dims.get(sid, (np.nan, np.nan))
    frame_area = img_w * img_h if not pd.isna(img_w) and not pd.isna(img_h) else np.nan

    geom_rows.append({
        "scan_frame_id": sid,
        "det_id": r.get("det_id", r.get("detection_id", "")),
        "video_id": r.get("video_id", ""),
        "timestamp": r.get("timestamp", ""),
        "score": safe_float(r[score_col]) if score_col else np.nan,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "bbox_width": width,
        "bbox_height": height,
        "bbox_area": area,
        "bbox_aspect_ratio": aspect,
        "bbox_center_x": cx,
        "bbox_center_y": cy,
        "image_width": img_w,
        "image_height": img_h,
        "bbox_area_fraction": area / frame_area if frame_area and frame_area > 0 else np.nan,
        "center_x_norm": cx / img_w if img_w and img_w > 0 else np.nan,
        "center_y_norm": cy / img_h if img_h and img_h > 0 else np.nan,
        "width_norm": width / img_w if img_w and img_w > 0 else np.nan,
        "height_norm": height / img_h if img_h and img_h > 0 else np.nan,
        "bbox_count_issue_type": r.get("bbox_count_issue_type", ""),
    })

geom = pd.DataFrame(geom_rows)

geom_path = FEAT / "week6_bbox_geometry_features.csv"
safe_to_csv(geom, geom_path)

# ------------------------------------------------------------
# 2) Group-spatial features per frame
# ------------------------------------------------------------
group_rows = []

for sid, g in geom.groupby("scan_frame_id"):
    boxes = g[["x1", "y1", "x2", "y2"]].values.astype(float)
    centers = g[["bbox_center_x", "bbox_center_y"]].values.astype(float)

    img_w = g["image_width"].dropna().iloc[0] if g["image_width"].notna().any() else np.nan
    img_h = g["image_height"].dropna().iloc[0] if g["image_height"].notna().any() else np.nan
    diag = math.sqrt(img_w ** 2 + img_h ** 2) if not pd.isna(img_w) and not pd.isna(img_h) else np.nan
    frame_area = img_w * img_h if not pd.isna(img_w) and not pd.isna(img_h) else np.nan

    pairwise_dists = []
    pairwise_ious = []

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            pairwise_dists.append(d / diag if diag and diag > 0 else d)
            pairwise_ious.append(iou_xyxy(boxes[i], boxes[j]))

    # Coarse spatial bins.
    left = int((g["center_x_norm"] < 1/3).sum()) if "center_x_norm" in g else 0
    middle_x = int(((g["center_x_norm"] >= 1/3) & (g["center_x_norm"] < 2/3)).sum()) if "center_x_norm" in g else 0
    right = int((g["center_x_norm"] >= 2/3).sum()) if "center_x_norm" in g else 0

    top = int((g["center_y_norm"] < 1/3).sum()) if "center_y_norm" in g else 0
    middle_y = int(((g["center_y_norm"] >= 1/3) & (g["center_y_norm"] < 2/3)).sum()) if "center_y_norm" in g else 0
    bottom = int((g["center_y_norm"] >= 2/3).sum()) if "center_y_norm" in g else 0

    group_rows.append({
        "scan_frame_id": sid,
        "video_id": g["video_id"].iloc[0] if "video_id" in g else "",
        "timestamp": g["timestamp"].iloc[0] if "timestamp" in g else "",
        "bbox_count": len(g),
        "mean_score": g["score"].mean(),
        "mean_bbox_area_fraction": g["bbox_area_fraction"].mean(),
        "sum_bbox_area_fraction": g["bbox_area_fraction"].sum(),
        "mean_bbox_aspect_ratio": g["bbox_aspect_ratio"].mean(),
        "std_center_x_norm": g["center_x_norm"].std(),
        "std_center_y_norm": g["center_y_norm"].std(),
        "mean_pairwise_center_distance_norm": float(np.mean(pairwise_dists)) if pairwise_dists else np.nan,
        "min_pairwise_center_distance_norm": float(np.min(pairwise_dists)) if pairwise_dists else np.nan,
        "max_pairwise_iou": float(np.max(pairwise_ious)) if pairwise_ious else np.nan,
        "mean_pairwise_iou": float(np.mean(pairwise_ious)) if pairwise_ious else np.nan,
        "left_roi_count": left,
        "middle_x_roi_count": middle_x,
        "right_roi_count": right,
        "top_roi_count": top,
        "middle_y_roi_count": middle_y,
        "bottom_roi_count": bottom,
        "bbox_count_issue_type": g["bbox_count_issue_type"].iloc[0] if "bbox_count_issue_type" in g else "",
    })

group_spatial = pd.DataFrame(group_rows)
group_spatial_path = FEAT / "week6_group_spatial_features_per_frame.csv"
safe_to_csv(group_spatial, group_spatial_path)

# ------------------------------------------------------------
# 3) Coarse ROI/resource proxy features
# ------------------------------------------------------------
roi_rows = []

for _, r in geom.iterrows():
    cx = r["center_x_norm"]
    cy = r["center_y_norm"]

    if pd.isna(cx) or pd.isna(cy):
        roi_x = "unknown"
        roi_y = "unknown"
        coarse_roi = "unknown"
    else:
        if cx < 1/3:
            roi_x = "left"
        elif cx < 2/3:
            roi_x = "middle"
        else:
            roi_x = "right"

        if cy < 1/3:
            roi_y = "top"
        elif cy < 2/3:
            roi_y = "middle"
        else:
            roi_y = "bottom"

        coarse_roi = f"{roi_y}_{roi_x}"

    near_edge = False
    if not pd.isna(cx) and not pd.isna(cy):
        near_edge = bool(cx < 0.15 or cx > 0.85 or cy < 0.15 or cy > 0.85)

    central_zone = False
    if not pd.isna(cx) and not pd.isna(cy):
        central_zone = bool(0.33 <= cx <= 0.67 and 0.33 <= cy <= 0.67)

    roi_rows.append({
        "scan_frame_id": r["scan_frame_id"],
        "det_id": r["det_id"],
        "video_id": r.get("video_id", ""),
        "timestamp": r.get("timestamp", ""),
        "coarse_roi_x": roi_x,
        "coarse_roi_y": roi_y,
        "coarse_roi_cell": coarse_roi,
        "near_frame_edge_proxy": near_edge,
        "central_zone_proxy": central_zone,
        "center_x_norm": cx,
        "center_y_norm": cy,
        "bbox_area_fraction": r["bbox_area_fraction"],
        "note": "Coarse ROI proxy from image position; not manually mapped physical feeder/drinker/enrichment ROI.",
    })

roi = pd.DataFrame(roi_rows)
roi_path = FEAT / "week6_coarse_roi_resource_proxy_features.csv"
safe_to_csv(roi, roi_path)

roi_frame = (
    roi.groupby(["scan_frame_id", "coarse_roi_cell"])
    .size()
    .reset_index(name="bbox_count_in_roi")
    .pivot(index="scan_frame_id", columns="coarse_roi_cell", values="bbox_count_in_roi")
    .fillna(0)
    .reset_index()
)

roi_frame_path = FEAT / "week6_coarse_roi_resource_proxy_features_per_frame.csv"
safe_to_csv(roi_frame, roi_frame_path)

# ------------------------------------------------------------
# 4) Crop descriptor baseline features
# ------------------------------------------------------------
crop_rows = []

if CV2_AVAILABLE and "frame_image_path" in dets.columns:
    for _, r in dets.iterrows():
        sid = r[scan_col]
        img_path = resolve_image_path(r.get("frame_image_path", ""))

        x1 = int(round(safe_float(r[x1_col], 0)))
        y1 = int(round(safe_float(r[y1_col], 0)))
        x2 = int(round(safe_float(r[x2_col], 0)))
        y2 = int(round(safe_float(r[y2_col], 0)))

        base = {
            "scan_frame_id": sid,
            "det_id": r.get("det_id", r.get("detection_id", "")),
            "video_id": r.get("video_id", ""),
            "timestamp": r.get("timestamp", ""),
            "crop_available": False,
            "crop_width": max(0, x2 - x1),
            "crop_height": max(0, y2 - y1),
            "mean_b": np.nan,
            "mean_g": np.nan,
            "mean_r": np.nan,
            "std_b": np.nan,
            "std_g": np.nan,
            "std_r": np.nan,
            "mean_gray": np.nan,
            "std_gray": np.nan,
            "edge_density": np.nan,
        }

        if img_path is not None:
            img = cv2.imread(str(img_path))

            if img is not None:
                h, w = img.shape[:2]
                x1c = max(0, min(w - 1, x1))
                x2c = max(0, min(w, x2))
                y1c = max(0, min(h - 1, y1))
                y2c = max(0, min(h, y2))

                crop = img[y1c:y2c, x1c:x2c]

                if crop.size > 0:
                    mean_bgr = crop.reshape(-1, 3).mean(axis=0)
                    std_bgr = crop.reshape(-1, 3).std(axis=0)
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 80, 160)

                    base.update({
                        "crop_available": True,
                        "crop_width": int(crop.shape[1]),
                        "crop_height": int(crop.shape[0]),
                        "mean_b": float(mean_bgr[0]),
                        "mean_g": float(mean_bgr[1]),
                        "mean_r": float(mean_bgr[2]),
                        "std_b": float(std_bgr[0]),
                        "std_g": float(std_bgr[1]),
                        "std_r": float(std_bgr[2]),
                        "mean_gray": float(gray.mean()),
                        "std_gray": float(gray.std()),
                        "edge_density": float((edges > 0).mean()),
                    })

        crop_rows.append(base)
else:
    # Create empty-but-documented rows if cv2/image paths unavailable.
    for _, r in dets.iterrows():
        crop_rows.append({
            "scan_frame_id": r[scan_col],
            "det_id": r.get("det_id", r.get("detection_id", "")),
            "video_id": r.get("video_id", ""),
            "timestamp": r.get("timestamp", ""),
            "crop_available": False,
            "note": "cv2 or frame_image_path unavailable",
        })

crop_desc = pd.DataFrame(crop_rows)
crop_desc_path = FEAT / "week6_crop_descriptor_baseline_features.csv"
safe_to_csv(crop_desc, crop_desc_path)

# Merge marker evidence when possible for a combined descriptor table.
combined = crop_desc.copy()

if len(markers):
    merge_keys = [c for c in ["scan_frame_id", "det_id"] if c in combined.columns and c in markers.columns]

    if merge_keys:
        marker_keep = merge_keys + [
            c for c in markers.columns
            if c not in merge_keys and c in [
                "best_marker_colour",
                "marker_confidence",
                "green_score",
                "blue_score",
                "purple_score",
                "red_score",
            ]
        ]
        combined = combined.merge(markers[marker_keep], on=merge_keys, how="left")

combined_path = FEAT / "week6_crop_descriptor_plus_marker_features.csv"
safe_to_csv(combined, combined_path)

# ------------------------------------------------------------
# 5) Trajectory feasibility table
# ------------------------------------------------------------
trajectory_rows = []

trajectory_rows.append({
    "feature_family": "trajectory",
    "implementation_status": "not_final_identity_resolved",
    "tested_available_inputs": "scanpoint frames, detector bboxes, candidate marker assignments",
    "reason": "Stable bbox-to-pig identity across time is not validated. Manual labels are colour-level, but detector bboxes are not final identity-resolved.",
    "usable_now": "frame-level group motion proxies only",
    "recommended_future_step": "Add tracking/identity association, then compute per-pig displacement, speed, and interaction trajectories.",
})

if len(assignments):
    trajectory_rows.append({
        "feature_family": "candidate_colour_tracklets",
        "implementation_status": "candidate_only",
        "tested_available_inputs": f"{len(assignments)} candidate bbox-to-colour assignments",
        "reason": "Marker-based assignments exist only for medium/high visible colour markers and require visual/manual confirmation.",
        "usable_now": "QC and candidate association analysis",
        "recommended_future_step": "Validate candidate assignments, separate red_neck/red_tail, then build short tracklets.",
    })

trajectory = pd.DataFrame(trajectory_rows)
trajectory_path = FEAT / "week6_trajectory_feature_feasibility_report.csv"
safe_to_csv(trajectory, trajectory_path)

# ------------------------------------------------------------
# 6) Feature test summary
# ------------------------------------------------------------
summary_rows = [
    {
        "feature_test": "bbox_geometry",
        "status": "implemented",
        "output": str(geom_path),
        "record_count": len(geom),
        "interpretation": "Per-detection geometric features for bbox size, location, and normalized area.",
    },
    {
        "feature_test": "group_spatial",
        "status": "implemented",
        "output": str(group_spatial_path),
        "record_count": len(group_spatial),
        "interpretation": "Per-frame group distribution, pairwise distance, overlap, and spatial spread features.",
    },
    {
        "feature_test": "coarse_roi_resource_proxy",
        "status": "implemented",
        "output": str(roi_path),
        "record_count": len(roi),
        "interpretation": "Coarse image-position ROI proxy; not a manually mapped physical resource annotation.",
    },
    {
        "feature_test": "crop_descriptor_baseline",
        "status": "implemented",
        "output": str(crop_desc_path),
        "record_count": len(crop_desc),
        "interpretation": "Simple crop colour/brightness/edge descriptors for downstream comparison.",
    },
    {
        "feature_test": "crop_descriptor_plus_marker",
        "status": "implemented",
        "output": str(combined_path),
        "record_count": len(combined),
        "interpretation": "Crop descriptors plus marker evidence when available.",
    },
    {
        "feature_test": "trajectory",
        "status": "feasibility_documented",
        "output": str(trajectory_path),
        "record_count": len(trajectory),
        "interpretation": "Trajectory features are not final because stable identity association is not validated.",
    },
]

summary = pd.DataFrame(summary_rows)
summary_path = STATS / "week6_lightweight_feature_extractor_test_summary.csv"
safe_to_csv(summary, summary_path)

# ------------------------------------------------------------
# 7) Notes
# ------------------------------------------------------------
note_path = NOTES / "week6_lightweight_feature_extractor_tests_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Lightweight Feature Extractor Tests\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This step expands Task 5 beyond detector boxes and marker features. "
        "It creates lightweight, auditable feature extractor outputs for bbox geometry, group-spatial context, coarse ROI/resource proxies, crop descriptors, and trajectory feasibility.\n\n"
    )

    f.write("## Outputs\n\n")
    f.write(summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write("- BBox geometry features are useful for object-level size and location analysis.\n")
    f.write("- Group-spatial features summarize group distribution, overlap, and interaction/contact proxies at scanpoint-frame level.\n")
    f.write("- Coarse ROI/resource features are image-position proxies only; they are not manually calibrated feeder/drinker/enrichment zones.\n")
    f.write("- Crop descriptor baselines provide simple colour/brightness/edge features and can be compared with marker features.\n")
    f.write("- Final trajectory features are intentionally not claimed because stable identity-resolved tracking is not validated yet.\n\n")

    f.write("## Key counts\n\n")
    f.write(f"- BBox geometry rows: `{len(geom)}`\n")
    f.write(f"- Group-spatial frame rows: `{len(group_spatial)}`\n")
    f.write(f"- ROI proxy rows: `{len(roi)}`\n")
    f.write(f"- Crop descriptor rows: `{len(crop_desc)}`\n")
    f.write(f"- Crop descriptor rows with available crop: `{int(crop_desc.get('crop_available', pd.Series(dtype=bool)).sum()) if 'crop_available' in crop_desc.columns else 0}`\n")
    f.write(f"- Trajectory feasibility rows: `{len(trajectory)}`\n")

print("Saved:")
print(geom_path)
print(group_spatial_path)
print(roi_path)
print(roi_frame_path)
print(crop_desc_path)
print(combined_path)
print(trajectory_path)
print(summary_path)
print(note_path)

print()
print("=== Feature extractor test summary ===")
print(summary.to_string(index=False))
