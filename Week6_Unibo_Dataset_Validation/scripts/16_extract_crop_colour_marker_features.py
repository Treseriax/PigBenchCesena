from pathlib import Path
import csv
import json
import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path.home() / "PigBench"
W6 = PROJECT_ROOT / "Week6_Unibo_Dataset_Validation"

DETECTIONS = W6 / "outputs/feature_extractors/week6_yolov8s_all_scanpoint_detections.csv"
LABELS = W6 / "outputs/unified_ground_truth/week6_scanpoint_frame_labels_long.csv"

OUT_FEAT = W6 / "outputs/feature_extractors"
OUT_VIS = W6 / "outputs/visual_label_check/colour_marker_crop_debug"
NOTES = W6 / "notes"

OUT_FEAT.mkdir(parents=True, exist_ok=True)
OUT_VIS.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


def clip_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w - 1))
    y2 = max(0, min(int(round(y2)), h - 1))

    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)

    return x1, y1, x2, y2


def mask_stats(mask):
    mask_u8 = mask.astype(np.uint8)
    pixel_count = int(mask_u8.sum())
    total = int(mask_u8.size)
    ratio = float(pixel_count / total) if total else 0.0

    if pixel_count == 0:
        return {
            "pixel_count": 0,
            "pixel_ratio": 0.0,
            "largest_component_area": 0,
            "largest_component_ratio": 0.0,
        }

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    if num_labels <= 1:
        largest = pixel_count
    else:
        largest = int(stats[1:, cv2.CC_STAT_AREA].max())

    return {
        "pixel_count": pixel_count,
        "pixel_ratio": ratio,
        "largest_component_area": largest,
        "largest_component_ratio": float(largest / total) if total else 0.0,
    }


def colour_masks(crop_bgr):
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Conservative marker thresholds. These are candidate features, not final identity labels.
    green = ((h >= 35) & (h <= 90) & (s >= 55) & (v >= 40))
    blue = ((h >= 90) & (h <= 135) & (s >= 55) & (v >= 35))
    purple = ((h >= 125) & (h <= 165) & (s >= 45) & (v >= 35))
    red1 = ((h <= 12) & (s >= 55) & (v >= 35))
    red2 = ((h >= 170) & (s >= 55) & (v >= 35))
    red = red1 | red2

    return {
        "green": green,
        "blue": blue,
        "purple": purple,
        "red": red,
    }


def score_from_stats(stats):
    # Combine dense colour pixels and coherent blob size.
    return float(stats["pixel_ratio"] * 0.6 + stats["largest_component_ratio"] * 0.4)


def save_crop_debug(frame, bbox, crop, out_path, text):
    x1, y1, x2, y2 = bbox

    vis = frame.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(
        vis,
        text,
        (x1, max(20, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    crop_resized = cv2.resize(crop, (180, 140), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), crop_resized)


if not DETECTIONS.exists():
    raise FileNotFoundError(DETECTIONS)

dets = pd.read_csv(DETECTIONS)
labels = pd.read_csv(LABELS) if LABELS.exists() else pd.DataFrame()

rows = []

for idx, r in dets.iterrows():
    image_path = Path(r["frame_image_path"])
    scan_frame_id = r["scan_frame_id"]

    img = cv2.imread(str(image_path))
    if img is None:
        rows.append({
            **r.to_dict(),
            "crop_status": "frame_read_failed",
            "crop_width": "",
            "crop_height": "",
            "best_marker_colour": "",
            "best_marker_score": "",
            "second_marker_score": "",
            "marker_score_margin": "",
            "marker_confidence": "none",
        })
        continue

    h, w = img.shape[:2]
    x1, y1, x2, y2 = clip_bbox(r["x1"], r["y1"], r["x2"], r["y2"], w, h)

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        rows.append({
            **r.to_dict(),
            "crop_status": "empty_crop",
            "crop_width": "",
            "crop_height": "",
            "best_marker_colour": "",
            "best_marker_score": "",
            "second_marker_score": "",
            "marker_score_margin": "",
            "marker_confidence": "none",
        })
        continue

    masks = colour_masks(crop)

    feature = {
        **r.to_dict(),
        "crop_status": "ok",
        "crop_x1": x1,
        "crop_y1": y1,
        "crop_x2": x2,
        "crop_y2": y2,
        "crop_width": int(x2 - x1),
        "crop_height": int(y2 - y1),
    }

    scores = {}

    for colour, mask in masks.items():
        stats = mask_stats(mask)
        score = score_from_stats(stats)
        scores[colour] = score

        feature[f"{colour}_pixel_count"] = stats["pixel_count"]
        feature[f"{colour}_pixel_ratio"] = stats["pixel_ratio"]
        feature[f"{colour}_largest_component_area"] = stats["largest_component_area"]
        feature[f"{colour}_largest_component_ratio"] = stats["largest_component_ratio"]
        feature[f"{colour}_score"] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_colour, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_score - second_score

    if best_score >= 0.020 and margin >= 0.008:
        confidence = "high"
    elif best_score >= 0.010 and margin >= 0.004:
        confidence = "medium"
    elif best_score >= 0.005:
        confidence = "low"
    else:
        confidence = "none"

    feature["best_marker_colour"] = best_colour if confidence != "none" else "no_marker_detected"
    feature["best_marker_score"] = best_score
    feature["second_marker_score"] = second_score
    feature["marker_score_margin"] = margin
    feature["marker_confidence"] = confidence

    # Save selected crop debug images only: high/medium candidates and first few low.
    debug_needed = confidence in ["high", "medium"]
    if debug_needed:
        out_crop = OUT_VIS / f"{scan_frame_id}_det{int(r['det_id']):02d}_{feature['best_marker_colour']}_{confidence}.jpg"
        cv2.imwrite(str(out_crop), crop)
        feature["crop_debug_path"] = str(out_crop)
    else:
        feature["crop_debug_path"] = ""

    rows.append(feature)

features = pd.DataFrame(rows)

features_csv = OUT_FEAT / "week6_crop_colour_marker_features.csv"
features_json = OUT_FEAT / "week6_crop_colour_marker_features.json"

safe_to_csv(features, features_csv)
features_json.write_text(json.dumps(features.to_dict(orient="records"), indent=2, ensure_ascii=False))

# Summary by predicted marker.
if len(features):
    marker_summary = (
        features.groupby(["best_marker_colour", "marker_confidence"])
        .agg(
            detection_count=("det_id", "count"),
            mean_best_score=("best_marker_score", "mean"),
            mean_margin=("marker_score_margin", "mean"),
            mean_detection_score=("score", "mean"),
        )
        .reset_index()
        .sort_values(["marker_confidence", "detection_count"], ascending=[True, False])
    )
else:
    marker_summary = pd.DataFrame()

marker_summary_path = OUT_FEAT / "week6_crop_colour_marker_summary.csv"
safe_to_csv(marker_summary, marker_summary_path)

# Candidate assignment table: keep only medium/high marker evidence.
candidate = features[features["marker_confidence"].isin(["medium", "high"])].copy()

candidate["candidate_colour_id"] = candidate["best_marker_colour"]
candidate.loc[candidate["best_marker_colour"] == "red", "candidate_colour_id"] = "red_marker_ambiguous_red_neck_or_red_tail"

candidate["assignment_status"] = "candidate_marker_based"
candidate.loc[candidate["candidate_colour_id"].str.contains("ambiguous", na=False), "assignment_status"] = "ambiguous_red_marker_candidate"
candidate["assignment_note"] = (
    "Candidate bbox-to-colour assignment from HSV crop marker features. "
    "Needs visual/manual confirmation before use as ground truth."
)

candidate_cols = [
    "scan_frame_id",
    "video_id",
    "timestamp",
    "frame_index",
    "det_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "score",
    "candidate_colour_id",
    "best_marker_colour",
    "best_marker_score",
    "marker_score_margin",
    "marker_confidence",
    "assignment_status",
    "assignment_note",
    "frame_image_path",
    "crop_debug_path",
]

candidate = candidate[[c for c in candidate_cols if c in candidate.columns]]

candidate_path = OUT_FEAT / "week6_candidate_bbox_to_colour_assignments.csv"
safe_to_csv(candidate, candidate_path)

# Frame-level candidate counts.
if len(features):
    frame_candidate_summary = (
        features.groupby("scan_frame_id")
        .agg(
            detections=("det_id", "count"),
            high_marker_candidates=("marker_confidence", lambda s: int((s == "high").sum())),
            medium_marker_candidates=("marker_confidence", lambda s: int((s == "medium").sum())),
            low_marker_candidates=("marker_confidence", lambda s: int((s == "low").sum())),
            no_marker=("marker_confidence", lambda s: int((s == "none").sum())),
        )
        .reset_index()
    )
else:
    frame_candidate_summary = pd.DataFrame()

frame_candidate_summary_path = OUT_FEAT / "week6_crop_colour_marker_frame_summary.csv"
safe_to_csv(frame_candidate_summary, frame_candidate_summary_path)

note_path = NOTES / "week6_crop_colour_marker_feature_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Crop Colour Marker Feature Notes\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This step scores each YOLOv8-s pig detection crop for visible colour-marker evidence. "
        "The goal is to explore candidate bbox-to-colour-ID association, not to create final identity ground truth.\n\n"
    )

    f.write("## Inputs\n\n")
    f.write(f"- Detections: `{DETECTIONS}`\n")
    f.write(f"- Manual labels: `{LABELS}`\n\n")

    f.write("## Outputs\n\n")
    f.write(f"- Crop marker features: `{features_csv}`\n")
    f.write(f"- Candidate bbox-to-colour assignments: `{candidate_path}`\n")
    f.write(f"- Marker summary: `{marker_summary_path}`\n")
    f.write(f"- Frame summary: `{frame_candidate_summary_path}`\n")
    f.write(f"- Debug crops: `{OUT_VIS}`\n\n")

    f.write("## Marker summary\n\n")
    f.write(marker_summary.to_markdown(index=False) if len(marker_summary) else "No marker summary.")
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "Green, blue, purple, and red markers are detected as colour evidence inside each bbox crop. "
        "Red evidence is ambiguous because the Excel labels contain both red_neck and red_tail. "
        "The no_color pig cannot be identified by a colour marker. "
        "Therefore these outputs should be used as candidate assignments and visual QC aids rather than final ground truth.\n"
    )

print("Saved:")
print(features_csv)
print(features_json)
print(marker_summary_path)
print(candidate_path)
print(frame_candidate_summary_path)
print(note_path)
print(OUT_VIS)

print()
print("=== Marker summary ===")
print(marker_summary.to_string(index=False) if len(marker_summary) else "No marker summary.")

print()
print("=== Candidate assignment count ===")
print(len(candidate))

print()
print("=== First 30 candidate assignments ===")
print(candidate.head(30).to_string(index=False) if len(candidate) else "No candidate assignments.")
