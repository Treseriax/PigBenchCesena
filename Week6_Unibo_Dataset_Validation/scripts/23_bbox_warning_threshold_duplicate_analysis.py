from pathlib import Path
import sys
import csv
import cv2
import json
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path.home() / "PigBench"
W6 = PROJECT_ROOT / "Week6_Unibo_Dataset_Validation"

OUT_STATS = W6 / "outputs/dataset_statistics"
OUT_GT = W6 / "outputs/unified_ground_truth"
OUT_FEAT = W6 / "outputs/feature_extractors"
OUT_VIS = W6 / "outputs/visual_label_check/bbox_warning_threshold_duplicate_analysis"
NOTES = W6 / "notes"

LOW = OUT_STATS / "week6_integrity_low_bbox_frames.csv"
HIGH = OUT_STATS / "week6_integrity_high_bbox_frames.csv"
FRAME_INDEX = OUT_GT / "week6_scanpoint_frame_index.csv"

CONFIG = PROJECT_ROOT / "detection/configs/yolov8/yolov8_s.py"
CHECKPOINT = PROJECT_ROOT / "detection/data/pretrained_weights/yolov8_pigs/yolov8_s.pth"

OUT_VIS.mkdir(parents=True, exist_ok=True)
OUT_FEAT.mkdir(parents=True, exist_ok=True)
OUT_STATS.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


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

    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def draw_boxes(image_path, dets, out_path, title):
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    for _, r in dets.iterrows():
        x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
        score = float(r["score"])
        det_id = int(r["det_id"])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            img,
            f"D{det_id} {score:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        img,
        title[:100],
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(out_path), img)
    return True


def safe_name(x):
    return (
        str(x)
        .replace("/", "_")
        .replace(" ", "_")
        .replace(":", "")
        .replace("-", "")
        .replace("T", "_")
    )


if not LOW.exists() or not HIGH.exists():
    raise FileNotFoundError("Integrity low/high bbox frame CSV files missing.")
if not FRAME_INDEX.exists():
    raise FileNotFoundError(FRAME_INDEX)

low = pd.read_csv(LOW)
high = pd.read_csv(HIGH)
frames = pd.read_csv(FRAME_INDEX)

low["warning_type"] = "low_bbox_count"
high["warning_type"] = "high_bbox_count"
warnings = pd.concat([low, high], ignore_index=True)

warnings = warnings.merge(frames, on="scan_frame_id", how="left")

assert CONFIG.exists(), CONFIG
assert CHECKPOINT.exists(), CHECKPOINT

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "detection"))

from mmdet.apis import init_detector, inference_detector

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = init_detector(str(CONFIG), str(CHECKPOINT), device=device)

raw_rows = []
threshold_rows = []
overlap_rows = []

thresholds = [0.10, 0.15, 0.25, 0.35, 0.50, 0.70]

for _, wr in warnings.iterrows():
    scan_frame_id = wr["scan_frame_id"]
    image_path = Path(str(wr["frame_image_path"]))
    warning_type = wr["warning_type"]

    if not image_path.exists():
        threshold_rows.append({
            "scan_frame_id": scan_frame_id,
            "warning_type": warning_type,
            "status": "missing_image",
        })
        continue

    result = inference_detector(model, str(image_path))
    pred = result.pred_instances

    bboxes = pred.bboxes.detach().cpu().numpy()
    scores = pred.scores.detach().cpu().numpy()
    labels = pred.labels.detach().cpu().numpy()

    for det_id, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
        if float(score) < 0.05:
            continue

        x1, y1, x2, y2 = [float(v) for v in bbox.tolist()]
        raw_rows.append({
            "scan_frame_id": scan_frame_id,
            "warning_type": warning_type,
            "image_path": str(image_path),
            "det_id": det_id,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "score": float(score),
            "class_id": int(label),
        })

    frame_raw = pd.DataFrame([r for r in raw_rows if r["scan_frame_id"] == scan_frame_id])

    for th in thresholds:
        kept = frame_raw[frame_raw["score"] >= th].copy()

        threshold_rows.append({
            "scan_frame_id": scan_frame_id,
            "warning_type": warning_type,
            "threshold": th,
            "bbox_count": len(kept),
            "max_score": float(kept["score"].max()) if len(kept) else "",
            "mean_score": float(kept["score"].mean()) if len(kept) else "",
            "frame_image_path": str(image_path),
            "video_id": wr.get("video_id", ""),
            "timestamp": wr.get("timestamp", ""),
            "video_match_status": wr.get("video_match_status", ""),
        })

        if th in [0.10, 0.25, 0.50]:
            out_img = OUT_VIS / f"{scan_frame_id}__{warning_type}__th{str(th).replace('.', '_')}.jpg"
            draw_boxes(
                image_path,
                kept,
                out_img,
                f"{scan_frame_id} {warning_type} threshold={th} boxes={len(kept)}",
            )

    # Duplicate / overlap analysis at threshold 0.25 and 0.50.
    for th in [0.25, 0.50]:
        kept = frame_raw[frame_raw["score"] >= th].copy().reset_index(drop=True)

        duplicate_pairs_05 = 0
        duplicate_pairs_07 = 0
        max_iou = 0.0

        for i in range(len(kept)):
            for j in range(i + 1, len(kept)):
                a = kept.loc[i, ["x1", "y1", "x2", "y2"]].astype(float).tolist()
                b = kept.loc[j, ["x1", "y1", "x2", "y2"]].astype(float).tolist()
                v = iou_xyxy(a, b)
                max_iou = max(max_iou, v)

                if v >= 0.50:
                    duplicate_pairs_05 += 1
                if v >= 0.70:
                    duplicate_pairs_07 += 1

        overlap_rows.append({
            "scan_frame_id": scan_frame_id,
            "warning_type": warning_type,
            "threshold": th,
            "bbox_count": len(kept),
            "duplicate_pairs_iou_ge_0_50": duplicate_pairs_05,
            "duplicate_pairs_iou_ge_0_70": duplicate_pairs_07,
            "max_pairwise_iou": max_iou,
        })

raw = pd.DataFrame(raw_rows)
threshold_df = pd.DataFrame(threshold_rows)
overlap_df = pd.DataFrame(overlap_rows)

raw_path = OUT_FEAT / "week6_bbox_warning_raw_detections_threshold005.csv"
threshold_path = OUT_STATS / "week6_bbox_warning_threshold_sensitivity.csv"
overlap_path = OUT_STATS / "week6_bbox_warning_duplicate_overlap_analysis.csv"

safe_to_csv(raw, raw_path)
safe_to_csv(threshold_df, threshold_path)
safe_to_csv(overlap_df, overlap_path)

# Summary.
if len(threshold_df):
    th_summary = (
        threshold_df.groupby(["warning_type", "threshold"])
        .agg(
            frame_count=("scan_frame_id", "count"),
            mean_bbox_count=("bbox_count", "mean"),
            min_bbox_count=("bbox_count", "min"),
            max_bbox_count=("bbox_count", "max"),
            frames_lt_6=("bbox_count", lambda s: int((s < 6).sum())),
            frames_gt_10=("bbox_count", lambda s: int((s > 10).sum())),
        )
        .reset_index()
    )
else:
    th_summary = pd.DataFrame()

th_summary_path = OUT_STATS / "week6_bbox_warning_threshold_sensitivity_summary.csv"
safe_to_csv(th_summary, th_summary_path)

if len(overlap_df):
    overlap_summary = (
        overlap_df.groupby(["warning_type", "threshold"])
        .agg(
            frame_count=("scan_frame_id", "count"),
            total_duplicate_pairs_iou_ge_0_50=("duplicate_pairs_iou_ge_0_50", "sum"),
            total_duplicate_pairs_iou_ge_0_70=("duplicate_pairs_iou_ge_0_70", "sum"),
            mean_max_pairwise_iou=("max_pairwise_iou", "mean"),
            max_pairwise_iou=("max_pairwise_iou", "max"),
        )
        .reset_index()
    )
else:
    overlap_summary = pd.DataFrame()

overlap_summary_path = OUT_STATS / "week6_bbox_warning_duplicate_overlap_summary.csv"
safe_to_csv(overlap_summary, overlap_summary_path)

note_path = NOTES / "week6_bbox_warning_threshold_duplicate_analysis_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Bbox Warning Threshold and Duplicate Analysis\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This controlled analysis reruns detector inference only on bbox-count warning frames and checks whether the warnings are sensitive to score threshold or caused by duplicate overlapping detections.\n\n"
    )

    f.write("## Outputs\n\n")
    f.write(f"- Raw detections >=0.05: `{raw_path}`\n")
    f.write(f"- Threshold sensitivity: `{threshold_path}`\n")
    f.write(f"- Threshold summary: `{th_summary_path}`\n")
    f.write(f"- Duplicate overlap analysis: `{overlap_path}`\n")
    f.write(f"- Duplicate overlap summary: `{overlap_summary_path}`\n")
    f.write(f"- Visualizations: `{OUT_VIS}`\n\n")

    f.write("## Threshold sensitivity summary\n\n")
    f.write(th_summary.to_markdown(index=False) if len(th_summary) else "No threshold summary.")
    f.write("\n\n")

    f.write("## Duplicate overlap summary\n\n")
    f.write(overlap_summary.to_markdown(index=False) if len(overlap_summary) else "No overlap summary.")
    f.write("\n\n")

    f.write("## Interpretation guide\n\n")
    f.write(
        "- If low bbox frames reach six boxes only at very low thresholds, extra boxes may be low-confidence and should not automatically be accepted.\n"
        "- If high bbox frames remain high even at 0.50 or 0.70, the issue may be duplicate/false-positive detections or multiple body-part boxes.\n"
        "- If many IoU>=0.50 duplicate pairs exist, post-filtering/NMS may help.\n"
        "- If duplicate overlap is low, high counts may come from distinct false positives rather than simple duplicates.\n"
    )

print("Saved:")
print(raw_path)
print(threshold_path)
print(th_summary_path)
print(overlap_path)
print(overlap_summary_path)
print(note_path)
print(OUT_VIS)

print()
print("=== Threshold sensitivity summary ===")
print(th_summary.to_string(index=False) if len(th_summary) else "No threshold summary.")

print()
print("=== Duplicate overlap summary ===")
print(overlap_summary.to_string(index=False) if len(overlap_summary) else "No overlap summary.")
