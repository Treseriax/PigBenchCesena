from pathlib import Path
import sys
import csv
import cv2
import json
import pandas as pd
import torch


PROJECT_ROOT = Path.home() / "PigBench"
W6 = PROJECT_ROOT / "Week6_Unibo_Dataset_Validation"

SCAN_FRAMES = W6 / "outputs/unified_ground_truth/week6_scanpoint_frame_index.csv"

OUT_FEAT = W6 / "outputs/feature_extractors"
OUT_VIS = W6 / "outputs/visual_label_check/yolov8s_all_scanpoint_detections"
NOTES = W6 / "notes"

OUT_FEAT.mkdir(parents=True, exist_ok=True)
OUT_VIS.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

CONFIG = PROJECT_ROOT / "detection/configs/yolov8/yolov8_s.py"
CHECKPOINT = PROJECT_ROOT / "detection/data/pretrained_weights/yolov8_pigs/yolov8_s.pth"

SCORE_THRESH = 0.25


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


def draw_boxes(image_path, detections, out_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    for _, r in detections.iterrows():
        x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
        score = float(r["score"])
        det_id = int(r["det_id"])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"D{det_id} {score:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 255, 0),
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


assert CONFIG.exists(), f"Missing config: {CONFIG}"
assert CHECKPOINT.exists(), f"Missing checkpoint: {CHECKPOINT}"
assert SCAN_FRAMES.exists(), f"Missing scan frame index: {SCAN_FRAMES}"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "detection"))

from mmdet.apis import init_detector, inference_detector

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Config:", CONFIG)
print("Checkpoint:", CHECKPOINT)
print("Device:", device)
print("Score threshold:", SCORE_THRESH)

model = init_detector(str(CONFIG), str(CHECKPOINT), device=device)

frames = pd.read_csv(SCAN_FRAMES)
frames = frames[frames["extraction_status"] == "ok"].copy()

all_det_rows = []
frame_summary_rows = []

for idx, row in frames.iterrows():
    scan_frame_id = row["scan_frame_id"]
    image_path = Path(row["frame_image_path"])

    if not image_path.exists():
        frame_summary_rows.append({
            "scan_frame_id": scan_frame_id,
            "image_path": str(image_path),
            "status": "missing_image",
            "detections_above_threshold": 0,
            "max_score": "",
            "visualization_path": "",
        })
        continue

    try:
        result = inference_detector(model, str(image_path))
        pred = result.pred_instances

        bboxes = pred.bboxes.detach().cpu().numpy()
        scores = pred.scores.detach().cpu().numpy()
        labels = pred.labels.detach().cpu().numpy()

        det_rows = []

        for det_id, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
            score = float(score)
            if score < SCORE_THRESH:
                continue

            x1, y1, x2, y2 = [float(v) for v in bbox.tolist()]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

            det_row = {
                "scan_frame_id": scan_frame_id,
                "video_id": row["video_id"],
                "video_path": row["video_path"],
                "timestamp": row["timestamp"],
                "timestamp_sec_in_video": row["timestamp_sec_in_video"],
                "frame_index": row["frame_index"],
                "frame_image_path": str(image_path),
                "det_id": det_id,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": (x1 + x2) / 2.0,
                "cy": (y1 + y2) / 2.0,
                "bbox_width": max(0.0, x2 - x1),
                "bbox_height": max(0.0, y2 - y1),
                "bbox_area": area,
                "score": score,
                "class_id": int(label),
                "detector": "mmdet_yolov8_s_pigbench",
                "score_threshold": SCORE_THRESH,
                "config": str(CONFIG),
                "checkpoint": str(CHECKPOINT),
                "video_match_status": row["video_match_status"],
                "video_mapping_confidence": row["video_mapping_confidence"],
            }
            det_rows.append(det_row)
            all_det_rows.append(det_row)

        det_df = pd.DataFrame(det_rows)

        out_img = OUT_VIS / f"{scan_frame_id}__{safe_name(row['timestamp'])}__detections.jpg"

        if len(det_df):
            draw_boxes(image_path, det_df, out_img)
            max_score = float(det_df["score"].max())
        else:
            img = cv2.imread(str(image_path))
            if img is not None:
                cv2.putText(
                    img,
                    "No detections above threshold",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imwrite(str(out_img), img)
            max_score = ""

        frame_summary_rows.append({
            "scan_frame_id": scan_frame_id,
            "image_path": str(image_path),
            "status": "ok",
            "detections_above_threshold": len(det_df),
            "max_score": max_score,
            "visualization_path": str(out_img),
            "video_id": row["video_id"],
            "timestamp": row["timestamp"],
            "frame_index": row["frame_index"],
            "video_match_status": row["video_match_status"],
        })

    except Exception as e:
        frame_summary_rows.append({
            "scan_frame_id": scan_frame_id,
            "image_path": str(image_path),
            "status": f"error: {type(e).__name__}: {e}",
            "detections_above_threshold": 0,
            "max_score": "",
            "visualization_path": "",
            "video_id": row["video_id"],
            "timestamp": row["timestamp"],
            "frame_index": row["frame_index"],
            "video_match_status": row["video_match_status"],
        })

detections = pd.DataFrame(all_det_rows)
frame_summary = pd.DataFrame(frame_summary_rows)

det_csv = OUT_FEAT / "week6_yolov8s_all_scanpoint_detections.csv"
det_json = OUT_FEAT / "week6_yolov8s_all_scanpoint_detections.json"
frame_summary_path = OUT_FEAT / "week6_yolov8s_all_scanpoint_frame_summary.csv"

safe_to_csv(detections, det_csv)
det_json.write_text(json.dumps(detections.to_dict(orient="records"), indent=2, ensure_ascii=False))
safe_to_csv(frame_summary, frame_summary_path)

if len(frame_summary):
    status_summary = (
        frame_summary.groupby("status")
        .agg(
            frame_count=("scan_frame_id", "count"),
            total_detections=("detections_above_threshold", "sum"),
            mean_detections_per_frame=("detections_above_threshold", "mean"),
            min_detections=("detections_above_threshold", "min"),
            max_detections=("detections_above_threshold", "max"),
        )
        .reset_index()
    )
else:
    status_summary = pd.DataFrame()

status_summary_path = OUT_FEAT / "week6_yolov8s_detection_status_summary.csv"
safe_to_csv(status_summary, status_summary_path)

if len(detections):
    detection_score_summary = pd.DataFrame([
        {
            "num_detections": len(detections),
            "mean_score": detections["score"].mean(),
            "median_score": detections["score"].median(),
            "min_score": detections["score"].min(),
            "max_score": detections["score"].max(),
            "mean_bbox_area": detections["bbox_area"].mean(),
            "median_bbox_area": detections["bbox_area"].median(),
        }
    ])
else:
    detection_score_summary = pd.DataFrame([
        {
            "num_detections": 0,
            "mean_score": "",
            "median_score": "",
            "min_score": "",
            "max_score": "",
            "mean_bbox_area": "",
            "median_bbox_area": "",
        }
    ])

score_summary_path = OUT_FEAT / "week6_yolov8s_detection_score_summary.csv"
safe_to_csv(detection_score_summary, score_summary_path)

note_path = NOTES / "week6_yolov8s_all_scanpoint_detection_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 YOLOv8-s All Scanpoint Detection Notes\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This step runs the PigBench YOLOv8-s detector on all 72 extracted manual scan-sampling frames. "
        "The resulting bounding boxes will be used for bbox visualization, crop-based features, colour-marker exploration, and later bbox-to-label association.\n\n"
    )

    f.write("## Detector\n\n")
    f.write(f"- Config: `{CONFIG}`\n")
    f.write(f"- Checkpoint: `{CHECKPOINT}`\n")
    f.write(f"- Device: `{device}`\n")
    f.write(f"- Score threshold: `{SCORE_THRESH}`\n\n")

    f.write("## Outputs\n\n")
    f.write(f"- Detections CSV: `{det_csv}`\n")
    f.write(f"- Frame summary CSV: `{frame_summary_path}`\n")
    f.write(f"- Detection visualizations: `{OUT_VIS}`\n\n")

    f.write("## Status summary\n\n")
    f.write(status_summary.to_markdown(index=False) if len(status_summary) else "No status summary.")
    f.write("\n\n")

    f.write("## Detection score summary\n\n")
    f.write(detection_score_summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "These detections provide pig bounding boxes for the scan-sampling frames, but they are not yet assigned to manual colour IDs. "
        "The next step is to combine detection bboxes with the manual colour/behaviour label panel for visual QC, then attempt crop-based colour-marker association.\n"
    )

print("Saved:")
print(det_csv)
print(det_json)
print(frame_summary_path)
print(status_summary_path)
print(score_summary_path)
print(note_path)
print(OUT_VIS)

print()
print("=== Detection status summary ===")
print(status_summary.to_string(index=False) if len(status_summary) else "No status summary.")

print()
print("=== Detection score summary ===")
print(detection_score_summary.to_string(index=False))

print()
print("=== First 20 detections ===")
print(detections.head(20).to_string(index=False) if len(detections) else "No detections.")
