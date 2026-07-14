from pathlib import Path
import sys
import csv
import cv2
import pandas as pd
import torch


PROJECT_ROOT = Path.home() / "PigBench"
W6 = PROJECT_ROOT / "Week6_Unibo_Dataset_Validation"

SCAN_FRAMES = W6 / "outputs/unified_ground_truth/week6_scanpoint_frame_index.csv"
OUT = W6 / "outputs/feature_extractors"
VIS = W6 / "outputs/visual_label_check/detector_smoke_test"

OUT.mkdir(parents=True, exist_ok=True)
VIS.mkdir(parents=True, exist_ok=True)

CONFIG = PROJECT_ROOT / "detection/configs/yolov8/yolov8_s.py"
CHECKPOINT = PROJECT_ROOT / "detection/data/pretrained_weights/yolov8_pigs/yolov8_s.pth"


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

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"pig {score:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), img)
    return True


status_rows = []

try:
    assert CONFIG.exists(), f"Missing config: {CONFIG}"
    assert CHECKPOINT.exists(), f"Missing checkpoint: {CHECKPOINT}"
    assert SCAN_FRAMES.exists(), f"Missing scan frame index: {SCAN_FRAMES}"

    print("Config:", CONFIG)
    print("Checkpoint:", CHECKPOINT)
    print("CUDA available:", torch.cuda.is_available())

    # Important for local PigBench imports.
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "detection"))

    from mmdet.apis import init_detector, inference_detector

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("Initializing detector on", device)
    model = init_detector(str(CONFIG), str(CHECKPOINT), device=device)

    frames = pd.read_csv(SCAN_FRAMES)
    frames = frames[frames["extraction_status"] == "ok"].copy()

    if len(frames) == 0:
        raise RuntimeError("No extracted scanpoint frames found.")

    # Pick first direct TLC frame for stable test.
    direct = frames[frames["video_match_status"] == "matched_tlc_hour_video"].copy()
    row = direct.iloc[0] if len(direct) else frames.iloc[0]

    image_path = Path(row["frame_image_path"])
    scan_frame_id = row["scan_frame_id"]

    print("Testing image:", image_path)

    result = inference_detector(model, str(image_path))

    pred = result.pred_instances

    bboxes = pred.bboxes.detach().cpu().numpy()
    scores = pred.scores.detach().cpu().numpy()
    labels = pred.labels.detach().cpu().numpy()

    rows = []

    for i, (bbox, score, label) in enumerate(zip(bboxes, scores, labels)):
        if float(score) < 0.25:
            continue

        x1, y1, x2, y2 = bbox.tolist()

        rows.append({
            "scan_frame_id": scan_frame_id,
            "image_path": str(image_path),
            "det_id": i,
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "score": float(score),
            "class_id": int(label),
            "detector": "mmdet_yolov8_s_pigbench",
            "config": str(CONFIG),
            "checkpoint": str(CHECKPOINT),
        })

    det = pd.DataFrame(rows)

    det_path = OUT / "yolov8s_smoke_test_detections.csv"
    safe_to_csv(det, det_path)

    out_img = VIS / f"{scan_frame_id}_yolov8s_smoke_test.jpg"

    if len(det):
        draw_ok = draw_boxes(image_path, det, out_img)
    else:
        img = cv2.imread(str(image_path))
        draw_ok = img is not None
        if draw_ok:
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

    status_rows.append({
        "status": "success",
        "device": device,
        "scan_frame_id": scan_frame_id,
        "image_path": str(image_path),
        "detections_above_0_25": len(det),
        "detection_csv": str(det_path),
        "visualization": str(out_img),
        "error": "",
    })

except Exception as e:
    status_rows.append({
        "status": "failed",
        "device": "",
        "scan_frame_id": "",
        "image_path": "",
        "detections_above_0_25": "",
        "detection_csv": "",
        "visualization": "",
        "error": f"{type(e).__name__}: {e}",
    })

status = pd.DataFrame(status_rows)

status_path = OUT / "yolov8s_smoke_test_status.csv"
safe_to_csv(status, status_path)

print()
print("=== Smoke test status ===")
print(status.to_string(index=False))

print()
print("Saved:")
print(status_path)
if status_rows[0]["status"] == "success":
    print(status_rows[0]["detection_csv"])
    print(status_rows[0]["visualization"])
