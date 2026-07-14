from pathlib import Path
import csv
import json
import traceback

import cv2
import numpy as np
import pandas as pd
import torch

from segment_anything import sam_model_registry, SamPredictor


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"

FEAT = W6 / "outputs" / "feature_extractors"
GT = W6 / "outputs" / "unified_ground_truth"
STATS = W6 / "outputs" / "dataset_statistics"
VIS = W6 / "outputs" / "visual_label_check"
NOTES = W6 / "notes"

DETS_PATH = FEAT / "week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv"
FRAME_INDEX_PATH = GT / "week6_scanpoint_frame_index.csv"

SAM_CHECKPOINT = Path("/work/models/SAM/sam_vit_b_01ec64.pth")
SAM_MODEL_TYPE = "vit_b"
DEVICE = "cpu"

OUT_DIR = VIS / "sam_box_prompt_smoke_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)
STATS.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)


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
        ROOT / value,
        Path.home() / value,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_detections():
    dets = pd.read_csv(DETS_PATH)

    if FRAME_INDEX_PATH.exists():
        frames = pd.read_csv(FRAME_INDEX_PATH)

        if "scan_frame_id" in dets.columns and "scan_frame_id" in frames.columns:
            frame_cols = ["scan_frame_id"]

            for col in [
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
                if col in frames.columns:
                    frame_cols.append(col)

            frame_cols = list(dict.fromkeys(frame_cols))

            dets = dets.merge(
                frames[frame_cols],
                on="scan_frame_id",
                how="left",
                suffixes=("", "_from_frame_index"),
            )

    return dets


def find_image_column(df):
    candidates = [
        "frame_image_path",
        "image_path",
        "frame_path",
        "scanpoint_frame_path",
        "frame_image_path_from_frame_index",
        "image_path_from_frame_index",
        "frame_path_from_frame_index",
        "scanpoint_frame_path_from_frame_index",
    ]

    for col in candidates:
        if col in df.columns:
            usable = 0
            for value in df[col].head(50):
                p = resolve_path(value)
                if p is not None and cv2.imread(str(p)) is not None:
                    usable += 1
            if usable > 0:
                return col

    return None


def draw_overlay(image_bgr, box, mask, label):
    out = image_bgr.copy()
    x1, y1, x2, y2 = [int(v) for v in box]

    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.drawContours(out, contours, -1, (0, 0, 255), 2)

    cv2.putText(
        out,
        label,
        (max(0, x1), max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return out


def mask_features(mask, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    mask_area = int(mask.sum())

    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        contour_area = float(cv2.contourArea(largest))
        perimeter = float(cv2.arcLength(largest, True))
        mx, my, mw, mh = cv2.boundingRect(largest)
    else:
        contour_area = np.nan
        perimeter = np.nan
        mx, my, mw, mh = np.nan, np.nan, np.nan, np.nan

    return {
        "bbox_area_pixels": bbox_area,
        "mask_area_pixels": mask_area,
        "mask_area_fraction_of_bbox": mask_area / bbox_area,
        "largest_contour_area": contour_area,
        "contour_perimeter": perimeter,
        "mask_bbox_x1": mx,
        "mask_bbox_y1": my,
        "mask_width": mw,
        "mask_height": mh,
        "mask_aspect_ratio": (mw / mh) if isinstance(mh, (int, float)) and mh and not pd.isna(mh) else np.nan,
    }


rows = []
summary_rows = []

try:
    torch.set_num_threads(4)

    if not DETS_PATH.exists():
        raise FileNotFoundError(f"Detection file not found: {DETS_PATH}")

    if not SAM_CHECKPOINT.exists():
        raise FileNotFoundError(f"Segment Anything Model checkpoint not found: {SAM_CHECKPOINT}")

    dets = load_detections()

    x1_col = first_existing_column(dets, ["x1", "bbox_x1", "xmin", "left"])
    y1_col = first_existing_column(dets, ["y1", "bbox_y1", "ymin", "top"])
    x2_col = first_existing_column(dets, ["x2", "bbox_x2", "xmax", "right"])
    y2_col = first_existing_column(dets, ["y2", "bbox_y2", "ymax", "bottom"])
    image_col = find_image_column(dets)

    if not all([x1_col, y1_col, x2_col, y2_col]):
        raise RuntimeError(f"Could not find bounding box columns. Columns: {list(dets.columns)}")

    if image_col is None:
        raise RuntimeError(f"Could not find readable image path column. Columns: {list(dets.columns)}")

    if "score" in dets.columns:
        dets = dets.sort_values("score", ascending=False)

    selected = []

    for _, r in dets.iterrows():
        img_path = resolve_path(r[image_col])
        if img_path is None:
            continue

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue

        height, width = image_bgr.shape[:2]

        x1 = max(0, min(width - 1, int(round(float(r[x1_col])))))
        y1 = max(0, min(height - 1, int(round(float(r[y1_col])))))
        x2 = max(0, min(width, int(round(float(r[x2_col])))))
        y2 = max(0, min(height, int(round(float(r[y2_col])))))

        if (x2 - x1) < 16 or (y2 - y1) < 16:
            continue

        selected.append((r, img_path, image_bgr, np.array([x1, y1, x2, y2], dtype=np.float32)))

        if len(selected) >= 5:
            break

    if len(selected) < 5:
        raise RuntimeError(f"Only found {len(selected)} usable detections for smoke test.")

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    for i, (r, img_path, image_bgr, box) in enumerate(selected):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        predictor.set_image(image_rgb)

        masks, scores, logits = predictor.predict(
            box=box,
            multimask_output=False,
        )

        mask = masks[0]
        sam_score = float(scores[0]) if len(scores) else np.nan

        scan_frame_id = str(r.get("scan_frame_id", f"sample_{i}"))
        det_id = str(r.get("det_id", i))

        label = f"SAM box prompt | frame {scan_frame_id} | detection {det_id}"
        overlay = draw_overlay(image_bgr, box, mask, label)

        overlay_path = OUT_DIR / f"sam_smoke_{i:02d}_{scan_frame_id}_det_{det_id}.jpg"
        cv2.imwrite(str(overlay_path), overlay)

        row = {
            "sample_id": i,
            "scan_frame_id": scan_frame_id,
            "det_id": det_id,
            "image_path": str(img_path),
            "overlay_path": str(overlay_path),
            "x1": float(box[0]),
            "y1": float(box[1]),
            "x2": float(box[2]),
            "y2": float(box[3]),
            "detector_score": r.get("score", ""),
            "sam_predicted_iou_score": sam_score,
            "sam_model_type": SAM_MODEL_TYPE,
            "sam_checkpoint": str(SAM_CHECKPOINT),
            "device": DEVICE,
            "status": "ok",
        }

        row.update(mask_features(mask, box))
        rows.append(row)

    result = pd.DataFrame(rows)

    area_ok = (
        (result["mask_area_fraction_of_bbox"].astype(float) > 0.02)
        & (result["mask_area_fraction_of_bbox"].astype(float) < 2.50)
    ).all()

    summary_rows = [
        {
            "check": "segment_anything_package_and_checkpoint",
            "status": "PASS",
            "evidence": f"checkpoint_exists=True; size_mb={round(SAM_CHECKPOINT.stat().st_size / (1024 * 1024), 2)}",
        },
        {
            "check": "device",
            "status": "PASS",
            "evidence": DEVICE,
        },
        {
            "check": "input_detections_loaded",
            "status": "PASS",
            "evidence": f"rows={len(dets)}; image_column={image_col}; bbox_columns={[x1_col, y1_col, x2_col, y2_col]}",
        },
        {
            "check": "sam_masks_generated",
            "status": "PASS" if len(result) == 5 else "FAIL",
            "evidence": f"generated_masks={len(result)}",
        },
        {
            "check": "reasonable_mask_area_fraction",
            "status": "PASS" if area_ok else "WARN",
            "evidence": json.dumps(result[["sample_id", "mask_area_fraction_of_bbox"]].to_dict(orient="records")),
        },
        {
            "check": "overlay_images_written",
            "status": "PASS" if all(Path(p).exists() for p in result["overlay_path"]) else "FAIL",
            "evidence": f"overlay_count={sum(Path(p).exists() for p in result['overlay_path'])}",
        },
    ]

except Exception as e:
    summary_rows = [
        {
            "check": "exception",
            "status": "FAIL",
            "evidence": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        }
    ]
    result = pd.DataFrame(rows)


summary = pd.DataFrame(summary_rows)

result_path = STATS / "week6_sam_box_prompt_smoke_test_results.csv"
summary_path = STATS / "week6_sam_box_prompt_smoke_test_summary.csv"
note_path = NOTES / "week6_sam_box_prompt_smoke_test_notes.md"

safe_to_csv(result, result_path)
safe_to_csv(summary, summary_path)

with open(note_path, "w") as f:
    f.write("# Week 6 Segment Anything Model Box-Prompt Smoke Test\n\n")
    f.write("## Purpose\n\n")
    f.write(
        "This smoke test validates Segment Anything Model segmentation using existing pig detector bounding boxes as box prompts. "
        "It runs on five sample detections before launching the full 540-bounding-box pipeline.\n\n"
    )

    f.write("## Summary\n\n")
    f.write(summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Sample results\n\n")
    if len(result):
        f.write(result.to_markdown(index=False))
    else:
        f.write("No masks generated.")
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    if len(result) == 5 and not summary["status"].astype(str).str.contains("FAIL").any():
        f.write(
            "The Segment Anything Model box-prompt smoke test passed. "
            "The next step is to run full Segment Anything Model segmentation over all 540 detector bounding boxes.\n"
        )
    else:
        f.write(
            "The Segment Anything Model smoke test did not fully pass. "
            "Inspect the summary and fix the issue before running the full pipeline.\n"
        )

print("Saved:")
print(result_path)
print(summary_path)
print(note_path)
print(OUT_DIR)

print()
print("=== SAM smoke test summary ===")
print(summary.to_string(index=False))

print()
print("=== SAM smoke test results ===")
print(result.to_string(index=False) if len(result) else "No results.")
