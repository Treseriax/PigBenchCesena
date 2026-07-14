from pathlib import Path
import csv
import json
import math
import traceback
from datetime import datetime

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

BASELINE_SEG_PATH = FEAT / "week6_preliminary_bbox_guided_segmentation_features.csv"

SAM_CHECKPOINT = Path("/work/models/SAM/sam_vit_b_01ec64.pth")
SAM_MODEL_TYPE = "vit_b"
DEVICE = "cpu"

OUT_DIR = VIS / "sam_box_prompt_full_segmentation"
MASK_DIR = OUT_DIR / "masks"
OVERLAY_DIR = OUT_DIR / "frame_overlays"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
STATS.mkdir(parents=True, exist_ok=True)
FEAT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)


FEATURES_PATH = FEAT / "week6_sam_box_prompt_segmentation_features.csv"
FEATURES_JSON_PATH = FEAT / "week6_sam_box_prompt_segmentation_features.json"
FRAME_SUMMARY_PATH = FEAT / "week6_sam_box_prompt_segmentation_frame_summary.csv"
QUALITY_SUMMARY_PATH = STATS / "week6_sam_box_prompt_segmentation_quality_summary.csv"
COMPARISON_PATH = STATS / "week6_sam_vs_grabcut_otsu_segmentation_comparison.csv"
STATUS_PATH = STATS / "week6_sam_box_prompt_segmentation_status_summary.csv"
CONTACT_SHEET_PATH = VIS / "week6_sam_box_prompt_segmentation_contact_sheet.jpg"
NOTE_PATH = NOTES / "week6_sam_box_prompt_full_segmentation_notes.md"
LOG_PROGRESS_PATH = STATS / "week6_sam_box_prompt_segmentation_progress.csv"


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

    if "scan_frame_id" not in dets.columns:
        dets["scan_frame_id"] = "unknown_frame"

    if "det_id" not in dets.columns:
        dets["det_id"] = dets.groupby("scan_frame_id").cumcount()

    dets["row_index"] = np.arange(len(dets))

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

            for value in df[col].head(100):
                p = resolve_path(value)

                if p is not None and cv2.imread(str(p)) is not None:
                    usable += 1

            if usable > 0:
                return col

    return None


def clip_box(row, x1_col, y1_col, x2_col, y2_col, width, height):
    x1 = max(0, min(width - 1, int(round(float(row[x1_col])))))
    y1 = max(0, min(height - 1, int(round(float(row[y1_col])))))
    x2 = max(0, min(width, int(round(float(row[x2_col])))))
    y2 = max(0, min(height, int(round(float(row[y2_col])))))

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def contour_features(mask):
    mask_u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {
            "largest_contour_area": np.nan,
            "contour_perimeter": np.nan,
            "mask_bbox_x1": np.nan,
            "mask_bbox_y1": np.nan,
            "mask_width": np.nan,
            "mask_height": np.nan,
            "mask_aspect_ratio": np.nan,
            "mask_solidity": np.nan,
            "mask_compactness": np.nan,
            "mask_centroid_x": np.nan,
            "mask_centroid_y": np.nan,
            "contour_count": 0,
        }, []

    largest = max(contours, key=cv2.contourArea)

    area = float(cv2.contourArea(largest))
    perimeter = float(cv2.arcLength(largest, True))

    x, y, w, h = cv2.boundingRect(largest)

    hull = cv2.convexHull(largest)
    hull_area = float(cv2.contourArea(hull))

    solidity = area / hull_area if hull_area > 0 else np.nan
    compactness = (4.0 * math.pi * area / (perimeter * perimeter)) if perimeter > 0 else np.nan

    moments = cv2.moments(largest)

    if moments["m00"] != 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        cx = np.nan
        cy = np.nan

    return {
        "largest_contour_area": area,
        "contour_perimeter": perimeter,
        "mask_bbox_x1": x,
        "mask_bbox_y1": y,
        "mask_width": w,
        "mask_height": h,
        "mask_aspect_ratio": (w / h) if h > 0 else np.nan,
        "mask_solidity": solidity,
        "mask_compactness": compactness,
        "mask_centroid_x": cx,
        "mask_centroid_y": cy,
        "contour_count": len(contours),
    }, contours


def mask_features(mask, box):
    x1, y1, x2, y2 = [int(v) for v in box]

    bbox_width = max(1, x2 - x1)
    bbox_height = max(1, y2 - y1)
    bbox_area = bbox_width * bbox_height

    mask_area = int(mask.sum())

    cf, contours = contour_features(mask)

    out = {
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area_pixels": bbox_area,
        "mask_area_pixels": mask_area,
        "mask_area_fraction_of_bbox": mask_area / bbox_area,
    }

    out.update(cf)
    return out, contours


def draw_frame_overlay(image_bgr, frame_rows, masks_and_boxes):
    out = image_bgr.copy()

    for item in masks_and_boxes:
        box = item["box"]
        mask = item["mask"]
        det_id = item["det_id"]
        score = item["sam_predicted_iou_score"]

        x1, y1, x2, y2 = [int(v) for v in box]

        mask_u8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.drawContours(out, contours, -1, (0, 0, 255), 2)

        label = f"SAM det {det_id} {score:.2f}"
        cv2.putText(
            out,
            label,
            (max(0, x1), max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return out


def create_contact_sheet(image_paths, out_path, thumb_w=360, thumb_h=240, cols=4):
    valid = []

    for p in image_paths:
        p = Path(p)

        if p.exists():
            img = cv2.imread(str(p))

            if img is not None:
                valid.append((p, img))

    if not valid:
        return False

    rows = math.ceil(len(valid) / cols)
    sheet = np.full((rows * thumb_h, cols * thumb_w, 3), 255, dtype=np.uint8)

    for i, (p, img) in enumerate(valid):
        r = i // cols
        c = i % cols

        resized = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

        y0 = r * thumb_h
        x0 = c * thumb_w

        sheet[y0:y0 + thumb_h, x0:x0 + thumb_w] = resized

        text = p.stem[:45]
        cv2.putText(
            sheet,
            text,
            (x0 + 6, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), sheet)
    return True


def compare_with_baseline(sam_df):
    if not BASELINE_SEG_PATH.exists():
        return pd.DataFrame([
            {
                "comparison": "baseline_file_missing",
                "status": "WARN",
                "evidence": str(BASELINE_SEG_PATH),
            }
        ])

    base = pd.read_csv(BASELINE_SEG_PATH)

    key_cols = []

    for col in ["scan_frame_id", "det_id"]:
        if col in base.columns and col in sam_df.columns:
            key_cols.append(col)

    if len(key_cols) < 2:
        return pd.DataFrame([
            {
                "comparison": "missing_join_keys",
                "status": "WARN",
                "evidence": f"baseline_cols={list(base.columns)}; sam_cols={list(sam_df.columns)}",
            }
        ])

    base_area_col = None

    for col in ["mask_area_pixels", "segmentation_area_pixels", "foreground_area_pixels", "mask_area"]:
        if col in base.columns:
            base_area_col = col
            break

    if base_area_col is None:
        return pd.DataFrame([
            {
                "comparison": "baseline_area_column_missing",
                "status": "WARN",
                "evidence": f"baseline_cols={list(base.columns)}",
            }
        ])

    merged = sam_df.merge(
        base[key_cols + [base_area_col]],
        on=key_cols,
        how="left",
        suffixes=("_sam", "_baseline"),
    )

    matched = merged[base_area_col].notna().sum()

    if matched == 0:
        return pd.DataFrame([
            {
                "comparison": "baseline_join_matched_rows",
                "status": "WARN",
                "evidence": "matched=0",
            }
        ])

    merged["baseline_mask_area_pixels"] = merged[base_area_col].astype(float)
    merged["sam_minus_baseline_area_pixels"] = merged["mask_area_pixels"].astype(float) - merged["baseline_mask_area_pixels"]
    merged["sam_to_baseline_area_ratio"] = merged["mask_area_pixels"].astype(float) / merged["baseline_mask_area_pixels"].replace(0, np.nan)

    comparison = pd.DataFrame([
        {
            "comparison": "matched_rows",
            "status": "PASS",
            "evidence": int(matched),
        },
        {
            "comparison": "mean_sam_mask_area_pixels",
            "status": "INFO",
            "evidence": float(merged["mask_area_pixels"].mean()),
        },
        {
            "comparison": "mean_baseline_mask_area_pixels",
            "status": "INFO",
            "evidence": float(merged["baseline_mask_area_pixels"].mean()),
        },
        {
            "comparison": "mean_sam_to_baseline_area_ratio",
            "status": "INFO",
            "evidence": float(merged["sam_to_baseline_area_ratio"].replace([np.inf, -np.inf], np.nan).mean()),
        },
        {
            "comparison": "median_sam_to_baseline_area_ratio",
            "status": "INFO",
            "evidence": float(merged["sam_to_baseline_area_ratio"].replace([np.inf, -np.inf], np.nan).median()),
        },
    ])

    detailed_path = STATS / "week6_sam_vs_grabcut_otsu_segmentation_comparison_detailed.csv"
    safe_to_csv(merged, detailed_path)

    return comparison


def main():
    start_time = datetime.now()

    torch.set_num_threads(4)

    if not DETS_PATH.exists():
        raise FileNotFoundError(DETS_PATH)

    if not SAM_CHECKPOINT.exists():
        raise FileNotFoundError(SAM_CHECKPOINT)

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

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    result_rows = []
    frame_rows = []
    failed_rows = []

    grouped = list(dets.groupby("scan_frame_id", sort=True))
    total_frames = len(grouped)
    total_dets = len(dets)

    print(f"Starting full SAM segmentation")
    print(f"Frames: {total_frames}")
    print(f"Detections: {total_dets}")
    print(f"Device: {DEVICE}")
    print(f"Image column: {image_col}")
    print(f"BBox columns: {[x1_col, y1_col, x2_col, y2_col]}")

    processed = 0

    for frame_i, (scan_frame_id, group) in enumerate(grouped, start=1):
        first_row = group.iloc[0]
        img_path = resolve_path(first_row[image_col])

        if img_path is None:
            for _, r in group.iterrows():
                failed_rows.append({
                    "scan_frame_id": scan_frame_id,
                    "det_id": r.get("det_id", ""),
                    "row_index": r.get("row_index", ""),
                    "reason": "image_path_not_resolved",
                })
            continue

        image_bgr = cv2.imread(str(img_path))

        if image_bgr is None:
            for _, r in group.iterrows():
                failed_rows.append({
                    "scan_frame_id": scan_frame_id,
                    "det_id": r.get("det_id", ""),
                    "row_index": r.get("row_index", ""),
                    "reason": "image_not_readable",
                })
            continue

        height, width = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        predictor.set_image(image_rgb)

        masks_for_overlay = []

        for _, r in group.iterrows():
            det_id = r.get("det_id", "")
            row_index = int(r.get("row_index", -1))

            try:
                box = clip_box(r, x1_col, y1_col, x2_col, y2_col, width, height)

                if (box[2] - box[0]) < 16 or (box[3] - box[1]) < 16:
                    raise RuntimeError("bbox_too_small_after_clipping")

                masks, scores, logits = predictor.predict(
                    box=box,
                    multimask_output=False,
                )

                mask = masks[0]
                sam_score = float(scores[0]) if len(scores) else np.nan

                mask_name = f"{scan_frame_id}_det_{det_id}_row_{row_index}_sam_mask.png"
                mask_path = MASK_DIR / mask_name
                cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255))

                features, contours = mask_features(mask, box)

                row = {
                    "scan_frame_id": scan_frame_id,
                    "det_id": det_id,
                    "row_index": row_index,
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
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

                row.update(features)
                result_rows.append(row)

                masks_for_overlay.append({
                    "box": box,
                    "mask": mask,
                    "det_id": det_id,
                    "sam_predicted_iou_score": sam_score,
                })

                processed += 1

            except Exception as e:
                failed_rows.append({
                    "scan_frame_id": scan_frame_id,
                    "det_id": det_id,
                    "row_index": row_index,
                    "reason": f"{type(e).__name__}: {e}",
                })

        overlay = draw_frame_overlay(image_bgr, group, masks_for_overlay)
        overlay_path = OVERLAY_DIR / f"{scan_frame_id}_sam_box_prompt_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay)

        frame_rows.append({
            "scan_frame_id": scan_frame_id,
            "image_path": str(img_path),
            "overlay_path": str(overlay_path),
            "detections_in_frame": len(group),
            "sam_masks_in_frame": len(masks_for_overlay),
            "failed_in_frame": len(group) - len(masks_for_overlay),
        })

        progress = pd.DataFrame([
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "frames_processed": frame_i,
                "frames_total": total_frames,
                "detections_processed_ok": processed,
                "detections_total": total_dets,
                "failed_rows": len(failed_rows),
            }
        ])

        safe_to_csv(progress, LOG_PROGRESS_PATH)

        if frame_i % 5 == 0 or frame_i == total_frames:
            print(
                f"[{frame_i}/{total_frames}] frames | "
                f"ok detections={processed}/{total_dets} | "
                f"failed={len(failed_rows)}",
                flush=True,
            )

            safe_to_csv(pd.DataFrame(result_rows), FEATURES_PATH)
            safe_to_csv(pd.DataFrame(frame_rows), FRAME_SUMMARY_PATH)

    result = pd.DataFrame(result_rows)
    frame_summary = pd.DataFrame(frame_rows)
    failed = pd.DataFrame(failed_rows)

    safe_to_csv(result, FEATURES_PATH)
    result.to_json(FEATURES_JSON_PATH, orient="records", indent=2)

    safe_to_csv(frame_summary, FRAME_SUMMARY_PATH)

    failed_path = STATS / "week6_sam_box_prompt_segmentation_failed_rows.csv"
    safe_to_csv(failed, failed_path)

    if len(result):
        quality = pd.DataFrame([
            {
                "metric": "total_input_detections",
                "value": total_dets,
            },
            {
                "metric": "successful_sam_masks",
                "value": len(result),
            },
            {
                "metric": "failed_sam_masks",
                "value": len(failed),
            },
            {
                "metric": "success_rate",
                "value": len(result) / total_dets if total_dets else 0,
            },
            {
                "metric": "frames_with_overlay",
                "value": int(frame_summary["overlay_path"].apply(lambda p: Path(p).exists()).sum()) if len(frame_summary) else 0,
            },
            {
                "metric": "mean_sam_predicted_iou_score",
                "value": float(result["sam_predicted_iou_score"].mean()),
            },
            {
                "metric": "median_sam_predicted_iou_score",
                "value": float(result["sam_predicted_iou_score"].median()),
            },
            {
                "metric": "mean_mask_area_fraction_of_bbox",
                "value": float(result["mask_area_fraction_of_bbox"].mean()),
            },
            {
                "metric": "median_mask_area_fraction_of_bbox",
                "value": float(result["mask_area_fraction_of_bbox"].median()),
            },
            {
                "metric": "sam_model_type",
                "value": SAM_MODEL_TYPE,
            },
            {
                "metric": "device",
                "value": DEVICE,
            },
        ])
    else:
        quality = pd.DataFrame([
            {
                "metric": "total_input_detections",
                "value": total_dets,
            },
            {
                "metric": "successful_sam_masks",
                "value": 0,
            },
            {
                "metric": "failed_sam_masks",
                "value": len(failed),
            },
            {
                "metric": "success_rate",
                "value": 0,
            },
        ])

    safe_to_csv(quality, QUALITY_SUMMARY_PATH)

    comparison = compare_with_baseline(result)
    safe_to_csv(comparison, COMPARISON_PATH)

    # Representative contact sheet: choose up to 16 frame overlays across the day.
    overlay_paths = list(frame_summary["overlay_path"]) if len(frame_summary) else []
    if len(overlay_paths) > 16:
        idxs = np.linspace(0, len(overlay_paths) - 1, 16).round().astype(int)
        selected_overlay_paths = [overlay_paths[i] for i in idxs]
    else:
        selected_overlay_paths = overlay_paths

    contact_sheet_ok = create_contact_sheet(selected_overlay_paths, CONTACT_SHEET_PATH)

    end_time = datetime.now()
    elapsed_sec = (end_time - start_time).total_seconds()

    status = pd.DataFrame([
        {
            "item": "sam_checkpoint_exists",
            "status": "PASS" if SAM_CHECKPOINT.exists() else "FAIL",
            "evidence": str(SAM_CHECKPOINT),
        },
        {
            "item": "input_detection_rows",
            "status": "PASS" if total_dets == 540 else "WARN",
            "evidence": total_dets,
        },
        {
            "item": "successful_sam_masks",
            "status": "PASS" if len(result) == total_dets else "WARN",
            "evidence": len(result),
        },
        {
            "item": "frame_overlays",
            "status": "PASS" if len(frame_summary) == 72 else "WARN",
            "evidence": len(frame_summary),
        },
        {
            "item": "contact_sheet",
            "status": "PASS" if contact_sheet_ok else "WARN",
            "evidence": str(CONTACT_SHEET_PATH),
        },
        {
            "item": "elapsed_seconds",
            "status": "INFO",
            "evidence": round(elapsed_sec, 2),
        },
    ])

    safe_to_csv(status, STATUS_PATH)

    with open(NOTE_PATH, "w") as f:
        f.write("# Week 6 Segment Anything Model Full Box-Prompt Segmentation\n\n")

        f.write("## Purpose\n\n")
        f.write(
            "This step upgrades the preliminary segmentation deliverable by applying Segment Anything Model to all detector bounding boxes. "
            "Each pig detector bounding box is used as a box prompt, and one mask is generated per detection.\n\n"
        )

        f.write("## Summary\n\n")
        f.write(quality.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Status checks\n\n")
        f.write(status.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Comparison with classical baseline\n\n")
        f.write(comparison.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Outputs\n\n")
        f.write(f"- Segment Anything Model feature table: `{FEATURES_PATH}`\n")
        f.write(f"- Segment Anything Model JSON features: `{FEATURES_JSON_PATH}`\n")
        f.write(f"- Frame summary: `{FRAME_SUMMARY_PATH}`\n")
        f.write(f"- Quality summary: `{QUALITY_SUMMARY_PATH}`\n")
        f.write(f"- Frame overlays: `{OVERLAY_DIR}`\n")
        f.write(f"- Mask images: `{MASK_DIR}`\n")
        f.write(f"- Contact sheet: `{CONTACT_SHEET_PATH}`\n\n")

        f.write("## Interpretation\n\n")
        if len(result) == total_dets:
            f.write(
                "The full Segment Anything Model box-prompt segmentation completed for every detector bounding box. "
                "This provides a stronger foundation-model-based segmentation route in addition to the earlier classical GrabCut/Otsu baseline. "
                "The masks are still automatic model outputs, not manual segmentation ground truth.\n"
            )
        else:
            f.write(
                "The Segment Anything Model segmentation produced outputs for a subset of detections. "
                "Failed rows should be inspected before treating the output as complete.\n"
            )

    print()
    print("Saved:")
    print(FEATURES_PATH)
    print(FEATURES_JSON_PATH)
    print(FRAME_SUMMARY_PATH)
    print(QUALITY_SUMMARY_PATH)
    print(COMPARISON_PATH)
    print(STATUS_PATH)
    print(CONTACT_SHEET_PATH)
    print(NOTE_PATH)

    print()
    print("=== Full SAM quality summary ===")
    print(quality.to_string(index=False))

    print()
    print("=== Full SAM status ===")
    print(status.to_string(index=False))


try:
    main()
except Exception as e:
    err_path = STATS / "week6_sam_box_prompt_segmentation_exception.txt"
    err_path.write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")

    print("FAILED")
    print(err_path)
    raise
