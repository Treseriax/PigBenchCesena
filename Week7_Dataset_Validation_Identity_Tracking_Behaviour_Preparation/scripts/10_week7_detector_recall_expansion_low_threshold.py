from pathlib import Path
from datetime import datetime
import csv
import math
import traceback

import cv2
import numpy as np
import pandas as pd
import torch


ROOT = Path.home() / "PigBench"

W6 = ROOT / "Week6_Unibo_Dataset_Validation"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

FRAME_INDEX_PATH = W6 / "outputs" / "unified_ground_truth" / "week6_scanpoint_frame_index.csv"
OLD_DETS_PATH = W6 / "outputs" / "feature_extractors" / "week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv"
V6_FRAME_SUMMARY = W7 / "outputs" / "roi_and_crate_mapping" / "week7_gt_pen_detection_selection_v6_frame_summary.csv"

OUT_ROI = W7 / "outputs" / "roi_and_crate_mapping"
OUT_VIS = W7 / "outputs" / "visualizations"
OUT_STATS = W7 / "outputs" / "dataset_statistics"
OUT_NOTES = W7 / "notes"

for p in [OUT_ROI, OUT_VIS, OUT_STATS, OUT_NOTES]:
    p.mkdir(parents=True, exist_ok=True)

OUT_DETS = OUT_ROI / "week7_detector_recall_expansion_low_threshold_detections.csv"
OUT_THRESH_SUMMARY = OUT_ROI / "week7_detector_recall_expansion_threshold_summary.csv"
OUT_NEW_CANDIDATES = OUT_ROI / "week7_detector_recall_expansion_new_candidates.csv"
OUT_FRAME_SUMMARY = OUT_ROI / "week7_detector_recall_expansion_frame_summary.csv"
OUT_MODEL_AUDIT = OUT_STATS / "week7_detector_recall_expansion_model_audit.csv"
OUT_CONTACT = OUT_VIS / "week7_detector_recall_expansion_problem_frames_contact_sheet.jpg"
OUT_NOTE = OUT_NOTES / "week7_detector_recall_expansion_low_threshold_notes.md"


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


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))

    return inter / float(area_a + area_b - inter)


def find_detector_config_and_checkpoint():
    config_candidates = []

    search_roots = [
        ROOT / "detection" / "configs",
        ROOT / "Week6_Unibo_Dataset_Validation" / "scripts",
        ROOT,
    ]

    for sr in search_roots:
        if sr.exists():
            for p in sr.rglob("*.py"):
                name = p.name.lower()
                text_hint = ""

                try:
                    text_hint = p.read_text(errors="ignore")[:20000].lower()
                except Exception:
                    text_hint = ""

                if "yolov8" in name or "yolov8" in text_hint:
                    if "pig" in name or "pig" in text_hint or "pigdetect" in text_hint:
                        config_candidates.append(p)

    # Prefer small/s config.
    config_candidates = sorted(
        set(config_candidates),
        key=lambda p: (
            0 if ("yolov8_s" in p.name.lower() or "yolov8-s" in p.name.lower()) else 1,
            len(str(p)),
        ),
    )

    checkpoint_candidates = []

    known = [
        ROOT / "detection" / "data" / "pretrained_weights" / "yolov8_pigs" / "yolov8_s.pth",
        Path("/work/pig/datasets/PigDetect-YOLOv8/yolov8_s.pth"),
    ]

    for p in known:
        if p.exists():
            checkpoint_candidates.append(p)

    for sr in [
        ROOT / "detection",
        ROOT / "Week6_Unibo_Dataset_Validation",
        Path("/work/pig/datasets"),
    ]:
        if sr.exists():
            for p in sr.rglob("*.pth"):
                name = p.name.lower()
                if "yolov8" in name and ("_s" in name or "-s" in name or "small" in name or name == "yolov8_s.pth"):
                    checkpoint_candidates.append(p)

    checkpoint_candidates = sorted(
        set(checkpoint_candidates),
        key=lambda p: (
            0 if p.name.lower() == "yolov8_s.pth" else 1,
            len(str(p)),
        ),
    )

    cfg = config_candidates[0] if config_candidates else None
    ckpt = checkpoint_candidates[0] if checkpoint_candidates else None

    return cfg, ckpt, config_candidates[:10], checkpoint_candidates[:10]


def extract_predictions(result):
    # mmdet v3 DetDataSample
    pred = result.pred_instances

    bboxes = pred.bboxes.detach().cpu().numpy()
    scores = pred.scores.detach().cpu().numpy()

    if hasattr(pred, "labels"):
        labels = pred.labels.detach().cpu().numpy()
    else:
        labels = np.zeros(len(scores), dtype=int)

    rows = []

    for i in range(len(scores)):
        x1, y1, x2, y2 = bboxes[i].tolist()

        rows.append({
            "pred_id": i,
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "score": float(scores[i]),
            "label": int(labels[i]),
        })

    return rows


def make_contact_sheet(frames, expanded, old_dets, problem_sids, output_path):
    image_col = first_col(frames, ["frame_image_path", "image_path", "frame_path", "scanpoint_frame_path"])

    if image_col is None:
        return False

    sample = frames[frames["scan_frame_id"].astype(str).isin(problem_sids)].copy()

    if len(sample) == 0:
        sample = frames.copy()

    sample = sample.drop_duplicates("scan_frame_id").sort_values("scan_frame_id")

    if len(sample) > 16:
        idx = np.linspace(0, len(sample) - 1, 16).round().astype(int)
        sample = sample.iloc[idx]

    thumbs = []
    thumb_w, thumb_h = 380, 250

    for _, fr in sample.iterrows():
        sid = str(fr["scan_frame_id"])
        img_path = resolve_path(fr[image_col])

        if img_path is None:
            continue

        img = cv2.imread(str(img_path))

        if img is None:
            continue

        overlay = img.copy()

        old_frame = old_dets[old_dets["scan_frame_id"].astype(str) == sid].copy()
        new_frame = expanded[
            (expanded["scan_frame_id"].astype(str) == sid)
            & (expanded["score"] >= 0.05)
        ].copy()

        # Old detections in grey.
        for _, d in old_frame.iterrows():
            x1, y1, x2, y2 = [int(round(float(d[c]))) for c in ["x1", "y1", "x2", "y2"]]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (120, 120, 120), 1)

        # New low threshold candidates.
        for _, d in new_frame.iterrows():
            x1, y1, x2, y2 = [int(round(float(d[c]))) for c in ["x1", "y1", "x2", "y2"]]

            is_new = bool(d["is_new_candidate_vs_week6"])
            score = float(d["score"])

            if is_new:
                color = (0, 165, 255)
                thickness = 2
                prefix = "NEW"
            else:
                color = (0, 220, 0)
                thickness = 1
                prefix = "OLD"

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

            if is_new or score < 0.25:
                cv2.putText(
                    overlay,
                    f"{prefix} {score:.2f}",
                    (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        old_count = len(old_frame)
        new_low_count = int(((new_frame["is_new_candidate_vs_week6"] == True) & (new_frame["score"] >= 0.05)).sum())

        title = f"{sid} | old={old_count} | low-th new={new_low_count}"

        cv2.putText(
            overlay,
            title,
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        thumbs.append(cv2.resize(overlay, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA))

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
# Load inputs.
# ---------------------------------------------------------------------
frames = pd.read_csv(FRAME_INDEX_PATH)
old_dets = pd.read_csv(OLD_DETS_PATH)
v6_summary = pd.read_csv(V6_FRAME_SUMMARY)

for c in ["x1", "y1", "x2", "y2"]:
    old_dets[c] = pd.to_numeric(old_dets[c], errors="coerce")

old_dets = old_dets.dropna(subset=["x1", "y1", "x2", "y2"]).copy()

image_col = first_col(frames, ["frame_image_path", "image_path", "frame_path", "scanpoint_frame_path"])

if image_col is None:
    raise RuntimeError("No image path column found in frame index.")

cfg, ckpt, cfg_candidates, ckpt_candidates = find_detector_config_and_checkpoint()

audit_rows = [
    {"item": "config_selected", "value": str(cfg), "exists": cfg.exists() if cfg else False},
    {"item": "checkpoint_selected", "value": str(ckpt), "exists": ckpt.exists() if ckpt else False},
    {"item": "config_candidates_top10", "value": " | ".join(map(str, cfg_candidates)), "exists": bool(cfg_candidates)},
    {"item": "checkpoint_candidates_top10", "value": " | ".join(map(str, ckpt_candidates)), "exists": bool(ckpt_candidates)},
    {"item": "torch_version", "value": torch.__version__, "exists": True},
    {"item": "cuda_available", "value": str(torch.cuda.is_available()), "exists": True},
]

safe_to_csv(pd.DataFrame(audit_rows), OUT_MODEL_AUDIT)

if cfg is None or ckpt is None:
    raise RuntimeError("Could not find detector config/checkpoint. See model audit CSV.")

# ---------------------------------------------------------------------
# Run detector.
# ---------------------------------------------------------------------
from mmdet.apis import init_detector, inference_detector

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = init_detector(str(cfg), str(ckpt), device=device)

rows = []

for _, fr in frames.drop_duplicates("scan_frame_id").sort_values("scan_frame_id").iterrows():
    sid = str(fr["scan_frame_id"])
    img_path = resolve_path(fr[image_col])

    if img_path is None:
        rows.append({
            "scan_frame_id": sid,
            "status": "missing_image",
            "image_path": str(fr[image_col]),
        })
        continue

    try:
        result = inference_detector(model, str(img_path))
        preds = extract_predictions(result)

        for p in preds:
            p["scan_frame_id"] = sid
            p["image_path"] = str(img_path)
            p["status"] = "ok"
            rows.append(p)

    except Exception as e:
        rows.append({
            "scan_frame_id": sid,
            "status": "inference_error",
            "image_path": str(img_path),
            "error": repr(e),
            "traceback": traceback.format_exc()[-2000:],
        })

expanded = pd.DataFrame(rows)

if "score" not in expanded.columns:
    raise RuntimeError("Detector produced no score column. Check inference errors.")

expanded = expanded[expanded["status"] == "ok"].copy()

# ---------------------------------------------------------------------
# Compare with old Week 6 detections.
# ---------------------------------------------------------------------
max_ious = []

for _, d in expanded.iterrows():
    sid = str(d["scan_frame_id"])
    old_frame = old_dets[old_dets["scan_frame_id"].astype(str) == sid]

    box = [float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"])]

    best = 0.0

    for _, o in old_frame.iterrows():
        obox = [float(o["x1"]), float(o["y1"]), float(o["x2"]), float(o["y2"])]
        best = max(best, bbox_iou(box, obox))

    max_ious.append(best)

expanded["max_iou_with_week6_detection"] = max_ious
expanded["is_new_candidate_vs_week6"] = expanded["max_iou_with_week6_detection"] < 0.50

safe_to_csv(expanded, OUT_DETS)

# Threshold summaries.
thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50]

summary_rows = []

for th in thresholds:
    sub = expanded[expanded["score"] >= th].copy()

    by_frame = sub.groupby("scan_frame_id").size()

    summary_rows.append({
        "threshold": th,
        "total_detections": len(sub),
        "mean_detections_per_frame": float(by_frame.mean()) if len(by_frame) else 0,
        "median_detections_per_frame": float(by_frame.median()) if len(by_frame) else 0,
        "min_detections_per_frame": int(by_frame.min()) if len(by_frame) else 0,
        "max_detections_per_frame": int(by_frame.max()) if len(by_frame) else 0,
        "new_candidates_vs_week6": int(sub["is_new_candidate_vs_week6"].sum()),
    })

threshold_summary = pd.DataFrame(summary_rows)
safe_to_csv(threshold_summary, OUT_THRESH_SUMMARY)

# New candidates.
new_candidates = expanded[
    (expanded["is_new_candidate_vs_week6"] == True)
    & (expanded["score"] >= 0.05)
].copy()

safe_to_csv(new_candidates, OUT_NEW_CANDIDATES)

# Frame summary including v6 strict count.
frame_rows = []

for sid, group in expanded.groupby("scan_frame_id"):
    v6_row = v6_summary[v6_summary["scan_frame_id"].astype(str) == str(sid)]

    if len(v6_row):
        strict_count = int(v6_row["strict_auto_safe_count_v6"].iloc[0])
        frame_needs_review = bool(str(v6_row["frame_needs_manual_review_v6"].iloc[0]).lower() == "true")
    else:
        strict_count = ""
        frame_needs_review = True

    frame_rows.append({
        "scan_frame_id": sid,
        "v6_strict_auto_safe_count": strict_count,
        "v6_frame_needs_manual_review": frame_needs_review,
        "expanded_count_score_ge_005": int((group["score"] >= 0.05).sum()),
        "expanded_count_score_ge_010": int((group["score"] >= 0.10).sum()),
        "expanded_count_score_ge_025": int((group["score"] >= 0.25).sum()),
        "new_candidate_count_score_ge_005": int(((group["score"] >= 0.05) & (group["is_new_candidate_vs_week6"] == True)).sum()),
        "new_candidate_count_score_ge_010": int(((group["score"] >= 0.10) & (group["is_new_candidate_vs_week6"] == True)).sum()),
    })

frame_summary = pd.DataFrame(frame_rows)
safe_to_csv(frame_summary, OUT_FRAME_SUMMARY)

problem_sids = set(
    v6_summary.loc[
        (v6_summary["strict_auto_safe_count_v6"] < 6)
        | (v6_summary["frame_needs_manual_review_v6"].astype(str).str.lower() == "true"),
        "scan_frame_id",
    ].astype(str)
)

contact_ok = make_contact_sheet(frames, expanded, old_dets, problem_sids, OUT_CONTACT)

# Notes.
with open(OUT_NOTE, "w") as f:
    f.write("# Week 7 Detector Recall Expansion with Low Threshold\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This step reruns the pig detector on the 72 scanpoint frames to recover possible missed pigs. "
        "The motivation is that some correct ground-truth pen pigs were not detected in the previous detection table, while some wrong-pen detections were still marked as strict-safe.\n\n"
    )

    f.write("## Model\n\n")
    f.write(f"- Config: `{cfg}`\n")
    f.write(f"- Checkpoint: `{ckpt}`\n")
    f.write(f"- Device: `{device}`\n\n")

    f.write("## Threshold summary\n\n")
    f.write(threshold_summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## New candidates\n\n")
    f.write(f"- New low-threshold candidates with score >= 0.05: `{len(new_candidates)}`\n")
    f.write(f"- Contact sheet: `{OUT_CONTACT}`\n")
    f.write(f"- Contact sheet generated: `{contact_ok}`\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "New low-threshold detections should not be accepted automatically. "
        "They are candidate boxes that may recover missed true pigs, but they also introduce more false positives. "
        "The next step is to merge useful new candidates into a v7 corrected candidate pool with manual review.\n\n"
    )

    f.write("## Outputs\n\n")
    for p in [
        OUT_DETS,
        OUT_THRESH_SUMMARY,
        OUT_NEW_CANDIDATES,
        OUT_FRAME_SUMMARY,
        OUT_MODEL_AUDIT,
        OUT_CONTACT,
    ]:
        f.write(f"- `{p}`\n")

print("Saved:")
print(OUT_DETS)
print(OUT_THRESH_SUMMARY)
print(OUT_NEW_CANDIDATES)
print(OUT_FRAME_SUMMARY)
print(OUT_MODEL_AUDIT)
print(OUT_CONTACT)
print(OUT_NOTE)

print()
print("=== Model audit ===")
print(pd.DataFrame(audit_rows).to_string(index=False))

print()
print("=== Threshold summary ===")
print(threshold_summary.to_string(index=False))

print()
print("=== New candidate summary ===")
print({
    "new_candidates_score_ge_005": len(new_candidates),
    "contact_sheet_generated": contact_ok,
})
