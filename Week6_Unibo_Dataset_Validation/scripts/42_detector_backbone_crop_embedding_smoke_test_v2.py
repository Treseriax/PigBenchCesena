from pathlib import Path
import csv
import json
import traceback
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False

try:
    from mmdet.apis import init_detector
    MMDET_AVAILABLE = True
except Exception:
    MMDET_AVAILABLE = False


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"

FEAT = W6 / "outputs/feature_extractors"
GT = W6 / "outputs/unified_ground_truth"
STATS = W6 / "outputs/dataset_statistics"
NOTES = W6 / "notes"

DETS_PATH = FEAT / "week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv"
FRAME_INDEX_PATH = GT / "week6_scanpoint_frame_index.csv"

CONFIG = ROOT / "detection/configs/yolov8/yolov8_s.py"
CHECKPOINT = ROOT / "detection/data/pretrained_weights/yolov8_pigs/yolov8_s.pth"

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


def get_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def resolve_path(p):
    if pd.isna(p):
        return None

    p = str(p).strip()

    if not p:
        return None

    candidates = [
        Path(p),
        W6 / p,
        ROOT / p,
        Path.home() / p,
    ]

    for c in candidates:
        if c.exists():
            return c

    return None


def load_detection_table():
    dets = pd.read_csv(DETS_PATH)
    frames = pd.read_csv(FRAME_INDEX_PATH)

    if "scan_frame_id" in frames.columns and "scan_frame_id" in dets.columns:
        frame_cols = ["scan_frame_id"]

        for c in [
            "frame_image_path",
            "video_id",
            "timestamp",
            "frame_index",
            "timestamp_sec_in_video",
            "video_match_status",
        ]:
            if c in frames.columns and c not in frame_cols:
                frame_cols.append(c)

        dets = dets.merge(
            frames[frame_cols],
            on="scan_frame_id",
            how="left",
            suffixes=("", "_from_frame_index"),
        )

    # Prefer original frame_image_path if present.
    if "frame_image_path" not in dets.columns and "frame_image_path_from_frame_index" in dets.columns:
        dets["frame_image_path"] = dets["frame_image_path_from_frame_index"]

    return dets


def extract_crop(row, x1_col, y1_col, x2_col, y2_col):
    img_path = resolve_path(row.get("frame_image_path", ""))

    if img_path is None:
        return None, "path_not_resolved"

    img = cv2.imread(str(img_path))

    if img is None:
        return None, "image_read_failed"

    h, w = img.shape[:2]

    try:
        x1 = max(0, min(w - 1, int(round(float(row[x1_col])))))
        y1 = max(0, min(h - 1, int(round(float(row[y1_col])))))
        x2 = max(0, min(w, int(round(float(row[x2_col])))))
        y2 = max(0, min(h, int(round(float(row[y2_col])))))
    except Exception:
        return None, "bbox_parse_failed"

    crop = img[y1:y2, x1:x2]

    if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
        return None, f"bad_crop_shape_{crop.shape if crop is not None else None}"

    return crop, "ok"


def crop_to_tensor(crop, size=640):
    # MMYOLO/MMDet checkpoints usually expect BGR-like image tensors before model preprocessing.
    # For feature smoke testing, resize crop and convert HWC BGR uint8 -> BCHW float.
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    arr = resized.transpose(2, 0, 1)
    tensor = torch.from_numpy(arr).float().unsqueeze(0)
    return tensor


def flatten_feats(feats):
    vecs = []

    if isinstance(feats, torch.Tensor):
        feats = [feats]

    if isinstance(feats, (list, tuple)):
        for f in feats:
            if not isinstance(f, torch.Tensor):
                continue

            if f.ndim == 4:
                pooled = F.adaptive_avg_pool2d(f, 1).flatten(1)
            elif f.ndim == 3:
                pooled = f.mean(dim=1)
            elif f.ndim == 2:
                pooled = f
            else:
                pooled = f.flatten(1)

            vecs.append(pooled)

    if not vecs:
        raise RuntimeError("No tensor feature maps returned by extract_feat.")

    return torch.cat(vecs, dim=1)


device = "cuda:0" if torch.cuda.is_available() else "cpu"

rows = []

basic_ok = CV2_AVAILABLE and MMDET_AVAILABLE and CONFIG.exists() and CHECKPOINT.exists()

rows.append({
    "check": "environment",
    "status": "PASS" if basic_ok else "FAIL",
    "details": f"cv2={CV2_AVAILABLE}; mmdet={MMDET_AVAILABLE}; config_exists={CONFIG.exists()}; checkpoint_exists={CHECKPOINT.exists()}; device={device}",
})

if not basic_ok:
    result = pd.DataFrame(rows)
    out = STATS / "week6_detector_backbone_crop_embedding_smoke_test_v2.csv"
    safe_to_csv(result, out)
    raise SystemExit("Environment not ready for detector-backed embedding smoke test.")

try:
    dets = load_detection_table()

    x1_col = get_col(dets, ["x1", "bbox_x1"])
    y1_col = get_col(dets, ["y1", "bbox_y1"])
    x2_col = get_col(dets, ["x2", "bbox_x2"])
    y2_col = get_col(dets, ["y2", "bbox_y2"])

    if not all([x1_col, y1_col, x2_col, y2_col]):
        raise RuntimeError(f"Missing bbox cols: {list(dets.columns)}")

    if "score" in dets.columns:
        dets = dets.sort_values("score", ascending=False)

    usable = []

    for _, r in dets.iterrows():
        crop, status = extract_crop(r, x1_col, y1_col, x2_col, y2_col)

        if status == "ok":
            usable.append((r, crop))

        if len(usable) >= 5:
            break

    rows.append({
        "check": "usable_crop_selection",
        "status": "PASS" if len(usable) >= 3 else "FAIL",
        "details": f"usable_crops={len(usable)}",
    })

    if len(usable) < 3:
        raise RuntimeError("Less than 3 usable crops found.")

    model = init_detector(str(CONFIG), str(CHECKPOINT), device=device)
    model.eval()

    rows.append({
        "check": "model_load",
        "status": "PASS",
        "details": f"class={model.__class__.__name__}; has_extract_feat={hasattr(model, 'extract_feat')}",
    })

    if not hasattr(model, "extract_feat"):
        raise RuntimeError("Model does not expose extract_feat.")

    embedding_rows = []

    with torch.no_grad():
        for i, (r, crop) in enumerate(usable):
            inp = crop_to_tensor(crop, size=640).to(device)

            feats = model.extract_feat(inp)
            vec = flatten_feats(feats)

            vec_np = vec.detach().cpu().numpy().reshape(-1)

            embedding_rows.append({
                "sample_id": i,
                "scan_frame_id": r.get("scan_frame_id", ""),
                "det_id": r.get("det_id", ""),
                "crop_shape": str(crop.shape),
                "embedding_dim": int(vec_np.shape[0]),
                "embedding_mean": float(np.mean(vec_np)),
                "embedding_std": float(np.std(vec_np)),
                "embedding_l2_norm": float(np.linalg.norm(vec_np)),
                "feature_maps_type": type(feats).__name__,
                "status": "ok",
            })

    emb_df = pd.DataFrame(embedding_rows)

    dim_ok = emb_df["embedding_dim"].nunique() == 1 and emb_df["embedding_dim"].iloc[0] > 0
    finite_ok = np.isfinite(emb_df[["embedding_mean", "embedding_std", "embedding_l2_norm"]].values).all()

    rows.append({
        "check": "embedding_extraction",
        "status": "PASS" if dim_ok and finite_ok else "FAIL",
        "details": f"embedding_dims={emb_df['embedding_dim'].tolist()}; finite_ok={finite_ok}",
    })

except Exception as e:
    rows.append({
        "check": "exception",
        "status": "FAIL",
        "details": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
    })
    emb_df = pd.DataFrame()

result = pd.DataFrame(rows)

result_path = STATS / "week6_detector_backbone_crop_embedding_smoke_test_v2.csv"
emb_preview_path = FEAT / "week6_detector_backbone_crop_embedding_smoke_test_v2_preview.csv"
decision_path = STATS / "week6_detector_backbone_crop_embedding_decision_v2.csv"

safe_to_csv(result, result_path)
safe_to_csv(emb_df, emb_preview_path)

passed = bool((result["status"] == "FAIL").sum() == 0)

decision = pd.DataFrame([
    {
        "decision": "can_run_full_detector_backbone_crop_embeddings",
        "value": passed,
        "reason": (
            "Detector model loaded and extract_feat produced finite fixed-length embeddings for sample crops."
            if passed
            else "Smoke test failed; do not run full embeddings yet."
        ),
    },
    {
        "decision": "selected_method",
        "value": "mmdet_yolov8s_pig_detector_backbone_global_avg_pool" if passed else "none",
        "reason": "Uses locally available pig detector checkpoint, not random/untrained embeddings.",
    },
])

safe_to_csv(decision, decision_path)

note_path = NOTES / "week6_detector_backbone_crop_embedding_smoke_test_v2_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Detector-Backbone Crop Embedding Smoke Test v2\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This step tests whether the locally available PigBench YOLOv8-s detector checkpoint can produce detector-backbone crop embeddings. "
        "This avoids generating dummy/untrained embeddings.\n\n"
    )

    f.write("## Smoke test result\n\n")
    f.write(result.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Embedding preview\n\n")
    f.write(emb_df.to_markdown(index=False) if len(emb_df) else "No embeddings generated.")
    f.write("\n\n")

    f.write("## Decision\n\n")
    f.write(decision.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    if passed:
        f.write(
            "The detector-backed embedding route is usable. "
            "Next step: extract embeddings for all 540 detector crops and save a full feature table.\n"
        )
    else:
        f.write(
            "The detector-backed embedding route is not yet usable. "
            "Inspect the failure before attempting full extraction.\n"
        )

print("Saved:")
print(result_path)
print(emb_preview_path)
print(decision_path)
print(note_path)

print()
print("=== Smoke test result ===")
print(result.to_string(index=False))

print()
print("=== Embedding preview ===")
print(emb_df.to_string(index=False) if len(emb_df) else "No embeddings generated.")

print()
print("=== Decision ===")
print(decision.to_string(index=False))
