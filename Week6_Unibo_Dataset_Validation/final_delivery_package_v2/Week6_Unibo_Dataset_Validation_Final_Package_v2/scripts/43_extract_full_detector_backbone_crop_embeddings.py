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

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


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
            "video_mapping_confidence",
        ]:
            if c in frames.columns and c not in frame_cols:
                frame_cols.append(c)

        dets = dets.merge(
            frames[frame_cols],
            on="scan_frame_id",
            how="left",
            suffixes=("", "_from_frame_index"),
        )

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
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    arr = resized.transpose(2, 0, 1)
    tensor = torch.from_numpy(arr).float()
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


if not CV2_AVAILABLE:
    raise RuntimeError("cv2 unavailable.")
if not MMDET_AVAILABLE:
    raise RuntimeError("mmdet unavailable.")
if not CONFIG.exists():
    raise FileNotFoundError(CONFIG)
if not CHECKPOINT.exists():
    raise FileNotFoundError(CHECKPOINT)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 8

dets = load_detection_table()

x1_col = get_col(dets, ["x1", "bbox_x1"])
y1_col = get_col(dets, ["y1", "bbox_y1"])
x2_col = get_col(dets, ["x2", "bbox_x2"])
y2_col = get_col(dets, ["y2", "bbox_y2"])

if not all([x1_col, y1_col, x2_col, y2_col]):
    raise RuntimeError(f"Missing bbox cols: {list(dets.columns)}")

model = init_detector(str(CONFIG), str(CHECKPOINT), device=device)
model.eval()

if not hasattr(model, "extract_feat"):
    raise RuntimeError("Model does not expose extract_feat.")

embedding_vectors = []
metadata_rows = []
status_rows = []

batch_tensors = []
batch_meta = []

for idx, row in dets.iterrows():
    crop, crop_status = extract_crop(row, x1_col, y1_col, x2_col, y2_col)

    meta = {
        "embedding_row_id": len(metadata_rows),
        "source_detection_row_index": idx,
        "scan_frame_id": row.get("scan_frame_id", ""),
        "det_id": row.get("det_id", ""),
        "video_id": row.get("video_id", ""),
        "timestamp": row.get("timestamp", ""),
        "timestamp_sec_in_video": row.get("timestamp_sec_in_video", ""),
        "frame_index": row.get("frame_index", ""),
        "frame_image_path": row.get("frame_image_path", ""),
        "x1": row.get(x1_col, ""),
        "y1": row.get(y1_col, ""),
        "x2": row.get(x2_col, ""),
        "y2": row.get(y2_col, ""),
        "score": row.get("score", ""),
        "bbox_count_issue_type": row.get("bbox_count_issue_type", ""),
        "detection_score_band": row.get("detection_score_band", ""),
        "recommended_use": row.get("recommended_use", ""),
        "crop_status": crop_status,
        "crop_shape": str(crop.shape) if crop is not None else "",
        "embedding_method": "mmdet_yolov8s_pig_detector_backbone_global_avg_pool",
        "config": str(CONFIG),
        "checkpoint": str(CHECKPOINT),
        "device": device,
    }

    if crop_status != "ok":
        status_rows.append({
            "source_detection_row_index": idx,
            "scan_frame_id": row.get("scan_frame_id", ""),
            "det_id": row.get("det_id", ""),
            "status": crop_status,
        })
        continue

    tensor = crop_to_tensor(crop, size=640)
    batch_tensors.append(tensor)
    batch_meta.append(meta)

    if len(batch_tensors) >= batch_size:
        x = torch.stack(batch_tensors, dim=0).to(device)

        with torch.no_grad():
            feats = model.extract_feat(x)
            vec = flatten_feats(feats).detach().cpu().numpy()

        for m, v in zip(batch_meta, vec):
            m["embedding_dim"] = int(v.shape[0])
            m["embedding_mean"] = float(np.mean(v))
            m["embedding_std"] = float(np.std(v))
            m["embedding_l2_norm"] = float(np.linalg.norm(v))
            metadata_rows.append(m)
            embedding_vectors.append(v.astype(np.float32))

        batch_tensors = []
        batch_meta = []

# Remaining batch.
if batch_tensors:
    x = torch.stack(batch_tensors, dim=0).to(device)

    with torch.no_grad():
        feats = model.extract_feat(x)
        vec = flatten_feats(feats).detach().cpu().numpy()

    for m, v in zip(batch_meta, vec):
        m["embedding_dim"] = int(v.shape[0])
        m["embedding_mean"] = float(np.mean(v))
        m["embedding_std"] = float(np.std(v))
        m["embedding_l2_norm"] = float(np.linalg.norm(v))
        metadata_rows.append(m)
        embedding_vectors.append(v.astype(np.float32))

metadata = pd.DataFrame(metadata_rows)
status = pd.DataFrame(status_rows)

if embedding_vectors:
    embeddings = np.vstack(embedding_vectors).astype(np.float32)
else:
    embeddings = np.zeros((0, 0), dtype=np.float32)

# Save NPY.
npy_path = FEAT / "week6_detector_backbone_crop_embeddings_896.npy"
np.save(npy_path, embeddings)

# Save metadata.
metadata_path = FEAT / "week6_detector_backbone_crop_embedding_metadata.csv"
safe_to_csv(metadata, metadata_path)

status_path = STATS / "week6_detector_backbone_crop_embedding_status.csv"
safe_to_csv(status, status_path)

# Save wide CSV.
emb_cols = [f"emb_{i:04d}" for i in range(embeddings.shape[1])]
emb_df = pd.DataFrame(embeddings, columns=emb_cols)

wide = pd.concat(
    [
        metadata[
            [
                "embedding_row_id",
                "source_detection_row_index",
                "scan_frame_id",
                "det_id",
                "video_id",
                "timestamp",
                "score",
                "bbox_count_issue_type",
                "detection_score_band",
                "recommended_use",
                "embedding_dim",
                "embedding_mean",
                "embedding_std",
                "embedding_l2_norm",
            ]
        ].reset_index(drop=True),
        emb_df.reset_index(drop=True),
    ],
    axis=1,
)

wide_path = FEAT / "week6_detector_backbone_crop_embeddings_896.csv"
safe_to_csv(wide, wide_path)

# PCA features for compact comparison.
pca_path = FEAT / "week6_detector_backbone_crop_embedding_pca_features.csv"
pca_summary_path = STATS / "week6_detector_backbone_crop_embedding_pca_summary.csv"

if SKLEARN_AVAILABLE and embeddings.shape[0] >= 10 and embeddings.shape[1] >= 2:
    n_components = min(16, embeddings.shape[0], embeddings.shape[1])
    scaled = StandardScaler().fit_transform(embeddings)
    pca = PCA(n_components=n_components, random_state=0)
    pca_values = pca.fit_transform(scaled)

    pca_cols = [f"pca_{i+1:02d}" for i in range(n_components)]
    pca_df = pd.DataFrame(pca_values, columns=pca_cols)

    pca_out = pd.concat(
        [
            metadata[
                [
                    "embedding_row_id",
                    "source_detection_row_index",
                    "scan_frame_id",
                    "det_id",
                    "video_id",
                    "timestamp",
                    "score",
                    "bbox_count_issue_type",
                    "detection_score_band",
                    "recommended_use",
                ]
            ].reset_index(drop=True),
            pca_df.reset_index(drop=True),
        ],
        axis=1,
    )

    safe_to_csv(pca_out, pca_path)

    pca_summary = pd.DataFrame([
        {
            "component": f"pca_{i+1:02d}",
            "explained_variance_ratio": float(v),
            "cumulative_explained_variance_ratio": float(np.sum(pca.explained_variance_ratio_[:i+1])),
        }
        for i, v in enumerate(pca.explained_variance_ratio_)
    ])

    safe_to_csv(pca_summary, pca_summary_path)
else:
    pca_out = pd.DataFrame()
    pca_summary = pd.DataFrame()
    safe_to_csv(pca_out, pca_path)
    safe_to_csv(pca_summary, pca_summary_path)

# Summary.
summary_rows = [
    {
        "metric": "input_detection_rows",
        "value": len(dets),
        "interpretation": "All detector rows considered for crop embedding extraction.",
    },
    {
        "metric": "successful_embedding_rows",
        "value": len(metadata),
        "interpretation": "Rows with usable crop and finite detector-backbone embedding.",
    },
    {
        "metric": "failed_crop_rows",
        "value": len(status),
        "interpretation": "Rows skipped due to crop loading/crop quality problems.",
    },
    {
        "metric": "embedding_dim",
        "value": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.shape[1] else 0,
        "interpretation": "Detector-backbone pooled feature dimension.",
    },
    {
        "metric": "embedding_method",
        "value": "mmdet_yolov8s_pig_detector_backbone_global_avg_pool",
        "interpretation": "Local PigBench YOLOv8-s detector checkpoint, not random/untrained model.",
    },
    {
        "metric": "device",
        "value": device,
        "interpretation": "Inference device.",
    },
    {
        "metric": "pca_components",
        "value": len(pca_summary),
        "interpretation": "Compact PCA features generated for downstream visualization/comparison.",
    },
]

summary = pd.DataFrame(summary_rows)
summary_path = STATS / "week6_detector_backbone_crop_embedding_summary.csv"
safe_to_csv(summary, summary_path)

# Verification.
finite_ok = bool(np.isfinite(embeddings).all()) if embeddings.size else False
dim_ok = bool(embeddings.shape[1] == 896) if embeddings.ndim == 2 and embeddings.shape[0] else False
count_ok = bool(len(metadata) == 540)

verification = pd.DataFrame([
    {
        "check": "embedding_count_540",
        "status": "PASS" if count_ok else "WARN",
        "observed": len(metadata),
        "expected": 540,
    },
    {
        "check": "embedding_dim_896",
        "status": "PASS" if dim_ok else "WARN",
        "observed": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.shape[1] else 0,
        "expected": 896,
    },
    {
        "check": "finite_embeddings",
        "status": "PASS" if finite_ok else "FAIL",
        "observed": finite_ok,
        "expected": True,
    },
    {
        "check": "npy_exists",
        "status": "PASS" if npy_path.exists() else "FAIL",
        "observed": npy_path.exists(),
        "expected": True,
    },
    {
        "check": "wide_csv_exists",
        "status": "PASS" if wide_path.exists() else "FAIL",
        "observed": wide_path.exists(),
        "expected": True,
    },
])

verification_path = STATS / "week6_detector_backbone_crop_embedding_verification.csv"
safe_to_csv(verification, verification_path)

# Notes.
note_path = NOTES / "week6_detector_backbone_crop_embeddings_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Detector-Backbone Crop Embeddings\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This step extracts learned crop embeddings for all detector bboxes using the locally available PigBench YOLOv8-s detector checkpoint. "
        "The method uses `model.extract_feat` and global average pooling over feature maps. "
        "This is not a random/untrained embedding route.\n\n"
    )

    f.write("## Method\n\n")
    f.write(f"- Config: `{CONFIG}`\n")
    f.write(f"- Checkpoint: `{CHECKPOINT}`\n")
    f.write("- Embedding method: `mmdet_yolov8s_pig_detector_backbone_global_avg_pool`\n")
    f.write("- Crop source: detector bbox crops from scanpoint frames\n")
    f.write("- Feature pooling: adaptive global average pooling per feature map, concatenated\n\n")

    f.write("## Outputs\n\n")
    f.write(f"- Wide embedding CSV: `{wide_path}`\n")
    f.write(f"- Embedding NPY: `{npy_path}`\n")
    f.write(f"- Metadata CSV: `{metadata_path}`\n")
    f.write(f"- PCA feature CSV: `{pca_path}`\n")
    f.write(f"- Summary CSV: `{summary_path}`\n")
    f.write(f"- Verification CSV: `{verification_path}`\n\n")

    f.write("## Summary\n\n")
    f.write(summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Verification\n\n")
    f.write(verification.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The detector-backbone embeddings provide a learned crop representation derived from the pig detector checkpoint. "
        "They can be used as a feature extractor comparison input alongside bbox geometry, ROI proxies, crop descriptors, marker features, and segmentation features. "
        "They should be described as detector-backed crop embeddings, not as DINO/CLIP/SAM features.\n"
    )

print("Saved:")
print(wide_path)
print(npy_path)
print(metadata_path)
print(pca_path)
print(pca_summary_path)
print(summary_path)
print(verification_path)
print(note_path)

print()
print("=== Embedding summary ===")
print(summary.to_string(index=False))

print()
print("=== Verification ===")
print(verification.to_string(index=False))
