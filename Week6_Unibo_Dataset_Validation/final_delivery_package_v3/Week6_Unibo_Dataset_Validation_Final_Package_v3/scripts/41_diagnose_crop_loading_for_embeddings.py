from pathlib import Path
import csv
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


def rel(path):
    try:
        return str(Path(path).relative_to(W6))
    except Exception:
        return str(path)


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


def get_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def try_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


dets = pd.read_csv(DETS_PATH)
frames = pd.read_csv(FRAME_INDEX_PATH)

diagnostics = []

diagnostics.append({
    "check": "cv2_available",
    "status": "PASS" if CV2_AVAILABLE else "FAIL",
    "details": str(CV2_AVAILABLE),
})

diagnostics.append({
    "check": "detections_table_rows",
    "status": "PASS" if len(dets) == 540 else "WARN",
    "details": f"rows={len(dets)}, columns={list(dets.columns)}",
})

diagnostics.append({
    "check": "frame_index_rows",
    "status": "PASS" if len(frames) == 72 else "WARN",
    "details": f"rows={len(frames)}, columns={list(frames.columns)}",
})

scan_col = get_col(dets, ["scan_frame_id", "frame_id"])
x1_col = get_col(dets, ["x1", "bbox_x1"])
y1_col = get_col(dets, ["y1", "bbox_y1"])
x2_col = get_col(dets, ["x2", "bbox_x2"])
y2_col = get_col(dets, ["y2", "bbox_y2"])
score_col = get_col(dets, ["score", "confidence"])

diagnostics.append({
    "check": "column_detection",
    "status": "PASS" if all([scan_col, x1_col, y1_col, x2_col, y2_col]) else "FAIL",
    "details": f"scan_col={scan_col}, x1={x1_col}, y1={y1_col}, x2={x2_col}, y2={y2_col}, score={score_col}",
})

# Merge carefully.
merged = dets.copy()

if scan_col and "scan_frame_id" in frames.columns:
    frame_cols = ["scan_frame_id"]
    for c in [
        "frame_image_path",
        "scan_frame_image_path",
        "image_path",
        "video_id",
        "timestamp",
        "frame_index",
        "timestamp_sec_in_video",
    ]:
        if c in frames.columns:
            frame_cols.append(c)

    frame_cols = list(dict.fromkeys(frame_cols))
    merged = merged.merge(
        frames[frame_cols],
        left_on=scan_col,
        right_on="scan_frame_id",
        how="left",
        suffixes=("", "_from_frame_index"),
    )

diagnostics.append({
    "check": "merged_table",
    "status": "PASS",
    "details": f"rows={len(merged)}, columns={list(merged.columns)}",
})

# Find image path columns.
image_path_candidates = [
    c for c in merged.columns
    if "image" in c.lower() and "path" in c.lower()
]

diagnostics.append({
    "check": "image_path_candidate_columns",
    "status": "PASS" if image_path_candidates else "FAIL",
    "details": str(image_path_candidates),
})

# Check path resolvability.
path_rows = []

for c in image_path_candidates:
    sample_values = merged[c].dropna().astype(str).head(20).tolist()
    resolved_count = 0
    readable_count = 0

    for v in sample_values:
        rp = resolve_path(v)

        if rp is not None:
            resolved_count += 1

            if CV2_AVAILABLE:
                img = cv2.imread(str(rp))
                if img is not None:
                    readable_count += 1

    path_rows.append({
        "column": c,
        "sample_non_null": len(sample_values),
        "resolved_count_first20": resolved_count,
        "cv2_readable_first20": readable_count,
        "sample_values": " | ".join(sample_values[:5]),
    })

path_df = pd.DataFrame(path_rows)

diagnostics.append({
    "check": "path_resolvability",
    "status": "PASS" if len(path_df) and path_df["cv2_readable_first20"].max() > 0 else "FAIL",
    "details": path_df.to_dict(orient="records") if len(path_df) else "no image path columns",
})

# BBox numeric/crop check.
crop_test_rows = []
usable_crop_count = 0

best_image_col = None

if len(path_df):
    path_df_sorted = path_df.sort_values("cv2_readable_first20", ascending=False)
    if path_df_sorted.iloc[0]["cv2_readable_first20"] > 0:
        best_image_col = path_df_sorted.iloc[0]["column"]

if best_image_col and all([x1_col, y1_col, x2_col, y2_col]):
    test = merged.copy()

    if score_col:
        test = test.sort_values(score_col, ascending=False)

    for idx, r in test.head(100).iterrows():
        p = resolve_path(r.get(best_image_col, ""))

        row = {
            "row_index": idx,
            "scan_frame_id": r.get(scan_col, ""),
            "det_id": r.get("det_id", r.get("detection_id", "")),
            "image_col": best_image_col,
            "image_value": r.get(best_image_col, ""),
            "path_resolved": p is not None,
            "image_readable": False,
            "bbox_raw": str([r.get(x1_col), r.get(y1_col), r.get(x2_col), r.get(y2_col)]),
            "bbox_numeric": False,
            "crop_shape": "",
            "usable_crop": False,
        }

        if p is not None and CV2_AVAILABLE:
            img = cv2.imread(str(p))

            if img is not None:
                row["image_readable"] = True
                h, w = img.shape[:2]

                x1 = try_float(r.get(x1_col))
                y1 = try_float(r.get(y1_col))
                x2 = try_float(r.get(x2_col))
                y2 = try_float(r.get(y2_col))

                if not any(pd.isna(v) for v in [x1, y1, x2, y2]):
                    row["bbox_numeric"] = True

                    x1i = max(0, min(w - 1, int(round(x1))))
                    y1i = max(0, min(h - 1, int(round(y1))))
                    x2i = max(0, min(w, int(round(x2))))
                    y2i = max(0, min(h, int(round(y2))))

                    crop = img[y1i:y2i, x1i:x2i]
                    row["crop_shape"] = str(crop.shape)

                    if crop.size > 0 and crop.shape[0] >= 16 and crop.shape[1] >= 16:
                        row["usable_crop"] = True
                        usable_crop_count += 1

        crop_test_rows.append(row)

crop_test_df = pd.DataFrame(crop_test_rows)

diagnostics.append({
    "check": "usable_crop_first100",
    "status": "PASS" if usable_crop_count > 0 else "FAIL",
    "details": f"best_image_col={best_image_col}, usable_crop_count_first100={usable_crop_count}",
})

diagnostics_df = pd.DataFrame(diagnostics)

diagnostics_path = STATS / "week6_embedding_crop_loading_diagnostics.csv"
path_check_path = STATS / "week6_embedding_image_path_resolution_check.csv"
crop_check_path = STATS / "week6_embedding_crop_test_first100.csv"

safe_to_csv(diagnostics_df, diagnostics_path)
safe_to_csv(path_df, path_check_path)
safe_to_csv(crop_test_df, crop_check_path)

note_path = NOTES / "week6_embedding_crop_loading_diagnostics_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Embedding Crop Loading Diagnostics\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "The learned embedding smoke test loaded the detector model but failed to find a usable crop. "
        "This diagnostic checks detection columns, frame image paths, path resolution, image readability, bbox numeric conversion, and crop extraction.\n\n"
    )

    f.write("## Diagnostics\n\n")
    f.write(diagnostics_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Image path resolution\n\n")
    f.write(path_df.to_markdown(index=False) if len(path_df) else "No image path candidates found.")
    f.write("\n\n")

    f.write("## Crop test preview\n\n")
    f.write(crop_test_df.head(20).to_markdown(index=False) if len(crop_test_df) else "No crop test rows generated.")
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    if usable_crop_count > 0:
        f.write(
            f"Crop loading is fixable. Best image path column is `{best_image_col}` and usable crops were found. "
            "The embedding extraction script should use this resolved image path logic.\n"
        )
    else:
        f.write(
            "No usable crop was found in the first 100 detections. "
            "Inspect image path columns and bbox coordinate columns before running full embedding extraction.\n"
        )

print("Saved:")
print(diagnostics_path)
print(path_check_path)
print(crop_check_path)
print(note_path)

print()
print("=== Diagnostics ===")
print(diagnostics_df.to_string(index=False))

print()
print("=== Image path check ===")
print(path_df.to_string(index=False) if len(path_df) else "No image path candidates.")

print()
print("=== Crop test preview ===")
print(crop_test_df.head(20).to_string(index=False) if len(crop_test_df) else "No crop test rows.")
