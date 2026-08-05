from pathlib import Path
from datetime import datetime
import json
import time

import cv2
import numpy as np
import pandas as pd

F = Path.home() / "PigBench" / "Full_Unibo_Behaviour_Pipeline"
V80 = F / "outputs/v80_final_project_completion"
V81 = F / "outputs/v81_performance_improvement"
OUT = V81 / "01_full_visual_temporal_features"
OUT.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(
    V80 / "05_clip_based_videomae/v80g2_split_A_clip_sampling_manifest.csv"
).fillna("")

manifest = manifest[manifest["sampling_status"] == "READY_FOR_EXTRACTION"].copy()

# Balanced-ish smoke: first 60 clips across train/val/test order.
smoke = pd.concat([
    manifest[manifest["split"] == "train"].head(30),
    manifest[manifest["split"] == "val"].head(15),
    manifest[manifest["split"] == "test"].head(15),
], ignore_index=True)

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def color_hist_features(frames_rgb, bins=8):
    feats = {}
    for ci, cname in enumerate(["r", "g", "b"]):
        vals = frames_rgb[:, :, :, ci].reshape(-1)
        hist, _ = np.histogram(vals, bins=bins, range=(0.0, 1.0), density=True)
        for i, h in enumerate(hist):
            feats[f"hist_{cname}_{i}"] = float(h)
    return feats

def grid_brightness_features(frames_rgb):
    feats = {}
    gray = frames_rgb.mean(axis=3)
    h = gray.shape[1]
    w = gray.shape[2]
    cells = {
        "tl": gray[:, :h//2, :w//2],
        "tr": gray[:, :h//2, w//2:],
        "bl": gray[:, h//2:, :w//2],
        "br": gray[:, h//2:, w//2:],
    }
    for name, cell in cells.items():
        feats[f"grid_{name}_brightness_mean"] = float(cell.mean())
        feats[f"grid_{name}_brightness_std"] = float(cell.std())
    return feats

def edge_features(frames_rgb):
    vals = []
    # use every 4th frame for speed
    for fr in frames_rgb[::4]:
        gray = (fr.mean(axis=2) * 255.0).astype(np.uint8)
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(sx * sx + sy * sy)
        vals.append(float(mag.mean()))
    vals = np.array(vals, dtype=np.float32)
    return {
        "edge_mean": float(vals.mean()) if len(vals) else 0.0,
        "edge_std": float(vals.std()) if len(vals) else 0.0,
        "edge_max": float(vals.max()) if len(vals) else 0.0,
    }

def temporal_segment_features(frames_rgb, segments=4):
    feats = {}
    chunks = np.array_split(frames_rgb, segments, axis=0)
    for i, ch in enumerate(chunks):
        bright = ch.mean(axis=(1, 2, 3))
        feats[f"seg{i}_brightness_mean"] = float(bright.mean())
        feats[f"seg{i}_brightness_std"] = float(bright.std())
        if len(ch) > 1:
            diff = np.abs(np.diff(ch, axis=0)).mean()
        else:
            diff = 0.0
        feats[f"seg{i}_motion_mean"] = float(diff)
    return feats

def extract_clip_features(row):
    video_path = Path(str(row["video_path"]))
    frame_indices = json.loads(str(row["sample_frame_indices"]))
    spatial = int(row["spatial_size"])

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    errors = []

    # use 112x112 feature resolution for speed; sampling is still from 32 planned frames.
    target_size = 112

    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            errors.append(f"read_failed_{i}_{idx}")
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
        frames.append(frame.astype(np.float32) / 255.0)

    cap.release()

    if len(frames) != len(frame_indices):
        return None, ";".join(errors)

    arr = np.stack(frames, axis=0)

    rgb_mean = arr.mean(axis=(0, 1, 2))
    rgb_std = arr.std(axis=(0, 1, 2))
    rgb_min = arr.min(axis=(0, 1, 2))
    rgb_max = arr.max(axis=(0, 1, 2))

    brightness = arr.mean(axis=(1, 2, 3))
    diffs = np.abs(np.diff(arr, axis=0))
    motion_per_step = diffs.mean(axis=(1, 2, 3))

    first_last_diff = np.abs(arr[-1] - arr[0])

    feats = {
        "clip_id": row["clip_id"],
        "split_A": row["split"],
        "video_id": row["video_id"],
        "video_filename": row["video_filename"],
        "tlc_camera": row["tlc_camera"],
        "room_pen": row["room_pen"],
        "identity_colour": row["identity_colour"],
        "behaviour_label": row["behaviour_label"],
        "clip_start_sec": safe_float(row["clip_start_sec"]),
        "clip_end_sec": safe_float(row["clip_end_sec"]),
        "num_frames": len(frames),
        "feature_resolution": target_size,

        "mean_r": float(rgb_mean[0]),
        "mean_g": float(rgb_mean[1]),
        "mean_b": float(rgb_mean[2]),
        "std_r": float(rgb_std[0]),
        "std_g": float(rgb_std[1]),
        "std_b": float(rgb_std[2]),
        "min_r": float(rgb_min[0]),
        "min_g": float(rgb_min[1]),
        "min_b": float(rgb_min[2]),
        "max_r": float(rgb_max[0]),
        "max_g": float(rgb_max[1]),
        "max_b": float(rgb_max[2]),

        "brightness_mean": float(brightness.mean()),
        "brightness_std": float(brightness.std()),
        "brightness_min": float(brightness.min()),
        "brightness_max": float(brightness.max()),

        "motion_mean": float(motion_per_step.mean()),
        "motion_std": float(motion_per_step.std()),
        "motion_min": float(motion_per_step.min()),
        "motion_max": float(motion_per_step.max()),

        "first_last_diff_mean": float(first_last_diff.mean()),
        "first_last_diff_std": float(first_last_diff.std()),
        "claim_scope": "v81_visual_temporal_features_no_model_training",
    }

    feats.update(color_hist_features(arr, bins=8))
    feats.update(grid_brightness_features(arr))
    feats.update(edge_features(arr))
    feats.update(temporal_segment_features(arr, segments=4))

    return feats, ""

rows = []
issues = []

start_time = time.time()

for i, (_, r) in enumerate(smoke.iterrows(), start=1):
    feats, err = extract_clip_features(r)
    if feats is None:
        issues.append({
            "clip_id": r["clip_id"],
            "issue_type": "feature_extraction_failed",
            "severity": "hard",
            "detail": err,
        })
    else:
        rows.append(feats)

    if i % 10 == 0:
        print(f"processed {i}/{len(smoke)} clips")

elapsed = time.time() - start_time

features = pd.DataFrame(rows)
features.to_csv(OUT / "v81b1_smoke_visual_temporal_features.csv", index=False)

if not issues:
    issues = [{
        "clip_id": "none",
        "issue_type": "none",
        "severity": "info",
        "detail": "60-clip visual-temporal feature smoke passed",
    }]

pd.DataFrame(issues).to_csv(OUT / "v81b1_smoke_issues.csv", index=False)

hard = sum(1 for x in issues if x["severity"] == "hard")

summary = pd.DataFrame([{
    "v81b1_decision": "visual_temporal_feature_smoke_passed" if hard == 0 else "visual_temporal_feature_smoke_has_blocking_issues",
    "requested_clips": len(smoke),
    "feature_rows": len(features),
    "feature_columns": len(features.columns) if len(features) else 0,
    "elapsed_seconds": round(elapsed, 2),
    "seconds_per_clip": round(elapsed / max(len(smoke), 1), 4),
    "hard_issue_count": hard,
    "ready_for_v81b2_full_feature_extraction": hard == 0,
    "claim_scope": "feature_extraction_smoke_only_no_training_yet",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
summary.to_csv(OUT / "v81b1_smoke_decision_summary.csv", index=False)

note = F / "notes/v81b1_visual_temporal_feature_smoke_notes.md"
note.write_text(
    "# v81b-1 Visual-Temporal Feature Smoke\n\n"
    f"- Decision: {summary.iloc[0]['v81b1_decision']}\n"
    f"- Requested clips: {len(smoke)}\n"
    f"- Feature rows: {len(features)}\n"
    f"- Feature columns: {summary.iloc[0]['feature_columns']}\n"
    f"- Elapsed seconds: {summary.iloc[0]['elapsed_seconds']}\n"
    f"- Seconds per clip: {summary.iloc[0]['seconds_per_clip']}\n"
    f"- Hard issues: {hard}\n"
    f"- Ready for v81b2 full feature extraction: {hard == 0}\n\n"
    "This stage tests the stronger visual-temporal feature extractor on a 60-clip subset. "
    "No model training is performed here.\n",
    encoding="utf-8",
)

print(summary.to_string(index=False))
print("=== class/split counts ===")
print(features.groupby(["split_A", "behaviour_label"]).size().reset_index(name="clips").to_string(index=False))
print("=== issues ===")
print(pd.DataFrame(issues).to_string(index=False))
