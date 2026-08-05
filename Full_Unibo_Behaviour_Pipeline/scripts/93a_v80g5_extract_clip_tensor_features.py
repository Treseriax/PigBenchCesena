from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

F = Path.home() / "PigBench" / "Full_Unibo_Behaviour_Pipeline"
IN_DIR = F / "outputs/v80_final_project_completion/05_clip_based_videomae/v80g4_limited_clip_tensor_extraction"
OUT = F / "outputs/v80_final_project_completion/05_clip_based_videomae/v80g5_limited_clip_visual_feature_baseline"
OUT.mkdir(parents=True, exist_ok=True)

report = pd.read_csv(IN_DIR / "v80g4_limited_extraction_report.csv").fillna("")
report = report[report["tensor_status"] == "OK"].copy()

rows = []

for _, r in report.iterrows():
    p = Path(r["tensor_path"])
    data = np.load(p)
    frames = data["frames"].astype(np.float32) / 255.0

    # T,H,W,C
    mean_rgb = frames.mean(axis=(0, 1, 2))
    std_rgb = frames.std(axis=(0, 1, 2))

    first = frames[0]
    last = frames[-1]
    diff = np.abs(last - first)

    temporal_diff = np.abs(np.diff(frames, axis=0)).mean(axis=(0, 1, 2))
    brightness = frames.mean(axis=(1, 2, 3))

    row = {
        "clip_id": r["clip_id"],
        "split": r["split"],
        "video_id": r["video_id"],
        "video_filename": r["video_filename"],
        "behaviour_label": r["behaviour_label"],
        "mean_r": float(mean_rgb[0]),
        "mean_g": float(mean_rgb[1]),
        "mean_b": float(mean_rgb[2]),
        "std_r": float(std_rgb[0]),
        "std_g": float(std_rgb[1]),
        "std_b": float(std_rgb[2]),
        "first_last_diff_mean": float(diff.mean()),
        "temporal_diff_r": float(temporal_diff[0]),
        "temporal_diff_g": float(temporal_diff[1]),
        "temporal_diff_b": float(temporal_diff[2]),
        "brightness_mean": float(brightness.mean()),
        "brightness_std": float(brightness.std()),
        "brightness_min": float(brightness.min()),
        "brightness_max": float(brightness.max()),
        "feature_source": "32_sampled_rgb_frames_224x224_npz",
        "claim_scope": "clip_visual_temporal_feature_baseline_not_final_videomae",
    }
    rows.append(row)

features = pd.DataFrame(rows)
features.to_csv(OUT / "v80g5_limited_clip_visual_temporal_features.csv", index=False)

summary = (
    features.groupby(["split", "behaviour_label"], dropna=False)
    .agg(clips=("clip_id", "count"))
    .reset_index()
)
summary.to_csv(OUT / "v80g5_feature_class_split_summary.csv", index=False)

decision = pd.DataFrame([{
    "v80g5a_decision": "clip_visual_temporal_features_created",
    "feature_rows": len(features),
    "train_clips": int((features["split"] == "train").sum()),
    "val_clips": int((features["split"] == "val").sum()),
    "test_clips": int((features["split"] == "test").sum()),
    "classes": features["behaviour_label"].nunique(),
    "ready_for_v80g5b_classifier": True,
    "claim_scope": "feature_extraction_only_no_model_training_yet",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
decision.to_csv(OUT / "v80g5a_decision_summary.csv", index=False)

note = F / "notes/v80g5a_clip_feature_extraction_notes.md"
note.write_text(
    "# v80g5-A Clip Visual-Temporal Feature Extraction\n\n"
    f"- Decision: clip_visual_temporal_features_created\n"
    f"- Feature rows: {len(features)}\n"
    f"- Classes: {features['behaviour_label'].nunique()}\n"
    "- Feature source: 32 sampled RGB frames per 10-second clip\n"
    "- Claim scope: feature extraction only, no final VideoMAE claim\n",
    encoding="utf-8",
)

print(decision.to_string(index=False))
print(summary.to_string(index=False))
