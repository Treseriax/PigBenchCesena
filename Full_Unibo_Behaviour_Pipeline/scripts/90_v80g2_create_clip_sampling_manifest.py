from pathlib import Path
from datetime import datetime
import json
import cv2
import pandas as pd
import numpy as np

F = Path.home() / "PigBench" / "Full_Unibo_Behaviour_Pipeline"
O = F / "outputs/v80_final_project_completion/05_clip_based_videomae"
O.mkdir(parents=True, exist_ok=True)

split_a_path = F / "outputs/v80_final_project_completion/03_splits/v80e_split_A_grouped_video_level.csv"
split_b_path = F / "outputs/v80_final_project_completion/03_splits/v80e_split_B_cross_camera_pen_level.csv"

NUM_FRAMES = 32
SPATIAL_SIZE = 224
RAW_ROOT = Path("/work/pig/datasets/Unibo")

def video_meta(video_filename):
    path = RAW_ROOT / str(video_filename)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {
            "video_path": str(path),
            "video_exists": path.exists(),
            "opencv_open": False,
            "fps": 0.0,
            "frame_count": 0,
            "width": 0,
            "height": 0,
        }

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()

    return {
        "video_path": str(path),
        "video_exists": path.exists(),
        "opencv_open": True,
        "fps": float(fps),
        "frame_count": int(frame_count),
        "width": int(width),
        "height": int(height),
    }

def build_manifest(split_path, split_col, protocol_name):
    df = pd.read_csv(split_path).fillna("")

    rows = []
    video_cache = {}

    for _, r in df.iterrows():
        vf = str(r["video_filename"])
        if vf not in video_cache:
            video_cache[vf] = video_meta(vf)

        meta = video_cache[vf]
        start = float(r["clip_start_sec_in_video"])
        end = float(r["clip_end_sec_in_video"])

        if end <= start:
            sample_times = []
            sample_frame_indices = []
            status = "INVALID_CLIP_TIME"
        elif not meta["opencv_open"] or meta["fps"] <= 0:
            sample_times = []
            sample_frame_indices = []
            status = "VIDEO_NOT_READABLE"
        else:
            sample_times = np.linspace(start, end, NUM_FRAMES, endpoint=False) + ((end - start) / NUM_FRAMES / 2.0)
            sample_frame_indices = [int(round(t * meta["fps"])) for t in sample_times]
            status = "READY_FOR_EXTRACTION"

        rows.append({
            "split_protocol": protocol_name,
            "split": r[split_col],
            "clip_id": r["clip_id"],
            "video_id": r["video_id"],
            "video_filename": vf,
            "video_path": meta["video_path"],
            "video_exists": meta["video_exists"],
            "opencv_open": meta["opencv_open"],
            "fps": meta["fps"],
            "frame_count": meta["frame_count"],
            "width": meta["width"],
            "height": meta["height"],
            "tlc_camera": r["tlc_camera"],
            "room_pen": r["room_pen"],
            "identity_colour": r["identity_colour"],
            "behaviour_label": r["behaviour_label"],
            "clip_start_sec": start,
            "clip_end_sec": end,
            "clip_duration_sec": end - start,
            "num_sampled_frames": NUM_FRAMES,
            "spatial_size": SPATIAL_SIZE,
            "sample_times_sec": json.dumps([round(float(x), 4) for x in sample_times]),
            "sample_frame_indices": json.dumps([int(x) for x in sample_frame_indices]),
            "sampling_status": status,
            "identity_dataset_status": r["identity_dataset_status"],
            "usable_for_behaviour_clip_dataset": r["usable_for_behaviour_clip_dataset"],
            "claim_boundary": "clip_frame_sampling_manifest_no_training_yet",
        })

    return pd.DataFrame(rows)

A = build_manifest(split_a_path, "split_A_grouped_video", "split_A_grouped_video_level")
B = build_manifest(split_b_path, "split_B_cross_camera", "split_B_cross_camera_pen_level")

A.to_csv(O / "v80g2_split_A_clip_sampling_manifest.csv", index=False)
B.to_csv(O / "v80g2_split_B_clip_sampling_manifest.csv", index=False)

summary = pd.DataFrame([
    {
        "manifest": "split_A",
        "rows": len(A),
        "ready_rows": int((A["sampling_status"] == "READY_FOR_EXTRACTION").sum()),
        "bad_rows": int((A["sampling_status"] != "READY_FOR_EXTRACTION").sum()),
        "clips": A["clip_id"].nunique(),
        "videos": A["video_id"].nunique(),
        "classes": A["behaviour_label"].nunique(),
        "num_frames": NUM_FRAMES,
        "spatial_size": SPATIAL_SIZE,
    },
    {
        "manifest": "split_B",
        "rows": len(B),
        "ready_rows": int((B["sampling_status"] == "READY_FOR_EXTRACTION").sum()),
        "bad_rows": int((B["sampling_status"] != "READY_FOR_EXTRACTION").sum()),
        "clips": B["clip_id"].nunique(),
        "videos": B["video_id"].nunique(),
        "classes": B["behaviour_label"].nunique(),
        "num_frames": NUM_FRAMES,
        "spatial_size": SPATIAL_SIZE,
    },
])
summary.to_csv(O / "v80g2_clip_sampling_manifest_summary.csv", index=False)

issues = []

for name, data in [("split_A", A), ("split_B", B)]:
    bad = int((data["sampling_status"] != "READY_FOR_EXTRACTION").sum())
    if bad > 0:
        issues.append({
            "item": name,
            "issue_type": "sampling_manifest_has_bad_rows",
            "severity": "hard",
            "detail": str(bad),
        })

if not issues:
    issues = [{
        "item": "none",
        "issue_type": "none",
        "severity": "info",
        "detail": "all clips ready for frame extraction",
    }]

pd.DataFrame(issues).to_csv(O / "v80g2_issues.csv", index=False)

hard = sum(1 for x in issues if x["severity"] == "hard")

decision = pd.DataFrame([{
    "v80g2_decision": "clip_sampling_manifest_created" if hard == 0 else "clip_sampling_manifest_has_blocking_issues",
    "split_A_rows": len(A),
    "split_B_rows": len(B),
    "num_sampled_frames_per_clip": NUM_FRAMES,
    "spatial_size": SPATIAL_SIZE,
    "split_A_ready_rows": int((A["sampling_status"] == "READY_FOR_EXTRACTION").sum()),
    "split_B_ready_rows": int((B["sampling_status"] == "READY_FOR_EXTRACTION").sum()),
    "hard_issue_count": hard,
    "ready_for_v80g3_pilot_frame_extraction": hard == 0,
    "claim_scope": "clip_sampling_manifest_no_model_training_yet",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
decision.to_csv(O / "v80g2_decision_summary.csv", index=False)

note = F / "notes/v80g2_clip_sampling_manifest_notes.md"
note.write_text(
    "# v80g2 Clip Frame Sampling Manifest\n\n"
    f"- Decision: {decision.iloc[0]['v80g2_decision']}\n"
    f"- Split A rows: {len(A)}\n"
    f"- Split B rows: {len(B)}\n"
    f"- Sampled frames per clip: {NUM_FRAMES}\n"
    f"- Spatial size target: {SPATIAL_SIZE}x{SPATIAL_SIZE}\n"
    f"- Split A ready rows: {decision.iloc[0]['split_A_ready_rows']}\n"
    f"- Split B ready rows: {decision.iloc[0]['split_B_ready_rows']}\n"
    f"- Hard issues: {hard}\n"
    f"- Ready for v80g3 pilot frame extraction: {hard == 0}\n\n"
    "This stage defines deterministic uniform frame sampling for each 10-second behaviour clip. "
    "No frames are extracted and no model is trained in this stage.\n",
    encoding="utf-8",
)

print(decision.to_string(index=False))
print("=== summary ===")
print(summary.to_string(index=False))
print("=== issues ===")
print(pd.DataFrame(issues).to_string(index=False))
