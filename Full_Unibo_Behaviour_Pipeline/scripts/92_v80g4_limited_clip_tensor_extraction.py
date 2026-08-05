from pathlib import Path
from datetime import datetime
import json
import cv2
import numpy as np
import pandas as pd

F = Path.home() / "PigBench" / "Full_Unibo_Behaviour_Pipeline"
O = F / "outputs/v80_final_project_completion/05_clip_based_videomae"
OUT = O / "v80g4_limited_clip_tensor_extraction"
TENSOR_DIR = OUT / "clip_tensors_npz"
PREVIEW_DIR = OUT / "preview_contact_sheets"

for d in [TENSOR_DIR, PREVIEW_DIR]:
    d.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(O / "v80g2_split_A_clip_sampling_manifest.csv").fillna("")
manifest = manifest[manifest["sampling_status"] == "READY_FOR_EXTRACTION"].copy()

LIMITS = {
    "train": 3,
    "val": 2,
    "test": 2,
}

selected_parts = []

for split, max_per_class in LIMITS.items():
    sub = manifest[manifest["split"] == split].copy()
    for label, g in sub.groupby("behaviour_label"):
        selected_parts.append(g.sort_values(["video_id", "clip_id"]).head(max_per_class))

selected = pd.concat(selected_parts, ignore_index=True)
selected = selected.drop_duplicates("clip_id").reset_index(drop=True)
selected.to_csv(OUT / "v80g4_limited_selected_clip_manifest.csv", index=False)

def make_contact_sheet(frames, cols=8):
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    rows = int(np.ceil(len(frames) / cols))
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, fr in enumerate(frames):
        r = i // cols
        c = i % cols
        sheet[r*h:(r+1)*h, c*w:(c+1)*w] = fr
    return sheet

rows = []

for idx, r in selected.iterrows():
    clip_id = str(r["clip_id"])
    video_path = Path(str(r["video_path"]))
    frame_indices = json.loads(str(r["sample_frame_indices"]))
    sample_times = json.loads(str(r["sample_times_sec"]))
    spatial_size = int(r["spatial_size"])

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    errors = []

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()

        if not ok or frame is None:
            errors.append(f"frame_{i}_idx_{frame_idx}_read_failed")
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (spatial_size, spatial_size), interpolation=cv2.INTER_AREA)
        frames.append(frame)

    cap.release()

    status = "OK" if len(frames) == len(frame_indices) else "INCOMPLETE"
    tensor_path = TENSOR_DIR / f"{clip_id}.npz"
    preview_path = PREVIEW_DIR / f"{clip_id}_contact_sheet.jpg"

    arr_shape = ""

    if frames:
        arr = np.stack(frames, axis=0).astype(np.uint8)
        arr_shape = str(arr.shape)

        np.savez_compressed(
            tensor_path,
            frames=arr,
            clip_id=clip_id,
            split=str(r["split"]),
            video_id=str(r["video_id"]),
            video_filename=str(r["video_filename"]),
            behaviour_label=str(r["behaviour_label"]),
            tlc_camera=str(r["tlc_camera"]),
            room_pen=str(r["room_pen"]),
            identity_colour=str(r["identity_colour"]),
            sample_times_sec=np.array(sample_times, dtype=np.float32),
            sample_frame_indices=np.array(frame_indices, dtype=np.int64),
        )

        if idx < 24:
            sheet = make_contact_sheet(frames, cols=8)
            if sheet is not None:
                cv2.imwrite(str(preview_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    rows.append({
        "clip_id": clip_id,
        "split": str(r["split"]),
        "video_id": str(r["video_id"]),
        "video_filename": str(r["video_filename"]),
        "behaviour_label": str(r["behaviour_label"]),
        "expected_frames": len(frame_indices),
        "read_frames": len(frames),
        "tensor_status": status,
        "array_shape": arr_shape,
        "tensor_path": str(tensor_path),
        "preview_path": str(preview_path) if preview_path.exists() else "",
        "errors": ";".join(errors),
    })

report = pd.DataFrame(rows)
report.to_csv(OUT / "v80g4_limited_extraction_report.csv", index=False)

summary = (
    report.groupby(["split", "behaviour_label"], dropna=False)
    .agg(clips=("clip_id", "count"), ok_clips=("tensor_status", lambda x: int((x == "OK").sum())))
    .reset_index()
)
summary.to_csv(OUT / "v80g4_limited_class_split_summary.csv", index=False)

bad = int((report["tensor_status"] != "OK").sum())

issues = []
if bad > 0:
    issues.append({
        "item": "limited_extraction",
        "issue_type": "incomplete_clip_tensors",
        "severity": "hard",
        "detail": str(bad),
    })

if not issues:
    issues = [{
        "item": "none",
        "issue_type": "none",
        "severity": "info",
        "detail": "limited clip tensor extraction passed",
    }]

pd.DataFrame(issues).to_csv(OUT / "v80g4_issues.csv", index=False)

hard = sum(1 for x in issues if x["severity"] == "hard")

decision = pd.DataFrame([{
    "v80g4_decision": "limited_clip_tensor_extraction_passed" if hard == 0 else "limited_clip_tensor_extraction_has_blocking_issues",
    "selected_clips": len(report),
    "successful_clip_tensors": int((report["tensor_status"] == "OK").sum()),
    "failed_clip_tensors": bad,
    "num_frames_per_clip": 32,
    "spatial_size": 224,
    "train_clips": int((report["split"] == "train").sum()),
    "val_clips": int((report["split"] == "val").sum()),
    "test_clips": int((report["split"] == "test").sum()),
    "hard_issue_count": hard,
    "ready_for_v80g5_limited_clip_model_training": hard == 0,
    "claim_scope": "limited_clip_tensor_extraction_no_final_videomae_claim_yet",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

decision.to_csv(OUT / "v80g4_decision_summary.csv", index=False)

note = F / "notes/v80g4_limited_clip_tensor_extraction_notes.md"
note.write_text(
    "# v80g4 Limited Clip Tensor Extraction\n\n"
    f"- Decision: {decision.iloc[0]['v80g4_decision']}\n"
    f"- Selected clips: {len(report)}\n"
    f"- Successful clip tensors: {decision.iloc[0]['successful_clip_tensors']}\n"
    f"- Failed clip tensors: {bad}\n"
    "- Frames per clip: 32\n"
    "- Spatial size: 224x224\n"
    f"- Train clips: {decision.iloc[0]['train_clips']}\n"
    f"- Val clips: {decision.iloc[0]['val_clips']}\n"
    f"- Test clips: {decision.iloc[0]['test_clips']}\n"
    f"- Hard issues: {hard}\n"
    f"- Ready for v80g5 limited clip model training: {hard == 0}\n\n"
    "This stage extracts a limited, class-aware clip tensor dataset from Split A for the first clip-based model pipeline test. "
    "It does not claim a final VideoMAE model yet.\n",
    encoding="utf-8",
)

print(decision.to_string(index=False))
print("=== class/split summary ===")
print(summary.to_string(index=False))
print("=== issues ===")
print(pd.DataFrame(issues).to_string(index=False))
