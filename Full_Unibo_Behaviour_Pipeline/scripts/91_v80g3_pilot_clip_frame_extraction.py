from pathlib import Path
from datetime import datetime
import json
import cv2
import numpy as np
import pandas as pd

F = Path.home() / "PigBench" / "Full_Unibo_Behaviour_Pipeline"
O = F / "outputs/v80_final_project_completion/05_clip_based_videomae"
PILOT = O / "v80g3_pilot_frame_extraction"
FRAME_DIR = PILOT / "frames"
TENSOR_DIR = PILOT / "clip_tensors_npz"
PREVIEW_DIR = PILOT / "preview_contact_sheets"

for d in [FRAME_DIR, TENSOR_DIR, PREVIEW_DIR]:
    d.mkdir(parents=True, exist_ok=True)

manifest_path = O / "v80g2_split_A_clip_sampling_manifest.csv"
df = pd.read_csv(manifest_path).fillna("")

# Balanced small pilot: first 2 train, 2 val, 2 test clips.
pilot_parts = []
for split in ["train", "val", "test"]:
    sub = df[(df["split"] == split) & (df["sampling_status"] == "READY_FOR_EXTRACTION")].head(2)
    pilot_parts.append(sub)

pilot = pd.concat(pilot_parts, ignore_index=True)

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

for _, r in pilot.iterrows():
    clip_id = str(r["clip_id"])
    video_path = Path(str(r["video_path"]))
    frame_indices = json.loads(str(r["sample_frame_indices"]))
    sample_times = json.loads(str(r["sample_times_sec"]))
    spatial_size = int(r["spatial_size"])

    cap = cv2.VideoCapture(str(video_path))
    clip_frames = []
    read_errors = []

    clip_frame_dir = FRAME_DIR / clip_id
    clip_frame_dir.mkdir(parents=True, exist_ok=True)

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()

        if not ok or frame is None:
            read_errors.append(f"frame_{i}_idx_{frame_idx}_read_failed")
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (spatial_size, spatial_size), interpolation=cv2.INTER_AREA)
        clip_frames.append(frame)

        out_jpg = clip_frame_dir / f"frame_{i:02d}_idx_{int(frame_idx):06d}.jpg"
        cv2.imwrite(str(out_jpg), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    cap.release()

    tensor_status = "OK" if len(clip_frames) == len(frame_indices) else "INCOMPLETE"
    arr_shape = ""

    if clip_frames:
        arr = np.stack(clip_frames, axis=0).astype(np.uint8)
        arr_shape = str(arr.shape)
        np.savez_compressed(
            TENSOR_DIR / f"{clip_id}.npz",
            frames=arr,
            clip_id=clip_id,
            behaviour_label=str(r["behaviour_label"]),
            split=str(r["split"]),
            video_id=str(r["video_id"]),
            sample_times_sec=np.array(sample_times, dtype=np.float32),
            sample_frame_indices=np.array(frame_indices, dtype=np.int64),
        )

        sheet = make_contact_sheet(clip_frames, cols=8)
        if sheet is not None:
            preview_path = PREVIEW_DIR / f"{clip_id}_contact_sheet.jpg"
            cv2.imwrite(str(preview_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    rows.append({
        "clip_id": clip_id,
        "split": str(r["split"]),
        "video_id": str(r["video_id"]),
        "video_filename": str(r["video_filename"]),
        "behaviour_label": str(r["behaviour_label"]),
        "expected_frames": len(frame_indices),
        "read_frames": len(clip_frames),
        "tensor_status": tensor_status,
        "array_shape": arr_shape,
        "frame_dir": str(clip_frame_dir),
        "tensor_path": str(TENSOR_DIR / f"{clip_id}.npz"),
        "preview_path": str(PREVIEW_DIR / f"{clip_id}_contact_sheet.jpg"),
        "read_errors": ";".join(read_errors),
    })

report = pd.DataFrame(rows)
report.to_csv(PILOT / "v80g3_pilot_extraction_report.csv", index=False)

bad = int((report["tensor_status"] != "OK").sum())

issues = []
if bad > 0:
    issues.append({
        "item": "pilot_extraction",
        "issue_type": "incomplete_clip_tensor",
        "severity": "hard",
        "detail": str(bad) + " pilot clips did not extract all expected frames",
    })

if not issues:
    issues = [{
        "item": "none",
        "issue_type": "none",
        "severity": "info",
        "detail": "pilot clip frame extraction passed",
    }]

pd.DataFrame(issues).to_csv(PILOT / "v80g3_issues.csv", index=False)

hard = sum(1 for x in issues if x["severity"] == "hard")

decision = pd.DataFrame([{
    "v80g3_decision": "pilot_clip_frame_extraction_passed" if hard == 0 else "pilot_clip_frame_extraction_has_blocking_issues",
    "pilot_clips": len(report),
    "expected_frames_per_clip": 32,
    "successful_clip_tensors": int((report["tensor_status"] == "OK").sum()),
    "failed_clip_tensors": bad,
    "hard_issue_count": hard,
    "ready_for_v80g4_full_or_limited_extraction": hard == 0,
    "claim_scope": "pilot_frame_extraction_only_no_model_training_yet",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

decision.to_csv(PILOT / "v80g3_decision_summary.csv", index=False)

note = F / "notes/v80g3_pilot_clip_frame_extraction_notes.md"
note.write_text(
    "# v80g3 Pilot Clip Frame Extraction\n\n"
    f"- Decision: {decision.iloc[0]['v80g3_decision']}\n"
    f"- Pilot clips: {len(report)}\n"
    "- Expected frames per clip: 32\n"
    f"- Successful clip tensors: {decision.iloc[0]['successful_clip_tensors']}\n"
    f"- Failed clip tensors: {bad}\n"
    f"- Hard issues: {hard}\n"
    f"- Ready for v80g4 full or limited extraction: {hard == 0}\n\n"
    "This stage extracts a small pilot set of uniformly sampled 224x224 RGB frames from the 10-second behaviour clips. "
    "It creates JPEG frames, compressed NPZ clip tensors, and contact-sheet previews. No model is trained here.\n",
    encoding="utf-8",
)

print(decision.to_string(index=False))
print("=== report ===")
print(report.to_string(index=False))
print("=== issues ===")
print(pd.DataFrame(issues).to_string(index=False))
