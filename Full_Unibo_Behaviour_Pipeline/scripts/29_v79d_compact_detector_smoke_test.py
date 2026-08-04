from pathlib import Path
from datetime import datetime
import sys
import csv
import traceback
import pandas as pd

ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

PILOT = FULL / "outputs/v79b_tracking_manifest_audit_and_run_plan/v79b_pilot_tracking_run_plan.csv"
CONFIG = ROOT / "detection/configs/yolov8/yolov8_s.py"
CKPT = ROOT / "detection/data/pretrained_weights/yolov8_pigs/yolov8_s.pth"

OUT = FULL / "outputs/v79d_compact_detector_smoke_test"
FRAMES = OUT / "annotated_frames"
OUT.mkdir(parents=True, exist_ok=True)
FRAMES.mkdir(parents=True, exist_ok=True)

DETECTIONS_CSV = OUT / "v79d_compact_detections.csv"
FRAME_CSV = OUT / "v79d_compact_frame_summary.csv"
DECISION_CSV = OUT / "v79d_compact_decision_summary.csv"
ISSUES_CSV = OUT / "v79d_compact_issues.csv"
NOTE = FULL / "notes/v79d_compact_detector_smoke_test_notes.md"
NOTE.parent.mkdir(parents=True, exist_ok=True)

SCORE_THR = 0.25
SAMPLE_SECONDS = [2, 10, 20]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "detection"))

def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s

def write(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")

issues = []

if not CONFIG.exists():
    issues.append({"item":"config","issue_type":"hard_missing_config","severity":"hard","detail":str(CONFIG)})

if not CKPT.exists():
    issues.append({"item":"checkpoint","issue_type":"hard_missing_checkpoint","severity":"hard","detail":str(CKPT)})

try:
    pilot = pd.read_csv(PILOT).fillna("")
except Exception as e:
    pilot = pd.DataFrame()
    issues.append({"item":"pilot_plan","issue_type":"hard_cannot_read_pilot_plan","severity":"hard","detail":str(e)})

for c in pilot.columns:
    if pilot[c].dtype == object:
        pilot[c] = pilot[c].map(clean)

model = None
device = ""

if not any(x["severity"] == "hard" for x in issues):
    try:
        import torch
        from mmdet.apis import init_detector

        try:
            from mmyolo.utils import register_all_modules
            register_all_modules(init_default_scope=True)
        except Exception:
            pass

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = init_detector(str(CONFIG), str(CKPT), device=device)

    except Exception:
        issues.append({
            "item":"detector_init",
            "issue_type":"hard_detector_init_failed",
            "severity":"hard",
            "detail":traceback.format_exc()[-1500:]
        })

det_rows = []
frame_rows = []

if model is not None:
    import cv2
    from mmdet.apis import inference_detector

    for _, r in pilot.iterrows():
        video_id = clean(r["video_id"])
        video_path = Path(clean(r["video_path"]))
        video_name = clean(r["video_filename"])

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            issues.append({
                "item":video_id,
                "issue_type":"hard_video_open_failed",
                "severity":"hard",
                "detail":str(video_path)
            })
            continue

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

        for sec in SAMPLE_SECONDS:
            frame_idx = int(round(sec * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()

            if not ok or frame is None:
                frame_rows.append({
                    "video_id":video_id,
                    "video_filename":video_name,
                    "sample_sec":sec,
                    "frame_index":frame_idx,
                    "frame_read_ok":False,
                    "detection_count":0,
                    "annotated_frame":""
                })
                continue

            result = inference_detector(model, frame)
            pred = result.pred_instances

            bboxes = pred.bboxes.detach().cpu().numpy()
            scores = pred.scores.detach().cpu().numpy()
            labels = pred.labels.detach().cpu().numpy()

            kept = 0
            for box, score, label in zip(bboxes, scores, labels):
                score = float(score)
                if score < SCORE_THR:
                    continue

                x1, y1, x2, y2 = [int(round(v)) for v in box]
                kept += 1

                det_rows.append({
                    "video_id":video_id,
                    "video_filename":video_name,
                    "sample_sec":sec,
                    "frame_index":frame_idx,
                    "label":int(label),
                    "score":round(score, 6),
                    "x1":x1,
                    "y1":y1,
                    "x2":x2,
                    "y2":y2
                })

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{score:.2f}", (x1, max(20, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            out_img = FRAMES / f"{video_id}_{frame_idx}.jpg"
            cv2.imwrite(str(out_img), frame)

            frame_rows.append({
                "video_id":video_id,
                "video_filename":video_name,
                "sample_sec":sec,
                "frame_index":frame_idx,
                "frame_read_ok":True,
                "detection_count":kept,
                "annotated_frame":str(out_img)
            })

        cap.release()

detections = pd.DataFrame(det_rows)
frames = pd.DataFrame(frame_rows)

write(detections, DETECTIONS_CSV)
write(frames, FRAME_CSV)

sampled_frames = len(frames)
frames_ok = int((frames["frame_read_ok"] == True).sum()) if len(frames) else 0
detections_count = len(detections)
zero_detection_frames = int((frames["detection_count"] == 0).sum()) if len(frames) else 0

if sampled_frames == 0:
    issues.append({"item":"sampling","issue_type":"hard_no_frames_sampled","severity":"hard","detail":"No frames sampled."})

if frames_ok == 0:
    issues.append({"item":"sampling","issue_type":"hard_no_frames_read","severity":"hard","detail":"No frames read."})

if detections_count == 0:
    issues.append({"item":"detector","issue_type":"hard_zero_detections","severity":"hard","detail":"Detector produced zero detections."})

if zero_detection_frames > 0:
    issues.append({
        "item":"detector",
        "issue_type":"warning_some_zero_detection_frames",
        "severity":"warning",
        "detail":f"{zero_detection_frames} sampled frames had zero detections."
    })

issues.append({
    "item":"scope",
    "issue_type":"info_detector_smoke_only",
    "severity":"info",
    "detail":"v79d compact tests detector on pilot frames only. It does not run full tracking."
})

issues_df = pd.DataFrame(issues)
write(issues_df, ISSUES_CSV)

hard_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0

decision = pd.DataFrame([{
    "v79d_compact_decision":"pilot_detector_smoke_test_passed" if hard_count == 0 else "pilot_detector_smoke_test_has_blocking_issues",
    "detector_config":str(CONFIG),
    "detector_checkpoint":str(CKPT),
    "device_used":device,
    "pilot_videos":len(pilot),
    "sampled_frames":sampled_frames,
    "frames_read_ok":frames_ok,
    "detections":detections_count,
    "zero_detection_frames":zero_detection_frames,
    "score_threshold":SCORE_THR,
    "hard_issue_count":hard_count,
    "warning_count":warning_count,
    "ready_for_pilot_tracking":bool(hard_count == 0 and detections_count > 0),
    "ready_for_full_tracking":False,
    "claim_scope":"compact_detector_smoke_test_only",
    "generated_at":datetime.now().isoformat(timespec="seconds")
}])

write(decision, DECISION_CSV)

NOTE.write_text(
    "# v79d Compact Detector Smoke Test\n\n"
    f"- Decision: {decision.iloc[0]['v79d_compact_decision']}\n"
    f"- Device used: {device}\n"
    f"- Pilot videos: {len(pilot)}\n"
    f"- Sampled frames: {sampled_frames}\n"
    f"- Frames read OK: {frames_ok}\n"
    f"- Detections: {detections_count}\n"
    f"- Zero-detection frames: {zero_detection_frames}\n"
    f"- Hard issues: {hard_count}\n"
    f"- Ready for pilot tracking: {bool(hard_count == 0 and detections_count > 0)}\n\n"
    "This stage only validates detector execution on pilot frames.\n",
    encoding="utf-8"
)

print("=== decision ===")
print(decision.to_string(index=False))

print("\n=== frame summary ===")
print(frames.to_string(index=False) if len(frames) else "empty")

print("\n=== detections sample ===")
print(detections.head(30).to_string(index=False) if len(detections) else "empty")

print("\n=== issues ===")
print(issues_df.to_string(index=False))
