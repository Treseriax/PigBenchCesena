from pathlib import Path
from datetime import datetime
import csv
import pandas as pd

ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

DETS = FULL / "outputs/v79d_compact_detector_smoke_test/v79d_compact_detections.csv"

OUT = FULL / "outputs/v79d2_compact_pilot_iou_tracker"
OUT.mkdir(parents=True, exist_ok=True)

TRACKS = OUT / "v79d2_pilot_iou_tracks.csv"
FRAME_SUMMARY = OUT / "v79d2_frame_track_summary.csv"
VIDEO_SUMMARY = OUT / "v79d2_video_track_summary.csv"
DECISION = OUT / "v79d2_decision_summary.csv"
ISSUES = OUT / "v79d2_issues.csv"
NOTE = FULL / "notes/v79d2_compact_pilot_iou_tracker_notes.md"
NOTE.parent.mkdir(parents=True, exist_ok=True)

IOU_THR = 0.30
MAX_MISSING_FRAME_STEPS = 2

def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s

def write(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    den = area_a + area_b - inter
    return inter / den if den > 0 else 0.0

issues = []

if not DETS.exists():
    issues.append({
        "item": "detections",
        "issue_type": "hard_missing_detection_csv",
        "severity": "hard",
        "detail": str(DETS),
    })
    dets = pd.DataFrame()
else:
    dets = pd.read_csv(DETS).fillna("")
    for c in dets.columns:
        if dets[c].dtype == object:
            dets[c] = dets[c].map(clean)

required = ["video_id", "video_filename", "sample_sec", "frame_index", "score", "x1", "y1", "x2", "y2"]
for c in required:
    if len(dets) and c not in dets.columns:
        issues.append({
            "item": c,
            "issue_type": "hard_missing_detection_column",
            "severity": "hard",
            "detail": f"Missing column: {c}",
        })

if len(dets) == 0:
    issues.append({
        "item": "detections",
        "issue_type": "hard_empty_detections",
        "severity": "hard",
        "detail": "Detection CSV is empty.",
    })

track_rows = []
frame_rows = []
video_rows = []

if len(dets) and not any(i["severity"] == "hard" for i in issues):
    numeric_cols = ["sample_sec", "frame_index", "score", "x1", "y1", "x2", "y2"]
    for c in numeric_cols:
        dets[c] = pd.to_numeric(dets[c], errors="coerce")

    for video_id, vg in dets.groupby("video_id"):
        vg = vg.sort_values(["frame_index", "score"], ascending=[True, False]).copy()

        active = {}
        next_track_id = 1
        frame_step = 0
        video_track_ids = set()

        for frame_index, fg in vg.groupby("frame_index"):
            fg = fg.sort_values("score", ascending=False)
            assigned = set()
            current_tracks = []

            for _, d in fg.iterrows():
                box = [float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"])]

                best_tid = None
                best_iou = 0.0

                for tid, tr in active.items():
                    if tid in assigned:
                        continue
                    val = iou(box, tr["bbox"])
                    if val > best_iou:
                        best_iou = val
                        best_tid = tid

                if best_tid is not None and best_iou >= IOU_THR:
                    tid = best_tid
                else:
                    tid = next_track_id
                    next_track_id += 1

                active[tid] = {
                    "bbox": box,
                    "last_frame_step": frame_step,
                }
                assigned.add(tid)
                video_track_ids.add(tid)
                current_tracks.append(tid)

                track_rows.append({
                    "video_id": clean(d["video_id"]),
                    "video_filename": clean(d["video_filename"]),
                    "sample_sec": float(d["sample_sec"]),
                    "frame_index": int(d["frame_index"]),
                    "track_id": int(tid),
                    "score": float(d["score"]),
                    "x1": int(d["x1"]),
                    "y1": int(d["y1"]),
                    "x2": int(d["x2"]),
                    "y2": int(d["y2"]),
                    "matched_iou": round(float(best_iou), 6),
                    "tracker_type": "simple_iou_sampled_frame_tracker",
                })

            active = {
                tid: tr for tid, tr in active.items()
                if frame_step - tr["last_frame_step"] <= MAX_MISSING_FRAME_STEPS
            }

            frame_rows.append({
                "video_id": clean(video_id),
                "frame_index": int(frame_index),
                "sample_sec": float(fg["sample_sec"].iloc[0]),
                "detection_count": int(len(fg)),
                "track_count": int(len(current_tracks)),
                "track_ids": ";".join(str(x) for x in sorted(current_tracks)),
            })

            frame_step += 1

        video_rows.append({
            "video_id": clean(video_id),
            "video_filename": clean(vg["video_filename"].iloc[0]),
            "sampled_frames": int(vg["frame_index"].nunique()),
            "detections": int(len(vg)),
            "unique_track_ids": int(len(video_track_ids)),
            "avg_detections_per_frame": round(len(vg) / max(1, vg["frame_index"].nunique()), 4),
        })

tracks = pd.DataFrame(track_rows)
frames = pd.DataFrame(frame_rows)
videos = pd.DataFrame(video_rows)

write(tracks, TRACKS)
write(frames, FRAME_SUMMARY)
write(videos, VIDEO_SUMMARY)

track_count = tracks["track_id"].nunique() if len(tracks) and "track_id" in tracks.columns else 0
track_rows_count = len(tracks)

if track_rows_count == 0:
    issues.append({
        "item": "tracker",
        "issue_type": "hard_no_track_rows_created",
        "severity": "hard",
        "detail": "No track rows were created.",
    })

if track_count == 0:
    issues.append({
        "item": "tracker",
        "issue_type": "hard_no_track_ids_created",
        "severity": "hard",
        "detail": "No track IDs were created.",
    })

issues.append({
    "item": "scope",
    "issue_type": "info_pilot_tracker_only",
    "severity": "info",
    "detail": "v79d2 assigns simple IoU track IDs on v79d sampled pilot detections only. It is not full continuous tracking.",
})

issues_df = pd.DataFrame(issues)
write(issues_df, ISSUES)

hard_count = int((issues_df["severity"] == "hard").sum()) if "severity" in issues_df.columns else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if "severity" in issues_df.columns else 0

decision = pd.DataFrame([{
    "v79d2_decision": "pilot_iou_tracker_smoke_test_passed" if hard_count == 0 else "pilot_iou_tracker_smoke_test_has_blocking_issues",
    "input_detection_rows": len(dets),
    "track_rows": track_rows_count,
    "unique_video_rows": len(videos),
    "unique_track_ids_total": int(track_count),
    "iou_threshold": IOU_THR,
    "hard_issue_count": hard_count,
    "warning_count": warning_count,
    "ready_for_full_tracking_run_script": bool(hard_count == 0 and track_rows_count > 0),
    "ready_for_full_tracking_execution": False,
    "claim_scope": "pilot_iou_tracker_smoke_test_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

write(decision, DECISION)

NOTE.write_text(
    "# v79d2 Compact Pilot IoU Tracker\n\n"
    f"- Decision: {decision.iloc[0]['v79d2_decision']}\n"
    f"- Input detection rows: {len(dets)}\n"
    f"- Track rows: {track_rows_count}\n"
    f"- Pilot videos: {len(videos)}\n"
    f"- Unique track IDs total: {int(track_count)}\n"
    f"- IoU threshold: {IOU_THR}\n"
    f"- Hard issues: {hard_count}\n"
    f"- Ready for full tracking run script: {bool(hard_count == 0 and track_rows_count > 0)}\n\n"
    "This is a sampled-frame tracker smoke test only, not full continuous tracking.\n",
    encoding="utf-8"
)

print("=== v79d2 decision ===")
print(decision.to_string(index=False))

print("\n=== video summary ===")
print(videos.to_string(index=False) if len(videos) else "empty")

print("\n=== frame summary ===")
print(frames.to_string(index=False) if len(frames) else "empty")

print("\n=== track sample ===")
print(tracks.head(50).to_string(index=False) if len(tracks) else "empty")

print("\n=== issues ===")
print(issues_df.to_string(index=False))
