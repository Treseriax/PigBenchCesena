from pathlib import Path
from datetime import datetime
import argparse
import os
import sys
import csv
import re
import traceback
import pandas as pd
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

RUN_PLAN = FULL / "outputs/v79b_tracking_manifest_audit_and_run_plan/v79b_tracking_run_plan_36_videos.csv"
CLIPS = FULL / "outputs/v79a_tracking_preparation_manifest/v79a_annotation_clip_manifest.csv"

CONFIG = ROOT / "detection/configs/yolov8/yolov8_s.py"
CKPT = ROOT / "detection/data/pretrained_weights/yolov8_pigs/yolov8_s.pth"

OUT = FULL / "outputs/v79e_full_36_video_sampled_tracking_run"
RUNS = OUT / "per_video_runs"
ANNOT = OUT / "annotated_sample_frames"
OUT.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)
ANNOT.mkdir(parents=True, exist_ok=True)

COMBINED_TRACKS = OUT / "v79e_all_sampled_tracks.csv"
COMBINED_DETS = OUT / "v79e_all_sampled_detections.csv"
WINDOWS_CSV = OUT / "v79e_tracking_windows.csv"
VIDEO_SUMMARY = OUT / "v79e_video_tracking_summary.csv"
DECISION = OUT / "v79e_decision_summary.csv"
ISSUES = OUT / "v79e_issues.csv"
NOTE = FULL / "notes/v79e_full_36_video_sampled_tracking_run_notes.md"
NOTE.parent.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "detection"))

def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s

def write(df, p):
    df.to_csv(p, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")

def safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", clean(s))

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
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ab = max(0, bx2 - bx1) * max(0, by2 - by1)
    den = aa + ab - inter
    return inter / den if den > 0 else 0.0

class IoUTracker:
    def __init__(self, thr=0.30, max_missing=2):
        self.thr = thr
        self.max_missing = max_missing
        self.next_id = 1
        self.active = {}

    def update(self, dets, frame_step):
        out = []
        assigned = set()

        dets = sorted(dets, key=lambda d: d["score"], reverse=True)

        for d in dets:
            best_tid = None
            best_iou = 0.0

            for tid, tr in self.active.items():
                if tid in assigned:
                    continue
                val = iou(d["bbox"], tr["bbox"])
                if val > best_iou:
                    best_iou = val
                    best_tid = tid

            if best_tid is not None and best_iou >= self.thr:
                tid = best_tid
            else:
                tid = self.next_id
                self.next_id += 1

            self.active[tid] = {"bbox": d["bbox"], "last_step": frame_step}
            assigned.add(tid)

            dd = dict(d)
            dd["track_local_id"] = tid
            dd["matched_iou"] = best_iou
            out.append(dd)

        self.active = {
            tid: tr for tid, tr in self.active.items()
            if frame_step - tr["last_step"] <= self.max_missing
        }

        return out

def init_detector_cpu():
    from mmdet.apis import init_detector

    try:
        from mmyolo.utils import register_all_modules
        register_all_modules(init_default_scope=True)
    except Exception:
        pass

    return init_detector(str(CONFIG), str(CKPT), device="cpu")

def infer(model, frame, score_thr):
    from mmdet.apis import inference_detector

    result = inference_detector(model, frame)
    pred = result.pred_instances

    bboxes = pred.bboxes.detach().cpu().numpy()
    scores = pred.scores.detach().cpu().numpy()
    labels = pred.labels.detach().cpu().numpy()

    rows = []
    for box, score, label in zip(bboxes, scores, labels):
        score = float(score)
        if score < score_thr:
            continue
        rows.append({
            "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            "score": score,
            "label": int(label),
        })
    return rows

def build_windows(clips):
    c = clips.copy()
    c["clip_start_sec_in_video"] = pd.to_numeric(c["clip_start_sec_in_video"], errors="coerce")
    c["clip_end_sec_in_video"] = pd.to_numeric(c["clip_end_sec_in_video"], errors="coerce")
    c = c.dropna(subset=["clip_start_sec_in_video", "clip_end_sec_in_video"])

    rows = []
    for (video_id, start, end), g in c.groupby(["video_id", "clip_start_sec_in_video", "clip_end_sec_in_video"]):
        rows.append({
            "window_id": f"{clean(video_id)}_w{len(rows):05d}",
            "video_id": clean(video_id),
            "window_start_sec": float(start),
            "window_end_sec": float(end),
            "linked_clip_count": int(len(g)),
            "clip_ids": ";".join(sorted(set(g["clip_id"].map(clean).tolist()))),
            "behaviour_labels": ";".join(sorted(set([clean(x) for x in g["behaviour_label"].tolist() if clean(x)]))),
            "identity_colours": ";".join(sorted(set([clean(x) for x in g["identity_colour"].tolist() if clean(x)]))),
        })
    return pd.DataFrame(rows)

def process_video(model, video_row, windows, args):
    import cv2

    video_id = clean(video_row["video_id"])
    video_path = Path(clean(video_row["video_path"]))
    video_name = clean(video_row["video_filename"])

    vout = RUNS / video_id
    vout.mkdir(parents=True, exist_ok=True)

    det_csv = vout / "detections.csv"
    track_csv = vout / "tracks.csv"
    summary_csv = vout / "summary.csv"

    if summary_csv.exists() and not args.overwrite:
        try:
            summ = pd.read_csv(summary_csv).fillna("")
            if len(summ) and clean(summ.iloc[0].get("run_status")) == "DONE":
                return "SKIPPED_EXISTING"
        except Exception:
            pass

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        summary = pd.DataFrame([{
            "video_id": video_id,
            "video_filename": video_name,
            "run_status": "FAILED_OPEN",
            "sampled_frames": 0,
            "detections": 0,
            "track_rows": 0,
            "unique_tracklets": 0,
            "error": str(video_path),
        }])
        write(summary, summary_csv)
        return "FAILED_OPEN"

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    det_rows = []
    track_rows = []
    sampled = 0
    zero_det = 0
    annotated_count = 0
    error = ""

    vwins = windows[windows["video_id"] == video_id].sort_values("window_start_sec").copy()

    try:
        for _, w in vwins.iterrows():
            window_id = clean(w["window_id"])
            start = float(w["window_start_sec"])
            end = float(w["window_end_sec"])

            tracker = IoUTracker(thr=args.iou_thr, max_missing=2)
            step = 0

            sample_times = np.arange(start, end, 1.0 / args.sample_fps)

            for t in sample_times:
                frame_idx = int(round(t * fps))
                if frame_total > 0 and frame_idx >= frame_total:
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                sampled += 1
                dets = infer(model, frame, args.score_thr)
                if len(dets) == 0:
                    zero_det += 1

                tracks = tracker.update(dets, step)

                for d in dets:
                    x1, y1, x2, y2 = [int(round(v)) for v in d["bbox"]]
                    det_rows.append({
                        "video_id": video_id,
                        "video_filename": video_name,
                        "window_id": window_id,
                        "time_sec": round(float(t), 3),
                        "frame_index": int(frame_idx),
                        "label": int(d["label"]),
                        "score": round(float(d["score"]), 6),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    })

                for tr in tracks:
                    x1, y1, x2, y2 = [int(round(v)) for v in tr["bbox"]]
                    local_id = int(tr["track_local_id"])
                    tracklet_id = f"{video_id}_{window_id}_trk{local_id:03d}"

                    track_rows.append({
                        "tracklet_id": tracklet_id,
                        "video_id": video_id,
                        "video_filename": video_name,
                        "window_id": window_id,
                        "time_sec": round(float(t), 3),
                        "frame_index": int(frame_idx),
                        "track_local_id": local_id,
                        "label": int(tr["label"]),
                        "score": round(float(tr["score"]), 6),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "matched_iou": round(float(tr["matched_iou"]), 6),
                    })

                    if annotated_count < args.max_annotated_per_video:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{tracklet_id.split('_')[-1]} {float(tr['score']):.2f}",
                                    (x1, max(20, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                if annotated_count < args.max_annotated_per_video:
                    img = ANNOT / f"{video_id}_{safe(window_id)}_{frame_idx}.jpg"
                    cv2.imwrite(str(img), frame)
                    annotated_count += 1

                step += 1

    except Exception:
        error = traceback.format_exc()[-2000:]

    cap.release()

    det_df = pd.DataFrame(det_rows)
    tr_df = pd.DataFrame(track_rows)
    write(det_df, det_csv)
    write(tr_df, track_csv)

    unique_tracklets = tr_df["tracklet_id"].nunique() if len(tr_df) and "tracklet_id" in tr_df.columns else 0

    status = "DONE" if not error else "FAILED_RUNTIME"
    summary = pd.DataFrame([{
        "video_id": video_id,
        "video_filename": video_name,
        "video_path": str(video_path),
        "run_status": status,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_total": frame_total,
        "windows": len(vwins),
        "sample_fps": args.sample_fps,
        "sampled_frames": sampled,
        "zero_detection_frames": zero_det,
        "detections": len(det_df),
        "track_rows": len(tr_df),
        "unique_tracklets": int(unique_tracklets),
        "annotated_frames_saved": annotated_count,
        "error": error,
    }])
    write(summary, summary_csv)

    return status

def combine_outputs(run_plan):
    dets = []
    tracks = []
    summaries = []

    for _, r in run_plan.iterrows():
        video_id = clean(r["video_id"])
        vout = RUNS / video_id

        d = vout / "detections.csv"
        t = vout / "tracks.csv"
        s = vout / "summary.csv"

        if d.exists():
            dets.append(pd.read_csv(d).fillna(""))
        if t.exists():
            tracks.append(pd.read_csv(t).fillna(""))
        if s.exists():
            summaries.append(pd.read_csv(s).fillna(""))

    det_df = pd.concat(dets, ignore_index=True) if dets else pd.DataFrame()
    tr_df = pd.concat(tracks, ignore_index=True) if tracks else pd.DataFrame()
    su_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()

    write(det_df, COMBINED_DETS)
    write(tr_df, COMBINED_TRACKS)
    write(su_df, VIDEO_SUMMARY)

    return det_df, tr_df, su_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-fps", type=float, default=1.0)
    ap.add_argument("--score-thr", type=float, default=0.25)
    ap.add_argument("--iou-thr", type=float, default=0.30)
    ap.add_argument("--max-videos", type=int, default=36)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-annotated-per-video", type=int, default=5)
    args = ap.parse_args()

    issues = []

    if not CONFIG.exists():
        issues.append({"item":"config","issue_type":"hard_missing_config","severity":"hard","detail":str(CONFIG)})
    if not CKPT.exists():
        issues.append({"item":"checkpoint","issue_type":"hard_missing_checkpoint","severity":"hard","detail":str(CKPT)})

    run_plan = pd.read_csv(RUN_PLAN).fillna("")
    clips = pd.read_csv(CLIPS).fillna("")
    for df in [run_plan, clips]:
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].map(clean)

    windows = build_windows(clips)
    write(windows, WINDOWS_CSV)

    if len(windows) == 0:
        issues.append({"item":"windows","issue_type":"hard_no_tracking_windows","severity":"hard","detail":"No windows created."})

    selected = run_plan.head(args.max_videos).copy()

    model = None
    if not any(i["severity"] == "hard" for i in issues):
        try:
            model = init_detector_cpu()
        except Exception:
            issues.append({
                "item":"detector_init",
                "issue_type":"hard_detector_cpu_init_failed",
                "severity":"hard",
                "detail":traceback.format_exc()[-2000:]
            })

    statuses = []
    if model is not None:
        for i, (_, vr) in enumerate(selected.iterrows(), start=1):
            print(f"[v79e] {i}/{len(selected)} {clean(vr['video_id'])} {clean(vr['video_filename'])}", flush=True)
            st = process_video(model, vr, windows, args)
            statuses.append(st)
            print(f"[v79e] status={st}", flush=True)

    det_df, tr_df, summary_df = combine_outputs(selected)

    if len(summary_df):
        failed = int((summary_df["run_status"] != "DONE").sum())
        done = int((summary_df["run_status"] == "DONE").sum())
    else:
        failed = 0
        done = 0

    if failed:
        issues.append({"item":"video_runs","issue_type":"hard_some_video_runs_failed","severity":"hard","detail":f"{failed} video runs failed."})

    if len(tr_df) == 0:
        issues.append({"item":"tracks","issue_type":"hard_no_tracks_created","severity":"hard","detail":"Combined track table is empty."})

    if len(det_df) == 0:
        issues.append({"item":"detections","issue_type":"hard_no_detections_created","severity":"hard","detail":"Combined detection table is empty."})

    issues.append({
        "item":"scope",
        "issue_type":"info_sampled_annotation_window_tracking",
        "severity":"info",
        "detail":"v79e runs CPU detector + simple IoU tracker on sampled frames inside annotation windows for selected videos."
    })

    issues_df = pd.DataFrame(issues)
    write(issues_df, ISSUES)

    hard_count = int((issues_df["severity"] == "hard").sum()) if "severity" in issues_df.columns else 0
    warning_count = int((issues_df["severity"] == "warning").sum()) if "severity" in issues_df.columns else 0

    unique_tracklets = tr_df["tracklet_id"].nunique() if len(tr_df) and "tracklet_id" in tr_df.columns else 0
    sampled_frames = int(summary_df["sampled_frames"].astype(float).sum()) if len(summary_df) and "sampled_frames" in summary_df.columns else 0
    detections = len(det_df)
    track_rows = len(tr_df)

    decision = pd.DataFrame([{
        "v79e_decision":"full_36_video_sampled_tracking_created" if hard_count == 0 else "full_36_video_sampled_tracking_has_blocking_issues",
        "selected_videos":len(selected),
        "video_runs_done":done,
        "video_runs_failed":failed,
        "tracking_windows":len(windows),
        "sample_fps":args.sample_fps,
        "score_threshold":args.score_thr,
        "iou_threshold":args.iou_thr,
        "sampled_frames":sampled_frames,
        "detection_rows":detections,
        "track_rows":track_rows,
        "unique_tracklets":int(unique_tracklets),
        "hard_issue_count":hard_count,
        "warning_count":warning_count,
        "ready_for_track_clip_fusion":bool(hard_count == 0 and track_rows > 0),
        "ready_for_final_tracking_claim":False,
        "claim_scope":"sampled_annotation_window_tracking_only",
        "generated_at":datetime.now().isoformat(timespec="seconds")
    }])
    write(decision, DECISION)

    NOTE.write_text(
        "# v79e Full 36-Video Sampled Tracking Run\n\n"
        f"- Decision: {decision.iloc[0]['v79e_decision']}\n"
        f"- Selected videos: {len(selected)}\n"
        f"- Video runs done: {done}\n"
        f"- Video runs failed: {failed}\n"
        f"- Tracking windows: {len(windows)}\n"
        f"- Sample FPS: {args.sample_fps}\n"
        f"- Sampled frames: {sampled_frames}\n"
        f"- Detection rows: {detections}\n"
        f"- Track rows: {track_rows}\n"
        f"- Unique tracklets: {int(unique_tracklets)}\n"
        f"- Hard issues: {hard_count}\n"
        f"- Ready for track-clip fusion: {bool(hard_count == 0 and track_rows > 0)}\n\n"
        "This is sampled annotation-window tracking, not full 25 FPS whole-video tracking.\n",
        encoding="utf-8"
    )

    print("=== v79e decision ===")
    print(decision.to_string(index=False))
    print("\n=== video summary ===")
    print(summary_df.to_string(index=False) if len(summary_df) else "empty")
    print("\n=== issues ===")
    print(issues_df.to_string(index=False))

if __name__ == "__main__":
    main()
