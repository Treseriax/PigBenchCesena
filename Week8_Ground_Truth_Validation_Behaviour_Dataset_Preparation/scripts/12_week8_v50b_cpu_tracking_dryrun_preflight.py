from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
import os
import sys
import traceback
import pandas as pd
import numpy as np


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"
REPO = ROOT

V45_CLIP_JSON = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_level_ground_truth.json"
V50A_DECISION = W8 / "outputs" / "v50a_tracking_source_coverage_audit" / "week8_v50a_decision_summary.csv"

OUT = W8 / "outputs" / "v50b_cpu_tracking_dryrun_preflight"
REPORTS = W8 / "reports"
NOTES = W8 / "notes"
PROGRESS = W8 / "progress"

for p in [OUT, REPORTS, NOTES, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_MODEL_CANDIDATES = OUT / "week8_v50b_model_candidates.csv"
OUT_CLIP_SUMMARY = OUT / "week8_v50b_dryrun_clip_summary.csv"
OUT_TRACKS = OUT / "week8_v50b_dryrun_tracking_rows.csv"
OUT_DECISION = OUT / "week8_v50b_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v50b_issues.csv"
OUT_REPORT = REPORTS / "week8_v50b_cpu_tracking_dryrun_preflight_report.md"
OUT_NOTE = NOTES / "week8_v50b_cpu_tracking_dryrun_preflight_notes.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def clean_str(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom


class SimpleIoUTracker:
    def __init__(self, iou_thr=0.30, max_missed=15):
        self.iou_thr = iou_thr
        self.max_missed = max_missed
        self.next_track_id = 1
        self.tracks = {}

    def update(self, detections, frame_index):
        assigned_det = set()
        assigned_track = set()
        matches = []

        track_ids = list(self.tracks.keys())

        candidates = []
        for tid in track_ids:
            tb = self.tracks[tid]["bbox"]
            for di, det in enumerate(detections):
                score = iou_xyxy(tb, det["bbox"])
                if score >= self.iou_thr:
                    candidates.append((score, tid, di))

        candidates.sort(reverse=True, key=lambda x: x[0])

        for score, tid, di in candidates:
            if tid in assigned_track or di in assigned_det:
                continue
            assigned_track.add(tid)
            assigned_det.add(di)
            matches.append((tid, di, score))

        output = []

        for tid, di, score in matches:
            det = detections[di]
            self.tracks[tid]["bbox"] = det["bbox"]
            self.tracks[tid]["last_frame"] = frame_index
            self.tracks[tid]["missed"] = 0
            output.append({
                **det,
                "track_id": tid,
                "match_iou": score,
                "track_status": "matched_existing",
            })

        for di, det in enumerate(detections):
            if di in assigned_det:
                continue
            tid = self.next_track_id
            self.next_track_id += 1
            self.tracks[tid] = {
                "bbox": det["bbox"],
                "last_frame": frame_index,
                "missed": 0,
            }
            output.append({
                **det,
                "track_id": tid,
                "match_iou": None,
                "track_status": "new_track",
            })

        for tid in list(self.tracks.keys()):
            if tid in assigned_track:
                continue
            self.tracks[tid]["missed"] = frame_index - self.tracks[tid]["last_frame"]
            if self.tracks[tid]["missed"] > self.max_missed:
                del self.tracks[tid]

        return output


def find_model_candidates():
    rows = []

    checkpoint_patterns = [
        REPO / "detection" / "data" / "pretrained_weights" / "yolov8_pigs" / "yolov8_s.pth",
        REPO / "detection" / "data" / "pretrained_weights" / "yolov8_pigs" / "yolov8_m.pth",
        REPO / "detection" / "data" / "pretrained_weights" / "yolov8_pigs" / "yolov8_l.pth",
        REPO / "detection" / "data" / "pretrained_weights" / "yolov8_pigs" / "yolov8_x.pth",
    ]

    checkpoints = []
    for p in checkpoint_patterns:
        if p.exists():
            checkpoints.append(p)

    for p in (REPO / "detection").glob("**/*.pth"):
        if p not in checkpoints and any(tok in p.name.lower() for tok in ["yolo", "pig"]):
            checkpoints.append(p)

    configs = []
    for p in (REPO / "detection").glob("**/*.py"):
        name = p.name.lower()
        if "yolov8" in name or "pig" in name or "coco" in name:
            try:
                txt = p.read_text(errors="ignore").lower()
            except Exception:
                txt = ""
            score = 0
            if "pigdetect" in txt:
                score += 5
            if "num_classes" in txt and "1" in txt:
                score += 3
            if "yolov8" in name:
                score += 3
            if "_s" in name or "s_" in name:
                score += 1
            configs.append((score, p))

    configs = [p for score, p in sorted(configs, key=lambda x: (-x[0], str(x[1])))]

    for ckpt in checkpoints:
        for cfg in configs[:20]:
            try:
                cfg_text = cfg.read_text(errors="ignore").lower()
            except Exception:
                cfg_text = ""
            cfg_score = 0
            if "pigdetect" in cfg_text:
                cfg_score += 5
            if "num_classes" in cfg_text:
                cfg_score += 2
            if "yolov8" in cfg.name.lower():
                cfg_score += 2
            if ckpt.stem.lower() in cfg.name.lower():
                cfg_score += 2
            if ckpt.name.lower().replace(".pth", "") in cfg.name.lower():
                cfg_score += 2

            rows.append({
                "config_path": str(cfg),
                "checkpoint_path": str(ckpt),
                "config_exists": cfg.exists(),
                "checkpoint_exists": ckpt.exists(),
                "priority_score": cfg_score,
                "config_name": cfg.name,
                "checkpoint_name": ckpt.name,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "config_path", "checkpoint_path", "config_exists", "checkpoint_exists",
            "priority_score", "config_name", "checkpoint_name"
        ])

    df = pd.DataFrame(rows).sort_values(
        ["priority_score", "checkpoint_name", "config_name"],
        ascending=[False, True, True],
    )
    return df


def init_mmdet_model(config_path, checkpoint_path, device):
    try:
        from mmyolo.utils import register_all_modules
        register_all_modules()
    except Exception:
        pass

    from mmdet.apis import init_detector
    model = init_detector(str(config_path), str(checkpoint_path), device=device)
    return model


def parse_mmdet_result(result, score_thr):
    pred = result.pred_instances
    bboxes = pred.bboxes.detach().cpu().numpy()
    scores = pred.scores.detach().cpu().numpy()

    if hasattr(pred, "labels"):
        labels = pred.labels.detach().cpu().numpy()
    else:
        labels = np.zeros(len(scores), dtype=int)

    detections = []
    for bbox, score, label in zip(bboxes, scores, labels):
        if float(score) < score_thr:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox.tolist()]
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "score": float(score),
            "label": int(label),
        })

    return detections


def run_clip_tracking(model, clip, max_frames, frame_stride, score_thr, iou_thr, max_missed):
    import cv2
    from mmdet.apis import inference_detector

    scan = clean_str(clip.get("scan_frame_id"))
    video_id = clean_str(clip.get("video_id"))
    clip_path = Path(clean_str(clip.get("clip_path")))

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return [], {
            "scan_frame_id": scan,
            "video_id": video_id,
            "clip_path": str(clip_path),
            "video_open_ok": False,
            "frame_count": 0,
            "processed_frames": 0,
            "tracking_rows": 0,
            "unique_tracks": 0,
            "status": "video_open_failed",
        }

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = SimpleIoUTracker(iou_thr=iou_thr, max_missed=max_missed)

    rows = []
    processed = 0
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if max_frames is not None and processed >= max_frames:
            break

        if frame_index % frame_stride != 0:
            frame_index += 1
            continue

        result = inference_detector(model, frame)
        detections = parse_mmdet_result(result, score_thr=score_thr)
        tracked = tracker.update(detections, frame_index)

        for t in tracked:
            x1, y1, x2, y2 = t["bbox"]
            rows.append({
                "dataset_version": "week8_v50b_dryrun",
                "scan_frame_id": scan,
                "video_id": video_id,
                "clip_path": str(clip_path),
                "frame_index_in_clip": int(frame_index),
                "timestamp_sec_in_clip": float(frame_index / fps) if fps and fps > 0 else None,
                "track_id": int(t["track_id"]),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "score": float(t["score"]),
                "label": int(t["label"]),
                "match_iou": t["match_iou"],
                "track_status": t["track_status"],
                "video_width": width,
                "video_height": height,
                "video_fps": fps,
                "video_frame_count": frame_count,
                "frame_stride": frame_stride,
                "score_thr": score_thr,
                "iou_thr": iou_thr,
            })

        processed += 1
        frame_index += 1

    cap.release()

    unique_tracks = len(set([r["track_id"] for r in rows])) if rows else 0

    summary = {
        "scan_frame_id": scan,
        "video_id": video_id,
        "clip_path": str(clip_path),
        "video_open_ok": True,
        "frame_count": int(frame_count),
        "video_width": int(width),
        "video_height": int(height),
        "video_fps": float(fps),
        "processed_frames": int(processed),
        "tracking_rows": int(len(rows)),
        "unique_tracks": int(unique_tracks),
        "status": "processed",
    }

    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-clips", type=int, default=3)
    parser.add_argument("--max-frames-per-clip", type=int, default=60)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--score-thr", type=float, default=0.30)
    parser.add_argument("--iou-thr", type=float, default=0.30)
    parser.add_argument("--max-missed", type=int, default=15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", default="")
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    issues = []

    for p in [V45_CLIP_JSON, V50A_DECISION]:
        if not p.exists():
            issues.append({
                "item": str(p),
                "issue_type": "hard_missing_required_input",
                "issue_detail": "Required previous-stage artifact is missing.",
                "severity": "hard",
            })

    if issues:
        issues_df = pd.DataFrame(issues)
        safe_to_csv(issues_df, OUT_ISSUES)
        decision = pd.DataFrame([{
            "v50b_decision": "cpu_tracking_dryrun_blocked_missing_inputs",
            "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
            "ready_for_v50c_full_72_tracking": False,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }])
        safe_to_csv(decision, OUT_DECISION)
        print(decision.to_string(index=False))
        return

    model_candidates = find_model_candidates()
    safe_to_csv(model_candidates, OUT_MODEL_CANDIDATES)

    if args.config and args.checkpoint:
        config_path = Path(args.config)
        checkpoint_path = Path(args.checkpoint)
    elif len(model_candidates):
        chosen = model_candidates.iloc[0]
        config_path = Path(chosen["config_path"])
        checkpoint_path = Path(chosen["checkpoint_path"])
    else:
        config_path = None
        checkpoint_path = None

    if config_path is None or checkpoint_path is None or not config_path.exists() or not checkpoint_path.exists():
        issues.append({
            "item": "model_config_checkpoint",
            "issue_type": "hard_model_config_or_checkpoint_not_found",
            "issue_detail": "Could not find usable detector config/checkpoint automatically.",
            "severity": "hard",
        })
        issues_df = pd.DataFrame(issues)
        safe_to_csv(issues_df, OUT_ISSUES)
        decision = pd.DataFrame([{
            "v50b_decision": "cpu_tracking_dryrun_blocked_model_not_found",
            "selected_config": str(config_path) if config_path else "",
            "selected_checkpoint": str(checkpoint_path) if checkpoint_path else "",
            "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
            "ready_for_v50c_full_72_tracking": False,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }])
        safe_to_csv(decision, OUT_DECISION)
        print(decision.to_string(index=False))
        return

    clip_data = json.loads(V45_CLIP_JSON.read_text())
    clips = clip_data.get("clips", [])

    selected_clips = clips[:args.max_clips]

    try:
        model = init_mmdet_model(config_path, checkpoint_path, args.device)
        model_init_ok = True
        model_error = ""
    except Exception as e:
        model_init_ok = False
        model_error = traceback.format_exc()
        issues.append({
            "item": "model_init",
            "issue_type": "hard_model_init_failed",
            "issue_detail": str(e),
            "severity": "hard",
        })

    all_rows = []
    clip_summaries = []

    if model_init_ok:
        for clip in selected_clips:
            try:
                rows, summary = run_clip_tracking(
                    model=model,
                    clip=clip,
                    max_frames=args.max_frames_per_clip,
                    frame_stride=max(1, args.frame_stride),
                    score_thr=args.score_thr,
                    iou_thr=args.iou_thr,
                    max_missed=args.max_missed,
                )
                all_rows.extend(rows)
                clip_summaries.append(summary)
            except Exception as e:
                clip_summaries.append({
                    "scan_frame_id": clean_str(clip.get("scan_frame_id")),
                    "video_id": clean_str(clip.get("video_id")),
                    "clip_path": clean_str(clip.get("clip_path")),
                    "video_open_ok": False,
                    "frame_count": 0,
                    "processed_frames": 0,
                    "tracking_rows": 0,
                    "unique_tracks": 0,
                    "status": "clip_processing_failed",
                    "error": str(e),
                })
                issues.append({
                    "item": clean_str(clip.get("scan_frame_id")),
                    "issue_type": "warning_clip_processing_failed",
                    "issue_detail": str(e),
                    "severity": "warning",
                })

    tracks_df = pd.DataFrame(all_rows)
    clip_summary_df = pd.DataFrame(clip_summaries)

    safe_to_csv(tracks_df, OUT_TRACKS)
    safe_to_csv(clip_summary_df, OUT_CLIP_SUMMARY)

    if model_init_ok and len(tracks_df) == 0:
        issues.append({
            "item": "dryrun_tracking_rows",
            "issue_type": "hard_no_tracking_rows_generated",
            "issue_detail": "Detector/tracker ran but generated zero tracking rows.",
            "severity": "hard",
        })

    if model_init_ok and len(clip_summary_df) and (clip_summary_df["tracking_rows"] == 0).any():
        zero_clips = clip_summary_df[clip_summary_df["tracking_rows"] == 0]["scan_frame_id"].tolist()
        issues.append({
            "item": "zero_tracking_clips",
            "issue_type": "warning_some_clips_zero_tracking_rows",
            "issue_detail": "|".join(zero_clips),
            "severity": "warning",
        })

    issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
    hard_issues = issues_df[issues_df["severity"] == "hard"] if len(issues_df) else pd.DataFrame()
    warnings = issues_df[issues_df["severity"] == "warning"] if len(issues_df) else pd.DataFrame()

    dryrun_processed_clips = int(len(clip_summary_df))
    dryrun_tracking_rows = int(len(tracks_df))
    dryrun_unique_tracks = int(tracks_df["track_id"].nunique()) if len(tracks_df) else 0

    ready_for_full = len(hard_issues) == 0 and model_init_ok and dryrun_tracking_rows > 0

    decision = pd.DataFrame([{
        "v50b_decision": "cpu_tracking_dryrun_passed" if ready_for_full else "cpu_tracking_dryrun_blocked",
        "selected_config": str(config_path),
        "selected_checkpoint": str(checkpoint_path),
        "device": args.device,
        "max_clips": int(args.max_clips),
        "max_frames_per_clip": int(args.max_frames_per_clip),
        "frame_stride": int(args.frame_stride),
        "score_thr": float(args.score_thr),
        "iou_thr": float(args.iou_thr),
        "model_init_ok": bool(model_init_ok),
        "dryrun_processed_clips": int(dryrun_processed_clips),
        "dryrun_tracking_rows": int(dryrun_tracking_rows),
        "dryrun_unique_tracks": int(dryrun_unique_tracks),
        "hard_issue_count": int(len(hard_issues)),
        "warning_count": int(len(warnings)),
        "issue_count": int(len(issues_df)),
        "ready_for_v50c_full_72_tracking": bool(ready_for_full),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])

    safe_to_csv(issues_df, OUT_ISSUES)
    safe_to_csv(decision, OUT_DECISION)

    report = (
        "Week 8 v50b CPU Tracking Dry-run Preflight Report\n\n"
        f"Decision: {decision.iloc[0]['v50b_decision']}\n"
        f"Selected config: {config_path}\n"
        f"Selected checkpoint: {checkpoint_path}\n"
        f"Device: {args.device}\n"
        f"Processed clips: {dryrun_processed_clips}\n"
        f"Tracking rows: {dryrun_tracking_rows}\n"
        f"Unique tracks: {dryrun_unique_tracks}\n"
        f"Hard issues: {len(hard_issues)}\n"
        f"Warnings: {len(warnings)}\n\n"
        "Purpose:\n"
        "This dry-run confirms whether the detector and simple IoU tracker can run safely before launching a full 72-clip tracking pass.\n\n"
        "Next:\n"
        "If ready_for_v50c_full_72_tracking is True, run the full 72-clip tracking stage with resume support.\n"
    )
    OUT_REPORT.write_text(report)

    OUT_NOTE.write_text(
        "# Week 8 v50b CPU Tracking Dry-run Preflight\n\n"
        "## Summary\n\n"
        f"- v50b decision: {decision.iloc[0]['v50b_decision']}\n"
        f"- Selected config: {config_path}\n"
        f"- Selected checkpoint: {checkpoint_path}\n"
        f"- Device: {args.device}\n"
        f"- Processed clips: {dryrun_processed_clips}\n"
        f"- Tracking rows: {dryrun_tracking_rows}\n"
        f"- Unique tracks: {dryrun_unique_tracks}\n"
        f"- Hard issue count: {len(hard_issues)}\n"
        f"- Warning count: {len(warnings)}\n"
        f"- Ready for v50c full 72 tracking: {ready_for_full}\n"
    )

    progress_row = pd.DataFrame([{
        "date": datetime.now().date().isoformat(),
        "stage": "v50b",
        "task_name": "CPU tracking dry-run preflight",
        "status": "PASS" if ready_for_full else "BLOCKED",
        "input_summary": str(V45_CLIP_JSON),
        "output_summary": str(OUT),
        "hard_issues": int(len(hard_issues)),
        "warnings": int(len(warnings)),
        "next_action": "v50c full 72-clip tracking run" if ready_for_full else "Fix detector/tracker preflight issues.",
    }])

    if OUT_PROGRESS.exists():
        old = pd.read_csv(OUT_PROGRESS)
        progress = pd.concat([old, progress_row], ignore_index=True)
    else:
        progress = progress_row

    safe_to_csv(progress, OUT_PROGRESS)

    print("Saved:")
    print(OUT_MODEL_CANDIDATES)
    print(OUT_CLIP_SUMMARY)
    print(OUT_TRACKS)
    print(OUT_DECISION)
    print(OUT_ISSUES)
    print(OUT_REPORT)
    print(OUT_NOTE)

    print()
    print("=== v50b decision ===")
    print(decision.to_string(index=False))

    print()
    print("=== v50b clip summary ===")
    if len(clip_summary_df):
        print(clip_summary_df.to_string(index=False))
    else:
        print("No clip summary rows.")

    print()
    print("=== v50b issues ===")
    if len(issues_df):
        print(issues_df.to_string(index=False))
    else:
        print("No issues found.")


if __name__ == "__main__":
    main()
