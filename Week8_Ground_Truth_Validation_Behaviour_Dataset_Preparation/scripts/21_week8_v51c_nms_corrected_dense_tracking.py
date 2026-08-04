from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
import os
import time
import importlib.util
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V45_CLIP_JSON = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_level_ground_truth.json"
V50B_DECISION = W8 / "outputs" / "v50b_cpu_tracking_dryrun_preflight" / "week8_v50b_decision_summary.csv"
V50B_SCRIPT = W8 / "scripts" / "12_week8_v50b_cpu_tracking_dryrun_preflight.py"
V51A_SCRIPT = W8 / "scripts" / "18_week8_v51a_tracking_maximization_sweep.py"

OUT = W8 / "outputs" / "v51c_nms_corrected_dense_tracking"
PER_CLIP = OUT / "per_clip_tracks"
PER_SUMMARY = OUT / "per_clip_summaries"
REPORTS = W8 / "reports"
NOTES = W8 / "notes"
PROGRESS = W8 / "progress"

for p in [OUT, PER_CLIP, PER_SUMMARY, REPORTS, NOTES, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_TRACKS = OUT / "week8_v51c_nms_dense_tracking_rows.csv"
OUT_SUMMARY = OUT / "week8_v51c_nms_dense_clip_summary.csv"
OUT_DECISION = OUT / "week8_v51c_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v51c_issues.csv"
OUT_REPORT = REPORTS / "week8_v51c_nms_corrected_dense_tracking_report.md"
OUT_NOTE = NOTES / "week8_v51c_nms_corrected_dense_tracking_notes.md"
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


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_detector_paths():
    d = pd.read_csv(V50B_DECISION)
    row = d.iloc[0]
    return Path(row["selected_config"]), Path(row["selected_checkpoint"])


def per_clip_paths(scan):
    return PER_CLIP / f"{scan}_tracks.csv", PER_SUMMARY / f"{scan}_summary.csv"


def run_clip_tracking_nms(v50b, v51a, model, clip, frame_stride, score_thr, tracker_iou_thr, nms_iou_thr, max_missed):
    import cv2
    from mmdet.apis import inference_detector

    scan = clean_str(clip.get("scan_frame_id"))
    video_id = clean_str(clip.get("video_id"))
    clip_path = Path(clean_str(clip.get("clip_path")))

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        cap.release()
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

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = v51a.SimpleIoUTrackerNMS(iou_thr=tracker_iou_thr, max_missed=max_missed)

    rows = []
    processed = 0
    frame_index = 0
    raw_detection_count = 0
    nms_kept_detection_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % frame_stride != 0:
            frame_index += 1
            continue

        result = inference_detector(model, frame)

        detections = v50b.parse_mmdet_result(result, score_thr=score_thr)
        raw_detection_count += len(detections)

        detections = v51a.nms_detections(detections, iou_thr=nms_iou_thr)
        nms_kept_detection_count += len(detections)

        tracked = tracker.update(detections, frame_index)

        for t in tracked:
            x1, y1, x2, y2 = t["bbox"]
            rows.append({
                "dataset_version": "week8_v51c_nms_corrected_dense_tracking",
                "scan_frame_id": scan,
                "video_id": video_id,
                "clip_path": str(clip_path),
                "frame_index_in_clip": int(frame_index),
                "timestamp_sec_in_clip": float(frame_index / fps) if fps > 0 else None,
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
                "tracker_iou_thr": tracker_iou_thr,
                "nms_iou_thr": nms_iou_thr,
                "max_missed": max_missed,
                "run_stage": "v51c",
                "parameter_source": "v51a_candidate_1_with_nms_correction",
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
        "frame_count": frame_count,
        "video_width": width,
        "video_height": height,
        "video_fps": fps,
        "processed_frames": processed,
        "tracking_rows": len(rows),
        "unique_tracks": unique_tracks,
        "raw_detection_count": raw_detection_count,
        "nms_kept_detection_count": nms_kept_detection_count,
        "nms_removed_detection_count": raw_detection_count - nms_kept_detection_count,
        "status": "processed",
        "run_status": "processed_v51c",
        "frame_stride": frame_stride,
        "score_thr": score_thr,
        "tracker_iou_thr": tracker_iou_thr,
        "nms_iou_thr": nms_iou_thr,
        "max_missed": max_missed,
        "parameter_source": "v51a_candidate_1_with_nms_correction",
    }

    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--score-thr", type=float, default=0.10)
    parser.add_argument("--tracker-iou-thr", type=float, default=0.15)
    parser.add_argument("--nms-iou-thr", type=float, default=0.55)
    parser.add_argument("--max-missed", type=int, default=35)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    issues = []

    for p in [V45_CLIP_JSON, V50B_DECISION, V50B_SCRIPT, V51A_SCRIPT]:
        if not p.exists():
            issues.append({
                "item": str(p),
                "issue_type": "hard_missing_required_input",
                "issue_detail": "Required input is missing.",
                "severity": "hard",
            })

    if issues:
        issues_df = pd.DataFrame(issues)
        safe_to_csv(issues_df, OUT_ISSUES)
        decision = pd.DataFrame([{
            "v51c_decision": "nms_dense_tracking_blocked_missing_inputs",
            "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
            "ready_for_v51d_nms_dense_tracking_qa": False,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }])
        safe_to_csv(decision, OUT_DECISION)
        print(decision.to_string(index=False))
        return

    clips = json.loads(V45_CLIP_JSON.read_text()).get("clips", [])
    config_path, checkpoint_path = read_detector_paths()

    v50b = load_module(V50B_SCRIPT, "week8_v50b_module")
    v51a = load_module(V51A_SCRIPT, "week8_v51a_module")

    print("Initializing detector for v51c NMS-corrected dense tracking...")
    print("Config:", config_path)
    print("Checkpoint:", checkpoint_path)
    print("Device:", args.device)
    print("frame_stride:", args.frame_stride)
    print("score_thr:", args.score_thr)
    print("tracker_iou_thr:", args.tracker_iou_thr)
    print("nms_iou_thr:", args.nms_iou_thr)
    print("max_missed:", args.max_missed)

    try:
        model = v50b.init_mmdet_model(config_path, checkpoint_path, args.device)
    except Exception as e:
        issues.append({
            "item": "model_init",
            "issue_type": "hard_model_init_failed",
            "issue_detail": str(e),
            "severity": "hard",
        })
        issues_df = pd.DataFrame(issues)
        safe_to_csv(issues_df, OUT_ISSUES)
        decision = pd.DataFrame([{
            "v51c_decision": "nms_dense_tracking_blocked_model_init_failed",
            "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
            "ready_for_v51d_nms_dense_tracking_qa": False,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }])
        safe_to_csv(decision, OUT_DECISION)
        print(decision.to_string(index=False))
        return

    start = time.time()
    summary_rows = []

    for idx, clip in enumerate(clips, start=1):
        scan = clean_str(clip.get("scan_frame_id"))
        track_path, summary_path = per_clip_paths(scan)

        print()
        print(f"[{idx}/{len(clips)}] {scan}")

        if args.resume and track_path.exists() and summary_path.exists():
            try:
                s = pd.read_csv(summary_path)
                if len(s):
                    row = s.iloc[0].to_dict()
                    row["run_status"] = "skipped_resume"
                    summary_rows.append(row)
                    print("  resume skip")
                    continue
            except Exception:
                pass

        try:
            rows, summary = run_clip_tracking_nms(
                v50b=v50b,
                v51a=v51a,
                model=model,
                clip=clip,
                frame_stride=args.frame_stride,
                score_thr=args.score_thr,
                tracker_iou_thr=args.tracker_iou_thr,
                nms_iou_thr=args.nms_iou_thr,
                max_missed=args.max_missed,
            )

            tracks_df = pd.DataFrame(rows)

            safe_to_csv(tracks_df, track_path)
            safe_to_csv(pd.DataFrame([summary]), summary_path)

            summary_rows.append(summary)

            print(
                f"  frames={summary.get('processed_frames')} "
                f"rows={summary.get('tracking_rows')} "
                f"tracks={summary.get('unique_tracks')} "
                f"nms_removed={summary.get('nms_removed_detection_count')}"
            )

        except Exception as e:
            issue = {
                "item": scan,
                "issue_type": "warning_clip_tracking_failed",
                "issue_detail": str(e),
                "severity": "warning",
            }
            issues.append(issue)

            summary = {
                "scan_frame_id": scan,
                "video_id": clean_str(clip.get("video_id")),
                "clip_path": clean_str(clip.get("clip_path")),
                "video_open_ok": False,
                "frame_count": 0,
                "processed_frames": 0,
                "tracking_rows": 0,
                "unique_tracks": 0,
                "raw_detection_count": 0,
                "nms_kept_detection_count": 0,
                "nms_removed_detection_count": 0,
                "status": "tracking_failed",
                "run_status": "failed_v51c",
                "error": str(e),
                "frame_stride": args.frame_stride,
                "score_thr": args.score_thr,
                "tracker_iou_thr": args.tracker_iou_thr,
                "nms_iou_thr": args.nms_iou_thr,
                "max_missed": args.max_missed,
                "parameter_source": "v51a_candidate_1_with_nms_correction",
            }

            safe_to_csv(pd.DataFrame([summary]), summary_path)
            summary_rows.append(summary)
            print("  FAILED:", e)

    all_tracks = []
    for clip in clips:
        scan = clean_str(clip.get("scan_frame_id"))
        track_path, _ = per_clip_paths(scan)
        if track_path.exists():
            try:
                df = pd.read_csv(track_path)
                if len(df):
                    all_tracks.append(df)
            except Exception as e:
                issues.append({
                    "item": scan,
                    "issue_type": "warning_failed_to_read_per_clip_tracks",
                    "issue_detail": str(e),
                    "severity": "warning",
                })

    full_tracks = pd.concat(all_tracks, ignore_index=True) if all_tracks else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)

    safe_to_csv(full_tracks, OUT_TRACKS)
    safe_to_csv(summary_df, OUT_SUMMARY)

    processed_clip_count = int(len(summary_df))
    successful_clip_count = int((summary_df["tracking_rows"].fillna(0).astype(float) > 0).sum()) if len(summary_df) else 0
    zero_tracking_clip_count = int((summary_df["tracking_rows"].fillna(0).astype(float) == 0).sum()) if len(summary_df) else 0
    tracking_rows_total = int(len(full_tracks))
    unique_scan_count = int(full_tracks["scan_frame_id"].nunique()) if len(full_tracks) and "scan_frame_id" in full_tracks.columns else 0
    unique_track_sum = int(full_tracks.groupby("scan_frame_id")["track_id"].nunique().sum()) if len(full_tracks) and "track_id" in full_tracks.columns else 0
    raw_detection_total = int(summary_df["raw_detection_count"].fillna(0).astype(float).sum()) if len(summary_df) else 0
    nms_kept_total = int(summary_df["nms_kept_detection_count"].fillna(0).astype(float).sum()) if len(summary_df) else 0
    nms_removed_total = int(summary_df["nms_removed_detection_count"].fillna(0).astype(float).sum()) if len(summary_df) else 0
    elapsed = time.time() - start

    if tracking_rows_total == 0:
        issues.append({
            "item": "nms_dense_tracking_rows",
            "issue_type": "hard_no_tracking_rows_generated",
            "issue_detail": "NMS dense tracking produced zero rows.",
            "severity": "hard",
        })

    if successful_clip_count < len(clips):
        issues.append({
            "item": "nms_dense_tracking_clip_coverage",
            "issue_type": "warning_some_clips_zero_tracking_rows",
            "issue_detail": f"{successful_clip_count}/{len(clips)} clips have tracking rows.",
            "severity": "warning",
        })

    issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
    hard_issues = issues_df[issues_df["severity"] == "hard"] if len(issues_df) else pd.DataFrame()
    warnings = issues_df[issues_df["severity"] == "warning"] if len(issues_df) else pd.DataFrame()

    ready = len(hard_issues) == 0 and successful_clip_count == len(clips)

    decision = pd.DataFrame([{
        "v51c_decision": "nms_corrected_dense_full_72_tracking_completed" if ready else "nms_corrected_dense_full_72_tracking_completed_with_warnings" if len(hard_issues) == 0 else "nms_corrected_dense_full_72_tracking_blocked",
        "parameter_source": "v51a_candidate_1_with_nms_correction",
        "device": args.device,
        "frame_stride": int(args.frame_stride),
        "score_thr": float(args.score_thr),
        "tracker_iou_thr": float(args.tracker_iou_thr),
        "nms_iou_thr": float(args.nms_iou_thr),
        "max_missed": int(args.max_missed),
        "clip_count_target": int(len(clips)),
        "processed_clip_count": processed_clip_count,
        "successful_clip_count": successful_clip_count,
        "zero_tracking_clip_count": zero_tracking_clip_count,
        "tracking_rows_total": tracking_rows_total,
        "unique_scan_count_in_tracks": unique_scan_count,
        "unique_track_total_by_clip_sum": unique_track_sum,
        "raw_detection_total": raw_detection_total,
        "nms_kept_detection_total": nms_kept_total,
        "nms_removed_detection_total": nms_removed_total,
        "elapsed_sec": float(elapsed),
        "hard_issue_count": int(len(hard_issues)),
        "warning_count": int(len(warnings)),
        "issue_count": int(len(issues_df)),
        "ready_for_v51d_nms_dense_tracking_qa": bool(ready),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])

    safe_to_csv(issues_df, OUT_ISSUES)
    safe_to_csv(decision, OUT_DECISION)

    OUT_REPORT.write_text(
        "Week 8 v51c NMS-corrected Dense Tracking Report\n\n"
        f"Decision: {decision.iloc[0]['v51c_decision']}\n"
        f"frame_stride: {args.frame_stride}\n"
        f"score_thr: {args.score_thr}\n"
        f"tracker_iou_thr: {args.tracker_iou_thr}\n"
        f"nms_iou_thr: {args.nms_iou_thr}\n"
        f"max_missed: {args.max_missed}\n"
        f"Processed clips: {processed_clip_count}\n"
        f"Successful clips: {successful_clip_count}\n"
        f"Zero tracking clips: {zero_tracking_clip_count}\n"
        f"Tracking rows total: {tracking_rows_total}\n"
        f"Unique track sum: {unique_track_sum}\n"
        f"Raw detection total: {raw_detection_total}\n"
        f"NMS kept detection total: {nms_kept_total}\n"
        f"NMS removed detection total: {nms_removed_total}\n"
        f"Elapsed seconds: {elapsed:.2f}\n"
        f"Hard issues: {len(hard_issues)}\n"
        f"Warnings: {len(warnings)}\n\n"
        "Next: run v51d QA comparing v51b raw dense vs v51c NMS-corrected dense tracking.\n"
    )

    OUT_NOTE.write_text(
        "# Week 8 v51c NMS-corrected Dense Tracking\n\n"
        "## Summary\n\n"
        f"- v51c decision: {decision.iloc[0]['v51c_decision']}\n"
        "- Parameter source: v51a candidate 1 with NMS correction\n"
        f"- frame_stride: {args.frame_stride}\n"
        f"- score_thr: {args.score_thr}\n"
        f"- tracker_iou_thr: {args.tracker_iou_thr}\n"
        f"- nms_iou_thr: {args.nms_iou_thr}\n"
        f"- max_missed: {args.max_missed}\n"
        f"- Processed clips: {processed_clip_count}\n"
        f"- Successful clips: {successful_clip_count}\n"
        f"- Zero tracking clips: {zero_tracking_clip_count}\n"
        f"- Tracking rows total: {tracking_rows_total}\n"
        f"- Unique track sum: {unique_track_sum}\n"
        f"- Raw detection total: {raw_detection_total}\n"
        f"- NMS kept detection total: {nms_kept_total}\n"
        f"- NMS removed detection total: {nms_removed_total}\n"
        f"- Hard issue count: {len(hard_issues)}\n"
        f"- Warning count: {len(warnings)}\n"
        f"- Ready for v51d QA: {ready}\n"
    )

    progress_row = pd.DataFrame([{
        "date": datetime.now().date().isoformat(),
        "stage": "v51c",
        "task_name": "NMS-corrected dense full tracking",
        "status": "PASS" if ready else "PASS_WITH_WARNINGS" if len(hard_issues) == 0 else "BLOCKED",
        "input_summary": str(V45_CLIP_JSON),
        "output_summary": str(OUT),
        "hard_issues": int(len(hard_issues)),
        "warnings": int(len(warnings)),
        "next_action": "v51d dense tracking QA and comparison" if len(hard_issues) == 0 else "Resolve hard issues.",
    }])

    if OUT_PROGRESS.exists():
        old = pd.read_csv(OUT_PROGRESS)
        progress = pd.concat([old, progress_row], ignore_index=True)
    else:
        progress = progress_row

    safe_to_csv(progress, OUT_PROGRESS)

    print()
    print("Saved:")
    print(OUT_TRACKS)
    print(OUT_SUMMARY)
    print(OUT_DECISION)
    print(OUT_ISSUES)
    print(OUT_REPORT)
    print(OUT_NOTE)

    print()
    print("=== v51c decision ===")
    print(decision.to_string(index=False))

    print()
    print("=== v51c issues ===")
    if len(issues_df):
        print(issues_df.to_string(index=False))
    else:
        print("No issues found.")


if __name__ == "__main__":
    main()
