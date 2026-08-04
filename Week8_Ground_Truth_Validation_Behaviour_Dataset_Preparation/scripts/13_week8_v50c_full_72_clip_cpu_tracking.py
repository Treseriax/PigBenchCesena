from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
import os
import sys
import time
import importlib.util
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V45_CLIP_JSON = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_level_ground_truth.json"
V50B_DECISION = W8 / "outputs" / "v50b_cpu_tracking_dryrun_preflight" / "week8_v50b_decision_summary.csv"
V50B_SCRIPT = W8 / "scripts" / "12_week8_v50b_cpu_tracking_dryrun_preflight.py"

OUT = W8 / "outputs" / "v50c_full_72_clip_cpu_tracking"
PER_CLIP = OUT / "per_clip_tracks"
PER_SUMMARY = OUT / "per_clip_summaries"
REPORTS = W8 / "reports"
NOTES = W8 / "notes"
PROGRESS = W8 / "progress"

for p in [OUT, PER_CLIP, PER_SUMMARY, REPORTS, NOTES, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_TRACKS = OUT / "week8_v50c_full_tracking_rows.csv"
OUT_CLIP_SUMMARY = OUT / "week8_v50c_clip_tracking_summary.csv"
OUT_DECISION = OUT / "week8_v50c_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v50c_issues.csv"
OUT_REPORT = REPORTS / "week8_v50c_full_72_clip_cpu_tracking_report.md"
OUT_NOTE = NOTES / "week8_v50c_full_72_clip_cpu_tracking_notes.md"
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


def load_v50b_module():
    spec = importlib.util.spec_from_file_location("week8_v50b_module", str(V50B_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_v50b_detector_paths():
    d = pd.read_csv(V50B_DECISION)
    row = d.iloc[0].to_dict()
    return Path(row["selected_config"]), Path(row["selected_checkpoint"])


def per_clip_paths(scan_frame_id):
    track_path = PER_CLIP / f"{scan_frame_id}_tracks.csv"
    summary_path = PER_SUMMARY / f"{scan_frame_id}_summary.csv"
    return track_path, summary_path


def summarize_existing_clip(scan_frame_id, clip):
    track_path, summary_path = per_clip_paths(scan_frame_id)

    if summary_path.exists():
        try:
            s = pd.read_csv(summary_path)
            if len(s):
                return s.iloc[0].to_dict()
        except Exception:
            pass

    if track_path.exists():
        try:
            t = pd.read_csv(track_path)
            return {
                "scan_frame_id": scan_frame_id,
                "video_id": clean_str(clip.get("video_id")),
                "clip_path": clean_str(clip.get("clip_path")),
                "video_open_ok": True,
                "frame_count": clean_str(clip.get("generated_frame_count")),
                "processed_frames": int(t["frame_index_in_clip"].nunique()) if "frame_index_in_clip" in t.columns else 0,
                "tracking_rows": int(len(t)),
                "unique_tracks": int(t["track_id"].nunique()) if "track_id" in t.columns else 0,
                "status": "resumed_existing_tracks",
            }
        except Exception:
            pass

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--score-thr", type=float, default=0.30)
    parser.add_argument("--iou-thr", type=float, default=0.30)
    parser.add_argument("--max-missed", type=int, default=15)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    issues = []
    required = [V45_CLIP_JSON, V50B_DECISION, V50B_SCRIPT]

    for p in required:
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
            "v50c_decision": "full_tracking_blocked_missing_inputs",
            "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
            "ready_for_v50d_tracking_qa": False,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }])
        safe_to_csv(decision, OUT_DECISION)
        print(decision.to_string(index=False))
        return

    clip_data = json.loads(V45_CLIP_JSON.read_text())
    clips = clip_data.get("clips", [])

    if args.max_clips and args.max_clips > 0:
        clips_to_process = clips[:args.max_clips]
    else:
        clips_to_process = clips

    config_path, checkpoint_path = read_v50b_detector_paths()

    if not config_path.exists() or not checkpoint_path.exists():
        issues.append({
            "item": "detector_paths",
            "issue_type": "hard_detector_config_or_checkpoint_missing",
            "issue_detail": f"config={config_path}; checkpoint={checkpoint_path}",
            "severity": "hard",
        })
        issues_df = pd.DataFrame(issues)
        safe_to_csv(issues_df, OUT_ISSUES)
        decision = pd.DataFrame([{
            "v50c_decision": "full_tracking_blocked_detector_missing",
            "selected_config": str(config_path),
            "selected_checkpoint": str(checkpoint_path),
            "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
            "ready_for_v50d_tracking_qa": False,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }])
        safe_to_csv(decision, OUT_DECISION)
        print(decision.to_string(index=False))
        return

    v50b = load_v50b_module()

    print("Initializing detector...")
    print("Config:", config_path)
    print("Checkpoint:", checkpoint_path)
    print("Device:", args.device)

    try:
        model = v50b.init_mmdet_model(config_path, checkpoint_path, args.device)
        model_init_ok = True
    except Exception as e:
        model_init_ok = False
        issues.append({
            "item": "model_init",
            "issue_type": "hard_model_init_failed",
            "issue_detail": str(e),
            "severity": "hard",
        })

    if not model_init_ok:
        issues_df = pd.DataFrame(issues)
        safe_to_csv(issues_df, OUT_ISSUES)
        decision = pd.DataFrame([{
            "v50c_decision": "full_tracking_blocked_model_init_failed",
            "selected_config": str(config_path),
            "selected_checkpoint": str(checkpoint_path),
            "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
            "ready_for_v50d_tracking_qa": False,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }])
        safe_to_csv(decision, OUT_DECISION)
        print(decision.to_string(index=False))
        return

    all_summary_rows = []
    start_time = time.time()

    for idx, clip in enumerate(clips_to_process, start=1):
        scan = clean_str(clip.get("scan_frame_id"))
        print()
        print(f"[{idx}/{len(clips_to_process)}] Processing {scan}")

        track_path, summary_path = per_clip_paths(scan)

        if args.resume and track_path.exists() and summary_path.exists():
            existing_summary = summarize_existing_clip(scan, clip)
            if existing_summary is not None:
                existing_summary["run_status"] = "skipped_resume"
                all_summary_rows.append(existing_summary)
                print(f"  resume skip: {track_path}")
                continue

        try:
            rows, summary = v50b.run_clip_tracking(
                model=model,
                clip=clip,
                max_frames=None,
                frame_stride=max(1, args.frame_stride),
                score_thr=args.score_thr,
                iou_thr=args.iou_thr,
                max_missed=args.max_missed,
            )

            for r in rows:
                r["dataset_version"] = "week8_v50c_full_tracking"
                r["run_stage"] = "v50c"
                r["frame_stride"] = int(args.frame_stride)
                r["score_thr"] = float(args.score_thr)
                r["iou_thr"] = float(args.iou_thr)

            tracks_df = pd.DataFrame(rows)
            summary["run_status"] = "processed_v50c"
            summary["frame_stride"] = int(args.frame_stride)
            summary["score_thr"] = float(args.score_thr)
            summary["iou_thr"] = float(args.iou_thr)

            safe_to_csv(tracks_df, track_path)
            safe_to_csv(pd.DataFrame([summary]), summary_path)

            all_summary_rows.append(summary)

            print(
                f"  frames={summary.get('processed_frames')} "
                f"rows={summary.get('tracking_rows')} "
                f"tracks={summary.get('unique_tracks')}"
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
                "status": "tracking_failed",
                "run_status": "failed_v50c",
                "error": str(e),
                "frame_stride": int(args.frame_stride),
                "score_thr": float(args.score_thr),
                "iou_thr": float(args.iou_thr),
            }

            safe_to_csv(pd.DataFrame([summary]), summary_path)
            all_summary_rows.append(summary)
            print("  FAILED:", e)

    # Aggregate per-clip outputs.
    all_track_dfs = []
    for clip in clips_to_process:
        scan = clean_str(clip.get("scan_frame_id"))
        track_path, _ = per_clip_paths(scan)
        if track_path.exists():
            try:
                t = pd.read_csv(track_path)
                if len(t):
                    all_track_dfs.append(t)
            except Exception as e:
                issues.append({
                    "item": scan,
                    "issue_type": "warning_failed_to_read_per_clip_track_csv",
                    "issue_detail": str(e),
                    "severity": "warning",
                })

    if all_track_dfs:
        full_tracks = pd.concat(all_track_dfs, ignore_index=True)
    else:
        full_tracks = pd.DataFrame()

    summary_df = pd.DataFrame(all_summary_rows)

    safe_to_csv(full_tracks, OUT_TRACKS)
    safe_to_csv(summary_df, OUT_CLIP_SUMMARY)

    processed_clip_count = int(len(summary_df))
    successful_clip_count = int((summary_df["tracking_rows"].fillna(0).astype(float) > 0).sum()) if len(summary_df) else 0
    zero_tracking_clip_count = int((summary_df["tracking_rows"].fillna(0).astype(float) == 0).sum()) if len(summary_df) else 0
    tracking_rows_total = int(len(full_tracks))
    unique_scan_count = int(full_tracks["scan_frame_id"].nunique()) if len(full_tracks) and "scan_frame_id" in full_tracks.columns else 0
    unique_track_total = int(full_tracks.groupby("scan_frame_id")["track_id"].nunique().sum()) if len(full_tracks) and "track_id" in full_tracks.columns else 0
    elapsed_sec = time.time() - start_time

    if tracking_rows_total == 0:
        issues.append({
            "item": "full_tracking_rows",
            "issue_type": "hard_no_tracking_rows_generated",
            "issue_detail": "Full tracking generated zero rows.",
            "severity": "hard",
        })

    if successful_clip_count < len(clips_to_process):
        issues.append({
            "item": "clip_coverage",
            "issue_type": "warning_some_clips_have_zero_tracking_rows",
            "issue_detail": f"{successful_clip_count}/{len(clips_to_process)} clips have tracking rows.",
            "severity": "warning",
        })

    issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
    hard_issues = issues_df[issues_df["severity"] == "hard"] if len(issues_df) else pd.DataFrame()
    warnings = issues_df[issues_df["severity"] == "warning"] if len(issues_df) else pd.DataFrame()

    ready_for_qa = len(hard_issues) == 0 and successful_clip_count == len(clips_to_process)

    decision = pd.DataFrame([{
        "v50c_decision": "full_72_clip_tracking_completed" if ready_for_qa else "full_72_clip_tracking_completed_with_warnings" if len(hard_issues) == 0 else "full_72_clip_tracking_blocked",
        "selected_config": str(config_path),
        "selected_checkpoint": str(checkpoint_path),
        "device": args.device,
        "frame_stride": int(args.frame_stride),
        "score_thr": float(args.score_thr),
        "iou_thr": float(args.iou_thr),
        "clip_count_target": int(len(clips_to_process)),
        "processed_clip_count": int(processed_clip_count),
        "successful_clip_count": int(successful_clip_count),
        "zero_tracking_clip_count": int(zero_tracking_clip_count),
        "tracking_rows_total": int(tracking_rows_total),
        "unique_scan_count_in_tracks": int(unique_scan_count),
        "unique_track_total_by_clip_sum": int(unique_track_total),
        "elapsed_sec": float(elapsed_sec),
        "hard_issue_count": int(len(hard_issues)),
        "warning_count": int(len(warnings)),
        "issue_count": int(len(issues_df)),
        "ready_for_v50d_tracking_qa": bool(ready_for_qa),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])

    safe_to_csv(issues_df, OUT_ISSUES)
    safe_to_csv(decision, OUT_DECISION)

    report = (
        "Week 8 v50c Full 72-Clip CPU Tracking Report\n\n"
        f"Decision: {decision.iloc[0]['v50c_decision']}\n"
        f"Target clips: {len(clips_to_process)}\n"
        f"Processed clips: {processed_clip_count}\n"
        f"Successful clips: {successful_clip_count}\n"
        f"Zero-tracking clips: {zero_tracking_clip_count}\n"
        f"Tracking rows total: {tracking_rows_total}\n"
        f"Unique scan count in tracks: {unique_scan_count}\n"
        f"Frame stride: {args.frame_stride}\n"
        f"Score threshold: {args.score_thr}\n"
        f"IoU threshold: {args.iou_thr}\n"
        f"Elapsed seconds: {elapsed_sec:.2f}\n"
        f"Hard issues: {len(hard_issues)}\n"
        f"Warnings: {len(warnings)}\n\n"
        "Interpretation:\n"
        "This stage creates full 72-clip tracking coverage for the Week 8 visual validation interface. "
        "The output is sampled by frame_stride and should be QA-checked before identity/behaviour integration.\n\n"
        "Next:\n"
        "Run v50d tracking QA and visual overlay diagnostics.\n"
    )
    OUT_REPORT.write_text(report)

    OUT_NOTE.write_text(
        "# Week 8 v50c Full 72-Clip CPU Tracking\n\n"
        "## Summary\n\n"
        f"- v50c decision: {decision.iloc[0]['v50c_decision']}\n"
        f"- Target clips: {len(clips_to_process)}\n"
        f"- Processed clips: {processed_clip_count}\n"
        f"- Successful clips: {successful_clip_count}\n"
        f"- Zero-tracking clips: {zero_tracking_clip_count}\n"
        f"- Tracking rows total: {tracking_rows_total}\n"
        f"- Frame stride: {args.frame_stride}\n"
        f"- Hard issue count: {len(hard_issues)}\n"
        f"- Warning count: {len(warnings)}\n"
        f"- Ready for v50d tracking QA: {ready_for_qa}\n\n"
        "## Output\n\n"
        f"- Full tracking rows: {OUT_TRACKS}\n"
        f"- Clip summary: {OUT_CLIP_SUMMARY}\n"
    )

    progress_row = pd.DataFrame([{
        "date": datetime.now().date().isoformat(),
        "stage": "v50c",
        "task_name": "Full 72-clip CPU tracking run",
        "status": "PASS" if ready_for_qa else "PASS_WITH_WARNINGS" if len(hard_issues) == 0 else "BLOCKED",
        "input_summary": str(V45_CLIP_JSON),
        "output_summary": str(OUT),
        "hard_issues": int(len(hard_issues)),
        "warnings": int(len(warnings)),
        "next_action": "v50d tracking QA and visual overlay diagnostics" if len(hard_issues) == 0 else "Resolve full tracking hard issues.",
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
    print(OUT_CLIP_SUMMARY)
    print(OUT_DECISION)
    print(OUT_ISSUES)
    print(OUT_REPORT)
    print(OUT_NOTE)

    print()
    print("=== v50c decision ===")
    print(decision.to_string(index=False))

    print()
    print("=== v50c issues ===")
    if len(issues_df):
        print(issues_df.to_string(index=False))
    else:
        print("No issues found.")


if __name__ == "__main__":
    main()
