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

OUT = W8 / "outputs" / "v51b_dense_full_72_tracking"
PER_CLIP = OUT / "per_clip_tracks"
PER_SUMMARY = OUT / "per_clip_summaries"
REPORTS = W8 / "reports"
NOTES = W8 / "notes"
PROGRESS = W8 / "progress"

for p in [OUT, PER_CLIP, PER_SUMMARY, REPORTS, NOTES, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_TRACKS = OUT / "week8_v51b_dense_full_tracking_rows.csv"
OUT_SUMMARY = OUT / "week8_v51b_dense_clip_tracking_summary.csv"
OUT_DECISION = OUT / "week8_v51b_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v51b_issues.csv"
OUT_REPORT = REPORTS / "week8_v51b_dense_full_72_tracking_report.md"
OUT_NOTE = NOTES / "week8_v51b_dense_full_72_tracking_notes.md"
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


def per_clip_paths(scan):
    return PER_CLIP / f"{scan}_tracks.csv", PER_SUMMARY / f"{scan}_summary.csv"


def read_detector_paths():
    d = pd.read_csv(V50B_DECISION)
    row = d.iloc[0]
    return Path(row["selected_config"]), Path(row["selected_checkpoint"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--score-thr", type=float, default=0.10)
    parser.add_argument("--iou-thr", type=float, default=0.15)
    parser.add_argument("--max-missed", type=int, default=35)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    issues = []

    for p in [V45_CLIP_JSON, V50B_DECISION, V50B_SCRIPT]:
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
            "v51b_decision": "dense_tracking_blocked_missing_inputs",
            "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
            "ready_for_v51c_dense_tracking_qa": False,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }])
        safe_to_csv(decision, OUT_DECISION)
        print(decision.to_string(index=False))
        return

    clips = json.loads(V45_CLIP_JSON.read_text()).get("clips", [])
    config_path, checkpoint_path = read_detector_paths()

    v50b = load_v50b_module()

    print("Initializing detector for v51b dense tracking...")
    print("Config:", config_path)
    print("Checkpoint:", checkpoint_path)
    print("Device:", args.device)
    print("frame_stride:", args.frame_stride)
    print("score_thr:", args.score_thr)
    print("iou_thr:", args.iou_thr)
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
            "v51b_decision": "dense_tracking_blocked_model_init_failed",
            "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
            "ready_for_v51c_dense_tracking_qa": False,
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
            rows, summary = v50b.run_clip_tracking(
                model=model,
                clip=clip,
                max_frames=None,
                frame_stride=args.frame_stride,
                score_thr=args.score_thr,
                iou_thr=args.iou_thr,
                max_missed=args.max_missed,
            )

            for r in rows:
                r["dataset_version"] = "week8_v51b_dense_full_tracking"
                r["run_stage"] = "v51b"
                r["parameter_source"] = "v51a_candidate_1_recall_max"

            tracks_df = pd.DataFrame(rows)

            summary["run_status"] = "processed_v51b"
            summary["frame_stride"] = args.frame_stride
            summary["score_thr"] = args.score_thr
            summary["iou_thr"] = args.iou_thr
            summary["max_missed"] = args.max_missed
            summary["parameter_source"] = "v51a_candidate_1_recall_max"

            safe_to_csv(tracks_df, track_path)
            safe_to_csv(pd.DataFrame([summary]), summary_path)

            summary_rows.append(summary)

            print(
                f"  processed_frames={summary.get('processed_frames')} "
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
                "run_status": "failed_v51b",
                "error": str(e),
                "frame_stride": args.frame_stride,
                "score_thr": args.score_thr,
                "iou_thr": args.iou_thr,
                "max_missed": args.max_missed,
                "parameter_source": "v51a_candidate_1_recall_max",
            }

            safe_to_csv(pd.DataFrame([summary]), summary_path)
            summary_rows.append(summary)
            print("  FAILED:", e)

    # Aggregate.
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
    elapsed = time.time() - start

    if tracking_rows_total == 0:
        issues.append({
            "item": "dense_tracking_rows",
            "issue_type": "hard_no_tracking_rows_generated",
            "issue_detail": "Dense tracking produced zero rows.",
            "severity": "hard",
        })

    if successful_clip_count < len(clips):
        issues.append({
            "item": "dense_tracking_clip_coverage",
            "issue_type": "warning_some_clips_zero_tracking_rows",
            "issue_detail": f"{successful_clip_count}/{len(clips)} clips have tracking rows.",
            "severity": "warning",
        })

    issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
    hard_issues = issues_df[issues_df["severity"] == "hard"] if len(issues_df) else pd.DataFrame()
    warnings = issues_df[issues_df["severity"] == "warning"] if len(issues_df) else pd.DataFrame()

    ready = len(hard_issues) == 0 and successful_clip_count == len(clips)

    decision = pd.DataFrame([{
        "v51b_decision": "dense_full_72_tracking_completed" if ready else "dense_full_72_tracking_completed_with_warnings" if len(hard_issues) == 0 else "dense_full_72_tracking_blocked",
        "parameter_source": "v51a_candidate_1_recall_max",
        "device": args.device,
        "frame_stride": int(args.frame_stride),
        "score_thr": float(args.score_thr),
        "iou_thr": float(args.iou_thr),
        "max_missed": int(args.max_missed),
        "clip_count_target": int(len(clips)),
        "processed_clip_count": processed_clip_count,
        "successful_clip_count": successful_clip_count,
        "zero_tracking_clip_count": zero_tracking_clip_count,
        "tracking_rows_total": tracking_rows_total,
        "unique_scan_count_in_tracks": unique_scan_count,
        "unique_track_total_by_clip_sum": unique_track_sum,
        "elapsed_sec": float(elapsed),
        "hard_issue_count": int(len(hard_issues)),
        "warning_count": int(len(warnings)),
        "issue_count": int(len(issues_df)),
        "ready_for_v51c_dense_tracking_qa": bool(ready),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])

    safe_to_csv(issues_df, OUT_ISSUES)
    safe_to_csv(decision, OUT_DECISION)

    OUT_REPORT.write_text(
        "Week 8 v51b Dense Full 72 Tracking Report\n\n"
        f"Decision: {decision.iloc[0]['v51b_decision']}\n"
        f"Parameter source: v51a candidate 1 recall max\n"
        f"frame_stride: {args.frame_stride}\n"
        f"score_thr: {args.score_thr}\n"
        f"iou_thr: {args.iou_thr}\n"
        f"max_missed: {args.max_missed}\n"
        f"Processed clips: {processed_clip_count}\n"
        f"Successful clips: {successful_clip_count}\n"
        f"Zero tracking clips: {zero_tracking_clip_count}\n"
        f"Tracking rows total: {tracking_rows_total}\n"
        f"Unique track sum: {unique_track_sum}\n"
        f"Elapsed seconds: {elapsed:.2f}\n"
        f"Hard issues: {len(hard_issues)}\n"
        f"Warnings: {len(warnings)}\n\n"
        "Next: run v51c dense tracking QA and compare against v50c/v50f.\n"
    )

    OUT_NOTE.write_text(
        "# Week 8 v51b Dense Full 72 Tracking\n\n"
        "## Summary\n\n"
        f"- v51b decision: {decision.iloc[0]['v51b_decision']}\n"
        "- Parameter source: v51a candidate 1 recall max\n"
        f"- frame_stride: {args.frame_stride}\n"
        f"- score_thr: {args.score_thr}\n"
        f"- iou_thr: {args.iou_thr}\n"
        f"- max_missed: {args.max_missed}\n"
        f"- Processed clips: {processed_clip_count}\n"
        f"- Successful clips: {successful_clip_count}\n"
        f"- Zero tracking clips: {zero_tracking_clip_count}\n"
        f"- Tracking rows total: {tracking_rows_total}\n"
        f"- Unique track sum: {unique_track_sum}\n"
        f"- Hard issue count: {len(hard_issues)}\n"
        f"- Warning count: {len(warnings)}\n"
        f"- Ready for v51c dense tracking QA: {ready}\n"
    )

    progress_row = pd.DataFrame([{
        "date": datetime.now().date().isoformat(),
        "stage": "v51b",
        "task_name": "Dense full 72-clip tracking",
        "status": "PASS" if ready else "PASS_WITH_WARNINGS" if len(hard_issues) == 0 else "BLOCKED",
        "input_summary": str(V45_CLIP_JSON),
        "output_summary": str(OUT),
        "hard_issues": int(len(hard_issues)),
        "warnings": int(len(warnings)),
        "next_action": "v51c dense tracking QA" if len(hard_issues) == 0 else "Resolve dense tracking hard issues.",
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
    print("=== v51b decision ===")
    print(decision.to_string(index=False))

    print()
    print("=== v51b issues ===")
    if len(issues_df):
        print(issues_df.to_string(index=False))
    else:
        print("No issues found.")


if __name__ == "__main__":
    main()
