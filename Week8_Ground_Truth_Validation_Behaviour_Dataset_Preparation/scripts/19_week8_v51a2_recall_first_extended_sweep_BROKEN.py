from pathlib import Path
from datetime import datetime
import argparse
import csv
import json
import os
import importlib.util
import pandas as pd
import numpy as np


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V45_CLIP_JSON = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_level_ground_truth.json"
V45_ANCHORS = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_object_propagated_annotations.csv"
V50B_DECISION = W8 / "outputs" / "v50b_cpu_tracking_dryrun_preflight" / "week8_v50b_decision_summary.csv"
V50B_SCRIPT = W8 / "scripts" / "12_week8_v50b_cpu_tracking_dryrun_preflight.py"
V50F_CLIP_RECALL = W8 / "outputs" / "v50f_anchor_tracking_recall_audit" / "week8_v50f_clip_tracking_recall_summary.csv"
V51A_CANDIDATES = W8 / "outputs" / "v51a_tracking_maximization_sweep" / "week8_v51a_candidate_parameter_results.csv"

OUT = W8 / "outputs" / "v51a2_recall_first_extended_sweep"
DETAIL = OUT / "candidate_details"
REPORTS = W8 / "reports"
NOTES = W8 / "notes"
PROGRESS = W8 / "progress"

for p in [OUT, DETAIL, REPORTS, NOTES, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_CANDIDATES = OUT / "week8_v51a2_candidate_parameter_results.csv"
OUT_SELECTED = OUT / "week8_v51a2_selected_recall_first_tracking_parameters.csv"
OUT_OBJECT_MATCHES = OUT / "week8_v51a2_anchor_object_candidate_matches.csv"
OUT_DECISION = OUT / "week8_v51a2_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v51a2_issues.csv"
OUT_REPORT = REPORTS / "week8_v51a2_recall_first_extended_sweep_report.md"
OUT_NOTE = NOTES / "week8_v51a2_recall_first_extended_sweep_notes.md"
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


def to_float(x, default=None):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


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


def nms_detections(detections, iou_thr):
    if not detections:
        return []

    dets = sorted(detections, key=lambda d: float(d["score"]), reverse=True)
    keep = []

    while dets:
        best = dets.pop(0)
        keep.append(best)
        remaining = []
        for d in dets:
            if iou_xyxy(best["bbox"], d["bbox"]) <= iou_thr:
                remaining.append(d)
        dets = remaining

    return keep


class SimpleIoUTracker:
    def __init__(self, iou_thr=0.10, max_missed=50):
        self.iou_thr = iou_thr
        self.max_missed = max_missed
        self.next_track_id = 1
        self.tracks = {}

    def update(self, detections, frame_index):
        assigned_det = set()
        assigned_track = set()
        matches = []

        candidates = []
        for tid, tr in self.tracks.items():
            for di, det in enumerate(detections):
                val = iou_xyxy(tr["bbox"], det["bbox"])
                if val >= self.iou_thr:
                    candidates.append((val, tid, di))

        candidates.sort(reverse=True, key=lambda x: x[0])

        for val, tid, di in candidates:
            if tid in assigned_track or di in assigned_det:
                continue
            assigned_track.add(tid)
            assigned_det.add(di)
            matches.append((tid, di, val))

        output = []

        for tid, di, val in matches:
            det = detections[di]
            self.tracks[tid]["bbox"] = det["bbox"]
            self.tracks[tid]["last_frame"] = frame_index
            output.append({
                **det,
                "track_id": tid,
                "match_iou": val,
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
            }
            output.append({
                **det,
                "track_id": tid,
                "match_iou": None,
                "track_status": "new_track",
            })

        for tid in list(self.tracks.keys()):
            if frame_index - self.tracks[tid]["last_frame"] > self.max_missed:
                del self.tracks[tid]

        return output


def load_v50b_module():
    spec = importlib.util.spec_from_file_location("week8_v50b_module", str(V50B_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_anchor_frame(clip):
    fps = to_float(clip.get("fps_used"), 25.0) or 25.0
    duration = to_float(clip.get("duration_sec"), 10.0) or 10.0
    scan = clean_str(clip.get("scan_frame_id"))

    if scan in ["scanframe_0000", "scanframe_0006", "scanframe_0012", "scanframe_0018"]:
        anchor_rel_sec = 0.0
    else:
        anchor_rel_sec = duration / 2.0

    return int(round(anchor_rel_sec * fps)), anchor_rel_sec


def nearest_frame(frames, target):
    if not frames:
        return None
    arr = np.array(sorted([int(x) for x in frames]))
    return int(arr[np.argmin(np.abs(arr - int(target)))])


def best_anchor_match(anchor_box, tracks_frame):
    best_iou = 0.0
    best_track_id = ""
    best_score = None

    for _, tr in tracks_frame.iterrows():
        tbox = [
            to_float(tr.get("x1")),
            to_float(tr.get("y1")),
            to_float(tr.get("x2")),
            to_float(tr.get("y2")),
        ]

        if not all(v is not None for v in tbox):
            continue

        val = iou_xyxy(anchor_box, tbox)

        if val > best_iou:
            best_iou = val
            best_track_id = clean_str(tr.get("track_id"))
            best_score = to_float(tr.get("score"))

    return best_iou, best_track_id, best_score


def run_clip_candidate(v50b, model, clip, frame_stride, score_thr, tracker_iou_thr, nms_iou_thr, max_missed):
    import cv2
    from mmdet.apis import inference_detector

    scan = clean_str(clip.get("scan_frame_id"))
    video_id = clean_str(clip.get("video_id"))
    clip_path = Path(clean_str(clip.get("clip_path")))

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        cap.release()
        return pd.DataFrame(), {
            "scan_frame_id": scan,
            "status": "video_open_failed",
            "processed_frames": 0,
            "tracking_rows": 0,
            "unique_tracks": 0,
        }

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = SimpleIoUTracker(iou_thr=tracker_iou_thr, max_missed=max_missed)

    rows = []
    raw_detection_count = 0
    nms_kept_count = 0
    processed = 0
    frame_index = 0

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

        detections = nms_detections(detections, nms_iou_thr)
        nms_kept_count += len(detections)

        tracked = tracker.update(detections, frame_index)

        for t in tracked:
            x1, y1, x2, y2 = t["bbox"]
            rows.append({
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
            })

        processed += 1
        frame_index += 1

    cap.release()

    df = pd.DataFrame(rows)

    summary = {
        "scan_frame_id": scan,
        "video_id": video_id,
        "processed_frames": processed,
        "tracking_rows": len(df),
        "unique_tracks": int(df["track_id"].nunique()) if len(df) else 0,
        "raw_detection_count": raw_detection_count,
        "nms_kept_detection_count": nms_kept_count,
        "nms_removed_detection_count": raw_detection_count - nms_kept_count,
        "status": "processed",
    }

    return df, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-test-clips", type=int, default=16)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    issues = []

    required = [V45_CLIP_JSON, V45_ANCHORS, V50B_DECISION, V50B_SCRIPT]
    for p in required:
        if not p.exists():
            issues.append({
                "item": str(p),
                "issue_type": "hard_missing_required_input",
                "issue_detail": "Required artifact missing.",
                "severity": "hard",
            })

    if issues:
        issues_df = pd.DataFrame(issues)
        safe_to_csv(issues_df, OUT_ISSUES)
        decision = pd.DataFrame([{
            "v51a2_decision": "recall_first_sweep_blocked_missing_inputs",
            "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
            "ready_for_v51b_dense_full_tracking": False,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }])
        safe_to_csv(decision, OUT_DECISION)
        print(decision.to_string(index=False))
        return

    clip_data = json.loads(V45_CLIP_JSON.read_text())
    clips = clip_data.get("clips", [])
    clip_map = {clean_str(c.get("scan_frame_id")): c for c in clips}

    anchors = pd.read_csv(V45_ANCHORS)
    for c in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
        anchors[c] = pd.to_numeric(anchors[c], errors="coerce")

    anchors_by_scan = {str(k): v.copy() for k, v in anchors.groupby("scan_frame_id")}

    # Focus mostly on hard clips, but include representative first clips.
    selected_scans = []

    if V50F_CLIP_RECALL.exists():
        recall = pd.read_csv(V50F_CLIP_RECALL)
        hard = recall.sort_values("filtered_anchor_recall_strong_or_moderate").head(args.max_test_clips)["scan_frame_id"].tolist()
        for s in hard:
            if s not in selected_scans:
                selected_scans.append(s)

    for s in list(clip_map.keys())[:6]:
        if s not in selected_scans:
            selected_scans.append(s)

    selected_scans = selected_scans[:args.max_test_clips]
    selected_clips = [clip_map[s] for s in selected_scans if s in clip_map]

    candidates = []
    cid = 1

    for score_thr in [0.03, 0.05, 0.08, 0.10]:
        for tracker_iou_thr in [0.10, 0.15]:
            for nms_iou_thr in [0.45, 0.55, 0.65]:
                candidates.append({
                    "candidate_id": cid,
                    "frame_stride": 1,
                    "score_thr": score_thr,
                    "tracker_iou_thr": tracker_iou_thr,
                    "nms_iou_thr": nms_iou_thr,
                    "max_missed": 50,
                })
                cid += 1

    v50b = load_v50b_module()

    v50b_decision = pd.read_csv(V50B_DECISION).iloc[0].to_dict()
    config_path = Path(v50b_decision["selected_config"])
    checkpoint_path = Path(v50b_decision["selected_checkpoint"])

    print("Initializing detector for v51a2 recall-first sweep...")
    print("Config:", config_path)
    print("Checkpoint:", checkpoint_path)
    print("Device:", args.device)

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
            "v51a2_decision": "recall_first_sweep_blocked_model_init_failed",
            "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
            "ready_for_v51b_dense_full_tracking": False,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }])
        safe_to_csv(decision, OUT_DECISION)
        print(decision.to_string(index=False))
        return

    candidate_rows = []
    object_rows = []

    for idx, cand in enumerate(candidates, start=1):
        print()
        print(f"=== v51a2 candidate {idx}/{len(candidates)}: {cand} ===")

        all_tracks = []

        for clip in selected_clips:
            scan = clean_str(clip.get("scan_frame_id"))
            print("  processing", scan)

            df, summary = run_clip_candidate(
                v50b=v50b,
                model=model,
                clip=clip,
                frame_stride=cand["frame_stride"],
                score_thr=cand["score_thr"],
                tracker_iou_thr=cand["tracker_iou_thr"],
                nms_iou_thr=cand["nms_iou_thr"],
                max_missed=cand["max_missed"],
            )

            if len(df):
                df["candidate_id"] = cand["candidate_id"]
                all_tracks.append(df)

        if all_tracks:
            tracks = pd.concat(all_tracks, ignore_index=True)
        else:
            tracks = pd.DataFrame()

        detail_path = DETAIL / f"candidate_{cand['candidate_id']:02d}_tracks_sample.csv"
        safe_to_csv(tracks.head(2000), detail_path)

        tracks_by_scan = {str(k): v.copy() for k, v in tracks.groupby("scan_frame_id")} if len(tracks) else {}

        total_anchor = 0
        strong = 0
        moderate = 0
        weak = 0
        miss = 0
        duplicate_pressures = []

        for clip in selected_clips:
            scan = clean_str(clip.get("scan_frame_id"))
            anchor_frame, _ = get_anchor_frame(clip)
            an = anchors_by_scan.get(scan, pd.DataFrame())
            tr = tracks_by_scan.get(scan, pd.DataFrame())

            if len(tr):
                frames = tr["frame_index_in_clip"].dropna().astype(int).unique().tolist()
                nf = nearest_frame(frames, anchor_frame)
                tr_frame = tr[tr["frame_index_in_clip"].astype(int) == int(nf)] if nf is not None else pd.DataFrame()
            else:
                nf = None
                tr_frame = pd.DataFrame()

            duplicate_pressures.append(len(tr_frame) / max(1, len(an)) if len(an) else 0.0)

            for _, a in an.iterrows():
                abox = [
                    to_float(a.get("bbox_x1")),
                    to_float(a.get("bbox_y1")),
                    to_float(a.get("bbox_x2")),
                    to_float(a.get("bbox_y2")),
                ]

                if not all(v is not None for v in abox):
                    continue

                best_iou, best_track_id, best_score = best_anchor_match(abox, tr_frame)

                if best_iou >= 0.50:
                    status = "strong"
                    strong += 1
                elif best_iou >= 0.30:
                    status = "moderate"
                    moderate += 1
                elif best_iou >= 0.10:
                    status = "weak"
                    weak += 1
                else:
                    status = "miss"
                    miss += 1

                total_anchor += 1

                object_rows.append({
                    **cand,
                    "scan_frame_id": scan,
                    "final_box_id": clean_str(a.get("final_box_id")),
                    "visual_marker_colour": clean_str(a.get("visual_marker_colour")),
                    "behaviour_code": clean_str(a.get("behaviour_code")),
                    "anchor_frame": anchor_frame,
                    "nearest_tracking_frame": nf,
                    "best_iou": best_iou,
                    "best_track_id": best_track_id,
                    "best_score": best_score,
                    "match_status": status,
                })

        strong_moderate_recall = (strong + moderate) / total_anchor if total_anchor else 0.0
        weak_inclusive_recall = (strong + moderate + weak) / total_anchor if total_anchor else 0.0
        miss_rate = miss / total_anchor if total_anchor else 1.0
        duplicate_pressure = float(np.mean(duplicate_pressures)) if duplicate_pressures else 0.0

        tracks_total = int(len(tracks))
        unique_tracks_total = int(tracks.groupby("scan_frame_id")["track_id"].nunique().sum()) if len(tracks) else 0

        # Recall-first score: prioritize weak-inclusive recall, then strong/moderate.
        # Very large duplicate pressure is penalized, but row count is only lightly penalized.
        recall_first_score = (
            weak_inclusive_recall
            + 0.50 * strong_moderate_recall
            - 0.04 * max(0.0, duplicate_pressure - 2.0)
            - 0.000004 * tracks_total
        )

        candidate_rows.append({
            **cand,
            "test_clip_count": len(selected_clips),
            "anchor_object_count": total_anchor,
            "strong_match_count": strong,
            "moderate_match_count": moderate,
            "weak_match_count": weak,
            "miss_count": miss,
            "strong_moderate_recall": strong_moderate_recall,
            "weak_inclusive_recall": weak_inclusive_recall,
            "miss_rate": miss_rate,
            "tracking_rows_total": tracks_total,
            "unique_tracks_total_by_clip_sum": unique_tracks_total,
            "avg_duplicate_pressure_tracks_per_anchor_frame_object": duplicate_pressure,
            "recall_first_score": recall_first_score,
            "detail_track_sample_path": str(detail_path),
        })

        print(
            f"  strong/mod={strong_moderate_recall:.3f} "
            f"weak_incl={weak_inclusive_recall:.3f} "
            f"miss={miss_rate:.3f} rows={tracks_total} "
            f"dup={duplicate_pressure:.2f} score={recall_first_score:.3f}"
        )

    candidates_df = pd.DataFrame(candidate_rows).sort_values(
        ["recall_first_score", "weak_inclusive_recall", "strong_moderate_recall"],
        ascending=[False, False, False],
    )

    object_matches = pd.DataFrame(object_rows)

    selected = candidates_df.iloc[0].to_dict()
    selected_df = pd.DataFrame([selected])

    safe_to_csv(candidates_df, OUT_CANDIDATES)
    safe_to_csv(object_matches, OUT_OBJECT_MATCHES)
    safe_to_csv(selected_df, OUT_SELECTED)

    if selected["weak_inclusive_recall"] < 0.80:
        issues.append({
            "item": "recall_first_selected_candidate",
            "issue_type": "warning_weak_inclusive_recall_below_0_80",
            "issue_detail": f"Selected weak-inclusive recall is {selected['weak_inclusive_recall']:.4f}.",
            "severity": "warning",
        })

    if selected["strong_moderate_recall"] < 0.65:
        issues.append({
            "item": "recall_first_selected_candidate",
            "issue_type": "warning_strong_moderate_recall_below_0_65",
            "issue_detail": f"Selected strong/moderate recall is {selected['strong_moderate_recall']:.4f}.",
            "severity": "warning",
        })

    if selected["avg_duplicate_pressure_tracks_per_anchor_frame_object"] > 2.5:
        issues.append({
            "item": "duplicate_pressure",
            "issue_type": "warning_high_duplicate_pressure",
            "issue_detail": f"Selected duplicate pressure is {selected['avg_duplicate_pressure_tracks_per_anchor_frame_object']:.4f}.",
            "severity": "warning",
        })

    issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
    hard_issues = issues_df[issues_df["severity"] == "hard"] if len(issues_df) else pd.DataFrame()
    warnings = issues_df[issues_df["severity"] == "warning"] if len(issues_df) else pd.DataFrame()

    ready = len(hard_issues) == 0

    decision = pd.DataFrame([{
        "v51a2_decision": "recall_first_extended_sweep_completed",
        "test_clip_count": int(len(selected_clips)),
        "candidate_count": int(len(candidates_df)),
        "selected_candidate_id": int(selected["candidate_id"]),
        "selected_frame_stride": int(selected["frame_stride"]),
        "selected_score_thr": float(selected["score_thr"]),
        "selected_tracker_iou_thr": float(selected["tracker_iou_thr"]),
        "selected_nms_iou_thr": float(selected["nms_iou_thr"]),
        "selected_max_missed": int(selected["max_missed"]),
        "selected_strong_moderate_recall": float(selected["strong_moderate_recall"]),
        "selected_weak_inclusive_recall": float(selected["weak_inclusive_recall"]),
        "selected_miss_rate": float(selected["miss_rate"]),
        "selected_tracking_rows_total_test_subset": int(selected["tracking_rows_total"]),
        "selected_avg_duplicate_pressure": float(selected["avg_duplicate_pressure_tracks_per_anchor_frame_object"]),
        "selected_recall_first_score": float(selected["recall_first_score"]),
_avg_duplicate_pressure": float(selected["avg        "hard_issue_count": int(len(hard_issues)),
        "warning_count": int(len(warnings)),
        "issue_count": int(len(issues_df)),
        "ready_for_v51b_dense_full_tracking": bool(ready),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])

    safe_to_csv(issues_df, OUT_ISSUES)
    safe_to_csv(decision, OUT_DECISION)

    report = (
        "Week 8 v51a2 Recall-first Extended Tracking Sweep Report\n\n"
        f"Decision: {decision.iloc[0]['v51a2_decision']}\n"
        f"Test clips: {len(selected_clips)}\n"
        f"Candidates tested: {len(candidates_df)}\n\n"
        "Selected recall-first parameters:\n"
        f"- frame_stride: {int(selected['frame_stride'])}\n"
        f"- score_thr: {float(selected['score_thr'])}\n"
        f"- tracker_iou_thr: {float(selected['tracker_iou_thr'])}\n"
        f"- nms_iou_thr: {float(selected['nms_iou_thr'])}\n"
        f"- max_missed: {int(selected['max_missed'])}\n\n"
        "Selected performance:\n"
        f"- strong/moderate recall: {float(selected['strong_moderate_recall']):.4f}\n"
        f"- weak-inclusive recall: {float(selected['weak_inclusive_recall']):.4f}\n"
        f"- miss rate: {float(selected['miss_rate']):.4f}\n"
        f"- duplicate pressure: {float(selected['avg_duplicate_pressure_tracks_per_anchor_frame_object']):.4f}\n"
        f"- recall-first score: {float(selected['recall_first_score']):.4f}\n\n"
        "Interpretation:\n"
        "This stage intentionally prioritizes maximum tracking recall before full 72-clip dense tracking.\n"
    )
    OUT_REPORT.write_text(report)

    OUT_NOTE.write_text(
        "# Week 8 v51a2 Recall-first Extended Tracking Sweep\n\n"
        "## Summary\n\n"
        "- v51a2 decision: recall_first_extended_sweep_completed\n"
        f"- Test clips: {len(selected_clips)}\n"
        f"- Candidates tested: {len(candidates_df)}\n"
        f"- Selected candidate: {int(selected['candidate_id'])}\n"
        f"- Selected frame_stride: {int(selected['frame_stride'])}\n"
        f"- Selected score_thr: {float(selected['score_thr'])}\n"
        f"- Selected tracker_iou_thr: {float(selected['tracker_iou_thr'])}\n"
        f"- Selected nms_iou_thr: {float(selected['nms_iou_thr'])}\n"
        f"- Selected max_missed: {int(selected['max_missed'])}\n"
        f"- Selected strong/moderate recall: {float(selected['strong_moderate_recall']):.4f}\n"
        f"- Selected weak-inclusive recall: {float(selected['weak_inclusive_recall']):.4f}\n"
        f"- Selected miss rate: {float(selected['miss_rate']):.4f}\n"
        f"- Selected duplicate pressure: {float(selected['avg_duplicate_pressure_tracks_per_anchor_frame_object']):.4f}\n"
        f"- Hard issue count: {len(hard_issues)}\n"
        f"- Warning count: {len(warnings)}\n"
        f"- Ready for v51b dense full tracking: {ready}\n\n"
        "## Next\n\n"
        "Run v51b full 72-clip dense tracking with the recall-first selected parameters.\n"
    )

    progress_row = pd.DataFrame([{
        "date": datetime.now().date().isoformat(),
        "stage": "v51a2",
        "task_name": "Recall-first extended tracking sweep",
        "status": "PASS" if ready else "BLOCKED",
        "input_summary": str(V45_CLIP_JSON),
        "output_summary": str(OUT),
        "hard_issues": int(len(hard_issues)),
        "warnings": int(len(warnings)),
        "next_action": "v51b full 72-clip dense tracking using recall-first parameters" if ready else "Resolve hard issues.",
    }])

    if OUT_PROGRESS.exists():
        old = pd.read_csv(OUT_PROGRESS)
        progress = pd.concat([old, progress_row], ignore_index=True)
    else:
        progress = progress_row

    safe_to_csv(progress, OUT_PROGRESS)

    print()
    print("Saved:")
    print(OUT_CANDIDATES)
    print(OUT_SELECTED)
    print(OUT_OBJECT_MATCHES)
    print(OUT_DECISION)
    print(OUT_ISSUES)
    print(OUT_REPORT)
    print(OUT_NOTE)

    print()
    print("=== v51a2 decision ===")
    print(decision.to_string(index=False))

    print()
    print("=== v51a2 top candidates ===")
    print(candidates_df.head(12).to_string(index=False))

    print()
    print("=== v51a2 issues ===")
    if len(issues_df):
        print(issues_df.to_string(index=False))
    else:
        print("No issues found.")


if __name__ == "__main__":
    main()
