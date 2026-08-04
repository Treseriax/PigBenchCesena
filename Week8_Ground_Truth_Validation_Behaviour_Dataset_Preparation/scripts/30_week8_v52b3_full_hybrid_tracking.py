from pathlib import Path
from datetime import datetime
import json
import csv
import pandas as pd
import numpy as np


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V45_CLIP_JSON = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_level_ground_truth.json"

V52A2_ROWS = W8 / "outputs" / "v52b1_gt_seeded_detector_linked_full" / "week8_v52b1_gt_seeded_detector_linked_rows.csv"
V52A4_ROWS = W8 / "outputs" / "v52b2_tracklet_level_gt_linking_full" / "week8_v52b2_tracklet_level_gt_linking_rows.csv"

OUT = W8 / "outputs" / "v52b3_hybrid_tracking_full"
GALLERY = OUT / "hybrid_tracking_gallery"
REPORTS = W8 / "reports"
NOTES = W8 / "notes"
PROGRESS = W8 / "progress"

for p in [OUT, GALLERY, REPORTS, NOTES, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_ROWS = OUT / "week8_v52b3_hybrid_tracking_rows.csv"
OUT_CLIP_SUMMARY = OUT / "week8_v52b3_clip_summary.csv"
OUT_OBJECT_SUMMARY = OUT / "week8_v52b3_object_summary.csv"
OUT_GALLERY_INDEX = OUT / "week8_v52b3_gallery_index.csv"
OUT_GALLERY_HTML = OUT / "index.html"
OUT_DECISION = OUT / "week8_v52b3_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v52b3_issues.csv"
OUT_REPORT = REPORTS / "week8_v52b3_hybrid_tracking_full_report.md"
OUT_NOTE = NOTES / "week8_v52b3_hybrid_tracking_full_notes.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def clean_str(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


def to_int(x, default=None):
    try:
        if pd.isna(x):
            return default
        return int(round(float(x)))
    except Exception:
        return default


def read_all_frames(video_path):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return [], {}
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "frames_read": len(frames),
    }


def draw_gallery(scan, clip, tracks_df, out_path):
    import cv2

    frames, meta = read_all_frames(clean_str(clip.get("clip_path")))
    if len(frames) == 0:
        return False, "video_read_failed"

    sample_frames = sorted(set([0, len(frames) // 2, len(frames) - 1]))
    panels = []

    colour_map = {
        "blue": (255, 80, 30),
        "green": (60, 220, 60),
        "cyan": (255, 220, 0),
        "red": (40, 40, 255),
        "pink": (220, 80, 255),
        "purple": (180, 70, 220),
        "unknown": (180, 180, 180),
        "not_visible": (130, 130, 130),
        "uncertain": (0, 220, 255),
        "unassigned": (160, 160, 160),
    }

    for fi in sample_frames:
        img = frames[fi].copy()
        h, w = img.shape[:2]

        g = tracks_df[
            (tracks_df["scan_frame_id"] == scan)
            & (tracks_df["frame_index_in_clip"] == fi)
            & (tracks_df["draw_ok"] == True)
        ].copy()

        for _, r in g.iterrows():
            x1 = to_int(r.get("x1"))
            y1 = to_int(r.get("y1"))
            x2 = to_int(r.get("x2"))
            y2 = to_int(r.get("y2"))

            if None in [x1, y1, x2, y2]:
                continue

            colour = clean_str(r.get("visual_marker_colour")) or "unknown"
            bgr = colour_map.get(colour, (255, 255, 255))

            source = clean_str(r.get("hybrid_source"))
            if source == "stable_tracklet":
                thickness = 3
            elif source == "recall_fallback":
                thickness = 1
            else:
                thickness = 2

            cv2.rectangle(img, (x1, y1), (x2, y2), bgr, thickness)

            label = f"{clean_str(r.get('behaviour_pig_id'))}/{colour}/{source}"
            cv2.putText(
                img,
                label,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                bgr,
                1,
            )

        cv2.rectangle(img, (0, 0), (w, 58), (0, 0, 0), -1)
        cv2.putText(
            img,
            f"{scan} hybrid tracking frame={fi}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img,
            "thick=stable tracklet, thin=recall fallback, missing not drawn",
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (230, 230, 230),
            1,
        )

        panels.append(img)

    combined = np.hstack(panels)
    ok = cv2.imwrite(str(out_path), combined)
    return bool(ok), "saved" if ok else "write_failed"


issues = []

for p in [V45_CLIP_JSON, V52A2_ROWS, V52A4_ROWS]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required input missing.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v52b3_decision": "hybrid_tracking_full_blocked_missing_inputs",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v52b_full_hybrid_tracking": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


clips = json.loads(V45_CLIP_JSON.read_text()).get("clips", [])
clip_map = {clean_str(c.get("scan_frame_id")): c for c in clips}

a2 = pd.read_csv(V52A2_ROWS)
a4 = pd.read_csv(V52A4_ROWS)

key_cols = ["scan_frame_id", "final_box_id", "frame_index_in_clip"]

for df in [a2, a4]:
    df["scan_frame_id"] = df["scan_frame_id"].astype(str).map(clean_str)
    df["final_box_id"] = df["final_box_id"].astype(str).map(clean_str)
    df["frame_index_in_clip"] = pd.to_numeric(df["frame_index_in_clip"], errors="coerce").astype("Int64")
    df["draw_ok"] = df["draw_ok"].astype(str).str.lower().isin(["true", "1", "yes"])

# Use only full intersection.
full_scans = sorted(set(a2["scan_frame_id"].unique()).intersection(set(a4["scan_frame_id"].unique())))

a2 = a2[a2["scan_frame_id"].isin(full_scans)].copy()
a4 = a4[a4["scan_frame_id"].isin(full_scans)].copy()

a2_index = {
    (r["scan_frame_id"], r["final_box_id"], int(r["frame_index_in_clip"])): r
    for _, r in a2.iterrows()
}

a4_index = {
    (r["scan_frame_id"], r["final_box_id"], int(r["frame_index_in_clip"])): r
    for _, r in a4.iterrows()
}

all_keys = sorted(set(a2_index.keys()).union(set(a4_index.keys())))

hybrid_rows = []

identity_cols = [
    "scan_frame_id",
    "video_id",
    "clip_path",
    "frame_index_in_clip",
    "final_box_id",
    "behaviour_pig_id",
    "visual_marker_colour",
    "behaviour_code",
    "anchor_frame_index",
]

for k in all_keys:
    r2 = a2_index.get(k)
    r4 = a4_index.get(k)

    use = None
    source = ""
    confidence = ""
    review_needed = True

    if r4 is not None and bool(r4.get("draw_ok")):
        use = r4
        source = "stable_tracklet"
        confidence = "high_stability"
        review_needed = False
    elif r2 is not None and bool(r2.get("draw_ok")):
        use = r2
        source = "recall_fallback"
        confidence = "medium_recall_review"
        review_needed = True
    elif r4 is not None:
        use = r4
        source = "missing"
        confidence = "missing"
        review_needed = True
    elif r2 is not None:
        use = r2
        source = "missing"
        confidence = "missing"
        review_needed = True
    else:
        continue

    out = {}

    for c in identity_cols:
        out[c] = use.get(c, "")

    for c in ["x1", "y1", "x2", "y2"]:
        out[c] = use.get(c, None)

    out["dataset_version"] = "week8_v52b3_hybrid_tracking_full"
    out["hybrid_source"] = source
    out["hybrid_confidence"] = confidence
    out["review_needed"] = bool(review_needed)
    out["draw_ok"] = bool(source in ["stable_tracklet", "recall_fallback"])

    out["v52b2_status"] = clean_str(r4.get("trajectory_status")) if r4 is not None else ""
    out["v52b1_status"] = clean_str(r2.get("trajectory_status")) if r2 is not None else ""

    out["v52b2_draw_ok"] = bool(r4.get("draw_ok")) if r4 is not None else False
    out["v52b1_draw_ok"] = bool(r2.get("draw_ok")) if r2 is not None else False

    out["method"] = "hybrid_stable_tracklet_then_recall_fallback"

    hybrid_rows.append(out)


hybrid = pd.DataFrame(hybrid_rows)
safe_to_csv(hybrid, OUT_ROWS)

clip_summary_rows = []

for scan, g in hybrid.groupby("scan_frame_id"):
    total = len(g)
    stable = int((g["hybrid_source"] == "stable_tracklet").sum())
    fallback = int((g["hybrid_source"] == "recall_fallback").sum())
    missing = int((g["hybrid_source"] == "missing").sum())
    draw = int((g["draw_ok"] == True).sum())

    object_count = int(g["final_box_id"].nunique())
    frame_count = int(g["frame_index_in_clip"].nunique())

    clip_summary_rows.append({
        "scan_frame_id": scan,
        "video_id": clean_str(g["video_id"].iloc[0]),
        "anchor_object_count": object_count,
        "frame_count": frame_count,
        "total_object_frame_rows": total,
        "stable_tracklet_rows": stable,
        "recall_fallback_rows": fallback,
        "missing_rows": missing,
        "draw_ok_rows": draw,
        "stable_tracklet_ratio": float(stable / max(1, total)),
        "recall_fallback_ratio": float(fallback / max(1, total)),
        "missing_ratio": float(missing / max(1, total)),
        "draw_ok_ratio": float(draw / max(1, total)),
        "review_needed_rows": int((g["review_needed"] == True).sum()),
        "review_needed_ratio": float((g["review_needed"] == True).sum() / max(1, total)),
        "status": "processed",
    })

clip_summary = pd.DataFrame(clip_summary_rows)
safe_to_csv(clip_summary, OUT_CLIP_SUMMARY)

object_summary_rows = []

for (scan, obj), g in hybrid.groupby(["scan_frame_id", "final_box_id"]):
    total = len(g)
    object_summary_rows.append({
        "scan_frame_id": scan,
        "final_box_id": obj,
        "behaviour_pig_id": clean_str(g["behaviour_pig_id"].iloc[0]),
        "visual_marker_colour": clean_str(g["visual_marker_colour"].iloc[0]),
        "behaviour_code": clean_str(g["behaviour_code"].iloc[0]),
        "rows_total": int(total),
        "stable_tracklet_rows": int((g["hybrid_source"] == "stable_tracklet").sum()),
        "recall_fallback_rows": int((g["hybrid_source"] == "recall_fallback").sum()),
        "missing_rows": int((g["hybrid_source"] == "missing").sum()),
        "draw_ok_rows": int((g["draw_ok"] == True).sum()),
        "draw_ok_ratio": float((g["draw_ok"] == True).sum() / max(1, total)),
        "review_needed_ratio": float((g["review_needed"] == True).sum() / max(1, total)),
    })

object_summary = pd.DataFrame(object_summary_rows)
safe_to_csv(object_summary, OUT_OBJECT_SUMMARY)

gallery_rows = []

for scan in full_scans:
    clip = clip_map.get(scan)
    if clip is None:
        continue

    out_img = GALLERY / f"{scan}_hybrid_tracking.jpg"
    ok, msg = draw_gallery(scan, clip, hybrid, out_img)

    s = clip_summary[clip_summary["scan_frame_id"] == scan].iloc[0].to_dict()

    gallery_rows.append({
        "scan_frame_id": scan,
        "overlay_path": str(out_img),
        "saved": bool(ok),
        "message": msg,
        "stable_tracklet_ratio": s["stable_tracklet_ratio"],
        "recall_fallback_ratio": s["recall_fallback_ratio"],
        "missing_ratio": s["missing_ratio"],
        "draw_ok_ratio": s["draw_ok_ratio"],
        "review_needed_ratio": s["review_needed_ratio"],
    })

gallery_index = pd.DataFrame(gallery_rows)
safe_to_csv(gallery_index, OUT_GALLERY_INDEX)

html = []
html.append("<!doctype html><html><head><meta charset='utf-8'>")
html.append("<title>Week 8 v52b3 Hybrid Tracking Full</title>")
html.append("<style>body{font-family:Arial;background:#111;color:#eee;margin:20px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(760px,1fr));gap:18px}.card{background:#1e1e1e;border:1px solid #333;border-radius:10px;padding:12px}img{width:100%;border:1px solid #444;border-radius:6px}.meta{font-size:13px;color:#bbb;line-height:1.4}.top{background:#2b2b2b;padding:12px;border-radius:10px;margin-bottom:18px}</style></head><body>")
html.append("<div class='top'><h1>Week 8 v52b3 Hybrid Tracking Full</h1>")
html.append("<p>Priority: stable tracklet boxes from v52b2. Fallback: recall-mode boxes from v52b1 when stable tracklet is missing.</p>")
html.append("<p>Thick boxes = stable tracklet. Thin boxes = recall fallback and should be reviewed.</p></div><div class='grid'>")

for _, r in gallery_index.iterrows():
    if not bool(r["saved"]):
        continue
    rel = Path(r["overlay_path"]).relative_to(OUT)
    html.append("<div class='card'>")
    html.append(f"<h3>{r['scan_frame_id']}</h3><img src='{rel.as_posix()}'>")
    html.append("<div class='meta'>")
    html.append(f"stable tracklet ratio: {float(r['stable_tracklet_ratio']):.3f}<br>")
    html.append(f"recall fallback ratio: {float(r['recall_fallback_ratio']):.3f}<br>")
    html.append(f"missing ratio: {float(r['missing_ratio']):.3f}<br>")
    html.append(f"draw ok ratio: {float(r['draw_ok_ratio']):.3f}<br>")
    html.append(f"review needed ratio: {float(r['review_needed_ratio']):.3f}<br>")
    html.append("</div></div>")

html.append("</div></body></html>")
OUT_GALLERY_HTML.write_text("\n".join(html))

total_rows = int(len(hybrid))
mean_stable = float(clip_summary["stable_tracklet_ratio"].mean()) if len(clip_summary) else 0.0
mean_fallback = float(clip_summary["recall_fallback_ratio"].mean()) if len(clip_summary) else 0.0
mean_missing = float(clip_summary["missing_ratio"].mean()) if len(clip_summary) else 1.0
mean_draw = float(clip_summary["draw_ok_ratio"].mean()) if len(clip_summary) else 0.0
mean_review = float(clip_summary["review_needed_ratio"].mean()) if len(clip_summary) else 1.0

if total_rows == 0:
    issues.append({
        "item": "hybrid_rows",
        "issue_type": "hard_no_rows",
        "issue_detail": "Hybrid tracking produced zero rows.",
        "severity": "hard",
    })

if mean_draw < 0.75:
    issues.append({
        "item": "hybrid_draw_ok",
        "issue_type": "warning_mean_draw_ok_below_0_75",
        "issue_detail": f"Mean draw-ok ratio is {mean_draw:.4f}.",
        "severity": "warning",
    })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
hard_issues = issues_df[issues_df["severity"] == "hard"] if len(issues_df) else pd.DataFrame()
warnings = issues_df[issues_df["severity"] == "warning"] if len(issues_df) else pd.DataFrame()

ready = len(hard_issues) == 0

decision = pd.DataFrame([{
    "v52b3_decision": "hybrid_tracking_full_completed" if ready else "hybrid_tracking_full_blocked",
    "full_clip_count": int(len(clip_summary)),
    "tracking_rows_total": total_rows,
    "mean_stable_tracklet_ratio": mean_stable,
    "mean_recall_fallback_ratio": mean_fallback,
    "mean_missing_ratio": mean_missing,
    "mean_draw_ok_ratio": mean_draw,
    "mean_review_needed_ratio": mean_review,
    "gallery_overlay_count": int(len(gallery_index)),
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "issue_count": int(len(issues_df)),
    "ready_for_v52b_full_hybrid_tracking": bool(ready),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(issues_df, OUT_ISSUES)
safe_to_csv(decision, OUT_DECISION)

OUT_REPORT.write_text(
    "Week 8 v52b3 Hybrid Tracking Full Report\n\n"
    f"Decision: {decision.iloc[0]['v52b3_decision']}\n"
    f"Full clips: {len(clip_summary)}\n"
    f"Tracking rows total: {total_rows}\n"
    f"Mean stable tracklet ratio: {mean_stable:.4f}\n"
    f"Mean recall fallback ratio: {mean_fallback:.4f}\n"
    f"Mean missing ratio: {mean_missing:.4f}\n"
    f"Mean draw-ok ratio: {mean_draw:.4f}\n"
    f"Mean review-needed ratio: {mean_review:.4f}\n"
    f"Hard issues: {len(hard_issues)}\n"
    f"Warnings: {len(warnings)}\n\n"
    "Interpretation:\n"
    "The hybrid method prioritizes identity-stable tracklet-level boxes and uses recall-mode detector-linked boxes only as fallback. "
    "Fallback rows are explicitly flagged for review.\n"
)

OUT_NOTE.write_text(
    "# Week 8 v52b3 Hybrid Tracking Full\n\n"
    "## Summary\n\n"
    f"- v52b3 decision: {decision.iloc[0]['v52b3_decision']}\n"
    f"- Full clips: {len(clip_summary)}\n"
    f"- Tracking rows total: {total_rows}\n"
    f"- Mean stable tracklet ratio: {mean_stable:.4f}\n"
    f"- Mean recall fallback ratio: {mean_fallback:.4f}\n"
    f"- Mean missing ratio: {mean_missing:.4f}\n"
    f"- Mean draw-ok ratio: {mean_draw:.4f}\n"
    f"- Mean review-needed ratio: {mean_review:.4f}\n"
    f"- Hard issue count: {len(hard_issues)}\n"
    f"- Warning count: {len(warnings)}\n"
    f"- Ready for v52b full hybrid tracking: {ready}\n\n"
    "## Interpretation\n\n"
    "v52b2 is more identity-stable but misses too much. v52b1 has better coverage but can be less stable. "
    "v52b3 combines them: stable tracklet first, recall fallback second, with review flags.\n\n"
    f"Gallery: {OUT_GALLERY_HTML}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v52b3",
    "task_name": "Hybrid tracking full",
    "status": "PASS" if ready else "BLOCKED",
    "input_summary": str(V52A2_ROWS) + " + " + str(V52A4_ROWS),
    "output_summary": str(OUT),
    "hard_issues": int(len(hard_issues)),
    "warnings": int(len(warnings)),
    "next_action": "v52b full hybrid tracking" if ready else "Review hybrid full issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_ROWS)
print(OUT_CLIP_SUMMARY)
print(OUT_OBJECT_SUMMARY)
print(OUT_GALLERY_INDEX)
print(OUT_GALLERY_HTML)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_REPORT)
print(OUT_NOTE)

print()
print("=== v52b3 decision ===")
print(decision.to_string(index=False))

print()
print("=== v52b3 issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")

print()
print("=== v52b3 clip summary ===")
print(clip_summary.to_string(index=False))
