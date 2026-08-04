from pathlib import Path
from datetime import datetime
import json
import csv
import pandas as pd
import numpy as np


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

LOCKED_JSON = W8 / "outputs" / "v43_setup_input_audit" / "week8_v43c_final_locked_inputs.json"
V45_CLIP_OBJECT_CSV = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_object_propagated_annotations.csv"

OUT = W8 / "outputs" / "v49b_exact_anchor_overlay_gallery"
IMG_DIR = OUT / "anchor_overlays_exact"
REPORTS = W8 / "reports"
NOTES = W8 / "notes"
PROGRESS = W8 / "progress"

for p in [OUT, IMG_DIR, REPORTS, NOTES, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_INDEX = OUT / "index.html"
OUT_OVERLAY_INDEX = OUT / "week8_v49b_anchor_overlay_index.csv"
OUT_DECISION = OUT / "week8_v49b_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v49b_issues.csv"
OUT_REPORT = REPORTS / "week8_v49b_exact_anchor_overlay_gallery_report.md"
OUT_NOTE = NOTES / "week8_v49b_exact_anchor_overlay_gallery_notes.md"
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


def to_int(x, default=None):
    try:
        if pd.isna(x):
            return default
        return int(round(float(x)))
    except Exception:
        return default


def read_locked():
    return json.loads(Path(LOCKED_JSON).read_text())


def path_for(locked, key):
    return Path(locked["locked_inputs"][key]["path"])


def draw_anchor_overlay(scan, clip_path, anchor_frame_idx, anchor_rel_sec, objects, out_path):
    try:
        import cv2
    except Exception:
        return {
            "saved": False,
            "message": "cv2_not_available",
            "video_open_ok": False,
            "video_width": None,
            "video_height": None,
            "video_frame_count": None,
            "video_fps": None,
        }

    clip_path = Path(clip_path)
    cap = cv2.VideoCapture(str(clip_path))

    if not cap.isOpened():
        cap.release()
        return {
            "saved": False,
            "message": "video_open_failed",
            "video_open_ok": False,
            "video_width": None,
            "video_height": None,
            "video_frame_count": None,
            "video_fps": None,
        }

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_count <= 0:
        frame_idx = 0
    else:
        frame_idx = max(0, min(frame_count - 1, int(anchor_frame_idx)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return {
            "saved": False,
            "message": "frame_read_failed",
            "video_open_ok": True,
            "video_width": width,
            "video_height": height,
            "video_frame_count": frame_count,
            "video_fps": fps,
        }

    colours = {
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

    for obj in objects:
        x1 = to_float(obj.get("bbox_x1"))
        y1 = to_float(obj.get("bbox_y1"))
        x2 = to_float(obj.get("bbox_x2"))
        y2 = to_float(obj.get("bbox_y2"))

        if any(v is None for v in [x1, y1, x2, y2]):
            continue

        x1i = int(round(x1))
        y1i = int(round(y1))
        x2i = int(round(x2))
        y2i = int(round(y2))

        colour = clean_str(obj.get("visual_marker_colour")) or "unknown"
        bgr = colours.get(colour, (255, 255, 255))

        cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), bgr, 3)

        label = f"{clean_str(obj.get('behaviour_pig_id'))}/{colour}/{clean_str(obj.get('behaviour_code'))}"
        y_text = max(20, y1i - 8)
        cv2.putText(frame, label, (x1i, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr, 2, cv2.LINE_AA)

    header_1 = f"{scan} exact anchor frame"
    header_2 = f"clip_frame={frame_idx} anchor_t={anchor_rel_sec:.3f}s objects={len(objects)}"
    cv2.rectangle(frame, (0, 0), (width, 58), (0, 0, 0), -1)
    cv2.putText(frame, header_1, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, header_2, (12, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved = cv2.imwrite(str(out_path), frame)

    return {
        "saved": bool(saved),
        "message": "saved" if saved else "write_failed",
        "video_open_ok": True,
        "video_width": width,
        "video_height": height,
        "video_frame_count": frame_count,
        "video_fps": fps,
    }


issues = []

for p in [LOCKED_JSON, V45_CLIP_OBJECT_CSV]:
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
        "v49b_decision": "exact_anchor_overlay_gallery_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_visual_inspection": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


locked = read_locked()
clip_index_path = path_for(locked, "clip_extraction_index_72")
clip_index = pd.read_csv(clip_index_path)
clip_obj = pd.read_csv(V45_CLIP_OBJECT_CSV)

obj_by_scan = {
    str(k): v.copy()
    for k, v in clip_obj.groupby("scan_frame_id")
}

rows = []

for _, clip in clip_index.iterrows():
    scan = clean_str(clip.get("scan_frame_id"))
    clip_path = clean_str(clip.get("clip_path"))

    fps_used = to_float(clip.get("fps_used"), 25.0) or 25.0
    start_sec = to_float(clip.get("start_sec"), 0.0) or 0.0
    end_sec = to_float(clip.get("end_sec"), start_sec + 10.0) or (start_sec + 10.0)
    duration_sec = to_float(clip.get("duration_sec"), end_sec - start_sec) or (end_sec - start_sec)
    center_sec = to_float(clip.get("center_sec"), None)

    if center_sec is None:
        anchor_rel_sec = duration_sec / 2.0
        anchor_source = "fallback_mid_clip"
    else:
        anchor_rel_sec = center_sec - start_sec
        anchor_source = "v26_center_sec_minus_start_sec"

    anchor_rel_sec = max(0.0, min(duration_sec, anchor_rel_sec))
    anchor_frame_idx = int(round(anchor_rel_sec * fps_used))

    objs_df = obj_by_scan.get(scan, pd.DataFrame())
    objects = objs_df.to_dict("records")

    out_path = IMG_DIR / f"{scan}_exact_anchor_overlay.jpg"

    result = draw_anchor_overlay(
        scan=scan,
        clip_path=clip_path,
        anchor_frame_idx=anchor_frame_idx,
        anchor_rel_sec=anchor_rel_sec,
        objects=objects,
        out_path=out_path,
    )

    rows.append({
        "scan_frame_id": scan,
        "video_id": clean_str(clip.get("video_id")),
        "clip_path": clip_path,
        "overlay_path": str(out_path),
        "overlay_saved": result["saved"],
        "message": result["message"],
        "object_count": len(objects),
        "fps_used": fps_used,
        "video_fps": result["video_fps"],
        "start_sec": start_sec,
        "center_sec": center_sec,
        "end_sec": end_sec,
        "duration_sec": duration_sec,
        "anchor_rel_sec": anchor_rel_sec,
        "anchor_frame_idx": anchor_frame_idx,
        "anchor_source": anchor_source,
        "video_open_ok": result["video_open_ok"],
        "video_width": result["video_width"],
        "video_height": result["video_height"],
        "video_frame_count": result["video_frame_count"],
        "behaviour_set": clean_str(clip.get("behaviour_codes_present")),
    })

index_df = pd.DataFrame(rows)

missing_overlays = int((index_df["overlay_saved"] == False).sum())
saved_overlays = int((index_df["overlay_saved"] == True).sum())

if missing_overlays > 0:
    issues.append({
        "item": "anchor_overlay_generation",
        "issue_type": "warning_some_overlays_not_saved",
        "issue_detail": f"{missing_overlays} overlays were not saved.",
        "severity": "warning",
    })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
hard_issues = issues_df[issues_df["severity"] == "hard"] if len(issues_df) else pd.DataFrame()
warnings = issues_df[issues_df["severity"] == "warning"] if len(issues_df) else pd.DataFrame()

ready = len(hard_issues) == 0

html_parts = []
html_parts.append("<!doctype html>")
html_parts.append("<html><head><meta charset='utf-8'>")
html_parts.append("<title>Week 8 v49b Exact Anchor Overlay Gallery</title>")
html_parts.append("<style>")
html_parts.append("body{font-family:Arial;background:#111;color:#eee;margin:20px}")
html_parts.append(".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}")
html_parts.append(".card{background:#1e1e1e;border:1px solid #333;border-radius:10px;padding:12px}")
html_parts.append("img{width:100%;border:1px solid #444;border-radius:6px}")
html_parts.append(".meta{font-size:13px;color:#bbb;line-height:1.4}")
html_parts.append(".top{background:#2b2b2b;padding:12px;border-radius:10px;margin-bottom:18px}")
html_parts.append("</style></head><body>")
html_parts.append("<div class='top'>")
html_parts.append("<h1>Week 8 v49b Exact Anchor Overlay Gallery</h1>")
html_parts.append("<p>This gallery draws bbox annotations directly on the exact annotation anchor frame extracted from the clip video. It bypasses the browser canvas overlay.</p>")
html_parts.append("<p>Use this to decide whether the problem is browser overlay/canvas timing or the annotation boxes themselves.</p>")
html_parts.append(f"<p>Saved overlays: {saved_overlays} / {len(index_df)}</p>")
html_parts.append("</div>")
html_parts.append("<div class='grid'>")

for _, r in index_df.iterrows():
    if not bool(r["overlay_saved"]):
        continue
    rel_img = Path(r["overlay_path"]).relative_to(OUT)
    html_parts.append("<div class='card'>")
    html_parts.append(f"<h3>{r['scan_frame_id']}</h3>")
    html_parts.append(f"<img src='{rel_img.as_posix()}'>")
    html_parts.append("<div class='meta'>")
    html_parts.append(f"video: {r['video_id']}<br>")
    html_parts.append(f"objects: {r['object_count']}<br>")
    html_parts.append(f"anchor_rel_sec: {r['anchor_rel_sec']}<br>")
    html_parts.append(f"anchor_frame_idx: {r['anchor_frame_idx']}<br>")
    html_parts.append(f"behaviour_set: {r['behaviour_set']}<br>")
    html_parts.append("</div>")
    html_parts.append("</div>")

html_parts.append("</div></body></html>")
OUT_INDEX.write_text("\n".join(html_parts))

decision = pd.DataFrame([{
    "v49b_decision": "exact_anchor_overlay_gallery_created" if ready else "exact_anchor_overlay_gallery_blocked",
    "clip_count": int(len(index_df)),
    "saved_overlay_count": int(saved_overlays),
    "missing_overlay_count": int(missing_overlays),
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "issue_count": int(len(issues_df)),
    "gallery_index": str(OUT_INDEX),
    "ready_for_visual_inspection": bool(ready),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(index_df, OUT_OVERLAY_INDEX)
safe_to_csv(issues_df, OUT_ISSUES)
safe_to_csv(decision, OUT_DECISION)

report = f"""Week 8 v49b Exact Anchor Overlay Gallery Report

Decision:
v49b decision: {decision.iloc[0]["v49b_decision"]}
Clip count: {len(index_df)}
Saved overlays: {saved_overlays}
Missing overlays: {missing_overlays}
Hard issue count: {len(hard_issues)}
Warning count: {len(warnings)}

Purpose:
This stage bypasses the browser interface and draws the annotation boxes directly on the exact anchor frame read from each clip video.

Interpretation rule:
If these exact-anchor overlays look correct, the problem is likely browser/canvas overlay or full-clip motion drift.
If these exact-anchor overlays look wrong, the issue is in the annotation boxes, identity association, or scanframe-to-clip mapping.

Gallery:
{OUT_INDEX}

Overlay index:
{OUT_OVERLAY_INDEX}
"""
OUT_REPORT.write_text(report)

OUT_NOTE.write_text(
    "# Week 8 v49b Exact Anchor Overlay Gallery\n\n"
    "## Summary\n\n"
    f"- v49b decision: {decision.iloc[0]['v49b_decision']}\n"
    f"- Clip count: {len(index_df)}\n"
    f"- Saved overlays: {saved_overlays}\n"
    f"- Missing overlays: {missing_overlays}\n"
    f"- Hard issue count: {len(hard_issues)}\n"
    f"- Warning count: {len(warnings)}\n\n"
    "## What to inspect\n\n"
    "Open the gallery and check whether the boxes are correct on the exact anchor frame.\n\n"
    "If exact-anchor overlays are correct, continue with browser/interface or tracking-refined full-clip boxes.\n"
    "If exact-anchor overlays are wrong, repair annotation/source mapping before tracking.\n\n"
    f"Gallery: {OUT_INDEX}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v49b",
    "task_name": "Exact anchor-frame overlay gallery",
    "status": "PASS" if ready else "BLOCKED",
    "input_summary": str(V45_CLIP_OBJECT_CSV),
    "output_summary": str(OUT),
    "hard_issues": int(len(hard_issues)),
    "warnings": int(len(warnings)),
    "next_action": "Inspect exact-anchor overlays, then decide interface fix vs annotation mapping repair vs tracking-refined boxes.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_OVERLAY_INDEX)
print(OUT_INDEX)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_REPORT)
print(OUT_NOTE)

print()
print("=== v49b decision ===")
print(decision.to_string(index=False))

print()
print("=== v49b issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")

print()
print("=== v49b overlay index head ===")
print(index_df.head(20).to_string(index=False))
