from pathlib import Path
from datetime import datetime
import csv
import math
import re
import shutil
import subprocess

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V25_ROOT = W7 / "outputs" / "tracking_temporal_preparation_v25"
CLIP_PLAN = V25_ROOT / "week7_tracking_temporal_v25_10sec_clip_window_plan.csv"

OUT_ROOT = W7 / "outputs" / "clip_extraction_temporal_qa_v26"
CLIP_ROOT = OUT_ROOT / "clips"
PREVIEW_ROOT = OUT_ROOT / "preview_frames"
CONTACT_ROOT = OUT_ROOT / "contact_sheets"

for p in [OUT_ROOT, CLIP_ROOT, PREVIEW_ROOT, CONTACT_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

OUT_INDEX = OUT_ROOT / "week7_clip_extraction_v26_index.csv"
OUT_SUMMARY = OUT_ROOT / "week7_clip_extraction_v26_summary.csv"
OUT_ISSUES = OUT_ROOT / "week7_clip_extraction_v26_issues.csv"
OUT_CONTACT_INDEX = OUT_ROOT / "week7_clip_extraction_v26_contact_sheet_index.csv"
OUT_README = OUT_ROOT / "README_clip_extraction_temporal_qa_v26.md"
OUT_NOTE = W7 / "notes" / "week7_clip_extraction_temporal_qa_v26_notes.md"

TARGET_DURATION_SEC = 10.0
HALF_WINDOW_SEC = TARGET_DURATION_SEC / 2.0


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def clean(v):
    if pd.isna(v):
        return ""
    return str(v).strip()


def slug(s):
    s = clean(s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def load_font(size=12):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def command_exists(cmd):
    return shutil.which(cmd) is not None


def run_cmd(args):
    proc = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def ffprobe_duration(video_path):
    if not command_exists("ffprobe"):
        return None

    args = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    code, out, err = run_cmd(args)

    if code != 0:
        return None

    try:
        dur = float(out.strip())
        if dur > 0:
            return dur
    except Exception:
        return None

    return None


def extract_clip(video_path, start_sec, duration_sec, out_path):
    args = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{start_sec:.3f}",
        "-i", str(video_path),
        "-t", f"{duration_sec:.3f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-an",
        str(out_path),
    ]

    return run_cmd(args)


def extract_preview(clip_path, preview_sec, out_path):
    args = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{preview_sec:.3f}",
        "-i", str(clip_path),
        "-frames:v", "1",
        str(out_path),
    ]

    return run_cmd(args)


def file_size(path):
    p = Path(path)
    return p.stat().st_size if p.exists() else 0


def make_contact_sheet(rows, out_path, title, cols=4, thumb_w=280, thumb_h=180):
    if not rows:
        return False

    font = load_font(12)
    title_font = load_font(16)

    pad = 8
    title_h = 40
    cell_w = thumb_w + 2 * pad
    cell_h = thumb_h + 62 + 2 * pad
    sheet_rows = math.ceil(len(rows) / cols)

    sheet = Image.new("RGB", (cols * cell_w, title_h + sheet_rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)

    draw.text((pad, pad), title, fill=(0, 0, 0), font=title_font)

    for i, row in enumerate(rows):
        r = i // cols
        c = i % cols

        x0 = c * cell_w + pad
        y0 = title_h + r * cell_h + pad

        preview_path = Path(row["preview_frame_path"])

        try:
            im = Image.open(preview_path).convert("RGB")
            im.thumbnail((thumb_w, thumb_h))
            bg = Image.new("RGB", (thumb_w, thumb_h), "white")
            bg.paste(im, ((thumb_w - im.width) // 2, (thumb_h - im.height) // 2))
            sheet.paste(bg, (x0, y0))
        except Exception:
            bg = Image.new("RGB", (thumb_w, thumb_h), "lightgray")
            d = ImageDraw.Draw(bg)
            d.text((8, 8), "PREVIEW ERROR", fill=(0, 0, 0), font=font)
            sheet.paste(bg, (x0, y0))

        label = (
            f"{row.get('scan_frame_id', '')}\n"
            f"{row.get('video_id', '')}\n"
            f"{row.get('start_sec', ''):.1f}s → {row.get('end_sec', ''):.1f}s"
        )

        draw.text((x0, y0 + thumb_h + 4), label, fill=(0, 0, 0), font=font)

    sheet.save(out_path, quality=95)
    return True


issues = []
index_rows = []

if not command_exists("ffmpeg"):
    issues.append({
        "scan_frame_id": "",
        "issue_type": "ffmpeg_missing",
        "issue_detail": "ffmpeg command not found in environment.",
    })

plan = pd.read_csv(CLIP_PLAN)

for c in [
    "scan_frame_id",
    "video_id",
    "timestamp",
    "candidate_video_path",
    "behaviour_codes_present",
    "clip_plan_status",
]:
    if c in plan.columns:
        plan[c] = plan[c].fillna("").astype(str).str.strip()

# Only planned clips.
planned = plan[plan["clip_plan_status"].astype(str).eq("planned")].copy()

for _, row in planned.iterrows():
    scan_frame_id = clean(row.get("scan_frame_id", ""))
    video_id = clean(row.get("video_id", ""))
    timestamp = clean(row.get("timestamp", ""))
    video_path = Path(clean(row.get("candidate_video_path", "")))

    fps = pd.to_numeric(pd.Series([row.get("fps_used", "")]), errors="coerce").iloc[0]
    center_frame = pd.to_numeric(pd.Series([row.get("center_frame_index", "")]), errors="coerce").iloc[0]

    if pd.isna(fps) or fps <= 0:
        fps = 25.0

    if pd.isna(center_frame):
        issues.append({
            "scan_frame_id": scan_frame_id,
            "issue_type": "missing_center_frame",
            "issue_detail": "Cannot compute temporal clip window.",
        })
        continue

    if not video_path.exists():
        issues.append({
            "scan_frame_id": scan_frame_id,
            "issue_type": "video_file_not_found",
            "issue_detail": str(video_path),
        })
        continue

    if not command_exists("ffmpeg"):
        continue

    center_sec = float(center_frame) / float(fps)

    video_duration = ffprobe_duration(video_path)

    # Professional adjustment:
    # Keep a 10-second clip whenever possible. For start-edge frames, shift right.
    start_sec = max(0.0, center_sec - HALF_WINDOW_SEC)
    duration_sec = TARGET_DURATION_SEC

    edge_adjustment = "none"

    if video_duration is not None:
        if video_duration < TARGET_DURATION_SEC:
            start_sec = 0.0
            duration_sec = video_duration
            edge_adjustment = "video_shorter_than_10sec"
        elif start_sec + TARGET_DURATION_SEC > video_duration:
            start_sec = max(0.0, video_duration - TARGET_DURATION_SEC)
            duration_sec = TARGET_DURATION_SEC
            edge_adjustment = "shifted_left_near_video_end"
        elif center_sec < HALF_WINDOW_SEC:
            start_sec = 0.0
            duration_sec = TARGET_DURATION_SEC
            edge_adjustment = "shifted_right_near_video_start"
    else:
        if center_sec < HALF_WINDOW_SEC:
            start_sec = 0.0
            edge_adjustment = "shifted_right_near_video_start_no_duration_probe"

    end_sec = start_sec + duration_sec

    start_frame_adjusted = int(round(start_sec * float(fps)))
    end_frame_adjusted = int(round(end_sec * float(fps)))

    out_name = (
        f"{slug(scan_frame_id)}__"
        f"{slug(video_id)}__"
        f"centerf_{int(round(float(center_frame))):06d}__"
        f"{int(round(start_sec * 1000)):08d}ms_{int(round(end_sec * 1000)):08d}ms.mp4"
    )

    clip_path = CLIP_ROOT / out_name
    preview_path = PREVIEW_ROOT / out_name.replace(".mp4", ".jpg")

    code, stdout, stderr = extract_clip(video_path, start_sec, duration_sec, clip_path)

    clip_ok = bool(code == 0 and clip_path.exists() and file_size(clip_path) > 0)

    if not clip_ok:
        issues.append({
            "scan_frame_id": scan_frame_id,
            "issue_type": "clip_extraction_failed",
            "issue_detail": f"{video_path} stderr={stderr[:500]}",
        })
        continue

    preview_sec = min(duration_sec / 2.0, max(0.0, duration_sec - 0.1))
    p_code, p_stdout, p_stderr = extract_preview(clip_path, preview_sec, preview_path)

    preview_ok = bool(p_code == 0 and preview_path.exists() and file_size(preview_path) > 0)

    if not preview_ok:
        issues.append({
            "scan_frame_id": scan_frame_id,
            "issue_type": "preview_extraction_failed",
            "issue_detail": f"{clip_path} stderr={p_stderr[:500]}",
        })

    index_rows.append({
        "scan_frame_id": scan_frame_id,
        "video_id": video_id,
        "timestamp": timestamp,
        "source_video_path": str(video_path),
        "clip_path": str(clip_path),
        "preview_frame_path": str(preview_path) if preview_ok else "",
        "fps_used": float(fps),
        "center_frame_index": int(round(float(center_frame))),
        "center_sec": round(center_sec, 4),
        "start_sec": round(start_sec, 4),
        "end_sec": round(end_sec, 4),
        "duration_sec": round(duration_sec, 4),
        "start_frame_adjusted": start_frame_adjusted,
        "end_frame_adjusted": end_frame_adjusted,
        "video_duration_sec": round(video_duration, 4) if video_duration is not None else "",
        "edge_adjustment": edge_adjustment,
        "behaviour_codes_present": clean(row.get("behaviour_codes_present", "")),
        "pig_rows": clean(row.get("pig_rows", "")),
        "clip_file_size_bytes": file_size(clip_path),
        "preview_ok": preview_ok,
        "clip_status": "extracted",
    })

index_df = pd.DataFrame(index_rows)
issues_df = pd.DataFrame(issues, columns=["scan_frame_id", "issue_type", "issue_detail"])

safe_to_csv(index_df, OUT_INDEX)
safe_to_csv(issues_df, OUT_ISSUES)

# Contact sheets by scanframe order.
contact_rows = []

if len(index_df):
    index_df["scan_num"] = index_df["scan_frame_id"].apply(
        lambda x: int(re.search(r"(\d+)$", str(x)).group(1)) if re.search(r"(\d+)$", str(x)) else 10**9
    )
    sorted_index = index_df.sort_values("scan_num").drop(columns=["scan_num"])

    preview_rows = sorted_index[sorted_index["preview_ok"] == True].to_dict("records")

    for idx in range(0, len(preview_rows), 24):
        chunk = preview_rows[idx:idx + 24]
        out = CONTACT_ROOT / f"clip_previews_part_{idx // 24 + 1:02d}.jpg"
        made = make_contact_sheet(chunk, out, f"v26 10-second clip previews | part {idx // 24 + 1}")

        if made:
            contact_rows.append({
                "sheet_type": "clip_previews",
                "sheet_path": str(out),
                "preview_count": len(chunk),
            })

contact_index = pd.DataFrame(contact_rows)
safe_to_csv(contact_index, OUT_CONTACT_INDEX)

summary = pd.DataFrame([{
    "source_clip_plan": str(CLIP_PLAN),
    "planned_rows": int(len(planned)),
    "clips_extracted": int(len(index_df)),
    "preview_frames_extracted": int(index_df["preview_ok"].sum()) if len(index_df) else 0,
    "contact_sheets_created": int(len(contact_index)),
    "issue_count": int(len(issues_df)),
    "clip_root": str(CLIP_ROOT),
    "preview_root": str(PREVIEW_ROOT),
    "contact_sheet_root": str(CONTACT_ROOT),
    "edge_adjusted_clips": int((index_df["edge_adjustment"] != "none").sum()) if len(index_df) else 0,
    "ready_for_v27_tracking_dry_run": bool(len(index_df) == len(planned) and len(issues_df) == 0),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(summary, OUT_SUMMARY)

ready = bool(summary.iloc[0]["ready_for_v27_tracking_dry_run"])

OUT_README.write_text(
    "# Week 7 Clip Extraction + Temporal QA v26\n\n"
    "## Purpose\n\n"
    "This step extracts 10-second scanpoint-centered clips from raw videos using the v25 temporal plan. "
    "It also creates preview frames and contact sheets for quick visual QA before detector/tracker dry-run.\n\n"
    "## Window policy\n\n"
    "- Target duration is 10 seconds.\n"
    "- For scanpoints near the video start, the window is shifted right to preserve 10 seconds when possible.\n"
    "- For scanpoints near the video end, the window is shifted left to preserve 10 seconds when possible.\n\n"
    "## Outputs\n\n"
    "- `clips/`\n"
    "- `preview_frames/`\n"
    "- `contact_sheets/`\n"
    "- `week7_clip_extraction_v26_index.csv`\n"
    "- `week7_clip_extraction_v26_summary.csv`\n"
    "- `week7_clip_extraction_v26_issues.csv`\n"
)

OUT_NOTE.write_text(
    "# Week 7 Clip Extraction + Temporal QA v26\n\n"
    "## Purpose\n\n"
    "This step extracts 10-second clips around each scanpoint and creates previews/contact sheets before tracking.\n\n"
    "## Summary\n\n"
    f"- Planned rows: `{int(summary.iloc[0]['planned_rows'])}`\n"
    f"- Clips extracted: `{int(summary.iloc[0]['clips_extracted'])}`\n"
    f"- Preview frames extracted: `{int(summary.iloc[0]['preview_frames_extracted'])}`\n"
    f"- Contact sheets created: `{int(summary.iloc[0]['contact_sheets_created'])}`\n"
    f"- Edge-adjusted clips: `{int(summary.iloc[0]['edge_adjusted_clips'])}`\n"
    f"- Issue count: `{int(summary.iloc[0]['issue_count'])}`\n"
    f"- Ready for v27 tracking dry-run: `{ready}`\n\n"
    "## Outputs\n\n"
    f"- Summary: `{OUT_SUMMARY}`\n"
    f"- Index: `{OUT_INDEX}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
    f"- Clips: `{CLIP_ROOT}`\n"
    f"- Preview frames: `{PREVIEW_ROOT}`\n"
    f"- Contact sheets: `{CONTACT_ROOT}`\n"
)

print("Saved:")
print(CLIP_ROOT)
print(PREVIEW_ROOT)
print(CONTACT_ROOT)
print(OUT_INDEX)
print(OUT_SUMMARY)
print(OUT_ISSUES)
print(OUT_CONTACT_INDEX)
print(OUT_README)
print(OUT_NOTE)

print()
print("=== v26 clip extraction summary ===")
print(summary.to_string(index=False))

print()
print("=== v26 issues ===")
if len(issues_df):
    print(issues_df.head(50).to_string(index=False))
else:
    print("No issues found.")
