from pathlib import Path
from datetime import datetime
import re
import csv
import json
import hashlib
import zipfile
import shutil
import subprocess
import pandas as pd

ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"
RAW_UNIBO = Path("/work/pig/datasets/Unibo")

OUT = FULL / "outputs" / "v78k7_overlay_based_video_identity_mapping"
PKG = OUT / "Full_Unibo_Overlay_Based_Video_Identity_Mapping"
CROPS = PKG / "overlay_crops"
NOTES = FULL / "notes"
REPORTS = FULL / "reports"
PROGRESS = FULL / "progress"

for p in [OUT, PKG, CROPS, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_VIDEO_TABLE = PKG / "v78k7_video_overlay_identity_table.csv"
OUT_CCODE_MAP = PKG / "v78k7_c_code_to_overlay_target_mapping.csv"
OUT_TARGET_MAP = PKG / "v78k7_overlay_target_to_c_code_mapping.csv"
OUT_CONFLICTS = PKG / "v78k7_overlay_mapping_conflicts.csv"
OUT_MANUAL = PKG / "v78k7_manual_overlay_review_template.csv"
OUT_HTML = PKG / "v78k7_overlay_crop_review_board.html"
OUT_DECISION = OUT / "v78k7_decision_summary.csv"
OUT_ISSUES = OUT / "v78k7_issues.csv"
OUT_README = PKG / "README_v78k7_Overlay_Based_Video_Identity_Mapping.md"
OUT_MANIFEST = PKG / "v78k7_manifest.json"
OUT_ZIP = OUT / "Full_Unibo_Overlay_Based_Video_Identity_Mapping.zip"
OUT_SHA = OUT / "Full_Unibo_Overlay_Based_Video_Identity_Mapping.sha256"
OUT_NOTE = NOTES / "v78k7_overlay_based_video_identity_mapping_notes.md"
OUT_REPORT = REPORTS / "v78k7_overlay_based_video_identity_mapping_report.md"
OUT_PROGRESS = PROGRESS / "full_unibo_pipeline_progress_log.csv"

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
ENCODED_RE = re.compile(r"(c\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", re.I)

def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s

def to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_date_anywhere(text):
    s = str(text)

    m = re.search(r"(20\d{2})[-_](\d{1,2})[-_](\d{1,2})", s)
    if m:
        y, mo, d = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"

    m = re.search(r"\b(\d{1,2})[-_](\d{1,2})[-_](20\d{2})\b", s)
    if m:
        d, mo, y = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"

    return ""

def parse_filename(path):
    name = path.name
    full = str(path)

    video_type = "unknown"
    c_code = ""
    date = ""
    start_time = ""
    start_hour = ""

    m = ENCODED_RE.search(name)
    if m:
        video_type = "encoded_c_code"
        c_code = m.group(1).lower()
        yy, mo, dd, hh, mm, ss = m.group(2), m.group(3), m.group(4), m.group(5), m.group(6), m.group(7)
        date = f"20{int(yy):02d}-{int(mo):02d}-{int(dd):02d}"
        start_time = f"{int(hh):02d}:{int(mm):02d}:{int(ss):02d}"
        start_hour = f"{int(hh):02d}:00"

    mt = re.search(r"tlc[\s_ -]*([1-6])", full, re.I)
    filename_tlc = f"TLC{mt.group(1)}" if mt else ""

    mp = re.search(r"\b([BCM])[\s_ -]*([0-9]+)\b", full, re.I)
    filename_pen = f"{mp.group(1).upper()}{int(mp.group(2))}" if mp else ""

    if video_type == "unknown" and filename_tlc:
        video_type = "friendly_tlc"

    if not date:
        date = parse_date_anywhere(full)

    if not start_hour:
        mh = re.search(r"\b([01]?\d|2[0-3])[:._-]?([0-5]\d)[:._-]?([0-5]\d)?\b", name)
        if mh:
            hh = int(mh.group(1))
            mm = int(mh.group(2))
            ss = int(mh.group(3) or 0)
            start_time = f"{hh:02d}:{mm:02d}:{ss:02d}"
            start_hour = f"{hh:02d}:00"

    return {
        "video_filename": name,
        "video_path": str(path),
        "video_parent": str(path.parent),
        "video_type": video_type,
        "filename_c_code": c_code,
        "filename_date": date,
        "filename_start_time": start_time,
        "filename_start_hour": start_hour,
        "filename_tlc_camera": filename_tlc,
        "filename_room_pen": filename_pen,
        "file_size_bytes": path.stat().st_size if path.exists() else 0,
    }

def extract_crop_cv2(video_path, out_path):
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, "cv2_open_failed"

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

        candidates = [int(fps * 2), int(fps * 8), 0]
        if total > 0:
            candidates.append(min(total - 1, int(total * 0.1)))

        frame = None
        for idx in candidates:
            if idx < 0:
                idx = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            if ok and fr is not None:
                frame = fr
                break

        cap.release()

        if frame is None:
            return False, "cv2_read_failed"

        h, w = frame.shape[:2]
        # Büyük crop alıyoruz çünkü overlay bazı videolarda birkaç satır olabilir.
        crop = frame[0:int(h * 0.22), 0:int(w * 0.45)]

        # OCR ve gözle kontrol için büyüt.
        scale = 2
        crop_big = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale))
        cv2.imwrite(str(out_path), crop_big)
        return True, ""
    except Exception as e:
        return False, f"cv2_error:{e}"

def extract_crop_ffmpeg(video_path, out_path):
    if not shutil.which("ffmpeg"):
        return False, "ffmpeg_not_available"
    cmd = [
        "ffmpeg", "-y",
        "-ss", "2",
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", "crop=iw*0.45:ih*0.22:0:0,scale=iw*2:ih*2",
        str(out_path),
    ]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        if r.returncode == 0 and out_path.exists():
            return True, ""
        return False, "ffmpeg_failed"
    except Exception as e:
        return False, f"ffmpeg_error:{e}"

def ocr_crop(path):
    # Optional OCR. If not available, leave blank and use visual review board.
    if not shutil.which("tesseract"):
        return "", "tesseract_not_available"

    try:
        import pytesseract
        from PIL import Image, ImageOps, ImageFilter
        img = Image.open(path).convert("L")
        img = ImageOps.autocontrast(img)
        img = img.filter(ImageFilter.SHARPEN)
        txt = pytesseract.image_to_string(img, config="--psm 6")
        return clean(txt.replace("\n", " ")), ""
    except Exception as e:
        return "", f"ocr_error:{e}"

def parse_overlay_text(text):
    t = clean(text)
    if not t:
        return "", "", "no_ocr_text"

    # Normalize common OCR confusions lightly.
    n = t.upper()
    n = n.replace("TLC.", "TLC ")
    n = n.replace("TLC:", "TLC ")
    n = n.replace("TL C", "TLC")
    n = re.sub(r"\s+", " ", n)

    patterns = [
        r"\bTLC\s*([1-6])\s*([BCM])\s*([0-9]{1,2})\b",
        r"\bTLC\s*([1-6]).{0,20}?\b([BCM])\s*([0-9]{1,2})\b",
        r"\bTLC\s*([1-6])\s*(BIG|SMALL)\b",
    ]

    for pat in patterns:
        m = re.search(pat, n, re.I)
        if m:
            tlc = f"TLC{m.group(1)}"
            if len(m.groups()) >= 3 and m.group(2).upper() in {"B", "C", "M"}:
                pen = f"{m.group(2).upper()}{int(m.group(3))}"
            else:
                pen = m.group(2).lower()
            return tlc, pen, "parsed_from_ocr"

    return "", "", "ocr_text_unparsed"

def html_img(path):
    if not path:
        return ""
    return Path(path).name

issues = []

if not RAW_UNIBO.exists():
    issues.append({
        "item": str(RAW_UNIBO),
        "issue_type": "hard_missing_raw_unibo_dir",
        "issue_detail": "Raw Unibo directory does not exist.",
        "severity": "hard",
    })

videos = []
if RAW_UNIBO.exists():
    for p in sorted(RAW_UNIBO.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            videos.append(p)

rows = []

for i, vp in enumerate(videos):
    base = parse_filename(vp)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", vp.stem)
    crop_path = CROPS / f"{i:04d}_{safe}_overlay_crop.jpg"

    ok, err = extract_crop_cv2(vp, crop_path)
    if not ok:
        ok, err2 = extract_crop_ffmpeg(vp, crop_path)
        if not ok:
            err = err + ";" + err2

    ocr_text = ""
    ocr_issue = ""
    overlay_tlc = ""
    overlay_pen = ""
    overlay_parse_status = "crop_not_available"

    if ok and crop_path.exists():
        ocr_text, ocr_issue = ocr_crop(crop_path)
        overlay_tlc, overlay_pen, overlay_parse_status = parse_overlay_text(ocr_text)

    final_tlc = overlay_tlc or base["filename_tlc_camera"]
    final_pen = overlay_pen or base["filename_room_pen"]

    if overlay_tlc and overlay_pen:
        final_source = "overlay_ocr"
        review_status = "auto_parsed_needs_visual_confirmation"
    elif base["filename_tlc_camera"] and base["filename_room_pen"]:
        final_source = "filename"
        review_status = "filename_parsed_needs_visual_confirmation"
    else:
        final_source = ""
        review_status = "needs_manual_overlay_review"

    rows.append({
        **base,
        "overlay_crop_path": str(crop_path) if crop_path.exists() else "",
        "crop_extraction_ok": bool(ok),
        "crop_extraction_issue": err,
        "ocr_text": ocr_text,
        "ocr_issue": ocr_issue,
        "overlay_tlc_camera": overlay_tlc,
        "overlay_room_pen": overlay_pen,
        "overlay_parse_status": overlay_parse_status,
        "final_tlc_camera": final_tlc,
        "final_room_pen": final_pen,
        "final_identity_source": final_source,
        "review_status": review_status,
    })

video_table = pd.DataFrame(rows)
to_csv(video_table, OUT_VIDEO_TABLE)

if video_table.empty:
    issues.append({
        "item": "video_table",
        "issue_type": "hard_empty_video_table",
        "issue_detail": "No videos were processed.",
        "severity": "hard",
    })

if not video_table.empty:
    failed_crops = int((video_table["crop_extraction_ok"] == False).sum())
    if failed_crops:
        issues.append({
            "item": "crop_extraction",
            "issue_type": "warning_some_overlay_crops_failed",
            "issue_detail": f"{failed_crops} videos failed overlay crop extraction.",
            "severity": "warning",
        })

    need_manual = int((video_table["review_status"] == "needs_manual_overlay_review").sum())
    if need_manual:
        issues.append({
            "item": "overlay_parse",
            "issue_type": "warning_some_videos_need_manual_overlay_review",
            "issue_detail": f"{need_manual} videos need manual overlay review.",
            "severity": "warning",
        })

# Manual review template.
manual_cols = [
    "video_filename", "video_path", "video_type", "filename_c_code", "filename_date",
    "filename_start_time", "filename_start_hour", "overlay_crop_path", "ocr_text",
    "overlay_tlc_camera", "overlay_room_pen", "final_tlc_camera", "final_room_pen",
    "manual_tlc_camera", "manual_room_pen", "manual_review_status", "manual_note"
]

manual = video_table.copy() if not video_table.empty else pd.DataFrame(columns=manual_cols)
manual["manual_tlc_camera"] = manual["final_tlc_camera"] if "final_tlc_camera" in manual.columns else ""
manual["manual_room_pen"] = manual["final_room_pen"] if "final_room_pen" in manual.columns else ""
manual["manual_review_status"] = manual["review_status"] if "review_status" in manual.columns else ""
manual["manual_note"] = ""
manual = manual[[c for c in manual_cols if c in manual.columns]]
to_csv(manual, OUT_MANUAL)

# Mapping tables.
if not video_table.empty:
    usable = video_table[
        (video_table["filename_c_code"] != "") &
        (video_table["final_tlc_camera"] != "") &
        (video_table["final_room_pen"] != "")
    ].copy()

    ccode_map = (
        usable.groupby(["filename_date", "filename_c_code", "final_tlc_camera", "final_room_pen"], dropna=False)
        .agg(
            video_count=("video_path", "count"),
            hours=("filename_start_hour", lambda x: ";".join(sorted(set([clean(v) for v in x if clean(v)])))),
            filenames=("video_filename", lambda x: ";".join(list(x)[:30])),
            identity_sources=("final_identity_source", lambda x: ";".join(sorted(set([clean(v) for v in x if clean(v)])))),
        )
        .reset_index()
        .sort_values(["filename_date", "filename_c_code", "final_tlc_camera", "final_room_pen"])
    )

    target_map = (
        usable.groupby(["filename_date", "final_tlc_camera", "final_room_pen"], dropna=False)
        .agg(
            c_codes=("filename_c_code", lambda x: ";".join(sorted(set([clean(v) for v in x if clean(v)])))),
            video_count=("video_path", "count"),
            hours=("filename_start_hour", lambda x: ";".join(sorted(set([clean(v) for v in x if clean(v)])))),
            filenames=("video_filename", lambda x: ";".join(list(x)[:30])),
            identity_sources=("final_identity_source", lambda x: ";".join(sorted(set([clean(v) for v in x if clean(v)])))),
        )
        .reset_index()
        .sort_values(["filename_date", "final_tlc_camera", "final_room_pen"])
    )
else:
    usable = pd.DataFrame()
    ccode_map = pd.DataFrame()
    target_map = pd.DataFrame()

to_csv(ccode_map, OUT_CCODE_MAP)
to_csv(target_map, OUT_TARGET_MAP)

# Conflict checks:
# 1) same date+c_code maps to multiple TLC/pen targets
# 2) same date+TLC/pen maps to multiple c-codes
conflict_rows = []

if not ccode_map.empty:
    g1 = ccode_map.groupby(["filename_date", "filename_c_code"])
    for (date, code), g in g1:
        targets = sorted(set([f"{r.final_tlc_camera}/{r.final_room_pen}" for _, r in g.iterrows()]))
        if len(targets) > 1:
            conflict_rows.append({
                "conflict_type": "same_date_c_code_multiple_targets",
                "date": date,
                "key": code,
                "values": ";".join(targets),
                "severity": "hard",
            })

if not target_map.empty:
    for _, r in target_map.iterrows():
        codes = [c for c in clean(r["c_codes"]).split(";") if c]
        if len(codes) > 1:
            conflict_rows.append({
                "conflict_type": "same_date_target_multiple_c_codes",
                "date": clean(r["filename_date"]),
                "key": f"{clean(r['final_tlc_camera'])}/{clean(r['final_room_pen'])}",
                "values": ";".join(codes),
                "severity": "warning",
            })

conflicts = pd.DataFrame(conflict_rows, columns=["conflict_type", "date", "key", "values", "severity"])
to_csv(conflicts, OUT_CONFLICTS)

if not conflicts.empty:
    hard_conflicts = int((conflicts["severity"] == "hard").sum())
    warning_conflicts = int((conflicts["severity"] == "warning").sum())

    if hard_conflicts:
        issues.append({
            "item": "overlay_mapping_conflicts",
            "issue_type": "hard_same_c_code_maps_to_multiple_targets",
            "issue_detail": f"{hard_conflicts} hard overlay mapping conflicts found.",
            "severity": "hard",
        })
    if warning_conflicts:
        issues.append({
            "item": "overlay_mapping_conflicts",
            "issue_type": "warning_target_maps_to_multiple_c_codes",
            "issue_detail": f"{warning_conflicts} target-to-multiple-c-code warnings found.",
            "severity": "warning",
        })

# HTML review board.
html = []
html.append("""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>v78k7 Overlay Crop Review Board</title>
<style>
body { font-family: Arial, sans-serif; background:#f4f4f4; margin:0; }
.header { background:#111; color:white; padding:18px 24px; position:sticky; top:0; z-index:10; }
.container { padding:22px; }
.section { background:white; padding:16px; margin-bottom:20px; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,.08); }
.grid { display:flex; flex-wrap:wrap; gap:12px; }
.card { width:31%; min-width:300px; background:#fafafa; border:1px solid #ccc; border-radius:8px; padding:8px; }
.card img { width:100%; border-radius:4px; margin-top:6px; border:1px solid #999; }
.ok { border:4px solid #2e8b57; }
.review { border:4px solid #c77d00; }
.bad { border:4px solid #a00000; }
small { color:#555; word-break:break-all; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th, td { border:1px solid #ccc; padding:5px; vertical-align:top; }
th { background:#eee; }
</style>
</head>
<body>
<div class="header">
<h1>v78k7 Overlay-Based Video Identity Mapping</h1>
<p>Use top-left video overlay as video identity evidence.</p>
</div>
<div class="container">
""")

html.append("<div class='section'><h2>C-code mapping</h2>")
html.append(ccode_map.to_html(index=False, escape=True) if not ccode_map.empty else "<p>No c-code mapping rows.</p>")
html.append("</div>")

html.append("<div class='section'><h2>Conflicts</h2>")
html.append(conflicts.to_html(index=False, escape=True) if not conflicts.empty else "<p>No conflicts detected.</p>")
html.append("</div>")

html.append("<div class='section'><h2>Overlay crops</h2><div class='grid'>")
for _, r in video_table.iterrows() if not video_table.empty else []:
    status = clean(r["review_status"])
    cls = "ok" if "auto" in status or "filename" in status else "review"
    crop = Path(clean(r["overlay_crop_path"]))
    img = f"overlay_crops/{crop.name}" if crop.exists() else ""
    title = f"{clean(r['video_type'])} | {clean(r['filename_c_code'])} | {clean(r['filename_date'])} {clean(r['filename_start_time'])}"
    parsed = f"{clean(r['final_tlc_camera'])} / {clean(r['final_room_pen'])}"
    html.append(f"<div class='card {cls}'><b>{title}</b><br><b>Parsed:</b> {parsed}<br>")
    html.append(f"<small>{clean(r['video_filename'])}</small><br>")
    html.append(f"<small>OCR: {clean(r['ocr_text'])}</small>")
    if img:
        html.append(f"<img src='{img}'>")
    html.append("</div>")
html.append("</div></div>")

html.append("</div></body></html>")
OUT_HTML.write_text("\n".join(html), encoding="utf-8")

issues.append({
    "item": "scope",
    "issue_type": "info_overlay_video_mapping_only",
    "issue_detail": "v78k7 maps video identity from top-left overlay and filename. It does not run tracking or create final behaviour clips.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
to_csv(issues_df, OUT_ISSUES)

hard_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

mapped_video_rows = int(((video_table["final_tlc_camera"] != "") & (video_table["final_room_pen"] != "")).sum()) if not video_table.empty else 0
encoded_rows = int((video_table["video_type"] == "encoded_c_code").sum()) if not video_table.empty else 0
encoded_mapped_rows = int(((video_table["video_type"] == "encoded_c_code") & (video_table["final_tlc_camera"] != "") & (video_table["final_room_pen"] != "")).sum()) if not video_table.empty else 0

readme = """# v78k7 Overlay-Based Video Identity Mapping

This stage uses the visible top-left video overlay as evidence for video identity.

Evidence hierarchy:
1. Overlay text such as `TLC 2 B6`
2. Filename c-code/date/time encoding
3. Manual visual confirmation from overlay crop board

This stage supersedes attempts to infer TLC-to-c-code mapping without reading the video overlay.
"""
OUT_README.write_text(readme, encoding="utf-8")
OUT_REPORT.write_text(readme, encoding="utf-8")

manifest = {
    "version": "v78k7_overlay_based_video_identity_mapping",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "videos_found": len(videos),
    "video_table_rows": int(len(video_table)),
    "encoded_video_rows": encoded_rows,
    "mapped_video_rows": mapped_video_rows,
    "encoded_mapped_rows": encoded_mapped_rows,
    "ccode_mapping_rows": int(len(ccode_map)),
    "target_mapping_rows": int(len(target_map)),
    "conflict_rows": int(len(conflicts)),
    "hard_issue_count": hard_count,
    "warning_count": warning_count,
    "claim_boundary": "overlay-based video identity mapping only",
}
OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

zip_hash = sha256_file(OUT_ZIP)
OUT_SHA.write_text(f"{zip_hash}  {OUT_ZIP.name}\n")

decision = pd.DataFrame([{
    "v78k7_decision": "overlay_video_identity_mapping_created" if hard_count == 0 else "overlay_video_identity_mapping_has_blocking_issues",
    "videos_found": len(videos),
    "video_table_rows": int(len(video_table)),
    "encoded_video_rows": encoded_rows,
    "mapped_video_rows": mapped_video_rows,
    "encoded_mapped_rows": encoded_mapped_rows,
    "ccode_mapping_rows": int(len(ccode_map)),
    "target_mapping_rows": int(len(target_map)),
    "conflict_rows": int(len(conflicts)),
    "manual_review_template": str(OUT_MANUAL),
    "review_board_html": str(OUT_HTML),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_issue_count": hard_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "ready_for_manual_overlay_review": bool(hard_count == 0 and len(video_table) > 0),
    "ready_for_video_mapping_use": bool(hard_count == 0 and encoded_mapped_rows == encoded_rows and encoded_rows > 0),
    "ready_for_tracking": False,
    "claim_scope": "overlay_based_video_identity_mapping_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# v78k7 Overlay-Based Video Identity Mapping\n\n"
    f"- Decision: {decision.iloc[0]['v78k7_decision']}\n"
    f"- Videos found: {len(videos)}\n"
    f"- Encoded video rows: {encoded_rows}\n"
    f"- Mapped video rows: {mapped_video_rows}\n"
    f"- Encoded mapped rows: {encoded_mapped_rows}\n"
    f"- C-code mapping rows: {len(ccode_map)}\n"
    f"- Target mapping rows: {len(target_map)}\n"
    f"- Conflict rows: {len(conflicts)}\n"
    f"- Hard issues: {hard_count}\n"
    f"- Ready for video mapping use: {bool(hard_count == 0 and encoded_mapped_rows == encoded_rows and encoded_rows > 0)}\n\n"
    "Core correction: video identity should be read from top-left overlay text, not inferred indirectly.\n",
    encoding="utf-8"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v78k7",
    "task_name": "Overlay-based video identity mapping",
    "status": "PASS" if hard_count == 0 else "NEEDS_FIX",
    "input_summary": "raw Unibo videos",
    "output_summary": str(PKG),
    "hard_issues": hard_count,
    "warnings": warning_count,
    "next_action": "Use overlay crop board/manual template to confirm or correct video identity mappings.",
}])

prog_path = PROGRESS / "full_unibo_pipeline_progress_log.csv"
if prog_path.exists():
    old = pd.read_csv(prog_path)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row
to_csv(progress, prog_path)

print("=== v78k7 decision ===")
print(decision.to_string(index=False))

print("\n=== video identity summary ===")
if not video_table.empty:
    print(video_table.groupby(["video_type", "review_status"]).size().reset_index(name="count").to_string(index=False))
else:
    print("none")

print("\n=== c-code mapping ===")
print(ccode_map.to_string(index=False) if not ccode_map.empty else "none")

print("\n=== target mapping ===")
print(target_map.to_string(index=False) if not target_map.empty else "none")

print("\n=== conflicts ===")
print(conflicts.to_string(index=False) if not conflicts.empty else "none")

print("\n=== manual review sample ===")
print(manual.head(40).to_string(index=False) if not manual.empty else "none")

print("\n=== issues ===")
print(issues_df.to_string(index=False))
