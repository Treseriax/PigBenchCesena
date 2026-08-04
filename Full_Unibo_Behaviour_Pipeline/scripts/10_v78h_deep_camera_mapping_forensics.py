from pathlib import Path
from datetime import datetime
import re
import csv
import json
import hashlib
import zipfile
import subprocess
import shutil

import pandas as pd

try:
    import cv2
except Exception:
    cv2 = None

try:
    import openpyxl
except Exception:
    openpyxl = None


ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"
UNIBO = Path("/work/pig/datasets/Unibo")
DATASETS = Path("/work/pig/datasets")

OUT = FULL / "outputs" / "v78h_deep_camera_mapping_forensics"
PKG = OUT / "Full_Unibo_Deep_Camera_Mapping_Forensics"
STATIC = PKG / "visual_board"
NOTES = FULL / "notes"
REPORTS = FULL / "reports"
PROGRESS = FULL / "progress"

for p in [OUT, PKG, STATIC, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_VIDEO_INV = PKG / "v78h_video_inventory.csv"
OUT_MP4_META = PKG / "v78h_mp4_metadata_hits.csv"
OUT_MP4_STRINGS = PKG / "v78h_mp4_header_string_hits.csv"
OUT_EXCEL_DEEP = PKG / "v78h_excel_deep_hits.csv"
OUT_FILE_HITS = PKG / "v78h_readable_file_keyword_hits.csv"
OUT_VISUAL_INDEX = PKG / "v78h_visual_camera_board.html"
OUT_QA = PKG / "v78h_quality_checks.csv"
OUT_README = PKG / "README_v78h_Deep_Camera_Mapping_Forensics.md"
OUT_MANIFEST = PKG / "v78h_manifest.json"

OUT_DECISION = OUT / "v78h_decision_summary.csv"
OUT_ISSUES = OUT / "v78h_issues.csv"
OUT_ZIP = OUT / "Full_Unibo_Deep_Camera_Mapping_Forensics.zip"
OUT_SHA = OUT / "Full_Unibo_Deep_Camera_Mapping_Forensics.sha256"
OUT_NOTE = NOTES / "v78h_deep_camera_mapping_forensics_notes.md"
OUT_REPORT = REPORTS / "v78h_deep_camera_mapping_forensics_report.md"

ENC_RE = re.compile(r"^(c\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2}).*\.mp4$", re.IGNORECASE)
TLC_RE = re.compile(r"TLC\s*([1-6])", re.IGNORECASE)
FRIENDLY_RE = re.compile(r"TLC\s*1.*B1.*?(\d{3,4})[-_]?(\d{3,4})", re.IGNORECASE)

KEYWORDS = [
    "tlc1", "tlc 1", "tlc2", "tlc 2", "tlc3", "tlc 3",
    "tlc4", "tlc 4", "tlc5", "tlc 5", "tlc6", "tlc 6",
    "c0000", "c0001", "c0002", "c0003", "c0100", "c0101",
    "camera", "telecamera", "cam", "layout", "map", "mapping",
    "mappa", "codice", "code", "b1", "b3", "b4", "b6",
    "c1", "c3", "c4", "c6", "m1", "m2", "m3", "m4",
    "big", "small"
]

SMALL_TEXT_EXTS = {".txt", ".md", ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml", ".ini", ".cfg", ".conf", ".log"}
EXCEL_EXTS = {".xlsx", ".xlsm"}


def clean(x):
    if x is None:
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


def has_keyword(text):
    low = clean(text).lower()
    return any(k in low for k in KEYWORDS)


def parse_encoded_video(path):
    m = ENC_RE.match(path.name)
    if not m:
        return None
    code, yy, mm, dd, HH, MM, SS = m.groups()
    return {
        "video_path": str(path),
        "video_filename": path.name,
        "video_type": "encoded",
        "camera_code": code.lower(),
        "date": f"20{yy}-{mm}-{dd}",
        "start_hhmm": f"{HH}:{MM}",
        "start_hhmmss": f"{HH}:{MM}:{SS}",
    }


def parse_friendly_video(path):
    name = path.name
    if "TLC" not in name.upper():
        return None
    tlc = TLC_RE.search(name)
    if not tlc:
        return None

    start = ""
    m = re.search(r"(\d{3,4})[-_](\d{3,4})", name)
    if m:
        raw = m.group(1).zfill(4)
        start = f"{raw[:2]}:{raw[2:]}"

    return {
        "video_path": str(path),
        "video_filename": path.name,
        "video_type": "friendly",
        "camera_code": "",
        "tlc_camera": f"TLC{tlc.group(1)}",
        "date": "",
        "start_hhmm": start,
        "start_hhmmss": "",
    }


def build_video_inventory():
    rows = []
    if not UNIBO.exists():
        return pd.DataFrame()

    for p in sorted(UNIBO.glob("*.mp4")):
        enc = parse_encoded_video(p)
        if enc:
            enc["tlc_camera"] = ""
            rows.append(enc)
            continue

        fr = parse_friendly_video(p)
        if fr:
            rows.append(fr)

    return pd.DataFrame(rows)


def run_ffprobe(path):
    if not shutil.which("ffprobe"):
        return ""

    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path)
    ]

    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=20)
        return out.decode("utf-8", errors="ignore")
    except Exception as e:
        return f"FFPROBE_ERROR: {e}"


def scan_mp4_metadata(video_df):
    rows = []

    for _, r in video_df.iterrows():
        path = Path(r["video_path"])
        meta = run_ffprobe(path)

        if not meta:
            continue

        lines = meta.splitlines()
        for i, line in enumerate(lines, start=1):
            if has_keyword(line):
                rows.append({
                    "video_filename": r["video_filename"],
                    "video_path": str(path),
                    "video_type": r.get("video_type", ""),
                    "camera_code": r.get("camera_code", ""),
                    "line": i,
                    "hit_text": line[:1000],
                })

    return pd.DataFrame(rows)


def scan_mp4_header_strings(video_df):
    rows = []

    for _, r in video_df.iterrows():
        path = Path(r["video_path"])

        try:
            size = path.stat().st_size
            chunks = []

            with open(path, "rb") as f:
                chunks.append(f.read(4 * 1024 * 1024))

                if size > 8 * 1024 * 1024:
                    f.seek(max(0, size - 4 * 1024 * 1024))
                    chunks.append(f.read(4 * 1024 * 1024))

            raw = b"\n".join(chunks)
            text = raw.decode("latin-1", errors="ignore")
            candidates = re.findall(r"[A-Za-z0-9_\-:/\. ]{4,120}", text)

            for s in candidates:
                if has_keyword(s):
                    rows.append({
                        "video_filename": r["video_filename"],
                        "video_path": str(path),
                        "video_type": r.get("video_type", ""),
                        "camera_code": r.get("camera_code", ""),
                        "hit_text": clean(s)[:500],
                    })
        except Exception as e:
            rows.append({
                "video_filename": r.get("video_filename", ""),
                "video_path": str(path),
                "video_type": r.get("video_type", ""),
                "camera_code": r.get("camera_code", ""),
                "hit_text": f"SCAN_ERROR: {e}",
            })

    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame()


def scan_excel_deep():
    rows = []

    if openpyxl is None:
        return pd.DataFrame([{
            "path": "",
            "sheet": "",
            "location": "",
            "hit_type": "openpyxl_missing",
            "text": "openpyxl is not available",
        }])

    excel_paths = []
    for base in [UNIBO, UNIBO / "excel", DATASETS]:
        if base.exists():
            excel_paths.extend(sorted(base.glob("*.xlsx")))
            excel_paths.extend(sorted(base.glob("*.xlsm")))

    excel_paths = sorted(set(excel_paths))

    for path in excel_paths:
        try:
            wb = openpyxl.load_workbook(path, data_only=False, read_only=False)

            # workbook properties
            props = wb.properties
            for attr in ["title", "subject", "creator", "keywords", "description", "category", "lastModifiedBy"]:
                val = clean(getattr(props, attr, ""))
                if val and has_keyword(val):
                    rows.append({
                        "path": str(path),
                        "sheet": "",
                        "location": f"workbook_property:{attr}",
                        "hit_type": "workbook_property",
                        "text": val,
                    })

            # defined names
            for dn in wb.defined_names.values():
                txt = clean(dn.name) + " " + clean(dn.attr_text)
                if has_keyword(txt):
                    rows.append({
                        "path": str(path),
                        "sheet": "",
                        "location": "defined_name",
                        "hit_type": "defined_name",
                        "text": txt,
                    })

            for ws in wb.worksheets:
                if has_keyword(ws.title):
                    rows.append({
                        "path": str(path),
                        "sheet": ws.title,
                        "location": "sheet_title",
                        "hit_type": "sheet_title",
                        "text": ws.title,
                    })

                if ws.sheet_state != "visible":
                    rows.append({
                        "path": str(path),
                        "sheet": ws.title,
                        "location": "sheet_state",
                        "hit_type": "hidden_sheet",
                        "text": ws.sheet_state,
                    })

                for row in ws.iter_rows():
                    row_vals = []
                    hit = False
                    for cell in row:
                        val = clean(cell.value)
                        row_vals.append(val)

                        if val and has_keyword(val):
                            hit = True
                            rows.append({
                                "path": str(path),
                                "sheet": ws.title,
                                "location": cell.coordinate,
                                "hit_type": "cell_value",
                                "text": val[:1000],
                            })

                        if cell.comment and has_keyword(cell.comment.text):
                            rows.append({
                                "path": str(path),
                                "sheet": ws.title,
                                "location": cell.coordinate,
                                "hit_type": "cell_comment",
                                "text": clean(cell.comment.text)[:1000],
                            })

                        if cell.hyperlink:
                            hyp = clean(cell.hyperlink.target) + " " + clean(cell.hyperlink.location) + " " + clean(cell.hyperlink.display)
                            if has_keyword(hyp):
                                rows.append({
                                    "path": str(path),
                                    "sheet": ws.title,
                                    "location": cell.coordinate,
                                    "hit_type": "cell_hyperlink",
                                    "text": hyp[:1000],
                                })

                    row_text = " ; ".join(v for v in row_vals if v)
                    if hit and ("c000" in row_text.lower() or "c010" in row_text.lower() or "camera" in row_text.lower() or "telecamera" in row_text.lower()):
                        rows.append({
                            "path": str(path),
                            "sheet": ws.title,
                            "location": "row_context",
                            "hit_type": "row_context",
                            "text": row_text[:1500],
                        })

            wb.close()
        except Exception as e:
            rows.append({
                "path": str(path),
                "sheet": "",
                "location": "",
                "hit_type": "excel_scan_error",
                "text": str(e)[:500],
            })

    return pd.DataFrame(rows)


def scan_readable_small_files():
    rows = []

    roots = [DATASETS, ROOT]
    for base in roots:
        if not base.exists():
            continue

        for path in base.rglob("*"):
            if not path.is_file():
                continue

            if "Full_Unibo_Behaviour_Pipeline/outputs" in str(path):
                continue

            if path.suffix.lower() not in SMALL_TEXT_EXTS:
                continue

            try:
                if path.stat().st_size > 5 * 1024 * 1024:
                    continue
            except Exception:
                continue

            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for ln, line in enumerate(f, start=1):
                        if has_keyword(line):
                            rows.append({
                                "path": str(path),
                                "line": ln,
                                "text": " ".join(line.split())[:1000],
                            })
            except Exception as e:
                rows.append({
                    "path": str(path),
                    "line": "",
                    "text": f"READ_ERROR: {e}",
                })

    return pd.DataFrame(rows)


def extract_frame(video_path, out_path, label, sec=5, width=520):
    if cv2 is None:
        return False

    if out_path.exists():
        return True

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
    else:
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return False

    h, w = frame.shape[:2]
    scale = width / max(1, w)
    frame = cv2.resize(frame, (width, int(h * scale)))

    cv2.rectangle(frame, (0, 0), (width, 44), (0, 0, 0), -1)
    cv2.putText(frame, label[:70], (8, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), frame)

    # zoom crops for overlays/signs
    h2, w2 = frame.shape[:2]
    crops = {
        "top": frame[0:int(h2*0.25), 0:w2],
        "bottom": frame[int(h2*0.75):h2, 0:w2],
        "left": frame[0:h2, 0:int(w2*0.25)],
        "right": frame[0:h2, int(w2*0.75):w2],
        "center": frame[int(h2*0.25):int(h2*0.75), int(w2*0.25):int(w2*0.75)],
    }

    stem = out_path.stem
    for name, crop in crops.items():
        if crop.size:
            crop_path = out_path.parent / f"{stem}__crop_{name}.jpg"
            cv2.imwrite(str(crop_path), crop)

    return True


def build_visual_board(video_df):
    if cv2 is None or video_df.empty:
        return 0

    encoded = video_df[video_df["video_type"] == "encoded"].copy()
    friendly = video_df[video_df["video_type"] == "friendly"].copy()

    hours = ["07:00", "08:00", "09:00", "10:00", "11:00", "12:00"]
    created = 0

    html = []
    html.append("""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>v78h Visual Camera Board</title>
<style>
body { font-family: Arial, sans-serif; background:#f5f5f5; margin:0; }
.header { background:#111; color:white; padding:16px 24px; position:sticky; top:0; z-index:10; }
.container { padding:24px; }
.section { background:white; padding:16px; border-radius:10px; margin-bottom:24px; box-shadow:0 2px 8px rgba(0,0,0,0.08); }
.grid { display:flex; flex-wrap:wrap; gap:14px; }
.card { width:540px; background:#fafafa; border:1px solid #ccc; border-radius:8px; padding:8px; }
.card img { max-width:100%; border-radius:4px; display:block; margin-top:6px; }
.small { font-size:13px; color:#555; }
.ref { border:4px solid #2d6cdf; background:#eef4ff; }
.code-title { font-size:22px; margin-top:0; }
</style>
</head>
<body>
<div class="header">
<h1>v78h Visual Camera Mapping Board</h1>
<p>Goal: look for visible overlays, labels, wall signs, pen labels, or geometry clues. This page does not assign mappings automatically.</p>
</div>
<div class="container">
""")

    html.append("<div class='section'><h2>Known anchor: TLC1 reference videos</h2><div class='grid'>")
    for _, r in friendly.iterrows():
        if r.get("tlc_camera") == "TLC1" and r.get("start_hhmm") in hours:
            img = STATIC / f"REF_TLC1_{r['start_hhmm'].replace(':','')}.jpg"
            if extract_frame(r["video_path"], img, f"REFERENCE TLC1 | {r['start_hhmm']} | {r['video_filename']}"):
                created += 1
            html.append(f"<div class='card ref'><b>TLC1 reference {r['start_hhmm']}</b><img src='visual_board/{img.name}'><div class='small'>{r['video_filename']}</div></div>")
    html.append("</div></div>")

    for code in sorted(encoded["camera_code"].dropna().unique()):
        subset = encoded[encoded["camera_code"] == code].copy()
        html.append(f"<div class='section'><h2 class='code-title'>{code}</h2><div class='grid'>")

        for hour in hours:
            rows = subset[subset["start_hhmm"] == hour]
            if rows.empty:
                continue
            r = rows.iloc[0]
            img = STATIC / f"{code}_{hour.replace(':','')}.jpg"
            if extract_frame(r["video_path"], img, f"{code} | {hour} | {r['video_filename']}"):
                created += 1

            html.append(f"<div class='card'><b>{code} | {hour}</b><img src='visual_board/{img.name}'><div class='small'>{r['video_filename']}</div>")

            for crop_name in ["top", "bottom", "left", "right", "center"]:
                crop = STATIC / f"{img.stem}__crop_{crop_name}.jpg"
                if crop.exists():
                    html.append(f"<details><summary>crop {crop_name}</summary><img src='visual_board/{crop.name}'></details>")

            html.append("</div>")

        html.append("</div></div>")

    html.append("</div></body></html>")
    OUT_VISUAL_INDEX.write_text("\n".join(html), encoding="utf-8")

    return created


issues = []

video_df = build_video_inventory()
to_csv(video_df, OUT_VIDEO_INV)

mp4_meta_df = scan_mp4_metadata(video_df)
to_csv(mp4_meta_df, OUT_MP4_META)

mp4_string_df = scan_mp4_header_strings(video_df)
to_csv(mp4_string_df, OUT_MP4_STRINGS)

excel_df = scan_excel_deep()
to_csv(excel_df, OUT_EXCEL_DEEP)

file_hits_df = scan_readable_small_files()
to_csv(file_hits_df, OUT_FILE_HITS)

visual_count = build_visual_board(video_df)

# QA
qa_rows = []

def qa(name, expected, actual, passed, severity, detail):
    qa_rows.append({
        "check_name": name,
        "expected": str(expected),
        "actual": str(actual),
        "passed": bool(passed),
        "severity": severity,
        "detail": detail,
    })

encoded_count = int((video_df["video_type"] == "encoded").sum()) if not video_df.empty else 0
friendly_count = int((video_df["video_type"] == "friendly").sum()) if not video_df.empty else 0
codes_count = int(video_df[video_df["video_type"] == "encoded"]["camera_code"].nunique()) if not video_df.empty else 0

qa("video_inventory_created", ">0", len(video_df), len(video_df) > 0, "hard", "Video inventory should exist.")
qa("encoded_videos_found", ">0", encoded_count, encoded_count > 0, "hard", "Encoded camera videos should exist.")
qa("camera_codes_found", ">=6", codes_count, codes_count >= 6, "hard", "Expected encoded camera codes should be found.")
qa("friendly_tlc_reference_found", ">0", friendly_count, friendly_count > 0, "hard", "Friendly TLC reference videos should exist.")
qa("visual_board_created", ">0", visual_count, visual_count > 0, "hard", "Visual board frames should be created.")
qa("mp4_metadata_scan_done", ">=0", len(mp4_meta_df), True, "info", "MP4 metadata scan is diagnostic.")
qa("mp4_header_string_scan_done", ">=0", len(mp4_string_df), True, "info", "MP4 header string scan is diagnostic.")
qa("excel_deep_scan_done", ">=0", len(excel_df), True, "info", "Excel deep scan is diagnostic.")
qa("readable_file_scan_done", ">=0", len(file_hits_df), True, "info", "Readable file keyword scan is diagnostic.")

qa_df = pd.DataFrame(qa_rows)
to_csv(qa_df, OUT_QA)

hard_quality_failures = int(((qa_df["severity"] == "hard") & (~qa_df["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v78h_quality_checks",
        "issue_type": "hard_forensics_failed",
        "issue_detail": f"{hard_quality_failures} hard QA checks failed.",
        "severity": "hard",
    })

if len(mp4_meta_df) == 0 and len(mp4_string_df) == 0:
    issues.append({
        "item": "mp4_metadata",
        "issue_type": "info_no_direct_mp4_metadata_mapping_found",
        "issue_detail": "No direct TLC/camera-code mapping found in MP4 metadata/header strings.",
        "severity": "info",
    })

issues.append({
    "item": "scope",
    "issue_type": "info_forensics_only",
    "issue_detail": "v78h searches metadata/Excel/readable files and creates visual boards. It does not apply mappings or run tracking.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

manifest = {
    "version": "v78h_deep_camera_mapping_forensics",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "video_rows": int(len(video_df)),
    "encoded_videos": encoded_count,
    "friendly_videos": friendly_count,
    "camera_codes": codes_count,
    "mp4_metadata_hits": int(len(mp4_meta_df)),
    "mp4_header_string_hits": int(len(mp4_string_df)),
    "excel_deep_hits": int(len(excel_df)),
    "readable_file_hits": int(len(file_hits_df)),
    "visual_frames_created": int(visual_count),
    "hard_issue_count": int(hard_issue_count),
    "claim_boundary": "forensics and visual evidence only; no mapping applied",
}
OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

readme = f"""# v78h Deep Camera Mapping Forensics

## Purpose

This stage exhausts non-permission-based evidence before asking for access:
1. MP4 metadata / ffprobe scan
2. MP4 header string scan
3. Deep Excel scan: sheet titles, hidden sheets, comments, hyperlinks, formulas, cell values
4. Readable small-file keyword scan
5. Visual camera board for human inspection

## Counts

- Video rows: {len(video_df)}
- Encoded videos: {encoded_count}
- Friendly reference videos: {friendly_count}
- Encoded camera codes: {codes_count}
- MP4 metadata hits: {len(mp4_meta_df)}
- MP4 header string hits: {len(mp4_string_df)}
- Excel deep hits: {len(excel_df)}
- Readable file hits: {len(file_hits_df)}
- Visual frames created: {visual_count}
- Hard issues: {hard_issue_count}

## Visual board

Open:

`{OUT_VISUAL_INDEX}`

Look for:
- camera overlay text
- wall labels
- pen labels
- physical signs
- geometry that connects to TLC naming
- whether c0002 remains visually consistent with TLC1 reference

This stage does not apply mappings.
"""

OUT_README.write_text(readme, encoding="utf-8")
OUT_REPORT.write_text(readme, encoding="utf-8")

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

zip_hash = sha256_file(OUT_ZIP)
OUT_SHA.write_text(f"{zip_hash}  {OUT_ZIP.name}\n")

decision = pd.DataFrame([{
    "v78h_decision": "deep_camera_mapping_forensics_completed" if hard_issue_count == 0 else "deep_camera_mapping_forensics_has_blocking_issues",
    "video_rows": int(len(video_df)),
    "encoded_videos": encoded_count,
    "friendly_videos": friendly_count,
    "camera_codes": codes_count,
    "mp4_metadata_hits": int(len(mp4_meta_df)),
    "mp4_header_string_hits": int(len(mp4_string_df)),
    "excel_deep_hits": int(len(excel_df)),
    "readable_file_hits": int(len(file_hits_df)),
    "visual_frames_created": int(visual_count),
    "visual_board_path": str(OUT_VISUAL_INDEX),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_quality_failures": int(hard_quality_failures),
    "hard_issue_count": int(hard_issue_count),
    "warning_count": int(warning_count),
    "info_count": int(info_count),
    "issue_count": int(len(issues_df)),
    "ready_for_visual_review": bool(hard_issue_count == 0 and visual_count > 0),
    "ready_for_auto_mapping": False,
    "ready_for_v79_full_tracking_preparation": False,
    "claim_scope": "deep_forensics_and_visual_board_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# v78h Deep Camera Mapping Forensics\n\n"
    f"- v78h decision: {decision.iloc[0]['v78h_decision']}\n"
    f"- Video rows: {len(video_df)}\n"
    f"- Encoded videos: {encoded_count}\n"
    f"- Friendly videos: {friendly_count}\n"
    f"- Camera codes: {codes_count}\n"
    f"- MP4 metadata hits: {len(mp4_meta_df)}\n"
    f"- MP4 header string hits: {len(mp4_string_df)}\n"
    f"- Excel deep hits: {len(excel_df)}\n"
    f"- Readable file hits: {len(file_hits_df)}\n"
    f"- Visual frames created: {visual_count}\n"
    f"- Visual board path: {OUT_VISUAL_INDEX}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Ready for visual review: {bool(hard_issue_count == 0 and visual_count > 0)}\n"
    f"- Ready for auto mapping: False\n"
    f"- Ready for v79 full tracking preparation: False\n\n"
    "This stage does not apply mappings. Use the visual board and hit tables to decide whether non-permission evidence is enough.\n",
    encoding="utf-8"
)

print("=== v78h decision ===")
print(decision.to_string(index=False))

print("\n=== MP4 metadata hits sample ===")
print(mp4_meta_df.head(50).to_string(index=False) if len(mp4_meta_df) else "none")

print("\n=== MP4 header string hits sample ===")
print(mp4_string_df.head(50).to_string(index=False) if len(mp4_string_df) else "none")

print("\n=== Excel deep hits sample ===")
print(excel_df.head(80).to_string(index=False) if len(excel_df) else "none")

print("\n=== readable file hits sample ===")
print(file_hits_df.head(80).to_string(index=False) if len(file_hits_df) else "none")

print("\n=== issues ===")
print(issues_df.to_string(index=False))
