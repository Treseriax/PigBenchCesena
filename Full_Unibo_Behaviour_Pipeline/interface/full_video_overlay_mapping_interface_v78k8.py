from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, quote, unquote
from pathlib import Path
from datetime import datetime
import html
import csv
import re
import mimetypes
import shutil
import subprocess
import pandas as pd

ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"
RAW_UNIBO = Path("/work/pig/datasets/Unibo")

K7_PKG = FULL / "outputs" / "v78k7_overlay_based_video_identity_mapping" / "Full_Unibo_Overlay_Based_Video_Identity_Mapping"
K7_TABLE = K7_PKG / "v78k7_video_overlay_identity_table.csv"

OUT = FULL / "outputs" / "v78k8_full_video_manual_overlay_mapping"
PKG = OUT / "Full_Unibo_Full_Video_Manual_Overlay_Mapping"
CROPS = PKG / "overlay_crops"
FRAMES = PKG / "full_frames"

for p in [OUT, PKG, CROPS, FRAMES, FULL / "notes", FULL / "reports", FULL / "progress", FULL / "config"]:
    p.mkdir(parents=True, exist_ok=True)

WORKING = PKG / "v78k8_FULL_VIDEO_MAPPING_WORKING.csv"
FINAL = PKG / "v78k8_FULL_VIDEO_MAPPING_FINAL.csv"
SUMMARY = PKG / "v78k8_full_video_mapping_summary.csv"
ISSUES = OUT / "v78k8_issues.csv"
DECISION = OUT / "v78k8_decision_summary.csv"
NOTE = FULL / "notes" / "v78k8_full_video_manual_overlay_mapping_notes.md"
CONFIG = FULL / "config" / "full_video_overlay_mapping_v78k8.csv"

VALID_TLC = ["", "TLC1", "TLC2", "TLC3", "TLC4", "TLC5", "TLC6"]
VALID_PENS = ["", "B1", "B3", "B4", "B6", "C1", "C3", "C4", "C6", "M1", "M2", "M3", "M4"]
STATUSES = ["UNREVIEWED", "RESOLVED", "UNCERTAIN", "UNREADABLE", "NO_OVERLAY", "EXCLUDE"]
CONFIDENCES = ["", "high", "medium_high", "medium", "low"]

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
ENCODED_RE = re.compile(r"(c\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", re.I)

def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s

def esc(x):
    return html.escape(clean(x))

def to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")

def read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path).fillna("")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(clean)
    return df

def parse_video_filename(path):
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

    if not start_time:
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

def extract_images(video_path, crop_path, frame_path):
    if crop_path.exists() and frame_path.exists():
        return True

    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        candidates = [int(fps * 2), int(fps * 8), 0]
        if total > 0:
            candidates.append(min(total - 1, int(total * 0.1)))

        frame = None
        for idx in candidates:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
            ok, fr = cap.read()
            if ok and fr is not None:
                frame = fr
                break
        cap.release()

        if frame is None:
            return False

        h, w = frame.shape[:2]
        crop = frame[0:int(h * 0.25), 0:int(w * 0.50)]

        crop_big = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2))
        frame_small = cv2.resize(frame, (min(960, w), int(h * min(960, w) / w)))

        cv2.imwrite(str(crop_path), crop_big)
        cv2.imwrite(str(frame_path), frame_small)
        return True

    except Exception:
        pass

    if shutil.which("ffmpeg"):
        try:
            subprocess.run([
                "ffmpeg", "-y", "-ss", "2", "-i", str(video_path),
                "-frames:v", "1",
                "-vf", "crop=iw*0.50:ih*0.25:0:0,scale=iw*2:ih*2",
                str(crop_path)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)

            subprocess.run([
                "ffmpeg", "-y", "-ss", "2", "-i", str(video_path),
                "-frames:v", "1",
                "-vf", "scale=960:-1",
                str(frame_path)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)

            return crop_path.exists() and frame_path.exists()
        except Exception:
            return False

    return False

def init_from_v78k7_or_raw():
    if K7_TABLE.exists():
        base = read_csv(K7_TABLE)
    else:
        rows = []
        for p in sorted(RAW_UNIBO.rglob("*")):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                rows.append(parse_video_filename(p))
        base = pd.DataFrame(rows)

    required = [
        "video_filename", "video_path", "video_type", "filename_c_code",
        "filename_date", "filename_start_time", "filename_start_hour",
        "filename_tlc_camera", "filename_room_pen", "file_size_bytes"
    ]
    for c in required:
        if c not in base.columns:
            base[c] = ""

    rows = []

    for i, r in base.reset_index(drop=True).iterrows():
        video_path = Path(clean(r["video_path"]))
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", video_path.stem if video_path.name else clean(r["video_filename"]))
        crop_path = CROPS / f"{i:04d}_{safe}_overlay_crop.jpg"
        frame_path = FRAMES / f"{i:04d}_{safe}_frame.jpg"

        if video_path.exists():
            extract_images(video_path, crop_path, frame_path)

        rows.append({
            "video_id": f"vid_{i:03d}",
            "video_index": i,
            "video_filename": clean(r["video_filename"]),
            "video_path": clean(r["video_path"]),
            "video_type": clean(r["video_type"]),
            "filename_c_code": clean(r["filename_c_code"]),
            "filename_date": clean(r["filename_date"]),
            "filename_start_time": clean(r["filename_start_time"]),
            "filename_start_hour": clean(r["filename_start_hour"]),
            "filename_tlc_camera_suggestion": clean(r["filename_tlc_camera"]),
            "filename_room_pen_suggestion": clean(r["filename_room_pen"]),
            "overlay_crop_path": str(crop_path) if crop_path.exists() else "",
            "full_frame_path": str(frame_path) if frame_path.exists() else "",
            "manual_tlc_camera": "",
            "manual_room_pen": "",
            "manual_overlay_text": "",
            "manual_review_status": "UNREVIEWED",
            "manual_confidence": "",
            "manual_note": "",
            "reviewed_at": "",
        })

    df = pd.DataFrame(rows).sort_values(["video_type", "filename_c_code", "filename_start_time", "video_filename"])
    df = df.reset_index(drop=True)
    df["video_index"] = range(len(df))
    return df

def load_state():
    if WORKING.exists():
        return read_csv(WORKING)
    df = init_from_v78k7_or_raw()
    to_csv(df, WORKING)
    return df

def save_state(df):
    df = df.copy()
    df["video_index"] = range(len(df))
    to_csv(df, WORKING)
    to_csv(df, CONFIG)

def allowed_file(path):
    p = Path(unquote(path)).resolve()
    roots = [PKG.resolve(), K7_PKG.resolve(), RAW_UNIBO.resolve(), FULL.resolve()]
    return p.exists() and any(str(p).startswith(str(r)) for r in roots)

def img_url(path):
    return "/img?path=" + quote(clean(path))

def validate_mapping(df, final=False):
    issues = []

    for _, r in df.iterrows():
        status = clean(r["manual_review_status"])
        tlc = clean(r["manual_tlc_camera"])
        pen = clean(r["manual_room_pen"])

        if status == "RESOLVED":
            if tlc not in VALID_TLC or not tlc:
                issues.append({
                    "video_id": clean(r["video_id"]),
                    "video_filename": clean(r["video_filename"]),
                    "issue_type": "resolved_missing_or_invalid_tlc",
                    "severity": "hard",
                })
            if pen not in VALID_PENS or not pen:
                issues.append({
                    "video_id": clean(r["video_id"]),
                    "video_filename": clean(r["video_filename"]),
                    "issue_type": "resolved_missing_or_invalid_pen",
                    "severity": "hard",
                })

    if final:
        unresolved = df[df["manual_review_status"] == "UNREVIEWED"]
        if len(unresolved):
            for _, r in unresolved.iterrows():
                issues.append({
                    "video_id": clean(r["video_id"]),
                    "video_filename": clean(r["video_filename"]),
                    "issue_type": "final_save_with_unreviewed_video",
                    "severity": "warning",
                })

    return pd.DataFrame(issues)

def write_reports(df, final=False):
    resolved = int((df["manual_review_status"] == "RESOLVED").sum())
    unreviewed = int((df["manual_review_status"] == "UNREVIEWED").sum())
    uncertain = int((df["manual_review_status"] == "UNCERTAIN").sum())
    unreadable = int((df["manual_review_status"] == "UNREADABLE").sum())
    no_overlay = int((df["manual_review_status"] == "NO_OVERLAY").sum())
    excluded = int((df["manual_review_status"] == "EXCLUDE").sum())

    encoded_total = int((df["video_type"] == "encoded_c_code").sum())
    encoded_resolved = int(((df["video_type"] == "encoded_c_code") & (df["manual_review_status"] == "RESOLVED")).sum())

    friendly_total = int((df["video_type"] == "friendly_tlc").sum())
    friendly_resolved = int(((df["video_type"] == "friendly_tlc") & (df["manual_review_status"] == "RESOLVED")).sum())

    summary = pd.DataFrame([{
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "video_rows": len(df),
        "resolved": resolved,
        "unreviewed": unreviewed,
        "uncertain": uncertain,
        "unreadable": unreadable,
        "no_overlay": no_overlay,
        "excluded": excluded,
        "encoded_total": encoded_total,
        "encoded_resolved": encoded_resolved,
        "friendly_total": friendly_total,
        "friendly_resolved": friendly_resolved,
        "all_videos_reviewed": bool(unreviewed == 0),
        "all_videos_resolved_or_marked": bool(unreviewed == 0),
        "ready_for_video_mapping_use": bool(unreviewed == 0 and resolved > 0),
    }])
    to_csv(summary, SUMMARY)

    issues = validate_mapping(df, final=final)
    if len(issues) == 0:
        issues = pd.DataFrame([{
            "video_id": "",
            "video_filename": "",
            "issue_type": "info_no_validation_issues" if final else "info_working_validation_ok",
            "severity": "info",
        }])
    to_csv(issues, ISSUES)

    hard_count = int((issues["severity"] == "hard").sum()) if "severity" in issues.columns else 0
    warning_count = int((issues["severity"] == "warning").sum()) if "severity" in issues.columns else 0

    decision = pd.DataFrame([{
        "v78k8_decision": "full_video_manual_mapping_final_saved" if final and hard_count == 0 else "full_video_manual_mapping_working",
        "video_rows": len(df),
        "resolved": resolved,
        "unreviewed": unreviewed,
        "encoded_total": encoded_total,
        "encoded_resolved": encoded_resolved,
        "friendly_total": friendly_total,
        "friendly_resolved": friendly_resolved,
        "hard_issue_count": hard_count,
        "warning_count": warning_count,
        "ready_for_video_mapping_use": bool(final and hard_count == 0 and unreviewed == 0 and resolved > 0),
        "ready_for_tracking": False,
        "working_csv": str(WORKING),
        "final_csv": str(FINAL if final else ""),
        "summary_csv": str(SUMMARY),
        "claim_scope": "manual_overlay_video_identity_mapping_only",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    to_csv(decision, DECISION)

    NOTE.write_text(
        "# v78k8 Full Video Manual Overlay Mapping\n\n"
        f"- Video rows: {len(df)}\n"
        f"- Resolved: {resolved}\n"
        f"- Unreviewed: {unreviewed}\n"
        f"- Encoded total: {encoded_total}\n"
        f"- Encoded resolved: {encoded_resolved}\n"
        f"- Friendly total: {friendly_total}\n"
        f"- Friendly resolved: {friendly_resolved}\n"
        f"- Hard issues: {hard_count}\n"
        f"- Ready for video mapping use: {bool(final and hard_count == 0 and unreviewed == 0 and resolved > 0)}\n\n"
        "This stage maps every video from visible overlay/manual review. It does not run tracking.\n",
        encoding="utf-8"
    )

def options(values, selected):
    out = []
    for v in values:
        label = v if v else "--"
        sel = "selected" if clean(v) == clean(selected) else ""
        out.append(f"<option value='{esc(v)}' {sel}>{esc(label)}</option>")
    return "\n".join(out)

def row_index_by_id(df, vid):
    matches = df.index[df["video_id"] == vid].tolist()
    if matches:
        return matches[0]
    return 0

class Handler(BaseHTTPRequestHandler):
    def send_html(self, body):
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def redirect(self, url):
        self.send_response(303)
        self.send_header("Location", url)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path == "/img":
            p = qs.get("path", [""])[0]
            if not allowed_file(p):
                self.send_response(404)
                self.end_headers()
                return
            path = Path(unquote(p))
            data = path.read_bytes()
            ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if parsed.path == "/final":
            df = load_state()
            save_state(df)
            to_csv(df, FINAL)
            write_reports(df, final=True)
            self.redirect("/")
            return

        self.render_main(qs)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        data = parse_qs(self.rfile.read(length).decode("utf-8"))

        df = load_state()
        vid = clean(data.get("video_id", [""])[0])
        idx = row_index_by_id(df, vid)

        for col in ["manual_tlc_camera", "manual_room_pen", "manual_overlay_text", "manual_review_status", "manual_confidence", "manual_note"]:
            df.loc[idx, col] = clean(data.get(col, [""])[0])

        df.loc[idx, "reviewed_at"] = datetime.now().isoformat(timespec="seconds")

        save_state(df)
        write_reports(df, final=False)

        action = clean(data.get("action", ["save"])[0])
        if action == "save_next":
            next_idx = min(idx + 1, len(df) - 1)
            self.redirect(f"/?video_id={quote(clean(df.loc[next_idx, 'video_id']))}")
        else:
            self.redirect(f"/?video_id={quote(vid)}")

    def render_main(self, qs):
        df = load_state()
        if df.empty:
            self.send_html("<h1>No videos found</h1>")
            return

        filter_status = clean(qs.get("status", ["ALL"])[0])
        filter_type = clean(qs.get("type", ["ALL"])[0])

        view = df.copy()
        if filter_status != "ALL":
            view = view[view["manual_review_status"] == filter_status]
        if filter_type != "ALL":
            view = view[view["video_type"] == filter_type]

        selected_id = clean(qs.get("video_id", [""])[0])
        if not selected_id or selected_id not in set(df["video_id"]):
            if not view.empty:
                selected_id = clean(view.iloc[0]["video_id"])
            else:
                selected_id = clean(df.iloc[0]["video_id"])

        idx = row_index_by_id(df, selected_id)
        row = df.loc[idx]

        resolved = int((df["manual_review_status"] == "RESOLVED").sum())
        unreviewed = int((df["manual_review_status"] == "UNREVIEWED").sum())

        nav_items = []
        for _, r in view.iterrows():
            st = clean(r["manual_review_status"])
            cls = "done" if st == "RESOLVED" else ("warn" if st in {"UNCERTAIN", "UNREADABLE", "NO_OVERLAY"} else "todo")
            label = f"{r['video_index']:02d} | {r['video_type']} | {r['filename_c_code']} | {r['filename_start_time']} | {st}"
            nav_items.append(f"<a class='{cls}' href='/?video_id={quote(clean(r['video_id']))}&status={quote(filter_status)}&type={quote(filter_type)}'>{esc(label)}</a>")

        crop_html = "<p>No overlay crop image.</p>"
        if clean(row["overlay_crop_path"]):
            crop_html = f"<img class='crop' src='{img_url(row['overlay_crop_path'])}'>"

        frame_html = "<p>No full frame image.</p>"
        if clean(row["full_frame_path"]):
            frame_html = f"<img class='frame' src='{img_url(row['full_frame_path'])}'>"

        body = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>v78k8 Full Video Overlay Mapping</title>
<style>
body {{ font-family: Arial, sans-serif; margin:0; background:#f4f4f4; }}
.header {{ background:#111; color:white; padding:12px 20px; position:sticky; top:0; z-index:50; }}
.wrap {{ display:flex; }}
.side {{ width:430px; background:white; padding:12px; height:calc(100vh - 58px); overflow:auto; border-right:1px solid #ccc; position:sticky; top:58px; }}
.main {{ flex:1; padding:20px; }}
.section {{ background:white; padding:16px; margin-bottom:18px; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,.08); }}
a {{ display:block; padding:6px; margin:3px 0; border-radius:5px; text-decoration:none; color:#111; font-size:13px; }}
a.done {{ background:#e5ffe5; }}
a.todo {{ background:#fff3cd; }}
a.warn {{ background:#ffe6cc; }}
.toplinks a {{ display:inline-block; background:#eee; margin-right:8px; }}
table {{ border-collapse: collapse; width:100%; }}
td, th {{ border:1px solid #ccc; padding:6px; vertical-align:top; }}
th {{ background:#eee; }}
.crop {{ width:100%; max-width:900px; border:3px solid #c77d00; border-radius:6px; }}
.frame {{ width:100%; max-width:900px; border:1px solid #999; border-radius:6px; }}
select, textarea, input {{ width:100%; padding:8px; margin:5px 0 12px 0; }}
button {{ padding:10px 14px; font-weight:bold; margin-right:8px; }}
.badge {{ display:inline-block; padding:4px 8px; border-radius:5px; background:#eee; margin-right:5px; }}
</style>
</head>
<body>
<div class="header">
<h2>v78k8 Full 84-Video Manual Overlay Mapping</h2>
</div>

<div class="wrap">
<div class="side">
<h3>Progress</h3>
<p><span class="badge">Resolved {resolved}</span><span class="badge">Unreviewed {unreviewed}</span><span class="badge">Total {len(df)}</span></p>

<div class="toplinks">
<a href="/?status=ALL&type={quote(filter_type)}">All</a>
<a href="/?status=UNREVIEWED&type={quote(filter_type)}">Unreviewed</a>
<a href="/?status=RESOLVED&type={quote(filter_type)}">Resolved</a>
<a href="/?status=UNCERTAIN&type={quote(filter_type)}">Uncertain</a>
</div>

<div class="toplinks">
<a href="/?status={quote(filter_status)}&type=ALL">All types</a>
<a href="/?status={quote(filter_status)}&type=encoded_c_code">Encoded</a>
<a href="/?status={quote(filter_status)}&type=friendly_tlc">Friendly</a>
</div>

<p><a class="done" href="/final">SAVE CURRENT TABLE AS FINAL</a></p>

<h3>Videos</h3>
{''.join(nav_items)}
</div>

<div class="main">

<div class="section">
<h2>{esc(row['video_index'])} — {esc(row['video_filename'])}</h2>
<table>
<tr><th>video_id</th><td>{esc(row['video_id'])}</td></tr>
<tr><th>type</th><td>{esc(row['video_type'])}</td></tr>
<tr><th>c-code</th><td>{esc(row['filename_c_code'])}</td></tr>
<tr><th>date</th><td>{esc(row['filename_date'])}</td></tr>
<tr><th>time</th><td>{esc(row['filename_start_time'])}</td></tr>
<tr><th>filename suggestion</th><td>{esc(row['filename_tlc_camera_suggestion'])} / {esc(row['filename_room_pen_suggestion'])}</td></tr>
<tr><th>path</th><td><small>{esc(row['video_path'])}</small></td></tr>
</table>
</div>

<div class="section">
<h3>Overlay crop — sol üst yazıyı buradan oku</h3>
{crop_html}
</div>

<div class="section">
<h3>Full frame context</h3>
{frame_html}
</div>

<div class="section">
<h3>Manual mapping</h3>
<form method="POST">
<input type="hidden" name="video_id" value="{esc(row['video_id'])}">

<label>TLC camera</label>
<select name="manual_tlc_camera">
{options(VALID_TLC, row['manual_tlc_camera'])}
</select>

<label>Room / Pen</label>
<select name="manual_room_pen">
{options(VALID_PENS, row['manual_room_pen'])}
</select>

<label>Overlay text exactly as seen</label>
<input name="manual_overlay_text" value="{esc(row['manual_overlay_text'])}" placeholder="example: Tlc 5 - M1">

<label>Review status</label>
<select name="manual_review_status">
{options(STATUSES, row['manual_review_status'])}
</select>

<label>Confidence</label>
<select name="manual_confidence">
{options(CONFIDENCES, row['manual_confidence'])}
</select>

<label>Note</label>
<textarea name="manual_note" rows="3">{esc(row['manual_note'])}</textarea>

<button name="action" value="save">Save</button>
<button name="action" value="save_next">Save & Next</button>
</form>
</div>

</div>
</div>
</body>
</html>
"""
        self.send_html(body)

if __name__ == "__main__":
    df = load_state()
    save_state(df)
    write_reports(df, final=False)
    print("Serving v78k8 full video manual overlay mapping on http://0.0.0.0:8555")
    print("Working CSV:", WORKING)
    ThreadingHTTPServer(("0.0.0.0", 8555), Handler).serve_forever()
