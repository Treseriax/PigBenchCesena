from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, quote, unquote
from pathlib import Path
from datetime import datetime
import html
import csv
import mimetypes
import pandas as pd

ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

V78I = FULL / "outputs" / "v78i_visual_stream_grouping_audit" / "Full_Unibo_Visual_Stream_Grouping_Audit"
V78J = FULL / "outputs" / "v78j_time_aware_visual_stream_resolver" / "Full_Unibo_Time_Aware_Visual_Stream_Resolver"
V78K = FULL / "outputs" / "v78k_pen_roi_resolution"

V78K.mkdir(parents=True, exist_ok=True)
BACKUP_DIR = V78K / "roi_backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_CSV = V78I / "v78i_frame_features.csv"
TARGET_TEMPLATE = V78J / "v78j_MANUAL_FILL_visual_stream_resolution.csv"

WORKING_CSV = V78K / "v78k_MANUAL_FILLED_pen_roi_resolution_WORKING.csv"
FINAL_CSV = V78K / "v78k_MANUAL_FILLED_pen_roi_resolution_FINAL.csv"
SUMMARY_CSV = V78K / "v78k_pen_roi_interface_summary.csv"
CONFIG_CSV = FULL / "config" / "pen_roi_resolution_v78k.csv"
CONFIG_CSV.parent.mkdir(parents=True, exist_ok=True)

GROUP_A_CODES = ["c0002", "c0000", "c0100"]
GROUP_B_CODES = ["c0001", "c0003", "c0101"]
ALL_CODES = GROUP_A_CODES + GROUP_B_CODES

UNRESOLVED_STATUSES = {
    "",
    "needs_pen_roi_resolution",
    "unresolved_after_manual_review",
}

ROI_COLS = [
    "target_id", "date", "camera", "pen", "unresolved_windows", "candidate_video_camera_codes",
    "selected_visual_group_id", "view_type", "video_camera_code", "sample_hour",
    "roi_x1_norm", "roi_y1_norm", "roi_x2_norm", "roi_y2_norm",
    "resolution_status", "confidence", "evidence_type", "reviewer_note", "saved_at"
]

def esc(x):
    return html.escape("" if x is None else str(x))

def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s

def read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path).fillna("")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(clean)
    return df

def write_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")

def init_roi_table():
    if WORKING_CSV.exists():
        df = read_csv(WORKING_CSV)
    elif FINAL_CSV.exists():
        df = read_csv(FINAL_CSV)
    else:
        base = read_csv(TARGET_TEMPLATE)
        rows = []
        for _, r in base.iterrows():
            rows.append({
                "target_id": clean(r.get("target_id")),
                "date": clean(r.get("date")),
                "camera": clean(r.get("camera")),
                "pen": clean(r.get("pen")),
                "unresolved_windows": clean(r.get("unresolved_windows")),
                "candidate_video_camera_codes": clean(r.get("candidate_video_camera_codes")),
                "selected_visual_group_id": clean(r.get("selected_visual_group_id")),
                "view_type": "",
                "video_camera_code": "",
                "sample_hour": "",
                "roi_x1_norm": "",
                "roi_y1_norm": "",
                "roi_x2_norm": "",
                "roi_y2_norm": "",
                "resolution_status": "needs_pen_roi_resolution",
                "confidence": "",
                "evidence_type": "",
                "reviewer_note": "",
                "saved_at": "",
            })
        df = pd.DataFrame(rows)

        # TLC1 rows are still not final until ROI is drawn.
        df.loc[df["camera"] == "TLC1", "selected_visual_group_id"] = "GROUP_A"

    for c in ROI_COLS:
        if c not in df.columns:
            df[c] = ""
    return df[ROI_COLS].copy()

def save_table(df, final=False):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BACKUP_DIR / f"v78k_pen_roi_backup_{ts}.csv"

    write_csv(df, backup)
    write_csv(df, WORKING_CSV)
    write_csv(df, CONFIG_CSV)

    if final:
        write_csv(df, FINAL_CSV)

    unresolved = int(df["resolution_status"].isin(UNRESOLVED_STATUSES).sum())
    resolved = int((df["resolution_status"] == "resolved_with_pen_roi").sum())
    not_visible = int((df["resolution_status"] == "not_visible_after_manual_review").sum())

    summary = pd.DataFrame([{
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "rows": len(df),
        "resolved_with_pen_roi": resolved,
        "not_visible_count": not_visible,
        "unresolved_count": unresolved,
        "working_csv": str(WORKING_CSV),
        "final_csv": str(FINAL_CSV) if final else "",
        "config_csv": str(CONFIG_CSV),
        "backup_csv": str(backup),
        "final_saved": bool(final),
        "ready_for_v78l_validation": bool(final and unresolved == 0),
    }])
    write_csv(summary, SUMMARY_CSV)
    return summary

def allowed_file(path):
    try:
        p = Path(path).resolve()
        roots = [FULL.resolve(), ROOT.resolve(), Path("/work/pig/datasets").resolve()]
        return p.exists() and any(str(p).startswith(str(r)) for r in roots)
    except Exception:
        return False

def image_url(path):
    return "/img?path=" + quote(str(path))

def feature_img_path(row):
    p = Path(clean(row.get("frame_image", "")))
    if p.exists():
        return p
    alt = V78I / "visual_stream_groups" / p.name
    if alt.exists():
        return alt
    return None

def group_for_code(code):
    if code in GROUP_A_CODES:
        return "GROUP_A"
    if code in GROUP_B_CODES:
        return "GROUP_B"
    return ""

def view_for_code(code):
    if code in ["c0100", "c0101"]:
        return "top"
    if code in ["c0000", "c0001", "c0002", "c0003"]:
        return "side"
    return ""

def choose_default_code(row):
    saved = clean(row.get("video_camera_code", ""))
    if saved:
        return saved
    if clean(row.get("selected_visual_group_id", "")) == "GROUP_A":
        return "c0002"
    if clean(row.get("selected_visual_group_id", "")) == "GROUP_B":
        return "c0001"
    return "c0002"

def choose_default_hour(row, features):
    saved = clean(row.get("sample_hour", ""))
    if saved:
        return saved
    hours = sorted([h for h in features["start_hhmm"].dropna().unique().tolist() if clean(h)])
    if "09:00" in hours:
        return "09:00"
    return hours[0] if hours else ""

def get_image_row(features, code, hour):
    df = features[features["camera_code"] == code].copy()
    if hour:
        df = df[df["start_hhmm"] == hour].copy()
    df = df.sort_values(["start_hhmm", "video_filename"])
    if df.empty:
        return None
    return df.iloc[0]

def save_roi_choice(df, data):
    target_id = clean(data.get("target_id", [""])[0])
    mode = clean(data.get("mode", ["ROI"])[0])
    code = clean(data.get("video_camera_code", [""])[0])
    hour = clean(data.get("sample_hour", [""])[0])
    conf = clean(data.get("confidence", ["medium_high"])[0])
    note = clean(data.get("note", [""])[0])

    x1 = clean(data.get("roi_x1_norm", [""])[0])
    y1 = clean(data.get("roi_y1_norm", [""])[0])
    x2 = clean(data.get("roi_x2_norm", [""])[0])
    y2 = clean(data.get("roi_y2_norm", [""])[0])

    idxs = df.index[df["target_id"] == target_id].tolist()
    if not idxs:
        return df

    idx = idxs[0]

    if mode == "NOT_VISIBLE":
        status = "not_visible_after_manual_review"
        group = ""
        view = "not_visible"
        code = ""
        x1 = y1 = x2 = y2 = ""
    elif mode == "UNRESOLVED":
        status = "unresolved_after_manual_review"
        group = ""
        view = "unresolved"
        code = ""
        x1 = y1 = x2 = y2 = ""
    else:
        group = group_for_code(code)
        view = view_for_code(code)
        has_roi = all(v != "" for v in [x1, y1, x2, y2])
        status = "resolved_with_pen_roi" if has_roi and group else "needs_pen_roi_resolution"

    df.loc[idx, "selected_visual_group_id"] = group
    df.loc[idx, "view_type"] = view
    df.loc[idx, "video_camera_code"] = code
    df.loc[idx, "sample_hour"] = hour
    df.loc[idx, "roi_x1_norm"] = x1
    df.loc[idx, "roi_y1_norm"] = y1
    df.loc[idx, "roi_x2_norm"] = x2
    df.loc[idx, "roi_y2_norm"] = y2
    df.loc[idx, "resolution_status"] = status
    df.loc[idx, "confidence"] = conf
    df.loc[idx, "evidence_type"] = "manual_pen_roi_browser_interface_v78k"
    df.loc[idx, "reviewer_note"] = note
    df.loc[idx, "saved_at"] = datetime.now().isoformat(timespec="seconds")

    return df

def thumb_grid(features, title, codes, hour):
    out = [f"<h3>{esc(title)}</h3><div class='thumbgrid'>"]
    df = features[features["camera_code"].isin(codes)].copy()
    if hour:
        df = df[df["start_hhmm"] == hour].copy()
    order = {c: i for i, c in enumerate(codes)}
    df["order"] = df["camera_code"].map(order)
    df = df.sort_values(["order", "start_hhmm", "video_filename"])

    for _, r in df.head(12).iterrows():
        p = feature_img_path(r)
        out.append("<div class='thumb'>")
        out.append(f"<b>{esc(r.get('camera_code'))} | {esc(r.get('start_hhmm'))}</b>")
        if p:
            out.append(f"<img src='{image_url(p)}'>")
        out.append(f"<small>{esc(r.get('video_filename'))}</small>")
        out.append("</div>")
    out.append("</div>")
    return "\n".join(out)

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
            p = unquote(qs.get("path", [""])[0])
            if not allowed_file(p):
                self.send_response(404)
                self.end_headers()
                return
            path = Path(p)
            data = path.read_bytes()
            ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if parsed.path == "/final":
            df = init_roi_table()
            unresolved = int(df["resolution_status"].isin(UNRESOLVED_STATUSES).sum())
            if unresolved == 0:
                save_table(df, final=True)
            self.redirect("/")
            return

        self.render_main(qs)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        data = parse_qs(raw)

        df = init_roi_table()
        df = save_roi_choice(df, data)
        save_table(df, final=False)

        target_id = clean(data.get("target_id", [""])[0])
        code = clean(data.get("video_camera_code", [""])[0])
        hour = clean(data.get("sample_hour", [""])[0])

        self.redirect(f"/?target_id={quote(target_id)}&code={quote(code)}&hour={quote(hour)}")

    def render_main(self, qs):
        missing = [p for p in [FEATURES_CSV, TARGET_TEMPLATE] if not p.exists()]
        if missing:
            self.send_html("<h1>Missing files</h1>" + "".join(f"<pre>{esc(p)}</pre>" for p in missing))
            return

        features = read_csv(FEATURES_CSV)
        df = init_roi_table()

        if df.empty:
            self.send_html("<h1>ROI table empty</h1>")
            return

        selected = clean(qs.get("target_id", [df.iloc[0]["target_id"]])[0])
        if selected not in set(df["target_id"]):
            selected = df.iloc[0]["target_id"]

        row = df[df["target_id"] == selected].iloc[0]

        code = clean(qs.get("code", [choose_default_code(row)])[0])
        if code not in ALL_CODES:
            code = "c0002"

        hour = clean(qs.get("hour", [choose_default_hour(row, features)])[0])
        hours = sorted([h for h in features["start_hhmm"].dropna().unique().tolist() if clean(h)])
        if hour not in hours and hours:
            hour = hours[0]

        image_row = get_image_row(features, code, hour)
        image_path = feature_img_path(image_row) if image_row is not None else None

        unresolved = int(df["resolution_status"].isin(UNRESOLVED_STATUSES).sum())

        nav = []
        for _, r in df.iterrows():
            tid = clean(r["target_id"])
            status = clean(r["resolution_status"])
            label = f"{tid} | {clean(r['camera'])} {clean(r['pen'])} | {status}"
            cls = "done" if status not in UNRESOLVED_STATUSES else "todo"
            nav.append(f"<a class='{cls}' href='/?target_id={quote(tid)}&code={quote(code)}&hour={quote(hour)}'>{esc(label)}</a>")

        code_options = "".join(
            f"<option value='{c}' {'selected' if c == code else ''}>{c} ({group_for_code(c)} {view_for_code(c)})</option>"
            for c in ALL_CODES
        )

        hour_options = "".join(
            f"<option value='{h}' {'selected' if h == hour else ''}>{h}</option>"
            for h in hours
        )

        x1 = clean(row.get("roi_x1_norm", ""))
        y1 = clean(row.get("roi_y1_norm", ""))
        x2 = clean(row.get("roi_x2_norm", ""))
        y2 = clean(row.get("roi_y2_norm", ""))

        img_html = f"<img id='mainImg' src='{image_url(image_path)}'>" if image_path else "<p class='bad'>No image for this code/hour.</p>"

        body = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>v78k Pen ROI Annotation</title>
<style>
body {{ font-family: Arial, sans-serif; margin:0; background:#f4f4f4; }}
.header {{ background:#111; color:white; padding:14px 22px; position:sticky; top:0; z-index:20; }}
.wrap {{ display:flex; }}
.side {{ width:380px; background:white; padding:14px; height:calc(100vh - 60px); overflow:auto; border-right:1px solid #ccc; position:sticky; top:60px; }}
.main {{ flex:1; padding:22px; }}
.section {{ background:white; padding:16px; margin-bottom:20px; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,.08); }}
a {{ display:block; padding:7px; margin:4px 0; text-decoration:none; border-radius:5px; color:#111; }}
a.todo {{ background:#fff3cd; }}
a.done {{ background:#e4ffe4; }}
.stage {{ position:relative; display:inline-block; max-width:100%; border:2px solid #333; background:#222; }}
.stage img {{ max-width:100%; display:block; }}
.stage canvas {{ position:absolute; left:0; top:0; cursor:crosshair; }}
.bad {{ color:#a00000; font-weight:bold; }}
button {{ padding:10px 16px; font-weight:bold; }}
select, textarea {{ width:100%; padding:8px; margin:6px 0 12px 0; }}
table td {{ padding:5px 10px; border-bottom:1px solid #ddd; }}
.thumbgrid {{ display:flex; flex-wrap:wrap; gap:10px; }}
.thumb {{ width:31%; min-width:220px; background:#fafafa; border:1px solid #ccc; padding:8px; border-radius:8px; }}
.thumb img {{ width:100%; }}
small {{ word-break:break-all; }}
</style>
</head>
<body>
<div class="header"><h2>v78k Pen ROI Annotation Interface</h2></div>
<div class="wrap">
<div class="side">
<h3>Progress</h3>
<p><b>Unresolved:</b> {unresolved} / {len(df)}</p>
<p><small>Working CSV:<br>{esc(WORKING_CSV)}</small></p>
{"<p><a class='done' href='/final'>SAVE FINAL CSV</a></p>" if unresolved == 0 else "<p>FINAL save için unresolved 0 olmalı.</p>"}
<h3>Targets</h3>
{''.join(nav)}
</div>

<div class="main">

<div class="section">
<h2>Current target: {esc(selected)}</h2>
<table>
<tr><td>Date</td><td>{esc(row.get("date",""))}</td></tr>
<tr><td>Camera</td><td>{esc(row.get("camera",""))}</td></tr>
<tr><td>Pen / area</td><td><b>{esc(row.get("pen",""))}</b></td></tr>
<tr><td>Status</td><td>{esc(row.get("resolution_status",""))}</td></tr>
<tr><td>Saved code</td><td>{esc(row.get("video_camera_code",""))}</td></tr>
<tr><td>Saved ROI</td><td>{esc(x1)}, {esc(y1)}, {esc(x2)}, {esc(y2)}</td></tr>
<tr><td>Note</td><td>{esc(row.get("reviewer_note",""))}</td></tr>
</table>
<p><b>Yapılacak iş:</b> Bu target'ın pen/bölgesini görüntü üzerinde rectangle ile işaretle. Emin değilsen UNRESOLVED seç.</p>
</div>

<div class="section">
<form method="GET">
<input type="hidden" name="target_id" value="{esc(selected)}">
<label>Video camera code / view</label>
<select name="code" onchange="this.form.submit()">{code_options}</select>
<label>Sample hour</label>
<select name="hour" onchange="this.form.submit()">{hour_options}</select>
</form>
</div>

<div class="section">
<h3>Draw ROI on selected image</h3>
<p>Mouse ile görüntü üzerinde sürükleyerek rectangle çiz. Sonra aşağıdan <b>ROI_SELECTED</b> moduyla Save de.</p>

<div class="stage" id="stage">
{img_html}
<canvas id="canvas"></canvas>
</div>

<form method="POST" id="saveForm">
<input type="hidden" name="target_id" value="{esc(selected)}">
<input type="hidden" name="video_camera_code" value="{esc(code)}">
<input type="hidden" name="sample_hour" value="{esc(hour)}">

<input type="hidden" name="roi_x1_norm" id="x1" value="{esc(x1)}">
<input type="hidden" name="roi_y1_norm" id="y1" value="{esc(y1)}">
<input type="hidden" name="roi_x2_norm" id="x2" value="{esc(x2)}">
<input type="hidden" name="roi_y2_norm" id="y2" value="{esc(y2)}">

<label>Mode</label>
<select name="mode">
<option value="ROI">ROI_SELECTED</option>
<option value="NOT_VISIBLE">NOT_VISIBLE</option>
<option value="UNRESOLVED">UNRESOLVED</option>
</select>

<label>Confidence</label>
<select name="confidence">
<option value="high">high</option>
<option value="medium_high" selected>medium_high</option>
<option value="medium">medium</option>
<option value="low">low</option>
<option value="none">none</option>
</select>

<label>Reviewer note</label>
<textarea name="note" rows="4">{esc(row.get("reviewer_note",""))}</textarea>

<button type="button" onclick="clearRect()">Clear ROI</button>
<button type="submit">Save this target</button>
</form>
</div>

<div class="section">
{thumb_grid(features, "GROUP_A reference thumbnails: c0002 / c0000 / c0100", GROUP_A_CODES, hour)}
</div>

<div class="section">
{thumb_grid(features, "GROUP_B reference thumbnails: c0001 / c0003 / c0101", GROUP_B_CODES, hour)}
</div>

</div>
</div>

<script>
let img = document.getElementById('mainImg');
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let dragging = false;
let startX = 0, startY = 0;
let rect = null;

function getHiddenRect() {{
    let x1 = parseFloat(document.getElementById('x1').value);
    let y1 = parseFloat(document.getElementById('y1').value);
    let x2 = parseFloat(document.getElementById('x2').value);
    let y2 = parseFloat(document.getElementById('y2').value);
    if ([x1,y1,x2,y2].some(v => isNaN(v))) return null;
    return {{x1:x1, y1:y1, x2:x2, y2:y2}};
}}

function resizeCanvas() {{
    if (!img) return;
    let r = img.getBoundingClientRect();
    canvas.width = r.width;
    canvas.height = r.height;
    canvas.style.width = r.width + 'px';
    canvas.style.height = r.height + 'px';
    draw();
}}

function draw() {{
    ctx.clearRect(0,0,canvas.width,canvas.height);
    if (!rect) rect = getHiddenRect();
    if (!rect) return;
    let x = rect.x1 * canvas.width;
    let y = rect.y1 * canvas.height;
    let w = (rect.x2 - rect.x1) * canvas.width;
    let h = (rect.y2 - rect.y1) * canvas.height;
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 4;
    ctx.strokeRect(x,y,w,h);
    ctx.fillStyle = 'rgba(255,0,0,0.18)';
    ctx.fillRect(x,y,w,h);
}}

function normPoint(evt) {{
    let r = canvas.getBoundingClientRect();
    let x = Math.max(0, Math.min(1, (evt.clientX - r.left) / r.width));
    let y = Math.max(0, Math.min(1, (evt.clientY - r.top) / r.height));
    return {{x:x, y:y}};
}}

canvas.addEventListener('mousedown', function(evt) {{
    dragging = true;
    let p = normPoint(evt);
    startX = p.x;
    startY = p.y;
    rect = {{x1:startX, y1:startY, x2:startX, y2:startY}};
    draw();
}});

canvas.addEventListener('mousemove', function(evt) {{
    if (!dragging) return;
    let p = normPoint(evt);
    rect = {{
        x1: Math.min(startX, p.x),
        y1: Math.min(startY, p.y),
        x2: Math.max(startX, p.x),
        y2: Math.max(startY, p.y)
    }};
    draw();
}});

canvas.addEventListener('mouseup', function(evt) {{
    dragging = false;
    if (!rect) return;
    document.getElementById('x1').value = rect.x1.toFixed(6);
    document.getElementById('y1').value = rect.y1.toFixed(6);
    document.getElementById('x2').value = rect.x2.toFixed(6);
    document.getElementById('y2').value = rect.y2.toFixed(6);
    draw();
}});

function clearRect() {{
    rect = null;
    document.getElementById('x1').value = '';
    document.getElementById('y1').value = '';
    document.getElementById('x2').value = '';
    document.getElementById('y2').value = '';
    draw();
}}

if (img) {{
    img.onload = resizeCanvas;
    window.addEventListener('resize', resizeCanvas);
    setTimeout(resizeCanvas, 500);
}}
</script>

</body>
</html>
"""
        self.send_html(body)

if __name__ == "__main__":
    print("Serving v78k Pen ROI interface on http://0.0.0.0:8548")
    ThreadingHTTPServer(("0.0.0.0", 8548), Handler).serve_forever()
