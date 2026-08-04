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

FEATURES_CSV = V78I / "v78i_frame_features.csv"
TARGETS_CSV = V78J / "v78j_excel_visual_mapping_targets.csv"
TEMPLATE_CSV = V78J / "v78j_MANUAL_FILL_visual_stream_resolution.csv"

WORKING_CSV = V78J / "v78j_MANUAL_FILLED_visual_stream_resolution_WORKING.csv"
FINAL_CSV = V78J / "v78j_MANUAL_FILLED_visual_stream_resolution_FINAL.csv"
CONFIG_CSV = FULL / "config" / "visual_stream_resolution_v78j.csv"
SUMMARY_CSV = V78J / "v78j_interface_save_summary.csv"
BACKUP_DIR = V78J / "interface_backups"

BACKUP_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_CSV.parent.mkdir(parents=True, exist_ok=True)

GROUP_INFO = {
    "GROUP_A": {
        "label": "GROUP_A: TLC1 anchor area",
        "desc": "c0002/c0000 side view, c0100 top view. TLC1 anchor burada.",
        "primary": "c0002", "secondary": "c0000", "top": "c0100",
        "codes": ["c0000", "c0002", "c0100"],
    },
    "GROUP_B": {
        "label": "GROUP_B: other visual area",
        "desc": "c0001/c0003 side view, c0101 top view.",
        "primary": "c0001", "secondary": "c0003", "top": "c0101",
        "codes": ["c0001", "c0003", "c0101"],
    },
    "NOT_VISIBLE": {
        "label": "NOT_VISIBLE: this target is not visible",
        "desc": "Bu pen/alan mevcut streamlerde güvenilir görünmüyor.",
        "primary": "", "secondary": "", "top": "", "codes": [],
    },
    "UNRESOLVED": {
        "label": "UNRESOLVED: not sure",
        "desc": "Emin değilsen yanlış mapping üretmemek için bunu seç.",
        "primary": "", "secondary": "", "top": "", "codes": [],
    },
}

UNRESOLVED_STATUSES = {"", "needs_manual_visual_resolution", "unresolved_after_manual_review"}


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


def load_resolution():
    if WORKING_CSV.exists():
        return read_csv(WORKING_CSV)
    if FINAL_CSV.exists():
        return read_csv(FINAL_CSV)
    return read_csv(TEMPLATE_CSV)


def save_resolution(df, final=False):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BACKUP_DIR / f"visual_stream_resolution_backup_{ts}.csv"

    write_csv(df, backup)
    write_csv(df, WORKING_CSV)
    write_csv(df, CONFIG_CSV)

    if final:
        write_csv(df, FINAL_CSV)

    unresolved = int(df["resolution_status"].isin(UNRESOLVED_STATUSES).sum())
    resolved = int((df["resolution_status"] == "resolved_by_manual_visual_grouping").sum())
    locked = int(df["resolution_status"].astype(str).str.contains("locked", na=False).sum())
    not_visible = int((df["resolution_status"] == "not_visible_after_manual_review").sum())

    summary = pd.DataFrame([{
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "rows": len(df),
        "resolved_count": resolved,
        "locked_count": locked,
        "not_visible_count": not_visible,
        "unresolved_count": unresolved,
        "working_csv": str(WORKING_CSV),
        "final_csv": str(FINAL_CSV) if final else "",
        "config_csv": str(CONFIG_CSV),
        "backup_csv": str(backup),
        "final_saved": bool(final),
        "ready_for_v78k_validation": bool(final and unresolved == 0),
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


def img_src(path):
    return "/img?path=" + quote(str(path))


def feature_img_path(row):
    p = Path(clean(row.get("frame_image", "")))
    if p.exists():
        return p
    alt = V78I / "visual_stream_groups" / p.name
    if alt.exists():
        return alt
    return None


def apply_choice(df, target_id, group_id, view_type, confidence, note):
    idxs = df.index[df["target_id"] == target_id].tolist()
    if not idxs:
        return df
    idx = idxs[0]
    info = GROUP_INFO[group_id]

    if group_id in {"GROUP_A", "GROUP_B"}:
        status = "resolved_by_manual_visual_grouping"
        primary = info["primary"]
        secondary = info["secondary"]
        top = info["top"]
    elif group_id == "NOT_VISIBLE":
        status = "not_visible_after_manual_review"
        view_type = "not_visible"
        primary = secondary = top = ""
    else:
        status = "unresolved_after_manual_review"
        view_type = "unresolved"
        primary = secondary = top = ""

    df.loc[idx, "selected_visual_group_id"] = group_id
    df.loc[idx, "selected_view_type"] = view_type
    df.loc[idx, "primary_video_camera_code"] = primary
    df.loc[idx, "secondary_video_camera_code"] = secondary
    df.loc[idx, "top_view_camera_code"] = top
    df.loc[idx, "resolution_status"] = status
    df.loc[idx, "confidence"] = confidence
    df.loc[idx, "evidence_type"] = "browser_visual_stream_grouping_interface_v78j"
    df.loc[idx, "reviewer_note"] = note
    return df


def image_cards(title, rows, max_cards=18):
    out = [f"<h3>{esc(title)}</h3><div class='grid'>"]
    if rows.empty:
        out.append("<p>No images.</p></div>")
        return "\n".join(out)

    for _, r in rows.head(max_cards).iterrows():
        p = feature_img_path(r)
        code = clean(r.get("camera_code", ""))
        tlc = clean(r.get("tlc_camera", ""))
        hour = clean(r.get("start_hhmm", ""))
        fname = clean(r.get("video_filename", ""))
        out.append("<div class='card'>")
        out.append(f"<b>{esc(code)} {esc(tlc)} | {esc(hour)}</b>")
        if p:
            out.append(f"<img src='{img_src(p)}'>")
        else:
            out.append("<p class='bad'>Image not found</p>")
        out.append(f"<small>{esc(fname)}</small>")
        out.append("</div>")
    out.append("</div>")
    return "\n".join(out)


def contact_sheets(target_id, targets):
    row = targets[targets["target_id"] == target_id]
    if row.empty:
        return ""
    sheets = clean(row.iloc[0].get("visual_contact_sheet", ""))
    if not sheets:
        return ""
    cards = ["<h3>Target contact sheet</h3><div class='grid'>"]
    found = 0
    for item in sheets.split(";"):
        p = Path(clean(item))
        if p.exists():
            found += 1
            cards.append("<div class='widecard'>")
            cards.append(f"<b>{esc(p.name)}</b>")
            cards.append(f"<img src='{img_src(p)}'>")
            cards.append("</div>")
    cards.append("</div>")
    return "\n".join(cards) if found else ""


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
            ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if parsed.path == "/final":
            df = load_resolution()
            unresolved = int(df["resolution_status"].isin(UNRESOLVED_STATUSES).sum())
            if unresolved == 0:
                save_resolution(df, final=True)
            self.redirect("/")
            return

        self.render_main(qs)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        data = parse_qs(raw)

        target_id = clean(data.get("target_id", [""])[0])
        group_id = clean(data.get("group_id", ["UNRESOLVED"])[0])
        view_type = clean(data.get("view_type", ["unresolved"])[0])
        confidence = clean(data.get("confidence", ["none"])[0])
        note = clean(data.get("note", [""])[0])

        if group_id not in GROUP_INFO:
            group_id = "UNRESOLVED"

        df = load_resolution()
        df = apply_choice(df, target_id, group_id, view_type, confidence, note)
        save_resolution(df, final=False)

        self.redirect("/?target_id=" + quote(target_id))

    def render_main(self, qs):
        missing = [p for p in [FEATURES_CSV, TARGETS_CSV, TEMPLATE_CSV] if not p.exists()]
        if missing:
            self.send_html("<h1>Missing files</h1>" + "".join(f"<pre>{esc(p)}</pre>" for p in missing))
            return

        df = load_resolution()
        features = read_csv(FEATURES_CSV)
        targets = read_csv(TARGETS_CSV)

        if df.empty:
            self.send_html("<h1>Resolution table empty</h1>")
            return

        selected = clean(qs.get("target_id", [df.iloc[0]["target_id"]])[0])
        if selected not in set(df["target_id"]):
            selected = df.iloc[0]["target_id"]

        hour = clean(qs.get("hour", ["ALL"])[0])
        hours = sorted([h for h in features["start_hhmm"].dropna().unique().tolist() if clean(h)])
        if hour != "ALL" and hour not in hours:
            hour = "ALL"

        row = df[df["target_id"] == selected].iloc[0]
        unresolved = int(df["resolution_status"].isin(UNRESOLVED_STATUSES).sum())

        filtered = features.copy()
        if hour != "ALL":
            filtered = filtered[filtered["start_hhmm"] == hour]

        ref = filtered[(filtered["video_type"] == "friendly") & (filtered["tlc_camera"] == "TLC1")].sort_values(["start_hhmm", "video_filename"])

        a_codes = GROUP_INFO["GROUP_A"]["codes"]
        b_codes = GROUP_INFO["GROUP_B"]["codes"]
        group_a = filtered[filtered["camera_code"].isin(a_codes)].copy()
        group_b = filtered[filtered["camera_code"].isin(b_codes)].copy()

        nav = []
        for _, r in df.iterrows():
            tid = clean(r["target_id"])
            status = clean(r["resolution_status"])
            label = f"{tid} | {clean(r['camera'])} {clean(r['pen'])} | {status}"
            cls = "done" if status not in UNRESOLVED_STATUSES else "todo"
            nav.append(f"<a class='{cls}' href='/?target_id={quote(tid)}&hour={quote(hour)}'>{esc(label)}</a>")

        hour_links = ["<a href='/?target_id=%s&hour=ALL'>ALL</a>" % quote(selected)]
        for h in hours:
            hour_links.append("<a href='/?target_id=%s&hour=%s'>%s</a>" % (quote(selected), quote(h), esc(h)))

        current_group = clean(row.get("selected_visual_group_id", ""))
        if current_group not in GROUP_INFO:
            current_group = "UNRESOLVED"

        body = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>v78j Visual Stream Grouping</title>
<style>
body {{ font-family: Arial, sans-serif; margin:0; background:#f4f4f4; }}
.header {{ background:#111; color:white; padding:16px 24px; position:sticky; top:0; z-index:10; }}
.wrap {{ display:flex; }}
.side {{ width:360px; background:white; padding:16px; height:calc(100vh - 70px); overflow:auto; border-right:1px solid #ccc; position:sticky; top:70px; }}
.main {{ flex:1; padding:24px; }}
.section {{ background:white; padding:16px; margin-bottom:20px; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,.08); }}
.grid {{ display:flex; flex-wrap:wrap; gap:12px; }}
.card {{ width:31%; min-width:260px; background:#fafafa; border:1px solid #ccc; border-radius:8px; padding:8px; }}
.widecard {{ width:48%; background:#fafafa; border:1px solid #ccc; border-radius:8px; padding:8px; }}
img {{ width:100%; border-radius:4px; margin-top:6px; }}
a {{ display:block; padding:7px; margin:4px 0; text-decoration:none; border-radius:5px; color:#111; }}
a.todo {{ background:#fff3cd; }}
a.done {{ background:#e4ffe4; }}
.bad {{ color:#a00000; font-weight:bold; }}
.rowtable td {{ padding:5px 10px; border-bottom:1px solid #ddd; }}
button {{ padding:10px 16px; font-weight:bold; }}
select, textarea {{ width:100%; padding:8px; margin:6px 0 12px 0; }}
.hourlinks a {{ display:inline-block; background:#eee; margin-right:6px; }}
</style>
</head>
<body>
<div class="header">
<h2>v78j Visual Stream Grouping Interface</h2>
</div>
<div class="wrap">
<div class="side">
<h3>Progress</h3>
<p><b>Unresolved:</b> {unresolved} / {len(df)}</p>
<p><b>Working CSV:</b><br><small>{esc(WORKING_CSV)}</small></p>
{"<p><a href='/final' class='done'>SAVE FINAL CSV</a></p>" if unresolved == 0 else "<p>FINAL save için unresolved 0 olmalı.</p>"}
<h3>Targets</h3>
{''.join(nav)}
</div>
<div class="main">
<div class="section">
<h2>Current target: {esc(selected)}</h2>
<table class="rowtable">
<tr><td>Date</td><td>{esc(row.get("date",""))}</td></tr>
<tr><td>Camera</td><td>{esc(row.get("camera",""))}</td></tr>
<tr><td>Pen</td><td>{esc(row.get("pen",""))}</td></tr>
<tr><td>Unresolved windows</td><td>{esc(row.get("unresolved_windows",""))}</td></tr>
<tr><td>Candidate codes</td><td>{esc(row.get("candidate_video_camera_codes",""))}</td></tr>
<tr><td>Current group</td><td>{esc(row.get("selected_visual_group_id",""))}</td></tr>
<tr><td>Status</td><td>{esc(row.get("resolution_status",""))}</td></tr>
<tr><td>Note</td><td>{esc(row.get("reviewer_note",""))}</td></tr>
</table>
</div>

<div class="section">
<h3>Hour filter</h3>
<div class="hourlinks">{''.join(hour_links)}</div>
</div>

<div class="section">
<h3>Decision guide</h3>
<p><b>GROUP_A</b>: TLC1 anchor tarafı. c0002/c0000 side, c0100 top.</p>
<p><b>GROUP_B</b>: diğer görsel alan. c0001/c0003 side, c0101 top.</p>
<p><b>NOT_VISIBLE</b>: bu target görüntülerde görünmüyor.</p>
<p><b>UNRESOLVED</b>: emin değilsen.</p>

<form method="POST">
<input type="hidden" name="target_id" value="{esc(selected)}">

<label>Visual group</label>
<select name="group_id">
{''.join(f'<option value="{gid}" {"selected" if gid == current_group else ""}>{esc(info["label"])}</option>' for gid, info in GROUP_INFO.items())}
</select>

<label>View type</label>
<select name="view_type">
{''.join(f'<option value="{v}" {"selected" if v == clean(row.get("selected_view_type","")) else ""}>{v}</option>' for v in ["side","top","side_and_top","not_visible","unresolved"])}
</select>

<label>Confidence</label>
<select name="confidence">
{''.join(f'<option value="{v}" {"selected" if v == clean(row.get("confidence","")) else ""}>{v}</option>' for v in ["high","medium_high","medium","low","none"])}
</select>

<label>Reviewer note</label>
<textarea name="note" rows="4">{esc(row.get("reviewer_note",""))}</textarea>

<button type="submit">Save this target</button>
</form>
</div>

<div class="section">
{contact_sheets(selected, targets)}
</div>

<div class="section">
{image_cards("TLC1 reference / anchor", ref, max_cards=12)}
</div>

<div class="section">
{image_cards("GROUP_A: c0000 / c0002 / c0100", group_a, max_cards=24)}
</div>

<div class="section">
{image_cards("GROUP_B: c0001 / c0003 / c0101", group_b, max_cards=24)}
</div>

</div>
</div>
</body>
</html>
"""
        self.send_html(body)


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8547
    print(f"Serving v78j interface on http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
