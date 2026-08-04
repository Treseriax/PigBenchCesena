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

K1 = FULL / "outputs" / "v78k1_date_specific_excel_pen_alias_resolver" / "Full_Unibo_Date_Specific_Excel_Pen_Alias_Resolver"
K1_TARGETS = K1 / "v78k1_v78j_targets_resolved_with_aliases.csv"

K3 = FULL / "outputs" / "v78k3_tlc_to_c_code_mapping_resolver" / "Full_Unibo_TLC_to_CCode_Mapping_Resolver"
K3_MATRIX = K3 / "v78k3_tlc_to_c_code_mapping_matrix.csv"

V78I = FULL / "outputs" / "v78i_visual_stream_grouping_audit" / "Full_Unibo_Visual_Stream_Grouping_Audit"
FEATURES_CSV = V78I / "v78i_frame_features.csv"

OUT = FULL / "outputs" / "v78k4_pattern_assisted_tlc_code_mapping"
OUT.mkdir(parents=True, exist_ok=True)

WORKING = OUT / "v78k4_manual_tlc_to_c_code_mapping_WORKING.csv"
FINAL = OUT / "v78k4_manual_tlc_to_c_code_mapping_FINAL.csv"
SUMMARY = OUT / "v78k4_interface_summary.csv"
CONFIG = FULL / "config" / "tlc_to_c_code_mapping_v78k4.csv"
CONFIG.parent.mkdir(parents=True, exist_ok=True)

ALL_CODES = ["c0000", "c0001", "c0002", "c0003", "c0100", "c0101"]
GROUP_A = ["c0002", "c0000", "c0100"]
GROUP_B = ["c0001", "c0003", "c0101"]

UNRESOLVED_VALUES = {"", "UNRESOLVED", "NEEDS_EXTERNAL_CONFIRMATION"}

def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s

def esc(x):
    return html.escape("" if x is None else str(x))

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

def code_group(code):
    if code in GROUP_A:
        return "GROUP_A"
    if code in GROUP_B:
        return "GROUP_B"
    return ""

def code_view(code):
    if code in ["c0100", "c0101"]:
        return "top"
    if code in ["c0000", "c0001", "c0002", "c0003"]:
        return "side"
    return ""

def init_table():
    if WORKING.exists():
        return read_csv(WORKING)
    if FINAL.exists():
        return read_csv(FINAL)

    matrix = read_csv(K3_MATRIX)
    rows = []

    for _, r in matrix.iterrows():
        tlc = clean(r.get("tlc_camera"))
        date = clean(r.get("date"))

        if tlc == "TLC1":
            decision = "RESOLVED"
            group = "GROUP_A"
            primary = "c0002"
            secondary = "c0000"
            top = "c0100"
            conf = "high"
            note = "Locked from TLC1 visual anchor."
        else:
            decision = "UNRESOLVED"
            group = ""
            primary = ""
            secondary = ""
            top = ""
            conf = ""
            note = ""

        rows.append({
            "date": date,
            "tlc_camera": tlc,
            "resolved_pens_from_excel": clean(r.get("resolved_pens_from_excel")),
            "target_ids": clean(r.get("target_ids")),
            "manual_decision": decision,
            "manual_visual_group": group,
            "manual_primary_side_code": primary,
            "manual_secondary_side_code": secondary,
            "manual_top_view_code": top,
            "manual_confidence": conf,
            "manual_evidence_note": note,
            "saved_at": "",
        })

    return pd.DataFrame(rows)

def save_table(df, final=False):
    now = datetime.now().isoformat(timespec="seconds")
    df = df.copy()
    write_csv(df, WORKING)
    write_csv(df, CONFIG)
    if final:
        write_csv(df, FINAL)

    unresolved = int(df["manual_decision"].isin(UNRESOLVED_VALUES).sum())
    resolved = int((df["manual_decision"] == "RESOLVED").sum())
    needs_external = int((df["manual_decision"] == "NEEDS_EXTERNAL_CONFIRMATION").sum())
    not_visible = int((df["manual_decision"] == "NOT_VISIBLE").sum())

    summary = pd.DataFrame([{
        "saved_at": now,
        "rows": len(df),
        "resolved": resolved,
        "not_visible": not_visible,
        "needs_external_confirmation": needs_external,
        "unresolved": unresolved,
        "working_csv": str(WORKING),
        "final_csv": str(FINAL) if final else "",
        "config_csv": str(CONFIG),
        "final_saved": bool(final),
    }])
    write_csv(summary, SUMMARY)
    return summary

def apply_choice(df, data):
    tlc = clean(data.get("tlc_camera", [""])[0])
    decision = clean(data.get("manual_decision", ["UNRESOLVED"])[0])
    primary = clean(data.get("manual_primary_side_code", [""])[0])
    secondary = clean(data.get("manual_secondary_side_code", [""])[0])
    top = clean(data.get("manual_top_view_code", [""])[0])
    conf = clean(data.get("manual_confidence", [""])[0])
    note = clean(data.get("manual_evidence_note", [""])[0])

    idxs = df.index[df["tlc_camera"] == tlc].tolist()
    if not idxs:
        return df

    idx = idxs[0]

    if decision == "RESOLVED":
        group = code_group(primary)
        if not secondary:
            secondary = "c0000" if primary == "c0002" else ("c0003" if primary == "c0001" else "")
        if not top:
            top = "c0100" if group == "GROUP_A" else ("c0101" if group == "GROUP_B" else "")
    else:
        group = ""
        primary = ""
        secondary = ""
        top = ""

    df.loc[idx, "manual_decision"] = decision
    df.loc[idx, "manual_visual_group"] = group
    df.loc[idx, "manual_primary_side_code"] = primary
    df.loc[idx, "manual_secondary_side_code"] = secondary
    df.loc[idx, "manual_top_view_code"] = top
    df.loc[idx, "manual_confidence"] = conf
    df.loc[idx, "manual_evidence_note"] = note
    df.loc[idx, "saved_at"] = datetime.now().isoformat(timespec="seconds")

    return df

def contact_sheets_for_tlc(tlc, targets):
    rows = targets[targets["camera"] == tlc].copy()
    out = ["<h3>Excel target contact sheets</h3><div class='grid'>"]
    found = 0

    for _, r in rows.iterrows():
        p = Path(clean(r.get("visual_contact_sheet", "")))
        if p.exists():
            found += 1
            title = f"{clean(r.get('camera'))} {clean(r.get('resolved_room_pen')) or clean(r.get('pen'))}"
            out.append("<div class='widecard'>")
            out.append(f"<b>{esc(title)}</b><br><small>{esc(p.name)}</small>")
            out.append(f"<img src='{image_url(p)}'>")
            out.append("</div>")

    out.append("</div>")
    return "\n".join(out) if found else "<p>No contact sheets found for this TLC.</p>"

def code_cards(features, codes, hour):
    f = features[features["camera_code"].isin(codes)].copy()
    if hour != "ALL":
        f = f[f["start_hhmm"] == hour].copy()
    f["order"] = f["camera_code"].map({c:i for i,c in enumerate(codes)})
    f = f.sort_values(["order", "start_hhmm", "video_filename"])

    out = ["<div class='grid'>"]
    if f.empty:
        out.append("<p>No images.</p>")
    for _, r in f.head(18).iterrows():
        p = feature_img_path(r)
        code = clean(r.get("camera_code"))
        h = clean(r.get("start_hhmm"))
        fname = clean(r.get("video_filename"))
        out.append("<div class='card'>")
        out.append(f"<b>{esc(code)} | {esc(h)} | {esc(code_group(code))} {esc(code_view(code))}</b>")
        if p:
            out.append(f"<img src='{image_url(p)}'>")
        else:
            out.append("<p class='bad'>Image not found</p>")
        out.append(f"<small>{esc(fname)}</small>")
        out.append("</div>")
    out.append("</div>")
    return "\n".join(out)

def friendly_cards(features, hour):
    f = features[(features["video_type"] == "friendly") & (features["tlc_camera"] == "TLC1")].copy()
    if hour != "ALL":
        f = f[f["start_hhmm"] == hour].copy()
    f = f.sort_values(["start_hhmm", "video_filename"])
    out = ["<div class='grid'>"]
    for _, r in f.head(12).iterrows():
        p = feature_img_path(r)
        out.append("<div class='card ref'>")
        out.append(f"<b>TLC1 reference | {esc(r.get('start_hhmm'))}</b>")
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
            df = init_table()
            save_table(df, final=True)
            self.redirect("/")
            return

        self.render_main(qs)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        data = parse_qs(raw)

        df = init_table()
        df = apply_choice(df, data)
        save_table(df, final=False)

        tlc = clean(data.get("tlc_camera", ["TLC1"])[0])
        self.redirect("/?tlc_camera=" + quote(tlc))

    def render_main(self, qs):
        features = read_csv(FEATURES_CSV)
        targets = read_csv(K1_TARGETS)
        df = init_table()

        if df.empty:
            self.send_html("<h1>Empty mapping table</h1>")
            return

        selected = clean(qs.get("tlc_camera", [df.iloc[0]["tlc_camera"]])[0])
        if selected not in set(df["tlc_camera"]):
            selected = df.iloc[0]["tlc_camera"]

        hours = sorted([h for h in features["start_hhmm"].dropna().unique().tolist() if clean(h)])
        hour = clean(qs.get("hour", ["09:00" if "09:00" in hours else "ALL"])[0])
        if hour != "ALL" and hour not in hours:
            hour = "ALL"

        row = df[df["tlc_camera"] == selected].iloc[0]
        unresolved = int(df["manual_decision"].isin(UNRESOLVED_VALUES).sum())

        nav = []
        for _, r in df.iterrows():
            tlc = clean(r["tlc_camera"])
            dec = clean(r["manual_decision"])
            cls = "done" if dec == "RESOLVED" else "todo"
            label = f"{tlc} | {clean(r['resolved_pens_from_excel'])} | {dec}"
            nav.append(f"<a class='{cls}' href='/?tlc_camera={quote(tlc)}&hour={quote(hour)}'>{esc(label)}</a>")

        hour_links = [f"<a href='/?tlc_camera={quote(selected)}&hour=ALL'>ALL</a>"]
        for h in hours:
            hour_links.append(f"<a href='/?tlc_camera={quote(selected)}&hour={quote(h)}'>{esc(h)}</a>")

        primary_options = "".join(
            f"<option value='{c}' {'selected' if c == clean(row.get('manual_primary_side_code')) else ''}>{c} ({code_group(c)} {code_view(c)})</option>"
            for c in [""] + ALL_CODES
        )

        decision_options = "".join(
            f"<option value='{d}' {'selected' if d == clean(row.get('manual_decision')) else ''}>{d}</option>"
            for d in ["RESOLVED", "NOT_VISIBLE", "UNRESOLVED", "NEEDS_EXTERNAL_CONFIRMATION"]
        )

        confidence_options = "".join(
            f"<option value='{c}' {'selected' if c == clean(row.get('manual_confidence')) else ''}>{c}</option>"
            for c in ["", "high", "medium_high", "medium", "low"]
        )

        body = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>v78k4 Pattern-Assisted TLC Mapping</title>
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
.grid {{ display:flex; flex-wrap:wrap; gap:12px; }}
.card {{ width:31%; min-width:260px; background:#fafafa; border:1px solid #ccc; border-radius:8px; padding:8px; }}
.widecard {{ width:48%; min-width:360px; background:#fafafa; border:1px solid #ccc; border-radius:8px; padding:8px; }}
.card img, .widecard img {{ width:100%; border-radius:4px; margin-top:6px; }}
.ref {{ border:4px solid #0a8a0a; }}
select, textarea {{ width:100%; padding:8px; margin:6px 0 12px 0; }}
button {{ padding:10px 16px; font-weight:bold; }}
.small, small {{ word-break:break-all; color:#555; }}
.hourlinks a {{ display:inline-block; background:#eee; margin-right:6px; }}
.bad {{ color:#a00000; }}
</style>
</head>
<body>
<div class="header"><h2>v78k4 Pattern-Assisted TLC-to-c-code Mapping</h2></div>

<div class="wrap">
<div class="side">
<h3>Progress</h3>
<p><b>Unresolved / external:</b> {unresolved} / {len(df)}</p>
<p><small>Working CSV:<br>{esc(WORKING)}</small></p>
<p><a class="done" href="/final">SAVE CURRENT AS FINAL</a></p>
<h3>TLC cameras</h3>
{''.join(nav)}
</div>

<div class="main">

<div class="section">
<h2>{esc(selected)} → pens {esc(row.get("resolved_pens_from_excel"))}</h2>
<p><b>Goal:</b> Bu TLC kamerasının encoded c-code streamlerinden hangisine benzediğini bul.</p>
<p><b>Not:</b> Bu pen-level ROI değil. Sadece camera-level stream mapping.</p>
</div>

<div class="section">
<h3>Hour filter</h3>
<div class="hourlinks">{''.join(hour_links)}</div>
</div>

<div class="section">
<h3>Manual decision</h3>
<form method="POST">
<input type="hidden" name="tlc_camera" value="{esc(selected)}">

<label>Decision</label>
<select name="manual_decision">{decision_options}</select>

<label>Primary side code</label>
<select name="manual_primary_side_code">{primary_options}</select>

<label>Confidence</label>
<select name="manual_confidence">{confidence_options}</select>

<label>Evidence note</label>
<textarea name="manual_evidence_note" rows="4">{esc(row.get("manual_evidence_note"))}</textarea>

<button type="submit">Save this TLC mapping</button>
</form>
</div>

<div class="section">
{contact_sheets_for_tlc(selected, targets)}
</div>

<div class="section">
<h3>TLC1 reference anchor</h3>
{friendly_cards(features, hour)}
</div>

<div class="section">
<h3>GROUP_A candidates: c0002 / c0000 / c0100</h3>
{code_cards(features, GROUP_A, hour)}
</div>

<div class="section">
<h3>GROUP_B candidates: c0001 / c0003 / c0101</h3>
{code_cards(features, GROUP_B, hour)}
</div>

</div>
</div>
</body>
</html>
"""
        self.send_html(body)

if __name__ == "__main__":
    print("Serving v78k4 pattern-assisted TLC mapping on http://0.0.0.0:8552")
    ThreadingHTTPServer(("0.0.0.0", 8552), Handler).serve_forever()
