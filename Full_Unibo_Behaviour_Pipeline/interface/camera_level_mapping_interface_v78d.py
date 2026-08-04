from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, quote, unquote_plus
import argparse
import csv
import html
import mimetypes


ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

PKG = FULL / "outputs" / "v78c_camera_code_mapping_resolver" / "Full_Unibo_Camera_Code_Mapping_Resolver"
TEMPLATE = FULL / "config" / "camera_code_mapping_TEMPLATE_TO_FILL.csv"
PEN_CONFIG = FULL / "config" / "camera_code_mapping.csv"
CAMERA_CONFIG = FULL / "config" / "camera_level_code_mapping.csv"
MANIFEST = PKG / "v78c_visual_atlas_manifest.csv"

ALLOWED_BASES = [
    PKG.resolve(),
    (FULL / "config").resolve(),
]

CAMERA_COLUMNS = [
    "date",
    "camera",
    "video_camera_code",
    "candidate_video_camera_codes",
    "candidate_pen_groups",
    "manual_note",
]

PEN_COLUMNS = [
    "date",
    "camera",
    "pen",
    "video_camera_code",
    "candidate_video_camera_codes",
    "unresolved_windows",
    "candidate_video_examples",
    "visual_contact_sheet",
    "manual_instruction",
]


def clean(x):
    return "" if x is None else str(x).strip()


def split_semicolon(x):
    x = clean(x)
    if not x:
        return []
    return [p.strip() for p in x.split(";") if p.strip()]


def read_csv_rows(path):
    rows = []
    if not path.exists():
        return rows

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: clean(v) for k, v in row.items()})

    return rows


def write_csv_rows(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)

    final_cols = list(columns)
    for row in rows:
        for key in row.keys():
            if key not in final_cols:
                final_cols.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=final_cols, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_template_rows():
    return read_csv_rows(TEMPLATE)


def load_manifest_rows():
    return read_csv_rows(MANIFEST)


def build_camera_rows():
    template = load_template_rows()

    groups = {}

    for row in template:
        date = clean(row.get("date"))
        camera = clean(row.get("camera"))
        pen = clean(row.get("pen"))
        codes = split_semicolon(row.get("candidate_video_camera_codes"))

        if not date or not camera or not codes:
            continue

        key = (date, camera)
        groups.setdefault(key, {
            "date": date,
            "camera": camera,
            "video_camera_code": "",
            "candidate_video_camera_codes": set(),
            "candidate_pen_groups": set(),
            "manual_note": "Select one c-code for the whole TLC camera. Pen-level ROI will be handled later.",
        })

        groups[key]["candidate_video_camera_codes"].update(codes)
        if pen:
            groups[key]["candidate_pen_groups"].add(pen)

    rows = []

    for key in sorted(groups.keys()):
        g = groups[key]
        rows.append({
            "date": g["date"],
            "camera": g["camera"],
            "video_camera_code": "",
            "candidate_video_camera_codes": ";".join(sorted(g["candidate_video_camera_codes"])),
            "candidate_pen_groups": ";".join(sorted(g["candidate_pen_groups"])),
            "manual_note": g["manual_note"],
        })

    if CAMERA_CONFIG.exists():
        old = read_csv_rows(CAMERA_CONFIG)
        old_map = {
            (clean(r.get("date")), clean(r.get("camera"))): clean(r.get("video_camera_code"))
            for r in old
        }

        for row in rows:
            key = (row["date"], row["camera"])
            if key in old_map:
                row["video_camera_code"] = old_map[key]

    return rows


def load_manifest_grouped():
    rows = load_manifest_rows()
    grouped = {}

    for row in rows:
        date = clean(row.get("date"))
        camera = clean(row.get("camera"))
        code = clean(row.get("candidate_code"))
        status = clean(row.get("thumbnail_status"))
        path = clean(row.get("thumbnail_path"))
        filename = clean(row.get("video_filename"))

        if not date or not camera or not code or status != "ok" or not path:
            continue

        key = (date, camera)
        grouped.setdefault(key, {})

        if code not in grouped[key]:
            grouped[key][code] = {
                "path": path,
                "filename": filename,
            }

    return grouped


def safe_file_response(path_str):
    if not path_str:
        return None

    try:
        p = Path(unquote_plus(path_str)).resolve()
    except Exception:
        return None

    allowed = any(str(p).startswith(str(base)) for base in ALLOWED_BASES)

    if not allowed:
        return None

    if not p.exists() or not p.is_file():
        return None

    return p


def expand_camera_to_pen_config(camera_rows):
    template = load_template_rows()

    selected = {
        (clean(r.get("date")), clean(r.get("camera"))): clean(r.get("video_camera_code"))
        for r in camera_rows
        if clean(r.get("video_camera_code"))
    }

    out = []

    for row in template:
        r = dict(row)
        date = clean(r.get("date"))
        camera = clean(r.get("camera"))
        candidate_codes = split_semicolon(r.get("candidate_video_camera_codes"))

        chosen = selected.get((date, camera), "")

        if chosen and chosen in candidate_codes:
            r["video_camera_code"] = chosen
        else:
            r["video_camera_code"] = ""

        out.append(r)

    write_csv_rows(PEN_CONFIG, out, PEN_COLUMNS)


def render_page(message=""):
    camera_rows = build_camera_rows()
    manifest = load_manifest_grouped()

    filled = sum(1 for r in camera_rows if clean(r.get("video_camera_code")))
    total = len(camera_rows)

    body = []

    body.append("""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>v78d Camera-Level Mapping Interface</title>
<style>
body {
    font-family: Arial, sans-serif;
    margin: 0;
    background: #f6f6f6;
}
.header {
    position: sticky;
    top: 0;
    z-index: 10;
    background: #111;
    color: white;
    padding: 14px 22px;
}
.header h1 {
    margin: 0 0 6px 0;
    font-size: 22px;
}
.header p {
    margin: 4px 0;
    color: #ddd;
}
.container {
    padding: 22px;
}
.instructions {
    background: #eef4ff;
    border: 1px solid #9db8ef;
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 20px;
    line-height: 1.45;
}
.card {
    background: white;
    border-radius: 10px;
    padding: 18px;
    margin-bottom: 22px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.codes {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
}
.code-card {
    width: 360px;
    border: 2px solid #ddd;
    border-radius: 8px;
    padding: 8px;
    background: #fafafa;
}
.code-card:hover {
    border-color: #2d6cdf;
}
.code-card img {
    max-width: 100%;
    border-radius: 4px;
    display: block;
}
.selected {
    border-color: #2d6cdf;
    background: #eef4ff;
}
.radio-line {
    font-size: 16px;
    margin-bottom: 7px;
}
.small {
    font-size: 13px;
    color: #666;
}
.blank-option {
    margin-bottom: 10px;
    padding: 8px;
    background: #fff;
    border: 1px dashed #999;
    border-radius: 6px;
}
.no-candidate {
    background: #fff8e8;
    border: 1px solid #e0c36a;
    padding: 10px;
    border-radius: 6px;
}
.message {
    background: #e9ffe9;
    border: 1px solid #65b565;
    color: #145214;
    padding: 10px;
    border-radius: 6px;
    margin-top: 10px;
}
button {
    padding: 9px 16px;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-weight: bold;
}
.save {
    background: #2d6cdf;
    color: white;
}
.path {
    font-family: monospace;
    background: #eee;
    padding: 2px 4px;
    border-radius: 4px;
}
</style>
</head>
<body>
<form method="POST" action="/save">
<div class="header">
<h1>v78d Camera-Level Mapping Interface</h1>
""")

    body.append(f"<p>Total camera rows: <b>{total}</b> | Filled: <b>{filled}</b></p>")
    body.append(f"<p>Camera-level output: <span class='path'>{html.escape(str(CAMERA_CONFIG))}</span></p>")
    body.append(f"<p>v78b config output: <span class='path'>{html.escape(str(PEN_CONFIG))}</span></p>")

    if message:
        body.append(f"<div class='message'>{html.escape(message)}</div>")

    body.append("""
<button class="save" type="submit">Save camera mapping and build v78b config</button>
</div>

<div class="container">
<div class="instructions">
<b>Bu interface artık pen değil, camera eşleştiriyor.</b><br>
Her kart bir <b>date + TLC camera</b> grubudur. Örneğin: <b>2021-07-22 | TLC2</b>.<br>
Altındaki görseller aday encoded camera code'lardır: <b>c0000, c0001, c0002, c0003, c0100, c0101</b>.<br>
Amaç: <b>TLC2 hangi c-code görüntüsüne karşılık geliyor?</b><br>
B3/B6 gibi pen isimlerini seçmeye çalışma. Pen-level ayrımı daha sonra ROI interface ile yapılacak.<br>
Emin değilsen blank bırak. Yanlış mapping, boş bırakmaktan daha kötüdür.
</div>
""")

    for idx, row in enumerate(camera_rows):
        date = clean(row.get("date"))
        camera = clean(row.get("camera"))
        current = clean(row.get("video_camera_code"))
        codes = split_semicolon(row.get("candidate_video_camera_codes"))
        pens = clean(row.get("candidate_pen_groups"))

        key = (date, camera)
        thumbs = manifest.get(key, {})

        body.append("<div class='card'>")
        body.append(f"<h2>{idx + 1}. {html.escape(date)} | {html.escape(camera)}</h2>")
        body.append(f"<div class='small'>Pen groups under this camera: <b>{html.escape(pens)}</b></div>")
        body.append(f"<div class='small'>Current selection: <b>{html.escape(current) if current else 'blank'}</b></div><br>")

        if not codes:
            body.append("<div class='no-candidate'>No candidate code. Leave blank.</div>")
            body.append(f"<input type='hidden' name='code_{idx}' value=''>")
            body.append("</div>")
            continue

        body.append("<div class='blank-option'>")
        checked_blank = "checked" if not current else ""
        body.append(f"<label><input type='radio' name='code_{idx}' value='' {checked_blank}> Leave blank / unresolved</label>")
        body.append("</div>")

        body.append("<div class='codes'>")

        for code in codes:
            selected_class = " selected" if current == code else ""
            checked = "checked" if current == code else ""
            thumb = thumbs.get(code)

            body.append(f"<div class='code-card{selected_class}'>")
            body.append("<div class='radio-line'>")
            body.append(f"<label><input type='radio' name='code_{idx}' value='{html.escape(code)}' {checked}> <b>{html.escape(code)}</b></label>")
            body.append("</div>")

            if thumb:
                img_url = "/image?path=" + quote(thumb["path"])
                body.append(f"<img src='{img_url}' alt='{html.escape(code)}'>")
                body.append(f"<div class='small'>{html.escape(thumb['filename'])}</div>")
            else:
                body.append("<div class='small'>No thumbnail for this code.</div>")

            body.append("</div>")

        body.append("</div>")
        body.append("</div>")

    body.append("""
<button class="save" type="submit">Save camera mapping and build v78b config</button>
</div>
</form>
</body>
</html>
""")

    return "\n".join(body).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            qs = parse_qs(parsed.query)
            msg = ""
            if qs.get("saved", ["0"])[0] == "1":
                msg = "Saved camera_level_code_mapping.csv and rebuilt camera_code_mapping.csv."
            content = render_page(msg)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        if parsed.path == "/image":
            qs = parse_qs(parsed.query)
            path = qs.get("path", [""])[0]
            p = safe_file_response(path)

            if p is None:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Image not found or not allowed.")
                return

            data = p.read_bytes()
            ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"

            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path != "/save":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        form = parse_qs(raw)

        rows = build_camera_rows()

        for idx, row in enumerate(rows):
            row["video_camera_code"] = clean(form.get(f"code_{idx}", [""])[0])

        write_csv_rows(CAMERA_CONFIG, rows, CAMERA_COLUMNS)
        expand_camera_to_pen_config(rows)

        self.send_response(303)
        self.send_header("Location", "/?saved=1")
        self.end_headers()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8542)
    args = parser.parse_args()

    if not TEMPLATE.exists():
        raise SystemExit(f"Missing template: {TEMPLATE}")

    if not MANIFEST.exists():
        raise SystemExit(f"Missing manifest: {MANIFEST}")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving camera-level mapping interface on http://{args.host}:{args.port}", flush=True)
    print(f"Camera output: {CAMERA_CONFIG}", flush=True)
    print(f"v78b config:   {PEN_CONFIG}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
