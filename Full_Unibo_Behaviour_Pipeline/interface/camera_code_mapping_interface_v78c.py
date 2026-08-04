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
CONFIG = FULL / "config" / "camera_code_mapping.csv"
MANIFEST = PKG / "v78c_visual_atlas_manifest.csv"

ALLOWED_BASES = [
    PKG.resolve(),
    (FULL / "config").resolve(),
]

REQUIRED_COLUMNS = [
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


def write_csv_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = list(REQUIRED_COLUMNS)
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_mapping_rows():
    if CONFIG.exists():
        rows = read_csv_rows(CONFIG)
    else:
        rows = read_csv_rows(TEMPLATE)
        if rows:
            write_csv_rows(CONFIG, rows)
    return rows


def load_manifest():
    rows = read_csv_rows(MANIFEST)
    grouped = {}

    for row in rows:
        key = (
            clean(row.get("date")),
            clean(row.get("camera")),
            clean(row.get("pen")),
        )
        grouped.setdefault(key, []).append(row)

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


def render_page(message=""):
    rows = load_mapping_rows()
    manifest = load_manifest()

    filled = sum(1 for r in rows if clean(r.get("video_camera_code")))
    candidate_rows = sum(1 for r in rows if clean(r.get("candidate_video_camera_codes")))
    total = len(rows)

    body = []

    body.append("""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>v78c Camera Code Mapping Interface</title>
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
.card h2 {
    margin-top: 0;
}
.meta {
    color: #555;
    margin-bottom: 12px;
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
.no-candidate {
    background: #fff8e8;
    border: 1px solid #e0c36a;
    padding: 10px;
    border-radius: 6px;
}
.small {
    font-size: 13px;
    color: #666;
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
.blank-option {
    margin-bottom: 10px;
    padding: 8px;
    background: #fff;
    border: 1px dashed #999;
    border-radius: 6px;
}
.path {
    font-family: monospace;
    background: #eee;
    padding: 2px 4px;
    border-radius: 4px;
}
details {
    margin-top: 10px;
}
</style>
</head>
<body>
<form method="POST" action="/save">
<div class="header">
<h1>v78c Camera Code Mapping Interface</h1>
""")

    body.append(f"<p>Total rows: <b>{total}</b> | Candidate rows: <b>{candidate_rows}</b> | Filled: <b>{filled}</b></p>")
    body.append(f"<p>Output file: <span class='path'>{html.escape(str(CONFIG))}</span></p>")

    if message:
        body.append(f"<div class='message'>{html.escape(message)}</div>")

    body.append("""
<button class="save" type="submit">Save mapping CSV</button>
</div>

<div class="container">
<div class="instructions">
<b>Nasıl seçeceksin?</b><br>
Her kart bir annotation grubudur: <b>date + TLC/camera + pen</b>.<br>
Altındaki görseller aday encoded video camera code'lardır: <b>c0000, c0001, c0002, c0003, c0100, c0101</b>.<br>
Görseldeki kafes/pen layout'u, kamera açısı, duvar/yemlik/zemin çizgisi hangi TLC/pen grubuna benziyorsa onu seç.<br>
<b>Emin değilsen Leave blank / unresolved bırak.</b> Yanlış eşleştirme tracking ve GT aşamasını bozar.
</div>
""")

    for idx, row in enumerate(rows):
        date = clean(row.get("date"))
        camera = clean(row.get("camera"))
        pen = clean(row.get("pen"))
        current = clean(row.get("video_camera_code"))
        unresolved = clean(row.get("unresolved_windows"))
        codes = split_semicolon(row.get("candidate_video_camera_codes"))
        examples = split_semicolon(row.get("candidate_video_examples"))
        contact_sheet = clean(row.get("visual_contact_sheet"))

        key = (date, camera, pen)
        thumbs = manifest.get(key, [])

        thumb_by_code = {}
        for t in thumbs:
            code = clean(t.get("candidate_code"))
            status = clean(t.get("thumbnail_status"))
            path = clean(t.get("thumbnail_path"))
            filename = clean(t.get("video_filename"))
            if code and status == "ok" and path and code not in thumb_by_code:
                thumb_by_code[code] = {
                    "path": path,
                    "filename": filename,
                }

        body.append("<div class='card'>")
        body.append(f"<h2>{idx + 1}. {html.escape(date)} | {html.escape(camera)} | {html.escape(pen)}</h2>")
        body.append(f"<div class='meta'>Unresolved windows: <b>{html.escape(unresolved)}</b> | Current selection: <b>{html.escape(current) if current else 'blank'}</b></div>")

        if not codes:
            body.append("<div class='no-candidate'>Bu satır için candidate camera code/görsel yok. Şimdilik boş bırak.</div>")
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
            thumb = thumb_by_code.get(code)

            body.append(f"<div class='code-card{selected_class}'>")
            body.append("<div class='radio-line'>")
            body.append(f"<label><input type='radio' name='code_{idx}' value='{html.escape(code)}' {checked}> <b>{html.escape(code)}</b></label>")
            body.append("</div>")

            if thumb:
                img_url = "/image?path=" + quote(thumb["path"])
                body.append(f"<img src='{img_url}' alt='{html.escape(code)}'>")
                body.append(f"<div class='small'>{html.escape(thumb['filename'])}</div>")
            else:
                body.append("<div class='small'>Bu code için ayrı thumbnail yok. Contact sheet veya örnek dosyalara bak.</div>")

            body.append("</div>")

        body.append("</div>")

        if contact_sheet:
            body.append("<details>")
            body.append("<summary>Contact sheet göster</summary>")
            img_url = "/image?path=" + quote(contact_sheet)
            body.append(f"<img src='{img_url}' style='max-width:1100px; margin-top:10px;'>")
            body.append("</details>")

        if examples:
            body.append("<details>")
            body.append("<summary>Candidate video examples</summary>")
            body.append("<div class='small'>" + html.escape('; '.join(examples[:30])) + "</div>")
            body.append("</details>")

        body.append("</div>")

    body.append("""
<button class="save" type="submit">Save mapping CSV</button>
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
                msg = "Saved camera_code_mapping.csv successfully."
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

            ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
            data = p.read_bytes()

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

        rows = load_mapping_rows()

        for idx, row in enumerate(rows):
            row["video_camera_code"] = clean(form.get(f"code_{idx}", [""])[0])

        write_csv_rows(CONFIG, rows)

        self.send_response(303)
        self.send_header("Location", "/?saved=1")
        self.end_headers()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8541)
    args = parser.parse_args()

    if not TEMPLATE.exists():
        raise SystemExit(f"Missing template: {TEMPLATE}")

    if not CONFIG.exists():
        rows = read_csv_rows(TEMPLATE)
        write_csv_rows(CONFIG, rows)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving v78c mapping interface on http://{args.host}:{args.port}", flush=True)
    print(f"Template: {TEMPLATE}", flush=True)
    print(f"Output:   {CONFIG}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
