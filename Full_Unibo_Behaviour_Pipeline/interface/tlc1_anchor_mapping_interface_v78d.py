from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, quote, unquote_plus
import argparse
import csv
import html
import mimetypes
import cv2


ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

V76_VIDEOS = FULL / "outputs" / "v76_full_annotation_video_mapping_audit" / "Full_Unibo_Annotation_Video_Mapping_Audit" / "v76_all_videos_inventory.csv"

TEMPLATE = FULL / "config" / "camera_code_mapping_TEMPLATE_TO_FILL.csv"
OUTPUT_CONFIG = FULL / "config" / "camera_code_mapping.csv"
OUTPUT_CAMERA_LEVEL = FULL / "config" / "camera_level_code_mapping.csv"

OUT = FULL / "outputs" / "v78d_tlc1_anchor_mapping_interface"
STATIC = OUT / "static"

OUT.mkdir(parents=True, exist_ok=True)
STATIC.mkdir(parents=True, exist_ok=True)

ALLOWED_BASES = [
    OUT.resolve(),
    STATIC.resolve(),
    (FULL / "config").resolve(),
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


def read_csv_rows(path):
    rows = []
    if not path.exists():
        return rows

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: clean(v) for k, v in row.items()})

    return rows


def write_csv_rows(path, rows, columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)

    if columns is None:
        columns = []
        for row in rows:
            for k in row:
                if k not in columns:
                    columns.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def split_semicolon(x):
    x = clean(x)
    if not x:
        return []
    return [p.strip() for p in x.split(";") if p.strip()]


def extract_thumb(video_path, out_path, label, sec=5, width=420):
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
    new_h = int(h * scale)
    frame = cv2.resize(frame, (width, new_h))

    cv2.rectangle(frame, (0, 0), (width, 42), (0, 0, 0), -1)
    cv2.putText(frame, label[:55], (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), frame)
    return True


def find_videos():
    videos = read_csv_rows(V76_VIDEOS)

    references = []
    encoded_by_hour_code = {}

    for row in videos:
        filename = clean(row.get("video_filename"))
        vtype = clean(row.get("video_name_type"))
        path = clean(row.get("video_path"))
        parsed_date = clean(row.get("parsed_date"))
        parsed_camera = clean(row.get("parsed_camera"))
        parsed_pen = clean(row.get("parsed_pen"))
        start = clean(row.get("parsed_start_hhmm"))
        code = clean(row.get("parsed_camera_code"))

        if vtype == "friendly_tlc_time_range" and parsed_camera == "TLC1" and parsed_pen == "B1":
            references.append({
                "filename": filename,
                "path": path,
                "start": start,
            })

        if vtype == "encoded_camera_datetime" and parsed_date == "2021-07-22" and code and start:
            encoded_by_hour_code.setdefault(start, {})
            encoded_by_hour_code[start][code] = {
                "filename": filename,
                "path": path,
                "code": code,
                "start": start,
            }

    references = sorted(references, key=lambda r: r["start"])
    return references, encoded_by_hour_code


def build_static():
    references, encoded_by_hour_code = find_videos()

    rows = []
    all_codes = set()

    for ref in references:
        start = ref["start"]
        ref_id = start.replace(":", "")

        ref_img = STATIC / f"ref_TLC1_B1_{ref_id}.jpg"
        extract_thumb(ref["path"], ref_img, f"REFERENCE TLC1 B1 | {start}")

        candidates = encoded_by_hour_code.get(start, {})
        cand_rows = []

        for code, c in sorted(candidates.items()):
            all_codes.add(code)
            cand_img = STATIC / f"candidate_{start.replace(':','')}_{code}.jpg"
            extract_thumb(c["path"], cand_img, f"{code} | {c['filename']}")

            cand_rows.append({
                "code": code,
                "filename": c["filename"],
                "path": c["path"],
                "thumb": str(cand_img),
            })

        rows.append({
            "start": start,
            "reference_filename": ref["filename"],
            "reference_path": ref["path"],
            "reference_thumb": str(ref_img),
            "candidates": cand_rows,
        })

    return rows, sorted(all_codes)


def load_existing_selection():
    rows = read_csv_rows(OUTPUT_CAMERA_LEVEL)
    for r in rows:
        if clean(r.get("date")) == "2021-07-22" and clean(r.get("camera")) == "TLC1":
            return clean(r.get("video_camera_code"))
    return ""


def save_mapping(selected_code):
    template_rows = read_csv_rows(TEMPLATE)
    output_rows = []

    for row in template_rows:
        r = dict(row)

        date = clean(r.get("date"))
        camera = clean(r.get("camera"))
        candidate_codes = split_semicolon(r.get("candidate_video_camera_codes"))

        if date == "2021-07-22" and camera == "TLC1" and selected_code in candidate_codes:
            r["video_camera_code"] = selected_code
        else:
            r["video_camera_code"] = ""

        output_rows.append(r)

    write_csv_rows(OUTPUT_CONFIG, output_rows, PEN_COLUMNS)

    camera_level = [{
        "date": "2021-07-22",
        "camera": "TLC1",
        "video_camera_code": selected_code,
        "method": "TLC1_B1_friendly_reference_visual_anchor",
        "note": "Only TLC1 is mapped. Other TLC cameras remain unresolved until external mapping/reference/ROI is available.",
    }]

    write_csv_rows(OUTPUT_CAMERA_LEVEL, camera_level)


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
    rows, all_codes = build_static()
    current = load_existing_selection()

    body = []

    body.append("""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>v78d TLC1 Anchor Mapping</title>
<style>
body { font-family: Arial, sans-serif; margin: 0; background: #f6f6f6; }
.header { position: sticky; top: 0; z-index: 10; background: #111; color: white; padding: 14px 22px; }
.header h1 { margin: 0 0 6px 0; font-size: 22px; }
.header p { margin: 4px 0; color: #ddd; }
.container { padding: 22px; }
.instructions { background: #eef4ff; border: 1px solid #9db8ef; padding: 12px; border-radius: 8px; margin-bottom: 20px; line-height: 1.45; }
.card { background: white; border-radius: 10px; padding: 18px; margin-bottom: 22px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.row { display: flex; gap: 18px; align-items: flex-start; flex-wrap: wrap; }
.ref { border: 4px solid #2d6cdf; background: #eef4ff; padding: 8px; border-radius: 8px; width: 430px; }
.cand { border: 3px solid #ddd; background: #fafafa; padding: 8px; border-radius: 8px; width: 430px; cursor: pointer; }
.cand:hover { border-color: #2d6cdf; background: #eef4ff; }
.cand.selected { border-color: #0a8f3c; background: #e9ffe9; }
img { max-width: 100%; border-radius: 4px; display: block; margin-top: 8px; }
.small { font-size: 13px; color: #666; }
.message { background: #e9ffe9; border: 1px solid #65b565; color: #145214; padding: 10px; border-radius: 6px; margin-top: 10px; }
.warning { background: #fff8e8; border: 1px solid #e0c36a; padding: 10px; border-radius: 6px; margin-bottom: 15px; }
button { padding: 9px 16px; border: none; border-radius: 6px; cursor: pointer; font-weight: bold; }
.save { background: #2d6cdf; color: white; }
.radio-line { font-size: 18px; margin-bottom: 6px; }
.top-select { background: white; padding: 12px; border-radius: 8px; margin-bottom: 18px; border: 1px solid #ccc; }
.code-radio { transform: scale(1.35); margin-right: 8px; }
</style>
<script>
function selectCode(code) {
    const radios = document.querySelectorAll("input[name='selected_code']");
    radios.forEach(r => {
        if (r.value === code) {
            r.checked = true;
        }
    });
    document.querySelectorAll(".cand").forEach(card => {
        if (card.dataset.code === code) {
            card.classList.add("selected");
        } else {
            card.classList.remove("selected");
        }
    });
}
window.addEventListener("DOMContentLoaded", () => {
    const checked = document.querySelector("input[name='selected_code']:checked");
    if (checked) {
        selectCode(checked.value);
    }
});
</script>
</head>
<body>
<form method="POST" action="/save">
<div class="header">
<h1>v78d TLC1 Anchor Mapping Interface</h1>
""")

    body.append(f"<p>Current TLC1 selection: <b>{html.escape(current) if current else 'blank'}</b></p>")

    if message:
        body.append(f"<div class='message'>{html.escape(message)}</div>")

    body.append("""
<button class="save" type="submit">Save TLC1 mapping</button>
</div>

<div class="container">
<div class="instructions">
<b>Bu interface sadece TLC1’i çözer.</b><br>
Solda named/reference video var: <b>TLC1 B1</b>.<br>
Sağda aynı saatteki encoded aday videolar var: <b>c0000, c0001, c0002, c0003, c0100, c0101</b>.<br>
Hangi c-code görüntüsü soldaki TLC1 B1 reference görüntüsüne en çok benziyorsa onu seç.<br>
<b>Artık her aday görselin üstünde seçme kutucuğu var. Görsel kartına tıklayınca da seçilir.</b>
</div>
""")

    body.append("<div class='top-select'>")
    body.append("<b>Genel seçim:</b><br><br>")
    body.append(f"<label><input class='code-radio' type='radio' name='selected_code' value='' {'checked' if not current else ''} onclick=\"selectCode('')\"> Leave blank / unresolved</label><br>")

    for code in all_codes:
        checked = "checked" if current == code else ""
        body.append(f"<label><input class='code-radio' type='radio' name='selected_code' value='{html.escape(code)}' {checked} onclick=\"selectCode('{html.escape(code)}')\"> <b>{html.escape(code)}</b></label><br>")

    body.append("</div>")

    for row in rows:
        body.append("<div class='card'>")
        body.append(f"<h2>Hour: {html.escape(row['start'])}</h2>")

        body.append("<div class='row'>")

        ref_url = "/image?path=" + quote(row["reference_thumb"])
        body.append("<div class='ref'>")
        body.append("<b>REFERENCE: TLC1 B1</b>")
        body.append(f"<img src='{ref_url}'>")
        body.append(f"<div class='small'>{html.escape(row['reference_filename'])}</div>")
        body.append("</div>")

        if not row["candidates"]:
            body.append("<div class='warning'>No encoded candidates for this hour.</div>")
        else:
            for cand in row["candidates"]:
                code = cand["code"]
                selected_class = " selected" if current == code else ""
                cand_url = "/image?path=" + quote(cand["thumb"])

                body.append(f"<div class='cand{selected_class}' data-code='{html.escape(code)}' onclick=\"selectCode('{html.escape(code)}')\">")
                body.append("<div class='radio-line'>")
                body.append(f"<label onclick='event.stopPropagation();'><input class='code-radio' type='radio' name='selected_code' value='{html.escape(code)}' {'checked' if current == code else ''} onclick=\"selectCode('{html.escape(code)}')\"> <b>Select {html.escape(code)}</b></label>")
                body.append("</div>")
                body.append(f"<img src='{cand_url}'>")
                body.append(f"<div class='small'>{html.escape(cand['filename'])}</div>")
                body.append("</div>")

        body.append("</div>")
        body.append("</div>")

    body.append("""
<button class="save" type="submit">Save TLC1 mapping</button>
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
                msg = "Saved TLC1 anchor mapping to camera_code_mapping.csv."
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

        selected = clean(form.get("selected_code", [""])[0])
        save_mapping(selected)

        self.send_response(303)
        self.send_header("Location", "/?saved=1")
        self.end_headers()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8543)
    args = parser.parse_args()

    if not V76_VIDEOS.exists():
        raise SystemExit(f"Missing video inventory: {V76_VIDEOS}")

    if not TEMPLATE.exists():
        raise SystemExit(f"Missing template: {TEMPLATE}")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving TLC1 anchor mapping interface on http://{args.host}:{args.port}", flush=True)
    print(f"Output config: {OUTPUT_CONFIG}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
