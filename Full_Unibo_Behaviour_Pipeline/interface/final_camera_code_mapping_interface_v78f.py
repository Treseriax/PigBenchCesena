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

DRAFT_MAP = FULL / "config" / "final_camera_code_mapping_DRAFT.csv"
FINAL_MAP = FULL / "config" / "final_camera_code_mapping.csv"
EXPANDED_V78B_CONFIG = FULL / "config" / "camera_code_mapping.csv"
CAMERA_LEVEL_MAP = FULL / "config" / "camera_level_code_mapping.csv"

OUT = FULL / "outputs" / "v78f_final_camera_code_mapping_interface"
STATIC = OUT / "static"
VALIDATION = OUT / "v78f_final_mapping_validation.csv"
ISSUES = OUT / "v78f_issues.csv"

OUT.mkdir(parents=True, exist_ok=True)
STATIC.mkdir(parents=True, exist_ok=True)

ALLOWED_BASES = [OUT.resolve(), STATIC.resolve(), (FULL / "config").resolve()]

CAMERAS = ["TLC1", "TLC2", "TLC3", "TLC4", "TLC5", "TLC6"]

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
        for r in rows:
            for k in r.keys():
                if k not in columns:
                    columns.append(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


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


def get_encoded_videos():
    rows = read_csv_rows(V76_VIDEOS)

    encoded = {}
    references = []

    for r in rows:
        filename = clean(r.get("video_filename"))
        vtype = clean(r.get("video_name_type"))
        path = clean(r.get("video_path"))
        parsed_date = clean(r.get("parsed_date"))
        parsed_camera = clean(r.get("parsed_camera"))
        parsed_pen = clean(r.get("parsed_pen"))
        start = clean(r.get("parsed_start_hhmm"))
        code = clean(r.get("parsed_camera_code"))

        if vtype == "encoded_camera_datetime" and parsed_date == "2021-07-22" and code:
            encoded.setdefault(code, [])
            encoded[code].append({
                "filename": filename,
                "path": path,
                "start": start,
                "code": code,
            })

        if vtype == "friendly_tlc_time_range" and parsed_camera == "TLC1" and parsed_pen == "B1":
            references.append({
                "filename": filename,
                "path": path,
                "start": start,
            })

    for code in encoded:
        encoded[code] = sorted(encoded[code], key=lambda x: x["start"])

    references = sorted(references, key=lambda x: x["start"])

    return encoded, references


def build_static():
    encoded, references = get_encoded_videos()

    code_cards = []

    for code in sorted(encoded.keys()):
        thumbs = []
        chosen = []

        # Use a few representative hours, not every video.
        for item in encoded[code]:
            if item["start"] in {"07:00", "08:00", "09:00", "10:00", "11:00", "12:00"}:
                chosen.append(item)

        if not chosen:
            chosen = encoded[code][:4]

        for item in chosen[:6]:
            thumb = STATIC / f"{code}_{item['start'].replace(':','')}.jpg"
            extract_thumb(item["path"], thumb, f"{code} | {item['start']} | {item['filename']}")
            thumbs.append({
                "start": item["start"],
                "filename": item["filename"],
                "thumb": str(thumb),
            })

        code_cards.append({
            "code": code,
            "video_count": len(encoded[code]),
            "thumbs": thumbs,
        })

    ref_thumbs = []

    for ref in references:
        if ref["start"] in {"07:00", "08:00", "09:00", "10:00", "11:00", "12:00"}:
            thumb = STATIC / f"REF_TLC1_B1_{ref['start'].replace(':','')}.jpg"
            extract_thumb(ref["path"], thumb, f"REFERENCE TLC1 B1 | {ref['start']}")
            ref_thumbs.append({
                "start": ref["start"],
                "filename": ref["filename"],
                "thumb": str(thumb),
            })

    return code_cards, ref_thumbs


def load_existing_mapping():
    selected = {cam: "" for cam in CAMERAS}

    if FINAL_MAP.exists():
        rows = read_csv_rows(FINAL_MAP)
    elif DRAFT_MAP.exists():
        rows = read_csv_rows(DRAFT_MAP)
    else:
        rows = read_csv_rows(CAMERA_LEVEL_MAP)

    for r in rows:
        cam = clean(r.get("camera"))
        code = clean(r.get("video_camera_code"))
        if cam in selected:
            selected[cam] = code

    # TLC1 visual anchor is our locked known mapping.
    selected["TLC1"] = "c0002"

    return selected


def validate_mapping(selected, valid_codes):
    rows = []
    issues = []

    used = {}

    for cam in CAMERAS:
        code = clean(selected.get(cam, ""))

        if not code:
            rows.append({
                "check_name": f"{cam}_filled",
                "expected": "non-empty",
                "actual": "",
                "passed": False,
                "severity": "hard",
                "detail": f"{cam} has no selected camera code.",
            })
            issues.append({
                "item": cam,
                "issue_type": "missing_camera_code",
                "issue_detail": f"{cam} must be mapped before final tracking.",
                "severity": "hard",
            })
            continue

        if code not in valid_codes:
            rows.append({
                "check_name": f"{cam}_valid_code",
                "expected": ";".join(valid_codes),
                "actual": code,
                "passed": False,
                "severity": "hard",
                "detail": f"{cam} selected code is not in available encoded camera codes.",
            })
            issues.append({
                "item": cam,
                "issue_type": "invalid_camera_code",
                "issue_detail": f"{cam} selected invalid code {code}.",
                "severity": "hard",
            })
        else:
            rows.append({
                "check_name": f"{cam}_valid_code",
                "expected": "valid encoded code",
                "actual": code,
                "passed": True,
                "severity": "hard",
                "detail": f"{cam} has a valid encoded camera code.",
            })

        used.setdefault(code, [])
        used[code].append(cam)

    for code, cams in used.items():
        if code and len(cams) > 1:
            rows.append({
                "check_name": f"{code}_unique",
                "expected": "used by one TLC only",
                "actual": ";".join(cams),
                "passed": False,
                "severity": "hard",
                "detail": f"{code} is assigned to multiple TLC cameras.",
            })
            issues.append({
                "item": code,
                "issue_type": "duplicate_camera_code_assignment",
                "issue_detail": f"{code} assigned to {cams}. Each encoded code can map to only one TLC camera.",
                "severity": "hard",
            })

    if clean(selected.get("TLC1")) != "c0002":
        rows.append({
            "check_name": "TLC1_locked_anchor",
            "expected": "c0002",
            "actual": clean(selected.get("TLC1")),
            "passed": False,
            "severity": "hard",
            "detail": "TLC1 must remain locked to c0002 based on manual visual anchor.",
        })
        issues.append({
            "item": "TLC1",
            "issue_type": "locked_anchor_modified",
            "issue_detail": "TLC1 must remain c0002.",
            "severity": "hard",
        })
    else:
        rows.append({
            "check_name": "TLC1_locked_anchor",
            "expected": "c0002",
            "actual": "c0002",
            "passed": True,
            "severity": "hard",
            "detail": "TLC1 locked anchor is preserved.",
        })

    return rows, issues


def write_expanded_v78b_config(selected):
    template = read_csv_rows(TEMPLATE)
    out = []

    for row in template:
        r = dict(row)
        date = clean(r.get("date"))
        cam = clean(r.get("camera"))
        candidates = split_semicolon(r.get("candidate_video_camera_codes"))
        chosen = clean(selected.get(cam, ""))

        if date == "2021-07-22" and chosen and chosen in candidates:
            r["video_camera_code"] = chosen
        else:
            r["video_camera_code"] = ""

        out.append(r)

    write_csv_rows(EXPANDED_V78B_CONFIG, out, PEN_COLUMNS)


def save_mapping(selected, valid_codes):
    draft_rows = []
    for cam in CAMERAS:
        method = "locked_TLC1_visual_anchor" if cam == "TLC1" else "manual_final_camera_map_confirmation"
        draft_rows.append({
            "date": "2021-07-22",
            "camera": cam,
            "video_camera_code": clean(selected.get(cam, "")),
            "method": method,
            "note": "Final one-to-one camera map for available 2021-07-22 encoded videos.",
        })

    write_csv_rows(DRAFT_MAP, draft_rows)

    qa_rows, issue_rows = validate_mapping(selected, valid_codes)
    write_csv_rows(VALIDATION, qa_rows)
    write_csv_rows(ISSUES, issue_rows, ["item", "issue_type", "issue_detail", "severity"])

    hard_issues = [i for i in issue_rows if i["severity"] == "hard"]

    if not hard_issues:
        write_csv_rows(FINAL_MAP, draft_rows)
        write_expanded_v78b_config(selected)
        return True, "Final mapping saved. camera_code_mapping.csv rebuilt for v78b."

    return False, f"Draft saved, but final mapping NOT accepted. Hard issues: {len(hard_issues)}. Check v78f_issues.csv."


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
    code_cards, ref_thumbs = build_static()
    selected = load_existing_mapping()
    valid_codes = [c["code"] for c in code_cards]

    body = []
    body.append("""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>v78f Final Camera-Code Mapping</title>
<style>
body { font-family: Arial, sans-serif; margin: 0; background: #f6f6f6; }
.header { position: sticky; top: 0; z-index: 10; background: #111; color: white; padding: 14px 22px; }
.header h1 { margin: 0 0 6px 0; font-size: 22px; }
.header p { margin: 4px 0; color: #ddd; }
.container { padding: 22px; }
.instructions { background: #eef4ff; border: 1px solid #9db8ef; padding: 12px; border-radius: 8px; margin-bottom: 20px; line-height: 1.45; }
.warning { background: #fff8e8; border: 1px solid #e0c36a; padding: 10px; border-radius: 6px; margin-bottom: 15px; }
.message { background: #e9ffe9; border: 1px solid #65b565; color: #145214; padding: 10px; border-radius: 6px; margin-top: 10px; }
.mapping { background: white; border-radius: 10px; padding: 18px; margin-bottom: 22px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
.row { display: grid; grid-template-columns: 120px 250px 1fr; gap: 12px; align-items: center; margin-bottom: 10px; }
select { padding: 8px; font-size: 15px; }
button { padding: 9px 16px; border: none; border-radius: 6px; cursor: pointer; font-weight: bold; }
.save { background: #2d6cdf; color: white; }
.gallery { display: flex; flex-wrap: wrap; gap: 18px; }
.card { background: white; border-radius: 10px; padding: 12px; margin-bottom: 22px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); width: 440px; }
.card img { max-width: 100%; border-radius: 4px; margin-top: 8px; }
.small { font-size: 13px; color: #666; }
.ref { border: 4px solid #2d6cdf; background: #eef4ff; }
.locked { background: #eee; }
</style>
</head>
<body>
<form method="POST" action="/save">
<div class="header">
<h1>v78f Final Camera-Code Mapping Gate</h1>
<p>TLC1 is locked to c0002. Remaining TLC cameras must be mapped one-to-one.</p>
<button class="save" type="submit">Save / Validate Final Mapping</button>
""")

    if message:
        body.append(f"<div class='message'>{html.escape(message)}</div>")

    body.append("""
</div>
<div class="container">
<div class="instructions">
<b>Bu aşamada amaç:</b> 2021-07-22 için TLC1–TLC6 kameralarını encoded camera code'lara bire bir bağlamak.<br>
<b>TLC1 = c0002 kilitli.</b> Çünkü named TLC1 B1 reference ile görsel anchor yaptık.<br>
TLC2–TLC6 için dataset/supervisor camera map veya kesin görsel doğrulama kullanılmalı. Aynı c-code iki TLC'ye atanırsa final kabul edilmez.
</div>
<div class="warning">
v78e otomatik önerilerini direkt kabul etmiyoruz: aynı code'u birden fazla TLC'ye önerdiği için kirli/çelişkili evidence var. Bu interface final validation gate'tir.
</div>
<div class="mapping">
<h2>Final mapping table</h2>
""")

    for cam in CAMERAS:
        locked = cam == "TLC1"
        body.append("<div class='row'>")
        body.append(f"<b>{cam}</b>")

        if locked:
            body.append("<select class='locked' name='TLC1' readonly><option value='c0002'>c0002</option></select>")
            body.append("<span class='small'>Locked visual anchor from TLC1 B1 reference.</span>")
        else:
            body.append(f"<select name='{cam}'>")
            body.append("<option value=''>-- unresolved --</option>")
            for code in valid_codes:
                sel = "selected" if selected.get(cam, "") == code else ""
                body.append(f"<option value='{html.escape(code)}' {sel}>{html.escape(code)}</option>")
            body.append("</select>")
            body.append("<span class='small'>Must be unique. Use verified camera map / supervisor confirmation.</span>")

        body.append("</div>")

    body.append("""
</div>

<h2>TLC1 reference frames</h2>
<div class="gallery">
""")

    for ref in ref_thumbs:
        url = "/image?path=" + quote(ref["thumb"])
        body.append("<div class='card ref'>")
        body.append(f"<b>REFERENCE TLC1 B1 | {html.escape(ref['start'])}</b>")
        body.append(f"<img src='{url}'>")
        body.append(f"<div class='small'>{html.escape(ref['filename'])}</div>")
        body.append("</div>")

    body.append("</div><h2>Encoded camera-code galleries</h2><div class='gallery'>")

    for card in code_cards:
        body.append("<div class='card'>")
        body.append(f"<h3>{html.escape(card['code'])}</h3>")
        body.append(f"<div class='small'>Video count: {card['video_count']}</div>")
        for t in card["thumbs"]:
            url = "/image?path=" + quote(t["thumb"])
            body.append(f"<img src='{url}'>")
            body.append(f"<div class='small'>{html.escape(t['filename'])}</div>")
        body.append("</div>")

    body.append("""
</div>
<button class="save" type="submit">Save / Validate Final Mapping</button>
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
            msg = qs.get("msg", [""])[0]
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

        code_cards, _ = build_static()
        valid_codes = [c["code"] for c in code_cards]

        selected = {}
        for cam in CAMERAS:
            selected[cam] = clean(form.get(cam, [""])[0])

        selected["TLC1"] = "c0002"

        ok, msg = save_mapping(selected, valid_codes)

        self.send_response(303)
        self.send_header("Location", "/?msg=" + quote(msg))
        self.end_headers()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8544)
    args = parser.parse_args()

    if not V76_VIDEOS.exists():
        raise SystemExit(f"Missing video inventory: {V76_VIDEOS}")
    if not TEMPLATE.exists():
        raise SystemExit(f"Missing template: {TEMPLATE}")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving v78f final mapping interface on http://{args.host}:{args.port}", flush=True)
    print(f"Draft map: {DRAFT_MAP}", flush=True)
    print(f"Final map: {FINAL_MAP}", flush=True)
    print(f"Expanded v78b config: {EXPANDED_V78B_CONFIG}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
