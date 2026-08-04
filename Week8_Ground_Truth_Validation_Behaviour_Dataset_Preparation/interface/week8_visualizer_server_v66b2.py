
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote
from datetime import datetime
import argparse
import csv
import json
import math
import mimetypes
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

DATA_JSON = W8 / "outputs" / "v66b2_adjust_only_bbox_editor" / "week8_v66b2_adjust_only_visualizer_data.json"
ASSIGNMENTS = W8 / "validation" / "week8_v63b_manual_gt_v2_assignments.csv"
STATIC = W8 / "interface" / "static_v66b2"


def clean(x):
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


def sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def write_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def load_data():
    return json.loads(DATA_JSON.read_text(encoding="utf-8"))


def save_bbox(payload):
    obj_id = clean(payload.get("canonical_gt_object_id"))
    x1 = clean(payload.get("manual_bbox_x1"))
    y1 = clean(payload.get("manual_bbox_y1"))
    x2 = clean(payload.get("manual_bbox_x2"))
    y2 = clean(payload.get("manual_bbox_y2"))

    if not obj_id:
        return {"ok": False, "error": "canonical_gt_object_id missing"}

    try:
        vals = [float(x1), float(y1), float(x2), float(y2)]
        if vals[2] <= vals[0] or vals[3] <= vals[1]:
            return {"ok": False, "error": "invalid bbox coordinates"}
    except Exception:
        return {"ok": False, "error": "bbox coordinates must be numeric"}

    df = pd.read_csv(ASSIGNMENTS).fillna("")
    for c in df.columns:
        df[c] = df[c].map(clean)

    mask = df["canonical_gt_object_id"] == obj_id
    if not mask.any():
        return {"ok": False, "error": f"object not found: {obj_id}"}

    idx = df.index[mask][0]

    df.at[idx, "manual_bbox_x1"] = x1
    df.at[idx, "manual_bbox_y1"] = y1
    df.at[idx, "manual_bbox_x2"] = x2
    df.at[idx, "manual_bbox_y2"] = y2

    df.at[idx, "manual_bbox_status"] = "bbox_ok"
    df.at[idx, "manual_identity_status"] = "identity_confirmed"
    df.at[idx, "manual_gt_v2_status"] = "gold_usable"
    df.at[idx, "manual_classification_use"] = "use_for_classification"
    df.at[idx, "manual_reviewer_note"] = clean(payload.get("manual_reviewer_note")) or "BBox adjusted in v66b2"
    df.at[idx, "assignment_updated_at"] = datetime.now().isoformat(timespec="seconds")
    df.at[idx, "manual_reviewed_at"] = datetime.now().isoformat(timespec="seconds")

    write_csv(df, ASSIGNMENTS)

    return {
        "ok": True,
        "canonical_gt_object_id": obj_id,
        "assignment_csv": str(ASSIGNMENTS),
    }


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload, code=200):
        body = json.dumps(sanitize(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text, code=200, content_type="text/plain; charset=utf-8"):
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, rel):
        path = STATIC / rel
        if not path.exists() or not path.is_file():
            self._send_text("Not found", 404)
            return
        data = path.read_bytes()
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_video(self, path_str):
        path = Path(path_str)
        if not path.exists() or not path.is_file():
            self._send_text(f"Video not found: {path}", 404)
            return

        file_size = path.stat().st_size
        ctype = mimetypes.guess_type(str(path))[0] or "video/mp4"
        range_header = self.headers.get("Range")

        if range_header:
            try:
                _, rng = range_header.split("=")
                start_s, end_s = rng.split("-")
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else file_size - 1
                end = min(end, file_size - 1)
                length = end - start + 1
                with open(path, "rb") as f:
                    f.seek(start)
                    data = f.read(length)

                self.send_response(206)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(length))
                self.end_headers()
                self.wfile.write(data)
                return
            except Exception:
                pass

        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/":
            self._serve_static("index.html")
            return
        if path.startswith("/static/"):
            self._serve_static(path.replace("/static/", "", 1))
            return
        if path == "/api/data":
            self._send_json(load_data())
            return
        if path == "/video":
            p = qs.get("path", [""])[0]
            self._serve_video(unquote(p))
            return
        self._send_text("Not found", 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(raw) if raw else {}
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, 400)
            return

        if parsed.path == "/api/save_bbox":
            result = save_bbox(payload)
            self._send_json(result, 200 if result.get("ok") else 400)
            return

        self._send_json({"ok": False, "error": "unknown endpoint"}, 404)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8529)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Week8 v66b2 adjust-only bbox editor running on http://{args.host}:{args.port}/")
    print(f"Assignment CSV: {ASSIGNMENTS}")
    server.serve_forever()


if __name__ == "__main__":
    main()
