
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote
from datetime import datetime
import argparse
import csv
import json
import math
import mimetypes
import os
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

UI_JSON = W8 / "outputs" / "v63a_canonical_gt_v2_schema" / "week8_v63a_visualizer_data.json"
ASSIGNMENTS = W8 / "validation" / "week8_v63b_manual_gt_v2_assignments.csv"
STATIC = W8 / "interface" / "static_v63b"

ASSIGNMENT_KEY = ["scan_frame_id", "canonical_colour_label_norm"]

UPDATE_FIELDS = [
    "manual_assigned_candidate_box_id",
    "manual_bbox_x1",
    "manual_bbox_y1",
    "manual_bbox_x2",
    "manual_bbox_y2",
    "manual_bbox_status",
    "manual_identity_status",
    "manual_colour_status",
    "manual_behaviour_status",
    "manual_gt_v2_status",
    "manual_classification_use",
    "manual_review_priority",
    "manual_reviewer_note",
    "manual_reviewed_by",
    "manual_reviewed_at",
    "assignment_created_at",
    "assignment_updated_at",
]


def clean_str(x):
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
    except Exception:
        pass
    s = str(x)
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
    return obj


def read_json_data():
    with open(UI_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return sanitize(data)


def read_assignments():
    if not ASSIGNMENTS.exists():
        return []
    df = pd.read_csv(ASSIGNMENTS)
    df = df.fillna("")
    return df.to_dict(orient="records")


def write_assignments_df(df):
    df.to_csv(
        ASSIGNMENTS,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def get_candidate_box(data, scan_frame_id, candidate_box_id):
    for sf in data.get("scanframes", []):
        if clean_str(sf.get("scan_frame_id")) != clean_str(scan_frame_id):
            continue
        for box in sf.get("candidate_boxes", []):
            if clean_str(box.get("candidate_box_id")) == clean_str(candidate_box_id):
                return box
    return None


def update_assignment(payload):
    df = pd.read_csv(ASSIGNMENTS).fillna("")

    scan = clean_str(payload.get("scan_frame_id"))
    colour = clean_str(payload.get("canonical_colour_label_norm"))

    if not scan or not colour:
        return {"ok": False, "error": "scan_frame_id and canonical_colour_label_norm are required."}

    mask = (
        (df["scan_frame_id"].astype(str) == scan)
        & (df["canonical_colour_label_norm"].astype(str) == colour)
    )

    if not mask.any():
        return {"ok": False, "error": f"No assignment row found for {scan} / {colour}."}

    now = datetime.now().isoformat(timespec="seconds")
    idx = df.index[mask][0]

    for field in UPDATE_FIELDS:
        if field not in df.columns:
            df[field] = ""

    if not clean_str(df.at[idx, "assignment_created_at"]):
        df.at[idx, "assignment_created_at"] = now

    df.at[idx, "assignment_updated_at"] = now
    df.at[idx, "manual_reviewed_at"] = now

    for field in UPDATE_FIELDS:
        if field in payload:
            df.at[idx, field] = clean_str(payload.get(field))

    candidate_box_id = clean_str(payload.get("manual_assigned_candidate_box_id"))
    if candidate_box_id:
        data = read_json_data()
        box = get_candidate_box(data, scan, candidate_box_id)
        if box:
            coord_map = {
                "manual_bbox_x1": "bbox_x1",
                "manual_bbox_y1": "bbox_y1",
                "manual_bbox_x2": "bbox_x2",
                "manual_bbox_y2": "bbox_y2",
            }
            for manual_col, box_col in coord_map.items():
                if not clean_str(df.at[idx, manual_col]):
                    df.at[idx, manual_col] = clean_str(box.get(box_col))

    write_assignments_df(df)

    return {
        "ok": True,
        "scan_frame_id": scan,
        "canonical_colour_label_norm": colour,
        "assignment_path": str(ASSIGNMENTS),
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

        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
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
                units, rng = range_header.split("=")
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
            self._send_json(read_json_data())
            return

        if path == "/api/assignments":
            self._send_json({
                "assignment_path": str(ASSIGNMENTS),
                "assignments": read_assignments(),
            })
            return

        if path == "/api/status":
            assignments = read_assignments()
            completed = 0
            for r in assignments:
                if clean_str(r.get("manual_gt_v2_status")):
                    completed += 1
            self._send_json({
                "assignment_path": str(ASSIGNMENTS),
                "total_assignment_rows": len(assignments),
                "completed_rows": completed,
                "pending_rows": len(assignments) - completed,
            })
            return

        if path == "/video":
            p = clean_str(qs.get("path", [""])[0])
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

        if parsed.path == "/api/save_assignment":
            result = update_assignment(payload)
            self._send_json(result, 200 if result.get("ok") else 400)
            return

        self._send_json({"ok": False, "error": "Unknown endpoint."}, 404)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8527)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Week8 v63b canonical GT assignment visualizer running on http://{args.host}:{args.port}/")
    print(f"Assignments CSV: {ASSIGNMENTS}")
    server.serve_forever()


if __name__ == "__main__":
    main()
