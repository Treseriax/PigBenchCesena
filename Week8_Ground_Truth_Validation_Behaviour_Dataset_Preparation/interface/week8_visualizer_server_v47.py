#!/usr/bin/env python3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, unquote
from datetime import datetime
import argparse
import csv
import json
import mimetypes
import re


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"
STATIC = W8 / "interface" / "static"
GT_JSON = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_level_ground_truth.json"
NOTES_CSV = W8 / "validation" / "week8_v47_visual_validation_notes.csv"


def load_dataset():
    data = json.loads(GT_JSON.read_text())
    clips = data.get("clips", [])
    by_scan = {str(c.get("scan_frame_id")): c for c in clips}
    return data, clips, by_scan


DATA, CLIPS, BY_SCAN = load_dataset()


def ensure_notes_header():
    NOTES_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not NOTES_CSV.exists():
        with open(NOTES_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "created_at",
                    "scan_frame_id",
                    "video_id",
                    "video_current_time_sec",
                    "estimated_frame_index_in_clip",
                    "issue_type",
                    "severity",
                    "note",
                    "visible_layers",
                ],
                quoting=csv.QUOTE_ALL,
                escapechar="\\",
            )
            writer.writeheader()


ensure_notes_header()


def json_response(handler, obj, status=200):
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(payload)


def text_response(handler, text, status=200):
    payload = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)


def serve_file(handler, path):
    path = Path(path)
    if not path.exists() or not path.is_file():
        text_response(handler, "Not found", status=404)
        return

    content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    data = path.read_bytes()

    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def serve_video(handler, path):
    path = Path(path)
    if not path.exists() or not path.is_file():
        text_response(handler, "Video not found", status=404)
        return

    file_size = path.stat().st_size
    range_header = handler.headers.get("Range")
    content_type = mimetypes.guess_type(str(path))[0] or "video/mp4"

    if range_header:
        match = re.match(r"bytes=(\d+)-(\d*)", range_header)
        if match:
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1

            handler.send_response(206)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Accept-Ranges", "bytes")
            handler.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            handler.send_header("Content-Length", str(length))
            handler.end_headers()

            with open(path, "rb") as f:
                f.seek(start)
                handler.wfile.write(f.read(length))
            return

    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Content-Length", str(file_size))
    handler.end_headers()

    with open(path, "rb") as f:
        handler.wfile.write(f.read())


class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            serve_file(self, STATIC / "index.html")
            return

        if path.startswith("/static/"):
            rel = path.replace("/static/", "", 1)
            serve_file(self, STATIC / rel)
            return

        if path == "/api/clips":
            items = []
            for c in CLIPS:
                items.append({
                    "scan_frame_id": c.get("scan_frame_id"),
                    "video_id": c.get("video_id"),
                    "split": c.get("split"),
                    "behaviour_set": c.get("behaviour_set"),
                    "object_count": len(c.get("objects", [])),
                    "training_ready_object_count": sum(1 for o in c.get("objects", []) if o.get("is_training_ready")),
                    "generated_frame_count": c.get("generated_frame_count"),
                    "duration_sec": c.get("duration_sec"),
                    "clip_path_exists": Path(c.get("clip_path", "")).exists(),
                })
            json_response(self, {
                "dataset_version": DATA.get("dataset_version"),
                "clip_count": len(items),
                "clips": items,
            })
            return

        if path.startswith("/api/clip/"):
            scan = unquote(path.replace("/api/clip/", "", 1))
            clip = BY_SCAN.get(scan)
            if not clip:
                json_response(self, {"error": "clip not found", "scan_frame_id": scan}, status=404)
                return
            json_response(self, clip)
            return

        if path.startswith("/media/"):
            scan_file = unquote(path.replace("/media/", "", 1))
            scan = scan_file.rsplit(".", 1)[0]
            clip = BY_SCAN.get(scan)
            if not clip:
                text_response(self, "Unknown scan_frame_id", status=404)
                return
            serve_video(self, clip.get("clip_path", ""))
            return

        if path == "/api/notes":
            ensure_notes_header()
            rows = []
            with open(NOTES_CSV, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            json_response(self, {"notes_path": str(NOTES_CSV), "count": len(rows), "notes": rows[-100:]})
            return

        text_response(self, "Not found", status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/note":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            try:
                payload = json.loads(raw)
            except Exception:
                json_response(self, {"ok": False, "error": "invalid json"}, status=400)
                return

            ensure_notes_header()
            row = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "scan_frame_id": payload.get("scan_frame_id", ""),
                "video_id": payload.get("video_id", ""),
                "video_current_time_sec": payload.get("video_current_time_sec", ""),
                "estimated_frame_index_in_clip": payload.get("estimated_frame_index_in_clip", ""),
                "issue_type": payload.get("issue_type", ""),
                "severity": payload.get("severity", ""),
                "note": payload.get("note", ""),
                "visible_layers": payload.get("visible_layers", ""),
            }

            with open(NOTES_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=list(row.keys()),
                    quoting=csv.QUOTE_ALL,
                    escapechar="\\",
                )
                writer.writerow(row)

            json_response(self, {"ok": True, "notes_path": str(NOTES_CSV), "row": row})
            return

        json_response(self, {"ok": False, "error": "unknown endpoint"}, status=404)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8513)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Week 8 v47 visualizer running at http://{args.host}:{args.port}/")
    print(f"Ground truth JSON: {GT_JSON}")
    print(f"Validation notes CSV: {NOTES_CSV}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
