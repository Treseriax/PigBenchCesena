from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import datetime
import json
import mimetypes
import csv
import os
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

STATIC = W8 / "interface" / "static_v52d"

V45_CLIP_JSON = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_level_ground_truth.json"
V45_ANCHORS = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_object_propagated_annotations.csv"
V52B3_TRACKS = W8 / "outputs" / "v52b3_hybrid_tracking_full" / "week8_v52b3_hybrid_tracking_rows.csv"
V52C_CLIP_QA = W8 / "outputs" / "v52c_full_hybrid_tracking_qa" / "week8_v52c_clip_quality_assessment.csv"
V52C_REVIEW_QUEUE = W8 / "outputs" / "v52c_full_hybrid_tracking_qa" / "week8_v52c_manual_review_queue.csv"

NOTES_CSV = W8 / "validation" / "week8_v52d_visual_validation_notes.csv"


def clean_str(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


def load_data():
    clips_raw = json.loads(V45_CLIP_JSON.read_text()).get("clips", [])
    clips = []
    for c in clips_raw:
        d = dict(c)
        d["scan_frame_id"] = clean_str(d.get("scan_frame_id"))
        d["clip_path"] = clean_str(d.get("clip_path"))
        d["video_id"] = clean_str(d.get("video_id"))
        clips.append(d)

    anchors = pd.read_csv(V45_ANCHORS)
    tracks = pd.read_csv(V52B3_TRACKS)
    clip_qa = pd.read_csv(V52C_CLIP_QA)

    for df in [anchors, tracks, clip_qa]:
        if "scan_frame_id" in df.columns:
            df["scan_frame_id"] = df["scan_frame_id"].astype(str).map(clean_str)

    for col in ["frame_index_in_clip", "x1", "y1", "x2", "y2"]:
        if col in tracks.columns:
            tracks[col] = pd.to_numeric(tracks[col], errors="coerce")

    for col in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
        if col in anchors.columns:
            anchors[col] = pd.to_numeric(anchors[col], errors="coerce")

    qa_map = {r["scan_frame_id"]: r.to_dict() for _, r in clip_qa.iterrows()}

    clip_list = []
    for c in clips:
        scan = c["scan_frame_id"]
        q = qa_map.get(scan, {})
        clip_list.append({
            "scan_frame_id": scan,
            "video_id": c.get("video_id", ""),
            "clip_path": c.get("clip_path", ""),
            "fps_used": c.get("fps_used", None),
            "duration_sec": c.get("duration_sec", None),
            "frame_count": int(q.get("frame_count", c.get("frame_count", 0)) or 0),
            "quality_tier": clean_str(q.get("quality_tier", "")),
            "manual_review_recommendation": clean_str(q.get("manual_review_recommendation", "")),
            "draw_ok_ratio": float(q.get("draw_ok_ratio", 0) or 0),
            "missing_ratio": float(q.get("missing_ratio", 0) or 0),
            "stable_tracklet_ratio": float(q.get("stable_tracklet_ratio", 0) or 0),
            "recall_fallback_ratio": float(q.get("recall_fallback_ratio", 0) or 0),
            "review_needed_ratio": float(q.get("review_needed_ratio", 0) or 0),
        })

    return clips, clip_list, anchors, tracks, clip_qa


CLIPS_RAW, CLIPS, ANCHORS, TRACKS, CLIP_QA = load_data()
CLIP_MAP = {c["scan_frame_id"]: c for c in CLIPS_RAW}


def json_response(handler, obj, status=200):
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def file_response(handler, path):
    path = Path(path)
    if not path.exists() or not path.is_file():
        handler.send_error(404)
        return

    ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    data = path.read_bytes()

    handler.send_response(200)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def video_response(handler, video_path):
    path = Path(video_path)
    if not path.exists() or not path.is_file():
        handler.send_error(404, "Video file not found")
        return

    file_size = path.stat().st_size
    range_header = handler.headers.get("Range")

    if range_header:
        try:
            units, rng = range_header.split("=")
            start_s, end_s = rng.split("-")
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1

            handler.send_response(206)
            handler.send_header("Content-Type", "video/mp4")
            handler.send_header("Accept-Ranges", "bytes")
            handler.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            handler.send_header("Content-Length", str(length))
            handler.end_headers()

            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    handler.wfile.write(chunk)
                    remaining -= len(chunk)
            return
        except Exception:
            pass

    handler.send_response(200)
    handler.send_header("Content-Type", "video/mp4")
    handler.send_header("Accept-Ranges", "bytes")
    handler.send_header("Content-Length", str(file_size))
    handler.end_headers()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            handler.wfile.write(chunk)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            return file_response(self, STATIC / "index.html")

        if path.startswith("/static/"):
            rel = path.replace("/static/", "", 1)
            return file_response(self, STATIC / rel)

        if path == "/api/clips":
            return json_response(self, {"clips": CLIPS})

        if path == "/api/clip":
            scan = qs.get("scan", [""])[0]
            clip = CLIP_MAP.get(scan)
            if clip is None:
                return json_response(self, {"error": "scan not found"}, status=404)

            qdf = CLIP_QA[CLIP_QA["scan_frame_id"] == scan].copy()
            qa = qdf.iloc[0].to_dict() if len(qdf) else {}

            return json_response(self, {"clip": clip, "qa": qa})

        if path == "/api/annotations":
            scan = qs.get("scan", [""])[0]
            if scan == "":
                return json_response(self, {"error": "scan missing"}, status=400)

            a = ANCHORS[ANCHORS["scan_frame_id"] == scan].copy()
            t = TRACKS[TRACKS["scan_frame_id"] == scan].copy()

            anchors_out = []
            for _, r in a.iterrows():
                anchors_out.append({
                    "scan_frame_id": scan,
                    "final_box_id": clean_str(r.get("final_box_id")),
                    "behaviour_pig_id": clean_str(r.get("behaviour_pig_id")),
                    "visual_marker_colour": clean_str(r.get("visual_marker_colour")),
                    "behaviour_code": clean_str(r.get("behaviour_code")),
                    "x1": float(r.get("bbox_x1", 0) or 0),
                    "y1": float(r.get("bbox_y1", 0) or 0),
                    "x2": float(r.get("bbox_x2", 0) or 0),
                    "y2": float(r.get("bbox_y2", 0) or 0),
                    "source": "anchor_gt",
                })

            tracks_out = []
            wanted_cols = [
                "scan_frame_id",
                "frame_index_in_clip",
                "final_box_id",
                "behaviour_pig_id",
                "visual_marker_colour",
                "behaviour_code",
                "x1",
                "y1",
                "x2",
                "y2",
                "hybrid_source",
                "hybrid_confidence",
                "review_needed",
                "draw_ok",
                "v52b2_status",
                "v52b1_status",
            ]

            for _, r in t.iterrows():
                d = {}
                for c in wanted_cols:
                    v = r.get(c, "")
                    if c in ["x1", "y1", "x2", "y2"]:
                        d[c] = float(v) if pd.notna(v) else 0
                    elif c == "frame_index_in_clip":
                        d[c] = int(v) if pd.notna(v) else 0
                    elif c in ["review_needed", "draw_ok"]:
                        d[c] = str(v).lower() in ["true", "1", "yes"]
                    else:
                        d[c] = clean_str(v)
                tracks_out.append(d)

            return json_response(self, {"scan_frame_id": scan, "anchors": anchors_out, "tracks": tracks_out})

        if path == "/video":
            scan = qs.get("scan", [""])[0]
            clip = CLIP_MAP.get(scan)
            if clip is None:
                self.send_error(404, "clip not found")
                return
            return video_response(self, clip.get("clip_path", ""))

        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/save_note":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(body)
        except Exception:
            return json_response(self, {"ok": False, "error": "invalid json"}, status=400)

        row = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "scan_frame_id": clean_str(payload.get("scan_frame_id")),
            "video_id": clean_str(payload.get("video_id")),
            "time_sec": clean_str(payload.get("time_sec")),
            "frame_index_in_clip": clean_str(payload.get("frame_index_in_clip")),
            "issue_type": clean_str(payload.get("issue_type")),
            "severity": clean_str(payload.get("severity")),
            "note": clean_str(payload.get("note")),
        }

        exists = NOTES_CSV.exists()
        with open(NOTES_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()), quoting=csv.QUOTE_ALL, escapechar="\\")
            if not exists:
                writer.writeheader()
            writer.writerow(row)

        return json_response(self, {"ok": True, "saved_to": str(NOTES_CSV)})


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8526)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Week 8 v52d visualizer running at http://{args.host}:{args.port}")
    print("Open through SSH tunnel/local browser, for example: http://127.0.0.1:8526/")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
