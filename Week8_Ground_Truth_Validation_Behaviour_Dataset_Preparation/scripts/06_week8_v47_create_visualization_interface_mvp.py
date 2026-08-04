from pathlib import Path
from datetime import datetime
import json
import csv
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

INTERFACE = W8 / "interface"
STATIC = INTERFACE / "static"
OUT = W8 / "outputs" / "v47_visualization_interface_mvp"
VALIDATION = W8 / "validation"
DOCS = W8 / "docs"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [INTERFACE, STATIC, OUT, VALIDATION, DOCS, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

V45_JSON = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_level_ground_truth.json"
V46_DECISION = W8 / "outputs" / "propagated_ground_truth" / "v46_propagation_qa" / "week8_v46_decision_summary.csv"

SERVER_PATH = INTERFACE / "week8_visualizer_server_v47.py"
INDEX_PATH = STATIC / "index.html"
APP_PATH = STATIC / "app.js"
STYLE_PATH = STATIC / "style.css"
NOTES_CSV = VALIDATION / "week8_v47_visual_validation_notes.csv"

OUT_DECISION = OUT / "week8_v47_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v47_issues.csv"
OUT_MANIFEST = OUT / "week8_v47_interface_manifest.csv"
OUT_REPORT = REPORTS / "week8_v47_visualization_interface_mvp_report.md"
OUT_NOTE = NOTES / "week8_v47_visualization_interface_mvp_notes.md"
OUT_QUICKSTART = DOCS / "week8_v47_visualizer_quickstart.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


issues = []

if not V45_JSON.exists():
    issues.append({
        "item": "v45_clip_level_ground_truth",
        "issue_type": "hard_missing_v45_json",
        "issue_detail": str(V45_JSON),
        "severity": "hard",
    })

if not V46_DECISION.exists():
    issues.append({
        "item": "v46_decision",
        "issue_type": "hard_missing_v46_decision",
        "issue_detail": str(V46_DECISION),
        "severity": "hard",
    })

clip_count = 0
object_count = 0
clip_paths_exist = 0

if V45_JSON.exists():
    data = json.loads(V45_JSON.read_text())
    clips = data.get("clips", [])
    clip_count = len(clips)
    object_count = sum(len(c.get("objects", [])) for c in clips)
    clip_paths_exist = sum(1 for c in clips if Path(c.get("clip_path", "")).exists())

    if clip_count != 72:
        issues.append({
            "item": "clip_count",
            "issue_type": "warning_unexpected_clip_count",
            "issue_detail": f"Expected 72, got {clip_count}",
            "severity": "warning",
        })

    if object_count != 429:
        issues.append({
            "item": "object_count",
            "issue_type": "warning_unexpected_object_count",
            "issue_detail": f"Expected 429, got {object_count}",
            "severity": "warning",
        })

    if clip_paths_exist != clip_count:
        issues.append({
            "item": "clip_paths",
            "issue_type": "warning_missing_clip_paths",
            "issue_detail": f"{clip_paths_exist}/{clip_count} clip paths exist.",
            "severity": "warning",
        })


server_code = r'''#!/usr/bin/env python3
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
'''

index_html = '''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Week 8 Behaviour Dataset Visualizer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <header>
    <div>
      <h1>Week 8 Behaviour Dataset Visualizer</h1>
      <p>Video + propagated ground truth validation interface</p>
    </div>
    <div id="statusBadge">Loading...</div>
  </header>

  <main>
    <aside>
      <h2>Clips</h2>
      <input id="clipSearch" type="text" placeholder="Search scanframe / behaviour / split">
      <select id="clipSelect" size="20"></select>

      <section class="panel">
        <h3>Overlay layers</h3>
        <label><input type="checkbox" id="showBoxes" checked> Bounding boxes</label>
        <label><input type="checkbox" id="showPigId" checked> Pig ID</label>
        <label><input type="checkbox" id="showColour" checked> Colour identity</label>
        <label><input type="checkbox" id="showBehaviour" checked> Behaviour label</label>
        <label><input type="checkbox" id="showTimestamp" checked> Frame / timestamp</label>
        <label><input type="checkbox" id="showFlags" checked> Validation flags</label>
      </section>

      <section class="panel">
        <h3>Validation note</h3>
        <label>Issue type</label>
        <select id="issueType">
          <option value="colour_error">colour_error</option>
          <option value="behaviour_error">behaviour_error</option>
          <option value="identity_switch">identity_switch</option>
          <option value="missing_label">missing_label</option>
          <option value="wrong_bbox">wrong_bbox</option>
          <option value="occlusion">occlusion</option>
          <option value="uncertain">uncertain</option>
          <option value="temporal_inconsistency">temporal_inconsistency</option>
          <option value="review_required">review_required</option>
          <option value="other">other</option>
        </select>

        <label>Severity</label>
        <select id="severity">
          <option value="info">info</option>
          <option value="warning">warning</option>
          <option value="hard">hard</option>
        </select>

        <textarea id="noteText" placeholder="Write validation note..."></textarea>
        <button id="saveNote">Save note</button>
        <div id="noteStatus"></div>
      </section>
    </aside>

    <section class="viewer">
      <div class="videoWrap">
        <video id="video" controls muted></video>
        <canvas id="overlay"></canvas>
      </div>

      <div class="infoGrid">
        <div><strong>Scan frame:</strong> <span id="scanInfo">-</span></div>
        <div><strong>Video ID:</strong> <span id="videoInfo">-</span></div>
        <div><strong>Split:</strong> <span id="splitInfo">-</span></div>
        <div><strong>Behaviour set:</strong> <span id="behaviourInfo">-</span></div>
        <div><strong>Current time:</strong> <span id="timeInfo">-</span></div>
        <div><strong>Estimated frame:</strong> <span id="frameInfo">-</span></div>
        <div><strong>Objects:</strong> <span id="objectInfo">-</span></div>
      </div>

      <section class="panel">
        <h3>Objects in current clip</h3>
        <table id="objectTable">
          <thead>
            <tr>
              <th>Box ID</th>
              <th>Colour</th>
              <th>Pig ID</th>
              <th>Behaviour</th>
              <th>Status</th>
              <th>Flags</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </section>
    </section>
  </main>

  <script src="/static/app.js"></script>
</body>
</html>
'''

app_js = '''let clips = [];
let currentClip = null;

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const clipSelect = document.getElementById("clipSelect");
const clipSearch = document.getElementById("clipSearch");

const layerIds = ["showBoxes", "showPigId", "showColour", "showBehaviour", "showTimestamp", "showFlags"];

const colourMap = {
  blue: "#2b6cff",
  green: "#22aa55",
  cyan: "#00bcd4",
  red: "#ff3333",
  pink: "#ff4fd8",
  purple: "#9b59ff",
  unknown: "#aaaaaa",
  not_visible: "#777777",
  uncertain: "#ffcc00",
  unassigned: "#999999"
};

function getLayers() {
  const layers = {};
  for (const id of layerIds) {
    layers[id] = document.getElementById(id).checked;
  }
  return layers;
}

async function loadClips() {
  const res = await fetch("/api/clips");
  const data = await res.json();
  clips = data.clips;
  document.getElementById("statusBadge").textContent = `${data.clip_count} clips loaded`;
  renderClipOptions();
}

function renderClipOptions() {
  const q = clipSearch.value.toLowerCase();
  clipSelect.innerHTML = "";

  for (const c of clips) {
    const label = `${c.scan_frame_id} | ${c.split || "unmapped"} | ${c.behaviour_set || "-"} | objects=${c.object_count}`;
    if (q && !label.toLowerCase().includes(q)) continue;

    const opt = document.createElement("option");
    opt.value = c.scan_frame_id;
    opt.textContent = label;
    clipSelect.appendChild(opt);
  }

  if (!currentClip && clipSelect.options.length > 0) {
    clipSelect.selectedIndex = 0;
    loadClip(clipSelect.value);
  }
}

async function loadClip(scan) {
  const res = await fetch(`/api/clip/${encodeURIComponent(scan)}`);
  currentClip = await res.json();

  video.src = `/media/${encodeURIComponent(scan)}.mp4`;
  video.load();

  document.getElementById("scanInfo").textContent = currentClip.scan_frame_id;
  document.getElementById("videoInfo").textContent = currentClip.video_id || "-";
  document.getElementById("splitInfo").textContent = currentClip.split || "unmapped";
  document.getElementById("behaviourInfo").textContent = currentClip.behaviour_set || "-";
  document.getElementById("objectInfo").textContent = `${currentClip.objects.length}`;

  renderObjectTable();
  drawOverlay();
}

function renderObjectTable() {
  const tbody = document.querySelector("#objectTable tbody");
  tbody.innerHTML = "";

  for (const o of currentClip.objects || []) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${o.final_box_id || "-"}</td>
      <td>${o.visual_marker_colour || "-"}</td>
      <td>${o.behaviour_pig_id || "-"}</td>
      <td>${o.behaviour_code || "-"}</td>
      <td>${o.identity_status || "-"}</td>
      <td>${(o.validation_flags || []).join("|")}</td>
    `;
    tbody.appendChild(tr);
  }
}

function resizeCanvasToVideo() {
  const rect = video.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
}

function labelTextForObject(o, layers) {
  const parts = [];
  if (layers.showPigId) parts.push(o.behaviour_pig_id || "pig?");
  if (layers.showColour) parts.push(o.visual_marker_colour || "colour?");
  if (layers.showBehaviour) parts.push(o.behaviour_code || "beh?");
  if (layers.showFlags && o.validation_flags && o.validation_flags.length) {
    parts.push(o.validation_flags.join("|"));
  }
  return parts.join(" / ");
}

function drawOverlay() {
  resizeCanvasToVideo();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!currentClip || !video.videoWidth || !video.videoHeight) {
    requestAnimationFrame(drawOverlay);
    return;
  }

  const layers = getLayers();
  const sx = canvas.width / video.videoWidth;
  const sy = canvas.height / video.videoHeight;

  if (layers.showBoxes) {
    for (const o of currentClip.objects || []) {
      const b = o.bbox_xyxy || [0, 0, 0, 0];
      const x1 = b[0] * sx;
      const y1 = b[1] * sy;
      const x2 = b[2] * sx;
      const y2 = b[3] * sy;
      const w = x2 - x1;
      const h = y2 - y1;

      const colour = colourMap[o.visual_marker_colour] || "#ffffff";
      ctx.strokeStyle = colour;
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, w, h);

      const label = labelTextForObject(o, layers);
      if (label) {
        ctx.font = "14px Arial";
        const textWidth = ctx.measureText(label).width;
        ctx.fillStyle = "rgba(0,0,0,0.75)";
        ctx.fillRect(x1, Math.max(0, y1 - 22), textWidth + 10, 20);
        ctx.fillStyle = colour;
        ctx.fillText(label, x1 + 5, Math.max(14, y1 - 7));
      }
    }
  }

  const fps = currentClip.fps_used || 25;
  const estFrame = Math.floor(video.currentTime * fps);
  document.getElementById("timeInfo").textContent = `${video.currentTime.toFixed(2)} s`;
  document.getElementById("frameInfo").textContent = `${estFrame}`;

  if (layers.showTimestamp) {
    ctx.font = "16px Arial";
    const msg = `t=${video.currentTime.toFixed(2)}s | frame≈${estFrame}`;
    ctx.fillStyle = "rgba(0,0,0,0.75)";
    ctx.fillRect(10, 10, ctx.measureText(msg).width + 16, 26);
    ctx.fillStyle = "white";
    ctx.fillText(msg, 18, 29);
  }

  requestAnimationFrame(drawOverlay);
}

async function saveNote() {
  if (!currentClip) return;

  const fps = currentClip.fps_used || 25;
  const layers = getLayers();

  const payload = {
    scan_frame_id: currentClip.scan_frame_id,
    video_id: currentClip.video_id || "",
    video_current_time_sec: Number(video.currentTime.toFixed(3)),
    estimated_frame_index_in_clip: Math.floor(video.currentTime * fps),
    issue_type: document.getElementById("issueType").value,
    severity: document.getElementById("severity").value,
    note: document.getElementById("noteText").value,
    visible_layers: Object.entries(layers).filter(([k, v]) => v).map(([k]) => k).join("|")
  };

  const res = await fetch("/api/note", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });

  const data = await res.json();
  const status = document.getElementById("noteStatus");

  if (data.ok) {
    status.textContent = "Saved.";
    document.getElementById("noteText").value = "";
  } else {
    status.textContent = "Save failed.";
  }
}

clipSelect.addEventListener("change", () => loadClip(clipSelect.value));
clipSearch.addEventListener("input", renderClipOptions);
document.getElementById("saveNote").addEventListener("click", saveNote);

for (const id of layerIds) {
  document.getElementById(id).addEventListener("change", drawOverlay);
}

video.addEventListener("loadedmetadata", drawOverlay);
window.addEventListener("resize", drawOverlay);

loadClips();
drawOverlay();
'''

style_css = '''* {
  box-sizing: border-box;
}

body {
  margin: 0;
  font-family: Arial, sans-serif;
  background: #111;
  color: #eee;
}

header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 22px;
  background: #1f1f1f;
  border-bottom: 1px solid #333;
}

header h1 {
  margin: 0 0 4px 0;
  font-size: 22px;
}

header p {
  margin: 0;
  color: #aaa;
}

#statusBadge {
  background: #2c7;
  color: #061;
  padding: 8px 12px;
  border-radius: 999px;
  font-weight: bold;
}

main {
  display: grid;
  grid-template-columns: 360px 1fr;
  gap: 16px;
  padding: 16px;
}

aside {
  background: #181818;
  border: 1px solid #333;
  border-radius: 10px;
  padding: 14px;
  height: calc(100vh - 110px);
  overflow: auto;
}

.viewer {
  min-width: 0;
}

input, select, textarea, button {
  width: 100%;
  margin: 6px 0 12px 0;
  padding: 8px;
  border-radius: 6px;
  border: 1px solid #444;
  background: #222;
  color: #eee;
}

button {
  cursor: pointer;
  background: #2b6cff;
  border-color: #2b6cff;
  font-weight: bold;
}

.panel {
  margin-top: 14px;
  padding: 12px;
  background: #202020;
  border: 1px solid #333;
  border-radius: 8px;
}

.panel h3 {
  margin-top: 0;
}

label {
  display: block;
  color: #ddd;
}

label input[type="checkbox"] {
  width: auto;
  margin-right: 8px;
}

.videoWrap {
  position: relative;
  background: black;
  border: 1px solid #333;
  border-radius: 10px;
  overflow: hidden;
  max-width: 100%;
}

video {
  display: block;
  width: 100%;
  max-height: 70vh;
  background: black;
}

canvas {
  position: absolute;
  left: 0;
  top: 0;
  pointer-events: none;
}

.infoGrid {
  display: grid;
  grid-template-columns: repeat(4, minmax(180px, 1fr));
  gap: 8px;
  margin: 12px 0;
}

.infoGrid div {
  background: #1d1d1d;
  border: 1px solid #333;
  border-radius: 8px;
  padding: 8px;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

th, td {
  padding: 6px;
  border-bottom: 1px solid #333;
  text-align: left;
}

th {
  color: #ccc;
}

#noteStatus {
  color: #7f7;
  min-height: 18px;
}
'''

quickstart = f"""# Week 8 v47 Behaviour Dataset Visualizer Quickstart

Purpose:
This interface is the Week 8 MVP visual validation tool. It loads extracted 10-second clips and the propagated ground-truth annotation generated in v45.

Supported functions:
- video playback
- bounding-box overlays
- pig ID / colour identity / behaviour label display
- timestamp and estimated frame display
- checkbox layer toggles
- manual validation note export

Run from the server:
cd ~/PigBench
python Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/interface/week8_visualizer_server_v47.py --port 8513

Open:
http://127.0.0.1:8513/

If port forwarding is needed from your local computer:
ssh -L 8513:127.0.0.1:8513 oyavuz@137.204.72.3

Stop:
Press Ctrl+C in the terminal running the server.

Validation notes:
{NOTES_CSV}

Scope:
v47 is an MVP visual validation interface. It uses the v45 propagated annotations. The v45 bounding boxes are scanpoint-anchor boxes repeated across the interval. Tracking-refined boxes can be integrated in a later refinement step.
"""

SERVER_PATH.write_text(server_code)
INDEX_PATH.write_text(index_html)
APP_PATH.write_text(app_js)
STYLE_PATH.write_text(style_css)
OUT_QUICKSTART.write_text(quickstart)

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

manifest_rows = [
    {"artifact": "server", "path": str(SERVER_PATH), "purpose": "Python HTTP server for interface and video/API endpoints."},
    {"artifact": "index_html", "path": str(INDEX_PATH), "purpose": "Main interface HTML."},
    {"artifact": "app_js", "path": str(APP_PATH), "purpose": "Canvas overlay, clip loading, notes saving."},
    {"artifact": "style_css", "path": str(STYLE_PATH), "purpose": "Interface styling."},
    {"artifact": "notes_csv", "path": str(NOTES_CSV), "purpose": "Manual visual validation notes output."},
    {"artifact": "quickstart", "path": str(OUT_QUICKSTART), "purpose": "How to run and use the interface."},
]

manifest_df = pd.DataFrame(manifest_rows)
safe_to_csv(manifest_df, OUT_MANIFEST)

hard_issues = [x for x in issues if x["severity"] == "hard"]
warnings = [x for x in issues if x["severity"] == "warning"]
ready_for_visual_review = len(hard_issues) == 0

decision = pd.DataFrame([{
    "v47_decision": "visualization_interface_mvp_created" if ready_for_visual_review else "visualization_interface_mvp_blocked",
    "clip_count": int(clip_count),
    "clip_paths_exist": int(clip_paths_exist),
    "object_count": int(object_count),
    "server_path": str(SERVER_PATH),
    "interface_url": "http://127.0.0.1:8513/",
    "notes_csv": str(NOTES_CSV),
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "issue_count": int(len(issues)),
    "ready_for_manual_visual_review": bool(ready_for_visual_review),
    "ready_for_v48_interface_validation_features": bool(ready_for_visual_review),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])

safe_to_csv(decision, OUT_DECISION)
safe_to_csv(issues_df, OUT_ISSUES)

report = f"""# Week 8 v47 Visualization Interface MVP Report

Decision:
- v47 decision: {decision.iloc[0]["v47_decision"]}
- Clip count: {clip_count}
- Existing clip paths: {clip_paths_exist}
- Object count: {object_count}
- Hard issue count: {len(hard_issues)}
- Warning count: {len(warnings)}
- Ready for manual visual review: {ready_for_visual_review}

Interface features:
- Video player
- Clip selector
- Canvas overlay
- Bounding box display
- Pig ID, colour identity and behaviour label display
- Timestamp and estimated frame display
- Checkbox layer toggles
- Manual issue note export

Run:
cd ~/PigBench
python Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/interface/week8_visualizer_server_v47.py --port 8513

Open:
http://127.0.0.1:8513/

Scope:
This is the MVP visual validation interface. It uses v45 propagated ground truth. The v45 bounding boxes are scanpoint-anchor boxes repeated across the interval.
"""

OUT_REPORT.write_text(report)

OUT_NOTE.write_text(
    "# Week 8 v47 Visualization Interface MVP\n\n"
    "## Summary\n\n"
    f"- v47 decision: {decision.iloc[0]['v47_decision']}\n"
    f"- Clip count: {clip_count}\n"
    f"- Existing clip paths: {clip_paths_exist}\n"
    f"- Object count: {object_count}\n"
    f"- Hard issue count: {len(hard_issues)}\n"
    f"- Ready for manual visual review: {ready_for_visual_review}\n\n"
    "## Run\n\n"
    "cd ~/PigBench\n"
    "python Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/interface/week8_visualizer_server_v47.py --port 8513\n\n"
    "Open http://127.0.0.1:8513/\n\n"
    f"Validation notes will be saved to {NOTES_CSV}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v47",
    "task_name": "Behaviour dataset visualization interface MVP",
    "status": "PASS" if ready_for_visual_review else "BLOCKED",
    "input_summary": str(V45_JSON),
    "output_summary": str(INTERFACE),
    "hard_issues": int(len(hard_issues)),
    "warnings": int(len(warnings)),
    "next_action": "Manual visual review and v48 interface validation features" if ready_for_visual_review else "Resolve interface creation issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(SERVER_PATH)
print(INDEX_PATH)
print(APP_PATH)
print(STYLE_PATH)
print(NOTES_CSV)
print(OUT_QUICKSTART)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_REPORT)
print(OUT_NOTE)

print()
print("=== v47 decision ===")
print(decision.to_string(index=False))

print()
print("=== v47 issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
