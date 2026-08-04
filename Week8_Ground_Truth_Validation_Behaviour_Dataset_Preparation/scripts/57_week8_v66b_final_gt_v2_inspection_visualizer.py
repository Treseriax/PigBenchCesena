from pathlib import Path
from datetime import datetime
import csv
import json
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V66A = W8 / "outputs" / "v66a_final_gt_v2_label_propagation"
OBJECTS = V66A / "week8_v66a_final_gt_v2_object_table.csv"
SCANFRAME_SUMMARY = V66A / "week8_v66a_scanframe_propagation_summary.csv"
QA = V66A / "week8_v66a_propagation_quality_checks.csv"
DECISION66A = V66A / "week8_v66a_decision_summary.csv"

OUT = W8 / "outputs" / "v66b_final_gt_v2_inspection_visualizer"
INTERFACE = W8 / "interface"
STATIC = INTERFACE / "static_v66b"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, INTERFACE, STATIC, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

SERVER = INTERFACE / "week8_visualizer_server_v66b.py"
INDEX = STATIC / "index.html"
APP = STATIC / "app.js"
STYLE = STATIC / "style.css"

OUT_VISUALIZER_JSON = OUT / "week8_v66b_final_gt_v2_visualizer_data.json"
OUT_DECISION = OUT / "week8_v66b_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v66b_issues.csv"
OUT_NOTE = NOTES / "week8_v66b_final_gt_v2_inspection_visualizer_notes.md"
OUT_REPORT = REPORTS / "week8_v66b_final_gt_v2_inspection_visualizer_report.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


def num(x):
    try:
        return float(clean(x))
    except Exception:
        return None


issues = []

for p in [OBJECTS, SCANFRAME_SUMMARY, QA, DECISION66A]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required v66b visualizer input is missing.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v66b_decision": "final_gt_v2_inspection_visualizer_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_visual_inspection": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


objects = pd.read_csv(OBJECTS).fillna("")
scan_summary = pd.read_csv(SCANFRAME_SUMMARY).fillna("")
decision66a = pd.read_csv(DECISION66A).fillna("")

for df in [objects, scan_summary, decision66a]:
    for c in df.columns:
        df[c] = df[c].map(clean)

if len(decision66a) == 0 or decision66a.iloc[0].get("ready_for_v66b_final_gt_visualizer", "False") != "True":
    issues.append({
        "item": str(DECISION66A),
        "issue_type": "hard_v66a_not_ready",
        "issue_detail": "v66a decision does not mark ready_for_v66b_final_gt_visualizer=True.",
        "severity": "hard",
    })

required_cols = [
    "scan_frame_id",
    "video_id",
    "canonical_colour_label_norm",
    "behaviour_code",
    "manual_bbox_x1",
    "manual_bbox_y1",
    "manual_bbox_x2",
    "manual_bbox_y2",
    "manual_bbox_status",
    "manual_identity_status",
    "manual_gt_v2_status",
    "classification_gate_v66a",
    "final_gt_v2_category_v66a",
    "clip_path",
]

missing_cols = [c for c in required_cols if c not in objects.columns]
if missing_cols:
    issues.append({
        "item": str(OBJECTS),
        "issue_type": "hard_missing_required_columns",
        "issue_detail": "Missing columns: " + ";".join(missing_cols),
        "severity": "hard",
    })

# Build visualizer JSON.
scanframes = []

for scan, g in objects.groupby("scan_frame_id"):
    g = g.copy()

    summary_row = scan_summary[scan_summary["scan_frame_id"] == scan]
    summary = summary_row.iloc[0].to_dict() if len(summary_row) else {}

    clip_path = clean(g["clip_path"].iloc[0]) if "clip_path" in g.columns else ""

    overlay_objects = []
    for _, r in g.iterrows():
        x1 = num(r.get("manual_bbox_x1", ""))
        y1 = num(r.get("manual_bbox_y1", ""))
        x2 = num(r.get("manual_bbox_x2", ""))
        y2 = num(r.get("manual_bbox_y2", ""))

        has_bbox = all(v is not None for v in [x1, y1, x2, y2]) and x2 > x1 and y2 > y1

        overlay_objects.append({
            "canonical_gt_object_id": clean(r.get("canonical_gt_object_id", "")),
            "scan_frame_id": clean(r.get("scan_frame_id", "")),
            "video_id": clean(r.get("video_id", "")),
            "canonical_colour_label_norm": clean(r.get("canonical_colour_label_norm", "")),
            "canonical_colour_label_raw": clean(r.get("canonical_colour_label_raw", "")),
            "behaviour_code": clean(r.get("behaviour_code", "")),
            "behaviour_raw": clean(r.get("behaviour_raw", "")),
            "manual_assigned_candidate_box_id": clean(r.get("manual_assigned_candidate_box_id", "")),
            "bbox": [x1, y1, x2, y2] if has_bbox else None,
            "has_valid_bbox": bool(has_bbox),
            "manual_bbox_status": clean(r.get("manual_bbox_status", "")),
            "manual_identity_status": clean(r.get("manual_identity_status", "")),
            "manual_gt_v2_status": clean(r.get("manual_gt_v2_status", "")),
            "manual_classification_use": clean(r.get("manual_classification_use", "")),
            "final_gt_v2_category": clean(r.get("final_gt_v2_category_v66a", "")),
            "classification_gate": clean(r.get("classification_gate_v66a", "")),
            "recommended_split": clean(r.get("recommended_split", "")),
            "reviewer_note": clean(r.get("manual_reviewer_note", "")),
            "bbox_propagation_status": clean(r.get("bbox_propagation_status", "")),
        })

    scanframes.append({
        "scan_frame_id": scan,
        "video_id": clean(g["video_id"].iloc[0]),
        "clip_path": clip_path,
        "scanframe_summary": summary,
        "objects": overlay_objects,
    })

data = {
    "visualizer_version": "v66b_final_gt_v2_inspection_visualizer",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "source_object_table": str(OBJECTS),
    "source_scanframe_summary": str(SCANFRAME_SUMMARY),
    "usage_note": "Inspection visualizer only. It does not edit GT. Static manual anchor boxes are shown for GT label inspection, not tracking-quality claims.",
    "category_legend": {
        "strict_gold_classification": "Use for strict classification baseline.",
        "caution_analysis": "Qualitative analysis only; excluded from strict classification.",
        "nonusable_fix_or_excluded": "Do not use for classification training/evaluation.",
    },
    "scanframes": scanframes,
}

OUT_VISUALIZER_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False))

server_code = r'''
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote
import argparse
import json
import mimetypes
import math


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

DATA_JSON = W8 / "outputs" / "v66b_final_gt_v2_inspection_visualizer" / "week8_v66b_final_gt_v2_visualizer_data.json"
STATIC = W8 / "interface" / "static_v66b"


def sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


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
            if not DATA_JSON.exists():
                self._send_json({"error": "data json missing", "path": str(DATA_JSON)}, 500)
                return
            self._send_json(json.loads(DATA_JSON.read_text(encoding="utf-8")))
            return

        if path == "/video":
            p = qs.get("path", [""])[0]
            self._serve_video(unquote(p))
            return

        self._send_text("Not found", 404)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8528)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Week8 v66b Final GT v2 inspection visualizer running on http://{args.host}:{args.port}/")
    print(f"Data JSON: {DATA_JSON}")
    server.serve_forever()


if __name__ == "__main__":
    main()
'''

html_code = r'''
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Week8 v66b Final GT v2 Inspection</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <div id="app">
    <aside id="sidebar">
      <h2>Final GT v2</h2>
      <div class="small">Inspection only. No GT editing here.</div>

      <label>Category</label>
      <select id="categoryFilter">
        <option value="">All</option>
        <option value="strict_gold_classification">strict_gold_classification</option>
        <option value="caution_analysis">caution_analysis</option>
        <option value="nonusable_fix_or_excluded">nonusable_fix_or_excluded</option>
      </select>

      <label>Behaviour</label>
      <select id="behaviourFilter">
        <option value="">All</option>
      </select>

      <label>Search scanframe</label>
      <input id="searchBox" placeholder="scanframe_0035">

      <label>
        <input type="checkbox" id="onlyProblemClips">
        show clips with nonusable/caution
      </label>

      <button id="refreshBtn">Refresh</button>

      <div id="summary" class="summary"></div>
      <div id="scanList"></div>
    </aside>

    <main>
      <section id="top">
        <div>
          <h1 id="title">Select scanframe</h1>
          <div id="subtitle"></div>
        </div>

        <div class="toggles">
          <label><input type="checkbox" id="showStrict" checked> strict</label>
          <label><input type="checkbox" id="showCaution" checked> caution</label>
          <label><input type="checkbox" id="showNonusable" checked> nonusable</label>
          <label><input type="checkbox" id="showLabels" checked> labels</label>
        </div>
      </section>

      <section id="videoSection">
        <div id="videoWrap">
          <video id="video" controls muted></video>
          <canvas id="overlay"></canvas>
        </div>

        <div id="seekButtons">
          <button data-seek="0">0s</button>
          <button data-seek="2.5">2.5s</button>
          <button data-seek="5">5s</button>
          <button data-seek="7.5">7.5s</button>
          <button data-seek="9.9">10s</button>
        </div>
      </section>

      <section id="objectPanel">
        <h2>Objects</h2>
        <div id="objects"></div>
      </section>
    </main>
  </div>

  <script src="/static/app.js"></script>
</body>
</html>
'''

css_code = r'''
* { box-sizing: border-box; }

body {
  margin: 0;
  font-family: Arial, sans-serif;
  background: #101010;
  color: #eee;
}

#app {
  display: flex;
  height: 100vh;
  width: 100vw;
}

#sidebar {
  width: 370px;
  padding: 14px;
  background: #181818;
  border-right: 1px solid #333;
  overflow-y: auto;
}

main {
  flex: 1;
  overflow-y: auto;
}

h1, h2 {
  margin: 0 0 8px 0;
}

.small {
  font-size: 12px;
  color: #aaa;
}

label {
  display: block;
  margin-top: 10px;
  margin-bottom: 4px;
  font-size: 13px;
}

select, input, button {
  width: 100%;
  background: #222;
  color: #eee;
  border: 1px solid #555;
  border-radius: 4px;
  padding: 7px;
}

button {
  cursor: pointer;
  margin-top: 8px;
}

button:hover {
  background: #333;
}

.summary {
  white-space: pre-wrap;
  color: #ccc;
  font-size: 12px;
  margin: 12px 0;
}

.scanItem {
  padding: 8px;
  border: 1px solid #333;
  border-radius: 6px;
  margin-bottom: 6px;
  background: #202020;
  cursor: pointer;
}

.scanItem:hover {
  background: #2a2a2a;
}

.scanItem.active {
  outline: 2px solid #ccc;
}

.badge {
  display: inline-block;
  padding: 2px 5px;
  border-radius: 4px;
  margin-top: 4px;
  font-size: 11px;
  background: #333;
}

#top {
  padding: 14px;
  border-bottom: 1px solid #333;
  display: flex;
  justify-content: space-between;
  gap: 20px;
}

.toggles {
  display: flex;
  gap: 12px;
  align-items: center;
}

.toggles label {
  margin: 0;
}

#videoSection {
  padding: 14px;
  border-bottom: 1px solid #333;
}

#videoWrap {
  position: relative;
  width: fit-content;
  max-width: 100%;
}

#video {
  max-width: calc(100vw - 420px);
  max-height: 55vh;
  background: black;
}

#overlay {
  position: absolute;
  left: 0;
  top: 0;
  pointer-events: none;
}

#seekButtons {
  display: flex;
  gap: 8px;
  margin-top: 8px;
}

#seekButtons button {
  width: auto;
}

#objectPanel {
  padding: 14px;
}

.objCard {
  border: 1px solid #333;
  border-radius: 8px;
  padding: 10px;
  margin-bottom: 8px;
  background: #181818;
}

.objHeader {
  display: flex;
  justify-content: space-between;
  gap: 10px;
}

.strict {
  color: #9ee59e;
}

.caution {
  color: #ffcf75;
}

.nonusable {
  color: #ff8888;
}

.kv {
  font-size: 12px;
  color: #bbb;
  margin-top: 5px;
  line-height: 1.35;
}
'''

js_code = r'''
let DATA = null;
let CURRENT = null;

function byId(id) { return document.getElementById(id); }

function clean(v) {
  if (v === null || v === undefined) return "";
  return String(v);
}

function catClass(cat) {
  if (cat === "strict_gold_classification") return "strict";
  if (cat === "caution_analysis") return "caution";
  return "nonusable";
}

async function loadData() {
  const resp = await fetch("/api/data");
  DATA = await resp.json();

  populateBehaviourFilter();
  renderSidebar();

  if (!CURRENT && DATA.scanframes.length) {
    selectScanframe(DATA.scanframes[0].scan_frame_id);
  }
}

function populateBehaviourFilter() {
  const set = new Set();
  for (const sf of DATA.scanframes || []) {
    for (const o of sf.objects || []) {
      if (o.behaviour_code) set.add(o.behaviour_code);
    }
  }

  const sel = byId("behaviourFilter");
  const current = sel.value;
  sel.innerHTML = '<option value="">All</option>';
  [...set].sort().forEach(b => {
    const opt = document.createElement("option");
    opt.value = b;
    opt.textContent = b;
    sel.appendChild(opt);
  });
  sel.value = current;
}

function scanframeMatches(sf) {
  const cat = byId("categoryFilter").value;
  const beh = byId("behaviourFilter").value;
  const q = byId("searchBox").value.trim().toLowerCase();
  const onlyProblems = byId("onlyProblemClips").checked;

  if (q && !sf.scan_frame_id.toLowerCase().includes(q)) return false;

  const objs = sf.objects || [];

  if (cat && !objs.some(o => o.final_gt_v2_category === cat)) return false;
  if (beh && !objs.some(o => o.behaviour_code === beh)) return false;

  if (onlyProblems) {
    const hasProblem = objs.some(o => o.final_gt_v2_category !== "strict_gold_classification");
    if (!hasProblem) return false;
  }

  return true;
}

function renderSidebar() {
  const list = byId("scanList");
  list.innerHTML = "";

  let counts = {
    scanframes: 0,
    strict_objects: 0,
    caution_objects: 0,
    nonusable_objects: 0,
  };

  for (const sf of DATA.scanframes || []) {
    if (!scanframeMatches(sf)) continue;

    counts.scanframes += 1;
    const objs = sf.objects || [];
    counts.strict_objects += objs.filter(o => o.final_gt_v2_category === "strict_gold_classification").length;
    counts.caution_objects += objs.filter(o => o.final_gt_v2_category === "caution_analysis").length;
    counts.nonusable_objects += objs.filter(o => o.final_gt_v2_category === "nonusable_fix_or_excluded").length;

    const div = document.createElement("div");
    div.className = "scanItem";
    if (CURRENT && CURRENT.scan_frame_id === sf.scan_frame_id) div.classList.add("active");

    const strict = objs.filter(o => o.final_gt_v2_category === "strict_gold_classification").length;
    const caution = objs.filter(o => o.final_gt_v2_category === "caution_analysis").length;
    const nonusable = objs.filter(o => o.final_gt_v2_category === "nonusable_fix_or_excluded").length;

    div.innerHTML = `
      <div><b>${sf.scan_frame_id}</b></div>
      <div class="small">${sf.video_id}</div>
      <span class="badge strict">strict=${strict}</span>
      <span class="badge caution">caution=${caution}</span>
      <span class="badge nonusable">nonusable=${nonusable}</span>
    `;

    div.onclick = () => selectScanframe(sf.scan_frame_id);
    list.appendChild(div);
  }

  byId("summary").textContent =
    `scanframes: ${counts.scanframes}\n` +
    `strict objects: ${counts.strict_objects}\n` +
    `caution objects: ${counts.caution_objects}\n` +
    `nonusable objects: ${counts.nonusable_objects}`;
}

function selectScanframe(scanId) {
  CURRENT = DATA.scanframes.find(x => x.scan_frame_id === scanId);

  byId("title").textContent = CURRENT.scan_frame_id;
  byId("subtitle").textContent = CURRENT.video_id;

  const video = byId("video");
  video.src = CURRENT.clip_path ? `/video?path=${encodeURIComponent(CURRENT.clip_path)}` : "";
  video.load();

  renderObjects();
  renderSidebar();
  setTimeout(drawOverlay, 250);
}

function visibleObject(o) {
  if (o.final_gt_v2_category === "strict_gold_classification" && !byId("showStrict").checked) return false;
  if (o.final_gt_v2_category === "caution_analysis" && !byId("showCaution").checked) return false;
  if (o.final_gt_v2_category === "nonusable_fix_or_excluded" && !byId("showNonusable").checked) return false;
  return true;
}

function renderObjects() {
  const box = byId("objects");
  box.innerHTML = "";

  const objs = CURRENT.objects || [];

  for (const o of objs) {
    const cls = catClass(o.final_gt_v2_category);
    const div = document.createElement("div");
    div.className = "objCard";

    div.innerHTML = `
      <div class="objHeader">
        <div>
          <b class="${cls}">${o.canonical_colour_label_norm}</b>
          <span> / behaviour: <b>${o.behaviour_code}</b></span>
        </div>
        <div class="${cls}">${o.final_gt_v2_category}</div>
      </div>
      <div class="kv">
        object: ${o.canonical_gt_object_id}<br>
        candidate box: ${o.manual_assigned_candidate_box_id}<br>
        bbox status: ${o.manual_bbox_status}<br>
        identity status: ${o.manual_identity_status}<br>
        gt status: ${o.manual_gt_v2_status}<br>
        classification gate: ${o.classification_gate}<br>
        split: ${o.recommended_split || "-"}<br>
        note: ${o.reviewer_note || "-"}
      </div>
    `;

    box.appendChild(div);
  }
}

function resizeCanvas() {
  const video = byId("video");
  const canvas = byId("overlay");
  const rect = video.getBoundingClientRect();

  canvas.width = rect.width;
  canvas.height = rect.height;
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
}

function drawOverlay() {
  const video = byId("video");
  const canvas = byId("overlay");
  const ctx = canvas.getContext("2d");

  resizeCanvas();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!CURRENT || !video.videoWidth || !video.videoHeight) return;

  const sx = canvas.width / video.videoWidth;
  const sy = canvas.height / video.videoHeight;

  for (const o of CURRENT.objects || []) {
    if (!visibleObject(o)) continue;
    if (!o.has_valid_bbox || !o.bbox) continue;

    const [x1, y1, x2, y2] = o.bbox;
    const cls = catClass(o.final_gt_v2_category);

    if (cls === "strict") ctx.strokeStyle = "lime";
    else if (cls === "caution") ctx.strokeStyle = "yellow";
    else ctx.strokeStyle = "red";

    ctx.lineWidth = cls === "strict" ? 2 : 3;
    ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);

    if (byId("showLabels").checked) {
      ctx.fillStyle = ctx.strokeStyle;
      ctx.font = "14px Arial";
      const label = `${o.canonical_colour_label_norm} / ${o.behaviour_code}`;
      ctx.fillText(label, x1 * sx + 4, y1 * sy + 16);
    }
  }
}

byId("categoryFilter").onchange = renderSidebar;
byId("behaviourFilter").onchange = renderSidebar;
byId("searchBox").oninput = renderSidebar;
byId("onlyProblemClips").onchange = renderSidebar;
byId("refreshBtn").onclick = loadData;

for (const id of ["showStrict", "showCaution", "showNonusable", "showLabels"]) {
  byId(id).onchange = drawOverlay;
}

byId("video").addEventListener("loadedmetadata", drawOverlay);
byId("video").addEventListener("timeupdate", drawOverlay);
window.addEventListener("resize", drawOverlay);

document.querySelectorAll("#seekButtons button").forEach(btn => {
  btn.onclick = () => {
    byId("video").currentTime = Number(btn.dataset.seek);
    drawOverlay();
  };
});

loadData();
'''

SERVER.write_text(server_code)
INDEX.write_text(html_code)
APP.write_text(js_code)
STYLE.write_text(css_code)

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0

decision = pd.DataFrame([{
    "v66b_decision": "final_gt_v2_inspection_visualizer_created" if hard_issue_count == 0 else "final_gt_v2_inspection_visualizer_created_with_blocking_issues",
    "scanframe_count": int(objects["scan_frame_id"].nunique()),
    "object_count": int(len(objects)),
    "strict_gold_objects": int((objects["final_gt_v2_category_v66a"] == "strict_gold_classification").sum()),
    "caution_objects": int((objects["final_gt_v2_category_v66a"] == "caution_analysis").sum()),
    "nonusable_objects": int((objects["final_gt_v2_category_v66a"] == "nonusable_fix_or_excluded").sum()),
    "visualizer_json": str(OUT_VISUALIZER_JSON),
    "server_path": str(SERVER),
    "static_dir": str(STATIC),
    "hard_issue_count": hard_issue_count,
    "issue_count": int(len(issues_df)),
    "ready_for_visual_inspection": bool(hard_issue_count == 0),
    "ready_for_v66c_tracking_helper_reattachment": bool(hard_issue_count == 0),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v66b Final GT v2 Inspection Visualizer\n\n"
    f"- v66b decision: {decision.iloc[0]['v66b_decision']}\n"
    f"- Scanframes: {objects['scan_frame_id'].nunique()}\n"
    f"- Objects: {len(objects)}\n"
    f"- Strict gold objects: {int((objects['final_gt_v2_category_v66a'] == 'strict_gold_classification').sum())}\n"
    f"- Caution objects: {int((objects['final_gt_v2_category_v66a'] == 'caution_analysis').sum())}\n"
    f"- Nonusable objects: {int((objects['final_gt_v2_category_v66a'] == 'nonusable_fix_or_excluded').sum())}\n"
    f"- Server: {SERVER}\n"
    f"- Static dir: {STATIC}\n"
    f"- Ready for visual inspection: {bool(hard_issue_count == 0)}\n\n"
    "This is an inspection interface. It does not edit GT. Static manual anchor boxes are shown for GT label inspection, not as tracking-quality evidence.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v66b Final GT v2 Inspection Visualizer Report\n\n"
    f"Decision: {decision.iloc[0]['v66b_decision']}\n\n"
    f"Visualizer JSON: `{OUT_VISUALIZER_JSON}`\n\n"
    f"Server: `{SERVER}`\n\n"
    "Use this tool to inspect final GT categories and behaviour labels over the 72 clips.\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v66b",
    "task_name": "Final GT v2 inspection visualizer",
    "status": "PASS" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": str(OBJECTS),
    "output_summary": str(OUT),
    "hard_issues": hard_issue_count,
    "warnings": 0,
    "next_action": "Inspect final GT v2 visually, then reattach tracking helper layer.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_VISUALIZER_JSON)
print(SERVER)
print(INDEX)
print(APP)
print(STYLE)
print(OUT_DECISION)
print(OUT_NOTE)

print()
print("=== v66b decision ===")
print(decision.to_string(index=False))

print()
print("=== issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
