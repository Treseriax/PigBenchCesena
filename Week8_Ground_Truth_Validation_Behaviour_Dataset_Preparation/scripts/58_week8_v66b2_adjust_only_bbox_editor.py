from pathlib import Path
from datetime import datetime
import csv
import json
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

ASSIGNMENTS = W8 / "validation" / "week8_v63b_manual_gt_v2_assignments.csv"
V66A_OBJECTS = W8 / "outputs" / "v66a_final_gt_v2_label_propagation" / "week8_v66a_final_gt_v2_object_table.csv"

OUT = W8 / "outputs" / "v66b2_adjust_only_bbox_editor"
INTERFACE = W8 / "interface"
STATIC = INTERFACE / "static_v66b2"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, INTERFACE, STATIC, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

ADJUST_TARGETS = OUT / "week8_v66b2_adjust_only_bbox_targets.csv"
VIS_JSON = OUT / "week8_v66b2_adjust_only_visualizer_data.json"
SERVER = INTERFACE / "week8_visualizer_server_v66b2.py"
INDEX = STATIC / "index.html"
APP = STATIC / "app.js"
STYLE = STATIC / "style.css"

DECISION = OUT / "week8_v66b2_decision_summary.csv"
ISSUES = OUT / "week8_v66b2_issues.csv"
NOTE = NOTES / "week8_v66b2_adjust_only_bbox_editor_notes.md"
REPORT = REPORTS / "week8_v66b2_adjust_only_bbox_editor_report.md"
PROGRESS_LOG = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


issues = []

for p in [ASSIGNMENTS, V66A_OBJECTS]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required input missing for adjust-only bbox editor.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, ISSUES)
    decision = pd.DataFrame([{
        "v66b2_decision": "adjust_only_bbox_editor_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_bbox_adjustment": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


assign = pd.read_csv(ASSIGNMENTS).fillna("")
objs = pd.read_csv(V66A_OBJECTS).fillna("")

for df in [assign, objs]:
    for c in df.columns:
        df[c] = df[c].map(clean)

# Only the rows that were explicitly marked as adjust/redraw/draw/new bb.
note = assign["manual_reviewer_note"].astype(str).str.lower()
adjust_note_mask = (
    note.str.contains("adjust", na=False)
    | note.str.contains("redraw", na=False)
    | note.str.contains("new bb", na=False)
    | note.str.contains("manually drawn", na=False)
    | note.str.contains("manual drawing", na=False)
    | note.str.contains("needs to drawn", na=False)
)

# Exclude "no colour visible" cases; keep actual bbox correction targets.
visible_or_fix_mask = (
    ~note.str.contains("no colour", na=False)
    & ~note.str.contains("colour not visible", na=False)
    & ~note.str.contains("color not visible", na=False)
)

target_mask = adjust_note_mask & visible_or_fix_mask

targets = assign[target_mask].copy()

# Merge clip_path and source video metadata from v66a object table.
meta_cols = [
    "canonical_gt_object_id",
    "clip_path",
    "source_video_path",
    "start_sec",
    "end_sec",
    "duration_sec",
    "fps_used",
    "generated_frame_count",
]
meta_cols = [c for c in meta_cols if c in objs.columns]

targets = targets.merge(
    objs[meta_cols].drop_duplicates("canonical_gt_object_id"),
    on="canonical_gt_object_id",
    how="left",
)

safe_to_csv(targets, ADJUST_TARGETS)

scanframes = []
for scan, g in targets.groupby("scan_frame_id"):
    scanframes.append({
        "scan_frame_id": scan,
        "video_id": g["video_id"].iloc[0],
        "clip_path": g["clip_path"].iloc[0] if "clip_path" in g.columns else "",
        "targets": g.to_dict(orient="records"),
    })

vis = {
    "version": "v66b2_adjust_only_bbox_editor",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "assignment_csv": str(ASSIGNMENTS),
    "target_csv": str(ADJUST_TARGETS),
    "rule": "Only explicit adjust/redraw/draw/new-bb targets are shown. Red-exclude no-colour-visible rows are not included.",
    "scanframes": scanframes,
}
VIS_JSON.write_text(json.dumps(vis, indent=2, ensure_ascii=False))

server_code = r'''
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
'''

html_code = r'''
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Week8 v66b2 Adjust-only BBox Editor</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <div id="app">
    <aside>
      <h2>v66b2 Adjust-only</h2>
      <div class="small">Only explicit adjust/redraw targets. No-colour-visible excluded rows are hidden.</div>
      <button id="refreshBtn">Refresh</button>
      <div id="summary"></div>
      <div id="list"></div>
    </aside>

    <main>
      <section id="top">
        <h1 id="title">Select target</h1>
        <div id="subtitle"></div>
      </section>

      <section id="videoSection">
        <div id="videoWrap">
          <video id="video" controls muted></video>
          <canvas id="overlay"></canvas>
        </div>
        <div class="help">
          Draw new bbox by dragging on the video. Then click Save.
        </div>
      </section>

      <section id="editor">
        <h2 id="objTitle"></h2>
        <div id="objInfo"></div>

        <div class="grid">
          <label>x1</label><input id="x1">
          <label>y1</label><input id="y1">
          <label>x2</label><input id="x2">
          <label>y2</label><input id="y2">
        </div>

        <label>Note</label>
        <input id="note" value="BBox adjusted in v66b2">

        <button id="saveBtn">Save bbox and mark strict gold</button>
        <span id="saveMsg"></span>
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
  background: #111;
  color: #eee;
  font-family: Arial, sans-serif;
}

#app {
  display: flex;
  height: 100vh;
}

aside {
  width: 370px;
  overflow-y: auto;
  padding: 14px;
  background: #181818;
  border-right: 1px solid #333;
}

main {
  flex: 1;
  overflow-y: auto;
}

h1, h2 {
  margin: 0 0 8px 0;
}

.small, .help {
  font-size: 12px;
  color: #aaa;
  line-height: 1.35;
}

button, input {
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

#summary {
  white-space: pre-wrap;
  font-size: 12px;
  color: #ccc;
  margin: 12px 0;
}

.item {
  border: 1px solid #333;
  background: #202020;
  padding: 8px;
  border-radius: 6px;
  margin-bottom: 6px;
  cursor: pointer;
}

.item:hover {
  background: #2a2a2a;
}

.item.active {
  outline: 2px solid #ddd;
}

#top, #videoSection, #editor {
  padding: 14px;
  border-bottom: 1px solid #333;
}

#videoWrap {
  position: relative;
  width: fit-content;
}

#video {
  max-width: calc(100vw - 420px);
  max-height: 58vh;
  background: black;
}

#overlay {
  position: absolute;
  left: 0;
  top: 0;
  cursor: crosshair;
}

.grid {
  display: grid;
  grid-template-columns: 60px 160px 60px 160px;
  gap: 8px;
  align-items: center;
  margin-top: 10px;
}

#saveMsg {
  margin-left: 10px;
  color: #9ee59e;
}
'''

js_code = r'''
let DATA = null;
let CURRENT_SCAN = null;
let CURRENT_TARGET = null;
let drawing = false;
let startPoint = null;
let currentBox = null;

function byId(id) { return document.getElementById(id); }
function clean(v) { return v === null || v === undefined ? "" : String(v); }

async function loadData() {
  const resp = await fetch("/api/data");
  DATA = await resp.json();
  renderList();

  if (!CURRENT_TARGET && DATA.scanframes.length && DATA.scanframes[0].targets.length) {
    selectTarget(DATA.scanframes[0].scan_frame_id, DATA.scanframes[0].targets[0].canonical_gt_object_id);
  }
}

function renderList() {
  const list = byId("list");
  list.innerHTML = "";

  let targetCount = 0;
  for (const sf of DATA.scanframes) targetCount += sf.targets.length;

  byId("summary").textContent = `scanframes: ${DATA.scanframes.length}\ntargets: ${targetCount}`;

  for (const sf of DATA.scanframes) {
    for (const t of sf.targets) {
      const div = document.createElement("div");
      div.className = "item";
      if (CURRENT_TARGET && CURRENT_TARGET.canonical_gt_object_id === t.canonical_gt_object_id) div.classList.add("active");

      div.innerHTML = `
        <b>${t.canonical_gt_object_id}</b><br>
        <span class="small">${sf.video_id}</span><br>
        <span class="small">${t.behaviour_code} / ${t.manual_bbox_status} / ${t.manual_reviewer_note}</span>
      `;

      div.onclick = () => selectTarget(sf.scan_frame_id, t.canonical_gt_object_id);
      list.appendChild(div);
    }
  }
}

function selectTarget(scanId, objectId) {
  CURRENT_SCAN = DATA.scanframes.find(s => s.scan_frame_id === scanId);
  CURRENT_TARGET = CURRENT_SCAN.targets.find(t => t.canonical_gt_object_id === objectId);

  byId("title").textContent = CURRENT_TARGET.canonical_gt_object_id;
  byId("subtitle").textContent = `${CURRENT_SCAN.video_id} / ${CURRENT_TARGET.canonical_colour_label_norm} / ${CURRENT_TARGET.behaviour_code}`;

  const video = byId("video");
  video.src = CURRENT_SCAN.clip_path ? `/video?path=${encodeURIComponent(CURRENT_SCAN.clip_path)}` : "";
  video.load();

  byId("objTitle").textContent = `${CURRENT_TARGET.canonical_colour_label_norm} / ${CURRENT_TARGET.behaviour_code}`;
  byId("objInfo").innerHTML = `
    old candidate: ${CURRENT_TARGET.manual_assigned_candidate_box_id || "-"}<br>
    old bbox status: ${CURRENT_TARGET.manual_bbox_status}<br>
    gt status: ${CURRENT_TARGET.manual_gt_v2_status}<br>
    note: ${CURRENT_TARGET.manual_reviewer_note}
  `;

  setInputsFromTarget();
  renderList();
  setTimeout(drawOverlay, 250);
}

function setInputsFromTarget() {
  byId("x1").value = clean(CURRENT_TARGET.manual_bbox_x1);
  byId("y1").value = clean(CURRENT_TARGET.manual_bbox_y1);
  byId("x2").value = clean(CURRENT_TARGET.manual_bbox_x2);
  byId("y2").value = clean(CURRENT_TARGET.manual_bbox_y2);

  const vals = [Number(byId("x1").value), Number(byId("y1").value), Number(byId("x2").value), Number(byId("y2").value)];
  if (vals.every(v => Number.isFinite(v)) && vals[2] > vals[0] && vals[3] > vals[1]) {
    currentBox = vals;
  } else {
    currentBox = null;
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

  if (!video.videoWidth || !video.videoHeight) return;

  const sx = canvas.width / video.videoWidth;
  const sy = canvas.height / video.videoHeight;

  if (currentBox) {
    const [x1, y1, x2, y2] = currentBox;
    ctx.strokeStyle = "yellow";
    ctx.lineWidth = 3;
    ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
    ctx.fillStyle = "yellow";
    ctx.font = "14px Arial";
    ctx.fillText(`${CURRENT_TARGET.canonical_colour_label_norm} / ${CURRENT_TARGET.behaviour_code}`, x1 * sx + 4, y1 * sy + 16);
  }
}

function canvasToVideoCoords(evt) {
  const video = byId("video");
  const canvas = byId("overlay");
  const rect = canvas.getBoundingClientRect();

  const cx = evt.clientX - rect.left;
  const cy = evt.clientY - rect.top;

  const vx = cx * video.videoWidth / canvas.width;
  const vy = cy * video.videoHeight / canvas.height;

  return [vx, vy];
}

byId("overlay").addEventListener("mousedown", evt => {
  if (!byId("video").videoWidth) return;
  drawing = true;
  startPoint = canvasToVideoCoords(evt);
});

byId("overlay").addEventListener("mousemove", evt => {
  if (!drawing || !startPoint) return;
  const p = canvasToVideoCoords(evt);
  const x1 = Math.min(startPoint[0], p[0]);
  const y1 = Math.min(startPoint[1], p[1]);
  const x2 = Math.max(startPoint[0], p[0]);
  const y2 = Math.max(startPoint[1], p[1]);
  currentBox = [x1, y1, x2, y2];
  updateInputs();
  drawOverlay();
});

byId("overlay").addEventListener("mouseup", evt => {
  drawing = false;
  startPoint = null;
});

function updateInputs() {
  if (!currentBox) return;
  byId("x1").value = currentBox[0].toFixed(3);
  byId("y1").value = currentBox[1].toFixed(3);
  byId("x2").value = currentBox[2].toFixed(3);
  byId("y2").value = currentBox[3].toFixed(3);
}

["x1", "y1", "x2", "y2"].forEach(id => {
  byId(id).oninput = () => {
    const vals = [Number(byId("x1").value), Number(byId("y1").value), Number(byId("x2").value), Number(byId("y2").value)];
    if (vals.every(v => Number.isFinite(v)) && vals[2] > vals[0] && vals[3] > vals[1]) {
      currentBox = vals;
      drawOverlay();
    }
  };
});

byId("saveBtn").onclick = async () => {
  const payload = {
    canonical_gt_object_id: CURRENT_TARGET.canonical_gt_object_id,
    manual_bbox_x1: byId("x1").value,
    manual_bbox_y1: byId("y1").value,
    manual_bbox_x2: byId("x2").value,
    manual_bbox_y2: byId("y2").value,
    manual_reviewer_note: byId("note").value || "BBox adjusted in v66b2"
  };

  const resp = await fetch("/api/save_bbox", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });

  const out = await resp.json();
  if (out.ok) {
    byId("saveMsg").textContent = "saved";
  } else {
    byId("saveMsg").textContent = out.error || "save failed";
  }
};

byId("refreshBtn").onclick = loadData;
byId("video").addEventListener("loadedmetadata", drawOverlay);
window.addEventListener("resize", drawOverlay);

loadData();
'''

SERVER.write_text(server_code)
INDEX.write_text(html_code)
APP.write_text(js_code)
STYLE.write_text(css_code)

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0

decision = pd.DataFrame([{
    "v66b2_decision": "adjust_only_bbox_editor_created" if hard_issue_count == 0 else "adjust_only_bbox_editor_created_with_issues",
    "adjust_target_count": int(len(targets)),
    "adjust_scanframe_count": int(targets["scan_frame_id"].nunique()) if len(targets) else 0,
    "target_csv": str(ADJUST_TARGETS),
    "visualizer_json": str(VIS_JSON),
    "server_path": str(SERVER),
    "hard_issue_count": hard_issue_count,
    "issue_count": int(len(issues_df)),
    "ready_for_adjust_only_bbox_review": bool(hard_issue_count == 0),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
safe_to_csv(decision, DECISION)

NOTE.write_text(
    "# Week 8 v66b2 Adjust-only BBox Editor\n\n"
    f"- v66b2 decision: {decision.iloc[0]['v66b2_decision']}\n"
    f"- Adjust target count: {len(targets)}\n"
    f"- Adjust scanframe count: {targets['scan_frame_id'].nunique() if len(targets) else 0}\n"
    f"- Server: {SERVER}\n"
    f"- Target CSV: {ADJUST_TARGETS}\n"
    f"- Ready for adjust-only bbox review: {bool(hard_issue_count == 0)}\n\n"
    "Only explicit adjust/redraw/draw/new-bb rows are included. No-colour-visible red-exclude rows are intentionally excluded.\n"
)

REPORT.write_text(
    "# Week 8 v66b2 Adjust-only BBox Editor Report\n\n"
    f"Decision: {decision.iloc[0]['v66b2_decision']}\n\n"
    f"Targets: `{ADJUST_TARGETS}`\n\n"
    f"Server: `{SERVER}`\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v66b2",
    "task_name": "Adjust-only bbox editor",
    "status": "PASS" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": str(ASSIGNMENTS),
    "output_summary": str(OUT),
    "hard_issues": hard_issue_count,
    "warnings": 0,
    "next_action": "Adjust the listed bbox targets, then rerun audit/export/propagation.",
}])

if PROGRESS_LOG.exists():
    old = pd.read_csv(PROGRESS_LOG)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, PROGRESS_LOG)

print("Saved:")
print(ADJUST_TARGETS)
print(VIS_JSON)
print(SERVER)
print(INDEX)
print(APP)
print(STYLE)
print(DECISION)
print(NOTE)

print()
print("=== v66b2 decision ===")
print(decision.to_string(index=False))

print()
print("=== adjust targets ===")
if len(targets):
    cols = [
        "canonical_gt_object_id",
        "scan_frame_id",
        "video_id",
        "canonical_colour_label_norm",
        "behaviour_code",
        "manual_bbox_status",
        "manual_gt_v2_status",
        "manual_classification_use",
        "manual_reviewer_note",
    ]
    print(targets[cols].to_string(index=False))
else:
    print("No adjust targets found.")

print()
print("=== issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
