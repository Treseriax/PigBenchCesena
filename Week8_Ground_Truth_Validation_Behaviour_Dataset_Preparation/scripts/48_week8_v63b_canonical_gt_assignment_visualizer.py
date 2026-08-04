from pathlib import Path
from datetime import datetime
import json
import csv
import shutil
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

UI_JSON = W8 / "outputs" / "v63a_canonical_gt_v2_schema" / "week8_v63a_visualizer_data.json"
MANUAL_TEMPLATE = W8 / "validation" / "week8_v63a_manual_gt_v2_assignment_template.csv"

INTERFACE = W8 / "interface"
STATIC = INTERFACE / "static_v63b"
VALIDATION = W8 / "validation"
OUT = W8 / "outputs" / "v63b_canonical_gt_assignment_visualizer"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [INTERFACE, STATIC, VALIDATION, OUT, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

SERVER = INTERFACE / "week8_visualizer_server_v63b.py"
INDEX = STATIC / "index.html"
APP = STATIC / "app.js"
STYLE = STATIC / "style.css"

ASSIGNMENTS = VALIDATION / "week8_v63b_manual_gt_v2_assignments.csv"
OUT_DECISION = OUT / "week8_v63b_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v63b_issues.csv"
OUT_NOTE = NOTES / "week8_v63b_canonical_gt_assignment_visualizer_notes.md"
OUT_REPORT = REPORTS / "week8_v63b_canonical_gt_assignment_visualizer_report.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


issues = []

if not UI_JSON.exists():
    issues.append({
        "item": str(UI_JSON),
        "issue_type": "hard_missing_v63a_visualizer_data",
        "issue_detail": "v63a visualizer data JSON is missing.",
        "severity": "hard",
    })

if not MANUAL_TEMPLATE.exists():
    issues.append({
        "item": str(MANUAL_TEMPLATE),
        "issue_type": "hard_missing_manual_template",
        "issue_detail": "v63a manual assignment template is missing.",
        "severity": "hard",
    })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v63b_decision": "canonical_gt_assignment_visualizer_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_manual_gt_assignment": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


# Initialize assignment CSV once.
if not ASSIGNMENTS.exists():
    template = pd.read_csv(MANUAL_TEMPLATE)
    template["assignment_created_at"] = ""
    template["assignment_updated_at"] = ""
    safe_to_csv(template, ASSIGNMENTS)


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
'''

html_code = r'''
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Week8 v63b Canonical GT Assignment</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <div id="app">
    <aside id="sidebar">
      <h2>v63b GT Assignment</h2>
      <div class="small">Canonical colours come from original Excel v62b2.</div>

      <div class="controls">
        <label>Filter</label>
        <select id="filterStatus">
          <option value="">All</option>
          <option value="P0_gt_anchor_problem">P0 GT anchor</option>
          <option value="P1_manual_major_or_critical_issue">P1 major/critical</option>
          <option value="P1_candidate_box_count_not_six">P1 box count</option>
          <option value="P2_review_required">P2 review</option>
          <option value="P3_standard_review">P3 standard</option>
        </select>

        <label>Search scanframe</label>
        <input id="searchBox" placeholder="scanframe_0039">

        <button id="refreshBtn">Refresh</button>
      </div>

      <div id="statusSummary" class="summary"></div>
      <div id="scanList"></div>
    </aside>

    <main id="main">
      <section id="videoPanel">
        <div id="topBar">
          <div>
            <h1 id="title">Select a scanframe</h1>
            <div id="subtitle"></div>
          </div>
          <div class="toggles">
            <label><input type="checkbox" id="showBoxes" checked> candidate boxes</label>
            <label><input type="checkbox" id="showLabels" checked> box labels</label>
          </div>
        </div>

        <div id="videoWrap">
          <video id="video" controls muted></video>
          <canvas id="overlay"></canvas>
        </div>

        <div id="playButtons">
          <button data-seek="0">0s</button>
          <button data-seek="2.5">2.5s</button>
          <button data-seek="5">5s</button>
          <button data-seek="7.5">7.5s</button>
          <button data-seek="9.9">10s</button>
        </div>
      </section>

      <section id="assignmentPanel">
        <h2>Canonical Excel targets</h2>
        <div class="small">
          Assign each canonical colour to the correct candidate box, or mark it as missing / wrong / needs redraw.
        </div>
        <div id="targetRows"></div>
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
  background: #111;
  color: #eee;
}

#app {
  display: flex;
  height: 100vh;
  width: 100vw;
}

#sidebar {
  width: 360px;
  overflow-y: auto;
  border-right: 1px solid #333;
  padding: 14px;
  background: #181818;
}

#main {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

#videoPanel {
  padding: 14px;
  border-bottom: 1px solid #333;
}

#assignmentPanel {
  padding: 14px;
  overflow-y: auto;
}

h1, h2, h3 {
  margin: 0 0 8px 0;
}

.small {
  color: #aaa;
  font-size: 12px;
  line-height: 1.35;
}

.controls {
  display: grid;
  gap: 6px;
  margin: 12px 0;
}

input, select, textarea, button {
  background: #222;
  color: #eee;
  border: 1px solid #555;
  border-radius: 4px;
  padding: 6px;
}

button {
  cursor: pointer;
}

button:hover {
  background: #333;
}

.summary {
  font-size: 12px;
  margin: 10px 0;
  color: #ccc;
  white-space: pre-wrap;
}

.scanItem {
  padding: 8px;
  border: 1px solid #333;
  border-radius: 6px;
  margin-bottom: 6px;
  cursor: pointer;
  background: #202020;
}

.scanItem:hover {
  background: #2b2b2b;
}

.scanItem.active {
  outline: 2px solid #aaa;
}

.badge {
  display: inline-block;
  font-size: 11px;
  padding: 2px 5px;
  border-radius: 4px;
  background: #333;
  margin-top: 4px;
}

#topBar {
  display: flex;
  justify-content: space-between;
  gap: 10px;
}

.toggles {
  display: flex;
  gap: 14px;
  font-size: 13px;
}

#videoWrap {
  position: relative;
  margin-top: 10px;
  width: fit-content;
  max-width: 100%;
}

#video {
  max-height: 52vh;
  max-width: calc(100vw - 410px);
  background: black;
}

#overlay {
  position: absolute;
  left: 0;
  top: 0;
  pointer-events: auto;
}

#playButtons {
  margin-top: 8px;
  display: flex;
  gap: 8px;
}

.targetCard {
  border: 1px solid #333;
  border-radius: 8px;
  padding: 10px;
  margin-bottom: 10px;
  background: #181818;
}

.targetHeader {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 8px;
}

.targetGrid {
  display: grid;
  grid-template-columns: 160px 1fr 160px 1fr;
  gap: 8px;
  align-items: center;
}

.targetGrid textarea {
  grid-column: span 3;
  min-height: 46px;
}

.colourName {
  font-weight: bold;
  font-size: 16px;
}

.behaviour {
  color: #ddd;
}

.saveMsg {
  font-size: 12px;
  color: #aaa;
  margin-left: 8px;
}

.warningText {
  color: #ffcf75;
}

.problemText {
  color: #ff8888;
}

.okText {
  color: #9ee59e;
}
'''

js_code = r'''
let DATA = null;
let ASSIGNMENTS = [];
let CURRENT = null;
let selectedBoxId = null;

const colourOrder = ["green", "blue", "purple", "red_neck", "red_tail", "no_colour"];

function byId(id) { return document.getElementById(id); }

function clean(v) {
  if (v === null || v === undefined) return "";
  return String(v);
}

function assignmentKey(scan, colour) {
  return `${scan}||${colour}`;
}

function buildAssignmentMap() {
  const m = {};
  for (const r of ASSIGNMENTS) {
    m[assignmentKey(r.scan_frame_id, r.canonical_colour_label_norm)] = r;
  }
  return m;
}

async function loadAll() {
  const dataResp = await fetch("/api/data");
  DATA = await dataResp.json();

  const assResp = await fetch("/api/assignments");
  const assData = await assResp.json();
  ASSIGNMENTS = assData.assignments || [];

  renderSidebar();
  if (!CURRENT && DATA.scanframes.length) {
    selectScanframe(DATA.scanframes[0].scan_frame_id);
  } else if (CURRENT) {
    selectScanframe(CURRENT.scan_frame_id);
  }
}

function renderSidebar() {
  const filter = byId("filterStatus").value;
  const query = byId("searchBox").value.trim().toLowerCase();

  const list = byId("scanList");
  list.innerHTML = "";

  const counts = {};
  for (const sf of DATA.scanframes) {
    counts[sf.status] = (counts[sf.status] || 0) + 1;
  }

  byId("statusSummary").textContent =
    Object.entries(counts).map(([k, v]) => `${k}: ${v}`).join("\n");

  for (const sf of DATA.scanframes) {
    if (filter && sf.status !== filter) continue;
    if (query && !sf.scan_frame_id.toLowerCase().includes(query)) continue;

    const div = document.createElement("div");
    div.className = "scanItem";
    if (CURRENT && CURRENT.scan_frame_id === sf.scan_frame_id) div.classList.add("active");

    const boxCount = (sf.candidate_boxes || []).length;
    const targetCount = (sf.canonical_targets || []).length;

    div.innerHTML = `
      <div><b>${sf.scan_frame_id}</b></div>
      <div class="small">${sf.video_id}</div>
      <div class="badge">${sf.status}</div>
      <div class="small">targets=${targetCount}, boxes=${boxCount}</div>
    `;

    div.onclick = () => selectScanframe(sf.scan_frame_id);
    list.appendChild(div);
  }
}

function selectScanframe(scanId) {
  CURRENT = DATA.scanframes.find(x => x.scan_frame_id === scanId);
  selectedBoxId = null;

  byId("title").textContent = CURRENT.scan_frame_id;
  byId("subtitle").innerHTML = `
    ${CURRENT.video_id} — <span class="${CURRENT.status.startsWith("P3") ? "okText" : "warningText"}">${CURRENT.status}</span>
    — gate: ${CURRENT.classification_gate}
  `;

  const clipPath = CURRENT.clip_meta && CURRENT.clip_meta.clip_path ? CURRENT.clip_meta.clip_path : "";
  const video = byId("video");
  video.src = clipPath ? `/video?path=${encodeURIComponent(clipPath)}` : "";
  video.load();

  renderTargets();
  resizeCanvas();
  setTimeout(drawOverlay, 200);
  renderSidebar();
}

function renderTargets() {
  const container = byId("targetRows");
  container.innerHTML = "";

  const amap = buildAssignmentMap();

  const targets = [...(CURRENT.canonical_targets || [])].sort((a, b) => {
    return colourOrder.indexOf(a.canonical_colour_label_norm) - colourOrder.indexOf(b.canonical_colour_label_norm);
  });

  for (const t of targets) {
    const key = assignmentKey(t.scan_frame_id, t.canonical_colour_label_norm);
    const a = amap[key] || t;

    const card = document.createElement("div");
    card.className = "targetCard";

    const options = [`<option value="">-- no candidate selected --</option>`];
    for (const b of CURRENT.candidate_boxes || []) {
      const id = clean(b.candidate_box_id);
      const prev = clean(b.deprecated_previous_visual_marker_colour);
      options.push(`<option value="${id}" ${clean(a.manual_assigned_candidate_box_id) === id ? "selected" : ""}>${id} / old=${prev}</option>`);
    }

    card.innerHTML = `
      <div class="targetHeader">
        <div>
          <div class="colourName">${t.canonical_colour_label_norm}</div>
          <div class="small">raw: ${t.canonical_colour_label_raw}</div>
        </div>
        <div class="behaviour">behaviour: <b>${t.behaviour_code}</b></div>
      </div>

      <div class="targetGrid">
        <label>Candidate box</label>
        <select class="candidateBox">${options.join("")}</select>

        <label>BBox status</label>
        <select class="bboxStatus">
          ${optionList(["", "bbox_ok", "bbox_wrong", "bbox_missing", "bbox_extra_false_positive", "bbox_needs_manual_redraw", "bbox_uncertain"], a.manual_bbox_status)}
        </select>

        <label>Identity status</label>
        <select class="identityStatus">
          ${optionList(["", "identity_confirmed", "identity_uncertain", "identity_switch", "identity_not_visible", "identity_missing"], a.manual_identity_status)}
        </select>

        <label>GT v2 status</label>
        <select class="gtStatus">
          ${optionList(["", "gold_usable", "silver_usable_with_caution", "red_exclude", "fix_required", "unknown_pending_review"], a.manual_gt_v2_status)}
        </select>

        <label>Classification use</label>
        <select class="classificationUse">
          ${optionList(["", "use_for_classification", "use_for_classification_with_caution", "exclude_from_classification", "pending_review"], a.manual_classification_use)}
        </select>

        <label>Reviewer</label>
        <input class="reviewedBy" value="${escapeHtml(clean(a.manual_reviewed_by) || "oyavuz")}">

        <label>Note</label>
        <textarea class="reviewerNote">${escapeHtml(clean(a.manual_reviewer_note))}</textarea>
      </div>

      <div style="margin-top:8px;">
        <button class="saveBtn">Save assignment</button>
        <button class="focusBtn">Focus selected box</button>
        <span class="saveMsg"></span>
      </div>

      <div class="small" style="margin-top:6px;">
        Excel cell: row ${t.excel_behaviour_row_0based}, col ${t.excel_behaviour_col_0based}
      </div>
    `;

    const candidateSelect = card.querySelector(".candidateBox");
    candidateSelect.onchange = () => {
      selectedBoxId = candidateSelect.value;
      drawOverlay();
    };

    card.querySelector(".focusBtn").onclick = () => {
      selectedBoxId = candidateSelect.value;
      drawOverlay();
    };

    card.querySelector(".saveBtn").onclick = async () => {
      const payload = {
        scan_frame_id: t.scan_frame_id,
        canonical_colour_label_norm: t.canonical_colour_label_norm,
        manual_assigned_candidate_box_id: candidateSelect.value,
        manual_bbox_status: card.querySelector(".bboxStatus").value,
        manual_identity_status: card.querySelector(".identityStatus").value,
        manual_colour_status: "canonical_excel_colour_confirmed",
        manual_behaviour_status: "canonical_excel_behaviour_confirmed",
        manual_gt_v2_status: card.querySelector(".gtStatus").value,
        manual_classification_use: card.querySelector(".classificationUse").value,
        manual_review_priority: CURRENT.status,
        manual_reviewer_note: card.querySelector(".reviewerNote").value,
        manual_reviewed_by: card.querySelector(".reviewedBy").value
      };

      const resp = await fetch("/api/save_assignment", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload)
      });

      const msg = card.querySelector(".saveMsg");
      const out = await resp.json();

      if (out.ok) {
        msg.textContent = "saved";
        msg.className = "saveMsg okText";
        await refreshAssignmentsOnly();
      } else {
        msg.textContent = out.error || "save failed";
        msg.className = "saveMsg problemText";
      }
    };

    container.appendChild(card);
  }
}

function optionList(values, selected) {
  return values.map(v => `<option value="${v}" ${clean(selected) === v ? "selected" : ""}>${v || "--"}</option>`).join("");
}

function escapeHtml(s) {
  return clean(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function refreshAssignmentsOnly() {
  const assResp = await fetch("/api/assignments");
  const assData = await assResp.json();
  ASSIGNMENTS = assData.assignments || [];
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

  if (!CURRENT || !byId("showBoxes").checked) return;
  if (!video.videoWidth || !video.videoHeight) return;

  const sx = canvas.width / video.videoWidth;
  const sy = canvas.height / video.videoHeight;

  for (const b of CURRENT.candidate_boxes || []) {
    const x1 = Number(b.bbox_x1) * sx;
    const y1 = Number(b.bbox_y1) * sy;
    const x2 = Number(b.bbox_x2) * sx;
    const y2 = Number(b.bbox_y2) * sy;

    const id = clean(b.candidate_box_id);
    const selected = selectedBoxId && id === selectedBoxId;

    ctx.lineWidth = selected ? 4 : 2;
    ctx.strokeStyle = selected ? "yellow" : "lime";
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    if (byId("showLabels").checked) {
      ctx.fillStyle = selected ? "yellow" : "lime";
      ctx.font = "14px Arial";
      ctx.fillText(id, x1 + 4, y1 + 16);
    }
  }
}

byId("filterStatus").onchange = renderSidebar;
byId("searchBox").oninput = renderSidebar;
byId("refreshBtn").onclick = loadAll;
byId("showBoxes").onchange = drawOverlay;
byId("showLabels").onchange = drawOverlay;

byId("video").addEventListener("loadedmetadata", drawOverlay);
byId("video").addEventListener("timeupdate", drawOverlay);
window.addEventListener("resize", drawOverlay);

document.querySelectorAll("#playButtons button").forEach(btn => {
  btn.onclick = () => {
    const t = Number(btn.dataset.seek);
    byId("video").currentTime = t;
    drawOverlay();
  };
});

loadAll();
'''

SERVER.write_text(server_code)
INDEX.write_text(html_code)
APP.write_text(js_code)
STYLE.write_text(css_code)

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

decision = pd.DataFrame([{
    "v63b_decision": "canonical_gt_assignment_visualizer_created",
    "server_path": str(SERVER),
    "static_dir": str(STATIC),
    "assignments_csv": str(ASSIGNMENTS),
    "source_ui_json": str(UI_JSON),
    "manual_template": str(MANUAL_TEMPLATE),
    "hard_issue_count": 0,
    "warning_count": 0,
    "issue_count": 0,
    "ready_for_manual_gt_assignment": True,
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v63b Canonical GT Assignment Visualizer\n\n"
    f"- v63b decision: {decision.iloc[0]['v63b_decision']}\n"
    f"- Server: {SERVER}\n"
    f"- Static dir: {STATIC}\n"
    f"- Assignments CSV: {ASSIGNMENTS}\n"
    f"- Ready for manual GT assignment: True\n\n"
    "Use this interface to assign each canonical Excel colour identity to the correct candidate box, or mark the target as missing/wrong/fix_required.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v63b Canonical GT Assignment Visualizer Report\n\n"
    f"Decision: {decision.iloc[0]['v63b_decision']}\n\n"
    f"Server: `{SERVER}`\n\n"
    f"Assignments CSV: `{ASSIGNMENTS}`\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v63b",
    "task_name": "Canonical GT assignment visualizer",
    "status": "PASS",
    "input_summary": str(UI_JSON),
    "output_summary": str(OUT),
    "hard_issues": 0,
    "warnings": 0,
    "next_action": "Manually assign canonical colour identities to candidate boxes.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(SERVER)
print(INDEX)
print(APP)
print(STYLE)
print(ASSIGNMENTS)
print(OUT_DECISION)
print(OUT_NOTE)

print()
print("=== v63b decision ===")
print(decision.to_string(index=False))
