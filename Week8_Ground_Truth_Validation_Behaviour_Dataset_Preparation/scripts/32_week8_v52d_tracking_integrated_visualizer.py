from pathlib import Path
from datetime import datetime
import csv
import json
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V45_CLIP_JSON = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_level_ground_truth.json"
V45_ANCHORS = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_object_propagated_annotations.csv"
V52B3_TRACKS = W8 / "outputs" / "v52b3_hybrid_tracking_full" / "week8_v52b3_hybrid_tracking_rows.csv"
V52C_CLIP_QA = W8 / "outputs" / "v52c_full_hybrid_tracking_qa" / "week8_v52c_clip_quality_assessment.csv"
V52C_REVIEW_QUEUE = W8 / "outputs" / "v52c_full_hybrid_tracking_qa" / "week8_v52c_manual_review_queue.csv"

INTERFACE = W8 / "interface"
STATIC = INTERFACE / "static_v52d"
VALIDATION = W8 / "validation"
OUT = W8 / "outputs" / "v52d_tracking_integrated_visualizer"
REPORTS = W8 / "reports"
NOTES = W8 / "notes"
PROGRESS = W8 / "progress"

for p in [INTERFACE, STATIC, VALIDATION, OUT, REPORTS, NOTES, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

SERVER_PATH = INTERFACE / "week8_visualizer_server_v52d.py"
INDEX_PATH = STATIC / "index.html"
APP_PATH = STATIC / "app.js"
STYLE_PATH = STATIC / "style.css"
NOTES_CSV = VALIDATION / "week8_v52d_visual_validation_notes.csv"

OUT_DECISION = OUT / "week8_v52d_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v52d_issues.csv"
OUT_MANIFEST = OUT / "week8_v52d_interface_manifest.csv"
OUT_REPORT = REPORTS / "week8_v52d_tracking_integrated_visualizer_report.md"
OUT_NOTE = NOTES / "week8_v52d_tracking_integrated_visualizer_notes.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


issues = []

for p in [V45_CLIP_JSON, V45_ANCHORS, V52B3_TRACKS, V52C_CLIP_QA]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required input missing.",
            "severity": "hard",
        })

if not NOTES_CSV.exists():
    safe_to_csv(pd.DataFrame(columns=[
        "created_at",
        "scan_frame_id",
        "video_id",
        "time_sec",
        "frame_index_in_clip",
        "issue_type",
        "severity",
        "note",
    ]), NOTES_CSV)


server_code = f'''from pathlib import Path
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

    qa_map = {{r["scan_frame_id"]: r.to_dict() for _, r in clip_qa.iterrows()}}

    clip_list = []
    for c in clips:
        scan = c["scan_frame_id"]
        q = qa_map.get(scan, {{}})
        clip_list.append({{
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
        }})

    return clips, clip_list, anchors, tracks, clip_qa


CLIPS_RAW, CLIPS, ANCHORS, TRACKS, CLIP_QA = load_data()
CLIP_MAP = {{c["scan_frame_id"]: c for c in CLIPS_RAW}}


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
            handler.send_header("Content-Range", f"bytes {{start}}-{{end}}/{{file_size}}")
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
            return json_response(self, {{"clips": CLIPS}})

        if path == "/api/clip":
            scan = qs.get("scan", [""])[0]
            clip = CLIP_MAP.get(scan)
            if clip is None:
                return json_response(self, {{"error": "scan not found"}}, status=404)

            qdf = CLIP_QA[CLIP_QA["scan_frame_id"] == scan].copy()
            qa = qdf.iloc[0].to_dict() if len(qdf) else {{}}

            return json_response(self, {{"clip": clip, "qa": qa}})

        if path == "/api/annotations":
            scan = qs.get("scan", [""])[0]
            if scan == "":
                return json_response(self, {{"error": "scan missing"}}, status=400)

            a = ANCHORS[ANCHORS["scan_frame_id"] == scan].copy()
            t = TRACKS[TRACKS["scan_frame_id"] == scan].copy()

            anchors_out = []
            for _, r in a.iterrows():
                anchors_out.append({{
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
                }})

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
                d = {{}}
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

            return json_response(self, {{"scan_frame_id": scan, "anchors": anchors_out, "tracks": tracks_out}})

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
            return json_response(self, {{"ok": False, "error": "invalid json"}}, status=400)

        row = {{
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "scan_frame_id": clean_str(payload.get("scan_frame_id")),
            "video_id": clean_str(payload.get("video_id")),
            "time_sec": clean_str(payload.get("time_sec")),
            "frame_index_in_clip": clean_str(payload.get("frame_index_in_clip")),
            "issue_type": clean_str(payload.get("issue_type")),
            "severity": clean_str(payload.get("severity")),
            "note": clean_str(payload.get("note")),
        }}

        exists = NOTES_CSV.exists()
        with open(NOTES_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()), quoting=csv.QUOTE_ALL, escapechar="\\")
            if not exists:
                writer.writeheader()
            writer.writerow(row)

        return json_response(self, {{"ok": True, "saved_to": str(NOTES_CSV)}})


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8526)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Week 8 v52d visualizer running at http://{{args.host}}:{{args.port}}")
    print("Open through SSH tunnel/local browser, for example: http://127.0.0.1:8526/")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
'''

index_html = '''<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Week 8 v52d Tracking Integrated Visualizer</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <header>
    <h1>Week 8 v52d Tracking Integrated Visualizer</h1>
    <p>Anchor GT + stable tracklet + recall fallback + behaviour/colour labels</p>
  </header>

  <main>
    <aside>
      <div class="panel">
        <h2>Clips</h2>
        <input id="clipSearch" placeholder="Search scan/video">
        <select id="qualityFilter">
          <option value="all">All clips</option>
          <option value="strong_visual_tracking_support">Strong</option>
          <option value="usable_with_review">Usable</option>
          <option value="limited_review_required">Limited</option>
          <option value="challenging_low_tracking_support">Challenging</option>
        </select>
        <div id="clipList"></div>
      </div>
    </aside>

    <section class="viewer">
      <div class="panel">
        <h2 id="clipTitle">Select a clip</h2>
        <div id="clipStats"></div>

        <div class="videoWrap">
          <video id="video" controls muted playsinline></video>
          <canvas id="overlay"></canvas>
        </div>

        <div class="controls">
          <label><input type="checkbox" id="showAnchor" checked> Anchor GT</label>
          <label><input type="checkbox" id="showStable" checked> Stable tracklet</label>
          <label><input type="checkbox" id="showFallback" checked> Recall fallback</label>
          <label><input type="checkbox" id="showReview" checked> Review/missing indicators</label>
          <label><input type="checkbox" id="showLabels" checked> Labels</label>
          <label><input type="checkbox" id="showBehaviour" checked> Behaviour</label>
          <label><input type="checkbox" id="showColour" checked> Colour</label>
        </div>

        <div id="frameInfo"></div>
      </div>

      <div class="panel">
        <h2>Manual validation note</h2>
        <div class="noteRow">
          <select id="issueType">
            <option value="bbox_ok">bbox_ok</option>
            <option value="bbox_wrong">bbox_wrong</option>
            <option value="identity_uncertain">identity_uncertain</option>
            <option value="identity_switch">identity_switch</option>
            <option value="behaviour_uncertain">behaviour_uncertain</option>
            <option value="missing_pig">missing_pig</option>
            <option value="false_positive">false_positive</option>
            <option value="other">other</option>
          </select>
          <select id="severity">
            <option value="info">info</option>
            <option value="minor">minor</option>
            <option value="major">major</option>
            <option value="critical">critical</option>
          </select>
        </div>
        <textarea id="noteText" placeholder="Write visual validation note here..."></textarea>
        <button id="saveNote">Save note</button>
        <div id="saveStatus"></div>
      </div>
    </section>
  </main>

  <script src="/static/app.js"></script>
</body>
</html>
'''

app_js = '''let clips = [];
let selectedClip = null;
let anchors = [];
let tracks = [];
let tracksByFrame = new Map();

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const colourMap = {
  blue: "#1e5bff",
  green: "#32e64b",
  cyan: "#00e5ff",
  red: "#ff2525",
  pink: "#ff3dd8",
  purple: "#a855ff",
  unknown: "#c8c8c8",
  not_visible: "#999999",
  uncertain: "#ffcc00",
  unassigned: "#aaaaaa"
};

function getColour(c) {
  c = (c || "unknown").toLowerCase();
  return colourMap[c] || "#ffffff";
}

function htmlEscape(s) {
  return String(s ?? "").replace(/[&<>"']/g, m => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  }[m]));
}

async function loadClips() {
  const res = await fetch("/api/clips");
  const data = await res.json();
  clips = data.clips || [];
  renderClipList();
}

function renderClipList() {
  const list = document.getElementById("clipList");
  const q = document.getElementById("clipSearch").value.toLowerCase();
  const filter = document.getElementById("qualityFilter").value;

  list.innerHTML = "";

  clips
    .filter(c => filter === "all" || c.quality_tier === filter)
    .filter(c => {
      const s = `${c.scan_frame_id} ${c.video_id}`.toLowerCase();
      return s.includes(q);
    })
    .forEach(c => {
      const div = document.createElement("div");
      div.className = "clipItem " + (c.quality_tier || "");
      div.innerHTML = `
        <b>${htmlEscape(c.scan_frame_id)}</b>
        <span>${htmlEscape(c.video_id)}</span>
        <small>${htmlEscape(c.quality_tier)} | draw ${Number(c.draw_ok_ratio).toFixed(2)} | review ${Number(c.review_needed_ratio).toFixed(2)}</small>
      `;
      div.onclick = () => selectClip(c.scan_frame_id);
      list.appendChild(div);
    });
}

async function selectClip(scan) {
  const clipRes = await fetch(`/api/clip?scan=${encodeURIComponent(scan)}`);
  const clipData = await clipRes.json();

  const annRes = await fetch(`/api/annotations?scan=${encodeURIComponent(scan)}`);
  const annData = await annRes.json();

  selectedClip = clipData.clip;
  selectedClip.qa = clipData.qa || {};
  anchors = annData.anchors || [];
  tracks = annData.tracks || [];

  tracksByFrame = new Map();
  for (const t of tracks) {
    const f = Number(t.frame_index_in_clip || 0);
    if (!tracksByFrame.has(f)) tracksByFrame.set(f, []);
    tracksByFrame.get(f).push(t);
  }

  document.getElementById("clipTitle").textContent = `${scan} — ${selectedClip.video_id || ""}`;
  document.getElementById("clipStats").innerHTML = `
    <b>quality:</b> ${htmlEscape(selectedClip.qa.quality_tier || "")}
    | <b>draw:</b> ${Number(selectedClip.qa.draw_ok_ratio || 0).toFixed(3)}
    | <b>stable:</b> ${Number(selectedClip.qa.stable_tracklet_ratio || 0).toFixed(3)}
    | <b>fallback:</b> ${Number(selectedClip.qa.recall_fallback_ratio || 0).toFixed(3)}
    | <b>missing:</b> ${Number(selectedClip.qa.missing_ratio || 0).toFixed(3)}
    | <b>review:</b> ${Number(selectedClip.qa.review_needed_ratio || 0).toFixed(3)}
  `;

  video.src = `/video?scan=${encodeURIComponent(scan)}`;
  video.load();
}

function resizeCanvas() {
  const rect = video.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
}

function frameIndex() {
  const fps = Number(selectedClip?.fps_used || 25);
  return Math.round(video.currentTime * fps);
}

function scaleBox(box) {
  const vw = video.videoWidth || 1;
  const vh = video.videoHeight || 1;
  const sx = canvas.width / vw;
  const sy = canvas.height / vh;
  return {
    x1: box.x1 * sx,
    y1: box.y1 * sy,
    x2: box.x2 * sx,
    y2: box.y2 * sy
  };
}

function drawBox(box, colour, lineWidth, dashed, label) {
  const b = scaleBox(box);
  ctx.save();
  ctx.strokeStyle = colour;
  ctx.lineWidth = lineWidth;
  if (dashed) ctx.setLineDash([7, 5]);
  ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);

  if (document.getElementById("showLabels").checked && label) {
    ctx.font = "13px Arial";
    ctx.fillStyle = colour;
    ctx.fillText(label, b.x1 + 3, Math.max(14, b.y1 - 4));
  }
  ctx.restore();
}

function makeLabel(r, source) {
  const parts = [];
  if (document.getElementById("showColour").checked) parts.push(r.visual_marker_colour || "unknown");
  if (document.getElementById("showBehaviour").checked) parts.push(r.behaviour_code || "");
  parts.push(source);
  return `${r.behaviour_pig_id || r.final_box_id || ""}/${parts.filter(Boolean).join("/")}`;
}

function drawOverlay() {
  resizeCanvas();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!selectedClip || !video.videoWidth) {
    requestAnimationFrame(drawOverlay);
    return;
  }

  const f = frameIndex();
  const current = tracksByFrame.get(f) || [];

  let stableCount = 0;
  let fallbackCount = 0;
  let missingCount = 0;
  let reviewCount = 0;

  if (document.getElementById("showAnchor").checked) {
    for (const a of anchors) {
      drawBox(a, "#ffffff", 1.5, true, makeLabel(a, "anchor"));
    }
  }

  for (const r of current) {
    if (r.hybrid_source === "stable_tracklet") stableCount++;
    if (r.hybrid_source === "recall_fallback") fallbackCount++;
    if (r.hybrid_source === "missing") missingCount++;
    if (r.review_needed) reviewCount++;

    if (!r.draw_ok) continue;

    const colour = getColour(r.visual_marker_colour);
    if (r.hybrid_source === "stable_tracklet" && document.getElementById("showStable").checked) {
      drawBox(r, colour, 3, false, makeLabel(r, "stable"));
    }
    if (r.hybrid_source === "recall_fallback" && document.getElementById("showFallback").checked) {
      drawBox(r, colour, 1.5, true, makeLabel(r, "fallback"));
    }
  }

  if (document.getElementById("showReview").checked) {
    document.getElementById("frameInfo").innerHTML = `
      <b>time:</b> ${video.currentTime.toFixed(2)}s
      | <b>frame:</b> ${f}
      | <b>stable:</b> ${stableCount}
      | <b>fallback:</b> ${fallbackCount}
      | <b>missing:</b> ${missingCount}
      | <b>review:</b> ${reviewCount}
    `;
  } else {
    document.getElementById("frameInfo").innerHTML = `<b>time:</b> ${video.currentTime.toFixed(2)}s | <b>frame:</b> ${f}`;
  }

  requestAnimationFrame(drawOverlay);
}

async function saveNote() {
  if (!selectedClip) return;

  const payload = {
    scan_frame_id: selectedClip.scan_frame_id,
    video_id: selectedClip.video_id || "",
    time_sec: video.currentTime.toFixed(3),
    frame_index_in_clip: frameIndex(),
    issue_type: document.getElementById("issueType").value,
    severity: document.getElementById("severity").value,
    note: document.getElementById("noteText").value
  };

  const res = await fetch("/api/save_note", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });

  const data = await res.json();
  document.getElementById("saveStatus").textContent = data.ok ? `Saved: ${data.saved_to}` : `Error: ${data.error}`;
  if (data.ok) document.getElementById("noteText").value = "";
}

document.getElementById("clipSearch").addEventListener("input", renderClipList);
document.getElementById("qualityFilter").addEventListener("change", renderClipList);
document.getElementById("saveNote").addEventListener("click", saveNote);

for (const id of ["showAnchor", "showStable", "showFallback", "showReview", "showLabels", "showBehaviour", "showColour"]) {
  document.getElementById(id).addEventListener("change", drawOverlay);
}

video.addEventListener("loadedmetadata", resizeCanvas);
window.addEventListener("resize", resizeCanvas);

loadClips();
drawOverlay();
'''

style_css = '''body {
  margin: 0;
  font-family: Arial, sans-serif;
  background: #111;
  color: #eee;
}

header {
  padding: 16px 22px;
  border-bottom: 1px solid #333;
  background: #181818;
}

header h1 {
  margin: 0 0 6px 0;
  font-size: 24px;
}

header p {
  margin: 0;
  color: #bbb;
}

main {
  display: grid;
  grid-template-columns: 360px 1fr;
  gap: 16px;
  padding: 16px;
}

.panel {
  background: #1d1d1d;
  border: 1px solid #333;
  border-radius: 10px;
  padding: 14px;
}

aside {
  max-height: calc(100vh - 110px);
  overflow: hidden;
}

#clipList {
  margin-top: 10px;
  max-height: calc(100vh - 250px);
  overflow-y: auto;
}

.clipItem {
  padding: 10px;
  border: 1px solid #333;
  border-radius: 8px;
  margin-bottom: 8px;
  cursor: pointer;
  background: #242424;
}

.clipItem:hover {
  background: #303030;
}

.clipItem span,
.clipItem small {
  display: block;
  color: #bbb;
  margin-top: 3px;
}

.clipItem.strong_visual_tracking_support {
  border-left: 5px solid #39d353;
}

.clipItem.usable_with_review {
  border-left: 5px solid #2f81f7;
}

.clipItem.limited_review_required {
  border-left: 5px solid #d29922;
}

.clipItem.challenging_low_tracking_support {
  border-left: 5px solid #f85149;
}

input, select, textarea, button {
  background: #111;
  color: #eee;
  border: 1px solid #444;
  border-radius: 6px;
  padding: 8px;
  margin-top: 8px;
}

input, select {
  width: 100%;
  box-sizing: border-box;
}

textarea {
  width: 100%;
  height: 90px;
  box-sizing: border-box;
}

button {
  cursor: pointer;
  background: #2f81f7;
  border: none;
}

.videoWrap {
  position: relative;
  width: 100%;
  max-width: 1280px;
  background: #000;
  margin-top: 12px;
}

video {
  width: 100%;
  display: block;
}

canvas {
  position: absolute;
  left: 0;
  top: 0;
  pointer-events: none;
}

.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  margin-top: 12px;
}

.controls label {
  background: #252525;
  padding: 6px 8px;
  border-radius: 6px;
}

.controls input {
  width: auto;
  margin: 0 5px 0 0;
}

#clipStats, #frameInfo, #saveStatus {
  color: #ccc;
  margin-top: 10px;
  line-height: 1.5;
}

.noteRow {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}

@media (max-width: 1100px) {
  main {
    grid-template-columns: 1fr;
  }
}
'''

SERVER_PATH.write_text(server_code)
INDEX_PATH.write_text(index_html)
APP_PATH.write_text(app_js)
STYLE_PATH.write_text(style_css)

manifest = pd.DataFrame([
    {"item": "server", "path": str(SERVER_PATH), "description": "Python HTTP server with video streaming, API endpoints and note saving."},
    {"item": "index", "path": str(INDEX_PATH), "description": "Main tracking-integrated visualizer UI."},
    {"item": "app_js", "path": str(APP_PATH), "description": "Browser logic for clip selection, overlay rendering and notes."},
    {"item": "style_css", "path": str(STYLE_PATH), "description": "Visualizer styling."},
    {"item": "notes_csv", "path": str(NOTES_CSV), "description": "Manual visual validation notes."},
])
safe_to_csv(manifest, OUT_MANIFEST)

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
ready = hard_issue_count == 0

decision = pd.DataFrame([{
    "v52d_decision": "tracking_integrated_visualizer_created" if ready else "tracking_integrated_visualizer_blocked",
    "server_path": str(SERVER_PATH),
    "static_dir": str(STATIC),
    "manual_notes_csv": str(NOTES_CSV),
    "hard_issue_count": hard_issue_count,
    "warning_count": int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0,
    "issue_count": int(len(issues_df)),
    "ready_for_manual_visual_validation": bool(ready),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
safe_to_csv(decision, OUT_DECISION)

OUT_REPORT.write_text(
    "# Week 8 v52d Tracking-integrated Visualizer Report\\n\\n"
    f"Decision: {decision.iloc[0]['v52d_decision']}\\n\\n"
    "The interface integrates anchor GT boxes, full hybrid tracking boxes, stable/fallback source labels, behaviour labels, colour labels, quality filters and manual note saving.\\n\\n"
    f"Server: {SERVER_PATH}\\n"
    f"Manual notes: {NOTES_CSV}\\n"
)

OUT_NOTE.write_text(
    "# Week 8 v52d Tracking-integrated Visualizer\\n\\n"
    "## Summary\\n\\n"
    f"- v52d decision: {decision.iloc[0]['v52d_decision']}\\n"
    f"- Server: {SERVER_PATH}\\n"
    f"- Static directory: {STATIC}\\n"
    f"- Manual notes CSV: {NOTES_CSV}\\n"
    f"- Hard issue count: {hard_issue_count}\\n"
    f"- Ready for manual visual validation: {ready}\\n\\n"
    "## Run\\n\\n"
    f"python {SERVER_PATH} --host 0.0.0.0 --port 8526\\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v52d",
    "task_name": "Tracking-integrated visualizer",
    "status": "PASS" if ready else "BLOCKED",
    "input_summary": str(V52B3_TRACKS),
    "output_summary": str(OUT),
    "hard_issues": hard_issue_count,
    "warnings": int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0,
    "next_action": "Manual visual validation and v52e validation summary" if ready else "Resolve v52d missing inputs.",
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
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_MANIFEST)
print(OUT_REPORT)
print(OUT_NOTE)

print()
print("=== v52d decision ===")
print(decision.to_string(index=False))

print()
print("=== v52d issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
