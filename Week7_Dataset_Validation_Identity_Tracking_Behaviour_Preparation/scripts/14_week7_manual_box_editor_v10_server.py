from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, unquote
from datetime import datetime
import argparse
import json
import csv
import mimetypes

import cv2
import pandas as pd


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

FRAME_INDEX_PATH = W6 / "outputs" / "unified_ground_truth" / "week6_scanpoint_frame_index.csv"
V7_POOL_PATH = W7 / "outputs" / "roi_and_crate_mapping" / "week7_gt_pen_candidate_pool_v7_corrected.csv"

IMG_DIR = W7 / "outputs" / "manual_annotation_v9" / "cvat_coco_import" / "images"

OUT_ROOT = W7 / "outputs" / "manual_annotation_v10_editor"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SAVED_JSON = OUT_ROOT / "week7_manual_box_editor_v10_saved_state.json"
FINAL_CSV = OUT_ROOT / "week7_manual_box_editor_v10_corrected_boxes.csv"
ALL_DECISIONS_CSV = OUT_ROOT / "week7_manual_box_editor_v10_all_box_decisions.csv"
FRAME_SUMMARY_CSV = OUT_ROOT / "week7_manual_box_editor_v10_frame_summary.csv"
NOTE_PATH = W7 / "notes" / "week7_manual_box_editor_v10_notes.md"
NOTE_PATH.parent.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def build_initial_state():
    frames = pd.read_csv(FRAME_INDEX_PATH)
    pool = pd.read_csv(V7_POOL_PATH)

    for c in ["x1", "y1", "x2", "y2", "candidate_score", "detector_score"]:
        if c in pool.columns:
            pool[c] = pd.to_numeric(pool[c], errors="coerce")

    pool = pool.dropna(subset=["x1", "y1", "x2", "y2"]).copy()

    if "v7_manual_review_required" in pool.columns:
        pool["v7_manual_review_required"] = to_bool_series(pool["v7_manual_review_required"])
    else:
        pool["v7_manual_review_required"] = True

    if "priority_new_recall_candidate_v7" in pool.columns:
        pool["priority_new_recall_candidate_v7"] = to_bool_series(pool["priority_new_recall_candidate_v7"])
    else:
        pool["priority_new_recall_candidate_v7"] = False

    state_frames = []

    for _, fr in frames.drop_duplicates("scan_frame_id").sort_values("scan_frame_id").iterrows():
        sid = str(fr["scan_frame_id"])
        img_path = IMG_DIR / f"{sid}.jpg"

        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        g = pool[pool["scan_frame_id"].astype(str) == sid].copy()

        # Editor'da çok kalabalık olmasın diye existing_ignored box'ları göstermiyoruz.
        g = g[g["v7_candidate_class"].astype(str) != "existing_ignored"].copy()

        boxes = []

        for _, r in g.iterrows():
            cls = str(r.get("v7_candidate_class", "unknown"))

            if cls == "strict_auto_safe":
                status = "keep"
            else:
                status = "pending"

            x1 = float(r["x1"])
            y1 = float(r["y1"])
            x2 = float(r["x2"])
            y2 = float(r["y2"])

            boxes.append({
                "id": str(r.get("v7_candidate_id", f"{sid}_candidate_{len(boxes)}")),
                "scan_frame_id": sid,
                "x": x1,
                "y": y1,
                "w": max(1.0, x2 - x1),
                "h": max(1.0, y2 - y1),
                "status": status,
                "label": "gt_pen_pig_candidate",
                "candidate_class": cls,
                "candidate_source": str(r.get("candidate_source", "")),
                "candidate_score": float(r.get("candidate_score", 0.0) or 0.0),
                "detector_score": float(r.get("detector_score", 0.0) or 0.0),
                "manual_note": "",
            })

        state_frames.append({
            "scan_frame_id": sid,
            "image_url": f"/images/{sid}.jpg",
            "width": int(w),
            "height": int(h),
            "expected_pigs": 6,
            "boxes": boxes,
        })

    state = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "last_saved_at": "",
        "version": "v10_manual_box_editor",
        "frames": state_frames,
    }

    return state


def load_state():
    if SAVED_JSON.exists():
        return json.loads(SAVED_JSON.read_text())
    return build_initial_state()


def generate_outputs(state):
    final_rows = []
    all_rows = []
    frame_rows = []

    for frame in state["frames"]:
        sid = frame["scan_frame_id"]
        keep_count = 0
        pending_count = 0
        rejected_count = 0
        deleted_count = 0

        for i, b in enumerate(frame.get("boxes", [])):
            status = str(b.get("status", "pending"))
            x = float(b.get("x", 0))
            y = float(b.get("y", 0))
            w = float(b.get("w", 0))
            h = float(b.get("h", 0))

            row = {
                "scan_frame_id": sid,
                "box_id": b.get("id", f"{sid}_box_{i}"),
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h,
                "width": w,
                "height": h,
                "status": status,
                "label": b.get("label", ""),
                "candidate_class": b.get("candidate_class", ""),
                "candidate_source": b.get("candidate_source", ""),
                "candidate_score": b.get("candidate_score", ""),
                "detector_score": b.get("detector_score", ""),
                "manual_note": b.get("manual_note", ""),
            }

            all_rows.append(row)

            if status == "keep":
                keep_count += 1
                final_rows.append({
                    "scan_frame_id": sid,
                    "final_box_id": f"{sid}_final_{keep_count:02d}",
                    "source_box_id": row["box_id"],
                    "source_type": row["candidate_class"],
                    "x1": x,
                    "y1": y,
                    "x2": x + w,
                    "y2": y + h,
                    "width": w,
                    "height": h,
                    "final_use_for_colour_matching": True,
                    "final_correction_status": "manual_corrected_keep",
                    "manual_note": row["manual_note"],
                })
            elif status == "pending":
                pending_count += 1
            elif status.startswith("reject"):
                rejected_count += 1
            elif status == "deleted":
                deleted_count += 1

        frame_rows.append({
            "scan_frame_id": sid,
            "expected_pigs": frame.get("expected_pigs", 6),
            "kept_boxes": keep_count,
            "pending_boxes": pending_count,
            "rejected_boxes": rejected_count,
            "deleted_boxes": deleted_count,
            "missing_vs_expected": max(0, int(frame.get("expected_pigs", 6)) - keep_count),
            "extra_vs_expected": max(0, keep_count - int(frame.get("expected_pigs", 6))),
            "frame_complete_candidate": keep_count == int(frame.get("expected_pigs", 6)) and pending_count == 0,
        })

    safe_to_csv(pd.DataFrame(final_rows), FINAL_CSV)
    safe_to_csv(pd.DataFrame(all_rows), ALL_DECISIONS_CSV)
    safe_to_csv(pd.DataFrame(frame_rows), FRAME_SUMMARY_CSV)

    NOTE_PATH.write_text(
        "# Week 7 Manual Box Editor v10\n\n"
        "## Purpose\n\n"
        "This custom browser editor replaces CVAT for the manual correction step. "
        "It is used to delete wrong-pen boxes, keep correct GT-pen boxes, adjust inaccurate boxes, and add missing boxes.\n\n"
        "## Main output\n\n"
        f"- Final corrected boxes CSV: `{FINAL_CSV}`\n"
        f"- All box decisions CSV: `{ALL_DECISIONS_CSV}`\n"
        f"- Frame summary CSV: `{FRAME_SUMMARY_CSV}`\n"
        f"- Saved editor state JSON: `{SAVED_JSON}`\n"
    )


HTML = r"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Week 7 Manual Box Editor v10</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 0;
      background: #f5f5f5;
    }
    #topbar {
      background: #222;
      color: white;
      padding: 10px 14px;
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    button {
      padding: 7px 10px;
      cursor: pointer;
      border: 1px solid #888;
      border-radius: 5px;
      background: white;
    }
    button.primary {
      background: #0b65c2;
      color: white;
      border-color: #0b65c2;
    }
    button.danger {
      background: #c62828;
      color: white;
      border-color: #c62828;
    }
    button.good {
      background: #2e7d32;
      color: white;
      border-color: #2e7d32;
    }
    button.warn {
      background: #ef6c00;
      color: white;
      border-color: #ef6c00;
    }
    #layout {
      display: grid;
      grid-template-columns: 260px 1fr 320px;
      height: calc(100vh - 54px);
    }
    #frameList {
      overflow: auto;
      background: white;
      border-right: 1px solid #ddd;
      padding: 10px;
    }
    .frameItem {
      padding: 8px;
      border-bottom: 1px solid #eee;
      cursor: pointer;
      font-size: 13px;
    }
    .frameItem.active {
      background: #e3f2fd;
      font-weight: bold;
    }
    .frameItem.problem {
      border-left: 5px solid #ef6c00;
    }
    .frameItem.ok {
      border-left: 5px solid #2e7d32;
    }
    #canvasPanel {
      overflow: auto;
      padding: 12px;
      text-align: center;
    }
    canvas {
      background: #ddd;
      border: 1px solid #444;
      cursor: crosshair;
      max-width: 100%;
    }
    #sidePanel {
      overflow: auto;
      background: white;
      border-left: 1px solid #ddd;
      padding: 12px;
    }
    .small {
      font-size: 12px;
      color: #555;
    }
    .legend span {
      display: inline-block;
      margin: 3px 0;
    }
    input, select, textarea {
      width: 100%;
      box-sizing: border-box;
      padding: 5px;
      margin: 3px 0 8px 0;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
    }
    .boxInfo {
      font-family: monospace;
      background: #eee;
      padding: 8px;
      border-radius: 6px;
      white-space: pre-wrap;
      font-size: 12px;
    }
    #statusMsg {
      color: #fff;
      font-weight: bold;
    }
  </style>
</head>
<body>
  <div id="topbar">
    <button onclick="prevFrame()">← Prev</button>
    <button onclick="nextFrame()">Next →</button>
    <button onclick="setMode('select')" id="selectBtn">Select/Move</button>
    <button onclick="setMode('draw')" id="drawBtn">Draw New Box</button>
    <button class="primary" onclick="saveState()">Save to server</button>
    <a href="/download_csv" target="_blank"><button>Download final CSV</button></a>
    <a href="/download_json" target="_blank"><button>Download JSON</button></a>
    <span id="statusMsg"></span>
  </div>

  <div id="layout">
    <div id="frameList"></div>

    <div id="canvasPanel">
      <h2 id="frameTitle"></h2>
      <canvas id="canvas"></canvas>
      <p class="small">
        Select/Move: kutuya tıkla ve sürükle. Draw New Box: eksik pig için sürükleyerek yeni kutu çiz.
      </p>
    </div>

    <div id="sidePanel">
      <h3>Selected Box</h3>
      <div id="selectedInfo" class="boxInfo">No selected box.</div>

      <h3>Actions</h3>
      <button class="good" onclick="setSelectedStatus('keep')">Keep / GT-pen pig</button>
      <button class="warn" onclick="setSelectedStatus('pending')">Pending</button>
      <button class="danger" onclick="setSelectedStatus('reject_wrong_pen')">Reject wrong pen</button>
      <button class="danger" onclick="setSelectedStatus('reject_duplicate')">Reject duplicate</button>
      <button class="danger" onclick="setSelectedStatus('reject_false_positive')">Reject false positive</button>
      <button onclick="setSelectedStatus('deleted')">Delete / hide from final</button>

      <h3>Edit coordinates</h3>
      <div class="row">
        <div><label>x</label><input id="xInput" type="number" step="1"></div>
        <div><label>y</label><input id="yInput" type="number" step="1"></div>
      </div>
      <div class="row">
        <div><label>w</label><input id="wInput" type="number" step="1"></div>
        <div><label>h</label><input id="hInput" type="number" step="1"></div>
      </div>
      <button onclick="applyCoords()">Apply coordinates</button>

      <h3>Manual note</h3>
      <textarea id="noteInput" rows="3" placeholder="optional note"></textarea>
      <button onclick="applyNote()">Save note to selected box</button>

      <h3>Frame summary</h3>
      <div id="frameSummary" class="boxInfo"></div>

      <h3>Legend</h3>
      <div class="legend small">
        <span style="color:green;font-weight:bold">Green</span>: keep / final GT-pen pig<br>
        <span style="color:darkorange;font-weight:bold">Orange</span>: pending / review<br>
        <span style="color:red;font-weight:bold">Red</span>: rejected<br>
        <span style="color:royalblue;font-weight:bold">Blue</span>: new manual box<br>
        <span style="color:gray;font-weight:bold">Gray</span>: deleted/ignored<br>
      </div>

      <h3>Rule</h3>
      <p class="small">
        Final CSV sadece <b>Keep</b> statüsündeki box’ları kullanır.
        Yanlış pen box’ı kalmamalı. Eksik doğru pig varsa Draw New Box ile ekle.
        Hedef: frame başına mümkünse 6 doğru GT-pen pig.
      </p>
    </div>
  </div>

<script>
let state = null;
let frameIndex = 0;
let selectedBoxId = null;
let mode = "select";
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");
let img = new Image();
let scale = 1.0;

let drag = null;
let drawStart = null;
let previewBox = null;

async function init() {
  state = await fetch("/data").then(r => r.json());
  renderFrameList();
  loadFrame(0);
}

function currentFrame() {
  return state.frames[frameIndex];
}

function activeBoxes(frame) {
  return frame.boxes || [];
}

function countFrame(frame) {
  let keep = 0, pending = 0, rejected = 0, deleted = 0;
  for (const b of activeBoxes(frame)) {
    if (b.status === "keep") keep++;
    else if (b.status === "pending") pending++;
    else if (b.status === "deleted") deleted++;
    else if ((b.status || "").startsWith("reject")) rejected++;
  }
  return {keep, pending, rejected, deleted};
}

function renderFrameList() {
  const div = document.getElementById("frameList");
  div.innerHTML = "";
  state.frames.forEach((f, i) => {
    const c = countFrame(f);
    const item = document.createElement("div");
    item.className = "frameItem " + (i === frameIndex ? "active " : "") + (c.keep === f.expected_pigs && c.pending === 0 ? "ok" : "problem");
    item.innerHTML = `<b>${f.scan_frame_id}</b><br>keep=${c.keep}, pending=${c.pending}, rejected=${c.rejected}`;
    item.onclick = () => loadFrame(i);
    div.appendChild(item);
  });
}

function loadFrame(i) {
  frameIndex = Math.max(0, Math.min(state.frames.length - 1, i));
  selectedBoxId = null;
  previewBox = null;
  const f = currentFrame();

  document.getElementById("frameTitle").innerText = `${f.scan_frame_id} (${frameIndex + 1}/${state.frames.length})`;

  img = new Image();
  img.onload = () => {
    const maxW = Math.max(700, window.innerWidth - 620);
    scale = Math.min(1.0, maxW / f.width);
    canvas.width = Math.round(f.width * scale);
    canvas.height = Math.round(f.height * scale);
    draw();
  };
  img.src = f.image_url;
  renderFrameList();
  updateSidePanel();
}

function nextFrame() {
  loadFrame(frameIndex + 1);
}

function prevFrame() {
  loadFrame(frameIndex - 1);
}

function setMode(m) {
  mode = m;
  document.getElementById("selectBtn").style.background = m === "select" ? "#bbdefb" : "white";
  document.getElementById("drawBtn").style.background = m === "draw" ? "#bbdefb" : "white";
}

function colorForBox(b) {
  if (b.status === "keep") return "lime";
  if (b.status === "pending") {
    if ((b.candidate_class || "").includes("new") || b.candidate_class === "manual_added") return "deepskyblue";
    return "orange";
  }
  if ((b.status || "").startsWith("reject")) return "red";
  if (b.status === "deleted") return "gray";
  return "yellow";
}

function draw() {
  const f = currentFrame();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  for (const b of activeBoxes(f)) {
    const x = b.x * scale;
    const y = b.y * scale;
    const w = b.w * scale;
    const h = b.h * scale;

    ctx.lineWidth = b.id === selectedBoxId ? 4 : 2;
    ctx.strokeStyle = colorForBox(b);
    if (b.status === "deleted") {
      ctx.globalAlpha = 0.35;
    } else {
      ctx.globalAlpha = 1.0;
    }
    ctx.strokeRect(x, y, w, h);
    ctx.globalAlpha = 1.0;

    const label = `${b.status}:${shortId(b.id)}`;
    ctx.font = "12px Arial";
    ctx.fillStyle = colorForBox(b);
    ctx.fillText(label, x + 3, Math.max(12, y - 4));
  }

  if (previewBox) {
    ctx.strokeStyle = "cyan";
    ctx.lineWidth = 2;
    ctx.strokeRect(previewBox.x * scale, previewBox.y * scale, previewBox.w * scale, previewBox.h * scale);
  }

  updateSidePanel();
}

function shortId(id) {
  id = String(id);
  return id.length > 18 ? id.slice(-18) : id;
}

function mousePos(e) {
  const r = canvas.getBoundingClientRect();
  return {
    x: (e.clientX - r.left) / scale,
    y: (e.clientY - r.top) / scale
  };
}

function findBoxAt(p) {
  const boxes = activeBoxes(currentFrame());
  for (let i = boxes.length - 1; i >= 0; i--) {
    const b = boxes[i];
    if (p.x >= b.x && p.x <= b.x + b.w && p.y >= b.y && p.y <= b.y + b.h) {
      return b;
    }
  }
  return null;
}

canvas.addEventListener("mousedown", (e) => {
  const p = mousePos(e);
  if (mode === "draw") {
    drawStart = p;
    previewBox = {x: p.x, y: p.y, w: 1, h: 1};
  } else {
    const b = findBoxAt(p);
    selectedBoxId = b ? b.id : null;
    if (b) {
      drag = {
        boxId: b.id,
        offsetX: p.x - b.x,
        offsetY: p.y - b.y
      };
    }
  }
  draw();
});

canvas.addEventListener("mousemove", (e) => {
  const p = mousePos(e);
  const f = currentFrame();

  if (mode === "draw" && drawStart) {
    previewBox = {
      x: Math.min(drawStart.x, p.x),
      y: Math.min(drawStart.y, p.y),
      w: Math.abs(p.x - drawStart.x),
      h: Math.abs(p.y - drawStart.y)
    };
    draw();
  } else if (mode === "select" && drag) {
    const b = f.boxes.find(x => x.id === drag.boxId);
    if (b) {
      b.x = Math.max(0, Math.min(f.width - b.w, p.x - drag.offsetX));
      b.y = Math.max(0, Math.min(f.height - b.h, p.y - drag.offsetY));
      draw();
    }
  }
});

canvas.addEventListener("mouseup", (e) => {
  const f = currentFrame();

  if (mode === "draw" && previewBox) {
    if (previewBox.w > 8 && previewBox.h > 8) {
      const id = `${f.scan_frame_id}_manual_${Date.now()}`;
      f.boxes.push({
        id: id,
        scan_frame_id: f.scan_frame_id,
        x: previewBox.x,
        y: previewBox.y,
        w: previewBox.w,
        h: previewBox.h,
        status: "keep",
        label: "gt_pen_pig",
        candidate_class: "manual_added",
        candidate_source: "manual_drawn_in_v10_editor",
        candidate_score: 1.0,
        detector_score: "",
        manual_note: "manual added box"
      });
      selectedBoxId = id;
    }
    previewBox = null;
    drawStart = null;
  }

  drag = null;
  draw();
  renderFrameList();
});

function getSelectedBox() {
  if (!selectedBoxId) return null;
  return currentFrame().boxes.find(b => b.id === selectedBoxId) || null;
}

function setSelectedStatus(status) {
  const b = getSelectedBox();
  if (!b) {
    alert("Select a box first.");
    return;
  }
  b.status = status;
  draw();
  renderFrameList();
}

function updateSidePanel() {
  const b = getSelectedBox();
  const info = document.getElementById("selectedInfo");

  if (!b) {
    info.innerText = "No selected box.";
    document.getElementById("xInput").value = "";
    document.getElementById("yInput").value = "";
    document.getElementById("wInput").value = "";
    document.getElementById("hInput").value = "";
    document.getElementById("noteInput").value = "";
  } else {
    info.innerText =
      `id: ${b.id}\n` +
      `status: ${b.status}\n` +
      `class: ${b.candidate_class}\n` +
      `score: ${b.candidate_score}\n` +
      `x,y,w,h: ${b.x.toFixed(1)}, ${b.y.toFixed(1)}, ${b.w.toFixed(1)}, ${b.h.toFixed(1)}`;

    document.getElementById("xInput").value = Math.round(b.x);
    document.getElementById("yInput").value = Math.round(b.y);
    document.getElementById("wInput").value = Math.round(b.w);
    document.getElementById("hInput").value = Math.round(b.h);
    document.getElementById("noteInput").value = b.manual_note || "";
  }

  const f = currentFrame();
  const c = countFrame(f);
  document.getElementById("frameSummary").innerText =
    `expected pigs: ${f.expected_pigs}\n` +
    `keep: ${c.keep}\n` +
    `pending: ${c.pending}\n` +
    `rejected: ${c.rejected}\n` +
    `deleted: ${c.deleted}\n` +
    `missing vs expected: ${Math.max(0, f.expected_pigs - c.keep)}\n` +
    `extra vs expected: ${Math.max(0, c.keep - f.expected_pigs)}`;
}

function applyCoords() {
  const b = getSelectedBox();
  const f = currentFrame();
  if (!b) {
    alert("Select a box first.");
    return;
  }
  b.x = Math.max(0, Number(document.getElementById("xInput").value));
  b.y = Math.max(0, Number(document.getElementById("yInput").value));
  b.w = Math.max(1, Number(document.getElementById("wInput").value));
  b.h = Math.max(1, Number(document.getElementById("hInput").value));

  if (b.x + b.w > f.width) b.w = f.width - b.x;
  if (b.y + b.h > f.height) b.h = f.height - b.y;

  draw();
}

function applyNote() {
  const b = getSelectedBox();
  if (!b) {
    alert("Select a box first.");
    return;
  }
  b.manual_note = document.getElementById("noteInput").value;
  draw();
}

async function saveState() {
  state.last_saved_at = new Date().toISOString();
  const res = await fetch("/save", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(state)
  });

  const data = await res.json();

  document.getElementById("statusMsg").innerText =
    `Saved. Final boxes: ${data.final_box_count}. Complete frames: ${data.complete_frames}/${data.total_frames}.`;

  renderFrameList();
}

document.addEventListener("keydown", (e) => {
  if (e.key === "ArrowRight") nextFrame();
  if (e.key === "ArrowLeft") prevFrame();
  if (e.key === "k") setSelectedStatus("keep");
  if (e.key === "r") setSelectedStatus("reject_wrong_pen");
  if (e.key === "p") setSelectedStatus("pending");
  if (e.key === "d") setSelectedStatus("deleted");
});

setMode("select");
init();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, content_type="text/plain"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self._send(200, HTML, "text/html; charset=utf-8")
            return

        if path == "/data":
            state = load_state()
            self._send(200, json.dumps(state), "application/json")
            return

        if path.startswith("/images/"):
            name = Path(unquote(path.replace("/images/", ""))).name
            p = IMG_DIR / name

            if not p.exists():
                self._send(404, "image not found")
                return

            ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
            self._send(200, p.read_bytes(), ctype)
            return

        if path == "/download_json":
            state = load_state()
            self._send(200, json.dumps(state, indent=2), "application/json")
            return

        if path == "/download_csv":
            state = load_state()
            generate_outputs(state)
            self._send(200, FINAL_CSV.read_text(), "text/csv")
            return

        self._send(404, "not found")

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path != "/save":
            self._send(404, "not found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)

        state = json.loads(body.decode("utf-8"))
        state["last_saved_at"] = datetime.now().isoformat(timespec="seconds")

        SAVED_JSON.write_text(json.dumps(state, indent=2))
        generate_outputs(state)

        frame_summary = pd.read_csv(FRAME_SUMMARY_CSV)

        payload = {
            "ok": True,
            "saved_json": str(SAVED_JSON),
            "final_csv": str(FINAL_CSV),
            "all_decisions_csv": str(ALL_DECISIONS_CSV),
            "frame_summary_csv": str(FRAME_SUMMARY_CSV),
            "final_box_count": int(pd.read_csv(FINAL_CSV).shape[0]) if FINAL_CSV.exists() and FINAL_CSV.stat().st_size > 0 else 0,
            "complete_frames": int(frame_summary["frame_complete_candidate"].sum()) if "frame_complete_candidate" in frame_summary.columns else 0,
            "total_frames": int(frame_summary.shape[0]),
        }

        self._send(200, json.dumps(payload), "application/json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8510)
    args = parser.parse_args()

    state = load_state()
    generate_outputs(state)

    print("Week 7 Manual Box Editor v10")
    print(f"Open: http://127.0.0.1:{args.port}/")
    print()
    print("Outputs:")
    print(SAVED_JSON)
    print(FINAL_CSV)
    print(ALL_DECISIONS_CSV)
    print(FRAME_SUMMARY_CSV)

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
