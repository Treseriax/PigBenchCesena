from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs, unquote
from datetime import datetime
import argparse
import csv
import json
import mimetypes
import os

import pandas as pd


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V15_ROOT = W7 / "outputs" / "colour_identity" / "segmented_enhanced_colour_evidence_v15"
V15_BOX_QA = V15_ROOT / "week7_segmented_enhanced_colour_evidence_v15_box_mask_qa.csv"
V15_FRAME_SUMMARY = V15_ROOT / "week7_segmented_enhanced_colour_evidence_v15_frame_summary.csv"

OUT_ROOT = W7 / "outputs" / "colour_identity" / "manual_colour_identity_v16"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

STATE_PATH = OUT_ROOT / "week7_manual_colour_identity_v16_state.json"
FINAL_CSV = OUT_ROOT / "week7_manual_colour_identity_v16_final_assignments.csv"
FRAME_SUMMARY_CSV = OUT_ROOT / "week7_manual_colour_identity_v16_frame_summary.csv"
NOTE_PATH = W7 / "notes" / "week7_manual_colour_identity_v16_notes.md"

VALID_COLOURS = ["blue", "green", "cyan", "red", "pink", "purple"]
SPECIAL_VALUES = ["uncertain", "not_visible", ""]


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def load_items():
    qa = pd.read_csv(V15_BOX_QA)
    fs = pd.read_csv(V15_FRAME_SUMMARY)

    overlay_map = {}
    if "overlay_path" in fs.columns:
        for _, r in fs.iterrows():
            overlay_map[str(r["scan_frame_id"])] = str(r["overlay_path"])

    items = []

    for _, r in qa.sort_values(["scan_frame_id", "final_box_id"]).iterrows():
        sid = str(r["scan_frame_id"])
        bid = str(r["final_box_id"])
        key = f"{sid}||{bid}"

        top = str(r.get("top_colour_v15", "")).strip().lower()
        if top not in VALID_COLOURS:
            top = ""

        conf = str(r.get("evidence_confidence_v15", "")).strip()

        default_status = "suggested_high_or_medium" if conf in ["high", "medium"] else "needs_check"

        items.append({
            "key": key,
            "scan_frame_id": sid,
            "final_box_id": bid,
            "x1": int(r.get("x1", 0)),
            "y1": int(r.get("y1", 0)),
            "x2": int(r.get("x2", 0)),
            "y2": int(r.get("y2", 0)),
            "suggested_colour_v15": top,
            "suggested_confidence_v15": conf,
            "top_score_v15": float(r.get("top_score_v15", 0.0)) if pd.notna(r.get("top_score_v15", 0.0)) else 0.0,
            "second_colour_v15": str(r.get("second_colour_v15", "")).strip(),
            "second_score_v15": float(r.get("second_score_v15", 0.0)) if pd.notna(r.get("second_score_v15", 0.0)) else 0.0,
            "masked_crop_path": str(r.get("masked_crop_path", "")),
            "enhanced_lab_clahe_path": str(r.get("enhanced_lab_clahe_path", "")),
            "enhanced_hsv_saturation_path": str(r.get("enhanced_hsv_saturation_path", "")),
            "enhanced_combo_path": str(r.get("enhanced_combo_path", "")),
            "frame_overlay_path": overlay_map.get(sid, ""),
            "default_status": default_status,
        })

    return items


ITEMS = load_items()
ITEM_BY_KEY = {x["key"]: x for x in ITEMS}
FRAMES = sorted({x["scan_frame_id"] for x in ITEMS})


def initial_state():
    assignments = {}
    statuses = {}
    notes = {}

    for item in ITEMS:
        key = item["key"]
        assignments[key] = item["suggested_colour_v15"]
        statuses[key] = item["default_status"]
        notes[key] = ""

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "assignments": assignments,
        "statuses": statuses,
        "notes": notes,
    }


def load_state():
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH, "r") as f:
                state = json.load(f)
        except Exception:
            state = initial_state()
    else:
        state = initial_state()

    # Add any new keys if missing.
    for item in ITEMS:
        key = item["key"]
        state.setdefault("assignments", {})
        state.setdefault("statuses", {})
        state.setdefault("notes", {})

        state["assignments"].setdefault(key, item["suggested_colour_v15"])
        state["statuses"].setdefault(key, item["default_status"])
        state["notes"].setdefault(key, "")

    return state


def write_state_and_outputs(state):
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")

    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)

    rows = []

    for item in ITEMS:
        key = item["key"]
        selected = str(state.get("assignments", {}).get(key, "")).strip().lower()
        status = str(state.get("statuses", {}).get(key, "")).strip()
        note = str(state.get("notes", {}).get(key, "")).strip()

        final_colour = selected if selected in VALID_COLOURS else ""
        final_status = status

        if selected == "uncertain":
            final_status = "uncertain"
        elif selected == "not_visible":
            final_status = "not_visible"
        elif selected not in VALID_COLOURS:
            final_status = "unassigned"

        rows.append({
            "scan_frame_id": item["scan_frame_id"],
            "final_box_id": item["final_box_id"],
            "final_colour_v16": final_colour,
            "manual_selection_v16": selected,
            "manual_status_v16": final_status,
            "manual_note_v16": note,
            "suggested_colour_v15": item["suggested_colour_v15"],
            "suggested_confidence_v15": item["suggested_confidence_v15"],
            "top_score_v15": item["top_score_v15"],
            "second_colour_v15": item["second_colour_v15"],
            "second_score_v15": item["second_score_v15"],
            "x1": item["x1"],
            "y1": item["y1"],
            "x2": item["x2"],
            "y2": item["y2"],
        })

    out = pd.DataFrame(rows)
    safe_to_csv(out, FINAL_CSV)

    frame_rows = []

    for sid, g in out.groupby("scan_frame_id", sort=True):
        valid = [c for c in g["final_colour_v16"].tolist() if c in VALID_COLOURS]
        counts = pd.Series(valid).value_counts()
        duplicates = [c for c, n in counts.items() if n > 1]
        missing = sorted(set(VALID_COLOURS) - set(valid))

        confirmed_like = int(g["manual_status_v16"].isin(["confirmed", "manual_confirmed", "suggested_high_or_medium"]).sum())
        uncertain_count = int((g["manual_status_v16"] == "uncertain").sum())
        not_visible_count = int((g["manual_status_v16"] == "not_visible").sum())
        unassigned_count = int((g["manual_status_v16"] == "unassigned").sum())

        frame_rows.append({
            "scan_frame_id": sid,
            "box_count": int(len(g)),
            "valid_colour_assignments": int(len(valid)),
            "unique_valid_colour_count": int(len(set(valid))),
            "duplicate_colours": " | ".join(duplicates),
            "missing_colours": " | ".join(missing),
            "confirmed_or_suggested_count": confirmed_like,
            "uncertain_count": uncertain_count,
            "not_visible_count": not_visible_count,
            "unassigned_count": unassigned_count,
            "frame_needs_review_v16": bool(len(duplicates) > 0 or uncertain_count > 0 or unassigned_count > 0),
        })

    frame_summary = pd.DataFrame(frame_rows)
    safe_to_csv(frame_summary, FRAME_SUMMARY_CSV)

    total = len(out)
    valid_total = int((out["final_colour_v16"].isin(VALID_COLOURS)).sum())
    uncertain_total = int((out["manual_status_v16"] == "uncertain").sum())
    not_visible_total = int((out["manual_status_v16"] == "not_visible").sum())
    frame_review = int(frame_summary["frame_needs_review_v16"].sum())

    NOTE_PATH.write_text(
        "# Week 7 Manual Colour Identity v16\n\n"
        "## Purpose\n\n"
        "This step manually verifies/corrects colour identity assignments using v15 segmentation-enhanced evidence as a suggestion layer.\n\n"
        "## Valid colours\n\n"
        "`blue`, `green`, `cyan`, `red`, `pink`, `purple`\n\n"
        "## Summary\n\n"
        f"- Total boxes: `{total}`\n"
        f"- Boxes with valid colour assignment: `{valid_total}`\n"
        f"- Uncertain boxes: `{uncertain_total}`\n"
        f"- Not-visible boxes: `{not_visible_total}`\n"
        f"- Frames still needing review: `{frame_review}`\n\n"
        "## Outputs\n\n"
        f"- State JSON: `{STATE_PATH}`\n"
        f"- Final assignments CSV: `{FINAL_CSV}`\n"
        f"- Frame summary CSV: `{FRAME_SUMMARY_CSV}`\n"
    )

    return {
        "total_boxes": total,
        "valid_colour_assignments": valid_total,
        "uncertain_boxes": uncertain_total,
        "not_visible_boxes": not_visible_total,
        "frames_needing_review": frame_review,
        "final_csv": str(FINAL_CSV),
        "frame_summary_csv": str(FRAME_SUMMARY_CSV),
    }


STATE = load_state()
write_state_and_outputs(STATE)


def is_allowed_path(p):
    try:
        rp = Path(p).expanduser().resolve()
    except Exception:
        return None

    allowed_roots = [
        W7.resolve(),
        ROOT.resolve(),
    ]

    for root in allowed_roots:
        try:
            rp.relative_to(root)
            return rp
        except ValueError:
            pass

    return None


HTML = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Week 7 Manual Colour Identity v16</title>
<style>
body {
  font-family: Arial, sans-serif;
  background: #f5f5f5;
  margin: 0;
}
header {
  position: sticky;
  top: 0;
  z-index: 10;
  background: #222;
  color: white;
  padding: 12px 18px;
}
button {
  cursor: pointer;
  border: 1px solid #999;
  border-radius: 6px;
  padding: 6px 9px;
  margin: 2px;
}
select, input {
  padding: 5px;
  margin: 3px;
}
.container {
  display: flex;
}
.sidebar {
  width: 260px;
  background: white;
  height: calc(100vh - 55px);
  overflow-y: auto;
  border-right: 1px solid #ddd;
  padding: 10px;
  position: sticky;
  top: 55px;
}
.main {
  flex: 1;
  padding: 18px;
}
.frame-btn {
  display: block;
  width: 100%;
  text-align: left;
  margin: 4px 0;
  background: #fafafa;
}
.frame-btn.review {
  border-left: 7px solid darkorange;
}
.frame-btn.ok {
  border-left: 7px solid green;
}
.frame-panel {
  background: white;
  padding: 14px;
  border-radius: 8px;
  margin-bottom: 20px;
}
.overlay {
  max-width: 100%;
  border: 1px solid #ccc;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 14px;
}
.card {
  background: #fff;
  border: 1px solid #ccc;
  border-radius: 8px;
  padding: 10px;
}
.card.review {
  border-left: 8px solid darkorange;
}
.card.confirmed {
  border-left: 8px solid green;
}
.crop-row {
  display: flex;
  gap: 6px;
  align-items: flex-start;
}
.crop-row img {
  max-width: 32%;
  border: 1px solid #ddd;
}
.color-btn {
  color: white;
  font-weight: bold;
}
.blue { background: #0055cc; }
.green { background: #008800; }
.cyan { background: #008b8b; }
.red { background: #cc0000; }
.pink { background: #cc6699; }
.purple { background: #7b1fa2; }
.special { background: #666; color: white; }
.warn {
  background: #fff4d5;
  border-left: 6px solid orange;
  padding: 8px;
  margin: 8px 0;
}
.okbox {
  background: #e8f5e9;
  border-left: 6px solid green;
  padding: 8px;
  margin: 8px 0;
}
.small {
  font-size: 12px;
  color: #555;
}
</style>
</head>
<body>
<header>
  <b>Week 7 Manual Colour Identity v16</b>
  <button onclick="saveAll()">Save</button>
  <button onclick="confirmCurrentFrame()">Confirm current frame</button>
  <span id="saveStatus"></span>
</header>

<div class="container">
  <div class="sidebar">
    <input id="filterBox" placeholder="filter frame..." oninput="renderSidebar()" style="width:95%;">
    <div id="summaryBox"></div>
    <div id="frameList"></div>
  </div>
  <div class="main">
    <div id="content"></div>
  </div>
</div>

<script>
const VALID = ["blue", "green", "cyan", "red", "pink", "purple"];
let DATA = null;
let STATE = null;
let currentFrame = null;

async function loadData() {
  const res = await fetch("/api/data");
  DATA = await res.json();
  STATE = DATA.state;
  currentFrame = DATA.frames[0];
  renderSidebar();
  renderFrame(currentFrame);
}

function imgUrl(path) {
  if (!path) return "";
  return "/file?path=" + encodeURIComponent(path);
}

function getFrameItems(sid) {
  return DATA.items.filter(x => x.scan_frame_id === sid);
}

function selected(key) {
  return (STATE.assignments[key] || "");
}

function statusOf(key) {
  return (STATE.statuses[key] || "");
}

function noteOf(key) {
  return (STATE.notes[key] || "");
}

function setColour(key, colour) {
  STATE.assignments[key] = colour;
  if (VALID.includes(colour)) {
    STATE.statuses[key] = "manual_confirmed";
  } else if (colour === "uncertain") {
    STATE.statuses[key] = "uncertain";
  } else if (colour === "not_visible") {
    STATE.statuses[key] = "not_visible";
  }
  renderFrame(currentFrame);
  renderSidebar();
}

function setStatus(key, st) {
  STATE.statuses[key] = st;
  renderFrame(currentFrame);
  renderSidebar();
}

function setNote(key, val) {
  STATE.notes[key] = val;
}

function frameQA(sid) {
  const items = getFrameItems(sid);
  const cols = items.map(x => selected(x.key)).filter(c => VALID.includes(c));
  const counts = {};
  cols.forEach(c => counts[c] = (counts[c] || 0) + 1);
  const dups = Object.keys(counts).filter(c => counts[c] > 1);
  const missing = VALID.filter(c => !cols.includes(c));
  const bad = items.filter(x => !VALID.includes(selected(x.key)) || ["uncertain","not_visible","unassigned","needs_check"].includes(statusOf(x.key)));
  return {cols, dups, missing, bad};
}

function frameNeedsReview(sid) {
  const q = frameQA(sid);
  return q.dups.length > 0 || q.bad.length > 0;
}

function renderSidebar() {
  const filter = document.getElementById("filterBox").value.toLowerCase();
  const list = document.getElementById("frameList");
  const summary = document.getElementById("summaryBox");

  const reviewCount = DATA.frames.filter(f => frameNeedsReview(f)).length;
  const validAssigned = DATA.items.filter(x => VALID.includes(selected(x.key))).length;

  summary.innerHTML = `
    <p class="small">
      Frames: ${DATA.frames.length}<br>
      Review frames: ${reviewCount}<br>
      Valid assigned boxes: ${validAssigned}/${DATA.items.length}
    </p>
  `;

  list.innerHTML = "";

  DATA.frames.forEach(f => {
    if (filter && !f.toLowerCase().includes(filter)) return;
    const q = frameQA(f);
    const btn = document.createElement("button");
    btn.className = "frame-btn " + (frameNeedsReview(f) ? "review" : "ok");
    btn.innerText = `${f} | missing:${q.missing.length} dup:${q.dups.length} bad:${q.bad.length}`;
    btn.onclick = () => { currentFrame = f; renderFrame(f); };
    list.appendChild(btn);
  });
}

function renderFrame(sid) {
  const content = document.getElementById("content");
  const items = getFrameItems(sid);
  const q = frameQA(sid);

  let qaHtml = "";
  if (q.dups.length || q.bad.length) {
    qaHtml = `<div class="warn">
      <b>Review needed</b><br>
      Duplicate colours: ${q.dups.join(", ") || "none"}<br>
      Missing colours: ${q.missing.join(", ") || "none"}<br>
      Boxes needing attention: ${q.bad.length}
    </div>`;
  } else {
    qaHtml = `<div class="okbox"><b>Frame looks consistent.</b> Missing colours: ${q.missing.join(", ") || "none"}</div>`;
  }

  const overlay = items.length ? items[0].frame_overlay_path : "";

  let cards = items.map(item => {
    const key = item.key;
    const sel = selected(key);
    const st = statusOf(key);
    const cardClass = (VALID.includes(sel) && !["needs_check","uncertain","not_visible"].includes(st)) ? "confirmed" : "review";

    const buttons = VALID.map(c =>
      `<button class="color-btn ${c}" onclick="setColour('${key}', '${c}')">${c}</button>`
    ).join("");

    return `
      <div class="card ${cardClass}">
        <h3>${item.final_box_id}</h3>
        <p>
          <b>Selected:</b> ${sel || "none"} |
          <b>Status:</b> ${st}<br>
          <b>v15 suggestion:</b> ${item.suggested_colour_v15 || "none"}
          (${item.suggested_confidence_v15}, score=${item.top_score_v15.toFixed(2)})
          <br>
          second: ${item.second_colour_v15 || "none"} (${item.second_score_v15.toFixed(2)})
        </p>

        <div>
          ${buttons}
          <button class="special" onclick="setColour('${key}', 'uncertain')">uncertain</button>
          <button class="special" onclick="setColour('${key}', 'not_visible')">not_visible</button>
        </div>

        <div>
          <label>Status:</label>
          <select onchange="setStatus('${key}', this.value)">
            ${["manual_confirmed","suggested_high_or_medium","needs_check","uncertain","not_visible"].map(s =>
              `<option value="${s}" ${st === s ? "selected" : ""}>${s}</option>`
            ).join("")}
          </select>
        </div>

        <div>
          <input style="width:95%;" value="${noteOf(key).replaceAll('"', '&quot;')}" placeholder="note..."
                 onchange="setNote('${key}', this.value)">
        </div>

        <div class="crop-row">
          <img src="${imgUrl(item.masked_crop_path)}" title="masked crop">
          <img src="${imgUrl(item.enhanced_combo_path)}" title="combo enhanced">
          <img src="${imgUrl(item.enhanced_hsv_saturation_path)}" title="hsv saturation">
        </div>
      </div>
    `;
  }).join("");

  content.innerHTML = `
    <div class="frame-panel">
      <h2>${sid}</h2>
      ${qaHtml}
      <img class="overlay" src="${imgUrl(overlay)}">
    </div>
    <div class="grid">${cards}</div>
  `;
}

function confirmCurrentFrame() {
  getFrameItems(currentFrame).forEach(item => {
    if (VALID.includes(selected(item.key))) {
      STATE.statuses[item.key] = "manual_confirmed";
    }
  });
  renderFrame(currentFrame);
  renderSidebar();
}

async function saveAll() {
  document.getElementById("saveStatus").innerText = "Saving...";
  const res = await fetch("/api/save", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(STATE)
  });
  const out = await res.json();
  document.getElementById("saveStatus").innerText = "Saved. Valid assigned: " + out.valid_colour_assignments + "/" + out.total_boxes + " | review frames: " + out.frames_needing_review;
}

loadData();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, status=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_text(self, text, status=200, content_type="text/html; charset=utf-8"):
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._send_text(HTML)
            return

        if parsed.path == "/api/data":
            state = load_state()
            payload = {
                "frames": FRAMES,
                "items": ITEMS,
                "state": state,
                "valid_colours": VALID_COLOURS,
                "outputs": {
                    "state_path": str(STATE_PATH),
                    "final_csv": str(FINAL_CSV),
                    "frame_summary_csv": str(FRAME_SUMMARY_CSV),
                }
            }
            self._send_json(payload)
            return

        if parsed.path == "/file":
            qs = parse_qs(parsed.query)
            raw = qs.get("path", [""])[0]
            raw = unquote(raw)
            rp = is_allowed_path(raw)

            if rp is None or not rp.exists() or not rp.is_file():
                self.send_response(404)
                self.end_headers()
                return

            ctype = mimetypes.guess_type(str(rp))[0] or "application/octet-stream"

            with open(rp, "rb") as f:
                data = f.read()

            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/save":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)

            try:
                state = json.loads(body.decode("utf-8"))
                summary = write_state_and_outputs(state)
                self._send_json(summary)
            except Exception as e:
                self._send_json({"error": repr(e)}, status=500)
            return

        self.send_response(404)
        self.end_headers()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8511)
    args = parser.parse_args()

    print("Manual colour identity editor v16")
    print(f"Open: http://127.0.0.1:{args.port}/")
    print(f"State: {STATE_PATH}")
    print(f"Final CSV: {FINAL_CSV}")
    print("Stop with Ctrl+C")

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
