from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from datetime import datetime
import json
import csv
import argparse
import mimetypes
import math

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V27A_SELECTED = W7 / "outputs" / "detector_tracker_preflight_v27a" / "week7_detector_tracker_preflight_v27a_selected_dryrun_clips.csv"
V28A_ROIS = W7 / "outputs" / "roi_filtered_threshold_sweep_v28a" / "week7_roi_filtered_threshold_sweep_v28a_roi_candidates.csv"

OUT_ROOT = W7 / "outputs" / "target_pen_polygon_roi_v28b"
OVERLAY_ROOT = OUT_ROOT / "polygon_overlay_review"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
OVERLAY_ROOT.mkdir(parents=True, exist_ok=True)

STATE_JSON = OUT_ROOT / "week7_target_pen_polygon_roi_v28b_state.json"
OUT_POLYGONS = OUT_ROOT / "week7_target_pen_polygon_roi_v28b_polygons.csv"
OUT_SUMMARY = OUT_ROOT / "week7_target_pen_polygon_roi_v28b_summary.csv"
OUT_ISSUES = OUT_ROOT / "week7_target_pen_polygon_roi_v28b_issues.csv"
OUT_OVERLAY_INDEX = OUT_ROOT / "week7_target_pen_polygon_roi_v28b_overlay_index.csv"
OUT_LIMITATIONS = OUT_ROOT / "week7_target_pen_polygon_roi_v28b_limitations.md"
OUT_README = OUT_ROOT / "README_target_pen_polygon_roi_v28b.md"
OUT_NOTE = W7 / "notes" / "week7_target_pen_polygon_roi_v28b_notes.md"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def clean(v):
    if pd.isna(v):
        return ""
    return str(v).strip()


def slug(s):
    s = clean(s)
    out = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "unknown"


def load_font(size=14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def polygon_area(points):
    if not points or len(points) < 3:
        return 0.0

    area = 0.0
    n = len(points)

    for i in range(n):
        x1, y1 = points[i]["x"], points[i]["y"]
        x2, y2 = points[(i + 1) % n]["x"], points[(i + 1) % n]["y"]
        area += x1 * y2 - x2 * y1

    return abs(area) / 2.0


def rect_to_polygon(x1, y1, x2, y2):
    return [
        {"x": float(x1), "y": float(y1)},
        {"x": float(x2), "y": float(y1)},
        {"x": float(x2), "y": float(y2)},
        {"x": float(x1), "y": float(y2)},
    ]


def build_initial_state():
    selected = pd.read_csv(V27A_SELECTED)
    rois = pd.read_csv(V28A_ROIS)

    for df in [selected, rois]:
        if "scan_frame_id" in df.columns:
            df["scan_frame_id"] = df["scan_frame_id"].fillna("").astype(str).str.strip()

    items = []

    for _, r in selected.iterrows():
        sid = clean(r.get("scan_frame_id", ""))
        img_path = clean(r.get("preview_frame_path", ""))

        if not img_path or not Path(img_path).exists():
            img_path = clean(r.get("clip_path", ""))

        width, height = 704, 576
        if img_path and Path(img_path).exists() and Path(img_path).suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
            try:
                im = Image.open(img_path)
                width, height = im.size
            except Exception:
                pass

        # Initial polygon is scanpoint_wide rectangle from v28a, only as starting point.
        rg = rois[(rois["scan_frame_id"] == sid) & (rois["roi_type"] == "scanpoint_wide")]

        if len(rg):
            rr = rg.iloc[0]
            initial_polygon = rect_to_polygon(
                float(rr["roi_x1"]),
                float(rr["roi_y1"]),
                float(rr["roi_x2"]),
                float(rr["roi_y2"]),
            )
        else:
            # Fallback: conservative central polygon.
            initial_polygon = rect_to_polygon(120, 50, width - 60, height - 90)

        items.append({
            "scan_frame_id": sid,
            "video_id": clean(r.get("video_id", "")),
            "selection_reason": clean(r.get("selection_reason", "")),
            "behaviour_codes_present": clean(r.get("behaviour_codes_present", "")),
            "expected_pig_rows": clean(r.get("pig_rows", "")),
            "image_path": img_path,
            "image_width": width,
            "image_height": height,
            "initial_polygon": initial_polygon,
            "manual_polygon": initial_polygon,
            "status": "needs_review",
            "manual_notes": "",
            "updated_at": "",
        })

    state = {
        "version": "v28b",
        "purpose": "Manual geometry-aware target-pen polygon ROI design before ROI-filtered tracking.",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "items": items,
    }

    STATE_JSON.write_text(json.dumps(state, indent=2))
    return state


def load_state():
    if STATE_JSON.exists():
        return json.loads(STATE_JSON.read_text())
    return build_initial_state()


def save_state(state):
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    STATE_JSON.write_text(json.dumps(state, indent=2))


def draw_overlay_for_item(item):
    img_path = Path(item["image_path"])

    if not img_path.exists():
        return None, "image_missing"

    try:
        im = Image.open(img_path).convert("RGB")
    except Exception as e:
        return None, f"image_open_failed: {repr(e)}"

    draw = ImageDraw.Draw(im)
    font = load_font(13)
    title_font = load_font(16)

    poly = item.get("manual_polygon", [])
    initial = item.get("initial_polygon", [])

    # Draw initial polygon in orange.
    if initial and len(initial) >= 3:
        pts = [(float(p["x"]), float(p["y"])) for p in initial]
        draw.line(pts + [pts[0]], fill=(255, 170, 0), width=3)

    # Draw manual polygon in green.
    if poly and len(poly) >= 3:
        pts = [(float(p["x"]), float(p["y"])) for p in poly]
        draw.polygon(pts, outline=(0, 210, 0))
        draw.line(pts + [pts[0]], fill=(0, 210, 0), width=5)

        for i, p in enumerate(poly):
            x, y = int(round(float(p["x"]))), int(round(float(p["y"])))
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(0, 210, 0))
            draw.text((x + 7, y - 7), str(i + 1), fill=(0, 120, 0), font=font)

    # Title panel.
    draw.rectangle((0, 0, im.width, 62), fill=(255, 255, 255))
    title = f"{item['scan_frame_id']} | {item['selection_reason']} | status={item['status']}"
    draw.text((8, 8), title, fill=(0, 0, 0), font=title_font)
    draw.text((8, 36), "orange=initial scanpoint_wide rectangle | green=manual polygon", fill=(0, 0, 0), font=font)

    out_path = OVERLAY_ROOT / f"{slug(item['scan_frame_id'])}_manual_polygon_roi_v28b.jpg"
    im.save(out_path, quality=95)

    return str(out_path), ""


def export_outputs():
    state = load_state()
    rows = []
    issues = []
    overlay_rows = []

    for item in state["items"]:
        poly = item.get("manual_polygon", [])
        area = polygon_area(poly)

        status = item.get("status", "needs_review")

        if len(poly) < 3:
            issues.append({
                "scan_frame_id": item["scan_frame_id"],
                "issue_type": "invalid_polygon",
                "issue_detail": "Manual polygon has fewer than 3 points.",
            })

        if status != "confirmed":
            issues.append({
                "scan_frame_id": item["scan_frame_id"],
                "issue_type": "polygon_not_confirmed",
                "issue_detail": f"status={status}",
            })

        overlay_path, overlay_issue = draw_overlay_for_item(item)

        if overlay_issue:
            issues.append({
                "scan_frame_id": item["scan_frame_id"],
                "issue_type": "overlay_generation_issue",
                "issue_detail": overlay_issue,
            })

        if overlay_path:
            overlay_rows.append({
                "scan_frame_id": item["scan_frame_id"],
                "video_id": item["video_id"],
                "overlay_path": overlay_path,
                "status": status,
            })

        rows.append({
            "scan_frame_id": item["scan_frame_id"],
            "video_id": item["video_id"],
            "selection_reason": item["selection_reason"],
            "behaviour_codes_present": item.get("behaviour_codes_present", ""),
            "expected_pig_rows": item.get("expected_pig_rows", ""),
            "image_path": item["image_path"],
            "image_width": item.get("image_width", ""),
            "image_height": item.get("image_height", ""),
            "polygon_points_json": json.dumps(poly),
            "polygon_point_count": len(poly),
            "polygon_area_px": round(area, 4),
            "status": status,
            "manual_notes": item.get("manual_notes", ""),
            "updated_at": item.get("updated_at", ""),
        })

    polygons = pd.DataFrame(rows)
    overlay_index = pd.DataFrame(overlay_rows)
    issues_df = pd.DataFrame(issues, columns=["scan_frame_id", "issue_type", "issue_detail"])

    confirmed = int((polygons["status"] == "confirmed").sum()) if len(polygons) else 0
    total = int(len(polygons))

    summary = pd.DataFrame([{
        "version": "v28b",
        "total_polygon_items": total,
        "confirmed_polygon_items": confirmed,
        "needs_review_items": int(total - confirmed),
        "issue_count": int(len(issues_df)),
        "state_json": str(STATE_JSON),
        "polygon_csv": str(OUT_POLYGONS),
        "overlay_review_dir": str(OVERLAY_ROOT),
        "ready_for_v28c_polygon_filtered_detector_dryrun": bool(total > 0 and confirmed == total and len(issues_df) == 0),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])

    safe_to_csv(polygons, OUT_POLYGONS)
    safe_to_csv(overlay_index, OUT_OVERLAY_INDEX)
    safe_to_csv(issues_df, OUT_ISSUES)
    safe_to_csv(summary, OUT_SUMMARY)

    OUT_LIMITATIONS.write_text(
        "# Week 7 v28b Polygon ROI Limitations\n\n"
        "## What this step does\n\n"
        "This step manually defines geometry-aware target-pen polygon ROIs for representative dry-run clips.\n\n"
        "## What this step does not do\n\n"
        "It does not perform final tracking and does not claim stable identity tracking.\n\n"
        "## Important notes\n\n"
        "1. Initial polygons are only starting rectangles derived from v28a scanpoint_wide ROI.\n"
        "2. Manual polygons must be visually confirmed before use.\n"
        "3. Polygon filtering is more appropriate than rectangular ROI because the pen boundary is perspective-distorted.\n"
        "4. v28c must test whether polygon filtering reduces leakage without cutting target pigs.\n"
        "5. v28d or later should evaluate improved tracking association after polygon filtering.\n"
    )

    OUT_README.write_text(
        "# Week 7 Manual Target-Pen Polygon ROI v28b\n\n"
        "## Purpose\n\n"
        "This package stores manually reviewed target-pen polygon ROI definitions.\n\n"
        "## Outputs\n\n"
        "- `week7_target_pen_polygon_roi_v28b_state.json`\n"
        "- `week7_target_pen_polygon_roi_v28b_polygons.csv`\n"
        "- `polygon_overlay_review/`\n"
        "- `week7_target_pen_polygon_roi_v28b_summary.csv`\n"
        "- `week7_target_pen_polygon_roi_v28b_issues.csv`\n\n"
        "## Next step\n\n"
        "If all polygon items are confirmed, proceed to v28c polygon-filtered detector dry-run.\n"
    )

    OUT_NOTE.write_text(
        "# Week 7 Manual Target-Pen Polygon ROI v28b\n\n"
        "## Purpose\n\n"
        "This step creates manually reviewed polygon ROIs for target-pen filtering.\n\n"
        "## Summary\n\n"
        f"- Total polygon items: `{total}`\n"
        f"- Confirmed polygon items: `{confirmed}`\n"
        f"- Needs review: `{total - confirmed}`\n"
        f"- Issue count: `{len(issues_df)}`\n"
        f"- Ready for v28c polygon-filtered detector dry-run: `{bool(total > 0 and confirmed == total and len(issues_df) == 0)}`\n\n"
        "## Outputs\n\n"
        f"- State JSON: `{STATE_JSON}`\n"
        f"- Polygon CSV: `{OUT_POLYGONS}`\n"
        f"- Overlay review: `{OVERLAY_ROOT}`\n"
        f"- Summary: `{OUT_SUMMARY}`\n"
        f"- Issues: `{OUT_ISSUES}`\n"
    )

    return {
        "summary": summary.to_dict(orient="records")[0],
        "polygons": str(OUT_POLYGONS),
        "overlays": str(OVERLAY_ROOT),
        "issues": str(OUT_ISSUES),
        "note": str(OUT_NOTE),
    }


HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Week7 v28b Polygon ROI Editor</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 16px; background: #f7f7f7; }
    .top { display: flex; gap: 16px; align-items: flex-start; }
    .panel { background: white; border: 1px solid #ccc; padding: 12px; border-radius: 8px; }
    canvas { background: #222; border: 1px solid #333; cursor: crosshair; max-width: 100%; }
    button { margin: 3px; padding: 7px 10px; }
    textarea { width: 420px; height: 90px; }
    .small { font-size: 12px; color: #555; }
    .ok { color: green; font-weight: bold; }
    .warn { color: #b55b00; font-weight: bold; }
    .bad { color: #b00000; font-weight: bold; }
    pre { background: #eee; padding: 8px; overflow-x: auto; }
  </style>
</head>
<body>
  <h2>Week 7 v28b — Manual Target-Pen Polygon ROI Editor</h2>

  <div class="top">
    <div class="panel">
      <canvas id="canvas"></canvas>
      <p class="small">
        Click to add/select points. Drag points to move. Use Undo/Clear if needed.
        Green polygon = manual ROI. Orange polygon = initial scanpoint_wide rectangle.
      </p>
    </div>

    <div class="panel" style="min-width: 450px;">
      <h3 id="title"></h3>
      <p id="meta"></p>

      <button onclick="prevItem()">Prev</button>
      <button onclick="nextItem()">Next</button>
      <button onclick="loadInitial()">Reset to initial rectangle</button>
      <button onclick="undoPoint()">Undo last point</button>
      <button onclick="clearPoly()">Clear polygon</button>
      <br>
      <button onclick="saveCurrent(false)">Save as needs_review</button>
      <button onclick="saveCurrent(true)">Save + Confirm</button>
      <button onclick="exportOutputs()">Export outputs</button>

      <p>Status: <span id="status"></span></p>

      <label>Manual notes:</label><br>
      <textarea id="notes"></textarea>

      <h4>Professional drawing rules</h4>
      <ul>
        <li>Include only the target pen region where labelled pigs should be tracked.</li>
        <li>Exclude neighbouring pen pigs on the left/right when possible.</li>
        <li>Exclude lower corridor / outside pig region when possible.</li>
        <li>Do not make polygon too tight; allow small movement during 10-sec clip.</li>
        <li>4–8 points is usually enough.</li>
      </ul>

      <h4>Coordinates</h4>
      <pre id="coords"></pre>

      <div id="message"></div>
    </div>
  </div>

<script>
let state = null;
let idx = 0;
let img = new Image();
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");
let scale = 1.0;
let selectedPoint = -1;
let dragging = false;

function getItem() { return state.items[idx]; }

async function loadState() {
  const res = await fetch('/api/state');
  state = await res.json();
  idx = 0;
  loadItem();
}

function naturalToCanvas(p) {
  return {x: p.x * scale, y: p.y * scale};
}

function canvasToNatural(x, y) {
  return {x: x / scale, y: y / scale};
}

function loadItem() {
  let item = getItem();
  img.onload = function() {
    let maxW = 900;
    scale = Math.min(1.0, maxW / img.naturalWidth);
    canvas.width = Math.round(img.naturalWidth * scale);
    canvas.height = Math.round(img.naturalHeight * scale);
    draw();
  };
  img.src = '/image?idx=' + idx + '&t=' + Date.now();

  document.getElementById("title").innerText = `${idx+1}/${state.items.length} | ${item.scan_frame_id}`;
  document.getElementById("meta").innerHTML =
    `<b>video:</b> ${item.video_id}<br>` +
    `<b>reason:</b> ${item.selection_reason}<br>` +
    `<b>behaviours:</b> ${item.behaviour_codes_present}<br>` +
    `<b>expected pigs:</b> ${item.expected_pig_rows}`;

  document.getElementById("notes").value = item.manual_notes || "";
  updateStatus();
  updateCoords();
}

function drawPolygon(poly, color, width, fill=false) {
  if (!poly || poly.length === 0) return;

  ctx.beginPath();
  let p0 = naturalToCanvas(poly[0]);
  ctx.moveTo(p0.x, p0.y);

  for (let i = 1; i < poly.length; i++) {
    let p = naturalToCanvas(poly[i]);
    ctx.lineTo(p.x, p.y);
  }

  if (poly.length >= 3) ctx.closePath();

  if (fill) {
    ctx.globalAlpha = 0.12;
    ctx.fillStyle = color;
    ctx.fill();
    ctx.globalAlpha = 1.0;
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();

  for (let i = 0; i < poly.length; i++) {
    let p = naturalToCanvas(poly[i]);
    ctx.beginPath();
    ctx.arc(p.x, p.y, i === selectedPoint ? 8 : 5, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.fillStyle = "white";
    ctx.font = "12px Arial";
    ctx.fillText(String(i+1), p.x + 7, p.y - 7);
  }
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  let item = getItem();
  drawPolygon(item.initial_polygon, "orange", 2, false);
  drawPolygon(item.manual_polygon, "lime", 4, true);
}

function getMouse(e) {
  let rect = canvas.getBoundingClientRect();
  return {x: e.clientX - rect.left, y: e.clientY - rect.top};
}

function findPoint(mx, my) {
  let poly = getItem().manual_polygon || [];
  for (let i = 0; i < poly.length; i++) {
    let p = naturalToCanvas(poly[i]);
    let d = Math.hypot(p.x - mx, p.y - my);
    if (d < 12) return i;
  }
  return -1;
}

canvas.addEventListener("mousedown", (e) => {
  let m = getMouse(e);
  let hit = findPoint(m.x, m.y);

  if (hit >= 0) {
    selectedPoint = hit;
    dragging = true;
  } else {
    let p = canvasToNatural(m.x, m.y);
    getItem().manual_polygon.push({x: Math.round(p.x * 10) / 10, y: Math.round(p.y * 10) / 10});
    selectedPoint = getItem().manual_polygon.length - 1;
  }

  draw();
  updateCoords();
});

canvas.addEventListener("mousemove", (e) => {
  if (!dragging || selectedPoint < 0) return;
  let m = getMouse(e);
  let p = canvasToNatural(m.x, m.y);
  getItem().manual_polygon[selectedPoint] = {
    x: Math.max(0, Math.min(img.naturalWidth, Math.round(p.x * 10) / 10)),
    y: Math.max(0, Math.min(img.naturalHeight, Math.round(p.y * 10) / 10))
  };
  draw();
  updateCoords();
});

canvas.addEventListener("mouseup", () => { dragging = false; });
canvas.addEventListener("mouseleave", () => { dragging = false; });

function updateStatus() {
  let s = getItem().status || "needs_review";
  let cls = s === "confirmed" ? "ok" : "warn";
  document.getElementById("status").innerHTML = `<span class="${cls}">${s}</span>`;
}

function updateCoords() {
  let poly = getItem().manual_polygon || [];
  document.getElementById("coords").innerText = JSON.stringify(poly, null, 2);
}

function prevItem() {
  saveCurrent(false, true);
  idx = Math.max(0, idx - 1);
  selectedPoint = -1;
  loadItem();
}

function nextItem() {
  saveCurrent(false, true);
  idx = Math.min(state.items.length - 1, idx + 1);
  selectedPoint = -1;
  loadItem();
}

function loadInitial() {
  getItem().manual_polygon = JSON.parse(JSON.stringify(getItem().initial_polygon));
  getItem().status = "needs_review";
  draw();
  updateStatus();
  updateCoords();
}

function undoPoint() {
  getItem().manual_polygon.pop();
  selectedPoint = -1;
  draw();
  updateCoords();
}

function clearPoly() {
  getItem().manual_polygon = [];
  getItem().status = "needs_review";
  selectedPoint = -1;
  draw();
  updateStatus();
  updateCoords();
}

async function saveCurrent(confirm=false, silent=false) {
  let item = getItem();
  item.manual_notes = document.getElementById("notes").value;
  item.status = confirm ? "confirmed" : (item.status || "needs_review");
  if (!confirm && item.status !== "confirmed") item.status = "needs_review";
  item.updated_at = new Date().toISOString();

  const res = await fetch('/api/save', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(state)
  });

  const data = await res.json();
  updateStatus();

  if (!silent) {
    document.getElementById("message").innerHTML = `<p class="ok">${data.message}</p>`;
  }
}

async function exportOutputs() {
  await saveCurrent(false, true);
  const res = await fetch('/api/export', { method: 'POST' });
  const data = await res.json();
  document.getElementById("message").innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
}

loadState();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, status=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, data, content_type="application/octet-stream", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._send_bytes(HTML.encode("utf-8"), "text/html")
            return

        if parsed.path == "/api/state":
            self._send_json(load_state())
            return

        if parsed.path == "/image":
            qs = parse_qs(parsed.query)
            idx = int(qs.get("idx", ["0"])[0])
            state = load_state()

            try:
                item = state["items"][idx]
                p = Path(item["image_path"])
                data = p.read_bytes()
                ctype = mimetypes.guess_type(str(p))[0] or "image/jpeg"
                self._send_bytes(data, ctype)
            except Exception as e:
                self._send_json({"error": repr(e)}, status=404)
            return

        self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/save":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            state = json.loads(body.decode("utf-8"))
            save_state(state)
            self._send_json({"ok": True, "message": "Saved state."})
            return

        if parsed.path == "/api/export":
            try:
                result = export_outputs()
                self._send_json({"ok": True, **result})
            except Exception as e:
                self._send_json({"ok": False, "error": repr(e)}, status=500)
            return

        self._send_json({"error": "not found"}, status=404)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8512)
    args = parser.parse_args()

    load_state()

    print("Week7 v28b manual polygon ROI editor")
    print(f"Open: http://127.0.0.1:{args.port}/")
    print(f"State: {STATE_JSON}")
    print("Stop with Ctrl+C")

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
