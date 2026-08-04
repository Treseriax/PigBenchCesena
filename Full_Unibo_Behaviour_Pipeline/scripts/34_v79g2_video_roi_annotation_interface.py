from pathlib import Path
from datetime import datetime
import json
import csv
import re
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import pandas as pd

ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

RUN_PLAN = FULL / "outputs/v79b_tracking_manifest_audit_and_run_plan/v79b_tracking_run_plan_36_videos.csv"

OUT = FULL / "outputs/v79g2_video_roi_annotation_interface"
FRAMES = OUT / "roi_reference_frames"
OUT.mkdir(parents=True, exist_ok=True)
FRAMES.mkdir(parents=True, exist_ok=True)

ROI_CSV = OUT / "v79g2_video_roi_annotations_WORKING.csv"
DECISION = OUT / "v79g2_decision_summary.csv"
ISSUES = OUT / "v79g2_issues.csv"
NOTE = FULL / "notes/v79g2_video_roi_annotation_interface_notes.md"
NOTE.parent.mkdir(parents=True, exist_ok=True)

PORT = 8557
REFERENCE_SEC = 10

def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s

def safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", clean(s))

def write(df, p):
    df.to_csv(p, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")

def load_run_plan():
    df = pd.read_csv(RUN_PLAN).fillna("")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(clean)
    return df

def extract_frames(run_plan):
    import cv2

    rows = []
    issues = []

    for _, r in run_plan.iterrows():
        video_id = clean(r["video_id"])
        video_path = Path(clean(r["video_path"]))
        frame_path = FRAMES / f"{video_id}_{safe(video_path.stem)}_roi_ref.jpg"

        exists = video_path.exists()
        ok_frame = frame_path.exists()

        width = height = 0
        fps = 0.0
        frame_index = 0
        err = ""

        if exists and not frame_path.exists():
            try:
                cap = cv2.VideoCapture(str(video_path))
                opened = cap.isOpened()
                if opened:
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
                    frame_index = int(round(REFERENCE_SEC * fps))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                    if ok and frame is not None:
                        height, width = frame.shape[:2]
                        ok_frame = bool(cv2.imwrite(str(frame_path), frame))
                cap.release()
            except Exception as e:
                err = str(e)

        if frame_path.exists():
            try:
                import cv2
                img = cv2.imread(str(frame_path))
                if img is not None:
                    height, width = img.shape[:2]
                    ok_frame = True
            except Exception:
                pass

        if not exists:
            issues.append({
                "item": video_id,
                "issue_type": "hard_video_missing",
                "severity": "hard",
                "detail": str(video_path),
            })

        if not ok_frame:
            issues.append({
                "item": video_id,
                "issue_type": "hard_reference_frame_missing",
                "severity": "hard",
                "detail": str(frame_path) + " " + err,
            })

        rows.append({
            "video_id": video_id,
            "video_filename": clean(r["video_filename"]),
            "video_path": str(video_path),
            "date": clean(r["date"]),
            "tlc_camera": clean(r["tlc_camera"]),
            "room_pen": clean(r["room_pen"]),
            "c_code": clean(r.get("c_code", "")),
            "clip_count": clean(r.get("clip_count", "")),
            "frame_path": str(frame_path),
            "frame_rel": "roi_reference_frames/" + frame_path.name,
            "frame_width": width,
            "frame_height": height,
            "reference_sec": REFERENCE_SEC,
        })

    return pd.DataFrame(rows), issues

def load_existing_roi():
    if ROI_CSV.exists():
        df = pd.read_csv(ROI_CSV).fillna("")
        return df
    return pd.DataFrame()

def init_roi_table(frame_table):
    old = load_existing_roi()
    old_map = {}
    if len(old):
        for _, r in old.iterrows():
            old_map[clean(r.get("video_id"))] = dict(r)

    rows = []
    for _, r in frame_table.iterrows():
        vid = clean(r["video_id"])
        prev = old_map.get(vid, {})
        rows.append({
            "video_id": vid,
            "video_filename": clean(r["video_filename"]),
            "video_path": clean(r["video_path"]),
            "date": clean(r["date"]),
            "tlc_camera": clean(r["tlc_camera"]),
            "room_pen": clean(r["room_pen"]),
            "c_code": clean(r["c_code"]),
            "frame_path": clean(r["frame_path"]),
            "frame_rel": clean(r["frame_rel"]),
            "frame_width": clean(r["frame_width"]),
            "frame_height": clean(r["frame_height"]),
            "roi_status": clean(prev.get("roi_status", "UNREVIEWED")) or "UNREVIEWED",
            "roi_points_json": clean(prev.get("roi_points_json", "")),
            "roi_point_count": clean(prev.get("roi_point_count", "")),
            "reviewer_note": clean(prev.get("reviewer_note", "")),
            "updated_at": clean(prev.get("updated_at", "")),
        })
    df = pd.DataFrame(rows)
    write(df, ROI_CSV)
    return df

run_plan = load_run_plan()
frame_table, frame_issues = extract_frames(run_plan)
roi_table = init_roi_table(frame_table)

hard_count = sum(1 for x in frame_issues if x["severity"] == "hard")

decision = pd.DataFrame([{
    "v79g2_decision": "roi_annotation_interface_ready" if hard_count == 0 else "roi_annotation_interface_has_blocking_issues",
    "video_rows": len(run_plan),
    "reference_frames": int(frame_table["frame_path"].map(lambda p: Path(p).exists()).sum()),
    "roi_csv": str(ROI_CSV),
    "hard_issue_count": hard_count,
    "ready_for_manual_roi_annotation": hard_count == 0,
    "claim_scope": "roi_annotation_interface_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
write(decision, DECISION)

issues = frame_issues + [{
    "item": "scope",
    "issue_type": "info_roi_annotation_interface_only",
    "severity": "info",
    "detail": "v79g2 creates a manual ROI annotation interface. It does not filter tracks yet.",
}]
write(pd.DataFrame(issues), ISSUES)

NOTE.write_text(
    "# v79g2 Video ROI Annotation Interface\n\n"
    f"- Decision: {decision.iloc[0]['v79g2_decision']}\n"
    f"- Video rows: {len(run_plan)}\n"
    f"- Reference frames: {decision.iloc[0]['reference_frames']}\n"
    f"- ROI CSV: {ROI_CSV}\n"
    f"- Hard issues: {hard_count}\n\n"
    "Draw one polygon ROI per video. Coordinates are stored in image pixel coordinates.\n",
    encoding="utf-8"
)

def current_roi_records():
    df = pd.read_csv(ROI_CSV).fillna("")
    records = []
    for _, r in df.iterrows():
        points = []
        raw = clean(r.get("roi_points_json", ""))
        if raw:
            try:
                points = json.loads(raw)
            except Exception:
                points = []
        rec = dict(r)
        rec["points"] = points
        records.append(rec)
    return records

HTML = r"""
<html>
<head>
<meta charset="utf-8">
<title>v79g2 ROI Annotation</title>
<style>
body{font-family:Arial;margin:18px}
#wrap{display:flex;gap:20px}
#left{width:360px}
#right{flex:1}
select,button,input,textarea{margin:4px;padding:6px}
canvas{border:1px solid #333;max-width:100%}
table{border-collapse:collapse}
td,th{border:1px solid #ccc;padding:4px;font-size:12px}
.ok{color:green;font-weight:bold}
.bad{color:red;font-weight:bold}
</style>
</head>
<body>
<h1>v79g2 Video ROI Annotation</h1>
<p>Draw polygon around ONLY the target pen region. Click points on the image. Save each video.</p>

<div id="wrap">
<div id="left">
<div><b>Status:</b> <span id="count"></span></div>
<select id="videoSelect" size="22" style="width:340px"></select><br>
<button onclick="undoPoint()">Undo point</button>
<button onclick="clearPoints()">Clear</button>
<button onclick="saveROI('RESOLVED')">Save ROI</button>
<button onclick="saveROI('UNCERTAIN')">Save uncertain</button>
<br>
<textarea id="note" rows="4" cols="42" placeholder="note optional"></textarea>
<p>
<b>Current:</b><br>
<span id="meta"></span>
</p>
<p>
Rules:<br>
1. Include target pen floor area.<br>
2. Exclude neighbouring pens as much as possible.<br>
3. If unsure, save as UNCERTAIN with note.<br>
4. Do not guess if the pen is not visible.
</p>
</div>

<div id="right">
<canvas id="canvas"></canvas>
</div>
</div>

<script>
let records = __RECORDS__;
let selected = 0;
let points = [];
let img = new Image();
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");

function statusIcon(r){
  if(r.roi_status === "RESOLVED") return "✅";
  if(r.roi_status === "UNCERTAIN") return "⚠️";
  return "⬜";
}

function refreshList(){
  let sel = document.getElementById("videoSelect");
  sel.innerHTML = "";
  records.forEach((r,i)=>{
    let opt = document.createElement("option");
    opt.value = i;
    opt.text = statusIcon(r)+" "+r.video_id+" "+r.tlc_camera+" "+r.room_pen+" "+r.video_filename;
    sel.add(opt);
  });
  sel.value = selected;
  let done = records.filter(r=>r.roi_status==="RESOLVED").length;
  let uncertain = records.filter(r=>r.roi_status==="UNCERTAIN").length;
  document.getElementById("count").innerHTML = done+"/"+records.length+" resolved, "+uncertain+" uncertain";
}

function loadSelected(){
  let r = records[selected];
  points = r.points || [];
  document.getElementById("note").value = r.reviewer_note || "";
  document.getElementById("meta").innerHTML =
    r.video_id+"<br>"+r.tlc_camera+" / "+r.room_pen+"<br>"+r.video_filename+"<br>Status: "+r.roi_status;
  img = new Image();
  img.onload = function(){
    canvas.width = img.width;
    canvas.height = img.height;
    draw();
  };
  img.src = r.frame_rel + "?t=" + Date.now();
}

function draw(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(img,0,0);
  if(points.length){
    ctx.lineWidth = 3;
    ctx.strokeStyle = "yellow";
    ctx.fillStyle = "rgba(255,255,0,0.18)";
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for(let i=1;i<points.length;i++) ctx.lineTo(points[i].x, points[i].y);
    if(points.length >= 3) ctx.closePath();
    ctx.stroke();
    if(points.length >= 3) ctx.fill();
  }
  ctx.fillStyle = "red";
  points.forEach((p,i)=>{
    ctx.beginPath();
    ctx.arc(p.x,p.y,5,0,Math.PI*2);
    ctx.fill();
    ctx.fillText(String(i+1),p.x+6,p.y-6);
  });
}

canvas.addEventListener("click", function(e){
  let rect = canvas.getBoundingClientRect();
  let scaleX = canvas.width / rect.width;
  let scaleY = canvas.height / rect.height;
  points.push({
    x: Math.round((e.clientX - rect.left)*scaleX),
    y: Math.round((e.clientY - rect.top)*scaleY)
  });
  draw();
});

document.getElementById("videoSelect").addEventListener("change", function(){
  selected = parseInt(this.value);
  loadSelected();
});

function undoPoint(){
  points.pop();
  draw();
}

function clearPoints(){
  points = [];
  draw();
}

async function saveROI(status){
  if(status === "RESOLVED" && points.length < 3){
    alert("Need at least 3 points for RESOLVED ROI");
    return;
  }
  let r = records[selected];
  let payload = {
    video_id: r.video_id,
    roi_status: status,
    points: points,
    reviewer_note: document.getElementById("note").value
  };
  let resp = await fetch("/save", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify(payload)
  });
  let data = await resp.json();
  if(!data.ok){
    alert("Save failed: "+data.error);
    return;
  }
  r.roi_status = status;
  r.points = points;
  r.roi_points_json = JSON.stringify(points);
  r.roi_point_count = points.length;
  r.reviewer_note = payload.reviewer_note;
  refreshList();
  if(selected < records.length-1){
    selected += 1;
    document.getElementById("videoSelect").value = selected;
    loadSelected();
  }
}

refreshList();
loadSelected();
</script>
</body>
</html>
"""

class Handler(BaseHTTPRequestHandler):
    def _send(self, code, content, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            html = HTML.replace("__RECORDS__", json.dumps(current_roi_records()))
            return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")

        if path.startswith("/roi_reference_frames/"):
            f = FRAMES / Path(path).name
            if f.exists():
                return self._send(200, f.read_bytes(), "image/jpeg")

        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/save":
            return self._send(404, b'{"ok":false}', "application/json")

        try:
            n = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(n).decode("utf-8"))

            video_id = clean(data.get("video_id"))
            status = clean(data.get("roi_status"))
            points = data.get("points", [])
            note = clean(data.get("reviewer_note"))

            df = pd.read_csv(ROI_CSV).fillna("")
            idxs = df.index[df["video_id"].astype(str) == video_id].tolist()
            if not idxs:
                raise ValueError("video_id not found")

            idx = idxs[0]
            df.loc[idx, "roi_status"] = status
            df.loc[idx, "roi_points_json"] = json.dumps(points)
            df.loc[idx, "roi_point_count"] = len(points)
            df.loc[idx, "reviewer_note"] = note
            df.loc[idx, "updated_at"] = datetime.now().isoformat(timespec="seconds")
            write(df, ROI_CSV)

            return self._send(200, b'{"ok":true}', "application/json")
        except Exception as e:
            out = json.dumps({"ok": False, "error": str(e)}).encode("utf-8")
            return self._send(500, out, "application/json")

print("=== v79g2 decision ===")
print(decision.to_string(index=False))
print()
print("ROI CSV:", ROI_CSV)
print("Open:")
print(f"http://137.204.72.3:{PORT}/index.html")
print()
print("Ctrl+C to stop server.")

ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
