from pathlib import Path
from datetime import datetime
import csv
import json
import hashlib
import zipfile
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V69C_OUT = W8 / "outputs" / "v69c_error_analysis"
V69C_PKG = V69C_OUT / "Week8_Feature_Baseline_Error_Analysis"
V69C_DECISION = V69C_OUT / "week8_v69c_decision_summary.csv"

PRED = V69C_PKG / "week8_v69c_best_model_predictions.csv"
ERRORS = V69C_PKG / "week8_v69c_best_model_errors.csv"
CONF = V69C_PKG / "week8_v69c_top_confusions.csv"
CLASS_SUMMARY = V69C_PKG / "week8_v69c_per_class_error_summary.csv"
SCAN_SUMMARY = V69C_PKG / "week8_v69c_scanframe_error_summary.csv"
VIDEO_SUMMARY = V69C_PKG / "week8_v69c_video_error_summary.csv"

OUT = W8 / "outputs" / "v69d_error_analysis_interface"
INTERFACE = W8 / "interface"
STATIC = INTERFACE / "static_v69d"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, INTERFACE, STATIC, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

SERVER = INTERFACE / "week8_error_analysis_server_v69d.py"
INDEX = STATIC / "index.html"
CSS = STATIC / "style.css"
JS = STATIC / "app.js"

OUT_DECISION = OUT / "week8_v69d_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v69d_issues.csv"
OUT_AUDIT = OUT / "week8_v69d_interface_input_audit.csv"
OUT_ZIP = OUT / "Week8_Error_Analysis_Interface_v69d.zip"
OUT_SHA256 = OUT / "Week8_Error_Analysis_Interface_v69d.sha256"
OUT_NOTE = NOTES / "week8_v69d_error_analysis_interface_notes.md"
OUT_REPORT = REPORTS / "week8_v69d_error_analysis_interface_report.md"
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


def read_csv_clean(path):
    df = pd.read_csv(path).fillna("")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(clean)
    return df


def bool_true(x):
    return str(x).strip().lower() == "true"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


issues = []

required = [V69C_DECISION, PRED, ERRORS, CONF, CLASS_SUMMARY, SCAN_SUMMARY, VIDEO_SUMMARY]
for p in required:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required v69c output missing for v69d interface.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v69d_decision": "error_analysis_interface_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_manual_error_inspection": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)

v69c_decision = read_csv_clean(V69C_DECISION)
if len(v69c_decision) == 0 or not bool_true(v69c_decision.iloc[0].get("ready_for_v70a_improved_baseline", "")):
    issues.append({
        "item": str(V69C_DECISION),
        "issue_type": "hard_v69c_not_ready",
        "issue_detail": "v69c must be completed before v69d interface.",
        "severity": "hard",
    })

pred = read_csv_clean(PRED)
errors = read_csv_clean(ERRORS)
conf = read_csv_clean(CONF)
cls = read_csv_clean(CLASS_SUMMARY)
scan = read_csv_clean(SCAN_SUMMARY)
video = read_csv_clean(VIDEO_SUMMARY)

required_pred_cols = [
    "split_policy",
    "crop_type",
    "model_type",
    "canonical_gt_object_id",
    "scan_frame_id",
    "video_id",
    "actual",
    "predicted",
    "correct",
    "confidence_like_score",
    "crop_path",
]

for c in required_pred_cols:
    if c not in pred.columns:
        issues.append({
            "item": c,
            "issue_type": "hard_missing_prediction_column",
            "issue_detail": "Prediction column missing.",
            "severity": "hard",
        })

crop_exists = 0
if "crop_path" in pred.columns:
    crop_exists = int(pred["crop_path"].map(lambda x: Path(str(x)).exists()).sum())

audit = pd.DataFrame([{
    "prediction_rows": len(pred),
    "error_rows": len(errors),
    "correct_rows": int((pred["correct"].astype(str) == "True").sum()) if "correct" in pred.columns else "",
    "crop_paths_existing": crop_exists,
    "top_confusion_rows": len(conf),
    "class_summary_rows": len(cls),
    "scanframe_summary_rows": len(scan),
    "video_summary_rows": len(video),
}])
safe_to_csv(audit, OUT_AUDIT)

if len(pred) != 150:
    issues.append({
        "item": "prediction_rows",
        "issue_type": "hard_unexpected_prediction_row_count",
        "issue_detail": f"Expected 150 predictions, found {len(pred)}.",
        "severity": "hard",
    })

if crop_exists != len(pred):
    issues.append({
        "item": "crop_paths",
        "issue_type": "hard_missing_prediction_crop_images",
        "issue_detail": f"Expected {len(pred)} existing crop paths, found {crop_exists}.",
        "severity": "hard",
    })

server_code = r'''
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote
import csv
import json
import mimetypes
import sys


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

STATIC = W8 / "interface" / "static_v69d"

V69C_PKG = W8 / "outputs" / "v69c_error_analysis" / "Week8_Feature_Baseline_Error_Analysis"
PRED = V69C_PKG / "week8_v69c_best_model_predictions.csv"
ERRORS = V69C_PKG / "week8_v69c_best_model_errors.csv"
CONF = V69C_PKG / "week8_v69c_top_confusions.csv"
CLASS_SUMMARY = V69C_PKG / "week8_v69c_per_class_error_summary.csv"
SCAN_SUMMARY = V69C_PKG / "week8_v69c_scanframe_error_summary.csv"
VIDEO_SUMMARY = V69C_PKG / "week8_v69c_video_error_summary.csv"


def load_csv(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def under_root(path):
    try:
        path.resolve().relative_to(ROOT.resolve())
        return True
    except Exception:
        return False


DATA = {
    "predictions": load_csv(PRED),
    "errors": load_csv(ERRORS),
    "confusions": load_csv(CONF),
    "class_summary": load_csv(CLASS_SUMMARY),
    "scan_summary": load_csv(SCAN_SUMMARY),
    "video_summary": load_csv(VIDEO_SUMMARY),
}


def summary():
    rows = DATA["predictions"]
    errors = [r for r in rows if str(r.get("correct", "")).lower() != "true"]

    by_policy = {}
    for r in rows:
        p = r.get("split_policy", "")
        by_policy.setdefault(p, {"total": 0, "correct": 0, "errors": 0})
        by_policy[p]["total"] += 1
        if str(r.get("correct", "")).lower() == "true":
            by_policy[p]["correct"] += 1
        else:
            by_policy[p]["errors"] += 1

    classes = sorted(set(r.get("actual", "") for r in rows))
    return {
        "prediction_rows": len(rows),
        "error_rows": len(errors),
        "correct_rows": len(rows) - len(errors),
        "classes": classes,
        "by_policy": by_policy,
        "claim_boundary": "v69d is manual visual inspection interface only; it does not change GT or train a model.",
    }


class Handler(BaseHTTPRequestHandler):
    def send_json(self, obj):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def send_file(self, path):
        path = Path(path)
        if not path.exists() or not path.is_file():
            self.send_error(404, "File not found")
            return

        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        b = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            return self.send_file(STATIC / "index.html")

        if path == "/api/summary":
            return self.send_json(summary())

        if path == "/api/predictions":
            return self.send_json(DATA["predictions"])

        if path == "/api/errors":
            return self.send_json(DATA["errors"])

        if path == "/api/confusions":
            return self.send_json(DATA["confusions"])

        if path == "/api/class_summary":
            return self.send_json(DATA["class_summary"])

        if path == "/api/scan_summary":
            return self.send_json(DATA["scan_summary"])

        if path == "/api/video_summary":
            return self.send_json(DATA["video_summary"])

        if path == "/image":
            qs = parse_qs(parsed.query)
            requested = unquote(qs.get("path", [""])[0])
            if not requested:
                self.send_error(400, "Missing path")
                return

            img = Path(requested)
            if not img.exists() or not under_root(img):
                self.send_error(403, "Image path not allowed")
                return

            return self.send_file(img)

        static_path = STATIC / path.lstrip("/")
        if static_path.exists():
            return self.send_file(static_path)

        self.send_error(404, "Not found")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8531)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Week8 v69d error analysis interface running at http://{args.host}:{args.port}/")
    print("Open locally via SSH tunnel / forwarded port: http://127.0.0.1:8531/")
    server.serve_forever()


if __name__ == "__main__":
    main()
'''

html = r'''<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Week8 v69d Error Analysis Interface</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <header>
    <h1>Week8 v69d Error Analysis Interface</h1>
    <p>Visual inspection for v69c feature-baseline predictions. GT is not modified here.</p>
  </header>

  <section id="summary"></section>

  <section id="controls">
    <label>Policy</label>
    <select id="policyFilter"><option value="">All</option></select>

    <label>Actual</label>
    <select id="actualFilter"><option value="">All</option></select>

    <label>Predicted</label>
    <select id="predFilter"><option value="">All</option></select>

    <label>Status</label>
    <select id="statusFilter">
      <option value="errors">Errors only</option>
      <option value="all">All</option>
      <option value="correct">Correct only</option>
    </select>

    <label>Search</label>
    <input id="searchBox" placeholder="scanframe / object / video">
  </section>

  <main>
    <aside>
      <h2>Top Confusions</h2>
      <div id="confusions"></div>

      <h2>Weak Classes</h2>
      <div id="classes"></div>
    </aside>

    <section id="grid"></section>
  </main>

  <script src="app.js"></script>
</body>
</html>
'''

css = r'''
body {
  margin: 0;
  background: #101010;
  color: #eeeeee;
  font-family: Arial, sans-serif;
}

header {
  padding: 16px;
  border-bottom: 1px solid #333;
}

h1 {
  margin: 0 0 6px 0;
}

p {
  color: #bbbbbb;
}

#summary {
  padding: 12px 16px;
  background: #151515;
  border-bottom: 1px solid #333;
  white-space: pre-wrap;
  font-size: 13px;
  color: #dddddd;
}

#controls {
  display: flex;
  gap: 10px;
  align-items: center;
  padding: 12px 16px;
  background: #1b1b1b;
  border-bottom: 1px solid #333;
  position: sticky;
  top: 0;
  z-index: 10;
}

select, input {
  background: #222;
  color: #eee;
  border: 1px solid #555;
  border-radius: 4px;
  padding: 7px;
}

main {
  display: grid;
  grid-template-columns: 360px 1fr;
  gap: 12px;
  padding: 12px;
}

aside {
  background: #171717;
  border: 1px solid #333;
  border-radius: 8px;
  padding: 10px;
  height: calc(100vh - 190px);
  overflow-y: auto;
}

aside h2 {
  font-size: 16px;
  margin-top: 8px;
}

#grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 12px;
}

.card {
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 8px;
  padding: 10px;
}

.card.error {
  border-color: #8a3b3b;
}

.card.correct {
  border-color: #3b7a46;
}

.badges {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  margin-bottom: 8px;
}

.badge {
  background: #333;
  color: #eee;
  padding: 3px 7px;
  border-radius: 4px;
  font-size: 12px;
}

.badge.error {
  background: #7a2828;
}

.badge.correct {
  background: #236d36;
}

.meta {
  color: #bbbbbb;
  font-size: 12px;
  line-height: 1.45;
  margin-top: 8px;
}

.card img {
  width: 100%;
  max-height: 260px;
  object-fit: contain;
  background: #eee;
  border-radius: 6px;
}

.rowlink {
  cursor: pointer;
  padding: 6px;
  border-bottom: 1px solid #333;
  font-size: 13px;
}

.rowlink:hover {
  background: #282828;
}
'''

js = r'''
let SUMMARY = null;
let PRED = [];
let CONF = [];
let CLS = [];

function id(x) { return document.getElementById(x); }

async function loadAll() {
  SUMMARY = await (await fetch("/api/summary")).json();
  PRED = await (await fetch("/api/predictions")).json();
  CONF = await (await fetch("/api/confusions")).json();
  CLS = await (await fetch("/api/class_summary")).json();

  populateFilters();
  renderSummary();
  renderSidebars();
  renderGrid();
}

function unique(arr) {
  return [...new Set(arr.filter(x => x !== undefined && x !== null && String(x).trim() !== ""))].sort();
}

function populateSelect(el, values) {
  for (const v of values) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    el.appendChild(opt);
  }
}

function populateFilters() {
  populateSelect(id("policyFilter"), unique(PRED.map(r => r.split_policy)));
  populateSelect(id("actualFilter"), unique(PRED.map(r => r.actual)));
  populateSelect(id("predFilter"), unique(PRED.map(r => r.predicted)));
}

function renderSummary() {
  let lines = [];
  lines.push(`Prediction rows: ${SUMMARY.prediction_rows}`);
  lines.push(`Correct rows: ${SUMMARY.correct_rows}`);
  lines.push(`Error rows: ${SUMMARY.error_rows}`);
  lines.push("");
  for (const [k, v] of Object.entries(SUMMARY.by_policy)) {
    lines.push(`${k}: total=${v.total}, correct=${v.correct}, errors=${v.errors}`);
  }
  lines.push("");
  lines.push(SUMMARY.claim_boundary);
  id("summary").textContent = lines.join("\n");
}

function setFilter(actual, predicted, policy) {
  if (policy) id("policyFilter").value = policy;
  if (actual) id("actualFilter").value = actual;
  if (predicted) id("predFilter").value = predicted;
  id("statusFilter").value = "errors";
  renderGrid();
}

function renderSidebars() {
  const c = id("confusions");
  c.innerHTML = "";
  for (const r of CONF.slice(0, 25)) {
    const div = document.createElement("div");
    div.className = "rowlink";
    div.textContent = `${r.split_policy} | ${r.actual} → ${r.predicted}: ${r.error_count}`;
    div.onclick = () => setFilter(r.actual, r.predicted, r.split_policy);
    c.appendChild(div);
  }

  const cls = id("classes");
  cls.innerHTML = "";
  const sorted = [...CLS].sort((a, b) => Number(a.recall) - Number(b.recall)).slice(0, 25);
  for (const r of sorted) {
    const div = document.createElement("div");
    div.className = "rowlink";
    div.textContent = `${r.split_policy} | ${r.actual}: recall=${Number(r.recall).toFixed(3)}, support=${r.support}`;
    div.onclick = () => setFilter(r.actual, "", r.split_policy);
    cls.appendChild(div);
  }
}

function keep(r) {
  const p = id("policyFilter").value;
  const a = id("actualFilter").value;
  const pr = id("predFilter").value;
  const st = id("statusFilter").value;
  const q = id("searchBox").value.trim().toLowerCase();

  if (p && r.split_policy !== p) return false;
  if (a && r.actual !== a) return false;
  if (pr && r.predicted !== pr) return false;

  const correct = String(r.correct).toLowerCase() === "true";
  if (st === "errors" && correct) return false;
  if (st === "correct" && !correct) return false;

  if (q) {
    const hay = [
      r.canonical_gt_object_id,
      r.scan_frame_id,
      r.video_id,
      r.actual,
      r.predicted,
      r.split_policy,
      r.crop_type,
      r.model_type
    ].join(" ").toLowerCase();
    if (!hay.includes(q)) return false;
  }

  return true;
}

function renderGrid() {
  const rows = PRED.filter(keep);
  const grid = id("grid");
  grid.innerHTML = "";

  for (const r of rows) {
    const correct = String(r.correct).toLowerCase() === "true";
    const card = document.createElement("div");
    card.className = "card " + (correct ? "correct" : "error");

    const imgUrl = "/image?path=" + encodeURIComponent(r.crop_path);

    card.innerHTML = `
      <div class="badges">
        <span class="badge ${correct ? "correct" : "error"}">${correct ? "CORRECT" : "ERROR"}</span>
        <span class="badge">${r.split_policy}</span>
        <span class="badge">${r.crop_type}</span>
        <span class="badge">${r.model_type}</span>
      </div>

      <img src="${imgUrl}">

      <div class="meta">
        actual: <b>${r.actual}</b> &nbsp; predicted: <b>${r.predicted}</b><br>
        confidence-like score: ${Number(r.confidence_like_score).toFixed(5)}<br>
        scanframe: ${r.scan_frame_id}<br>
        object: ${r.canonical_gt_object_id}<br>
        video: ${r.video_id}
      </div>
    `;

    grid.appendChild(card);
  }
}

for (const el of ["policyFilter", "actualFilter", "predFilter", "statusFilter", "searchBox"]) {
  window.addEventListener("DOMContentLoaded", () => {
    id(el).addEventListener("input", renderGrid);
    id(el).addEventListener("change", renderGrid);
  });
}

loadAll();
'''

SERVER.write_text(server_code)
INDEX.write_text(html)
CSS.write_text(css)
JS.write_text(js)

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in [SERVER, INDEX, CSS, JS, OUT_AUDIT, OUT_ISSUES]:
        if p.exists():
            z.write(p, p.relative_to(W8))

zip_hash = sha256_file(OUT_ZIP)
OUT_SHA256.write_text(f"{zip_hash}  {OUT_ZIP.name}\n")

decision = pd.DataFrame([{
    "v69d_decision": "error_analysis_interface_created" if hard_issue_count == 0 else "error_analysis_interface_has_blocking_issues",
    "prediction_rows": int(len(pred)),
    "error_rows": int(len(errors)),
    "top_confusion_rows": int(len(conf)),
    "class_summary_rows": int(len(cls)),
    "crop_paths_existing": int(crop_exists),
    "server_path": str(SERVER),
    "static_dir": str(STATIC),
    "default_port": 8531,
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "issue_count": int(len(issues_df)),
    "ready_for_manual_error_inspection": bool(hard_issue_count == 0),
    "ready_for_v70a_improved_baseline": bool(hard_issue_count == 0),
    "claim_scope": "interactive_error_inspection_only_no_gt_change_no_model_training",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v69d Error Analysis Interface\n\n"
    f"- v69d decision: {decision.iloc[0]['v69d_decision']}\n"
    f"- Prediction rows: {len(pred)}\n"
    f"- Error rows: {len(errors)}\n"
    f"- Top confusion rows: {len(conf)}\n"
    f"- Class summary rows: {len(cls)}\n"
    f"- Crop paths existing: {crop_exists}\n"
    f"- Server: {SERVER}\n"
    f"- Static dir: {STATIC}\n"
    f"- Default port: 8531\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Ready for manual error inspection: {bool(hard_issue_count == 0)}\n\n"
    "This interface is for visual error inspection only. It does not modify GT and does not train a model.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v69d Error Analysis Interface Report\n\n"
    f"Decision: {decision.iloc[0]['v69d_decision']}\n\n"
    f"Server: {SERVER}\n\n"
    f"Default URL: http://127.0.0.1:8531/\n\n"
    f"ZIP: {OUT_ZIP}\n\n"
    f"SHA256: {zip_hash}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v69d",
    "task_name": "Interactive error analysis interface",
    "status": "PASS" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": str(V69C_PKG),
    "output_summary": str(SERVER),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Inspect errors visually, then run v70a improved baseline." if hard_issue_count == 0 else "Fix v69d interface hard issues.",
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
print(CSS)
print(JS)
print(OUT_AUDIT)
print(OUT_DECISION)
print(OUT_NOTE)
print(OUT_ZIP)
print(OUT_SHA256)

print()
print("=== v69d decision ===")
print(decision.to_string(index=False))

print()
print("=== issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
