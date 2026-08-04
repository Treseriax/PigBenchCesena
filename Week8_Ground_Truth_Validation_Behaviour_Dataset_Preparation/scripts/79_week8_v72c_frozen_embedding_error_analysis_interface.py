from pathlib import Path
from datetime import datetime
import csv
import json
import hashlib
import zipfile
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V72B = W8 / "outputs" / "v72b_frozen_embedding_baseline"
V72B_PKG = V72B / "Week8_Frozen_Embedding_Baseline"

V72B_DECISION = V72B / "week8_v72b_decision_summary.csv"
BEST = V72B_PKG / "week8_v72b_best_test_summary.csv"
PRED = V72B_PKG / "week8_v72b_selected_test_predictions.csv"
COMPARE = V72B_PKG / "week8_v72b_vs_v69b_champion_comparison.csv"

OUT = W8 / "outputs" / "v72c_frozen_embedding_error_analysis_interface"
PKG = OUT / "Week8_Frozen_Embedding_Error_Analysis_Interface"
INTERFACE = W8 / "interface"
STATIC = INTERFACE / "static_v72c"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, PKG, INTERFACE, STATIC, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_CHAMPION_PRED = PKG / "week8_v72c_champion_predictions.csv"
OUT_ERRORS = PKG / "week8_v72c_champion_errors.csv"
OUT_CORRECT = PKG / "week8_v72c_champion_correct_predictions.csv"
OUT_CONF = PKG / "week8_v72c_top_confusions.csv"
OUT_CLASS = PKG / "week8_v72c_per_class_error_summary.csv"
OUT_SCAN = PKG / "week8_v72c_scanframe_error_summary.csv"
OUT_VIDEO = PKG / "week8_v72c_video_error_summary.csv"
OUT_QA = PKG / "week8_v72c_quality_checks.csv"
OUT_README = PKG / "README_Week8_Frozen_Embedding_Error_Analysis_Interface.md"
OUT_MANIFEST = PKG / "week8_v72c_manifest.json"

SERVER = INTERFACE / "week8_frozen_embedding_error_server_v72c.py"
INDEX = STATIC / "index.html"
CSS = STATIC / "style.css"
JS = STATIC / "app.js"

OUT_DECISION = OUT / "week8_v72c_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v72c_issues.csv"
OUT_ZIP = OUT / "Week8_Frozen_Embedding_Error_Analysis_Interface.zip"
OUT_SHA = OUT / "Week8_Frozen_Embedding_Error_Analysis_Interface.sha256"
OUT_NOTE = NOTES / "week8_v72c_frozen_embedding_error_analysis_interface_notes.md"
OUT_REPORT = REPORTS / "week8_v72c_frozen_embedding_error_analysis_interface_report.md"
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

for p in [V72B_DECISION, BEST, PRED, COMPARE]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required v72b output missing.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v72c_decision": "frozen_embedding_error_analysis_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v73_final_modeling_package": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


d72b = read_csv_clean(V72B_DECISION)
best = read_csv_clean(BEST)
pred = read_csv_clean(PRED)
compare = read_csv_clean(COMPARE)

if len(d72b) == 0 or not bool_true(d72b.iloc[0].get("ready_for_v72c_error_analysis", "")):
    issues.append({
        "item": str(V72B_DECISION),
        "issue_type": "hard_v72b_not_ready",
        "issue_detail": "v72b must be ready before v72c.",
        "severity": "hard",
    })

champion_rows = []

for _, b in best.iterrows():
    policy = clean(b["split_policy"])
    crop = clean(b["crop_type"])
    cfg = clean(b["selected_config_name"])

    rows = pred[
        (pred["split_policy"] == policy)
        & (pred["crop_type"] == crop)
        & (pred["selected_config_name"] == cfg)
    ].copy()

    champion_rows.append(rows)

champion = pd.concat(champion_rows, ignore_index=True) if champion_rows else pd.DataFrame()
champion["correct_bool"] = champion["correct"].astype(str).str.lower() == "true"

errors = champion[champion["correct_bool"] == False].copy()
correct = champion[champion["correct_bool"] == True].copy()

safe_to_csv(champion.drop(columns=["correct_bool"]), OUT_CHAMPION_PRED)
safe_to_csv(errors.drop(columns=["correct_bool"]), OUT_ERRORS)
safe_to_csv(correct.drop(columns=["correct_bool"]), OUT_CORRECT)

conf = (
    errors.groupby(["split_policy", "crop_type", "selected_config_name", "actual", "predicted"])
    .size()
    .reset_index(name="error_count")
    .sort_values(["split_policy", "error_count"], ascending=[True, False])
)
safe_to_csv(conf, OUT_CONF)

class_summary = (
    champion.groupby(["split_policy", "crop_type", "selected_config_name", "actual"])
    .agg(
        support=("actual", "size"),
        correct=("correct_bool", "sum"),
        mean_confidence=("confidence_like_score", lambda s: pd.to_numeric(s, errors="coerce").mean()),
    )
    .reset_index()
)
class_summary["errors"] = class_summary["support"] - class_summary["correct"]
class_summary["recall"] = class_summary["correct"] / class_summary["support"].replace(0, pd.NA)
class_summary = class_summary.sort_values(["split_policy", "recall", "support"], ascending=[True, True, True])
safe_to_csv(class_summary, OUT_CLASS)

scan_summary = (
    champion.groupby(["split_policy", "scan_frame_id"])
    .agg(
        test_objects=("actual", "size"),
        errors=("correct_bool", lambda s: int((~s).sum())),
        correct=("correct_bool", "sum"),
    )
    .reset_index()
)
scan_summary["error_rate"] = scan_summary["errors"] / scan_summary["test_objects"].replace(0, pd.NA)
scan_summary = scan_summary.sort_values(["split_policy", "error_rate", "errors"], ascending=[True, False, False])
safe_to_csv(scan_summary, OUT_SCAN)

video_summary = (
    champion.groupby(["split_policy", "video_id"])
    .agg(
        test_objects=("actual", "size"),
        errors=("correct_bool", lambda s: int((~s).sum())),
        correct=("correct_bool", "sum"),
    )
    .reset_index()
)
video_summary["error_rate"] = video_summary["errors"] / video_summary["test_objects"].replace(0, pd.NA)
video_summary = video_summary.sort_values(["split_policy", "error_rate", "errors"], ascending=[True, False, False])
safe_to_csv(video_summary, OUT_VIDEO)

qa_rows = []

def add_qa(name, expected, actual, passed, severity, detail):
    qa_rows.append({
        "check_name": name,
        "expected": str(expected),
        "actual": str(actual),
        "passed": bool(passed),
        "severity": severity,
        "detail": detail,
    })

current_rows = champion[champion["split_policy"] == "current_recommended_split"]
group_rows = champion[champion["split_policy"] == "selected_group_aware_split"]

crop_exists = int(champion["crop_path"].map(lambda p: Path(str(p)).exists()).sum()) if len(champion) else 0

add_qa("champion_prediction_rows", 150, len(champion), len(champion) == 150, "hard", "Expected 78 current + 72 group-aware champion predictions.")
add_qa("current_champion_rows", 78, len(current_rows), len(current_rows) == 78, "hard", "Current split champion should have 78 test predictions.")
add_qa("group_champion_rows", 72, len(group_rows), len(group_rows) == 72, "hard", "Group-aware split champion should have 72 test predictions.")
add_qa("crop_paths_existing", len(champion), crop_exists, crop_exists == len(champion), "hard", "All champion prediction crop paths should exist.")
add_qa("error_rows_nonempty", ">0", len(errors), len(errors) > 0, "info", "Errors should exist because the baseline is not perfect.")
add_qa("top_confusions_nonempty", ">0", len(conf), len(conf) > 0, "hard", "Top confusion table should be created.")
add_qa("class_summary_nonempty", ">0", len(class_summary), len(class_summary) > 0, "hard", "Per-class error summary should exist.")
add_qa("scan_summary_nonempty", ">0", len(scan_summary), len(scan_summary) > 0, "hard", "Scanframe summary should exist.")
add_qa("video_summary_nonempty", ">0", len(video_summary), len(video_summary) > 0, "hard", "Video summary should exist.")

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_QA)

hard_quality_failures = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v72c_quality_checks",
        "issue_type": "hard_error_analysis_quality_failed",
        "issue_detail": f"{hard_quality_failures} hard checks failed.",
        "severity": "hard",
    })

issues.append({
    "item": "claim_scope",
    "issue_type": "info_error_analysis_interface_only",
    "issue_detail": "v72c provides visual error inspection for frozen embedding baseline. It does not modify GT or train a model.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

server_code = r'''
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote
import csv
import json
import mimetypes


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

STATIC = W8 / "interface" / "static_v72c"
PKG = W8 / "outputs" / "v72c_frozen_embedding_error_analysis_interface" / "Week8_Frozen_Embedding_Error_Analysis_Interface"

FILES = {
    "predictions": PKG / "week8_v72c_champion_predictions.csv",
    "errors": PKG / "week8_v72c_champion_errors.csv",
    "confusions": PKG / "week8_v72c_top_confusions.csv",
    "class_summary": PKG / "week8_v72c_per_class_error_summary.csv",
    "scan_summary": PKG / "week8_v72c_scanframe_error_summary.csv",
    "video_summary": PKG / "week8_v72c_video_error_summary.csv",
}


def load_csv(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


DATA = {k: load_csv(v) for k, v in FILES.items()}


def under_root(path):
    try:
        Path(path).resolve().relative_to(ROOT.resolve())
        return True
    except Exception:
        return False


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

    return {
        "prediction_rows": len(rows),
        "correct_rows": len(rows) - len(errors),
        "error_rows": len(errors),
        "by_policy": by_policy,
        "claim_boundary": "v72c is frozen embedding baseline error inspection only. It does not change GT or train a model.",
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

        for key in DATA:
            if path == f"/api/{key}":
                return self.send_json(DATA[key])

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
    parser.add_argument("--port", type=int, default=8532)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)

    print(f"Week8 v72c frozen embedding error interface running at http://{args.host}:{args.port}/")
    print("Open locally via SSH tunnel / forwarded port: http://127.0.0.1:8532/")

    server.serve_forever()


if __name__ == "__main__":
    main()
'''

html = r'''<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Week8 v72c Frozen Embedding Error Analysis</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <header>
    <h1>Week8 v72c Frozen Embedding Error Analysis</h1>
    <p>Champion v72b frozen detector embedding predictions. GT is not modified here.</p>
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
  grid-template-columns: 380px 1fr;
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

  for (const r of CONF.slice(0, 30)) {
    const div = document.createElement("div");
    div.className = "rowlink";
    div.textContent = `${r.split_policy} | ${r.actual} → ${r.predicted}: ${r.error_count}`;
    div.onclick = () => setFilter(r.actual, r.predicted, r.split_policy);
    c.appendChild(div);
  }

  const cls = id("classes");
  cls.innerHTML = "";

  const sorted = [...CLS].sort((a, b) => Number(a.recall) - Number(b.recall)).slice(0, 30);

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
      r.selected_config_name
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
        <span class="badge">${r.selected_config_name}</span>
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

window.addEventListener("DOMContentLoaded", () => {
  for (const el of ["policyFilter", "actualFilter", "predFilter", "statusFilter", "searchBox"]) {
    id(el).addEventListener("input", renderGrid);
    id(el).addEventListener("change", renderGrid);
  }
});

loadAll();
'''

SERVER.write_text(server_code)
INDEX.write_text(html)
CSS.write_text(css)
JS.write_text(js)

manifest = {
    "version": "week8_v72c_frozen_embedding_error_analysis_interface",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "champion_prediction_rows": int(len(champion)),
    "error_rows": int(len(errors)),
    "correct_rows": int(len(correct)),
    "top_confusion_rows": int(len(conf)),
    "server_path": str(SERVER),
    "static_dir": str(STATIC),
    "default_port": 8532,
    "claim_boundary": "error analysis and interface only; no GT modification and no model training",
}
OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

OUT_README.write_text(
    "# Week8 v72c Frozen Embedding Error Analysis Interface\n\n"
    "This package contains champion prediction/error tables and a browser interface for visual inspection.\n\n"
    "It does not modify GT and does not train a model.\n\n"
    f"- Champion predictions: {len(champion)}\n"
    f"- Errors: {len(errors)}\n"
    f"- Correct: {len(correct)}\n"
    f"- Top confusions: {len(conf)}\n"
    f"- Server: {SERVER}\n"
    f"- Default port: 8532\n"
)

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))
    for p in [SERVER, INDEX, CSS, JS]:
        if p.exists():
            z.write(p, p.relative_to(W8))

zip_hash = sha256_file(OUT_ZIP)
OUT_SHA.write_text(f"{zip_hash}  {OUT_ZIP.name}\n")

decision = pd.DataFrame([{
    "v72c_decision": "frozen_embedding_error_analysis_interface_created" if hard_issue_count == 0 else "frozen_embedding_error_analysis_interface_has_blocking_issues",
    "champion_prediction_rows": int(len(champion)),
    "error_rows": int(len(errors)),
    "correct_rows": int(len(correct)),
    "current_errors": int((current_rows["correct"].astype(str).str.lower() != "true").sum()) if len(current_rows) else 0,
    "group_errors": int((group_rows["correct"].astype(str).str.lower() != "true").sum()) if len(group_rows) else 0,
    "top_confusion_rows": int(len(conf)),
    "class_summary_rows": int(len(class_summary)),
    "scan_summary_rows": int(len(scan_summary)),
    "video_summary_rows": int(len(video_summary)),
    "server_path": str(SERVER),
    "static_dir": str(STATIC),
    "default_port": 8532,
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_quality_failures": hard_quality_failures,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v73_final_modeling_package": bool(hard_issue_count == 0),
    "claim_scope": "frozen_embedding_error_analysis_interface_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v72c Frozen Embedding Error Analysis Interface\n\n"
    f"- v72c decision: {decision.iloc[0]['v72c_decision']}\n"
    f"- Champion prediction rows: {len(champion)}\n"
    f"- Error rows: {len(errors)}\n"
    f"- Correct rows: {len(correct)}\n"
    f"- Current errors: {decision.iloc[0]['current_errors']}\n"
    f"- Group-aware errors: {decision.iloc[0]['group_errors']}\n"
    f"- Top confusion rows: {len(conf)}\n"
    f"- Server: {SERVER}\n"
    f"- Default port: 8532\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Ready for v73 final modeling package: {bool(hard_issue_count == 0)}\n\n"
    "This interface is for visual error inspection only. It does not modify GT and does not train a model.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v72c Frozen Embedding Error Analysis Interface Report\n\n"
    f"Decision: {decision.iloc[0]['v72c_decision']}\n\n"
    f"Champion prediction rows: {len(champion)}\n\n"
    f"Errors: {len(errors)}\n\n"
    f"ZIP: {OUT_ZIP}\n\n"
    f"SHA256: {zip_hash}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v72c",
    "task_name": "Frozen embedding error analysis interface",
    "status": "PASS" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": str(V72B_PKG),
    "output_summary": str(PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Prepare v73 final modeling package." if hard_issue_count == 0 else "Fix v72c hard issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("=== v72c decision ===")
print(decision.to_string(index=False))

print("\n=== top confusions ===")
print(conf.head(20).to_string(index=False))

print("\n=== weakest classes ===")
print(class_summary.head(20).to_string(index=False))

print("\n=== issues ===")
print(issues_df.to_string(index=False) if len(issues_df) else "No issues found.")
