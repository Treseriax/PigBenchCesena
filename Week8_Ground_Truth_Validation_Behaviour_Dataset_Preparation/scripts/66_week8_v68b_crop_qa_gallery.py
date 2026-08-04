from pathlib import Path
from datetime import datetime
import csv
import json
import hashlib
import zipfile
import shutil
import pandas as pd
import numpy as np


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V68A = W8 / "outputs" / "v68a_strict_gold_anchor_crop_materialization"
DATASET = V68A / "Week8_StrictGold_AnchorFrame_Crop_Dataset"

METADATA = DATASET / "metadata" / "week8_v68a_strict_gold_anchor_crop_metadata.csv"
FAILED = DATASET / "audit" / "week8_v68a_failed_crop_rows.csv"
V68A_DECISION = V68A / "week8_v68a_decision_summary.csv"

OUT = W8 / "outputs" / "v68b_crop_qa_gallery"
GALLERY = OUT / "Week8_StrictGold_Crop_QA_Gallery"
THUMBS = GALLERY / "thumbs"
AUDIT = GALLERY / "audit"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, GALLERY, THUMBS, AUDIT, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_HTML = GALLERY / "index.html"
OUT_CSS = GALLERY / "style.css"
OUT_JS = GALLERY / "app.js"
OUT_DATA_JSON = GALLERY / "week8_v68b_gallery_data.json"

OUT_IMAGE_AUDIT = AUDIT / "week8_v68b_image_integrity_audit.csv"
OUT_QA = AUDIT / "week8_v68b_crop_qa_quality_checks.csv"
OUT_BEHAVIOUR = AUDIT / "week8_v68b_gallery_behaviour_distribution.csv"
OUT_SPLIT = AUDIT / "week8_v68b_gallery_split_distribution.csv"
OUT_RARE = AUDIT / "week8_v68b_rare_class_warnings.csv"

OUT_DECISION = OUT / "week8_v68b_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v68b_issues.csv"
OUT_ZIP = OUT / "Week8_StrictGold_Crop_QA_Gallery.zip"
OUT_SHA256 = OUT / "Week8_StrictGold_Crop_QA_Gallery.sha256"
OUT_NOTE = NOTES / "week8_v68b_crop_qa_gallery_notes.md"
OUT_REPORT = REPORTS / "week8_v68b_crop_qa_gallery_report.md"
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


def safe_name(s):
    import re
    s = clean(s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def rel_to_gallery(path):
    return str(Path(path).relative_to(GALLERY)).replace("\\", "/")


issues = []

for p in [METADATA, FAILED, V68A_DECISION]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required v68a output missing for crop QA gallery.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v68b_decision": "crop_qa_gallery_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v68c_split_leakage_audit": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


v68a_decision = read_csv_clean(V68A_DECISION)
if len(v68a_decision) == 0 or not bool_true(v68a_decision.iloc[0].get("ready_for_v68b_crop_qa_gallery", "")):
    issues.append({
        "item": str(V68A_DECISION),
        "issue_type": "hard_v68a_not_ready",
        "issue_detail": "v68a must be ready before v68b crop QA gallery.",
        "severity": "hard",
    })

metadata = read_csv_clean(METADATA)
failed = read_csv_clean(FAILED)

try:
    import cv2
except Exception as e:
    issues.append({
        "item": "opencv",
        "issue_type": "hard_opencv_import_failed",
        "issue_detail": str(e),
        "severity": "hard",
    })

if any(i["severity"] == "hard" for i in issues):
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v68b_decision": "crop_qa_gallery_blocked",
        "metadata_rows": int(len(metadata)),
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v68c_split_leakage_audit": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    print(issues_df.to_string(index=False))
    raise SystemExit(1)


def make_thumb(src_path, dst_path, label_top, label_bottom, thumb_w=220, thumb_h=220):
    img = cv2.imread(str(src_path))
    if img is None:
        return False, "image_read_failed", 0, 0

    h, w = img.shape[:2]
    canvas = np.full((thumb_h, thumb_w, 3), 245, dtype=np.uint8)

    image_h = thumb_h - 46
    scale = min(thumb_w / max(w, 1), image_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x0 = (thumb_w - new_w) // 2
    y0 = 4
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

    cv2.putText(canvas, label_top[:28], (6, thumb_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, label_bottom[:30], (6, thumb_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1, cv2.LINE_AA)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    return bool(ok), "ok" if ok else "thumb_write_failed", w, h


image_audit_rows = []
gallery_rows = []

for _, r in metadata.iterrows():
    obj_id = clean(r["canonical_gt_object_id"])
    behaviour = clean(r["behaviour_code"])
    split = clean(r["recommended_split"])
    colour = clean(r["canonical_colour_label_norm"])
    scan = clean(r["scan_frame_id"])

    tight_path = Path(clean(r["tight_crop_path"]))
    context_path = Path(clean(r["context10_crop_path"]))
    anchor_path = Path(clean(r["anchor_frame_path"]))

    tight_thumb = THUMBS / "tight" / safe_name(split) / safe_name(behaviour) / f"{safe_name(obj_id)}.jpg"
    context_thumb = THUMBS / "context10" / safe_name(split) / safe_name(behaviour) / f"{safe_name(obj_id)}.jpg"

    label_top = f"{behaviour} | {colour}"
    label_bottom = f"{split} | {scan}"

    tight_ok, tight_status, tight_w, tight_h = make_thumb(tight_path, tight_thumb, label_top, label_bottom)
    context_ok, context_status, context_w, context_h = make_thumb(context_path, context_thumb, label_top, label_bottom)

    anchor_exists = anchor_path.exists()

    image_audit_rows.append({
        "canonical_gt_object_id": obj_id,
        "scan_frame_id": scan,
        "behaviour_code": behaviour,
        "recommended_split": split,
        "tight_crop_path": str(tight_path),
        "tight_crop_exists": tight_path.exists(),
        "tight_crop_read_ok": tight_ok,
        "tight_crop_status": tight_status,
        "tight_crop_width": tight_w,
        "tight_crop_height": tight_h,
        "context10_crop_path": str(context_path),
        "context10_crop_exists": context_path.exists(),
        "context10_crop_read_ok": context_ok,
        "context10_crop_status": context_status,
        "context10_crop_width": context_w,
        "context10_crop_height": context_h,
        "anchor_frame_path": str(anchor_path),
        "anchor_frame_exists": anchor_exists,
        "thumb_tight_path": str(tight_thumb),
        "thumb_context10_path": str(context_thumb),
    })

    gallery_rows.append({
        "canonical_gt_object_id": obj_id,
        "scan_frame_id": scan,
        "video_id": clean(r["video_id"]),
        "behaviour_code": behaviour,
        "canonical_colour_label_norm": colour,
        "recommended_split": split,
        "tight_thumb": rel_to_gallery(tight_thumb),
        "context10_thumb": rel_to_gallery(context_thumb),
        "tight_crop_path": str(tight_path),
        "context10_crop_path": str(context_path),
        "anchor_frame_idx": clean(r.get("anchor_frame_idx", "")),
        "manual_assigned_candidate_box_id": clean(r.get("manual_assigned_candidate_box_id", "")),
        "claim_boundary": "strict_gold_anchor_frame_crop_for_baseline_only",
    })

image_audit = pd.DataFrame(image_audit_rows)
gallery_df = pd.DataFrame(gallery_rows)

safe_to_csv(image_audit, OUT_IMAGE_AUDIT)

behaviour_dist = (
    gallery_df.groupby("behaviour_code")
    .size()
    .reset_index(name="gallery_count")
    .sort_values("gallery_count", ascending=False)
)

split_dist = (
    gallery_df.groupby(["recommended_split", "behaviour_code"])
    .size()
    .reset_index(name="gallery_count")
    .sort_values(["recommended_split", "behaviour_code"])
)

rare = behaviour_dist[behaviour_dist["gallery_count"] < 10].copy()
rare["warning"] = "rare class; do not claim reliable standalone metrics"

safe_to_csv(behaviour_dist, OUT_BEHAVIOUR)
safe_to_csv(split_dist, OUT_SPLIT)
safe_to_csv(rare, OUT_RARE)

gallery_json = {
    "version": "week8_v68b_crop_qa_gallery",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "summary": {
        "metadata_rows": int(len(metadata)),
        "gallery_rows": int(len(gallery_df)),
        "behaviour_classes": int(gallery_df["behaviour_code"].nunique()) if len(gallery_df) else 0,
        "scanframes": int(gallery_df["scan_frame_id"].nunique()) if len(gallery_df) else 0,
        "tight_thumbnails": int(image_audit["tight_crop_read_ok"].sum()) if len(image_audit) else 0,
        "context10_thumbnails": int(image_audit["context10_crop_read_ok"].sum()) if len(image_audit) else 0,
    },
    "rows": gallery_rows,
}

OUT_DATA_JSON.write_text(json.dumps(gallery_json, indent=2, ensure_ascii=False))

html = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Week8 Strict-Gold Crop QA Gallery</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <header>
    <h1>Week8 Strict-Gold Crop QA Gallery</h1>
    <p>Only strict gold anchor-frame crops. Tracking is not used for labels.</p>
  </header>

  <section id="controls">
    <label>Split</label>
    <select id="splitFilter">
      <option value="">All</option>
    </select>

    <label>Behaviour</label>
    <select id="behaviourFilter">
      <option value="">All</option>
    </select>

    <label>Search</label>
    <input id="searchBox" placeholder="scanframe_0005 or object id">

    <label>
      <input type="checkbox" id="contextOnly">
      show context crop first
    </label>
  </section>

  <section id="summary"></section>
  <section id="grid"></section>

  <script src="app.js"></script>
</body>
</html>
"""

css = """
body {
  margin: 0;
  background: #111;
  color: #eee;
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
  color: #bbb;
}

#controls {
  display: flex;
  gap: 10px;
  align-items: center;
  padding: 12px 16px;
  background: #181818;
  position: sticky;
  top: 0;
  z-index: 10;
  border-bottom: 1px solid #333;
}

select, input {
  background: #222;
  color: #eee;
  border: 1px solid #555;
  padding: 7px;
  border-radius: 4px;
}

#summary {
  padding: 12px 16px;
  color: #ccc;
  white-space: pre-wrap;
  font-size: 13px;
}

#grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(480px, 1fr));
  gap: 12px;
  padding: 16px;
}

.card {
  background: #1b1b1b;
  border: 1px solid #333;
  border-radius: 8px;
  padding: 10px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 8px;
}

.behaviour {
  font-weight: bold;
}

.meta {
  color: #bbb;
  font-size: 12px;
  line-height: 1.4;
}

.images {
  display: flex;
  gap: 8px;
  margin-top: 8px;
}

.images img {
  width: 220px;
  height: 220px;
  object-fit: contain;
  background: #eee;
  border-radius: 4px;
}

.badge {
  background: #333;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 12px;
}
"""

js = """
let DATA = null;

function byId(id) {
  return document.getElementById(id);
}

async function loadData() {
  const resp = await fetch("week8_v68b_gallery_data.json");
  DATA = await resp.json();
  populateFilters();
  render();
}

function populateFilters() {
  const splits = new Set();
  const behaviours = new Set();

  for (const r of DATA.rows) {
    splits.add(r.recommended_split);
    behaviours.add(r.behaviour_code);
  }

  for (const s of [...splits].sort()) {
    const opt = document.createElement("option");
    opt.value = s;
    opt.textContent = s;
    byId("splitFilter").appendChild(opt);
  }

  for (const b of [...behaviours].sort()) {
    const opt = document.createElement("option");
    opt.value = b;
    opt.textContent = b;
    byId("behaviourFilter").appendChild(opt);
  }
}

function matches(r) {
  const split = byId("splitFilter").value;
  const behaviour = byId("behaviourFilter").value;
  const q = byId("searchBox").value.trim().toLowerCase();

  if (split && r.recommended_split !== split) return false;
  if (behaviour && r.behaviour_code !== behaviour) return false;

  if (q) {
    const hay = [
      r.canonical_gt_object_id,
      r.scan_frame_id,
      r.video_id,
      r.behaviour_code,
      r.canonical_colour_label_norm
    ].join(" ").toLowerCase();
    if (!hay.includes(q)) return false;
  }

  return true;
}

function render() {
  const grid = byId("grid");
  grid.innerHTML = "";

  const rows = DATA.rows.filter(matches);
  const contextFirst = byId("contextOnly").checked;

  const counts = {};
  for (const r of rows) {
    counts[r.behaviour_code] = (counts[r.behaviour_code] || 0) + 1;
  }

  byId("summary").textContent =
    `shown: ${rows.length} / ${DATA.rows.length}\\n` +
    Object.entries(counts).sort((a,b) => b[1] - a[1]).map(([k,v]) => `${k}: ${v}`).join("\\n");

  for (const r of rows) {
    const card = document.createElement("div");
    card.className = "card";

    const first = contextFirst ? r.context10_thumb : r.tight_thumb;
    const second = contextFirst ? r.tight_thumb : r.context10_thumb;
    const firstLabel = contextFirst ? "context10" : "tight";
    const secondLabel = contextFirst ? "tight" : "context10";

    card.innerHTML = `
      <div class="card-header">
        <div>
          <span class="behaviour">${r.behaviour_code}</span>
          <span class="badge">${r.recommended_split}</span>
          <span class="badge">${r.canonical_colour_label_norm}</span>
        </div>
        <div class="meta">${r.scan_frame_id}</div>
      </div>

      <div class="meta">
        object: ${r.canonical_gt_object_id}<br>
        video: ${r.video_id}<br>
        anchor frame: ${r.anchor_frame_idx}<br>
        candidate/manual bbox id: ${r.manual_assigned_candidate_box_id}
      </div>

      <div class="images">
        <div>
          <div class="meta">${firstLabel}</div>
          <img src="${first}">
        </div>
        <div>
          <div class="meta">${secondLabel}</div>
          <img src="${second}">
        </div>
      </div>
    `;

    grid.appendChild(card);
  }
}

byId("splitFilter").onchange = render;
byId("behaviourFilter").onchange = render;
byId("searchBox").oninput = render;
byId("contextOnly").onchange = render;

loadData();
"""

OUT_HTML.write_text(html)
OUT_CSS.write_text(css)
OUT_JS.write_text(js)

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

add_qa("metadata_rows", 372, len(metadata), len(metadata) == 372, "hard", "v68a metadata should contain 372 strict crop rows.")
add_qa("failed_v68a_rows", 0, len(failed), len(failed) == 0, "hard", "v68a should have zero failed crop rows.")
add_qa("gallery_rows", 372, len(gallery_df), len(gallery_df) == 372, "hard", "Gallery should contain one row per crop.")
add_qa("tight_thumbnails_created", 372, int(image_audit["tight_crop_read_ok"].sum()), int(image_audit["tight_crop_read_ok"].sum()) == 372, "hard", "All tight thumbnails should be readable and created.")
add_qa("context10_thumbnails_created", 372, int(image_audit["context10_crop_read_ok"].sum()), int(image_audit["context10_crop_read_ok"].sum()) == 372, "hard", "All context thumbnails should be readable and created.")
add_qa("behaviour_classes", 11, gallery_df["behaviour_code"].nunique(), gallery_df["behaviour_code"].nunique() == 11, "hard", "Gallery should preserve all 11 behaviour classes.")
add_qa("scanframes", 70, gallery_df["scan_frame_id"].nunique(), gallery_df["scan_frame_id"].nunique() == 70, "hard", "Gallery should cover 70 strict-gold scanframes.")
add_qa("html_exists", True, OUT_HTML.exists(), OUT_HTML.exists(), "hard", "Gallery index.html should exist.")
add_qa("rare_class_warnings_documented", True, len(rare) >= 1, len(rare) >= 1, "info", "Rare classes should be documented as warnings, not hard failures.")

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_QA)

hard_quality_failures = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())
if hard_quality_failures:
    issues.append({
        "item": "v68b_quality_checks",
        "issue_type": "hard_crop_qa_gallery_quality_check_failed",
        "issue_detail": f"{hard_quality_failures} hard QA checks failed.",
        "severity": "hard",
    })

if len(rare):
    for _, r in rare.iterrows():
        issues.append({
            "item": r["behaviour_code"],
            "issue_type": "info_rare_class_low_crop_count",
            "issue_detail": f"Only {r['gallery_count']} crops in strict-gold crop dataset.",
            "severity": "info",
        })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

manifest = {
    "gallery_version": "week8_v68b_crop_qa_gallery",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "source_dataset": str(DATASET),
    "gallery_rows": int(len(gallery_df)),
    "behaviour_classes": int(gallery_df["behaviour_code"].nunique()) if len(gallery_df) else 0,
    "scanframes": int(gallery_df["scan_frame_id"].nunique()) if len(gallery_df) else 0,
    "tight_thumbnails": int(image_audit["tight_crop_read_ok"].sum()) if len(image_audit) else 0,
    "context10_thumbnails": int(image_audit["context10_crop_read_ok"].sum()) if len(image_audit) else 0,
    "tracking_used": False,
    "usage": "visual QA before leakage audit and baseline classification",
    "files": {
        "index_html": str(OUT_HTML),
        "gallery_json": str(OUT_DATA_JSON),
        "image_integrity_audit": str(OUT_IMAGE_AUDIT),
        "quality_checks": str(OUT_QA),
    },
}

(GALLERY / "week8_v68b_gallery_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

readme = f"""# Week8 Strict-Gold Crop QA Gallery

Open index.html in a browser, or serve this folder with:

python -m http.server 8530

This gallery contains only strict gold anchor-frame crops.

Key numbers:

- Gallery rows: {len(gallery_df)}
- Behaviour classes: {gallery_df["behaviour_code"].nunique() if len(gallery_df) else 0}
- Scanframes: {gallery_df["scan_frame_id"].nunique() if len(gallery_df) else 0}
- Tight thumbnails: {int(image_audit["tight_crop_read_ok"].sum()) if len(image_audit) else 0}
- Context10 thumbnails: {int(image_audit["context10_crop_read_ok"].sum()) if len(image_audit) else 0}

Manual GT v2 is the source of truth. Tracking is not used for crop labels.
"""

(GALLERY / "README_Week8_StrictGold_Crop_QA_Gallery.md").write_text(readme)

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(GALLERY.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

zip_hash = sha256_file(OUT_ZIP)
OUT_SHA256.write_text(f"{zip_hash}  {OUT_ZIP.name}\n")

decision = pd.DataFrame([{
    "v68b_decision": "crop_qa_gallery_completed" if hard_issue_count == 0 else "crop_qa_gallery_has_blocking_issues",
    "gallery_rows": int(len(gallery_df)),
    "behaviour_classes": int(gallery_df["behaviour_code"].nunique()) if len(gallery_df) else 0,
    "scanframes": int(gallery_df["scan_frame_id"].nunique()) if len(gallery_df) else 0,
    "tight_thumbnails": int(image_audit["tight_crop_read_ok"].sum()) if len(image_audit) else 0,
    "context10_thumbnails": int(image_audit["context10_crop_read_ok"].sum()) if len(image_audit) else 0,
    "rare_class_count": int(len(rare)),
    "gallery_dir": str(GALLERY),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_quality_failures": hard_quality_failures,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v68c_split_leakage_audit": bool(hard_issue_count == 0),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v68b Crop QA Gallery\n\n"
    f"- v68b decision: {decision.iloc[0]['v68b_decision']}\n"
    f"- Gallery rows: {len(gallery_df)}\n"
    f"- Behaviour classes: {gallery_df['behaviour_code'].nunique() if len(gallery_df) else 0}\n"
    f"- Scanframes: {gallery_df['scan_frame_id'].nunique() if len(gallery_df) else 0}\n"
    f"- Tight thumbnails: {int(image_audit['tight_crop_read_ok'].sum()) if len(image_audit) else 0}\n"
    f"- Context10 thumbnails: {int(image_audit['context10_crop_read_ok'].sum()) if len(image_audit) else 0}\n"
    f"- Rare class warnings: {len(rare)}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Gallery: {GALLERY}\n"
    f"- ZIP: {OUT_ZIP}\n"
    f"- SHA256: {zip_hash}\n"
    f"- Ready for v68c split leakage audit: {bool(hard_issue_count == 0)}\n\n"
    "This is visual QA for strict-gold crops only. Tracking is not used for crop labels.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v68b Crop QA Gallery Report\n\n"
    f"Decision: {decision.iloc[0]['v68b_decision']}\n\n"
    f"Gallery directory: {GALLERY}\n\n"
    f"ZIP: {OUT_ZIP}\n\n"
    f"SHA256: {zip_hash}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v68b",
    "task_name": "Strict-gold crop QA gallery",
    "status": "PASS" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": str(METADATA),
    "output_summary": str(GALLERY),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Run v68c split leakage audit." if hard_issue_count == 0 else "Fix crop QA gallery issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_HTML)
print(OUT_DATA_JSON)
print(OUT_IMAGE_AUDIT)
print(OUT_QA)
print(OUT_BEHAVIOUR)
print(OUT_SPLIT)
print(OUT_RARE)
print(OUT_ZIP)
print(OUT_SHA256)
print(OUT_DECISION)
print(OUT_NOTE)

print()
print("=== v68b decision ===")
print(decision.to_string(index=False))

print()
print("=== QA ===")
print(qa.to_string(index=False))

print()
print("=== rare classes ===")
if len(rare):
    print(rare.to_string(index=False))
else:
    print("No rare class warnings.")

print()
print("=== issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
