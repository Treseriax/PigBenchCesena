from pathlib import Path
from datetime import datetime
import csv
import json
import hashlib
import zipfile
import pandas as pd
import cv2
import numpy as np


ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

V76 = FULL / "outputs" / "v76_full_annotation_video_mapping_audit"
V76_PKG = V76 / "Full_Unibo_Annotation_Video_Mapping_Audit"
V76_VIDEOS = V76_PKG / "v76_all_videos_inventory.csv"

V78B = FULL / "outputs" / "v78b_corrected_video_mapping_from_v77c"
V78B_PKG = V78B / "Full_Unibo_Corrected_Video_Mapping_From_v77c"
V78B_DECISION = V78B / "v78b_decision_summary.csv"
V78B_UNRESOLVED = V78B_PKG / "v78b_unresolved_annotation_windows.csv"
V78B_CANDIDATES = V78B_PKG / "v78b_mapping_candidates_long.csv"

CONFIG = FULL / "config"
OUT = FULL / "outputs" / "v78c_camera_code_mapping_resolver"
PKG = OUT / "Full_Unibo_Camera_Code_Mapping_Resolver"
ATLAS = PKG / "visual_camera_code_atlas"
NOTES = FULL / "notes"
REPORTS = FULL / "reports"
PROGRESS = FULL / "progress"

for p in [OUT, PKG, ATLAS, NOTES, REPORTS, PROGRESS, CONFIG]:
    p.mkdir(parents=True, exist_ok=True)

OUT_GROUPS = PKG / "v78c_unresolved_mapping_groups.csv"
OUT_TEMPLATE = PKG / "v78c_clean_camera_code_mapping_template.csv"
OUT_CODE_INVENTORY = PKG / "v78c_encoded_video_code_inventory.csv"
OUT_ATLAS_INDEX = PKG / "v78c_visual_atlas_index.html"
OUT_ATLAS_CSV = PKG / "v78c_visual_atlas_manifest.csv"
OUT_QA = PKG / "v78c_quality_checks.csv"
OUT_README = PKG / "README_v78c_Camera_Code_Mapping_Resolver.md"
OUT_MANIFEST = PKG / "v78c_manifest.json"

OUT_DECISION = OUT / "v78c_decision_summary.csv"
OUT_ISSUES = OUT / "v78c_issues.csv"
OUT_ZIP = OUT / "Full_Unibo_Camera_Code_Mapping_Resolver.zip"
OUT_SHA = OUT / "Full_Unibo_Camera_Code_Mapping_Resolver.sha256"
OUT_NOTE = NOTES / "v78c_camera_code_mapping_resolver_notes.md"
OUT_REPORT = REPORTS / "v78c_camera_code_mapping_resolver_report.md"
OUT_PROGRESS = PROGRESS / "full_unibo_pipeline_progress_log.csv"

FINAL_CONFIG_TEMPLATE = CONFIG / "camera_code_mapping_TEMPLATE_TO_FILL.csv"


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


def split_semicolon(x):
    s = clean(x)
    if not s:
        return []
    return [a.strip() for a in s.split(";") if a.strip()]


def safe_name(x):
    s = clean(x)
    for ch in [" ", "/", "\\", ":", ";", "|", ","]:
        s = s.replace(ch, "_")
    return s


def extract_thumbnail(video_path, label, out_path, second=5.0, width=360):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "open_failed"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(second * fps))
    else:
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return False, "read_failed"

    h, w = frame.shape[:2]
    scale = width / max(1, w)
    new_h = int(h * scale)
    frame = cv2.resize(frame, (width, new_h))

    cv2.rectangle(frame, (0, 0), (width, 36), (0, 0, 0), -1)
    cv2.putText(frame, label[:48], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), frame)
    return True, "ok"


def make_contact_sheet(image_paths, labels, out_path, thumb_w=360):
    thumbs = []

    for p, label in zip(image_paths, labels):
        img = cv2.imread(str(p))
        if img is None:
            continue
        thumbs.append(img)

    if not thumbs:
        return False

    max_h = max(img.shape[0] for img in thumbs)
    padded = []

    for img in thumbs:
        h, w = img.shape[:2]
        if h < max_h:
            pad = np.zeros((max_h - h, w, 3), dtype=np.uint8)
            img = np.vstack([img, pad])
        padded.append(img)

    cols = 2
    rows = []

    for i in range(0, len(padded), cols):
        row_imgs = padded[i:i+cols]
        if len(row_imgs) < cols:
            blank = np.zeros_like(row_imgs[0])
            row_imgs.append(blank)
        rows.append(np.hstack(row_imgs))

    sheet = np.vstack(rows)
    cv2.imwrite(str(out_path), sheet)
    return True


issues = []

for p in [V76_VIDEOS, V78B_DECISION, V78B_UNRESOLVED, V78B_CANDIDATES]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "v78c requires v76 videos and v78b unresolved/candidate outputs.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    raise SystemExit("Missing required inputs.")

videos = read_csv_clean(V76_VIDEOS)
decision_b = read_csv_clean(V78B_DECISION)
unresolved = read_csv_clean(V78B_UNRESOLVED)
candidates = read_csv_clean(V78B_CANDIDATES)

if len(decision_b) == 0 or not bool_true(decision_b.iloc[0].get("ready_for_mapping_resolution", "")):
    issues.append({
        "item": "v78b_decision",
        "issue_type": "hard_v78b_not_ready_for_mapping_resolution",
        "issue_detail": "v78b must be ready for mapping resolution before v78c.",
        "severity": "hard",
    })

encoded = videos[videos["video_name_type"] == "encoded_camera_datetime"].copy()
if len(encoded):
    code_inventory = (
        encoded.groupby(["parsed_date", "parsed_camera_code"])
        .agg(
            video_count=("video_path", "nunique"),
            first_start=("parsed_start_hhmm", "min"),
            last_start=("parsed_start_hhmm", "max"),
            examples=("video_filename", lambda x: ";".join(list(x)[:20])),
        )
        .reset_index()
        .sort_values(["parsed_date", "parsed_camera_code"])
    )
else:
    code_inventory = pd.DataFrame(columns=["parsed_date", "parsed_camera_code", "video_count", "first_start", "last_start", "examples"])

safe_to_csv(code_inventory, OUT_CODE_INVENTORY)

# Clean unresolved groups only from actual unresolved rows.
if len(unresolved):
    unresolved_groups = (
        unresolved.groupby(["date", "camera", "pen"])
        .agg(
            unresolved_windows=("annotation_window_id", "nunique"),
            candidate_codes=("candidate_video_codes", lambda x: ";".join(sorted(set(";".join(x).split(";"))))),
            candidate_filenames=("candidate_video_filenames", lambda x: ";".join(sorted(set(";".join(x).split(";")))[:40])),
        )
        .reset_index()
        .sort_values(["date", "camera", "pen"])
    )
else:
    unresolved_groups = pd.DataFrame(columns=["date", "camera", "pen", "unresolved_windows", "candidate_codes", "candidate_filenames"])

safe_to_csv(unresolved_groups, OUT_GROUPS)

template_rows = []
atlas_rows = []

video_by_filename = {clean(r["video_filename"]): clean(r["video_path"]) for _, r in videos.iterrows()}

for _, g in unresolved_groups.iterrows():
    date = clean(g["date"])
    camera = clean(g["camera"])
    pen = clean(g["pen"])

    codes = split_semicolon(g["candidate_codes"])
    files = split_semicolon(g["candidate_filenames"])

    # Pick at most one representative file per camera code, preferably earliest filename.
    selected = []

    for code in codes:
        code_files = [f for f in files if f.startswith(code)]
        if not code_files:
            continue
        selected.append(sorted(code_files)[0])

    # If filename prefix does not exactly match parsed code string, fallback by v76 lookup.
    if not selected:
        for code in codes:
            subset = videos[
                (videos["parsed_camera_code"] == code)
                & (videos["parsed_date"] == date)
            ]
            if len(subset):
                selected.append(clean(subset.sort_values("parsed_start_hhmm").iloc[0]["video_filename"]))

    selected = selected[:12]

    thumb_paths = []
    labels = []

    group_dir = ATLAS / f"{safe_name(date)}__{safe_name(camera)}__{safe_name(pen)}"
    group_dir.mkdir(parents=True, exist_ok=True)

    for fname in selected:
        path = video_by_filename.get(fname, "")
        if not path:
            continue

        vrow = videos[videos["video_filename"] == fname]
        code = clean(vrow.iloc[0]["parsed_camera_code"]) if len(vrow) else ""
        start = clean(vrow.iloc[0]["parsed_start_hhmm"]) if len(vrow) else ""

        out_img = group_dir / f"{safe_name(code)}__{safe_name(start)}__{safe_name(fname)}.jpg"
        label = f"{code} | {start} | {fname}"

        ok, status = extract_thumbnail(path, label, out_img)

        atlas_rows.append({
            "date": date,
            "camera": camera,
            "pen": pen,
            "candidate_code": code,
            "video_filename": fname,
            "video_path": path,
            "thumbnail_path": str(out_img),
            "thumbnail_status": status,
        })

        if ok:
            thumb_paths.append(out_img)
            labels.append(label)

    contact_path = ATLAS / f"{safe_name(date)}__{safe_name(camera)}__{safe_name(pen)}__contact_sheet.jpg"
    contact_ok = make_contact_sheet(thumb_paths, labels, contact_path)

    template_rows.append({
        "date": date,
        "camera": camera,
        "pen": pen,
        "video_camera_code": "",
        "candidate_video_camera_codes": ";".join(codes),
        "unresolved_windows": int(g["unresolved_windows"]),
        "candidate_video_examples": ";".join(selected),
        "visual_contact_sheet": str(contact_path) if contact_ok else "",
        "manual_instruction": "Inspect contact sheet, choose one video_camera_code, then save filled table as Full_Unibo_Behaviour_Pipeline/config/camera_code_mapping.csv",
    })

template = pd.DataFrame(template_rows)
atlas_manifest = pd.DataFrame(atlas_rows)

safe_to_csv(template, OUT_TEMPLATE)
safe_to_csv(template, FINAL_CONFIG_TEMPLATE)
safe_to_csv(atlas_manifest, OUT_ATLAS_CSV)

# HTML atlas
html = [
    "<html><head><meta charset='utf-8'><title>v78c Camera Code Atlas</title></head><body>",
    "<h1>v78c Camera Code Mapping Visual Atlas</h1>",
    "<p>For each unresolved annotation group, inspect candidate encoded video thumbnails and fill video_camera_code in camera_code_mapping.csv.</p>",
]

for _, r in template.iterrows():
    html.append(f"<h2>{r['date']} | {r['camera']} | {r['pen']} | unresolved={r['unresolved_windows']}</h2>")
    html.append(f"<p>Candidate codes: {r['candidate_video_camera_codes']}</p>")
    cp = clean(r["visual_contact_sheet"])
    if cp:
        rel = Path(cp).relative_to(PKG)
        html.append(f"<img src='{rel}' style='max-width:1100px; border:1px solid #ccc;'>")
    else:
        html.append("<p>No contact sheet created.</p>")

html.append("</body></html>")
OUT_ATLAS_INDEX.write_text("\n".join(html))

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

unresolved_group_count = len(unresolved_groups)
template_count = len(template)
atlas_thumb_ok = int((atlas_manifest["thumbnail_status"] == "ok").sum()) if len(atlas_manifest) else 0
encoded_code_count = int(encoded["parsed_camera_code"].nunique()) if len(encoded) else 0

add_qa("unresolved_rows_exist", ">0", len(unresolved), len(unresolved) > 0, "hard", "There must be unresolved rows to resolve.")
add_qa("unresolved_groups_created", ">0", unresolved_group_count, unresolved_group_count > 0, "hard", "Unresolved groups should be created.")
add_qa("clean_mapping_template_created", ">0", template_count, template_count > 0, "hard", "Clean camera-code mapping template should be created.")
add_qa("encoded_camera_codes_found", ">0", encoded_code_count, encoded_code_count > 0, "hard", "Encoded camera codes should exist.")
add_qa("visual_thumbnails_created", ">0", atlas_thumb_ok, atlas_thumb_ok > 0, "warning", "Visual thumbnails should be created for manual review.")
add_qa("html_atlas_created", True, OUT_ATLAS_INDEX.exists(), OUT_ATLAS_INDEX.exists(), "hard", "HTML atlas should be created.")
add_qa("final_config_template_written", True, FINAL_CONFIG_TEMPLATE.exists(), FINAL_CONFIG_TEMPLATE.exists(), "hard", "A fillable config template should be written.")

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_QA)

hard_quality_failures = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())
warning_quality_failures = int(((qa["severity"] == "warning") & (~qa["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v78c_quality_checks",
        "issue_type": "hard_camera_code_resolver_failed",
        "issue_detail": f"{hard_quality_failures} hard QA checks failed.",
        "severity": "hard",
    })

if warning_quality_failures:
    issues.append({
        "item": "v78c_quality_checks",
        "issue_type": "warning_visual_atlas_incomplete",
        "issue_detail": f"{warning_quality_failures} warning QA checks failed.",
        "severity": "warning",
    })

issues.append({
    "item": "manual_mapping_required",
    "issue_type": "info_fill_camera_code_mapping_csv",
    "issue_detail": f"Fill {FINAL_CONFIG_TEMPLATE} and save/copy it as Full_Unibo_Behaviour_Pipeline/config/camera_code_mapping.csv, then rerun v78b.",
    "severity": "info",
})

issues.append({
    "item": "scope",
    "issue_type": "info_mapping_resolver_only",
    "issue_detail": "v78c creates a camera-code mapping resolver and visual atlas. It does not run tracking or model training.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum())
warning_count = int((issues_df["severity"] == "warning").sum()) if "warning" in set(issues_df["severity"]) else 0
info_count = int((issues_df["severity"] == "info").sum())

manifest = {
    "version": "v78c_camera_code_mapping_resolver",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "unresolved_rows": int(len(unresolved)),
    "unresolved_groups": int(unresolved_group_count),
    "template_rows": int(template_count),
    "encoded_camera_code_count": int(encoded_code_count),
    "visual_thumbnails_ok": int(atlas_thumb_ok),
    "html_atlas": str(OUT_ATLAS_INDEX),
    "config_template": str(FINAL_CONFIG_TEMPLATE),
    "hard_issue_count": int(hard_issue_count),
    "claim_boundary": "camera-code mapping resolver only; no tracking/model",
}

OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

readme = f"""# v78c Camera-Code Mapping Resolver

## Purpose

v78c resolves the bottleneck discovered in v78b: encoded video filenames use camera codes, while Excel sheets use TLC/camera/pen names.

This stage creates:
- unresolved mapping groups,
- clean camera-code mapping template,
- encoded video camera-code inventory,
- visual contact sheets for manual review.

## Main counts

- Unresolved annotation rows from v78b: {len(unresolved)}
- Unresolved groups: {unresolved_group_count}
- Template rows: {template_count}
- Encoded camera-code count: {encoded_code_count}
- Visual thumbnails created: {atlas_thumb_ok}
- Hard issues: {hard_issue_count}

## Manual next step

Inspect:

`{OUT_ATLAS_INDEX}`

Fill:

`{FINAL_CONFIG_TEMPLATE}`

Then save/copy the filled file as:

`Full_Unibo_Behaviour_Pipeline/config/camera_code_mapping.csv`

and rerun v78b.
"""

OUT_README.write_text(readme)
OUT_REPORT.write_text(readme)

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

zip_hash = sha256_file(OUT_ZIP)
OUT_SHA.write_text(f"{zip_hash}  {OUT_ZIP.name}\n")

decision = pd.DataFrame([{
    "v78c_decision": "camera_code_mapping_resolver_completed" if hard_issue_count == 0 else "camera_code_mapping_resolver_has_blocking_issues",
    "unresolved_rows_from_v78b": int(len(unresolved)),
    "unresolved_mapping_groups": int(unresolved_group_count),
    "camera_code_mapping_template_rows": int(template_count),
    "encoded_camera_code_count": int(encoded_code_count),
    "visual_thumbnails_created": int(atlas_thumb_ok),
    "html_atlas_path": str(OUT_ATLAS_INDEX),
    "config_template_path": str(FINAL_CONFIG_TEMPLATE),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_quality_failures": int(hard_quality_failures),
    "warning_quality_failures": int(warning_quality_failures),
    "hard_issue_count": int(hard_issue_count),
    "warning_count": int(warning_count),
    "info_count": int(info_count),
    "issue_count": int(len(issues_df)),
    "ready_for_manual_camera_code_mapping": bool(hard_issue_count == 0),
    "ready_for_v78b_rerun_after_mapping": False,
    "claim_scope": "camera_code_mapping_resolver_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# v78c Camera-Code Mapping Resolver\n\n"
    f"- v78c decision: {decision.iloc[0]['v78c_decision']}\n"
    f"- Unresolved rows from v78b: {len(unresolved)}\n"
    f"- Unresolved mapping groups: {unresolved_group_count}\n"
    f"- Camera-code mapping template rows: {template_count}\n"
    f"- Encoded camera-code count: {encoded_code_count}\n"
    f"- Visual thumbnails created: {atlas_thumb_ok}\n"
    f"- HTML atlas path: {OUT_ATLAS_INDEX}\n"
    f"- Config template path: {FINAL_CONFIG_TEMPLATE}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Ready for manual camera-code mapping: {bool(hard_issue_count == 0)}\n\n"
    "Next: inspect the visual atlas, fill video_camera_code values, save as config/camera_code_mapping.csv, then rerun v78b.\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v78c",
    "task_name": "Camera-code mapping resolver",
    "status": "PASS_MANUAL_MAPPING_REQUIRED" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": str(V78B_UNRESOLVED),
    "output_summary": str(PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Fill camera_code_mapping.csv and rerun v78b." if hard_issue_count == 0 else "Fix v78c hard issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("=== v78c decision ===")
print(decision.to_string(index=False))

print("\n=== clean camera-code mapping template ===")
print(template.to_string(index=False))

print("\n=== encoded video code inventory ===")
print(code_inventory.to_string(index=False))

print("\n=== QA ===")
print(qa.to_string(index=False))

print("\n=== issues ===")
print(issues_df.to_string(index=False))
