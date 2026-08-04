from pathlib import Path
from datetime import datetime
import csv
import json
import hashlib
import zipfile
import pandas as pd


ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

V78C_TEMPLATE = FULL / "config" / "camera_code_mapping_TEMPLATE_TO_FILL.csv"
V78I_PKG = FULL / "outputs" / "v78i_visual_stream_grouping_audit" / "Full_Unibo_Visual_Stream_Grouping_Audit"
V78I_OBS = V78I_PKG / "v78i_human_visual_observations.csv"
V78I_FEATURES = V78I_PKG / "v78i_frame_features.csv"

OUT = FULL / "outputs" / "v78j_time_aware_visual_stream_resolver"
PKG = OUT / "Full_Unibo_Time_Aware_Visual_Stream_Resolver"
NOTES = FULL / "notes"
REPORTS = FULL / "reports"
PROGRESS = FULL / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_GROUPS = PKG / "v78j_visual_stream_group_definitions.csv"
OUT_TARGETS = PKG / "v78j_excel_visual_mapping_targets.csv"
OUT_TEMPLATE = PKG / "v78j_MANUAL_FILL_visual_stream_resolution.csv"
OUT_LOCKED = PKG / "v78j_locked_anchor_rows.csv"
OUT_HTML = PKG / "v78j_visual_stream_resolution_board.html"
OUT_QA = PKG / "v78j_quality_checks.csv"
OUT_README = PKG / "README_v78j_Time_Aware_Visual_Stream_Resolver.md"
OUT_MANIFEST = PKG / "v78j_manifest.json"

OUT_DECISION = OUT / "v78j_decision_summary.csv"
OUT_ISSUES = OUT / "v78j_issues.csv"
OUT_ZIP = OUT / "Full_Unibo_Time_Aware_Visual_Stream_Resolver.zip"
OUT_SHA = OUT / "Full_Unibo_Time_Aware_Visual_Stream_Resolver.sha256"
OUT_NOTE = NOTES / "v78j_time_aware_visual_stream_resolver_notes.md"
OUT_REPORT = REPORTS / "v78j_time_aware_visual_stream_resolver_report.md"
OUT_PROGRESS = PROGRESS / "full_unibo_pipeline_progress_log.csv"


def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path).fillna("")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(clean)
    return df


issues = []

if not V78C_TEMPLATE.exists():
    issues.append({
        "item": str(V78C_TEMPLATE),
        "issue_type": "hard_missing_v78c_template",
        "issue_detail": "Need v78c camera_code_mapping_TEMPLATE_TO_FILL.csv.",
        "severity": "hard",
    })

if not V78I_OBS.exists():
    issues.append({
        "item": str(V78I_OBS),
        "issue_type": "hard_missing_v78i_observations",
        "issue_detail": "Need v78i human observations.",
        "severity": "hard",
    })

if issues:
    issues_df = pd.DataFrame(issues)
    to_csv(issues_df, OUT_ISSUES)
    raise SystemExit("Missing required inputs for v78j.")

template = read_csv(V78C_TEMPLATE)
obs = read_csv(V78I_OBS)
features = read_csv(V78I_FEATURES)

# 1) Official stream groups from human review.
groups = pd.DataFrame([
    {
        "visual_group_id": "GROUP_A",
        "relation": "side_or_duplicate_view",
        "camera_codes": "c0000;c0002",
        "primary_code": "c0002",
        "secondary_code": "c0000",
        "top_view_code": "c0100",
        "anchor_status": "contains_locked_TLC1_anchor",
        "evidence_source": "manual_visual_review_v78i",
        "confidence": "high_for_c0000_c0002_same_view; medium_high_for_c0100_top_view",
        "notes": "User confirmed TLC1 anchor matches c0002 and observed c0000 visually same as c0002; c0100 appears top-view of this group.",
    },
    {
        "visual_group_id": "GROUP_B",
        "relation": "side_or_duplicate_view",
        "camera_codes": "c0001;c0003",
        "primary_code": "c0001",
        "secondary_code": "c0003",
        "top_view_code": "c0101",
        "anchor_status": "no_named_TLC_anchor",
        "evidence_source": "manual_visual_review_v78i",
        "confidence": "medium_high_for_c0001_c0003_same_view; medium_high_for_c0101_top_view",
        "notes": "User observed c0001 and c0003 visually similar/same; c0101 appears top-view of this group.",
    },
])
to_csv(groups, OUT_GROUPS)

# 2) Build mapping targets from v78c unresolved template.
required_cols = ["date", "camera", "pen", "unresolved_windows"]
for c in required_cols:
    if c not in template.columns:
        template[c] = ""

targets = (
    template.groupby(["date", "camera", "pen"], dropna=False)
    .agg(
        unresolved_windows=("unresolved_windows", lambda x: sum(int(float(v)) if clean(v) else 0 for v in x)),
        candidate_video_camera_codes=("candidate_video_camera_codes", lambda x: ";".join(sorted(set(";".join(map(str, x)).split(";"))))),
        visual_contact_sheet=("visual_contact_sheet", lambda x: ";".join(sorted(set(map(str, x)))[:3])),
    )
    .reset_index()
    .sort_values(["date", "camera", "pen"])
)

# Keep only rows with available candidate c-codes for now.
targets = targets[targets["candidate_video_camera_codes"].astype(str).str.contains("c000|c010", regex=True, na=False)].copy()
targets["target_id"] = ["target_%03d" % (i + 1) for i in range(len(targets))]
targets = targets[["target_id", "date", "camera", "pen", "unresolved_windows", "candidate_video_camera_codes", "visual_contact_sheet"]]
to_csv(targets, OUT_TARGETS)

# 3) Manual fill table.
rows = []
locked_rows = []

for _, r in targets.iterrows():
    camera = clean(r["camera"])
    pen = clean(r["pen"])

    base = {
        "target_id": r["target_id"],
        "date": r["date"],
        "camera": camera,
        "pen": pen,
        "unresolved_windows": r["unresolved_windows"],
        "candidate_video_camera_codes": r["candidate_video_camera_codes"],
        "selected_visual_group_id": "",
        "selected_view_type": "",
        "primary_video_camera_code": "",
        "secondary_video_camera_code": "",
        "top_view_camera_code": "",
        "resolution_status": "needs_manual_visual_resolution",
        "confidence": "",
        "evidence_type": "",
        "reviewer_note": "",
        "allowed_values_visual_group": "GROUP_A;GROUP_B;NOT_VISIBLE;UNRESOLVED",
        "allowed_values_view_type": "side;top;side_and_top;not_visible;unresolved",
    }

    # Known high-confidence anchor: TLC1 rows belong to visual group A because c0002 is locked anchor.
    if camera == "TLC1":
        base.update({
            "selected_visual_group_id": "GROUP_A",
            "selected_view_type": "side_and_top",
            "primary_video_camera_code": "c0002",
            "secondary_video_camera_code": "c0000",
            "top_view_camera_code": "c0100",
            "resolution_status": "locked_from_TLC1_c0002_visual_anchor",
            "confidence": "high",
            "evidence_type": "named_friendly_TLC1_reference_plus_manual_visual_confirmation",
            "reviewer_note": "TLC1/c0002 locked. c0000 duplicate and c0100 top-view kept as auxiliary views.",
        })
        locked_rows.append(base.copy())

    rows.append(base)

manual = pd.DataFrame(rows)
to_csv(manual, OUT_TEMPLATE)

locked_df = pd.DataFrame(locked_rows)
to_csv(locked_df, OUT_LOCKED)

# 4) Build HTML board with instructions and frame links from v78i.
html = []
html.append("""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>v78j Time-Aware Visual Stream Resolver</title>
<style>
body { font-family: Arial, sans-serif; background:#f5f5f5; margin:0; }
.header { background:#111; color:white; padding:18px 24px; position:sticky; top:0; z-index:10; }
.container { padding:24px; }
.section { background:white; padding:18px; margin-bottom:24px; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,0.08); }
.warn { background:#fff3cd; border:1px solid #d6b44c; padding:12px; border-radius:8px; }
.ok { background:#e9ffe9; border:1px solid #60b060; padding:12px; border-radius:8px; }
table { border-collapse: collapse; width:100%; font-size:13px; }
th, td { border:1px solid #ccc; padding:5px; vertical-align:top; }
th { background:#eee; }
.grid { display:flex; flex-wrap:wrap; gap:12px; }
.card { width:520px; background:#fafafa; border:1px solid #ccc; border-radius:8px; padding:8px; }
.card img { width:100%; border-radius:4px; }
.small { font-size:12px; color:#555; }
</style>
</head>
<body>
<div class="header">
<h1>v78j Time-Aware Visual Stream Resolver</h1>
<p>Do not force TLC→single c-code mapping. Resolve each Excel target into a visual stream group or mark unresolved/not visible.</p>
</div>
<div class="container">
""")

html.append("<div class='section warn'><h2>Core rule</h2>")
html.append("<p><b>Fixed one-to-one TLC→c-code mapping is invalid or unproven.</b> Use visual groups:</p>")
html.append("<ul><li>GROUP_A: c0000/c0002 side or duplicate view, c0100 top view. TLC1/c0002 anchor locked.</li>")
html.append("<li>GROUP_B: c0001/c0003 side or duplicate view, c0101 top view.</li></ul>")
html.append("</div>")

html.append("<div class='section'><h2>Visual group definitions</h2>")
html.append(groups.to_html(index=False, escape=True))
html.append("</div>")

html.append("<div class='section'><h2>Manual fill targets</h2>")
html.append("<p>Fill this CSV on the server:</p>")
html.append(f"<code>{OUT_TEMPLATE}</code>")
html.append("<p>For each target choose GROUP_A, GROUP_B, NOT_VISIBLE, or UNRESOLVED. TLC1 is already locked.</p>")
html.append(manual.to_html(index=False, escape=True))
html.append("</div>")

html.append("<div class='section'><h2>Frame evidence from v78i</h2>")
if not features.empty and "frame_image" in features.columns:
    html.append("<div class='grid'>")
    for _, r in features.iterrows():
        img = clean(r.get("frame_image", ""))
        if not img:
            continue
        p = Path(img)
        rel = "../v78i_visual_stream_grouping_audit/Full_Unibo_Visual_Stream_Grouping_Audit/visual_stream_groups/" + p.name
        title = f"{r.get('video_type','')} | {r.get('camera_code','')} {r.get('tlc_camera','')} | {r.get('start_hhmm','')}"
        html.append(f"<div class='card'><b>{title}</b><img src='{rel}'><div class='small'>{r.get('video_filename','')}</div></div>")
    html.append("</div>")
else:
    html.append("<p>No v78i frame images found.</p>")
html.append("</div>")

html.append("</div></body></html>")
OUT_HTML.write_text("\n".join(html), encoding="utf-8")

# 5) QA / decisions.
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

add_qa("visual_groups_defined", "2", len(groups), len(groups) == 2, "hard", "GROUP_A and GROUP_B should be defined.")
add_qa("targets_created", ">0", len(targets), len(targets) > 0, "hard", "Excel visual mapping targets should be created.")
add_qa("locked_tlc1_rows", ">0", len(locked_df), len(locked_df) > 0, "hard", "TLC1 anchor rows should be locked to GROUP_A.")
add_qa("manual_template_created", "exists", OUT_TEMPLATE.exists(), OUT_TEMPLATE.exists(), "hard", "Manual fill template should exist.")
add_qa("html_board_created", "exists", OUT_HTML.exists(), OUT_HTML.exists(), "hard", "HTML review board should exist.")
add_qa("non_tlc1_rows_require_manual_review", ">=0", int((manual["camera"] != "TLC1").sum()), True, "info", "Non-TLC1 rows intentionally require manual visual resolution.")

qa_df = pd.DataFrame(qa_rows)
to_csv(qa_df, OUT_QA)

hard_quality_failures = int(((qa_df["severity"] == "hard") & (~qa_df["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v78j_quality_checks",
        "issue_type": "hard_resolver_template_failed",
        "issue_detail": f"{hard_quality_failures} hard QA checks failed.",
        "severity": "hard",
    })

issues.append({
    "item": "scope",
    "issue_type": "info_resolver_template_only",
    "issue_detail": "v78j creates a time-aware visual stream resolver template. It does not yet finalize mapping or run tracking.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

manifest = {
    "version": "v78j_time_aware_visual_stream_resolver",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "visual_groups": int(len(groups)),
    "targets": int(len(targets)),
    "manual_template_rows": int(len(manual)),
    "locked_tlc1_rows": int(len(locked_df)),
    "html_board": str(OUT_HTML),
    "hard_issue_count": int(hard_issue_count),
    "claim_boundary": "resolver template only; no final mapping applied",
}
OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

readme = f"""# v78j Time-Aware Visual Stream Resolver

## Purpose

v78j replaces the unsafe fixed TLC→c-code mapping with a visual-stream resolver.

## Visual groups

- GROUP_A: c0000/c0002 side or duplicate view, c0100 top view. TLC1/c0002 anchor locked.
- GROUP_B: c0001/c0003 side or duplicate view, c0101 top view.

## Manual action

Fill:

`{OUT_TEMPLATE}`

Allowed `selected_visual_group_id`:
- GROUP_A
- GROUP_B
- NOT_VISIBLE
- UNRESOLVED

Allowed `selected_view_type`:
- side
- top
- side_and_top
- not_visible
- unresolved

TLC1 rows are prefilled and locked from the visual anchor. Other rows must be resolved by visual evidence or explicitly marked unresolved/not visible.

## Important

This stage does not run tracking and does not finalize mappings.
"""

OUT_README.write_text(readme, encoding="utf-8")
OUT_REPORT.write_text(readme, encoding="utf-8")

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

zip_hash = sha256_file(OUT_ZIP)
OUT_SHA.write_text(f"{zip_hash}  {OUT_ZIP.name}\n")

decision = pd.DataFrame([{
    "v78j_decision": "time_aware_visual_stream_resolver_template_completed" if hard_issue_count == 0 else "time_aware_visual_stream_resolver_template_has_blocking_issues",
    "visual_groups": int(len(groups)),
    "targets": int(len(targets)),
    "manual_template_rows": int(len(manual)),
    "locked_tlc1_rows": int(len(locked_df)),
    "manual_template_path": str(OUT_TEMPLATE),
    "html_board_path": str(OUT_HTML),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_quality_failures": int(hard_quality_failures),
    "hard_issue_count": int(hard_issue_count),
    "warning_count": int(warning_count),
    "info_count": int(info_count),
    "issue_count": int(len(issues_df)),
    "ready_for_manual_visual_resolution": bool(hard_issue_count == 0),
    "ready_for_final_mapping": False,
    "ready_for_v79_full_tracking_preparation": False,
    "claim_scope": "time_aware_visual_stream_resolver_template_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# v78j Time-Aware Visual Stream Resolver\n\n"
    f"- v78j decision: {decision.iloc[0]['v78j_decision']}\n"
    f"- Visual groups: {len(groups)}\n"
    f"- Mapping targets: {len(targets)}\n"
    f"- Manual template rows: {len(manual)}\n"
    f"- Locked TLC1 rows: {len(locked_df)}\n"
    f"- Manual template: {OUT_TEMPLATE}\n"
    f"- HTML board: {OUT_HTML}\n"
    f"- Ready for manual visual resolution: {bool(hard_issue_count == 0)}\n"
    f"- Ready for final mapping: False\n"
    f"- Ready for v79 full tracking preparation: False\n\n"
    "Fill the manual template using visual evidence. Do not proceed to tracking until v78k validates completed resolutions.\n",
    encoding="utf-8"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v78j",
    "task_name": "Time-aware visual stream resolver template",
    "status": "PASS_READY_FOR_MANUAL_RESOLUTION" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": "v78c targets + v78i visual stream groups",
    "output_summary": str(PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Manually fill v78j_MANUAL_FILL_visual_stream_resolution.csv, then run v78k validation.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

to_csv(progress, OUT_PROGRESS)

print("=== v78j decision ===")
print(decision.to_string(index=False))

print("\n=== visual groups ===")
print(groups.to_string(index=False))

print("\n=== manual fill template preview ===")
print(manual.to_string(index=False))

print("\n=== QA ===")
print(qa_df.to_string(index=False))

print("\n=== issues ===")
print(issues_df.to_string(index=False))
