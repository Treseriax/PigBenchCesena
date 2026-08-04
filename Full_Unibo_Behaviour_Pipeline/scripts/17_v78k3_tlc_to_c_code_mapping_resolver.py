from pathlib import Path
from datetime import datetime
import csv
import json
import hashlib
import zipfile
import shutil
import pandas as pd

ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

K1_PKG = FULL / "outputs" / "v78k1_date_specific_excel_pen_alias_resolver" / "Full_Unibo_Date_Specific_Excel_Pen_Alias_Resolver"
K1_TARGETS = K1_PKG / "v78k1_v78j_targets_resolved_with_aliases.csv"
K1_CANONICAL = K1_PKG / "v78k1_date_specific_canonical_tlc_room_pen_pattern.csv"

K2_PKG = FULL / "outputs" / "v78k2_tracking_readiness_resolver" / "Full_Unibo_Tracking_Readiness_Resolver"
K2_READINESS = K2_PKG / "v78k2_target_tracking_readiness_table.csv"

V78I_PKG = FULL / "outputs" / "v78i_visual_stream_grouping_audit" / "Full_Unibo_Visual_Stream_Grouping_Audit"
V78I_OBS = V78I_PKG / "v78i_human_visual_observations.csv"
V78I_FEATURES = V78I_PKG / "v78i_frame_features.csv"
V78I_PAIRWISE = V78I_PKG / "v78i_pairwise_visual_similarity.csv"
V78I_TLC1 = V78I_PKG / "v78i_tlc1_anchor_similarity_by_hour.csv"

OUT = FULL / "outputs" / "v78k3_tlc_to_c_code_mapping_resolver"
PKG = OUT / "Full_Unibo_TLC_to_CCode_Mapping_Resolver"
STATIC = PKG / "review_images"
NOTES = FULL / "notes"
REPORTS = FULL / "reports"
PROGRESS = FULL / "progress"

for p in [OUT, PKG, STATIC, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_MATRIX = PKG / "v78k3_tlc_to_c_code_mapping_matrix.csv"
OUT_REVIEW = PKG / "v78k3_manual_review_template.csv"
OUT_LOCKED = PKG / "v78k3_locked_mapping_rows.csv"
OUT_GROUPS = PKG / "v78k3_visual_stream_group_definitions.csv"
OUT_HTML = PKG / "v78k3_tlc_to_c_code_review_board.html"
OUT_COUNTS = PKG / "v78k3_mapping_status_counts.csv"
OUT_DECISION = OUT / "v78k3_decision_summary.csv"
OUT_ISSUES = OUT / "v78k3_issues.csv"
OUT_README = PKG / "README_v78k3_TLC_to_CCode_Mapping_Resolver.md"
OUT_MANIFEST = PKG / "v78k3_manifest.json"
OUT_ZIP = OUT / "Full_Unibo_TLC_to_CCode_Mapping_Resolver.zip"
OUT_SHA = OUT / "Full_Unibo_TLC_to_CCode_Mapping_Resolver.sha256"
OUT_NOTE = NOTES / "v78k3_tlc_to_c_code_mapping_resolver_notes.md"
OUT_REPORT = REPORTS / "v78k3_tlc_to_c_code_mapping_resolver_report.md"
OUT_PROGRESS = PROGRESS / "full_unibo_pipeline_progress_log.csv"


def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def read_csv(path):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path).fillna("")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(clean)
    return df


def to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_review_image(src_text):
    src_text = clean(src_text)
    if not src_text:
        return ""
    src = Path(src_text)
    if not src.exists():
        alt = V78I_PKG / "visual_stream_groups" / src.name
        if alt.exists():
            src = alt
    if not src.exists():
        return ""
    dst = STATIC / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
    return f"review_images/{dst.name}"


issues = []

targets = read_csv(K1_TARGETS)
canonical = read_csv(K1_CANONICAL)
readiness = read_csv(K2_READINESS)
obs = read_csv(V78I_OBS)
features = read_csv(V78I_FEATURES)
pairwise = read_csv(V78I_PAIRWISE)
tlc1_anchor = read_csv(V78I_TLC1)

required_inputs = [
    (K1_TARGETS, targets, "v78k1 targets"),
    (K2_READINESS, readiness, "v78k2 readiness table"),
    (V78I_FEATURES, features, "v78i frame features"),
]

for path, df, name in required_inputs:
    if df.empty:
        issues.append({
            "item": str(path),
            "issue_type": f"hard_missing_{name.replace(' ', '_')}",
            "issue_detail": f"Required input is missing or empty: {name}",
            "severity": "hard",
        })

# Visual group definitions based on v78i human review.
groups = pd.DataFrame([
    {
        "visual_group_id": "GROUP_A",
        "side_or_duplicate_codes": "c0002;c0000",
        "primary_side_code": "c0002",
        "secondary_side_code": "c0000",
        "top_view_code": "c0100",
        "known_tlc_anchor": "TLC1",
        "status": "locked_for_TLC1_only",
        "evidence": "User visually confirmed TLC1/c0002 anchor; c0000 appears same/duplicate; c0100 appears top-view of same area.",
    },
    {
        "visual_group_id": "GROUP_B",
        "side_or_duplicate_codes": "c0001;c0003",
        "primary_side_code": "c0001",
        "secondary_side_code": "c0003",
        "top_view_code": "c0101",
        "known_tlc_anchor": "",
        "status": "unassigned_visual_group",
        "evidence": "User observed c0001 and c0003 visually similar/same; c0101 appears top-view of same area.",
    },
])
to_csv(groups, OUT_GROUPS)

matrix_rows = []
review_rows = []
locked_rows = []

if not targets.empty:
    # Make TLC-level target summaries.
    target_summary = (
        targets.groupby(["date", "camera"], dropna=False)
        .agg(
            resolved_pens=("resolved_room_pen", lambda x: ";".join(sorted(set([clean(v) for v in x if clean(v)])))),
            target_ids=("target_id", lambda x: ";".join(x)),
            unresolved_windows=("unresolved_windows", lambda x: sum(int(float(v)) if clean(v) else 0 for v in x)),
        )
        .reset_index()
        .rename(columns={"camera": "tlc_camera"})
    )

    for _, r in target_summary.iterrows():
        date = clean(r["date"])
        tlc = clean(r["tlc_camera"])
        pens = clean(r["resolved_pens"])

        if date == "2021-07-22" and tlc == "TLC1":
            status = "LOCKED_TLC1_VISUAL_ANCHOR"
            group = "GROUP_A"
            primary = "c0002"
            secondary = "c0000"
            top = "c0100"
            confidence = "high"
            evidence = "TLC1 friendly reference visually matched c0002; v78i records c0000 duplicate/same-view and c0100 top-view relation."
            action = "Keep locked. Do not generalize to TLC2-TLC6."
            ready = True
        else:
            status = "NEEDS_MANUAL_OR_EXTERNAL_TLC_TO_CCODE_MAPPING"
            group = ""
            primary = ""
            secondary = ""
            top = ""
            confidence = ""
            evidence = "No trusted TLC-to-c-code evidence available for this TLC camera."
            action = "Resolve by visual evidence, dataset layout, metadata, or supervisor confirmation. Do not guess."
            ready = False

        row = {
            "date": date,
            "tlc_camera": tlc,
            "resolved_pens_from_excel": pens,
            "target_ids": clean(r["target_ids"]),
            "unresolved_windows": int(r["unresolved_windows"]),
            "mapping_status": status,
            "selected_visual_group_id": group,
            "primary_side_code": primary,
            "secondary_side_code": secondary,
            "top_view_code": top,
            "confidence": confidence,
            "evidence": evidence,
            "recommended_action": action,
            "ready_for_camera_level_video_mapping": ready,
            "ready_for_pen_level_tracking": False,
        }

        matrix_rows.append(row)

        if status.startswith("LOCKED"):
            locked_rows.append(row)
        else:
            review_rows.append({
                **row,
                "manual_selected_visual_group_id": "",
                "manual_primary_side_code": "",
                "manual_secondary_side_code": "",
                "manual_top_view_code": "",
                "manual_decision": "UNRESOLVED",
                "manual_confidence": "",
                "manual_evidence_note": "",
                "allowed_visual_groups": "GROUP_A;GROUP_B;NOT_VISIBLE;UNRESOLVED",
                "allowed_decisions": "RESOLVED;NOT_VISIBLE;UNRESOLVED;NEEDS_EXTERNAL_CONFIRMATION",
            })

matrix = pd.DataFrame(matrix_rows)
review = pd.DataFrame(review_rows)
locked = pd.DataFrame(locked_rows)

to_csv(matrix, OUT_MATRIX)
to_csv(review, OUT_REVIEW)
to_csv(locked, OUT_LOCKED)

if matrix.empty:
    issues.append({
        "item": "mapping_matrix",
        "issue_type": "hard_empty_mapping_matrix",
        "issue_detail": "No TLC-to-c-code mapping matrix rows were created.",
        "severity": "hard",
    })

if not matrix.empty:
    counts = (
        matrix.groupby("mapping_status")
        .size()
        .reset_index(name="tlc_camera_count")
        .sort_values("mapping_status")
    )
else:
    counts = pd.DataFrame(columns=["mapping_status", "tlc_camera_count"])
to_csv(counts, OUT_COUNTS)

# Build local image review board.
feature_rows = []
if not features.empty:
    # Keep key hours only to reduce size.
    keep_hours = {"07:00", "08:00", "09:00", "10:00", "11:00", "12:00"}
    f = features.copy()
    f = f[f["start_hhmm"].isin(keep_hours) | (f["video_type"] == "friendly")].copy()
    for _, r in f.iterrows():
        local = copy_review_image(clean(r.get("frame_image")))
        feature_rows.append({
            "video_type": clean(r.get("video_type")),
            "camera_code": clean(r.get("camera_code")),
            "tlc_camera": clean(r.get("tlc_camera")),
            "start_hhmm": clean(r.get("start_hhmm")),
            "video_filename": clean(r.get("video_filename")),
            "local_image": local,
        })
features_local = pd.DataFrame(feature_rows)

html = []
html.append("""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>v78k3 TLC-to-c-code Mapping Review Board</title>
<style>
body { font-family: Arial, sans-serif; background:#f5f5f5; margin:0; }
.header { background:#111; color:white; padding:18px 24px; position:sticky; top:0; z-index:10; }
.container { padding:24px; }
.section { background:white; padding:18px; margin-bottom:24px; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,.08); }
.warn { background:#fff3cd; border:1px solid #d6b44c; padding:12px; border-radius:8px; }
.ok { background:#e9ffe9; border:1px solid #60b060; padding:12px; border-radius:8px; }
table { border-collapse: collapse; width:100%; font-size:13px; }
td, th { border:1px solid #ccc; padding:5px; vertical-align:top; }
th { background:#eee; }
.grid { display:flex; flex-wrap:wrap; gap:12px; }
.card { width:31%; min-width:260px; background:#fafafa; border:1px solid #ccc; border-radius:8px; padding:8px; }
.card img { width:100%; border-radius:4px; margin-top:6px; }
.small { font-size:12px; color:#555; word-break:break-all; }
.groupA { border:4px solid #2d6cdf; }
.groupB { border:4px solid #8a2be2; }
.ref { border:4px solid #0a8a0a; }
</style>
</head>
<body>
<div class="header">
<h1>v78k3 TLC-to-c-code Mapping Review Board</h1>
<p>Purpose: resolve TLC camera streams to encoded c-code streams without changing tracking labels.</p>
</div>
<div class="container">
""")

html.append("<div class='section warn'><h2>Core boundary</h2>")
html.append("<p>This board is for <b>TLC camera-level video mapping</b>, not pen-level ROI and not final tracking.</p>")
html.append("<p>TLC1 is locked only because it has a friendly reference anchor. TLC2-TLC6 remain unresolved unless strong evidence is found.</p>")
html.append("</div>")

html.append("<div class='section'><h2>Visual stream groups</h2>")
html.append(groups.to_html(index=False, escape=True))
html.append("</div>")

html.append("<div class='section'><h2>TLC-to-c-code mapping matrix</h2>")
html.append(matrix.to_html(index=False, escape=True) if not matrix.empty else "<p>empty</p>")
html.append("</div>")

html.append("<div class='section'><h2>Manual review template path</h2>")
html.append(f"<p><code>{OUT_REVIEW}</code></p>")
html.append(review.to_html(index=False, escape=True) if not review.empty else "<p>No unresolved TLC cameras.</p>")
html.append("</div>")

def image_section(title, df, css):
    html.append(f"<div class='section'><h2>{title}</h2><div class='grid'>")
    if df.empty:
        html.append("<p>No images.</p>")
    else:
        for _, r in df.iterrows():
            img = clean(r.get("local_image"))
            title2 = f"{clean(r.get('video_type'))} | {clean(r.get('camera_code'))} {clean(r.get('tlc_camera'))} | {clean(r.get('start_hhmm'))}"
            html.append(f"<div class='card {css}'><b>{title2}</b>")
            if img:
                html.append(f"<img src='{img}'>")
            html.append(f"<div class='small'>{clean(r.get('video_filename'))}</div></div>")
    html.append("</div></div>")

if not features_local.empty:
    image_section(
        "TLC1 friendly reference images",
        features_local[(features_local["video_type"] == "friendly") & (features_local["tlc_camera"] == "TLC1")].sort_values(["start_hhmm", "video_filename"]),
        "ref"
    )
    image_section(
        "GROUP_A candidate images: c0002 / c0000 / c0100",
        features_local[features_local["camera_code"].isin(["c0002", "c0000", "c0100"])].sort_values(["camera_code", "start_hhmm", "video_filename"]),
        "groupA"
    )
    image_section(
        "GROUP_B candidate images: c0001 / c0003 / c0101",
        features_local[features_local["camera_code"].isin(["c0001", "c0003", "c0101"])].sort_values(["camera_code", "start_hhmm", "video_filename"]),
        "groupB"
    )

html.append("</div></body></html>")
OUT_HTML.write_text("\n".join(html), encoding="utf-8")

# Issues / decision.
locked_count = int(len(locked))
unresolved_count = int(len(review))
ready_camera_level_count = int(matrix["ready_for_camera_level_video_mapping"].sum()) if not matrix.empty else 0

if unresolved_count:
    issues.append({
        "item": "tlc_to_c_code_mapping",
        "issue_type": "info_unresolved_tlc_cameras",
        "issue_detail": f"{unresolved_count} TLC cameras remain unresolved for encoded c-code mapping.",
        "severity": "info",
    })

issues.append({
    "item": "scope",
    "issue_type": "info_mapping_resolver_only",
    "issue_detail": "v78k3 creates TLC-to-c-code mapping matrix and review board. It does not run tracking.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
to_csv(issues_df, OUT_ISSUES)

hard_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

readme = """# v78k3 TLC-to-c-code Mapping Resolver

This package creates a TLC camera-level mapping matrix.

Known:
- TLC1 is locked to GROUP_A / c0002 primary, c0000 duplicate, c0100 top-view.
- TLC2-TLC6 require manual/external evidence before mapping.

Boundary:
- This does not provide pen-level ROI.
- This does not run tracking.
- This does not create final behaviour labels.
"""
OUT_README.write_text(readme, encoding="utf-8")
OUT_REPORT.write_text(readme, encoding="utf-8")

manifest = {
    "version": "v78k3_tlc_to_c_code_mapping_resolver",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "mapping_matrix_rows": int(len(matrix)),
    "locked_count": locked_count,
    "unresolved_count": unresolved_count,
    "ready_camera_level_count": ready_camera_level_count,
    "hard_issue_count": hard_count,
    "warning_count": warning_count,
    "claim_boundary": "TLC-to-c-code mapping resolver only; no tracking",
}
OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

zip_hash = sha256_file(OUT_ZIP)
OUT_SHA.write_text(f"{zip_hash}  {OUT_ZIP.name}\n")

decision = pd.DataFrame([{
    "v78k3_decision": "tlc_to_c_code_mapping_matrix_created_unresolved_remain" if unresolved_count else "tlc_to_c_code_mapping_resolved",
    "mapping_matrix_rows": int(len(matrix)),
    "locked_tlc_camera_rows": locked_count,
    "unresolved_tlc_camera_rows": unresolved_count,
    "ready_camera_level_video_mapping_rows": ready_camera_level_count,
    "manual_review_template_path": str(OUT_REVIEW),
    "review_board_html": str(OUT_HTML),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_issue_count": hard_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "ready_for_v79_tracking_preparation": False,
    "ready_for_full_pen_level_tracking": False,
    "claim_scope": "tlc_to_c_code_mapping_resolver_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# v78k3 TLC-to-c-code Mapping Resolver\n\n"
    f"- Decision: {decision.iloc[0]['v78k3_decision']}\n"
    f"- Mapping matrix rows: {len(matrix)}\n"
    f"- Locked TLC rows: {locked_count}\n"
    f"- Unresolved TLC rows: {unresolved_count}\n"
    f"- Ready camera-level video mapping rows: {ready_camera_level_count}\n"
    f"- Review board: {OUT_HTML}\n"
    f"- Manual review template: {OUT_REVIEW}\n"
    f"- Ready for v79 tracking preparation: False\n\n"
    "TLC1 is locked via visual anchor. TLC2-TLC6 require additional evidence before mapping to encoded c-codes.\n",
    encoding="utf-8"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v78k3",
    "task_name": "TLC-to-c-code mapping resolver",
    "status": "MATRIX_CREATED_UNRESOLVED_REMAIN" if unresolved_count else "PASS",
    "input_summary": "v78k1 targets + v78k2 readiness + v78i visual observations",
    "output_summary": str(PKG),
    "hard_issues": hard_count,
    "warnings": warning_count,
    "next_action": "Review unresolved TLC2-TLC6 mapping evidence or obtain external camera layout confirmation.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row
to_csv(progress, OUT_PROGRESS)

print("=== v78k3 decision ===")
print(decision.to_string(index=False))

print("\n=== mapping matrix ===")
print(matrix.to_string(index=False) if not matrix.empty else "none")

print("\n=== manual review template ===")
print(review.to_string(index=False) if not review.empty else "none")

print("\n=== status counts ===")
print(counts.to_string(index=False) if not counts.empty else "none")

print("\n=== issues ===")
print(issues_df.to_string(index=False))
