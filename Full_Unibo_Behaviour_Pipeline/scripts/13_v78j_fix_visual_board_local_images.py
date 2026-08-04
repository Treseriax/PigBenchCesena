from pathlib import Path
from datetime import datetime
import shutil
import csv
import pandas as pd


ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

V78I_PKG = FULL / "outputs" / "v78i_visual_stream_grouping_audit" / "Full_Unibo_Visual_Stream_Grouping_Audit"
V78J_OUT = FULL / "outputs" / "v78j_time_aware_visual_stream_resolver"
V78J_PKG = V78J_OUT / "Full_Unibo_Time_Aware_Visual_Stream_Resolver"

FEATURES = V78I_PKG / "v78i_frame_features.csv"
GROUPS = V78J_PKG / "v78j_visual_stream_group_definitions.csv"
MANUAL = V78J_PKG / "v78j_MANUAL_FILL_visual_stream_resolution.csv"
TARGETS = V78J_PKG / "v78j_excel_visual_mapping_targets.csv"

STATIC = V78J_PKG / "visual_stream_groups"
CONTACT = V78J_PKG / "contact_sheets"
STATIC.mkdir(parents=True, exist_ok=True)
CONTACT.mkdir(parents=True, exist_ok=True)

OUT_HTML_FIXED = V78J_PKG / "v78j_visual_stream_resolution_board_FIXED.html"
OUT_HTML_ORIGINAL = V78J_PKG / "v78j_visual_stream_resolution_board.html"
OUT_SUMMARY = V78J_OUT / "v78j_visual_board_image_fix_summary.csv"
OUT_NOTE = FULL / "notes" / "v78j_visual_board_image_fix_notes.md"


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


features = read_csv(FEATURES)
groups = read_csv(GROUPS)
manual = read_csv(MANUAL)
targets = read_csv(TARGETS)

copied = []
missing = []

# 1) Copy v78i frame images locally into v78j package.
if not features.empty and "frame_image" in features.columns:
    for _, r in features.iterrows():
        src_text = clean(r.get("frame_image", ""))
        if not src_text:
            missing.append({
                "type": "frame_image",
                "source": "",
                "reason": "empty frame_image path",
            })
            continue

        src = Path(src_text)
        if not src.exists():
            # fallback: maybe only basename exists in v78i visual folder
            alt = V78I_PKG / "visual_stream_groups" / src.name
            if alt.exists():
                src = alt

        if not src.exists():
            missing.append({
                "type": "frame_image",
                "source": src_text,
                "reason": "source image not found",
            })
            continue

        dst = STATIC / src.name
        shutil.copy2(src, dst)

        copied.append({
            "type": "frame_image",
            "source": str(src),
            "destination": str(dst),
            "filename": dst.name,
            "video_type": clean(r.get("video_type", "")),
            "camera_code": clean(r.get("camera_code", "")),
            "tlc_camera": clean(r.get("tlc_camera", "")),
            "hour": clean(r.get("start_hhmm", "")),
            "video_filename": clean(r.get("video_filename", "")),
        })

# 2) Copy contact sheets locally too, if target table has them.
if not targets.empty and "visual_contact_sheet" in targets.columns:
    for _, r in targets.iterrows():
        sheets = clean(r.get("visual_contact_sheet", ""))
        if not sheets:
            continue

        for item in sheets.split(";"):
            src_text = clean(item)
            if not src_text:
                continue

            src = Path(src_text)
            if not src.exists():
                missing.append({
                    "type": "contact_sheet",
                    "source": src_text,
                    "reason": "source contact sheet not found",
                })
                continue

            dst = CONTACT / src.name
            shutil.copy2(src, dst)

            copied.append({
                "type": "contact_sheet",
                "source": str(src),
                "destination": str(dst),
                "filename": dst.name,
                "video_type": "",
                "camera_code": "",
                "tlc_camera": clean(r.get("camera", "")),
                "hour": "",
                "video_filename": f"{clean(r.get('target_id',''))} {clean(r.get('camera',''))} {clean(r.get('pen',''))}",
            })

copied_df = pd.DataFrame(copied)
missing_df = pd.DataFrame(missing)

# 3) Build robust HTML with local image links only.
html = []
html.append("""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>v78j Visual Stream Resolution Board FIXED</title>
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
.card img { width:100%; border-radius:4px; display:block; margin-top:6px; }
.small { font-size:12px; color:#555; word-break:break-all; }
.groupA { border:4px solid #2d6cdf; }
.groupB { border:4px solid #8a2be2; }
.ref { border:4px solid #0a8a0a; }
.missing { color:#a00000; font-weight:bold; }
.code { font-family: monospace; background:#eee; padding:2px 4px; border-radius:3px; }
</style>
</head>
<body>
<div class="header">
<h1>v78j Visual Stream Resolution Board — FIXED LOCAL IMAGES</h1>
<p>Images are copied locally into the v78j package, so they should display from this server folder.</p>
</div>
<div class="container">
""")

html.append("<div class='section warn'><h2>Core rule</h2>")
html.append("<p><b>Do not force TLC → single c-code mapping.</b> Use visual groups:</p>")
html.append("<ul>")
html.append("<li><b>GROUP_A:</b> c0000 / c0002 side or duplicate view, c0100 top view. TLC1/c0002 anchor locked.</li>")
html.append("<li><b>GROUP_B:</b> c0001 / c0003 side or duplicate view, c0101 top view.</li>")
html.append("</ul>")
html.append("</div>")

html.append("<div class='section ok'><h2>Manual fill CSV</h2>")
html.append(f"<p>Fill this file after visual review:</p><p class='code'>{MANUAL}</p>")
html.append("<p>Allowed selected_visual_group_id: GROUP_A, GROUP_B, NOT_VISIBLE, UNRESOLVED</p>")
html.append("</div>")

html.append("<div class='section'><h2>Visual group definitions</h2>")
if not groups.empty:
    html.append(groups.to_html(index=False, escape=True))
else:
    html.append("<p class='missing'>Missing group definitions.</p>")
html.append("</div>")

html.append("<div class='section'><h2>Manual resolution targets</h2>")
if not manual.empty:
    html.append(manual.to_html(index=False, escape=True))
else:
    html.append("<p class='missing'>Missing manual template.</p>")
html.append("</div>")

# Contact sheets
html.append("<div class='section'><h2>Contact sheets from v78c/v78j targets</h2>")
contact_files = sorted(CONTACT.glob("*.jpg"))
if contact_files:
    html.append("<div class='grid'>")
    for p in contact_files:
        html.append(f"<div class='card'><b>{p.name}</b><img src='contact_sheets/{p.name}'></div>")
    html.append("</div>")
else:
    html.append("<p>No contact sheets copied or found.</p>")
html.append("</div>")

# Frame cards grouped.
html.append("<div class='section'><h2>Known TLC1 friendly reference</h2><div class='grid'>")
if not copied_df.empty:
    ref_rows = copied_df[(copied_df["type"] == "frame_image") & (copied_df["video_type"] == "friendly")].copy()
    ref_rows = ref_rows.sort_values(["hour", "filename"])
    for _, r in ref_rows.iterrows():
        fn = clean(r["filename"])
        title = f"{clean(r['tlc_camera'])} reference | {clean(r['hour'])}"
        html.append(f"<div class='card ref'><b>{title}</b><img src='visual_stream_groups/{fn}'><div class='small'>{clean(r['video_filename'])}</div></div>")
else:
    html.append("<p class='missing'>No copied frame images.</p>")
html.append("</div></div>")

def section_for_codes(title, codes, css_class):
    html.append(f"<div class='section'><h2>{title}</h2><div class='grid'>")
    if copied_df.empty:
        html.append("<p class='missing'>No copied images.</p>")
    else:
        df = copied_df[
            (copied_df["type"] == "frame_image") &
            (copied_df["camera_code"].isin(codes))
        ].copy()
        df["code_order"] = df["camera_code"].map({c:i for i,c in enumerate(codes)})
        df = df.sort_values(["code_order", "hour", "filename"])
        for _, r in df.iterrows():
            fn = clean(r["filename"])
            title2 = f"{clean(r['camera_code'])} | {clean(r['hour'])}"
            html.append(f"<div class='card {css_class}'><b>{title2}</b><img src='visual_stream_groups/{fn}'><div class='small'>{clean(r['video_filename'])}</div></div>")
    html.append("</div></div>")

section_for_codes("GROUP_A evidence: c0000 / c0002 side + c0100 top", ["c0000", "c0002", "c0100"], "groupA")
section_for_codes("GROUP_B evidence: c0001 / c0003 side + c0101 top", ["c0001", "c0003", "c0101"], "groupB")

html.append("<div class='section'><h2>Image copy summary</h2>")
summary_rows = [{
    "copied_images": len(copied_df),
    "missing_images": len(missing_df),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "static_folder": str(STATIC),
    "contact_folder": str(CONTACT),
}]
summary_df = pd.DataFrame(summary_rows)
html.append(summary_df.to_html(index=False, escape=True))
if not missing_df.empty:
    html.append("<h3>Missing image sources</h3>")
    html.append(missing_df.head(100).to_html(index=False, escape=True))
html.append("</div>")

html.append("</div></body></html>")

OUT_HTML_FIXED.write_text("\n".join(html), encoding="utf-8")

# Overwrite original too, so old URL works.
OUT_HTML_ORIGINAL.write_text("\n".join(html), encoding="utf-8")

# Summary CSV
summary = pd.DataFrame([{
    "status": "visual_board_fixed",
    "copied_images": len(copied_df),
    "missing_images": len(missing_df),
    "features_rows": len(features),
    "manual_rows": len(manual),
    "group_rows": len(groups),
    "fixed_html": str(OUT_HTML_FIXED),
    "original_html_overwritten": str(OUT_HTML_ORIGINAL),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
to_csv(summary, OUT_SUMMARY)

OUT_NOTE.write_text(
    "# v78j Visual Board Image Fix\n\n"
    f"- Status: visual_board_fixed\n"
    f"- Copied images: {len(copied_df)}\n"
    f"- Missing images: {len(missing_df)}\n"
    f"- Fixed HTML: {OUT_HTML_FIXED}\n"
    f"- Original HTML overwritten: {OUT_HTML_ORIGINAL}\n"
    f"- Static image folder: {STATIC}\n"
    f"- Contact sheet folder: {CONTACT}\n\n"
    "Open the fixed board through the v78j HTTP server.\n",
    encoding="utf-8"
)

print("=== v78j visual board image fix summary ===")
print(summary.to_string(index=False))

print("\n=== copied image sample ===")
print(copied_df.head(30).to_string(index=False) if len(copied_df) else "none")

print("\n=== missing image sample ===")
print(missing_df.head(30).to_string(index=False) if len(missing_df) else "none")
