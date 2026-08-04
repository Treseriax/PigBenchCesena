from pathlib import Path
from datetime import datetime
import csv
import re
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V45_CLIP_OBJECTS = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_object_propagated_annotations.csv"
V61B_SHORTLIST = W8 / "outputs" / "v61b_canonical_annotation_source_shortlist" / "week8_v61b_canonical_annotation_shortlist.csv"

EXCEL_DIR = Path("/work/pig/datasets/Unibo/excel")

OUT = W8 / "outputs" / "v62a_canonical_excel_structure_audit"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_SOURCE_VIDEO_SUMMARY = OUT / "week8_v62a_source_video_summary.csv"
OUT_EXCEL_SHEET_GRID_MARKERS = OUT / "week8_v62a_excel_sheet_grid_markers.csv"
OUT_RELEVANT_SHEET_PREVIEW = OUT / "week8_v62a_relevant_sheet_preview.csv"
OUT_SOURCE_TO_EXCEL_CANDIDATE = OUT / "week8_v62a_source_to_excel_candidate_alignment.csv"
OUT_DECISION = OUT / "week8_v62a_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v62a_issues.csv"
OUT_NOTE = NOTES / "week8_v62a_canonical_excel_structure_audit_notes.md"
OUT_REPORT = REPORTS / "week8_v62a_canonical_excel_structure_audit_report.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def clean_str(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", clean_str(s).lower())


def extract_video_tokens(video_id):
    s = clean_str(video_id)
    sl = s.lower()

    cam = ""
    room = ""
    pen = ""
    hour = ""

    m = re.search(r"tlc\s*([0-9]+)", sl)
    if m:
        cam = m.group(1)

    m = re.search(r"\b([bcm])\s*([0-9]+)\b", sl)
    if m:
        room = m.group(1).upper()
        pen = m.group(2)

    m = re.search(r"(\d{3,4})[-_]?(\d{3,4})", sl)
    if m:
        start = m.group(1).zfill(4)
        end = m.group(2).zfill(4)
        hour = f"{start[:2]}:00-{end[:2]}:00"

    return cam, room, pen, hour


def likely_sheet_score(video_id, path, sheet):
    cam, room, pen, hour = extract_video_tokens(video_id)
    text = f"{path.name} {sheet}".lower()
    score = 0

    if cam and f"tlc {cam}" in text:
        score += 5
    if cam and f"tlc{cam}" in text.replace(" ", ""):
        score += 5
    if room and room.lower() in text:
        score += 3
    if pen and re.search(rf"\b{pen}\b", text):
        score += 3
    if "giorno 1" in text and ("tlc 1" in video_id.lower() or "c0001210722" in video_id.lower()):
        score += 2

    return score


def grid_markers(path, sheet):
    rows = []
    try:
        df = pd.read_excel(path, sheet_name=sheet, header=None, dtype=object)
    except Exception as e:
        return [{
            "path": str(path),
            "sheet": sheet,
            "status": "read_error",
            "error": str(e),
        }]

    keywords = [
        "Fascia oraria", "Periodo di osservazione",
        "verde", "green", "blu", "blue", "rosso", "red",
        "rosa", "pink", "viola", "purple", "ciano", "cyan",
        "PI", "SI", "LAI", "STI", "NU", "BE", "DE", "AN", "IN", "IA", "BOX",
    ]

    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            val = clean_str(df.iat[r, c])
            if not val:
                continue

            val_l = val.lower()
            hit = ""
            for k in keywords:
                if k.lower() == val_l or k.lower() in val_l:
                    hit = k
                    break

            if hit:
                context_vals = []
                c0 = max(0, c - 3)
                c1 = min(df.shape[1], c + 8)
                for cc in range(c0, c1):
                    context_vals.append(clean_str(df.iat[r, cc]))
                rows.append({
                    "path": str(path),
                    "sheet": sheet,
                    "row_index_0based": r,
                    "col_index_0based": c,
                    "matched_keyword": hit,
                    "cell_value": val,
                    "row_context": " | ".join(context_vals),
                    "status": "ok",
                    "error": "",
                })

    return rows


def sheet_preview(path, sheet, max_rows=80, max_cols=80):
    try:
        df = pd.read_excel(path, sheet_name=sheet, header=None, dtype=object)
    except Exception as e:
        return pd.DataFrame([{
            "path": str(path),
            "sheet": sheet,
            "row_index_0based": "",
            "preview": "",
            "error": str(e),
        }])

    out_rows = []
    rr = min(max_rows, df.shape[0])
    cc = min(max_cols, df.shape[1])

    for r in range(rr):
        vals = [clean_str(df.iat[r, c]) for c in range(cc)]
        # Keep rows that contain useful information.
        joined = " | ".join(vals)
        if any(tok in joined.lower() for tok in [
            "fascia", "periodo", "verde", "green", "blu", "blue",
            "rosso", "red", "rosa", "pink", "viola", "purple",
            "ciano", "cyan", "lai", "sti", "box"
        ]):
            out_rows.append({
                "path": str(path),
                "sheet": sheet,
                "row_index_0based": r,
                "preview": joined,
                "error": "",
            })

    return pd.DataFrame(out_rows)


issues = []

for p in [V45_CLIP_OBJECTS, V61B_SHORTLIST]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required v62a input is missing.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v62a_decision": "canonical_excel_structure_audit_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v62b_canonical_excel_extraction": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


clip_objects = pd.read_csv(V45_CLIP_OBJECTS)
shortlist = pd.read_csv(V61B_SHORTLIST)

source_summary = (
    clip_objects.groupby("video_id")
    .agg(
        scanframe_count=("scan_frame_id", "nunique"),
        object_count=("final_box_id", "count"),
        first_scanframe=("scan_frame_id", "min"),
        last_scanframe=("scan_frame_id", "max"),
    )
    .reset_index()
    .sort_values("video_id")
)

source_summary[["camera_guess", "room_guess", "pen_guess", "hour_guess"]] = source_summary["video_id"].apply(
    lambda x: pd.Series(extract_video_tokens(x))
)

safe_to_csv(source_summary, OUT_SOURCE_VIDEO_SUMMARY)

# Candidate alignment.
align_rows = []
for _, v in source_summary.iterrows():
    video_id = clean_str(v["video_id"])
    for _, s in shortlist.iterrows():
        path = Path(clean_str(s["path"]))
        sheet = clean_str(s["sheet"])
        score = likely_sheet_score(video_id, path, sheet)
        if score > 0:
            align_rows.append({
                "video_id": video_id,
                "scanframe_count": int(v["scanframe_count"]),
                "camera_guess": v["camera_guess"],
                "room_guess": v["room_guess"],
                "pen_guess": v["pen_guess"],
                "hour_guess": v["hour_guess"],
                "candidate_path": str(path),
                "candidate_sheet": sheet,
                "alignment_score": score,
            })

alignment = pd.DataFrame(align_rows)
if len(alignment):
    alignment = alignment.sort_values(["video_id", "alignment_score"], ascending=[True, False])
else:
    alignment = pd.DataFrame(columns=[
        "video_id", "scanframe_count", "camera_guess", "room_guess", "pen_guess", "hour_guess",
        "candidate_path", "candidate_sheet", "alignment_score"
    ])

safe_to_csv(alignment, OUT_SOURCE_TO_EXCEL_CANDIDATE)

# Inspect unique likely relevant sheets, but keep it bounded.
relevant = alignment[alignment["alignment_score"] >= 8][["candidate_path", "candidate_sheet"]].drop_duplicates()

# Ensure TLC 1 B1 original file is inspected because current validated data heavily uses it.
extra = pd.DataFrame([{
    "candidate_path": str(EXCEL_DIR / "Giorno 1 - 22_7_2021 Tutti.xlsx"),
    "candidate_sheet": "TLC 1 B1",
}])

relevant = pd.concat([relevant, extra], ignore_index=True).drop_duplicates()
relevant = relevant.head(20)

marker_rows = []
preview_frames = []

for _, r in relevant.iterrows():
    path = Path(r["candidate_path"])
    sheet = clean_str(r["candidate_sheet"])
    if not path.exists():
        marker_rows.append({
            "path": str(path),
            "sheet": sheet,
            "status": "missing_file",
            "error": "file does not exist",
        })
        continue

    marker_rows.extend(grid_markers(path, sheet))
    preview_frames.append(sheet_preview(path, sheet))

markers = pd.DataFrame(marker_rows)
safe_to_csv(markers, OUT_EXCEL_SHEET_GRID_MARKERS)

if preview_frames:
    preview = pd.concat(preview_frames, ignore_index=True)
else:
    preview = pd.DataFrame(columns=["path", "sheet", "row_index_0based", "preview", "error"])

safe_to_csv(preview, OUT_RELEVANT_SHEET_PREVIEW)

if len(alignment) == 0:
    issues.append({
        "item": "source_to_excel_alignment",
        "issue_type": "warning_no_source_alignment",
        "issue_detail": "Could not align source video IDs to Excel sheets automatically.",
        "severity": "warning",
    })

if len(preview) == 0:
    issues.append({
        "item": "excel_preview",
        "issue_type": "warning_no_relevant_preview_rows",
        "issue_detail": "No relevant sheet preview rows found.",
        "severity": "warning",
    })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0

decision = pd.DataFrame([{
    "v62a_decision": "canonical_excel_structure_audit_completed" if hard_issue_count == 0 else "canonical_excel_structure_audit_blocked",
    "source_video_count": int(source_summary["video_id"].nunique()),
    "alignment_candidate_rows": int(len(alignment)),
    "relevant_sheet_count": int(len(relevant)),
    "marker_row_count": int(len(markers)),
    "preview_row_count": int(len(preview)),
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v62b_canonical_excel_extraction": bool(hard_issue_count == 0),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v62a Canonical Excel Structure Audit\n\n"
    f"- v62a decision: {decision.iloc[0]['v62a_decision']}\n"
    f"- Source videos: {source_summary['video_id'].nunique()}\n"
    f"- Alignment candidate rows: {len(alignment)}\n"
    f"- Relevant sheets inspected: {len(relevant)}\n"
    f"- Marker rows found: {len(markers)}\n"
    f"- Preview rows found: {len(preview)}\n"
    f"- Ready for v62b canonical Excel extraction: {hard_issue_count == 0}\n\n"
    "Interpretation: v62a inspects the original Excel files as raw grids. v62b should parse colour rows and behaviour observations from these canonical sheets, not from derived Week7/Week8 outputs.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v62a Canonical Excel Structure Audit Report\n\n"
    f"Decision: {decision.iloc[0]['v62a_decision']}\n\n"
    f"Source video summary: `{OUT_SOURCE_VIDEO_SUMMARY}`\n\n"
    f"Source-to-Excel candidate alignment: `{OUT_SOURCE_TO_EXCEL_CANDIDATE}`\n\n"
    f"Excel sheet grid markers: `{OUT_EXCEL_SHEET_GRID_MARKERS}`\n\n"
    f"Relevant sheet preview: `{OUT_RELEVANT_SHEET_PREVIEW}`\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v62a",
    "task_name": "Canonical Excel structure audit",
    "status": "PASS" if hard_issue_count == 0 else "BLOCKED",
    "input_summary": str(V61B_SHORTLIST),
    "output_summary": str(OUT),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Parse canonical colour and behaviour rows from original Excel sheets.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_SOURCE_VIDEO_SUMMARY)
print(OUT_SOURCE_TO_EXCEL_CANDIDATE)
print(OUT_EXCEL_SHEET_GRID_MARKERS)
print(OUT_RELEVANT_SHEET_PREVIEW)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_NOTE)

print()
print("=== v62a decision ===")
print(decision.to_string(index=False))

print()
print("=== source video summary ===")
print(source_summary.to_string(index=False))

print()
print("=== source-to-excel candidate alignment head ===")
print(alignment.head(40).to_string(index=False))

print()
print("=== relevant sheet preview head ===")
pd.set_option("display.max_colwidth", 220)
print(preview.head(40).to_string(index=False))

print()
print("=== issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
