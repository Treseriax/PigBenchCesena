from pathlib import Path
from datetime import datetime
import re
import csv
import json
import hashlib
import zipfile

import pandas as pd

try:
    import openpyxl
except Exception:
    openpyxl = None


ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

OUT = FULL / "outputs" / "v78g_server_raw_camera_mapping_search"
PKG = OUT / "Full_Unibo_Server_Raw_Camera_Mapping_Search"
NOTES = FULL / "notes"
REPORTS = FULL / "reports"
PROGRESS = FULL / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_FILENAME_HITS = PKG / "v78g_filename_hits.csv"
OUT_TEXT_HITS = PKG / "v78g_text_hits.csv"
OUT_EXCEL_HITS = PKG / "v78g_excel_cell_hits.csv"
OUT_PAIR_EVIDENCE = PKG / "v78g_tlc_code_pair_evidence.csv"
OUT_POSSIBLE_LAYOUT_FILES = PKG / "v78g_possible_layout_or_doc_files.csv"
OUT_SUMMARY = PKG / "v78g_candidate_mapping_summary.csv"
OUT_QA = PKG / "v78g_quality_checks.csv"
OUT_README = PKG / "README_v78g_Server_Raw_Camera_Mapping_Search.md"
OUT_MANIFEST = PKG / "v78g_manifest.json"

OUT_DECISION = OUT / "v78g_decision_summary.csv"
OUT_ISSUES = OUT / "v78g_issues.csv"
OUT_ZIP = OUT / "Full_Unibo_Server_Raw_Camera_Mapping_Search.zip"
OUT_SHA = OUT / "Full_Unibo_Server_Raw_Camera_Mapping_Search.sha256"
OUT_NOTE = NOTES / "v78g_server_raw_camera_mapping_search_notes.md"
OUT_REPORT = REPORTS / "v78g_server_raw_camera_mapping_search_report.md"
OUT_PROGRESS = PROGRESS / "full_unibo_pipeline_progress_log.csv"

SEARCH_ROOTS = [
    Path("/work/pig/datasets"),
    ROOT,
]

EXCLUDE_PARTS = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    "outputs",
    "final_delivery_package_v2",
    "final_delivery_package_v3",
    "Full_Unibo_Behaviour_Pipeline/outputs",
    "Week6_Unibo_Dataset_Validation/outputs",
    "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation/outputs",
    "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation/outputs",
}

TEXT_EXTS = {
    ".txt", ".md", ".csv", ".tsv", ".json", ".jsonl",
    ".yaml", ".yml", ".ini", ".cfg", ".conf", ".py",
    ".sh", ".log"
}

EXCEL_EXTS = {".xlsx", ".xlsm"}

DOC_EXTS = {
    ".pdf", ".doc", ".docx", ".ppt", ".pptx",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp",
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

KEY_RE = re.compile(
    r"(TLC\s*[1-6]|c0000|c0001|c0002|c0003|c0100|c0101|"
    r"camera|telecamera|cam|layout|map|mappa|mapping|code|codice|"
    r"B1|B3|B4|B6|C1|C3|C4|C6|M1|M2|M3|M4|big|small)",
    re.IGNORECASE,
)

TLC_RE = re.compile(r"\bTLC\s*([1-6])\b", re.IGNORECASE)
CODE_RE = re.compile(r"\b(c0000|c0001|c0002|c0003|c0100|c0101)\b", re.IGNORECASE)


def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def excluded(path):
    s = str(path)
    for part in EXCLUDE_PARTS:
        if part in s:
            return True
    return False


def tlcs(text):
    return sorted({f"TLC{m.group(1)}" for m in TLC_RE.finditer(text or "")})


def codes(text):
    return sorted({m.group(1).lower() for m in CODE_RE.finditer(text or "")})


def add_pair(pair_rows, source_path, source_type, location, text):
    found_tlcs = tlcs(text)
    found_codes = codes(text)

    if not found_tlcs or not found_codes:
        return

    snippet = " ".join(str(text).split())
    if len(snippet) > 700:
        snippet = snippet[:700] + "..."

    for t in found_tlcs:
        for c in found_codes:
            pair_rows.append({
                "source_path": str(source_path),
                "source_type": source_type,
                "location": location,
                "tlc_camera": t,
                "video_camera_code": c,
                "evidence_snippet": snippet,
            })


def file_kind(path):
    ext = path.suffix.lower()
    if ext in TEXT_EXTS:
        return "text"
    if ext in EXCEL_EXTS:
        return "excel"
    if ext in DOC_EXTS:
        return "doc_or_image"
    if ext in VIDEO_EXTS:
        return "video"
    return "other"


filename_hits = []
text_hits = []
excel_hits = []
pair_rows = []
layout_files = []
issues = []

files_seen = 0
files_scanned_text = 0
files_scanned_excel = 0
files_skipped_large = 0

for root in SEARCH_ROOTS:
    if not root.exists():
        issues.append({
            "item": str(root),
            "issue_type": "warning_search_root_missing",
            "issue_detail": "Search root does not exist.",
            "severity": "warning",
        })
        continue

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if excluded(path):
            continue

        files_seen += 1

        ext = path.suffix.lower()
        kind = file_kind(path)
        name_text = str(path)

        # filename-level hit
        if KEY_RE.search(name_text):
            filename_hits.append({
                "path": str(path),
                "file_name": path.name,
                "file_ext": ext,
                "kind": kind,
                "size_bytes": path.stat().st_size if path.exists() else "",
                "reason": "filename/path contains TLC, camera, code, layout, map, pen, or related term",
            })

            if kind in {"doc_or_image", "other"}:
                layout_files.append({
                    "path": str(path),
                    "file_name": path.name,
                    "file_ext": ext,
                    "kind": kind,
                    "size_bytes": path.stat().st_size if path.exists() else "",
                    "reason": "possible layout/document/image/manual evidence from filename",
                })

        # skip giant non-video files
        try:
            size = path.stat().st_size
        except Exception:
            size = 0

        if size > 30 * 1024 * 1024 and ext not in VIDEO_EXTS:
            files_skipped_large += 1
            continue

        # text content scan
        if ext in TEXT_EXTS:
            files_scanned_text += 1
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_no, line in enumerate(f, start=1):
                        if KEY_RE.search(line):
                            snippet = " ".join(line.split())
                            text_hits.append({
                                "path": str(path),
                                "line": line_no,
                                "text": snippet[:1000],
                            })
                            add_pair(pair_rows, path, "text_line", f"L{line_no}", line)
            except Exception as e:
                issues.append({
                    "item": str(path),
                    "issue_type": "warning_text_scan_failed",
                    "issue_detail": str(e)[:500],
                    "severity": "warning",
                })

        # excel content scan
        if ext in EXCEL_EXTS and openpyxl is not None:
            files_scanned_excel += 1
            try:
                wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
                for ws in wb.worksheets:
                    for row in ws.iter_rows():
                        row_values = []
                        hit_cells = []
                        for cell in row:
                            value = cell.value
                            s = clean(value)
                            row_values.append(s)
                            if s and KEY_RE.search(s):
                                hit_cells.append((cell.coordinate, s))

                        if hit_cells:
                            row_text = " ; ".join(v for v in row_values if v)
                            for coord, val in hit_cells:
                                excel_hits.append({
                                    "path": str(path),
                                    "sheet": ws.title,
                                    "cell": coord,
                                    "value": val,
                                    "row_text": row_text[:1500],
                                })
                            add_pair(pair_rows, path, "excel_row", f"{ws.title}", row_text)
                wb.close()
            except Exception as e:
                issues.append({
                    "item": str(path),
                    "issue_type": "warning_excel_scan_failed",
                    "issue_detail": str(e)[:500],
                    "severity": "warning",
                })

filename_df = pd.DataFrame(filename_hits)
text_df = pd.DataFrame(text_hits)
excel_df = pd.DataFrame(excel_hits)
pair_df = pd.DataFrame(pair_rows).drop_duplicates() if pair_rows else pd.DataFrame(
    columns=["source_path", "source_type", "location", "tlc_camera", "video_camera_code", "evidence_snippet"]
)
layout_df = pd.DataFrame(layout_files).drop_duplicates() if layout_files else pd.DataFrame(
    columns=["path", "file_name", "file_ext", "kind", "size_bytes", "reason"]
)

safe_to_csv(filename_df, OUT_FILENAME_HITS)
safe_to_csv(text_df, OUT_TEXT_HITS)
safe_to_csv(excel_df, OUT_EXCEL_HITS)
safe_to_csv(pair_df, OUT_PAIR_EVIDENCE)
safe_to_csv(layout_df, OUT_POSSIBLE_LAYOUT_FILES)

if len(pair_df):
    summary = (
        pair_df.groupby(["tlc_camera", "video_camera_code"])
        .agg(
            evidence_rows=("source_path", "count"),
            source_count=("source_path", "nunique"),
            example_sources=("source_path", lambda x: ";".join(sorted(set(x))[:10])),
            example_snippets=("evidence_snippet", lambda x: " || ".join(list(x)[:3])),
        )
        .reset_index()
        .sort_values(["tlc_camera", "evidence_rows"], ascending=[True, False])
    )
else:
    summary = pd.DataFrame(columns=[
        "tlc_camera", "video_camera_code", "evidence_rows",
        "source_count", "example_sources", "example_snippets"
    ])

safe_to_csv(summary, OUT_SUMMARY)

# Quality checks
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

add_qa("files_seen", ">0", files_seen, files_seen > 0, "hard", "Server files should be visible.")
add_qa("filename_hits", ">=0", len(filename_df), len(filename_df) >= 0, "info", "Filename hits are diagnostic.")
add_qa("text_or_excel_hits", ">=0", len(text_df) + len(excel_df), len(text_df) + len(excel_df) >= 0, "info", "Content hits are diagnostic.")
add_qa("pair_evidence_rows", ">=0", len(pair_df), len(pair_df) >= 0, "info", "TLC-code co-occurrence evidence is diagnostic.")
add_qa("possible_layout_files", ">=0", len(layout_df), len(layout_df) >= 0, "info", "Layout/doc/image candidate files are diagnostic.")

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_QA)

hard_quality_failures = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v78g_quality_checks",
        "issue_type": "hard_raw_search_failed",
        "issue_detail": f"{hard_quality_failures} hard QA checks failed.",
        "severity": "hard",
    })

if len(pair_df) == 0:
    issues.append({
        "item": "tlc_code_pair_evidence",
        "issue_type": "info_no_direct_tlc_code_pair_found",
        "issue_detail": "No direct TLC↔c-code pair was found in raw server files. External supervisor/layout confirmation may be required.",
        "severity": "info",
    })

issues.append({
    "item": "scope",
    "issue_type": "info_raw_search_only",
    "issue_detail": "v78g searches raw/server files for mapping evidence. It does not modify mappings, run tracking, or train models.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

direct_pair_sources = int(pair_df["source_path"].nunique()) if len(pair_df) else 0

# We do NOT auto-apply anything here.
ready_for_manual_review = bool(hard_issue_count == 0)
ready_for_auto_mapping = False

manifest = {
    "version": "v78g_server_raw_camera_mapping_search",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "files_seen": int(files_seen),
    "files_scanned_text": int(files_scanned_text),
    "files_scanned_excel": int(files_scanned_excel),
    "files_skipped_large": int(files_skipped_large),
    "filename_hits": int(len(filename_df)),
    "text_hits": int(len(text_df)),
    "excel_hits": int(len(excel_df)),
    "pair_evidence_rows": int(len(pair_df)),
    "direct_pair_sources": int(direct_pair_sources),
    "possible_layout_files": int(len(layout_df)),
    "hard_issue_count": int(hard_issue_count),
    "claim_boundary": "raw server evidence search only; no mapping applied",
}

OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

readme = f"""# v78g Server Raw Camera Mapping Search

## Purpose

This stage searches raw/server-accessible files for evidence that maps Excel TLC camera names to encoded camera-code videos.

It is intentionally evidence-only:
- It does not apply mappings.
- It does not run tracking.
- It does not train a model.
- It does not guess unresolved TLC cameras.

## Counts

- Files seen: {files_seen}
- Text files scanned: {files_scanned_text}
- Excel files scanned: {files_scanned_excel}
- Filename hits: {len(filename_df)}
- Text hits: {len(text_df)}
- Excel hits: {len(excel_df)}
- TLC-code pair evidence rows: {len(pair_df)}
- Direct pair evidence sources: {direct_pair_sources}
- Possible layout/doc/image files: {len(layout_df)}
- Hard issues: {hard_issue_count}

## Next

Review:
1. `v78g_tlc_code_pair_evidence.csv`
2. `v78g_candidate_mapping_summary.csv`
3. `v78g_possible_layout_or_doc_files.csv`

If these files contain a reliable mapping, use it in v78f. If not, external supervisor/layout confirmation is required.
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
    "v78g_decision": "server_raw_camera_mapping_search_completed" if hard_issue_count == 0 else "server_raw_camera_mapping_search_has_blocking_issues",
    "files_seen": int(files_seen),
    "files_scanned_text": int(files_scanned_text),
    "files_scanned_excel": int(files_scanned_excel),
    "filename_hits": int(len(filename_df)),
    "text_hits": int(len(text_df)),
    "excel_hits": int(len(excel_df)),
    "tlc_code_pair_evidence_rows": int(len(pair_df)),
    "direct_pair_sources": int(direct_pair_sources),
    "possible_layout_files": int(len(layout_df)),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_quality_failures": int(hard_quality_failures),
    "hard_issue_count": int(hard_issue_count),
    "warning_count": int(warning_count),
    "info_count": int(info_count),
    "issue_count": int(len(issues_df)),
    "ready_for_manual_review": ready_for_manual_review,
    "ready_for_auto_mapping": ready_for_auto_mapping,
    "ready_for_v79_full_tracking_preparation": False,
    "claim_scope": "raw_server_mapping_evidence_search_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# v78g Server Raw Camera Mapping Search\n\n"
    f"- v78g decision: {decision.iloc[0]['v78g_decision']}\n"
    f"- Files seen: {files_seen}\n"
    f"- Text files scanned: {files_scanned_text}\n"
    f"- Excel files scanned: {files_scanned_excel}\n"
    f"- Filename hits: {len(filename_df)}\n"
    f"- Text hits: {len(text_df)}\n"
    f"- Excel hits: {len(excel_df)}\n"
    f"- TLC-code pair evidence rows: {len(pair_df)}\n"
    f"- Direct pair evidence sources: {direct_pair_sources}\n"
    f"- Possible layout/doc/image files: {len(layout_df)}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Ready for manual review: {ready_for_manual_review}\n"
    f"- Ready for auto mapping: {ready_for_auto_mapping}\n"
    f"- Ready for v79 full tracking preparation: False\n\n"
    "This stage searches raw/server files only and does not apply mappings.\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v78g",
    "task_name": "Server raw camera mapping evidence search",
    "status": "PASS_SEARCH_COMPLETED" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": "/work/pig/datasets + ~/PigBench raw/non-output files",
    "output_summary": str(PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Review raw mapping evidence and possible layout files; update v78f only with reliable confirmed mapping.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("=== v78g decision ===")
print(decision.to_string(index=False))

print("\n=== candidate mapping summary ===")
print(summary.to_string(index=False))

print("\n=== possible layout/doc/image files ===")
print(layout_df.head(80).to_string(index=False))

print("\n=== TLC-code pair evidence sample ===")
print(pair_df.head(80).to_string(index=False))

print("\n=== QA ===")
print(qa.to_string(index=False))

print("\n=== issues ===")
print(issues_df.to_string(index=False))
