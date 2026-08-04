from pathlib import Path
from datetime import datetime
import re
import csv
import json
import hashlib
import zipfile
import pandas as pd


ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"

V78C_TEMPLATE = FULL / "config" / "camera_code_mapping_TEMPLATE_TO_FILL.csv"
CURRENT_MAPPING = FULL / "config" / "camera_code_mapping.csv"
CURRENT_CAMERA_MAPPING = FULL / "config" / "camera_level_code_mapping.csv"

OUT = FULL / "outputs" / "v78e_evidence_based_camera_code_mapping_audit"
PKG = OUT / "Full_Unibo_Evidence_Based_Camera_Code_Mapping_Audit"
NOTES = FULL / "notes"
REPORTS = FULL / "reports"
PROGRESS = FULL / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_RAW_EVIDENCE = PKG / "v78e_raw_camera_code_tlc_evidence.csv"
OUT_SUMMARY = PKG / "v78e_camera_code_tlc_evidence_summary.csv"
OUT_PROPOSED = PKG / "v78e_proposed_camera_code_mapping.csv"
OUT_UNRESOLVED = PKG / "v78e_unresolved_camera_mapping_questions.csv"
OUT_CONFLICTS = PKG / "v78e_camera_mapping_conflicts.csv"
OUT_CURRENT = PKG / "v78e_current_mapping_snapshot.csv"
OUT_QA = PKG / "v78e_quality_checks.csv"
OUT_README = PKG / "README_v78e_Evidence_Based_Camera_Code_Mapping_Audit.md"
OUT_MANIFEST = PKG / "v78e_manifest.json"

OUT_DECISION = OUT / "v78e_decision_summary.csv"
OUT_ISSUES = OUT / "v78e_issues.csv"
OUT_ZIP = OUT / "Full_Unibo_Evidence_Based_Camera_Code_Mapping_Audit.zip"
OUT_SHA = OUT / "Full_Unibo_Evidence_Based_Camera_Code_Mapping_Audit.sha256"
OUT_NOTE = NOTES / "v78e_evidence_based_camera_code_mapping_audit_notes.md"
OUT_REPORT = REPORTS / "v78e_evidence_based_camera_code_mapping_audit_report.md"
OUT_PROGRESS = PROGRESS / "full_unibo_pipeline_progress_log.csv"

CODE_RE = re.compile(r"\b(c\d{4})\d{10,}\.mp4\b", re.IGNORECASE)
CODE_ONLY_RE = re.compile(r"\b(c\d{4})\b", re.IGNORECASE)
TLC_RE = re.compile(r"\bTLC\s*([1-6])\b", re.IGNORECASE)

EXCLUDED_PATH_HINTS = [
    "v78b_mapping_candidates_long",
    "v78b_unresolved_annotation_windows",
    "v78c_clean_camera_code_mapping_template",
    "camera_code_mapping_TEMPLATE_TO_FILL",
    "v78c_visual_atlas_manifest",
    "v78c_unresolved_mapping_groups",
    "v78c_camera_code_mapping_resolver",
    "v78d_tlc1_anchor_mapping_interface/static",
]

INCLUDED_EXTS = {".csv", ".txt", ".md", ".json", ".jsonl"}


def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def read_csv_clean(path):
    df = pd.read_csv(path).fillna("")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(clean)
    return df


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_weight(path_text, text):
    low = (path_text + " " + text).lower()
    w = 1

    if any(k in low for k in ["manual", "review", "reviewed", "validated", "final_gt", "strict", "gold"]):
        w += 5

    if any(k in low for k in ["week8", "v64", "v65", "v66", "v67", "v68"]):
        w += 3

    if any(k in low for k in ["current_mapping", "camera_level_code_mapping", "camera_code_mapping.csv"]):
        w += 6

    if any(k in low for k in ["candidate", "template", "unresolved", "mapping_candidates"]):
        w -= 4

    return max(w, 0)


def parse_codes(text):
    codes = set(m.group(1).lower() for m in CODE_RE.finditer(text))
    codes.update(m.group(1).lower() for m in CODE_ONLY_RE.finditer(text))
    return sorted(codes)


def parse_tlcs(text):
    return sorted(set(f"TLC{m.group(1)}" for m in TLC_RE.finditer(text)))


def should_skip(path):
    s = str(path)
    if path.suffix.lower() not in INCLUDED_EXTS:
        return True

    if any(h in s for h in EXCLUDED_PATH_HINTS):
        return True

    try:
        if path.stat().st_size > 50 * 1024 * 1024:
            return True
    except Exception:
        return True

    return False


def add_evidence(rows, source_path, source_kind, text, row_id=""):
    codes = parse_codes(text)
    tlcs = parse_tlcs(text)

    if not codes or not tlcs:
        return

    weight = source_weight(str(source_path), text)

    if weight <= 0:
        return

    snippet = text.replace("\n", " ")
    if len(snippet) > 800:
        snippet = snippet[:800] + "..."

    for code in codes:
        for tlc in tlcs:
            rows.append({
                "source_path": str(source_path),
                "source_kind": source_kind,
                "row_id": row_id,
                "tlc_camera": tlc,
                "video_camera_code": code,
                "evidence_weight": weight,
                "evidence_snippet": snippet,
            })


issues = []

for p in [V78C_TEMPLATE]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "v78e requires v78c template.",
            "severity": "hard",
        })

if issues:
    safe_to_csv(pd.DataFrame(issues), OUT_ISSUES)
    raise SystemExit("Missing required input.")

evidence_rows = []

# Current confirmed mapping is valid evidence, because it came from the manual visual TLC1 anchor.
if CURRENT_CAMERA_MAPPING.exists():
    current_cam = read_csv_clean(CURRENT_CAMERA_MAPPING)
    for i, r in current_cam.iterrows():
        date = clean(r.get("date", ""))
        cam = clean(r.get("camera", ""))
        code = clean(r.get("video_camera_code", ""))
        if date and cam and code:
            add_evidence(
                evidence_rows,
                CURRENT_CAMERA_MAPPING,
                "current_manual_camera_level_mapping",
                f"{date} {cam} {code} {clean(r.get('method',''))} {clean(r.get('note',''))}",
                row_id=str(i),
            )

if CURRENT_MAPPING.exists():
    current_map = read_csv_clean(CURRENT_MAPPING)
    snapshot = current_map.copy()
    safe_to_csv(snapshot, OUT_CURRENT)
else:
    current_map = pd.DataFrame()
    safe_to_csv(pd.DataFrame(), OUT_CURRENT)

# Search previous pipeline outputs for actual evidence.
search_roots = [
    ROOT / "Week6_Unibo_Dataset_Validation",
    ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation",
    ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation",
    FULL,
]

files_scanned = 0
files_with_evidence = set()

for base in search_roots:
    if not base.exists():
        continue

    for path in base.rglob("*"):
        if should_skip(path):
            continue

        files_scanned += 1

        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path, dtype=str, nrows=200000).fillna("")
                for idx, row in df.iterrows():
                    text = " ; ".join(clean(v) for v in row.values)
                    before = len(evidence_rows)
                    add_evidence(evidence_rows, path, "csv_row", text, row_id=str(idx))
                    if len(evidence_rows) > before:
                        files_with_evidence.add(str(path))
            else:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for line_no, line in enumerate(f, start=1):
                        before = len(evidence_rows)
                        add_evidence(evidence_rows, path, "text_line", line, row_id=str(line_no))
                        if len(evidence_rows) > before:
                            files_with_evidence.add(str(path))
        except Exception as e:
            issues.append({
                "item": str(path),
                "issue_type": "warning_scan_failed",
                "issue_detail": str(e)[:500],
                "severity": "warning",
            })

raw = pd.DataFrame(evidence_rows)

if len(raw):
    raw = raw.drop_duplicates()
else:
    raw = pd.DataFrame(columns=[
        "source_path", "source_kind", "row_id", "tlc_camera",
        "video_camera_code", "evidence_weight", "evidence_snippet"
    ])

safe_to_csv(raw, OUT_RAW_EVIDENCE)

if len(raw):
    summary = (
        raw.groupby(["tlc_camera", "video_camera_code"])
        .agg(
            evidence_rows=("source_path", "count"),
            evidence_weight_sum=("evidence_weight", "sum"),
            source_count=("source_path", "nunique"),
            example_sources=("source_path", lambda x: ";".join(sorted(set(x))[:10])),
            example_snippets=("evidence_snippet", lambda x: " || ".join(list(x)[:3])),
        )
        .reset_index()
        .sort_values(["tlc_camera", "evidence_weight_sum"], ascending=[True, False])
    )
else:
    summary = pd.DataFrame(columns=[
        "tlc_camera", "video_camera_code", "evidence_rows", "evidence_weight_sum",
        "source_count", "example_sources", "example_snippets"
    ])

safe_to_csv(summary, OUT_SUMMARY)

template = read_csv_clean(V78C_TEMPLATE)
camera_targets = (
    template[template["candidate_video_camera_codes"].astype(str).str.len() > 0]
    .groupby(["date", "camera"])
    .agg(
        pens=("pen", lambda x: ";".join(sorted(set(map(str, x))))),
        candidate_codes=("candidate_video_camera_codes", lambda x: ";".join(sorted(set(";".join(x).split(";"))))),
        unresolved_windows=("unresolved_windows", lambda x: sum(int(float(v)) if clean(v) else 0 for v in x)),
    )
    .reset_index()
    .sort_values(["date", "camera"])
)

proposed_rows = []
conflict_rows = []
unresolved_rows = []

for _, target in camera_targets.iterrows():
    date = clean(target["date"])
    camera = clean(target["camera"])
    candidate_codes = [c for c in clean(target["candidate_codes"]).split(";") if c]

    evid = summary[summary["tlc_camera"] == camera].copy()

    # Only date-specific direct evidence is weakly available from text snippets.
    # We keep the date in the proposed row, but evidence itself is TLC/code-level.
    if len(evid) == 0:
        status = "unresolved_no_evidence"
        selected_code = ""
        reason = "No reliable evidence found in previous outputs."
    else:
        evid = evid.sort_values("evidence_weight_sum", ascending=False)
        top = evid.iloc[0]
        top_code = clean(top["video_camera_code"])
        top_weight = int(top["evidence_weight_sum"])

        second_weight = 0
        second_code = ""

        if len(evid) > 1:
            second = evid.iloc[1]
            second_weight = int(second["evidence_weight_sum"])
            second_code = clean(second["video_camera_code"])

        if top_code in candidate_codes and top_weight >= 6 and top_weight > second_weight:
            status = "proposed_from_evidence"
            selected_code = top_code
            reason = f"Top evidence code {top_code}, weight={top_weight}, second={second_code}:{second_weight}"
        elif top_code in candidate_codes and top_weight >= 6 and top_weight == second_weight:
            status = "conflict_equal_evidence"
            selected_code = ""
            reason = f"Equal/ambiguous evidence: top={top_code}:{top_weight}, second={second_code}:{second_weight}"
        else:
            status = "unresolved_evidence_not_strong_enough"
            selected_code = ""
            reason = f"Evidence exists but not enough to select automatically. top={top_code}:{top_weight}, second={second_code}:{second_weight}"

    row = {
        "date": date,
        "camera": camera,
        "pens": clean(target["pens"]),
        "candidate_codes": ";".join(candidate_codes),
        "selected_video_camera_code": selected_code,
        "proposal_status": status,
        "reason": reason,
        "unresolved_windows": int(target["unresolved_windows"]) if clean(target["unresolved_windows"]) else 0,
    }

    proposed_rows.append(row)

    if status.startswith("conflict"):
        conflict_rows.append(row)

    if not selected_code:
        unresolved_rows.append(row)

proposed = pd.DataFrame(proposed_rows)
conflicts = pd.DataFrame(conflict_rows)
unresolved = pd.DataFrame(unresolved_rows)

safe_to_csv(proposed, OUT_PROPOSED)
safe_to_csv(conflicts, OUT_CONFLICTS)
safe_to_csv(unresolved, OUT_UNRESOLVED)

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

evidence_count = len(raw)
summary_count = len(summary)
target_count = len(camera_targets)
proposed_count = int((proposed["selected_video_camera_code"].astype(str).str.len() > 0).sum()) if len(proposed) else 0
unresolved_count = len(unresolved)
conflict_count = len(conflicts)

add_qa("files_scanned", ">0", files_scanned, files_scanned > 0, "hard", "Previous outputs should be scanned.")
add_qa("evidence_rows_created", ">=1", evidence_count, evidence_count >= 1, "hard", "At least current TLC1 manual evidence should exist.")
add_qa("camera_targets_created", ">0", target_count, target_count > 0, "hard", "Camera targets should be created from v78c template.")
add_qa("at_least_one_mapping_proposed", ">=1", proposed_count, proposed_count >= 1, "hard", "At least TLC1 should be proposed from current anchor/evidence.")
add_qa("conflicts_reported", ">=0", conflict_count, conflict_count >= 0, "info", "Conflicts are reported instead of hidden.")
add_qa("unresolved_reported", ">=0", unresolved_count, unresolved_count >= 0, "info", "Unresolved cameras are reported explicitly.")

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_QA)

hard_quality_failures = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v78e_quality_checks",
        "issue_type": "hard_evidence_audit_failed",
        "issue_detail": f"{hard_quality_failures} hard QA checks failed.",
        "severity": "hard",
    })

if conflict_count > 0:
    issues.append({
        "item": "camera_mapping_conflicts",
        "issue_type": "warning_conflicting_camera_code_evidence",
        "issue_detail": f"{conflict_count} target cameras have conflicting/equal evidence.",
        "severity": "warning",
    })

if unresolved_count > 0:
    issues.append({
        "item": "unresolved_camera_mapping",
        "issue_type": "info_unresolved_camera_mapping_requires_reference_or_manual_confirmation",
        "issue_detail": f"{unresolved_count} camera/date targets still need reference/manual confirmation.",
        "severity": "info",
    })

issues.append({
    "item": "scope",
    "issue_type": "info_evidence_audit_only",
    "issue_detail": "v78e audits evidence and proposes mappings. It does not run tracking/model training.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum())
warning_count = int((issues_df["severity"] == "warning").sum()) if "warning" in set(issues_df["severity"]) else 0
info_count = int((issues_df["severity"] == "info").sum())

all_targets_resolved = bool(target_count > 0 and unresolved_count == 0 and conflict_count == 0 and hard_issue_count == 0)

manifest = {
    "version": "v78e_evidence_based_camera_code_mapping_audit",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "files_scanned": int(files_scanned),
    "files_with_evidence": int(len(files_with_evidence)),
    "raw_evidence_rows": int(evidence_count),
    "summary_rows": int(summary_count),
    "camera_targets": int(target_count),
    "proposed_mappings": int(proposed_count),
    "unresolved_targets": int(unresolved_count),
    "conflict_targets": int(conflict_count),
    "hard_issue_count": int(hard_issue_count),
    "all_targets_resolved": all_targets_resolved,
    "claim_boundary": "evidence audit and mapping proposals only; no tracking/model",
}

OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

readme = f"""# v78e Evidence-Based Camera-Code Mapping Audit

## Purpose

This stage searches previous Week6/Week7/Week8/Full pipeline outputs for evidence that connects encoded video camera codes such as `c0002` to Excel/TLC camera names such as `TLC1`.

It avoids forced assumptions. If evidence is missing or conflicting, it reports the unresolved item explicitly.

## Main counts

- Files scanned: {files_scanned}
- Files with evidence: {len(files_with_evidence)}
- Raw evidence rows: {evidence_count}
- Evidence summary rows: {summary_count}
- Camera/date targets: {target_count}
- Proposed mappings: {proposed_count}
- Unresolved targets: {unresolved_count}
- Conflict targets: {conflict_count}
- Hard issues: {hard_issue_count}
- All targets resolved: {all_targets_resolved}

## Next step

If unresolved targets remain, use the unresolved table as the authoritative checklist for manual/supervisor confirmation. Do not proceed to full tracking until the mapping is resolved or the scope is explicitly reduced.
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
    "v78e_decision": "evidence_based_camera_mapping_audit_completed" if hard_issue_count == 0 else "evidence_based_camera_mapping_audit_has_blocking_issues",
    "files_scanned": int(files_scanned),
    "files_with_evidence": int(len(files_with_evidence)),
    "raw_evidence_rows": int(evidence_count),
    "camera_targets": int(target_count),
    "proposed_mappings": int(proposed_count),
    "unresolved_targets": int(unresolved_count),
    "conflict_targets": int(conflict_count),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_quality_failures": int(hard_quality_failures),
    "hard_issue_count": int(hard_issue_count),
    "warning_count": int(warning_count),
    "info_count": int(info_count),
    "issue_count": int(len(issues_df)),
    "all_camera_targets_resolved": all_targets_resolved,
    "ready_for_v78f_mapping_completion": bool(hard_issue_count == 0 and not all_targets_resolved),
    "ready_for_v79_full_tracking_preparation": all_targets_resolved,
    "claim_scope": "evidence_audit_and_mapping_proposals_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# v78e Evidence-Based Camera-Code Mapping Audit\n\n"
    f"- v78e decision: {decision.iloc[0]['v78e_decision']}\n"
    f"- Files scanned: {files_scanned}\n"
    f"- Files with evidence: {len(files_with_evidence)}\n"
    f"- Raw evidence rows: {evidence_count}\n"
    f"- Camera targets: {target_count}\n"
    f"- Proposed mappings: {proposed_count}\n"
    f"- Unresolved targets: {unresolved_count}\n"
    f"- Conflict targets: {conflict_count}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- All camera targets resolved: {all_targets_resolved}\n"
    f"- Ready for v78f mapping completion: {bool(hard_issue_count == 0 and not all_targets_resolved)}\n"
    f"- Ready for v79 full tracking preparation: {all_targets_resolved}\n\n"
    "This stage does not run tracking. It turns the remaining mapping problem into an auditable evidence table and unresolved checklist.\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v78e",
    "task_name": "Evidence-based camera-code mapping audit",
    "status": "PASS_ALL_RESOLVED" if all_targets_resolved else ("PASS_NEEDS_MAPPING_COMPLETION" if hard_issue_count == 0 else "NEEDS_FIX"),
    "input_summary": str(V78C_TEMPLATE),
    "output_summary": str(PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Proceed to v79 full tracking preparation." if all_targets_resolved else "Complete unresolved camera mappings using v78e unresolved checklist.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("=== v78e decision ===")
print(decision.to_string(index=False))

print("\n=== proposed mappings ===")
print(proposed.to_string(index=False))

print("\n=== evidence summary ===")
print(summary.to_string(index=False))

print("\n=== unresolved mapping questions ===")
print(unresolved.to_string(index=False))

print("\n=== conflicts ===")
print(conflicts.to_string(index=False))

print("\n=== QA ===")
print(qa.to_string(index=False))

print("\n=== issues ===")
print(issues_df.to_string(index=False))
