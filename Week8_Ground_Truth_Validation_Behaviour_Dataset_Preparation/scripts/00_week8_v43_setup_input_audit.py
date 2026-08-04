from pathlib import Path
from datetime import datetime
import json
import csv
import hashlib
import pandas as pd


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

DIRS = [
    "inputs_audit",
    "outputs",
    "outputs/v43_setup_input_audit",
    "outputs/propagated_ground_truth",
    "outputs/identity_assignment",
    "outputs/scene_change",
    "scripts",
    "interface",
    "interface/static",
    "docs",
    "validation",
    "reports",
    "progress",
    "figures",
    "final_package",
    "notes",
]

for d in DIRS:
    (W8 / d).mkdir(parents=True, exist_ok=True)

OUT = W8 / "outputs" / "v43_setup_input_audit"
OUT.mkdir(parents=True, exist_ok=True)

OUT_CANDIDATES = OUT / "week8_v43_candidate_artifacts.csv"
OUT_SELECTED = OUT / "week8_v43_selected_inputs.json"
OUT_AUDIT = OUT / "week8_v43_input_audit_summary.csv"
OUT_DECISION = OUT / "week8_v43_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v43_issues.csv"
OUT_REPORT = OUT / "week8_v43_setup_input_audit_report.md"
OUT_PROGRESS = W8 / "progress" / "week8_experiment_progress_log.csv"
OUT_NOTE = W8 / "notes" / "week8_v43_setup_input_audit_notes.md"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def sha256_file(path):
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def quick_line_count(path):
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    if p.suffix.lower() not in [".csv", ".txt", ".md", ".json"]:
        return ""
    try:
        with open(p, "r", errors="ignore") as f:
            n = sum(1 for _ in f)
        if p.suffix.lower() == ".csv":
            return max(0, n - 1)
        return n
    except Exception:
        return ""


def find_matches(base, patterns):
    matches = []

    for pattern in patterns:
        pattern_path = base / pattern

        if any(ch in pattern for ch in ["*", "?", "["]):
            found = list(base.glob(pattern))
        else:
            found = [pattern_path] if pattern_path.exists() else []

        for p in found:
            if p.exists():
                matches.append(p)

    # unique + sort by modified time newest first
    unique = []
    seen = set()
    for p in matches:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            unique.append(p)

    unique.sort(key=lambda x: x.stat().st_mtime if x.exists() else 0, reverse=True)
    return unique


def path_info(p):
    p = Path(p)
    exists = p.exists()
    is_file = p.is_file() if exists else False
    is_dir = p.is_dir() if exists else False

    file_count = ""
    mp4_count = ""
    csv_count = ""
    image_count = ""

    if is_dir:
        files = [x for x in p.rglob("*") if x.is_file()]
        file_count = len(files)
        mp4_count = len([x for x in files if x.suffix.lower() == ".mp4"])
        csv_count = len([x for x in files if x.suffix.lower() == ".csv"])
        image_count = len([x for x in files if x.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    return {
        "exists": bool(exists),
        "is_file": bool(is_file),
        "is_dir": bool(is_dir),
        "size_bytes": p.stat().st_size if exists and is_file else "",
        "file_count_if_dir": file_count,
        "mp4_count_if_dir": mp4_count,
        "csv_count_if_dir": csv_count,
        "image_count_if_dir": image_count,
        "line_or_row_count_if_text": quick_line_count(p),
        "sha256_if_file": sha256_file(p) if is_file and p.stat().st_size < 300 * 1024 * 1024 else "",
        "modified_at": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds") if exists else "",
    }


artifact_specs = [
    {
        "key": "week7_root",
        "required": True,
        "base": ROOT,
        "patterns": ["Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"],
        "purpose": "Main Week 7 project directory.",
    },
    {
        "key": "raw_unibo_dataset",
        "required": False,
        "base": Path("/"),
        "patterns": ["work/pig/datasets/Unibo"],
        "purpose": "Original Unibo videos. Useful for full video interface, but v26 clips are enough for first interface MVP.",
    },
    {
        "key": "v11_corrected_boxes",
        "required": True,
        "base": W7,
        "patterns": [
            "outputs/final_corrected_gt_pen_boxes_v11/*for_colour_matching*.csv",
            "outputs/final_corrected_gt_pen_boxes_v11/*.csv",
        ],
        "purpose": "Final corrected pig boxes over annotated scanpoint frames.",
    },
    {
        "key": "v17_final_colour_identity",
        "required": True,
        "base": W7,
        "patterns": [
            "outputs/colour_identity/final_colour_identity_v17_fixed/*.csv",
            "outputs/colour_identity/final_colour_identity_v17_fixed/**/*.csv",
        ],
        "purpose": "Final colour identity lock / usable identity rows.",
    },
    {
        "key": "v18c_behaviour_fusion",
        "required": True,
        "base": W7,
        "patterns": [
            "outputs/behaviour_label_fusion_v18c_verified_crosswalk/*.csv",
            "outputs/behaviour_label_fusion_v18c_verified_crosswalk/**/*.csv",
        ],
        "purpose": "Verified colour-to-behaviour crosswalk and fused behaviour rows.",
    },
    {
        "key": "v19_final_fusion_stats",
        "required": False,
        "base": W7,
        "patterns": [
            "outputs/final_fusion_qa_dataset_stats_v19/*.csv",
            "outputs/final_fusion_qa_dataset_stats_v19/**/*.csv",
        ],
        "purpose": "Final fusion QA and dataset statistics.",
    },
    {
        "key": "v21_primary_split",
        "required": True,
        "base": W7,
        "patterns": [
            "outputs/primary_split_v21/*all*.csv",
            "outputs/primary_split_v21/*.csv",
        ],
        "purpose": "Primary train/val/test split.",
    },
    {
        "key": "v26_clip_index",
        "required": True,
        "base": W7,
        "patterns": [
            "outputs/clip_extraction_temporal_qa_v26/*clip*index*.csv",
            "outputs/clip_extraction_temporal_qa_v26/*.csv",
        ],
        "purpose": "10-second clip extraction index.",
    },
    {
        "key": "v26_clip_directory",
        "required": True,
        "base": W7,
        "patterns": [
            "outputs/clip_extraction_temporal_qa_v26/clips",
        ],
        "purpose": "Extracted 10-second clips.",
    },
    {
        "key": "v34b_clip_level_index",
        "required": True,
        "base": W7,
        "patterns": [
            "outputs/week7_baseline_behaviour_representation_dataset_v34b_split_fixed/*clip*index*split*.csv",
            "outputs/week7_baseline_behaviour_representation_dataset_v34b_split_fixed/*.csv",
        ],
        "purpose": "Split-fixed clip-level behaviour index for propagation and v38/v39 continuity.",
    },
    {
        "key": "v28d_dense_polygon_tracking",
        "required": True,
        "base": W7,
        "patterns": [
            "outputs/dense_polygon_filtered_tracking_v28d/*.csv",
            "outputs/dense_polygon_filtered_tracking_v28d/**/*.csv",
        ],
        "purpose": "Dense polygon-filtered tracking output for colour+tracking identity work.",
    },
    {
        "key": "v29b_colour_constrained_linking",
        "required": True,
        "base": W7,
        "patterns": [
            "outputs/colour_constrained_tracklet_linking_v29b/*.csv",
            "outputs/colour_constrained_tracklet_linking_v29b/**/*.csv",
        ],
        "purpose": "Colour-constrained tracklet identity linking prototype.",
    },
    {
        "key": "v29c_full_tracklet_colour_evidence",
        "required": False,
        "base": W7,
        "patterns": [
            "outputs/full_tracklet_colour_evidence_v29c/*.csv",
            "outputs/full_tracklet_colour_evidence_v29c/**/*.csv",
        ],
        "purpose": "Full-tracklet HSV colour evidence.",
    },
    {
        "key": "v29d_conservative_identity_arbitration",
        "required": True,
        "base": W7,
        "patterns": [
            "outputs/conservative_identity_arbitration_v29d/*fixed*.csv",
            "outputs/conservative_identity_arbitration_v29d/*.csv",
            "outputs/conservative_identity_arbitration_v29d/**/*.csv",
        ],
        "purpose": "Conservative identity arbitration and review queue.",
    },
    {
        "key": "v29e_final_identity_linking_package",
        "required": False,
        "base": W7,
        "patterns": [
            "outputs/final_identity_linking_report_package_v29e/**/*.csv",
            "outputs/final_identity_linking_report_package_v29e/**/*.md",
            "outputs/final_identity_linking_report_package_v29e/**/*.zip",
        ],
        "purpose": "Final identity linking evidence package.",
    },
    {
        "key": "v40_behaviour_temporal_evidence",
        "required": False,
        "base": W7,
        "patterns": [
            "outputs/week7_behaviour_temporal_evidence_package_v40/Week7_Behaviour_Temporal_Evidence_Package_v40.zip",
            "outputs/week7_behaviour_temporal_evidence_package_v40/*.csv",
            "outputs/week7_behaviour_temporal_evidence_package_v40/*.md",
        ],
        "purpose": "Week 7 behaviour temporal evidence package.",
    },
    {
        "key": "v41b_final_report_ready_package",
        "required": False,
        "base": W7,
        "patterns": [
            "outputs/week7_final_report_ready_summary_visual_package_v41/Week7_Final_Report_Ready_Summary_Visual_Package_v41b_FIXED.zip",
        ],
        "purpose": "Week 7 final fixed report-ready package.",
    },
    {
        "key": "v42_final_audit",
        "required": False,
        "base": W7,
        "patterns": [
            "outputs/week7_final_audit_delivery_verification_v42/week7_v42_final_audit_delivery_decision.csv",
            "outputs/week7_final_audit_delivery_verification_v42/Week7_Final_Delivery_Verification_v42.zip",
        ],
        "purpose": "Week 7 final delivery audit verification.",
    },
]

candidate_rows = []
audit_rows = []
selected_inputs = {
    "stage": "week8_v43_setup_input_audit",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "root": str(ROOT),
    "week7_root": str(W7),
    "week8_root": str(W8),
    "selected_inputs": {},
}

issues = []

for spec in artifact_specs:
    key = spec["key"]
    matches = find_matches(spec["base"], spec["patterns"])

    selected = matches[0] if matches else None

    audit_rows.append({
        "key": key,
        "required": bool(spec["required"]),
        "match_count": int(len(matches)),
        "selected_path": str(selected) if selected else "",
        "purpose": spec["purpose"],
        "status": "FOUND" if selected else ("MISSING_REQUIRED" if spec["required"] else "MISSING_OPTIONAL"),
    })

    if selected:
        selected_inputs["selected_inputs"][key] = {
            "path": str(selected),
            "required": bool(spec["required"]),
            "purpose": spec["purpose"],
        }

    if not selected and spec["required"]:
        issues.append({
            "item": key,
            "issue_type": "hard_missing_required_input",
            "issue_detail": spec["purpose"],
        })

    if not selected and not spec["required"]:
        issues.append({
            "item": key,
            "issue_type": "warning_missing_optional_input",
            "issue_detail": spec["purpose"],
        })

    if len(matches) > 1:
        issues.append({
            "item": key,
            "issue_type": "warning_multiple_candidate_inputs",
            "issue_detail": f"{len(matches)} candidates found; newest modified selected.",
        })

    for idx, p in enumerate(matches):
        info = path_info(p)
        candidate_rows.append({
            "key": key,
            "candidate_rank": int(idx + 1),
            "selected": bool(idx == 0),
            "required": bool(spec["required"]),
            "path": str(p),
            "purpose": spec["purpose"],
            **info,
        })

candidate_df = pd.DataFrame(candidate_rows)
audit_df = pd.DataFrame(audit_rows)
issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])

safe_to_csv(candidate_df, OUT_CANDIDATES)
safe_to_csv(audit_df, OUT_AUDIT)
safe_to_csv(issues_df, OUT_ISSUES)

OUT_SELECTED.write_text(json.dumps(selected_inputs, indent=2))

hard_issues = [x for x in issues if str(x["issue_type"]).startswith("hard_")]
warnings = [x for x in issues if str(x["issue_type"]).startswith("warning_")]

required_total = sum(1 for x in artifact_specs if x["required"])
required_found = sum(
    1 for row in audit_rows
    if row["required"] and row["status"] == "FOUND"
)
optional_total = sum(1 for x in artifact_specs if not x["required"])
optional_found = sum(
    1 for row in audit_rows
    if (not row["required"]) and row["status"] == "FOUND"
)

ready_for_v44 = len(hard_issues) == 0

decision = pd.DataFrame([{
    "v43_decision": "week8_setup_input_audit_passed" if ready_for_v44 else "week8_setup_input_audit_blocked",
    "week8_root": str(W8),
    "required_inputs_total": int(required_total),
    "required_inputs_found": int(required_found),
    "optional_inputs_total": int(optional_total),
    "optional_inputs_found": int(optional_found),
    "candidate_artifact_rows": int(len(candidate_df)),
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "issue_count": int(len(issues_df)),
    "ready_for_v44_ground_truth_schema": bool(ready_for_v44),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

# Initial progress log
progress_exists = OUT_PROGRESS.exists()
progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v43",
    "task_name": "Week 8 setup and input audit",
    "status": "PASS" if ready_for_v44 else "BLOCKED",
    "input_summary": "Week 7 outputs and raw dataset locations scanned.",
    "output_summary": str(OUT),
    "hard_issues": int(len(hard_issues)),
    "warnings": int(len(warnings)),
    "next_action": "v44 ground-truth schema and field dictionary" if ready_for_v44 else "Resolve missing required inputs.",
}])

if progress_exists:
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

def markdown_table(df, max_rows=50):
    if df is None or len(df) == 0:
        return "_No rows._"
    rows = df.head(max_rows).to_dict("records")
    cols = list(df.columns)
    out = []
    out.append("| " + " | ".join(cols) + " |")
    out.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for r in rows:
        vals = []
        for c in cols:
            x = str(r.get(c, ""))
            x = x.replace("\n", " ").replace("|", "/")
            vals.append(x)
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)

report = f"""# Week 8 v43 Setup and Input Audit Report

## Purpose

This stage prepares the Week 8 professional workflow by creating the working directory structure and auditing the Week 7 outputs needed for ground-truth validation, behaviour label propagation, visualization interface development, and identity assignment improvement.

## Decision

- v43 decision: `{decision.iloc[0]['v43_decision']}`
- Required inputs found: `{required_found}` / `{required_total}`
- Optional inputs found: `{optional_found}` / `{optional_total}`
- Hard issue count: `{len(hard_issues)}`
- Warning count: `{len(warnings)}`
- Ready for v44 ground-truth schema: `{ready_for_v44}`

## Input audit summary

{markdown_table(audit_df)}

## Interpretation

If all required inputs are present, Week 8 can proceed to ground-truth schema design and behaviour label propagation. Optional missing inputs should be documented but should not block v44 unless they are needed for a specific later task.

## Next step

v44 should define the Week 8 ground-truth JSON schema, field dictionary, annotation rules, behaviour labels, colour identities, timestamp convention, and validation flags.
"""

OUT_REPORT.write_text(report)

OUT_NOTE.write_text(
    "# Week 8 v43 Setup and Input Audit\n\n"
    "## Summary\n\n"
    f"- Week 8 root: `{W8}`\n"
    f"- Required inputs found: `{required_found}` / `{required_total}`\n"
    f"- Optional inputs found: `{optional_found}` / `{optional_total}`\n"
    f"- Hard issue count: `{len(hard_issues)}`\n"
    f"- Warning count: `{len(warnings)}`\n"
    f"- Ready for v44 ground-truth schema: `{ready_for_v44}`\n\n"
    "## Main outputs\n\n"
    f"- Candidate artifacts: `{OUT_CANDIDATES}`\n"
    f"- Input audit summary: `{OUT_AUDIT}`\n"
    f"- Selected inputs JSON: `{OUT_SELECTED}`\n"
    f"- Decision summary: `{OUT_DECISION}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
    f"- Report: `{OUT_REPORT}`\n"
    f"- Progress log: `{OUT_PROGRESS}`\n"
)

print("Saved:")
print(OUT_CANDIDATES)
print(OUT_AUDIT)
print(OUT_SELECTED)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_REPORT)
print(OUT_PROGRESS)
print(OUT_NOTE)

print()
print("=== v43 decision ===")
print(decision.to_string(index=False))

print()
print("=== v43 issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
