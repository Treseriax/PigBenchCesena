from pathlib import Path
from datetime import datetime
import json
import csv
import hashlib
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V43B_JSON = W8 / "outputs" / "v43_setup_input_audit" / "week8_v43b_canonical_selected_inputs.json"

OUT = W8 / "outputs" / "v43_setup_input_audit"
OUT_FINAL_JSON = OUT / "week8_v43c_final_locked_inputs.json"
OUT_AUDIT = OUT / "week8_v43c_final_locked_input_audit.csv"
OUT_DECISION = OUT / "week8_v43c_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v43c_issues.csv"
OUT_REPORT = OUT / "week8_v43c_final_input_lock_report.md"
OUT_NOTE = W8 / "notes" / "week8_v43c_final_input_lock_notes.md"
OUT_PROGRESS = W8 / "progress" / "week8_experiment_progress_log.csv"


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


def row_count(path):
    try:
        with open(path, "r", errors="ignore") as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return ""


def cols(path):
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def has_cols(path, required):
    c = [str(x).lower() for x in cols(path)]
    return all(any(r.lower() in x for x in c) for r in required)


def read_json(path):
    return json.loads(Path(path).read_text())


def file_info(path):
    p = Path(path)
    exists = p.exists()
    return {
        "path": str(p),
        "exists": bool(exists),
        "row_count": row_count(p) if exists and p.suffix.lower() == ".csv" else "",
        "column_count": len(cols(p)) if exists and p.suffix.lower() == ".csv" else "",
        "columns_preview": " | ".join(cols(p)[:50]) if exists and p.suffix.lower() == ".csv" else "",
        "size_bytes": p.stat().st_size if exists and p.is_file() else "",
        "sha256": sha256_file(p) if exists and p.is_file() and p.stat().st_size < 250 * 1024 * 1024 else "",
    }


data = read_json(V43B_JSON)
ci = data["canonical_inputs"]

issues = []
audit_rows = []

# Start from v43b canonical inputs.
locked = {
    "stage": "week8_v43c_final_input_lock",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "week8_root": str(W8),
    "input_manifest_source": str(V43B_JSON),
    "locked_inputs": {},
    "lock_rules": {
        "box_level_identity_source": "Use behaviour_fusion_box_level if standalone v17 identity file is frame-level QA.",
        "training_ready_source": "Use primary_split_all_rows as the 374-row model/training-ready set.",
        "propagation_source": "Use behaviour_fusion_box_level for all 429 box-level objects and clip_extraction_index for all 72 clips.",
        "visualizer_source": "Use clip_extraction_index + propagated ground truth produced in v45.",
        "tracking_identity_source": "Use v28d/v29b/v29c/v29d for tracklet-level identity evidence.",
    }
}

def add_locked(key, source_key, role, purpose, note=""):
    item = dict(ci[source_key])
    item["source_key"] = source_key
    item["role"] = role
    item["purpose_week8"] = purpose
    item["lock_note"] = note
    locked["locked_inputs"][key] = item

    info = file_info(item["path"])
    audit_rows.append({
        "locked_key": key,
        "source_key": source_key,
        "role": role,
        "status": "LOCKED" if info["exists"] else "MISSING",
        "purpose_week8": purpose,
        "lock_note": note,
        **info,
    })

    if not info["exists"]:
        issues.append({
            "item": key,
            "issue_type": "hard_locked_input_missing",
            "issue_detail": item["path"],
        })


# Core box/behaviour propagation inputs.
add_locked(
    "corrected_boxes_box_level_429",
    "corrected_boxes_box_level",
    "box_geometry_reference",
    "Final corrected pig boxes at scanpoint frames.",
)

add_locked(
    "behaviour_fusion_box_level_429",
    "behaviour_fusion_box_level",
    "box_identity_behaviour_reference",
    "Main 429-row box-level table containing geometry, colour identity, pig ID and behaviour labels.",
)

# Final colour identity: v43b standalone v17 selected frame QA, so use box-level fusion as effective box-level identity source.
v17_path = Path(ci["final_colour_identity_box_level"]["path"])
v17_rows = row_count(v17_path)
v17_is_frame_qa = "frame_qa" in v17_path.name.lower() or str(v17_rows) == "72"

if v17_is_frame_qa:
    add_locked(
        "final_colour_identity_effective_box_level_429",
        "behaviour_fusion_box_level",
        "effective_box_level_colour_identity",
        "Box-level final colour identity columns are taken from the v18c fused box-level dataset because the standalone v17 selected file is frame-level QA.",
        note=f"v43b standalone v17 file was `{v17_path.name}` with {v17_rows} rows; kept as QA, not as box-level identity table.",
    )
    add_locked(
        "final_colour_identity_frame_qa_72",
        "final_colour_identity_box_level",
        "frame_level_colour_identity_QA",
        "Frame-level QA summary for colour identity completeness and soft/hard issues.",
        note="QA/supporting input only.",
    )
    issues.append({
        "item": "final_colour_identity_box_level",
        "issue_type": "info_reclassified_v17_frame_qa",
        "issue_detail": "v43b selected v17 frame_qa.csv; v43c uses behaviour_fusion_box_level as effective box-level identity source.",
    })
else:
    add_locked(
        "final_colour_identity_box_level",
        "final_colour_identity_box_level",
        "box_level_colour_identity",
        "Standalone box-level final colour identity assignments.",
    )

# Training/model-ready canonical rows should be 374 primary split.
add_locked(
    "training_ready_primary_split_374",
    "primary_split_all_rows",
    "training_ready_behaviour_rows",
    "374-row training/model-ready pig-level dataset with primary split labels.",
    note="This is the canonical training-ready source, replacing the v43b behaviour_training_ready_rows alias that pointed to the 429-row box-level table.",
)

# Clip inputs.
add_locked(
    "clip_extraction_index_72",
    "clip_extraction_index",
    "clip_interval_reference",
    "72 extracted 10-second clip records with paths, timing and scanframe mapping.",
)

add_locked(
    "clip_level_multilabel_index_72",
    "clip_level_multilabel_split_fixed_index",
    "clip_level_multilabel_reference",
    "72-row clip-level behaviour index with split and multi-label behaviour columns.",
)

# Tracking / identity evidence.
add_locked(
    "dense_polygon_tracking_rows_987",
    "dense_polygon_tracking_rows",
    "tracking_detections_reference",
    "Dense polygon-filtered tracking rows for selected tracking dry-run clips.",
)

add_locked(
    "colour_constrained_tracklet_candidates_50",
    "colour_constrained_tracklet_assignments",
    "tracklet_identity_candidate_reference",
    "Colour-constrained tracklet identity candidates.",
)

add_locked(
    "full_tracklet_colour_evidence_50",
    "full_tracklet_colour_evidence",
    "secondary_tracklet_colour_evidence",
    "Full-tracklet colour evidence used as secondary support only.",
)

add_locked(
    "conservative_identity_arbitration_50",
    "conservative_identity_arbitration_rows",
    "conservative_tracklet_identity_reference",
    "Conservative identity arbitration output; v29c does not overwrite v29b.",
)

add_locked(
    "identity_review_queue_41",
    "identity_review_queue",
    "identity_review_reference",
    "Tracklet identity review queue.",
)

# Sanity checks
bf_path = Path(locked["locked_inputs"]["behaviour_fusion_box_level_429"]["path"])
split_path = Path(locked["locked_inputs"]["training_ready_primary_split_374"]["path"])
clip_path = Path(locked["locked_inputs"]["clip_extraction_index_72"]["path"])
clip_level_path = Path(locked["locked_inputs"]["clip_level_multilabel_index_72"]["path"])

checks = [
    ("behaviour_fusion_box_level_429_rows", row_count(bf_path), "429"),
    ("training_ready_primary_split_374_rows", row_count(split_path), "374"),
    ("clip_extraction_index_72_rows", row_count(clip_path), "72"),
    ("clip_level_multilabel_index_72_rows", row_count(clip_level_path), "72"),
]

for item, got, expected in checks:
    if str(got) != str(expected):
        issues.append({
            "item": item,
            "issue_type": "hard_unexpected_row_count",
            "issue_detail": f"expected {expected}, got {got}",
        })

required_col_checks = [
    ("behaviour_fusion_box_level_429", bf_path, ["scan_frame_id", "final_box_id", "x1", "y1", "x2", "y2", "behaviour"]),
    ("training_ready_primary_split_374", split_path, ["scan_frame_id", "final_box_id", "split", "behaviour"]),
    ("clip_extraction_index_72", clip_path, ["scan_frame_id", "clip_path", "start_sec", "end_sec"]),
    ("clip_level_multilabel_index_72", clip_level_path, ["scan_frame_id", "clip_path", "split"]),
]

for item, path, required in required_col_checks:
    if not has_cols(path, required):
        issues.append({
            "item": item,
            "issue_type": "hard_missing_expected_columns",
            "issue_detail": f"required keywords: {required}; cols={cols(path)[:50]}",
        })

audit_df = pd.DataFrame(audit_rows)
issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])

hard_issues = [x for x in issues if str(x["issue_type"]).startswith("hard_")]
warnings = [x for x in issues if str(x["issue_type"]).startswith("warning_")]
infos = [x for x in issues if str(x["issue_type"]).startswith("info_")]

ready_for_v44 = len(hard_issues) == 0

decision = pd.DataFrame([{
    "v43c_decision": "final_input_lock_passed" if ready_for_v44 else "final_input_lock_blocked",
    "locked_input_count": int(len(locked["locked_inputs"])),
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "info_count": int(len(infos)),
    "issue_count": int(len(issues_df)),
    "behaviour_fusion_rows": row_count(bf_path),
    "training_ready_rows": row_count(split_path),
    "clip_index_rows": row_count(clip_path),
    "clip_level_index_rows": row_count(clip_level_path),
    "ready_for_v44_ground_truth_schema": bool(ready_for_v44),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

OUT_FINAL_JSON.write_text(json.dumps(locked, indent=2))
safe_to_csv(audit_df, OUT_AUDIT)
safe_to_csv(issues_df, OUT_ISSUES)
safe_to_csv(decision, OUT_DECISION)

# Progress log
progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v43c",
    "task_name": "Final canonical input lock",
    "status": "PASS" if ready_for_v44 else "BLOCKED",
    "input_summary": str(V43B_JSON),
    "output_summary": str(OUT_FINAL_JSON),
    "hard_issues": int(len(hard_issues)),
    "warnings": int(len(warnings)),
    "next_action": "v44 ground-truth schema and field dictionary" if ready_for_v44 else "Fix locked input mappings.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

report = f"""# Week 8 v43c Final Input Lock Report

## Purpose

This stage locks the final canonical inputs for Week 8. It corrects the v43b ambiguity where some valid files were selected but needed role clarification before schema design and label propagation.

## Decision

- v43c decision: `{decision.iloc[0]['v43c_decision']}`
- Locked input count: `{len(locked['locked_inputs'])}`
- Hard issue count: `{len(hard_issues)}`
- Warning count: `{len(warnings)}`
- Info count: `{len(infos)}`
- Ready for v44 ground-truth schema: `{ready_for_v44}`

## Important lock decisions

1. The 429-row v18c box-level fusion table is the main box-level identity and behaviour reference.
2. The v17 selected frame QA file is kept as a QA/supporting file, not as the effective box-level identity table.
3. The 374-row v21 primary split table is the canonical training-ready/model-ready pig-level dataset.
4. The 72-row v26 clip index is the canonical 10-second clip interval reference.
5. The 72-row v34b clip-level multi-label index is the canonical clip-level behaviour/split reference.
6. v28d/v29b/v29c/v29d/v29e outputs remain identity/tracking evidence, not full production tracking ground truth.

## Key row counts

- Behaviour fusion box-level rows: `{row_count(bf_path)}`
- Training-ready primary split rows: `{row_count(split_path)}`
- Clip extraction index rows: `{row_count(clip_path)}`
- Clip-level multi-label index rows: `{row_count(clip_level_path)}`

## Next step

Proceed to v44: define the ground-truth JSON schema, field dictionary, annotation rules, colour identifiers, behaviour label dictionary, propagation rule, and validation flags.
"""

OUT_REPORT.write_text(report)

OUT_NOTE.write_text(
    "# Week 8 v43c Final Input Lock\n\n"
    "## Summary\n\n"
    f"- v43c decision: `{decision.iloc[0]['v43c_decision']}`\n"
    f"- Locked input count: `{len(locked['locked_inputs'])}`\n"
    f"- Behaviour fusion rows: `{row_count(bf_path)}`\n"
    f"- Training-ready rows: `{row_count(split_path)}`\n"
    f"- Clip index rows: `{row_count(clip_path)}`\n"
    f"- Clip-level index rows: `{row_count(clip_level_path)}`\n"
    f"- Hard issue count: `{len(hard_issues)}`\n"
    f"- Ready for v44 ground-truth schema: `{ready_for_v44}`\n\n"
    "## Final lock decisions\n\n"
    "- Use v18c box-level fusion as the main 429-row identity + behaviour reference.\n"
    "- Use v21 primary split as the canonical 374-row training-ready set.\n"
    "- Use v26 clip extraction index as the 72-clip interval reference.\n"
    "- Keep v17 frame QA as supporting QA, not box-level identity.\n\n"
    "## Outputs\n\n"
    f"- Final locked inputs JSON: `{OUT_FINAL_JSON}`\n"
    f"- Audit CSV: `{OUT_AUDIT}`\n"
    f"- Decision CSV: `{OUT_DECISION}`\n"
    f"- Issues CSV: `{OUT_ISSUES}`\n"
    f"- Report: `{OUT_REPORT}`\n"
)

print("Saved:")
print(OUT_FINAL_JSON)
print(OUT_AUDIT)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_REPORT)
print(OUT_NOTE)
print(OUT_PROGRESS)

print()
print("=== v43c decision ===")
print(decision.to_string(index=False))

print()
print("=== v43c issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")

print()
print("=== v43c locked audit ===")
print(audit_df[["locked_key", "source_key", "role", "status", "row_count", "column_count", "path"]].to_string(index=False))
