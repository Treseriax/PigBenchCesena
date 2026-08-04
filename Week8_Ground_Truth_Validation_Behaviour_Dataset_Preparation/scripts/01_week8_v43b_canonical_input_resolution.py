from pathlib import Path
from datetime import datetime
import csv
import json
import hashlib
import pandas as pd


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

OUT = W8 / "outputs" / "v43_setup_input_audit"
OUT.mkdir(parents=True, exist_ok=True)

OUT_CANDIDATES = OUT / "week8_v43b_canonical_input_candidates.csv"
OUT_SELECTED = OUT / "week8_v43b_canonical_selected_inputs.json"
OUT_AUDIT = OUT / "week8_v43b_canonical_input_audit.csv"
OUT_DECISION = OUT / "week8_v43b_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v43b_issues.csv"
OUT_REPORT = OUT / "week8_v43b_canonical_input_resolution_report.md"
OUT_NOTE = W8 / "notes" / "week8_v43b_canonical_input_resolution_notes.md"
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


def read_csv_light(path, max_rows=5000):
    try:
        return pd.read_csv(path, nrows=max_rows)
    except Exception:
        return pd.DataFrame()


def full_row_count(path):
    try:
        with open(path, "r", errors="ignore") as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return ""


def has_any_col(cols, keywords):
    cols_l = [str(c).lower() for c in cols]
    return any(any(k.lower() in c for c in cols_l) for k in keywords)


def has_all_keywords(cols, keywords):
    cols_l = [str(c).lower() for c in cols]
    return all(any(k.lower() in c for c in cols_l) for k in keywords)


def collect_csvs(base):
    base = Path(base)
    if not base.exists():
        return []
    return sorted([p for p in base.rglob("*.csv") if p.is_file()])


def name_score(path, positive=None, negative=None):
    positive = positive or []
    negative = negative or []
    s = str(path.name).lower()
    score = 0
    for w in positive:
        if w.lower() in s:
            score += 5
    for w in negative:
        if w.lower() in s:
            score -= 7
    return score


def classify_file(path, spec):
    df = read_csv_light(path)
    cols = list(df.columns)
    n_rows = full_row_count(path)

    score = 0
    reasons = []

    score += name_score(
        path,
        positive=spec.get("positive_name", []),
        negative=spec.get("negative_name", []),
    )

    for kw in spec.get("required_col_keywords", []):
        if has_any_col(cols, [kw]):
            score += 8
            reasons.append(f"has_col_keyword:{kw}")
        else:
            score -= 12
            reasons.append(f"missing_col_keyword:{kw}")

    for kw in spec.get("positive_col_keywords", []):
        if has_any_col(cols, [kw]):
            score += 4
            reasons.append(f"has_positive_col:{kw}")

    for kw in spec.get("negative_col_keywords", []):
        if has_any_col(cols, [kw]):
            score -= 4
            reasons.append(f"has_negative_col:{kw}")

    try:
        rn = int(n_rows)
    except Exception:
        rn = 0

    target_min = spec.get("target_min_rows")
    target_max = spec.get("target_max_rows")

    if target_min is not None and rn >= target_min:
        score += 5
        reasons.append(f"rows>=min:{target_min}")

    if target_max is not None and rn <= target_max:
        score += 3
        reasons.append(f"rows<=max:{target_max}")

    if rn <= 2:
        score -= 10
        reasons.append("very_few_rows")

    if "summary" in path.name.lower():
        score -= 20
        reasons.append("filename_summary_penalty")

    if "issue" in path.name.lower() or "issues" in path.name.lower():
        score -= 20
        reasons.append("filename_issues_penalty")

    if "decision" in path.name.lower():
        score -= 20
        reasons.append("filename_decision_penalty")

    if "report" in path.name.lower():
        score -= 8
        reasons.append("filename_report_penalty")

    return {
        "score": score,
        "row_count": n_rows,
        "columns": " | ".join([str(c) for c in cols[:40]]),
        "column_count": len(cols),
        "reasons": " ; ".join(reasons),
    }


specs = {
    "corrected_boxes_box_level": {
        "required": True,
        "base": W7 / "outputs" / "final_corrected_gt_pen_boxes_v11",
        "purpose": "Box-level corrected pig bounding boxes for annotated scanpoint frames.",
        "positive_name": ["for_colour_matching", "corrected", "boxes", "final"],
        "negative_name": ["summary", "issue", "decision", "overlay_index"],
        "required_col_keywords": ["scan", "box"],
        "positive_col_keywords": ["x1", "y1", "x2", "y2", "bbox", "colour", "final"],
        "target_min_rows": 300,
        "target_max_rows": 600,
    },
    "final_colour_identity_box_level": {
        "required": True,
        "base": W7 / "outputs" / "colour_identity" / "final_colour_identity_v17_fixed",
        "purpose": "Box-level final colour identity assignments.",
        "positive_name": ["final", "colour", "identity", "box", "fixed"],
        "negative_name": ["summary", "issue", "decision"],
        "required_col_keywords": ["colour", "identity"],
        "positive_col_keywords": ["scan", "box", "not_visible", "uncertain", "usable"],
        "target_min_rows": 300,
        "target_max_rows": 600,
    },
    "behaviour_fusion_box_level": {
        "required": True,
        "base": W7 / "outputs" / "behaviour_label_fusion_v18c_verified_crosswalk",
        "purpose": "Box-level fused identity and behaviour labels.",
        "positive_name": ["box", "level", "fusion", "behaviour", "all"],
        "negative_name": ["summary", "issue", "decision", "crosswalk"],
        "required_col_keywords": ["behaviour", "colour"],
        "positive_col_keywords": ["scan", "box", "pig", "label", "status"],
        "target_min_rows": 300,
        "target_max_rows": 600,
    },
    "behaviour_training_ready_rows": {
        "required": True,
        "base": W7 / "outputs" / "behaviour_label_fusion_v18c_verified_crosswalk",
        "purpose": "Training-ready rows after behaviour fusion.",
        "positive_name": ["training", "ready", "matched", "behaviour"],
        "negative_name": ["summary", "issue", "decision", "crosswalk"],
        "required_col_keywords": ["behaviour", "colour"],
        "positive_col_keywords": ["split", "crop", "scan", "pig", "label"],
        "target_min_rows": 300,
        "target_max_rows": 450,
    },
    "primary_split_all_rows": {
        "required": True,
        "base": W7 / "outputs" / "primary_split_v21",
        "purpose": "Primary split all rows.",
        "positive_name": ["all", "primary", "split"],
        "negative_name": ["summary", "issue", "decision", "class"],
        "required_col_keywords": ["split", "behaviour"],
        "positive_col_keywords": ["train", "val", "test", "scan"],
        "target_min_rows": 300,
        "target_max_rows": 450,
    },
    "clip_extraction_index": {
        "required": True,
        "base": W7 / "outputs" / "clip_extraction_temporal_qa_v26",
        "purpose": "10-second clip extraction index with clip paths and scanframe mapping.",
        "positive_name": ["clip", "index", "temporal", "extraction"],
        "negative_name": ["summary", "issue", "decision", "contact"],
        "required_col_keywords": ["clip"],
        "positive_col_keywords": ["scan", "path", "start", "end", "timestamp", "frame"],
        "target_min_rows": 60,
        "target_max_rows": 100,
    },
    "clip_level_multilabel_split_fixed_index": {
        "required": True,
        "base": W7 / "outputs" / "week7_baseline_behaviour_representation_dataset_v34b_split_fixed",
        "purpose": "Clip-level multi-label split-fixed behaviour index.",
        "positive_name": ["clip", "multilabel", "index", "split", "fixed"],
        "negative_name": ["summary", "issue", "decision", "class"],
        "required_col_keywords": ["clip", "split"],
        "positive_col_keywords": ["label__", "behaviour", "scan", "path"],
        "target_min_rows": 60,
        "target_max_rows": 100,
    },
    "dense_polygon_tracking_rows": {
        "required": True,
        "base": W7 / "outputs" / "dense_polygon_filtered_tracking_v28d",
        "purpose": "Dense polygon-filtered tracking detections/track rows.",
        "positive_name": ["track", "tracking", "dense", "rows", "detections"],
        "negative_name": ["summary", "issue", "decision", "contact"],
        "required_col_keywords": ["track"],
        "positive_col_keywords": ["bbox", "frame", "scan", "dense", "clip"],
        "target_min_rows": 500,
    },
    "colour_constrained_tracklet_assignments": {
        "required": True,
        "base": W7 / "outputs" / "colour_constrained_tracklet_linking_v29b",
        "purpose": "Colour-constrained tracklet identity linking assignments.",
        "positive_name": ["tracklet", "assignment", "linking", "colour", "identity"],
        "negative_name": ["summary", "issue", "decision"],
        "required_col_keywords": ["track"],
        "positive_col_keywords": ["colour", "assigned", "confidence", "scan"],
        "target_min_rows": 20,
        "target_max_rows": 100,
    },
    "full_tracklet_colour_evidence": {
        "required": False,
        "base": W7 / "outputs" / "full_tracklet_colour_evidence_v29c",
        "purpose": "Full-tracklet colour evidence rows.",
        "positive_name": ["tracklet", "colour", "evidence", "full"],
        "negative_name": ["summary", "issue", "decision"],
        "required_col_keywords": ["track"],
        "positive_col_keywords": ["colour", "evidence", "confidence", "scan"],
        "target_min_rows": 20,
        "target_max_rows": 100,
    },
    "conservative_identity_arbitration_rows": {
        "required": True,
        "base": W7 / "outputs" / "conservative_identity_arbitration_v29d",
        "purpose": "Conservative identity arbitration table.",
        "positive_name": ["arbitration", "identity", "tracklet"],
        "negative_name": ["summary", "issue", "decision"],
        "required_col_keywords": ["track"],
        "positive_col_keywords": ["status", "colour", "v29b", "v29c", "accepted", "review"],
        "target_min_rows": 20,
        "target_max_rows": 100,
    },
    "identity_review_queue": {
        "required": False,
        "base": W7 / "outputs" / "conservative_identity_arbitration_v29d",
        "purpose": "Identity review queue from conservative arbitration.",
        "positive_name": ["review", "queue"],
        "negative_name": ["summary", "issue", "decision"],
        "required_col_keywords": ["track"],
        "positive_col_keywords": ["review", "colour", "status", "reason"],
        "target_min_rows": 20,
    },
}

candidate_rows = []
selected = {
    "stage": "week8_v43b_canonical_input_resolution",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "week7_root": str(W7),
    "week8_root": str(W8),
    "canonical_inputs": {},
}
audit_rows = []
issues = []

for key, spec in specs.items():
    csvs = collect_csvs(spec["base"])

    if not spec["base"].exists():
        issues.append({
            "item": key,
            "issue_type": "hard_missing_input_directory" if spec["required"] else "warning_missing_optional_input_directory",
            "issue_detail": str(spec["base"]),
        })
        csvs = []

    scored = []

    for p in csvs:
        info = classify_file(p, spec)
        scored.append((info["score"], p, info))

        candidate_rows.append({
            "key": key,
            "path": str(p),
            "required": bool(spec["required"]),
            "purpose": spec["purpose"],
            "score": info["score"],
            "row_count": info["row_count"],
            "column_count": info["column_count"],
            "columns_preview": info["columns"],
            "reasons": info["reasons"],
            "size_bytes": p.stat().st_size,
            "modified_at": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
            "sha256": sha256_file(p) if p.stat().st_size < 200 * 1024 * 1024 else "",
        })

    scored.sort(key=lambda x: x[0], reverse=True)

    best = scored[0] if scored else None

    if best is None:
        audit_rows.append({
            "key": key,
            "required": bool(spec["required"]),
            "status": "MISSING_REQUIRED" if spec["required"] else "MISSING_OPTIONAL",
            "selected_path": "",
            "selected_score": "",
            "row_count": "",
            "purpose": spec["purpose"],
        })

        issues.append({
            "item": key,
            "issue_type": "hard_missing_required_canonical_input" if spec["required"] else "warning_missing_optional_canonical_input",
            "issue_detail": spec["purpose"],
        })
        continue

    score, path, info = best

    status = "SELECTED"

    if score < 0:
        status = "LOW_CONFIDENCE_SELECTED"
        issues.append({
            "item": key,
            "issue_type": "warning_low_confidence_canonical_selection",
            "issue_detail": f"Selected {path} with low score {score}. Review manually.",
        })

    # Flag suspicious if chosen file is summary/issues/decision despite being required.
    lower_name = path.name.lower()
    if spec["required"] and any(x in lower_name for x in ["summary", "issues", "issue", "decision"]):
        issues.append({
            "item": key,
            "issue_type": "hard_suspicious_required_selection",
            "issue_detail": f"Required canonical input selected suspicious file: {path}",
        })

    selected["canonical_inputs"][key] = {
        "path": str(path),
        "required": bool(spec["required"]),
        "purpose": spec["purpose"],
        "score": score,
        "row_count": info["row_count"],
        "columns_preview": info["columns"],
        "sha256": sha256_file(path) if path.stat().st_size < 200 * 1024 * 1024 else "",
    }

    audit_rows.append({
        "key": key,
        "required": bool(spec["required"]),
        "status": status,
        "selected_path": str(path),
        "selected_score": score,
        "row_count": info["row_count"],
        "purpose": spec["purpose"],
    })

candidate_df = pd.DataFrame(candidate_rows)
candidate_df = candidate_df.sort_values(["key", "score"], ascending=[True, False])

audit_df = pd.DataFrame(audit_rows)
issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])

hard_issues = [x for x in issues if str(x["issue_type"]).startswith("hard_")]
warnings = [x for x in issues if str(x["issue_type"]).startswith("warning_")]

ready_for_v44 = len(hard_issues) == 0

safe_to_csv(candidate_df, OUT_CANDIDATES)
safe_to_csv(audit_df, OUT_AUDIT)
safe_to_csv(issues_df, OUT_ISSUES)
OUT_SELECTED.write_text(json.dumps(selected, indent=2))

required_total = sum(1 for x in specs.values() if x["required"])
required_selected = sum(
    1 for row in audit_rows
    if row["required"] and row["status"] in ["SELECTED", "LOW_CONFIDENCE_SELECTED"]
)

optional_total = sum(1 for x in specs.values() if not x["required"])
optional_selected = sum(
    1 for row in audit_rows
    if (not row["required"]) and row["status"] in ["SELECTED", "LOW_CONFIDENCE_SELECTED"]
)

decision = pd.DataFrame([{
    "v43b_decision": "canonical_input_resolution_passed" if ready_for_v44 else "canonical_input_resolution_needs_review",
    "required_canonical_inputs_total": int(required_total),
    "required_canonical_inputs_selected": int(required_selected),
    "optional_canonical_inputs_total": int(optional_total),
    "optional_canonical_inputs_selected": int(optional_selected),
    "candidate_rows": int(len(candidate_df)),
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "issue_count": int(len(issues_df)),
    "ready_for_v44_ground_truth_schema": bool(ready_for_v44),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

# Progress log
progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v43b",
    "task_name": "Canonical Week 8 input resolution",
    "status": "PASS" if ready_for_v44 else "NEEDS_REVIEW",
    "input_summary": "Week 7 output directories scanned and canonical data files selected.",
    "output_summary": str(OUT_SELECTED),
    "hard_issues": int(len(hard_issues)),
    "warnings": int(len(warnings)),
    "next_action": "v44 ground-truth schema and field dictionary" if ready_for_v44 else "Manually review suspicious selections.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)


def markdown_table(df, max_rows=40):
    if df is None or len(df) == 0:
        return "_No rows._"
    d = df.head(max_rows)
    cols = list(d.columns)
    out = []
    out.append("| " + " | ".join(cols) + " |")
    out.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in d.iterrows():
        vals = []
        for c in cols:
            x = str(r.get(c, ""))
            x = x.replace("\n", " ").replace("|", "/")
            vals.append(x)
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)

report = f"""# Week 8 v43b Canonical Input Resolution Report

## Purpose

v43 confirmed that all Week 7 input categories exist, but some automatically selected paths were summary or issue files. v43b resolves canonical data files for Week 8 schema design, label propagation, visualization and identity-validation tasks.

## Decision

- v43b decision: `{decision.iloc[0]['v43b_decision']}`
- Required canonical inputs selected: `{required_selected}` / `{required_total}`
- Optional canonical inputs selected: `{optional_selected}` / `{optional_total}`
- Hard issue count: `{len(hard_issues)}`
- Warning count: `{len(warnings)}`
- Ready for v44 ground-truth schema: `{ready_for_v44}`

## Canonical input audit

{markdown_table(audit_df)}

## Important note

If a selected required input still points to a summary, issue or decision file, v43b blocks the pipeline. Week 8 propagation should only use row-level data files, not summary-only outputs.

## Next step

Proceed to v44 only if v43b passes. v44 will define the ground-truth JSON schema and field dictionary.
"""

OUT_REPORT.write_text(report)

OUT_NOTE.write_text(
    "# Week 8 v43b Canonical Input Resolution\n\n"
    "## Summary\n\n"
    f"- Required canonical inputs selected: `{required_selected}` / `{required_total}`\n"
    f"- Optional canonical inputs selected: `{optional_selected}` / `{optional_total}`\n"
    f"- Candidate rows: `{len(candidate_df)}`\n"
    f"- Hard issue count: `{len(hard_issues)}`\n"
    f"- Warning count: `{len(warnings)}`\n"
    f"- Ready for v44 ground-truth schema: `{ready_for_v44}`\n\n"
    "## Outputs\n\n"
    f"- Candidates: `{OUT_CANDIDATES}`\n"
    f"- Canonical audit: `{OUT_AUDIT}`\n"
    f"- Selected canonical inputs: `{OUT_SELECTED}`\n"
    f"- Decision: `{OUT_DECISION}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
    f"- Report: `{OUT_REPORT}`\n"
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
print("=== v43b decision ===")
print(decision.to_string(index=False))

print()
print("=== v43b canonical audit ===")
print(audit_df.to_string(index=False))

print()
print("=== v43b issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
