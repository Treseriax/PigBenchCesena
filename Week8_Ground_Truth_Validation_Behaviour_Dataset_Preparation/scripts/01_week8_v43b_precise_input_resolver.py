from pathlib import Path
from datetime import datetime
import json
import csv
import hashlib
import pandas as pd


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

OUT = W8 / "outputs" / "v43b_precise_input_resolver"
OUT.mkdir(parents=True, exist_ok=True)

OUT_CANDIDATES = OUT / "week8_v43b_precise_input_candidates.csv"
OUT_SELECTED_CSV = OUT / "week8_v43b_precise_selected_inputs.csv"
OUT_SELECTED_JSON = OUT / "week8_v43b_precise_selected_inputs.json"
OUT_DECISION = OUT / "week8_v43b_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v43b_issues.csv"
OUT_REPORT = OUT / "week8_v43b_precise_input_resolver_report.md"
OUT_NOTE = W8 / "notes" / "week8_v43b_precise_input_resolver_notes.md"
OUT_PROGRESS = W8 / "progress" / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def clean(v):
    if pd.isna(v):
        return ""
    return str(v).strip()


def sha256_file(path):
    p = Path(path)
    if not p.exists() or not p.is_file():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def quick_row_count(path):
    p = Path(path)
    try:
        with open(p, "r", errors="ignore") as f:
            n = sum(1 for _ in f)
        return max(0, n - 1)
    except Exception:
        return ""


def read_columns(path):
    try:
        df = pd.read_csv(path, nrows=3)
        return list(df.columns)
    except Exception:
        return []


def has_col_like(columns, keywords):
    cols = [str(c).lower() for c in columns]
    for kw in keywords:
        kw = kw.lower()
        if any(kw in c for c in cols):
            return True
    return False


def score_file(path, spec):
    p = Path(path)
    name = p.name.lower()
    full = str(p).lower()
    cols = read_columns(p)
    row_count = quick_row_count(p)

    score = 0
    reasons = []

    for token in spec.get("positive_name_tokens", []):
        if token.lower() in name or token.lower() in full:
            score += 8
            reasons.append(f"+name:{token}")

    for token in spec.get("negative_name_tokens", []):
        if token.lower() in name or token.lower() in full:
            score -= 12
            reasons.append(f"-name:{token}")

    for group in spec.get("column_groups", []):
        if has_col_like(cols, group):
            score += 10
            reasons.append("+cols:" + "/".join(group))
        else:
            score -= 4
            reasons.append("-cols:" + "/".join(group))

    min_rows = spec.get("expected_min_rows")
    if isinstance(row_count, int) and min_rows is not None:
        if row_count >= min_rows:
            score += 8
            reasons.append(f"+rows>={min_rows}")
        else:
            score -= 8
            reasons.append(f"-rows<{min_rows}")

    preferred_exact = spec.get("preferred_exact_names", [])
    for exact in preferred_exact:
        if p.name == exact:
            score += 40
            reasons.append(f"+exact:{exact}")

    summary_like_tokens = [
        "summary",
        "decision",
        "issue",
        "issues",
        "manifest",
        "report",
        "readme",
        "hard_issues",
        "warnings",
        "notes",
    ]

    is_summary_like = any(t in name for t in summary_like_tokens)

    if is_summary_like and spec.get("avoid_summary_like", True):
        score -= 25
        reasons.append("-summary_like")

    return {
        "score": score,
        "columns": cols,
        "row_count": row_count,
        "is_summary_like": is_summary_like,
        "reasons": " ; ".join(reasons),
    }


def scan_csvs(base_dir):
    base = Path(base_dir)
    if not base.exists():
        return []
    return sorted([p for p in base.rglob("*.csv") if p.is_file()])


specs = [
    {
        "key": "corrected_scanpoint_boxes",
        "required": True,
        "base_dir": W7 / "outputs" / "final_corrected_gt_pen_boxes_v11",
        "purpose": "Corrected pig boxes on annotated scanpoint frames.",
        "positive_name_tokens": ["final_corrected", "gt_pen_boxes", "for_colour_matching", "boxes"],
        "negative_name_tokens": ["summary", "issue", "overlay_index", "review"],
        "preferred_exact_names": [
            "week7_final_corrected_gt_pen_boxes_v11_for_colour_matching.csv",
        ],
        "column_groups": [
            ["scan_frame", "scanframe"],
            ["x1", "bbox_x1", "left"],
            ["y1", "bbox_y1", "top"],
            ["x2", "bbox_x2", "right"],
            ["y2", "bbox_y2", "bottom"],
        ],
        "expected_min_rows": 350,
        "avoid_summary_like": True,
    },
    {
        "key": "final_colour_identity_rows",
        "required": True,
        "base_dir": W7 / "outputs" / "colour_identity" / "final_colour_identity_v17_fixed",
        "purpose": "Final per-box colour identity rows after manual correction and lock.",
        "positive_name_tokens": ["final_colour_identity", "colour_identity", "box", "rows", "fixed"],
        "negative_name_tokens": ["summary", "issue", "issues", "report", "decision"],
        "column_groups": [
            ["scan_frame", "scanframe"],
            ["colour", "color", "visual_marker"],
            ["identity", "status", "final"],
        ],
        "expected_min_rows": 350,
        "avoid_summary_like": True,
    },
    {
        "key": "behaviour_fusion_box_level_rows",
        "required": True,
        "base_dir": W7 / "outputs" / "behaviour_label_fusion_v18c_verified_crosswalk",
        "purpose": "Box-level identity + behaviour fusion rows.",
        "positive_name_tokens": ["box_level", "behaviour_label_fusion", "fusion", "training", "matched"],
        "negative_name_tokens": ["summary", "issue", "issues", "unmatched", "report", "decision"],
        "column_groups": [
            ["scan_frame", "scanframe"],
            ["behaviour", "behavior"],
            ["colour", "color", "visual_marker", "pig_id"],
        ],
        "expected_min_rows": 350,
        "avoid_summary_like": True,
    },
    {
        "key": "primary_split_all_rows",
        "required": True,
        "base_dir": W7 / "outputs" / "primary_split_v21",
        "purpose": "Primary split assignment rows.",
        "positive_name_tokens": ["primary_split", "all"],
        "negative_name_tokens": ["summary", "issue", "report", "decision"],
        "preferred_exact_names": [
            "week7_primary_split_v21_all.csv",
        ],
        "column_groups": [
            ["split"],
            ["scan_frame", "scanframe"],
            ["behaviour", "behavior"],
        ],
        "expected_min_rows": 300,
        "avoid_summary_like": True,
    },
    {
        "key": "clip_extraction_index_v26",
        "required": True,
        "base_dir": W7 / "outputs" / "clip_extraction_temporal_qa_v26",
        "purpose": "10-second extracted clip index with clip path and timing.",
        "positive_name_tokens": ["clip", "index", "temporal", "extraction"],
        "negative_name_tokens": ["summary", "issue", "report", "decision"],
        "column_groups": [
            ["clip_path", "clip"],
            ["scan_frame", "scanframe"],
            ["start", "end", "center", "timestamp", "ms"],
        ],
        "expected_min_rows": 70,
        "avoid_summary_like": True,
    },
    {
        "key": "clip_directory_v26",
        "required": True,
        "path": W7 / "outputs" / "clip_extraction_temporal_qa_v26" / "clips",
        "purpose": "Directory containing extracted 10-second clips.",
        "type": "directory",
    },
    {
        "key": "clip_level_multilabel_index_v34b",
        "required": True,
        "base_dir": W7 / "outputs" / "week7_baseline_behaviour_representation_dataset_v34b_split_fixed",
        "purpose": "Split-fixed clip-level multi-label behaviour index.",
        "positive_name_tokens": ["clip_level", "multilabel", "behaviour", "index", "split_fixed"],
        "negative_name_tokens": ["summary", "issue", "report", "decision", "readiness"],
        "preferred_exact_names": [
            "week7_v34b_clip_level_multilabel_behaviour_index_split_fixed.csv",
        ],
        "column_groups": [
            ["clip_path"],
            ["scan_frame", "scanframe"],
            ["split"],
            ["label__"],
        ],
        "expected_min_rows": 70,
        "avoid_summary_like": True,
    },
    {
        "key": "dense_polygon_tracking_rows_v28d",
        "required": True,
        "base_dir": W7 / "outputs" / "dense_polygon_filtered_tracking_v28d",
        "purpose": "Dense polygon-filtered tracking rows for identity-over-time work.",
        "positive_name_tokens": ["dense", "polygon", "tracking", "tracks", "detections"],
        "negative_name_tokens": ["summary", "decision", "issue", "issues", "report"],
        "column_groups": [
            ["track_id", "dense_track"],
            ["frame"],
            ["x1", "bbox", "left"],
            ["y1", "top"],
        ],
        "expected_min_rows": 500,
        "avoid_summary_like": True,
    },
    {
        "key": "colour_constrained_tracklet_linking_rows_v29b",
        "required": True,
        "base_dir": W7 / "outputs" / "colour_constrained_tracklet_linking_v29b",
        "purpose": "Colour-constrained tracklet linking rows.",
        "positive_name_tokens": ["tracklet", "linking", "assignment", "assigned", "identity"],
        "negative_name_tokens": ["summary", "decision", "issue", "issues", "report"],
        "column_groups": [
            ["tracklet", "track"],
            ["colour", "color"],
            ["identity", "assigned", "status"],
        ],
        "expected_min_rows": 20,
        "avoid_summary_like": True,
    },
    {
        "key": "full_tracklet_colour_evidence_v29c",
        "required": False,
        "base_dir": W7 / "outputs" / "full_tracklet_colour_evidence_v29c",
        "purpose": "Full-tracklet colour evidence rows.",
        "positive_name_tokens": ["tracklet", "colour", "color", "evidence"],
        "negative_name_tokens": ["summary", "decision", "issue", "issues", "report"],
        "column_groups": [
            ["tracklet", "track"],
            ["colour", "color"],
            ["confidence", "evidence", "status"],
        ],
        "expected_min_rows": 20,
        "avoid_summary_like": True,
    },
    {
        "key": "conservative_identity_arbitration_rows_v29d",
        "required": True,
        "base_dir": W7 / "outputs" / "conservative_identity_arbitration_v29d",
        "purpose": "Conservative identity arbitration rows.",
        "positive_name
