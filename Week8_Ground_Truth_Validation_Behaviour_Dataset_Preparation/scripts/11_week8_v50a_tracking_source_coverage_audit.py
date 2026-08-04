from pathlib import Path
from datetime import datetime
import json
import csv
import pandas as pd
import numpy as np


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

LOCKED_JSON = W8 / "outputs" / "v43_setup_input_audit" / "week8_v43c_final_locked_inputs.json"
V45_CLIP_JSON = W8 / "outputs" / "propagated_ground_truth" / "v45_label_propagation" / "week8_v45_clip_level_ground_truth.json"
V49C_DECISION = W8 / "outputs" / "v49c_anchor_validation_verdict" / "week8_v49c_decision_summary.csv"

OUT = W8 / "outputs" / "v50a_tracking_source_coverage_audit"
REPORTS = W8 / "reports"
NOTES = W8 / "notes"
PROGRESS = W8 / "progress"

for p in [OUT, REPORTS, NOTES, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_TRACKING_SOURCE_CANDIDATES = OUT / "week8_v50a_tracking_source_candidates.csv"
OUT_TRACKING_COLUMN_INVENTORY = OUT / "week8_v50a_tracking_column_inventory.csv"
OUT_TRACKING_COVERAGE_BY_CLIP = OUT / "week8_v50a_tracking_coverage_by_clip.csv"
OUT_NORMALIZED_SAMPLE = OUT / "week8_v50a_normalized_tracking_sample.csv"
OUT_DECISION = OUT / "week8_v50a_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v50a_issues.csv"
OUT_REPORT = REPORTS / "week8_v50a_tracking_source_coverage_audit_report.md"
OUT_NOTE = NOTES / "week8_v50a_tracking_source_coverage_audit_notes.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def clean_str(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


def read_json(path):
    return json.loads(Path(path).read_text())


def first_existing_col(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def maybe_numeric(s):
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return s


def path_for_locked(locked, key):
    try:
        return Path(locked["locked_inputs"][key]["path"])
    except Exception:
        return None


def find_tracking_sources(locked):
    rows = []

    locked_inputs = locked.get("locked_inputs", {})
    for key, item in locked_inputs.items():
        p = Path(item.get("path", ""))
        if any(tok in key.lower() for tok in ["track", "tracking", "arbitration", "identity"]):
            rows.append({
                "source_type": "locked_input",
                "key": key,
                "path": str(p),
                "exists": p.exists(),
                "priority": 1 if "dense_polygon_tracking" in key.lower() or "tracking_rows" in key.lower() else 2,
            })

    fallback_patterns = [
        "**/*tracking*v28d*tracks*.csv",
        "**/*dense*polygon*tracking*.csv",
        "**/*tracklet*arbitration*.csv",
        "**/*identity*arbitration*.csv",
        "**/*tracklet*.csv",
    ]

    for pattern in fallback_patterns:
        for p in W7.glob(pattern):
            rows.append({
                "source_type": "fallback_glob_week7",
                "key": pattern,
                "path": str(p),
                "exists": p.exists(),
                "priority": 3,
            })

    if rows:
        df = pd.DataFrame(rows).drop_duplicates(subset=["path"]).sort_values(["priority", "path"])
    else:
        df = pd.DataFrame(columns=["source_type", "key", "path", "exists", "priority"])

    return df


issues = []

for p in [LOCKED_JSON, V45_CLIP_JSON, V49C_DECISION]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required previous-stage artifact is missing.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v50a_decision": "tracking_source_coverage_audit_blocked_missing_inputs",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v50b_full_tracking_run": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


locked = read_json(LOCKED_JSON)
clip_data = read_json(V45_CLIP_JSON)
clips = clip_data.get("clips", [])

clip_rows = []
for c in clips:
    clip_rows.append({
        "scan_frame_id": clean_str(c.get("scan_frame_id")),
        "video_id": clean_str(c.get("video_id")),
        "clip_path": clean_str(c.get("clip_path")),
        "split": clean_str(c.get("split")),
        "generated_frame_count": c.get("generated_frame_count"),
        "object_count": len(c.get("objects", [])),
        "behaviour_set": clean_str(c.get("behaviour_set")),
    })

clips_df = pd.DataFrame(clip_rows)

tracking_sources = find_tracking_sources(locked)
safe_to_csv(tracking_sources, OUT_TRACKING_SOURCE_CANDIDATES)

selected_tracking_path = None
selected_source_key = ""

if len(tracking_sources):
    existing = tracking_sources[tracking_sources["exists"] == True].copy()
    if len(existing):
        selected = existing.sort_values(["priority", "path"]).iloc[0]
        selected_tracking_path = Path(selected["path"])
        selected_source_key = selected["key"]

if selected_tracking_path is None or not selected_tracking_path.exists():
    issues.append({
        "item": "tracking_source",
        "issue_type": "hard_no_tracking_source_found",
        "issue_detail": "No existing tracking CSV was found from locked inputs or Week7 fallback search.",
        "severity": "hard",
    })

    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)

    decision = pd.DataFrame([{
        "v50a_decision": "tracking_source_coverage_audit_blocked_no_tracking_source",
        "selected_tracking_path": "",
        "clip_count": int(len(clips_df)),
        "tracking_covered_clip_count": 0,
        "coverage_ratio": 0.0,
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "warning_count": int((issues_df["severity"] == "warning").sum()),
        "issue_count": int(len(issues_df)),
        "ready_for_v50b_full_tracking_run": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


tracking = pd.read_csv(selected_tracking_path)

column_inventory_rows = []
for c in tracking.columns:
    non_null = int(tracking[c].notna().sum())
    example = ""
    if non_null > 0:
        example = clean_str(tracking[c].dropna().iloc[0])
    column_inventory_rows.append({
        "selected_tracking_path": str(selected_tracking_path),
        "column_name": c,
        "dtype": str(tracking[c].dtype),
        "non_null_count": non_null,
        "example_value": example[:250],
    })

column_inventory = pd.DataFrame(column_inventory_rows)
safe_to_csv(column_inventory, OUT_TRACKING_COLUMN_INVENTORY)

scan_col = first_existing_col(tracking, [
    "scan_frame_id",
    "scanframe_id",
    "scan_frame",
    "clip_scan_frame_id",
    "clip_id",
    "clip_name",
])

clip_path_col = first_existing_col(tracking, [
    "clip_path",
    "video_clip_path",
    "source_clip_path",
    "video_path",
])

video_id_col = first_existing_col(tracking, [
    "video_id",
    "video_name",
    "source_video_id",
])

frame_col = first_existing_col(tracking, [
    "frame_index_in_clip",
    "frame_idx",
    "frame_index",
    "frame",
    "sampled_frame_index",
    "clip_frame_idx",
    "frame_id",
])

track_col = first_existing_col(tracking, [
    "track_id",
    "tracklet_id",
    "dense_track_id",
    "local_track_id",
    "id",
])

x1_col = first_existing_col(tracking, ["bbox_x1", "x1", "left", "xmin"])
y1_col = first_existing_col(tracking, ["bbox_y1", "y1", "top", "ymin"])
x2_col = first_existing_col(tracking, ["bbox_x2", "x2", "right", "xmax"])
y2_col = first_existing_col(tracking, ["bbox_y2", "y2", "bottom", "ymax"])

w_col = first_existing_col(tracking, ["bbox_w", "w", "width"])
h_col = first_existing_col(tracking, ["bbox_h", "h", "height"])

conf_col = first_existing_col(tracking, [
    "score",
    "confidence",
    "conf",
    "det_conf",
    "detection_score",
])

normalized = pd.DataFrame()
normalized["source_row_index"] = np.arange(len(tracking))

if scan_col:
    normalized["scan_frame_id"] = tracking[scan_col].astype(str).map(clean_str)
else:
    normalized["scan_frame_id"] = ""

if clip_path_col:
    normalized["clip_path"] = tracking[clip_path_col].astype(str).map(clean_str)
else:
    normalized["clip_path"] = ""

if video_id_col:
    normalized["video_id"] = tracking[video_id_col].astype(str).map(clean_str)
else:
    normalized["video_id"] = ""

if frame_col:
    normalized["frame_index_in_clip"] = pd.to_numeric(tracking[frame_col], errors="coerce")
else:
    normalized["frame_index_in_clip"] = np.nan

if track_col:
    normalized["track_id"] = tracking[track_col].astype(str).map(clean_str)
else:
    normalized["track_id"] = ""

if x1_col and y1_col and x2_col and y2_col:
    normalized["bbox_x1"] = pd.to_numeric(tracking[x1_col], errors="coerce")
    normalized["bbox_y1"] = pd.to_numeric(tracking[y1_col], errors="coerce")
    normalized["bbox_x2"] = pd.to_numeric(tracking[x2_col], errors="coerce")
    normalized["bbox_y2"] = pd.to_numeric(tracking[y2_col], errors="coerce")
elif x1_col and y1_col and w_col and h_col:
    x1 = pd.to_numeric(tracking[x1_col], errors="coerce")
    y1 = pd.to_numeric(tracking[y1_col], errors="coerce")
    w = pd.to_numeric(tracking[w_col], errors="coerce")
    h = pd.to_numeric(tracking[h_col], errors="coerce")
    normalized["bbox_x1"] = x1
    normalized["bbox_y1"] = y1
    normalized["bbox_x2"] = x1 + w
    normalized["bbox_y2"] = y1 + h
else:
    normalized["bbox_x1"] = np.nan
    normalized["bbox_y1"] = np.nan
    normalized["bbox_x2"] = np.nan
    normalized["bbox_y2"] = np.nan

if conf_col:
    normalized["confidence"] = pd.to_numeric(tracking[conf_col], errors="coerce")
else:
    normalized["confidence"] = np.nan

# Try to recover scan_frame_id from clip path basename if scan col is absent.
if normalized["scan_frame_id"].eq("").all() and normalized["clip_path"].ne("").any():
    # Match any scan_frame_id whose string appears in the clip path.
    scan_ids = clips_df["scan_frame_id"].tolist()

    def infer_scan_from_path(p):
        p = clean_str(p)
        for s in scan_ids:
            if s and s in p:
                return s
        return ""

    normalized["scan_frame_id"] = normalized["clip_path"].map(infer_scan_from_path)

# If clip path is missing but scan exists, attach clip_path from v45.
clip_path_map = dict(zip(clips_df["scan_frame_id"], clips_df["clip_path"]))
video_id_map = dict(zip(clips_df["scan_frame_id"], clips_df["video_id"]))

normalized["clip_path_from_v45"] = normalized["scan_frame_id"].map(clip_path_map).fillna("")
normalized["video_id_from_v45"] = normalized["scan_frame_id"].map(video_id_map).fillna("")

normalized["effective_clip_path"] = normalized["clip_path"]
normalized.loc[normalized["effective_clip_path"].eq(""), "effective_clip_path"] = normalized.loc[
    normalized["effective_clip_path"].eq(""), "clip_path_from_v45"
]

normalized["effective_video_id"] = normalized["video_id"]
normalized.loc[normalized["effective_video_id"].eq(""), "effective_video_id"] = normalized.loc[
    normalized["effective_video_id"].eq(""), "video_id_from_v45"
]

normalized["bbox_valid_basic"] = (
    normalized["bbox_x1"].notna()
    & normalized["bbox_y1"].notna()
    & normalized["bbox_x2"].notna()
    & normalized["bbox_y2"].notna()
    & (normalized["bbox_x2"] > normalized["bbox_x1"])
    & (normalized["bbox_y2"] > normalized["bbox_y1"])
)

safe_to_csv(normalized.head(500), OUT_NORMALIZED_SAMPLE)

coverage_rows = []

for _, clip in clips_df.iterrows():
    scan = clip["scan_frame_id"]
    g = normalized[normalized["scan_frame_id"] == scan].copy()

    tracking_rows = len(g)
    unique_frames = int(g["frame_index_in_clip"].dropna().nunique()) if tracking_rows else 0
    unique_tracks = int(g["track_id"].replace("", np.nan).dropna().nunique()) if tracking_rows else 0
    valid_bbox_rows = int(g["bbox_valid_basic"].sum()) if tracking_rows else 0

    generated_frames = int(clip["generated_frame_count"]) if pd.notna(clip["generated_frame_count"]) else 0
    frame_coverage_ratio = unique_frames / generated_frames if generated_frames > 0 else 0.0

    if tracking_rows == 0:
        coverage_status = "no_tracking_rows"
    elif frame_coverage_ratio >= 0.8:
        coverage_status = "high_frame_coverage"
    elif frame_coverage_ratio >= 0.2:
        coverage_status = "partial_frame_coverage"
    else:
        coverage_status = "low_sampled_coverage"

    coverage_rows.append({
        "scan_frame_id": scan,
        "video_id": clip["video_id"],
        "split": clip["split"],
        "behaviour_set": clip["behaviour_set"],
        "clip_path": clip["clip_path"],
        "generated_frame_count": generated_frames,
        "object_count_anchor": int(clip["object_count"]),
        "tracking_rows": int(tracking_rows),
        "tracking_unique_frames": int(unique_frames),
        "tracking_unique_tracks": int(unique_tracks),
        "tracking_valid_bbox_rows": int(valid_bbox_rows),
        "tracking_frame_coverage_ratio": float(frame_coverage_ratio),
        "coverage_status": coverage_status,
    })

coverage = pd.DataFrame(coverage_rows)
safe_to_csv(coverage, OUT_TRACKING_COVERAGE_BY_CLIP)

covered_clip_count = int((coverage["tracking_rows"] > 0).sum())
high_or_partial_clip_count = int(coverage["coverage_status"].isin(["high_frame_coverage", "partial_frame_coverage"]).sum())
coverage_ratio = covered_clip_count / len(coverage) if len(coverage) else 0.0

missing_clip_count = int((coverage["tracking_rows"] == 0).sum())
valid_bbox_rows_total = int(normalized["bbox_valid_basic"].sum())
tracking_rows_total = int(len(normalized))

detected_columns = {
    "scan_col": scan_col or "",
    "clip_path_col": clip_path_col or "",
    "video_id_col": video_id_col or "",
    "frame_col": frame_col or "",
    "track_col": track_col or "",
    "x1_col": x1_col or "",
    "y1_col": y1_col or "",
    "x2_col": x2_col or "",
    "y2_col": y2_col or "",
    "w_col": w_col or "",
    "h_col": h_col or "",
    "conf_col": conf_col or "",
}

if not frame_col:
    issues.append({
        "item": "tracking_frame_column",
        "issue_type": "warning_frame_column_not_detected",
        "issue_detail": "Could not confidently detect frame index column in tracking source.",
        "severity": "warning",
    })

if not track_col:
    issues.append({
        "item": "tracking_track_id_column",
        "issue_type": "warning_track_id_column_not_detected",
        "issue_detail": "Could not confidently detect track id column in tracking source.",
        "severity": "warning",
    })

if valid_bbox_rows_total == 0:
    issues.append({
        "item": "tracking_bbox_columns",
        "issue_type": "hard_no_valid_tracking_bboxes",
        "issue_detail": "No valid bbox rows could be normalized from tracking source.",
        "severity": "hard",
    })

if covered_clip_count < len(coverage):
    issues.append({
        "item": "tracking_clip_coverage",
        "issue_type": "warning_partial_tracking_clip_coverage",
        "issue_detail": f"Tracking source covers {covered_clip_count}/{len(coverage)} clips. Full 72-clip tracking run is needed before final interface integration.",
        "severity": "warning",
    })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])

hard_issues = issues_df[issues_df["severity"] == "hard"] if len(issues_df) else pd.DataFrame()
warnings = issues_df[issues_df["severity"] == "warning"] if len(issues_df) else pd.DataFrame()

# Decision logic:
# If tracking source covers all 72 clips and bbox/frame/track columns are usable, we can integrate.
# Otherwise next step is a full 72-clip tracking run.
existing_tracking_ready_for_full_integration = (
    len(hard_issues) == 0
    and covered_clip_count == len(coverage)
    and frame_col is not None
    and track_col is not None
    and valid_bbox_rows_total > 0
)

ready_for_v50b_full_run = len(hard_issues) == 0 and not existing_tracking_ready_for_full_integration
ready_for_v50c_integration = existing_tracking_ready_for_full_integration

if existing_tracking_ready_for_full_integration:
    v50a_decision = "existing_tracking_source_covers_all_clips_ready_for_integration"
else:
    v50a_decision = "existing_tracking_source_partial_full_72_tracking_run_needed"

decision_payload = {
    "v50a_decision": v50a_decision,
    "selected_tracking_path": str(selected_tracking_path),
    "selected_source_key": selected_source_key,
    "tracking_rows_total": int(tracking_rows_total),
    "valid_tracking_bbox_rows_total": int(valid_bbox_rows_total),
    "clip_count": int(len(coverage)),
    "tracking_covered_clip_count": int(covered_clip_count),
    "missing_tracking_clip_count": int(missing_clip_count),
    "coverage_ratio": float(coverage_ratio),
    "high_or_partial_clip_count": int(high_or_partial_clip_count),
    "scan_col": detected_columns["scan_col"],
    "frame_col": detected_columns["frame_col"],
    "track_col": detected_columns["track_col"],
    "bbox_x1_col": detected_columns["x1_col"],
    "bbox_y1_col": detected_columns["y1_col"],
    "bbox_x2_col": detected_columns["x2_col"],
    "bbox_y2_col": detected_columns["y2_col"],
    "hard_issue_count": int(len(hard_issues)),
    "warning_count": int(len(warnings)),
    "issue_count": int(len(issues_df)),
    "existing_tracking_ready_for_full_integration": bool(existing_tracking_ready_for_full_integration),
    "ready_for_v50b_full_72_tracking_run": bool(ready_for_v50b_full_run),
    "ready_for_v50c_tracking_integration": bool(ready_for_v50c_integration),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}

decision = pd.DataFrame([decision_payload])
safe_to_csv(issues_df, OUT_ISSUES)
safe_to_csv(decision, OUT_DECISION)

report = f"""Week 8 v50a Tracking Source Coverage Audit Report

Decision:
{v50a_decision}

Selected tracking source:
{selected_tracking_path}

Coverage:
- Tracking rows total: {tracking_rows_total}
- Valid tracking bbox rows: {valid_bbox_rows_total}
- Covered clips: {covered_clip_count}/{len(coverage)}
- Missing tracking clips: {missing_clip_count}
- Coverage ratio: {coverage_ratio:.4f}

Detected columns:
- scan column: {detected_columns["scan_col"]}
- frame column: {detected_columns["frame_col"]}
- track column: {detected_columns["track_col"]}
- bbox columns: {detected_columns["x1_col"]}, {detected_columns["y1_col"]}, {detected_columns["x2_col"]}, {detected_columns["y2_col"]}

Interpretation:
v49c validated anchor-frame annotations. v50a checks whether existing tracking outputs are sufficient to replace fixed anchor boxes with tracking-refined per-frame boxes.

If existing tracking covers only a subset of the 72 clips, it should not be integrated as the final Week 8 viewer source. The next step should be a CPU-safe full 72-clip tracking run or a clearly scoped partial demonstration.

Outputs:
- Tracking source candidates: {OUT_TRACKING_SOURCE_CANDIDATES}
- Column inventory: {OUT_TRACKING_COLUMN_INVENTORY}
- Coverage by clip: {OUT_TRACKING_COVERAGE_BY_CLIP}
- Normalized sample: {OUT_NORMALIZED_SAMPLE}
- Issues: {OUT_ISSUES}

Next:
- If ready_for_v50b_full_72_tracking_run=True, run full 72-clip tracking.
- If ready_for_v50c_tracking_integration=True, integrate existing tracking into the visualizer.
"""

OUT_REPORT.write_text(report)

OUT_NOTE.write_text(
    "# Week 8 v50a Tracking Source Coverage Audit\n\n"
    "## Summary\n\n"
    f"- v50a decision: {v50a_decision}\n"
    f"- Selected tracking source: {selected_tracking_path}\n"
    f"- Tracking rows total: {tracking_rows_total}\n"
    f"- Valid tracking bbox rows: {valid_bbox_rows_total}\n"
    f"- Covered clips: {covered_clip_count}/{len(coverage)}\n"
    f"- Missing tracking clips: {missing_clip_count}\n"
    f"- Coverage ratio: {coverage_ratio:.4f}\n"
    f"- Hard issue count: {len(hard_issues)}\n"
    f"- Warning count: {len(warnings)}\n"
    f"- Ready for v50b full 72-clip tracking run: {ready_for_v50b_full_run}\n"
    f"- Ready for v50c tracking integration: {ready_for_v50c_integration}\n\n"
    "## Interpretation\n\n"
    "Existing tracking should only be used for final visual validation if it covers all 72 clips with usable frame, track, and bbox columns. "
    "If coverage is partial, the next step is a full 72-clip tracking run.\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v50a",
    "task_name": "Tracking source coverage audit",
    "status": "PASS" if len(hard_issues) == 0 else "BLOCKED",
    "input_summary": str(selected_tracking_path),
    "output_summary": str(OUT),
    "hard_issues": int(len(hard_issues)),
    "warnings": int(len(warnings)),
    "next_action": "v50b full 72-clip tracking run" if ready_for_v50b_full_run else "v50c tracking integration" if ready_for_v50c_integration else "Resolve tracking source issues.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_TRACKING_SOURCE_CANDIDATES)
print(OUT_TRACKING_COLUMN_INVENTORY)
print(OUT_TRACKING_COVERAGE_BY_CLIP)
print(OUT_NORMALIZED_SAMPLE)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_REPORT)
print(OUT_NOTE)

print()
print("=== v50a decision ===")
print(decision.to_string(index=False))

print()
print("=== detected columns ===")
for k, v in detected_columns.items():
    print(f"{k}: {v}")

print()
print("=== coverage summary ===")
print(coverage["coverage_status"].value_counts(dropna=False).to_string())

print()
print("=== covered clips head ===")
print(coverage[coverage["tracking_rows"] > 0].head(20).to_string(index=False))

print()
print("=== missing clips head ===")
print(coverage[coverage["tracking_rows"] == 0].head(20).to_string(index=False))

print()
print("=== issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
