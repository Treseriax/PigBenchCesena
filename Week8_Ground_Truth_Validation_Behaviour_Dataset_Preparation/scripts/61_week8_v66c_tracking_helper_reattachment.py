from pathlib import Path
from datetime import datetime
import csv
import re
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"
OUTPUTS = W8 / "outputs"

GT_OBJECTS = OUTPUTS / "v66a_final_gt_v2_label_propagation" / "week8_v66a_final_gt_v2_object_table.csv"
GT_SCANFRAME = OUTPUTS / "v66a_final_gt_v2_label_propagation" / "week8_v66a_scanframe_propagation_summary.csv"

OUT = OUTPUTS / "v66c_tracking_helper_reattachment"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_INVENTORY = OUT / "week8_v66c_tracking_helper_file_inventory.csv"
OUT_SCANFRAME = OUT / "week8_v66c_scanframe_tracking_helper_summary.csv"
OUT_OBJECT = OUT / "week8_v66c_final_gt_v2_object_tracking_helper_summary.csv"
OUT_DIRECT = OUT / "week8_v66c_direct_object_tracking_helper_matches.csv"
OUT_DECISION = OUT / "week8_v66c_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v66c_issues.csv"
OUT_NOTE = NOTES / "week8_v66c_tracking_helper_reattachment_notes.md"
OUT_REPORT = REPORTS / "week8_v66c_tracking_helper_reattachment_report.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


def line_count(path):
    try:
        with open(path, "rb") as f:
            return max(sum(1 for _ in f) - 1, 0)
    except Exception:
        return -1


def normalize_scanframe(value):
    s = clean(value)
    m = re.search(r"scanframe[_-]?(\d{4})", s)
    if m:
        return f"scanframe_{m.group(1)}"
    return s


def find_scan_col(cols):
    preferred = [
        "scan_frame_id",
        "scanframe_id",
        "scanframe",
        "scan_frame",
        "clip_id",
        "clip_name",
        "video_clip_id",
    ]
    for c in preferred:
        if c in cols:
            return c
    for c in cols:
        lc = c.lower()
        if "scan" in lc and "frame" in lc:
            return c
    for c in cols:
        lc = c.lower()
        if "clip" in lc and "id" in lc:
            return c
    return ""


def numeric_metric_cols(cols):
    keys = [
        "draw",
        "missing",
        "stable",
        "fallback",
        "coverage",
        "ratio",
        "recall",
        "precision",
        "match",
        "tracked",
        "tracking",
        "quality",
        "score",
        "iou",
    ]
    out = []
    for c in cols:
        lc = c.lower()
        if any(k in lc for k in keys):
            out.append(c)
    return out[:25]


issues = []

if not GT_OBJECTS.exists():
    issues.append({
        "item": str(GT_OBJECTS),
        "issue_type": "hard_missing_final_gt_objects",
        "issue_detail": "Final GT v2 object table missing.",
        "severity": "hard",
    })

if not GT_SCANFRAME.exists():
    issues.append({
        "item": str(GT_SCANFRAME),
        "issue_type": "hard_missing_final_gt_scanframe_summary",
        "issue_detail": "Final GT v2 scanframe summary missing.",
        "severity": "hard",
    })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v66c_decision": "tracking_helper_reattachment_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v67_report_ready_package": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


gt = pd.read_csv(GT_OBJECTS).fillna("")
gt_scan = pd.read_csv(GT_SCANFRAME).fillna("")

for df in [gt, gt_scan]:
    for c in df.columns:
        df[c] = df[c].map(clean) if df[c].dtype == object else df[c]

# Find v52 tracking/helper outputs.
tracking_csvs = []
for p in OUTPUTS.rglob("*.csv"):
    rel = str(p.relative_to(OUTPUTS)).lower()
    if "v52" in rel and ("track" in rel or "hybrid" in rel or "quality" in rel or "helper" in rel):
        tracking_csvs.append(p)

tracking_csvs = sorted(set(tracking_csvs))

inventory_rows = []
scanframe_evidence_rows = []
direct_object_rows = []

for p in tracking_csvs:
    try:
        head = pd.read_csv(p, nrows=5).fillna("")
        cols = list(head.columns)
        rows = line_count(p)
        scan_col = find_scan_col(cols)
        has_object_id = "canonical_gt_object_id" in cols
        metric_cols = numeric_metric_cols(cols)

        inventory_rows.append({
            "file_path": str(p),
            "relative_path": str(p.relative_to(W8)),
            "row_count": rows,
            "column_count": len(cols),
            "scan_column": scan_col,
            "has_canonical_gt_object_id": has_object_id,
            "metric_columns_detected": ";".join(metric_cols),
            "columns_preview": ";".join(cols[:40]),
        })

        # Scanframe-level aggregation.
        if scan_col:
            use_cols = [scan_col] + metric_cols
            if has_object_id:
                use_cols.append("canonical_gt_object_id")

            use_cols = list(dict.fromkeys([c for c in use_cols if c in cols]))
            df = pd.read_csv(p, usecols=use_cols).fillna("")

            for c in df.columns:
                df[c] = df[c].map(clean) if df[c].dtype == object else df[c]

            df["scan_frame_id"] = df[scan_col].map(normalize_scanframe)

            agg = df.groupby("scan_frame_id").size().reset_index(name="helper_row_count")
            agg["source_file"] = str(p)
            agg["source_relative_path"] = str(p.relative_to(W8))
            agg["source_row_count"] = rows

            # Add numeric metric means when possible.
            for mc in metric_cols:
                if mc in df.columns:
                    numeric = pd.to_numeric(df[mc], errors="coerce")
                    tmp = pd.DataFrame({
                        "scan_frame_id": df["scan_frame_id"],
                        mc: numeric,
                    })
                    mean_df = tmp.groupby("scan_frame_id")[mc].mean().reset_index()
                    mean_df = mean_df.rename(columns={mc: f"mean_{mc}"})
                    agg = agg.merge(mean_df, on="scan_frame_id", how="left")

            scanframe_evidence_rows.append(agg)

            if has_object_id:
                direct = df[df["canonical_gt_object_id"].astype(str).str.strip() != ""].copy()
                if len(direct):
                    direct_agg = (
                        direct.groupby("canonical_gt_object_id")
                        .size()
                        .reset_index(name="direct_tracking_helper_rows")
                    )
                    direct_agg["source_file"] = str(p)
                    direct_object_rows.append(direct_agg)

    except Exception as e:
        inventory_rows.append({
            "file_path": str(p),
            "relative_path": str(p.relative_to(W8)),
            "row_count": -1,
            "column_count": -1,
            "scan_column": "",
            "has_canonical_gt_object_id": False,
            "metric_columns_detected": "",
            "columns_preview": "",
            "read_error": str(e),
        })


inventory = pd.DataFrame(inventory_rows)
safe_to_csv(inventory, OUT_INVENTORY)

if len(inventory) == 0:
    issues.append({
        "item": str(OUTPUTS),
        "issue_type": "warning_no_v52_tracking_csv_found",
        "issue_detail": "No v52 tracking/helper CSV files found. Tracking helper layer cannot be attached.",
        "severity": "warning",
    })

if scanframe_evidence_rows:
    scan_ev = pd.concat(scanframe_evidence_rows, ignore_index=True)

    # Collapse to scanframe summary.
    source_count = scan_ev.groupby("scan_frame_id")["source_file"].nunique().reset_index(name="tracking_helper_source_file_count")
    helper_rows = scan_ev.groupby("scan_frame_id")["helper_row_count"].sum().reset_index(name="tracking_helper_total_rows")

    scan_helper = source_count.merge(helper_rows, on="scan_frame_id", how="outer")

    # keep means from available columns
    mean_cols = [c for c in scan_ev.columns if c.startswith("mean_")]
    for mc in mean_cols:
        tmp = scan_ev.groupby("scan_frame_id")[mc].mean().reset_index()
        scan_helper = scan_helper.merge(tmp, on="scan_frame_id", how="left")

else:
    scan_helper = pd.DataFrame({
        "scan_frame_id": gt["scan_frame_id"].drop_duplicates().tolist(),
        "tracking_helper_source_file_count": 0,
        "tracking_helper_total_rows": 0,
    })

# Merge GT scanframe summary.
scan_out = gt_scan.merge(scan_helper, on="scan_frame_id", how="left")
scan_out["tracking_helper_source_file_count"] = scan_out["tracking_helper_source_file_count"].fillna(0).astype(int)
scan_out["tracking_helper_total_rows"] = scan_out["tracking_helper_total_rows"].fillna(0).astype(int)

def support_level(row):
    if row["tracking_helper_total_rows"] <= 0:
        return "no_tracking_helper_available"
    if row["tracking_helper_source_file_count"] >= 2:
        return "tracking_helper_available_multiple_sources"
    return "tracking_helper_available"

scan_out["tracking_helper_support_level"] = scan_out.apply(support_level, axis=1)
scan_out["tracking_usage_boundary"] = "diagnostic_helper_only_does_not_override_manual_gt_v2"

safe_to_csv(scan_out, OUT_SCANFRAME)

# Direct object-level matching if available.
if direct_object_rows:
    direct_all = pd.concat(direct_object_rows, ignore_index=True)
    direct_sum = (
        direct_all.groupby("canonical_gt_object_id")["direct_tracking_helper_rows"]
        .sum()
        .reset_index()
    )
else:
    direct_sum = pd.DataFrame(columns=["canonical_gt_object_id", "direct_tracking_helper_rows"])

safe_to_csv(direct_sum, OUT_DIRECT)

object_out = gt.merge(
    scan_out[[
        "scan_frame_id",
        "tracking_helper_source_file_count",
        "tracking_helper_total_rows",
        "tracking_helper_support_level",
        "tracking_usage_boundary",
    ]],
    on="scan_frame_id",
    how="left",
)

object_out = object_out.merge(direct_sum, on="canonical_gt_object_id", how="left")
object_out["direct_tracking_helper_rows"] = object_out["direct_tracking_helper_rows"].fillna(0).astype(int)

object_out["tracking_helper_scope"] = object_out["direct_tracking_helper_rows"].map(
    lambda x: "direct_object_level_tracking_helper" if x > 0 else "scanframe_level_tracking_helper_only"
)

object_out["final_identity_source"] = "manual_gt_v2"
object_out["tracking_identity_role"] = "helper_diagnostic_not_source_of_truth"

safe_to_csv(object_out, OUT_OBJECT)

# QA / decision.
scanframes_with_helper = int((scan_out["tracking_helper_total_rows"] > 0).sum())
direct_object_matches = int((object_out["direct_tracking_helper_rows"] > 0).sum())

if scanframes_with_helper == 0:
    issues.append({
        "item": "tracking_helper_scanframes",
        "issue_type": "warning_no_scanframe_tracking_helper_attached",
        "issue_detail": "No scanframes received tracking helper rows. This is not blocking for GT, but tracking reattachment is limited.",
        "severity": "warning",
    })

if direct_object_matches == 0:
    issues.append({
        "item": "tracking_helper_objects",
        "issue_type": "info_no_direct_object_tracking_matches",
        "issue_detail": "No direct canonical_gt_object_id matches found in v52 tracking outputs. Reattachment is scanframe-level helper only.",
        "severity": "info",
    })

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

decision = pd.DataFrame([{
    "v66c_decision": "tracking_helper_reattachment_completed" if hard_issue_count == 0 else "tracking_helper_reattachment_has_blocking_issues",
    "final_gt_object_rows": int(len(gt)),
    "final_gt_scanframes": int(gt["scan_frame_id"].nunique()),
    "tracking_csv_files_found": int(len(inventory)),
    "scanframes_with_tracking_helper": scanframes_with_helper,
    "direct_object_tracking_matches": direct_object_matches,
    "strict_gold_objects": int((gt["final_gt_v2_category_v66a"] == "strict_gold_classification").sum()) if "final_gt_v2_category_v66a" in gt.columns else "",
    "caution_objects": int((gt["final_gt_v2_category_v66a"] == "caution_analysis").sum()) if "final_gt_v2_category_v66a" in gt.columns else "",
    "nonusable_objects": int((gt["final_gt_v2_category_v66a"] == "nonusable_fix_or_excluded").sum()) if "final_gt_v2_category_v66a" in gt.columns else "",
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v67_report_ready_package": bool(hard_issue_count == 0),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v66c Tracking Helper Reattachment\n\n"
    f"- v66c decision: {decision.iloc[0]['v66c_decision']}\n"
    f"- Final GT object rows: {len(gt)}\n"
    f"- Final GT scanframes: {gt['scan_frame_id'].nunique()}\n"
    f"- Tracking CSV files found: {len(inventory)}\n"
    f"- Scanframes with tracking helper: {scanframes_with_helper}\n"
    f"- Direct object tracking matches: {direct_object_matches}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Ready for v67 report-ready package: {bool(hard_issue_count == 0)}\n\n"
    "Tracking is attached only as a helper/diagnostic layer. Manual GT v2 remains the identity and label source of truth.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v66c Tracking Helper Reattachment Report\n\n"
    f"Decision: {decision.iloc[0]['v66c_decision']}\n\n"
    f"Inventory: `{OUT_INVENTORY}`\n\n"
    f"Scanframe helper summary: `{OUT_SCANFRAME}`\n\n"
    f"Object helper summary: `{OUT_OBJECT}`\n\n"
    "Important: tracking helper outputs do not override final manual GT v2.\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v66c",
    "task_name": "Tracking helper reattachment",
    "status": "PASS" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": str(GT_OBJECTS),
    "output_summary": str(OUT),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Build Week8 report-ready package on solid GT v2 foundation.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("Saved:")
print(OUT_INVENTORY)
print(OUT_SCANFRAME)
print(OUT_OBJECT)
print(OUT_DIRECT)
print(OUT_DECISION)
print(OUT_NOTE)

print()
print("=== v66c decision ===")
print(decision.to_string(index=False))

print()
print("=== inventory top ===")
if len(inventory):
    cols = ["relative_path", "row_count", "scan_column", "has_canonical_gt_object_id", "metric_columns_detected"]
    print(inventory[cols].head(30).to_string(index=False))
else:
    print("No tracking CSVs found.")

print()
print("=== scanframe helper counts ===")
print(scan_out["tracking_helper_support_level"].value_counts().to_string())

print()
print("=== issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
