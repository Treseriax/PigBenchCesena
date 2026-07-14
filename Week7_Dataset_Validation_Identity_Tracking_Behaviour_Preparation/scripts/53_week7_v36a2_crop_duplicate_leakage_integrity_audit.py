from pathlib import Path
from datetime import datetime
import csv
import json
import pandas as pd


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V36A = W7 / "outputs" / "week7_crop_feature_extraction_preflight_v36a"

BASELINE_IN = V36A / "week7_v36a_crop_baseline_ready_subset.csv"
LOADING_AUDIT_IN = V36A / "week7_v36a_crop_image_loading_audit.csv"
V36A_CONFIG_IN = V36A / "week7_v36a_crop_feature_extraction_config.json"

OUT_ROOT = W7 / "outputs" / "week7_crop_duplicate_leakage_integrity_audit_v36a2"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_DUP_GROUPS = OUT_ROOT / "week7_v36a2_duplicate_crop_path_groups.csv"
OUT_DUP_ROWS = OUT_ROOT / "week7_v36a2_duplicate_crop_path_rows.csv"
OUT_CONFLICTS = OUT_ROOT / "week7_v36a2_leakage_or_label_conflict_groups.csv"
OUT_DEDUP_INDEX = OUT_ROOT / "week7_v36a2_deduplicated_crop_baseline_ready_subset.csv"
OUT_REMOVED_DUPLICATES = OUT_ROOT / "week7_v36a2_removed_duplicate_rows.csv"
OUT_ORIGINAL_VS_DEDUP_COUNTS = OUT_ROOT / "week7_v36a2_original_vs_dedup_class_split_counts.csv"
OUT_SAMPLE_ID_AUDIT = OUT_ROOT / "week7_v36a2_sample_id_integrity_audit.csv"
OUT_V36B_CONFIG = OUT_ROOT / "week7_v36a2_recommended_v36b_feature_extraction_config.json"
OUT_REPORT = OUT_ROOT / "week7_v36a2_crop_duplicate_leakage_integrity_report.md"
OUT_DECISION = OUT_ROOT / "week7_v36a2_crop_duplicate_leakage_integrity_decision_summary.csv"
OUT_ISSUES = OUT_ROOT / "week7_v36a2_crop_duplicate_leakage_integrity_issues.csv"
OUT_NOTE = W7 / "notes" / "week7_v36a2_crop_duplicate_leakage_integrity_notes.md"


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


def read_df(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def resolve_path(value):
    s = clean(value)
    if not s:
        return ""

    p = Path(s)

    if p.is_absolute():
        return str(p)

    candidates = [
        ROOT / p,
        W7 / p,
        Path.cwd() / p,
    ]

    for c in candidates:
        if c.exists():
            return str(c)

    return str(ROOT / p)


def md_table(rows, columns):
    if not rows:
        return "_No rows available._"

    out = []
    out.append("| " + " | ".join(columns) + " |")
    out.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for r in rows:
        vals = []
        for c in columns:
            x = str(r.get(c, ""))
            x = x.replace("\n", " ").replace("|", "/")
            vals.append(x)
        out.append("| " + " | ".join(vals) + " |")

    return "\n".join(out)


def count_by_class_split(df, label_col, split_col):
    behaviours = sorted(df[label_col].dropna().astype(str).unique().tolist())
    rows = []

    for b in behaviours:
        g = df[df[label_col].astype(str) == b]

        row = {
            "behaviour_code": b,
            "total_rows": int(len(g)),
            "train_rows": int((g[split_col].astype(str) == "train").sum()),
            "val_rows": int((g[split_col].astype(str) == "val").sum()),
            "test_rows": int((g[split_col].astype(str) == "test").sum()),
        }

        rows.append(row)

    return pd.DataFrame(rows)


issues = []

baseline = read_df(BASELINE_IN)
loading_audit = read_df(LOADING_AUDIT_IN)
config = read_json(V36A_CONFIG_IN)

if len(baseline) == 0:
    issues.append({
        "item": "baseline_subset",
        "issue_type": "hard_missing_or_empty_input",
        "issue_detail": str(BASELINE_IN),
    })

if not config:
    issues.append({
        "item": "v36a_config",
        "issue_type": "hard_missing_or_unreadable_config",
        "issue_detail": str(V36A_CONFIG_IN),
    })

if len(issues):
    issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
    safe_to_csv(issues_df, OUT_ISSUES)
    print("Hard input issue:")
    print(issues_df.to_string(index=False))
    raise SystemExit(1)

feature_input_column = config.get("feature_input_column", "crop_path")
label_column = config.get("label_column", "behaviour_code")
split_column = config.get("split_column", "split")
eligible_classes = config.get("eligible_classes", [])

sample_id_col = "v36a_baseline_sample_id"
if sample_id_col not in baseline.columns:
    sample_id_col = "v34_crop_sample_id" if "v34_crop_sample_id" in baseline.columns else ""

for col in [feature_input_column, label_column, split_column]:
    if col not in baseline.columns:
        issues.append({
            "item": col,
            "issue_type": "hard_missing_required_column",
            "issue_detail": f"Column `{col}` missing from baseline subset.",
        })

if len(issues):
    issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
    safe_to_csv(issues_df, OUT_ISSUES)
    print("Hard column issue:")
    print(issues_df.to_string(index=False))
    raise SystemExit(1)

baseline = baseline.copy().reset_index(drop=True)
baseline["v36a2_original_row_index"] = baseline.index
baseline["resolved_crop_path"] = baseline[feature_input_column].map(resolve_path)
baseline["resolved_crop_path_exists"] = baseline["resolved_crop_path"].map(lambda x: bool(x and Path(x).exists()))

baseline[label_column] = baseline[label_column].fillna("").astype(str).str.strip()
baseline[split_column] = baseline[split_column].fillna("").astype(str).str.strip()

if "scan_frame_id" in baseline.columns:
    baseline["scan_frame_id"] = baseline["scan_frame_id"].fillna("").astype(str).str.strip()

if sample_id_col and sample_id_col in baseline.columns:
    baseline[sample_id_col] = baseline[sample_id_col].fillna("").astype(str).str.strip()

# ---------------------------------------------------------------------
# Sample ID integrity audit
# ---------------------------------------------------------------------

sample_rows = []

if sample_id_col and sample_id_col in baseline.columns:
    sid_counts = baseline[sample_id_col].value_counts(dropna=False)

    for sid, cnt in sid_counts.items():
        if cnt > 1:
            g = baseline[baseline[sample_id_col] == sid]
            sample_rows.append({
                "sample_id": sid,
                "row_count": int(cnt),
                "split_values": " | ".join(sorted(g[split_column].dropna().astype(str).unique().tolist())),
                "label_values": " | ".join(sorted(g[label_column].dropna().astype(str).unique().tolist())),
                "crop_path_count": int(g["resolved_crop_path"].nunique()),
            })

sample_id_audit = pd.DataFrame(sample_rows)
safe_to_csv(sample_id_audit, OUT_SAMPLE_ID_AUDIT)

duplicate_sample_id_count = int(len(sample_id_audit))

# ---------------------------------------------------------------------
# Duplicate crop path groups
# ---------------------------------------------------------------------

group_rows = []
duplicate_row_frames = []
conflict_rows = []

valid_path_df = baseline[baseline["resolved_crop_path"].astype(str).str.len() > 0].copy()

for path, g in valid_path_df.groupby("resolved_crop_path", dropna=False):
    row_count = int(len(g))
    split_values = sorted(g[split_column].dropna().astype(str).unique().tolist())
    label_values = sorted(g[label_column].dropna().astype(str).unique().tolist())
    scan_values = sorted(g["scan_frame_id"].dropna().astype(str).unique().tolist()) if "scan_frame_id" in g.columns else []
    sample_values = sorted(g[sample_id_col].dropna().astype(str).unique().tolist()) if sample_id_col and sample_id_col in g.columns else []

    split_count = len(split_values)
    label_count = len(label_values)

    if row_count == 1:
        duplicate_type = "unique"
    elif split_count > 1 and label_count > 1:
        duplicate_type = "cross_split_leakage_and_label_conflict"
    elif split_count > 1:
        duplicate_type = "cross_split_leakage"
    elif label_count > 1:
        duplicate_type = "same_path_label_conflict"
    else:
        duplicate_type = "same_path_same_split_same_label_duplicate"

    is_duplicate = row_count > 1
    is_hard_conflict = duplicate_type in [
        "cross_split_leakage_and_label_conflict",
        "cross_split_leakage",
        "same_path_label_conflict",
    ]

    group_row = {
        "resolved_crop_path": path,
        "row_count": row_count,
        "duplicate_extra_rows": max(0, row_count - 1),
        "split_count": split_count,
        "split_values": " | ".join(split_values),
        "label_count": label_count,
        "label_values": " | ".join(label_values),
        "scan_frame_count": len(scan_values),
        "scan_frame_values": " | ".join(scan_values),
        "sample_id_count": len(sample_values),
        "sample_id_values": " | ".join(sample_values),
        "duplicate_type": duplicate_type,
        "is_duplicate_group": bool(is_duplicate),
        "is_hard_conflict": bool(is_hard_conflict),
    }

    group_rows.append(group_row)

    if is_duplicate:
        tmp = g.copy()
        tmp["duplicate_type"] = duplicate_type
        tmp["duplicate_group_row_count"] = row_count
        duplicate_row_frames.append(tmp)

    if is_hard_conflict:
        conflict_rows.append(group_row)

duplicate_groups = pd.DataFrame(group_rows)
duplicate_groups_sorted = duplicate_groups.sort_values(
    ["is_hard_conflict", "is_duplicate_group", "row_count", "resolved_crop_path"],
    ascending=[False, False, False, True],
)

safe_to_csv(duplicate_groups_sorted, OUT_DUP_GROUPS)

if duplicate_row_frames:
    duplicate_rows = pd.concat(duplicate_row_frames, ignore_index=True)
else:
    duplicate_rows = pd.DataFrame(columns=list(baseline.columns) + ["duplicate_type", "duplicate_group_row_count"])

safe_to_csv(duplicate_rows, OUT_DUP_ROWS)

conflicts = pd.DataFrame(conflict_rows)
safe_to_csv(conflicts, OUT_CONFLICTS)

# ---------------------------------------------------------------------
# Deduplicated baseline index
# ---------------------------------------------------------------------

# Keep one row per resolved crop path. This removes pure duplicate rows.
# If hard conflicts exist, the dedup file is still written, but decision will not mark it as clean.
sort_cols = []
if split_column in baseline.columns:
    split_order = {"train": 0, "val": 1, "test": 2}
    baseline["_split_sort"] = baseline[split_column].map(lambda x: split_order.get(str(x), 9))
    sort_cols.append("_split_sort")

sort_cols += ["resolved_crop_path", "v36a2_original_row_index"]

dedup = baseline.sort_values(sort_cols).drop_duplicates(
    subset=["resolved_crop_path"],
    keep="first",
).copy()

removed = baseline[~baseline["v36a2_original_row_index"].isin(dedup["v36a2_original_row_index"])].copy()

if "_split_sort" in dedup.columns:
    dedup = dedup.drop(columns=["_split_sort"])
if "_split_sort" in removed.columns:
    removed = removed.drop(columns=["_split_sort"])
if "_split_sort" in baseline.columns:
    baseline = baseline.drop(columns=["_split_sort"])

dedup = dedup.reset_index(drop=True)
dedup["v36a2_dedup_sample_id"] = ["v36a2_dedup_%04d" % i for i in range(len(dedup))]

safe_to_csv(dedup, OUT_DEDUP_INDEX)
safe_to_csv(removed, OUT_REMOVED_DUPLICATES)

# ---------------------------------------------------------------------
# Original vs dedup class/split count comparison
# ---------------------------------------------------------------------

orig_counts = count_by_class_split(baseline, label_column, split_column)
dedup_counts = count_by_class_split(dedup, label_column, split_column)

comparison = orig_counts.merge(
    dedup_counts,
    on="behaviour_code",
    how="outer",
    suffixes=("_original", "_dedup"),
).fillna(0)

for col in [
    "total_rows_original",
    "train_rows_original",
    "val_rows_original",
    "test_rows_original",
    "total_rows_dedup",
    "train_rows_dedup",
    "val_rows_dedup",
    "test_rows_dedup",
]:
    if col in comparison.columns:
        comparison[col] = comparison[col].astype(int)

comparison["removed_rows"] = comparison["total_rows_original"] - comparison["total_rows_dedup"]

comparison["dedup_has_train_val_test"] = (
    (comparison["train_rows_dedup"] > 0)
    & (comparison["val_rows_dedup"] > 0)
    & (comparison["test_rows_dedup"] > 0)
)

comparison["eligible_for_main_crop_baseline"] = comparison["behaviour_code"].isin(eligible_classes)

safe_to_csv(comparison, OUT_ORIGINAL_VS_DEDUP_COUNTS)

# ---------------------------------------------------------------------
# Issues and decision
# ---------------------------------------------------------------------

total_baseline_rows = int(len(baseline))
unique_crop_paths = int(baseline["resolved_crop_path"].nunique())
duplicate_group_count = int((duplicate_groups["is_duplicate_group"] == True).sum()) if len(duplicate_groups) else 0
duplicate_rows_total = int(len(duplicate_rows))
duplicate_extra_rows = int(duplicate_groups["duplicate_extra_rows"].sum()) if len(duplicate_groups) else 0

cross_split_leakage_groups = int((duplicate_groups["duplicate_type"].isin([
    "cross_split_leakage",
    "cross_split_leakage_and_label_conflict",
])).sum()) if len(duplicate_groups) else 0

label_conflict_groups = int((duplicate_groups["duplicate_type"].isin([
    "same_path_label_conflict",
    "cross_split_leakage_and_label_conflict",
])).sum()) if len(duplicate_groups) else 0

hard_issue_list = []
warning_list = []

if cross_split_leakage_groups > 0:
    hard_issue_list.append({
        "item": "cross_split_leakage",
        "issue_type": "hard_duplicate_crop_path_cross_split_leakage",
        "issue_detail": f"{cross_split_leakage_groups} duplicate crop path groups appear across multiple splits.",
    })

if label_conflict_groups > 0:
    hard_issue_list.append({
        "item": "label_conflict",
        "issue_type": "hard_duplicate_crop_path_label_conflict",
        "issue_detail": f"{label_conflict_groups} duplicate crop path groups have multiple behaviour labels.",
    })

if duplicate_extra_rows > 0:
    warning_list.append({
        "item": "pure_duplicate_rows",
        "issue_type": "warning_duplicate_rows_removed_in_dedup_index",
        "issue_detail": f"{duplicate_extra_rows} extra duplicate rows can be removed using the v36a2 dedup index.",
    })

if duplicate_sample_id_count > 0:
    warning_list.append({
        "item": "duplicate_sample_ids",
        "issue_type": "warning_duplicate_sample_ids_found",
        "issue_detail": f"{duplicate_sample_id_count} sample IDs appear more than once.",
    })

# Check eligible classes after dedup.
dedup_eligible = comparison[comparison["eligible_for_main_crop_baseline"] == True].copy()

missing_split_classes = dedup_eligible[dedup_eligible["dedup_has_train_val_test"] == False]

if len(missing_split_classes):
    hard_issue_list.append({
        "item": "dedup_class_split_coverage",
        "issue_type": "hard_dedup_baseline_class_missing_split",
        "issue_detail": "After deduplication, at least one eligible baseline class is missing train/val/test coverage: "
                        + " | ".join(missing_split_classes["behaviour_code"].astype(str).tolist()),
    })

all_issues = hard_issue_list + warning_list
issues_df = pd.DataFrame(all_issues, columns=["item", "issue_type", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int(len(hard_issue_list))
warning_count = int(len(warning_list))

ready_for_v36b = (
    hard_issue_count == 0
    and len(dedup) > 0
    and len(dedup_eligible) > 0
)

v36b_config = {
    "stage": "v36a2_crop_duplicate_leakage_integrity_audit",
    "recommended_input_for_v36b": str(OUT_DEDUP_INDEX),
    "original_input": str(BASELINE_IN),
    "label_column": label_column,
    "split_column": split_column,
    "feature_input_column": feature_input_column,
    "resolved_path_column": "resolved_crop_path",
    "eligible_classes": eligible_classes,
    "duplicate_policy": "Use deduplicated index for main baseline feature extraction.",
    "hard_conflict_policy": "Do not proceed if duplicate crop paths cross splits or have conflicting labels.",
    "same_split_same_label_duplicate_policy": "Remove duplicate rows to avoid overweighting identical crop images.",
    "ready_for_v36b": bool(ready_for_v36b),
}

OUT_V36B_CONFIG.write_text(json.dumps(v36b_config, indent=2))

decision = pd.DataFrame([{
    "v36a2_decision": "crop_duplicate_leakage_integrity_audit_passed" if ready_for_v36b else "crop_duplicate_leakage_integrity_audit_issues_found",
    "original_baseline_rows": int(total_baseline_rows),
    "unique_crop_paths": int(unique_crop_paths),
    "duplicate_group_count": int(duplicate_group_count),
    "duplicate_rows_total": int(duplicate_rows_total),
    "duplicate_extra_rows_removed_in_dedup_index": int(duplicate_extra_rows),
    "deduplicated_baseline_rows": int(len(dedup)),
    "cross_split_leakage_groups": int(cross_split_leakage_groups),
    "label_conflict_groups": int(label_conflict_groups),
    "duplicate_sample_id_count": int(duplicate_sample_id_count),
    "eligible_classes": " | ".join(eligible_classes),
    "hard_issue_count": int(hard_issue_count),
    "warning_count": int(warning_count),
    "issue_count": int(len(issues_df)),
    "ready_for_v36b_full_crop_feature_extraction": bool(ready_for_v36b),
    "recommended_v36b_input": str(OUT_DEDUP_INDEX),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

# ---------------------------------------------------------------------
# Report and note
# ---------------------------------------------------------------------

dup_summary_rows = duplicate_groups_sorted[duplicate_groups_sorted["is_duplicate_group"] == True].head(20).to_dict("records")
dup_table = md_table(
    dup_summary_rows,
    [
        "row_count",
        "duplicate_extra_rows",
        "split_values",
        "label_values",
        "scan_frame_values",
        "duplicate_type",
        "is_hard_conflict",
    ],
)

comparison_table = md_table(
    comparison.to_dict("records"),
    [
        "behaviour_code",
        "total_rows_original",
        "total_rows_dedup",
        "removed_rows",
        "train_rows_dedup",
        "val_rows_dedup",
        "test_rows_dedup",
        "eligible_for_main_crop_baseline",
    ],
)

report = f"""# Week 7 v36a2 Crop Duplicate / Leakage / Sample Integrity Audit

## Purpose

This audit checks whether the crop-level baseline subset can be safely used for full crop feature extraction.

It focuses on duplicate crop paths, cross-split leakage, conflicting labels and whether a deduplicated baseline index should be used for v36b.

## Inputs

- Original v36a baseline subset: `{BASELINE_IN}`
- v36a loading audit: `{LOADING_AUDIT_IN}`
- v36a config: `{V36A_CONFIG_IN}`

## Main results

- Original baseline rows: `{total_baseline_rows}`
- Unique crop paths: `{unique_crop_paths}`
- Duplicate groups: `{duplicate_group_count}`
- Duplicate rows total: `{duplicate_rows_total}`
- Duplicate extra rows removed in dedup index: `{duplicate_extra_rows}`
- Deduplicated baseline rows: `{len(dedup)}`
- Cross-split leakage groups: `{cross_split_leakage_groups}`
- Label conflict groups: `{label_conflict_groups}`
- Duplicate sample ID count: `{duplicate_sample_id_count}`
- Hard issue count: `{hard_issue_count}`
- Warning count: `{warning_count}`
- Ready for v36b: `{ready_for_v36b}`

## Duplicate group preview

{dup_table}

## Original vs deduplicated class/split counts

{comparison_table}

## Interpretation

If duplicate crop paths are repeated only within the same split and same label, they are not leakage, but they can overweight identical images. Therefore the deduplicated index is recommended for v36b full feature extraction.

If cross-split leakage or label conflict exists, v36b should not proceed until those conflicts are resolved.

## Recommended v36b input

`{OUT_DEDUP_INDEX}`
"""

OUT_REPORT.write_text(report)

OUT_NOTE.write_text(
    "# Week 7 v36a2 Crop Duplicate / Leakage / Sample Integrity Audit\n\n"
    "## Summary\n\n"
    f"- Original baseline rows: `{total_baseline_rows}`\n"
    f"- Unique crop paths: `{unique_crop_paths}`\n"
    f"- Duplicate groups: `{duplicate_group_count}`\n"
    f"- Duplicate rows total: `{duplicate_rows_total}`\n"
    f"- Duplicate extra rows removed in dedup index: `{duplicate_extra_rows}`\n"
    f"- Deduplicated baseline rows: `{len(dedup)}`\n"
    f"- Cross-split leakage groups: `{cross_split_leakage_groups}`\n"
    f"- Label conflict groups: `{label_conflict_groups}`\n"
    f"- Duplicate sample ID count: `{duplicate_sample_id_count}`\n"
    f"- Hard issue count: `{hard_issue_count}`\n"
    f"- Warning count: `{warning_count}`\n"
    f"- Ready for v36b full crop feature extraction: `{ready_for_v36b}`\n\n"
    "## Outputs\n\n"
    f"- Duplicate groups: `{OUT_DUP_GROUPS}`\n"
    f"- Duplicate rows: `{OUT_DUP_ROWS}`\n"
    f"- Conflicts: `{OUT_CONFLICTS}`\n"
    f"- Deduplicated index: `{OUT_DEDUP_INDEX}`\n"
    f"- Removed duplicate rows: `{OUT_REMOVED_DUPLICATES}`\n"
    f"- Original vs dedup counts: `{OUT_ORIGINAL_VS_DEDUP_COUNTS}`\n"
    f"- Recommended v36b config: `{OUT_V36B_CONFIG}`\n"
    f"- Report: `{OUT_REPORT}`\n"
    f"- Decision: `{OUT_DECISION}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
)

print("Saved:")
print(OUT_DUP_GROUPS)
print(OUT_DUP_ROWS)
print(OUT_CONFLICTS)
print(OUT_DEDUP_INDEX)
print(OUT_REMOVED_DUPLICATES)
print(OUT_ORIGINAL_VS_DEDUP_COUNTS)
print(OUT_SAMPLE_ID_AUDIT)
print(OUT_V36B_CONFIG)
print(OUT_REPORT)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_NOTE)

print()
print("=== v36a2 decision ===")
print(decision.to_string(index=False))

print()
print("=== v36a2 issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
