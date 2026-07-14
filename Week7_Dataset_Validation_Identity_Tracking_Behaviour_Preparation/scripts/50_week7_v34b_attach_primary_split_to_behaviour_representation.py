from pathlib import Path
from datetime import datetime
import csv
import pandas as pd


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V34 = W7 / "outputs" / "week7_baseline_behaviour_representation_dataset_v34"

CROP_IN = V34 / "week7_v34_crop_level_baseline_behaviour_index.csv"
CLIP_IN = V34 / "week7_v34_clip_level_multilabel_behaviour_index.csv"
CLASS_IN = V34 / "week7_v34_behaviour_class_readiness.csv"

OUT_ROOT = W7 / "outputs" / "week7_baseline_behaviour_representation_dataset_v34b_split_fixed"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_CROP = OUT_ROOT / "week7_v34b_crop_level_baseline_behaviour_index_split_fixed.csv"
OUT_CLIP = OUT_ROOT / "week7_v34b_clip_level_multilabel_behaviour_index_split_fixed.csv"
OUT_SPLIT_MAP = OUT_ROOT / "week7_v34b_primary_split_map_used.csv"
OUT_SPLIT_READINESS = OUT_ROOT / "week7_v34b_split_readiness_checks.csv"
OUT_CLASS_SPLIT = OUT_ROOT / "week7_v34b_class_by_split_readiness.csv"
OUT_CANDIDATES = OUT_ROOT / "week7_v34b_split_source_candidates.csv"
OUT_DECISION = OUT_ROOT / "week7_v34b_split_fix_decision_summary.csv"
OUT_ISSUES = OUT_ROOT / "week7_v34b_split_fix_issues.csv"
OUT_REPORT = OUT_ROOT / "week7_v34b_split_fix_report.md"
OUT_NOTE = W7 / "notes" / "week7_v34b_split_fix_notes.md"


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


def norm_split(v):
    s = clean(v).lower()
    if s in ["train", "training", "tr"]:
        return "train"
    if s in ["val", "valid", "validation", "dev"]:
        return "val"
    if s in ["test", "testing", "te"]:
        return "test"
    return clean(v)


def find_scan_col(df):
    for c in df.columns:
        if c == "scan_frame_id":
            return c
    for c in df.columns:
        cl = c.lower()
        if "scan" in cl and "frame" in cl:
            return c
    return None


def find_split_col(df):
    exact = ["split", "primary_split", "subset", "set", "dataset_split"]
    for c in exact:
        if c in df.columns:
            return c

    for c in df.columns:
        cl = c.lower()
        if "split" in cl or cl in ["subset", "set"]:
            return c

    return None


def candidate_score(path, df, scan_col, split_col):
    s = str(path).lower()
    vals = set(df[split_col].dropna().astype(str).str.lower().map(norm_split).unique().tolist())

    score = 0
    if "primary_split_v21" in s:
        score += 100
    if "v21" in s:
        score += 80
    if "final_audit_package_v24" in s:
        score += 20
    if "summary" in s:
        score -= 30
    if "issues" in s:
        score -= 100
    if "manifest" in s:
        score -= 100
    if "decision" in s:
        score -= 60
    if {"train", "val", "test"}.issubset(vals):
        score += 100
    elif {"train", "val"}.issubset(vals) or {"train", "test"}.issubset(vals):
        score += 40

    score += min(len(df), 500) / 10
    score += df[scan_col].nunique()

    return score


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


issues = []

crop = read_df(CROP_IN)
clip = read_df(CLIP_IN)
class_ready = read_df(CLASS_IN)

for key, path, df in [
    ("v34_crop_index", CROP_IN, crop),
    ("v34_clip_index", CLIP_IN, clip),
]:
    if not path.exists() or len(df) == 0:
        issues.append({
            "item": key,
            "issue_type": "missing_or_empty_required_input",
            "issue_detail": str(path),
        })

if len(issues):
    issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
    safe_to_csv(issues_df, OUT_ISSUES)
    print(issues_df.to_string(index=False))
    raise SystemExit(1)

for df in [crop, clip]:
    if "scan_frame_id" in df.columns:
        df["scan_frame_id"] = df["scan_frame_id"].fillna("").astype(str).str.strip()

# ---------------------------------------------------------------------
# Search for the best primary split source.
# ---------------------------------------------------------------------

candidate_rows = []

for path in sorted((W7 / "outputs").rglob("*.csv")):
    s = str(path).lower()

    if "week7_baseline_behaviour_representation_dataset_v34b_split_fixed" in s:
        continue

    # Avoid huge irrelevant linked detection rows when possible.
    if any(bad in s for bad in ["linked_track_detections", "zip_inventory", "file_inventory"]):
        continue

    try:
        df = pd.read_csv(path)
    except Exception:
        continue

    if len(df) == 0:
        continue

    scan_col = find_scan_col(df)
    split_col = find_split_col(df)

    if scan_col is None or split_col is None:
        continue

    vals = sorted(set(df[split_col].dropna().astype(str).map(norm_split).unique().tolist()))
    valid_like = any(v in vals for v in ["train", "val", "test"])

    if not valid_like:
        continue

    score = candidate_score(path, df, scan_col, split_col)

    candidate_rows.append({
        "path": str(path),
        "rows": int(len(df)),
        "scan_col": scan_col,
        "split_col": split_col,
        "unique_scan_frames": int(df[scan_col].nunique()),
        "split_values": " | ".join(vals),
        "score": round(float(score), 4),
    })

candidates = pd.DataFrame(candidate_rows).sort_values("score", ascending=False) if candidate_rows else pd.DataFrame()
safe_to_csv(candidates, OUT_CANDIDATES)

if len(candidates) == 0:
    issues.append({
        "item": "primary_split_source",
        "issue_type": "no_split_source_found",
        "issue_detail": "Could not find a CSV with scan_frame_id and train/val/test split information.",
    })

    issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
    safe_to_csv(issues_df, OUT_ISSUES)

    decision = pd.DataFrame([{
        "v34b_decision": "split_fix_failed_no_split_source_found",
        "split_source_found": False,
        "crop_rows": int(len(crop)),
        "clip_rows": int(len(clip)),
        "issue_count": int(len(issues_df)),
        "ready_for_v35": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)

    print("No split source found.")
    print(issues_df.to_string(index=False))
    raise SystemExit(0)

best = candidates.iloc[0].to_dict()
best_path = Path(best["path"])
best_df = pd.read_csv(best_path)
scan_col = best["scan_col"]
split_col = best["split_col"]

best_df[scan_col] = best_df[scan_col].fillna("").astype(str).str.strip()
best_df[split_col] = best_df[split_col].map(norm_split)

# Build scan_frame_id -> split by mode.
split_map_rows = []
for scan_id, g in best_df.groupby(scan_col):
    scan_id = clean(scan_id)
    if not scan_id:
        continue

    vals = g[split_col].dropna().astype(str).map(norm_split)
    vals = vals[vals.astype(str).str.len() > 0]

    if len(vals) == 0:
        continue

    mode_val = vals.mode().iloc[0]

    split_map_rows.append({
        "scan_frame_id": scan_id,
        "split": mode_val,
        "source_row_count": int(len(g)),
        "unique_split_values_in_source": " | ".join(sorted(vals.unique().tolist())),
        "source_path": str(best_path),
    })

split_map = pd.DataFrame(split_map_rows)
safe_to_csv(split_map, OUT_SPLIT_MAP)

map_dict = dict(zip(split_map["scan_frame_id"], split_map["split"]))

# ---------------------------------------------------------------------
# Apply split to crop and clip index.
# ---------------------------------------------------------------------

crop_fixed = crop.copy()
clip_fixed = clip.copy()

crop_fixed["split"] = crop_fixed["scan_frame_id"].map(map_dict).fillna("")
clip_fixed["split"] = clip_fixed["scan_frame_id"].map(map_dict).fillna("")

crop_unmapped = int((crop_fixed["split"].astype(str).str.len() == 0).sum())
clip_unmapped = int((clip_fixed["split"].astype(str).str.len() == 0).sum())

if crop_unmapped > 0:
    issues.append({
        "item": "crop_split_mapping",
        "issue_type": "unmapped_crop_rows_warning",
        "issue_detail": f"{crop_unmapped} crop rows could not be mapped to split.",
    })

if clip_unmapped > 0:
    issues.append({
        "item": "clip_split_mapping",
        "issue_type": "unmapped_clip_rows_warning",
        "issue_detail": f"{clip_unmapped} clip rows could not be mapped to split.",
    })

safe_to_csv(crop_fixed, OUT_CROP)
safe_to_csv(clip_fixed, OUT_CLIP)

# ---------------------------------------------------------------------
# Split readiness.
# ---------------------------------------------------------------------

split_rows = []
behaviours = sorted(crop_fixed["behaviour_code"].dropna().astype(str).unique().tolist())

for split_name, g in crop_fixed[crop_fixed["split"].astype(str).str.len() > 0].groupby("split"):
    row = {
        "split": split_name,
        "crop_level_rows": int(len(g)),
        "unique_scan_frames": int(g["scan_frame_id"].nunique()),
        "unique_behaviours": int(g["behaviour_code"].nunique()),
        "behaviour_set": " | ".join(sorted(g["behaviour_code"].dropna().astype(str).unique().tolist())),
    }

    for b in behaviours:
        row[f"count__{b}"] = int((g["behaviour_code"] == b).sum())

    split_rows.append(row)

split_readiness = pd.DataFrame(split_rows)
safe_to_csv(split_readiness, OUT_SPLIT_READINESS)

# ---------------------------------------------------------------------
# Class by split readiness.
# ---------------------------------------------------------------------

class_rows = []

for b in behaviours:
    row = {
        "behaviour_code": b,
        "total_crop_rows": int((crop_fixed["behaviour_code"] == b).sum()),
    }

    present_splits = []
    for split_name in ["train", "val", "test"]:
        cnt = int(((crop_fixed["behaviour_code"] == b) & (crop_fixed["split"] == split_name)).sum())
        row[f"{split_name}_rows"] = cnt
        if cnt > 0:
            present_splits.append(split_name)

    row["splits_present"] = " | ".join(present_splits)
    row["present_in_all_splits"] = set(["train", "val", "test"]).issubset(set(present_splits))

    total = row["total_crop_rows"]

    if total < 10:
        row["readiness"] = "report_only_or_group_until_more_data"
    elif row["present_in_all_splits"] and total >= 25:
        row["readiness"] = "baseline_ready_all_splits"
    elif total >= 25:
        row["readiness"] = "baseline_ready_but_split_coverage_warning"
    else:
        row["readiness"] = "limited_use_with_caution"

    class_rows.append(row)

class_by_split = pd.DataFrame(class_rows).sort_values(
    ["total_crop_rows", "behaviour_code"],
    ascending=[False, True],
)
safe_to_csv(class_by_split, OUT_CLASS_SPLIT)

# ---------------------------------------------------------------------
# Final decision.
# ---------------------------------------------------------------------

# Treat unmapped scanframe_0000/0001 clips as expected if they have no crop rows.
# But keep the count visible.
issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)

split_source_found = True
split_values = sorted(crop_fixed["split"].dropna().astype(str).unique().tolist())
nonempty_split_values = [s for s in split_values if s]

has_train_val_test = set(["train", "val", "test"]).issubset(set(nonempty_split_values))
mapped_crop_rows = int((crop_fixed["split"].astype(str).str.len() > 0).sum())
mapped_clip_rows = int((clip_fixed["split"].astype(str).str.len() > 0).sum())

hard_issue_count = len([
    x for x in issues
    if not str(x["issue_type"]).endswith("_warning")
])

ready_for_v35 = (
    split_source_found
    and has_train_val_test
    and mapped_crop_rows > 0
    and hard_issue_count == 0
)

decision = pd.DataFrame([{
    "v34b_decision": "primary_split_attached_to_behaviour_representation" if ready_for_v35 else "split_fix_created_with_warnings_or_issues",
    "split_source_found": bool(split_source_found),
    "split_source_path": str(best_path),
    "split_source_score": best["score"],
    "crop_rows": int(len(crop_fixed)),
    "clip_rows": int(len(clip_fixed)),
    "mapped_crop_rows": mapped_crop_rows,
    "mapped_clip_rows": mapped_clip_rows,
    "unmapped_crop_rows": crop_unmapped,
    "unmapped_clip_rows": clip_unmapped,
    "split_values": " | ".join(nonempty_split_values),
    "has_train_val_test": bool(has_train_val_test),
    "class_rows": int(len(class_by_split)),
    "classes_present_in_all_splits": int(class_by_split["present_in_all_splits"].sum()),
    "hard_issue_count": int(hard_issue_count),
    "issue_count": int(len(issues_df)),
    "ready_for_v35_baseline_experiment_or_feature_extraction": bool(ready_for_v35),
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

split_table = md_table(
    split_readiness.to_dict("records"),
    ["split", "crop_level_rows", "unique_scan_frames", "unique_behaviours", "behaviour_set"],
)

class_table = md_table(
    class_by_split.to_dict("records"),
    ["behaviour_code", "total_crop_rows", "train_rows", "val_rows", "test_rows", "splits_present", "readiness"],
)

report = f"""# Week 7 v34b Split Fix Report

## Purpose

v34 created the baseline behaviour representation dataset, but the crop-level index did not contain the primary split assignment. This v34b step attaches the primary split back to the crop-level and clip-level behaviour representation indices.

## Split source used

- Source path: `{best_path}`
- Scan column: `{scan_col}`
- Split column: `{split_col}`
- Source score: `{best['score']}`

## Summary

- Crop rows: `{len(crop_fixed)}`
- Clip rows: `{len(clip_fixed)}`
- Mapped crop rows: `{mapped_crop_rows}`
- Mapped clip rows: `{mapped_clip_rows}`
- Unmapped crop rows: `{crop_unmapped}`
- Unmapped clip rows: `{clip_unmapped}`
- Split values: `{', '.join(nonempty_split_values)}`
- Has train/val/test: `{has_train_val_test}`
- Hard issue count: `{hard_issue_count}`
- Issue count: `{len(issues_df)}`
- Ready for v35: `{ready_for_v35}`

## Split readiness

{split_table}

## Class-by-split readiness

{class_table}

## Interpretation

The split-fixed indices should be used instead of the original v34 indices for any baseline experiment or feature-extraction stage that requires train/validation/test separation.
"""

OUT_REPORT.write_text(report)

OUT_NOTE.write_text(
    "# Week 7 v34b Split Fix\n\n"
    "## Summary\n\n"
    f"- Split source: `{best_path}`\n"
    f"- Crop rows: `{len(crop_fixed)}`\n"
    f"- Clip rows: `{len(clip_fixed)}`\n"
    f"- Mapped crop rows: `{mapped_crop_rows}`\n"
    f"- Mapped clip rows: `{mapped_clip_rows}`\n"
    f"- Unmapped crop rows: `{crop_unmapped}`\n"
    f"- Unmapped clip rows: `{clip_unmapped}`\n"
    f"- Split values: `{', '.join(nonempty_split_values)}`\n"
    f"- Has train/val/test: `{has_train_val_test}`\n"
    f"- Hard issue count: `{hard_issue_count}`\n"
    f"- Issue count: `{len(issues_df)}`\n"
    f"- Ready for v35: `{ready_for_v35}`\n\n"
    "## Outputs\n\n"
    f"- Crop fixed index: `{OUT_CROP}`\n"
    f"- Clip fixed index: `{OUT_CLIP}`\n"
    f"- Split map: `{OUT_SPLIT_MAP}`\n"
    f"- Split readiness: `{OUT_SPLIT_READINESS}`\n"
    f"- Class by split readiness: `{OUT_CLASS_SPLIT}`\n"
    f"- Decision: `{OUT_DECISION}`\n"
    f"- Issues: `{OUT_ISSUES}`\n"
)

print("Saved:")
print(OUT_CROP)
print(OUT_CLIP)
print(OUT_SPLIT_MAP)
print(OUT_SPLIT_READINESS)
print(OUT_CLASS_SPLIT)
print(OUT_CANDIDATES)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_REPORT)
print(OUT_NOTE)

print()
print("=== v34b decision ===")
print(decision.to_string(index=False))

print()
print("=== v34b issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
