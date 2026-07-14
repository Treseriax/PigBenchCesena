from pathlib import Path
from datetime import datetime
import csv
import shutil
import pandas as pd


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V20_ROOT = W7 / "outputs" / "split_strategy_design_v20"
V20_DATASETS = V20_ROOT / "split_datasets" / "frame_stratified"

SRC_ALL = V20_DATASETS / "week7_v20_frame_stratified_all_splits.csv"
SRC_TRAIN = V20_DATASETS / "week7_v20_frame_stratified_train.csv"
SRC_VAL = V20_DATASETS / "week7_v20_frame_stratified_val.csv"
SRC_TEST = V20_DATASETS / "week7_v20_frame_stratified_test.csv"

SRC_CANDIDATE_SUMMARY = V20_ROOT / "week7_split_strategy_design_v20_candidate_summary.csv"
SRC_CLASS_DIST = V20_ROOT / "week7_split_strategy_design_v20_class_distribution_by_candidate.csv"
SRC_LEAKAGE = V20_ROOT / "week7_split_strategy_design_v20_leakage_report.csv"
SRC_RECOMMENDATION = V20_ROOT / "week7_split_strategy_design_v20_recommendation.md"

OUT_ROOT = W7 / "outputs" / "primary_split_v21"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_ALL = OUT_ROOT / "week7_primary_split_v21_all.csv"
OUT_TRAIN = OUT_ROOT / "week7_primary_split_v21_train.csv"
OUT_VAL = OUT_ROOT / "week7_primary_split_v21_val.csv"
OUT_TEST = OUT_ROOT / "week7_primary_split_v21_test.csv"

OUT_SUMMARY = OUT_ROOT / "week7_primary_split_v21_summary.csv"
OUT_CLASS_DIST = OUT_ROOT / "week7_primary_split_v21_class_distribution.csv"
OUT_FRAME_DIST = OUT_ROOT / "week7_primary_split_v21_frame_distribution.csv"
OUT_VIDEO_LEAKAGE = OUT_ROOT / "week7_primary_split_v21_video_leakage_report.csv"
OUT_README = OUT_ROOT / "README_primary_split_v21.md"
OUT_NOTE = W7 / "notes" / "week7_primary_split_v21_notes.md"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def copy_file(src, dst):
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copy2(src, dst)


copy_file(SRC_ALL, OUT_ALL)
copy_file(SRC_TRAIN, OUT_TRAIN)
copy_file(SRC_VAL, OUT_VAL)
copy_file(SRC_TEST, OUT_TEST)

all_df = pd.read_csv(OUT_ALL)
train_df = pd.read_csv(OUT_TRAIN)
val_df = pd.read_csv(OUT_VAL)
test_df = pd.read_csv(OUT_TEST)

for df in [all_df, train_df, val_df, test_df]:
    for c in ["split_v20", "scan_frame_id", "video_key_v20", "behaviour_code", "behaviour_label"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()

# Class distribution for locked primary split.
dist_rows = []
for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    total = len(df)
    vc = df["behaviour_code"].value_counts().to_dict()

    for behaviour_code in sorted(all_df["behaviour_code"].unique()):
        label_values = all_df.loc[all_df["behaviour_code"] == behaviour_code, "behaviour_label"].dropna().unique()
        label = label_values[0] if len(label_values) else ""

        count = int(vc.get(behaviour_code, 0))
        dist_rows.append({
            "split": split_name,
            "behaviour_code": behaviour_code,
            "behaviour_label": label,
            "count": count,
            "split_total_rows": total,
            "percentage_within_split": round(count / total * 100, 2) if total else 0,
        })

class_dist = pd.DataFrame(dist_rows)
safe_to_csv(class_dist, OUT_CLASS_DIST)

# Frame distribution.
frame_rows = []
for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    frame_rows.append({
        "split": split_name,
        "rows": int(len(df)),
        "frames": int(df["scan_frame_id"].nunique()),
        "videos": int(df["video_key_v20"].nunique()),
        "behaviour_classes_present": int(df["behaviour_code"].nunique()),
        "missing_behaviour_classes": " | ".join(
            sorted(set(all_df["behaviour_code"].unique()) - set(df["behaviour_code"].unique()))
        ),
    })

frame_dist = pd.DataFrame(frame_rows)
safe_to_csv(frame_dist, OUT_FRAME_DIST)

# Video leakage report.
video_split_counts = (
    all_df.groupby("video_key_v20")["split_v20"]
    .nunique()
    .reset_index(name="split_count")
)

leaking_videos = video_split_counts[video_split_counts["split_count"] > 1].copy()

video_rows = []
for _, r in leaking_videos.iterrows():
    video_id = r["video_key_v20"]
    g = all_df[all_df["video_key_v20"] == video_id]
    video_rows.append({
        "video_key_v20": video_id,
        "split_count": int(r["split_count"]),
        "splits": " | ".join(sorted(g["split_v20"].unique())),
        "rows": int(len(g)),
        "frames": int(g["scan_frame_id"].nunique()),
        "note": "Expected for frame_stratified primary split; not suitable for strict video-level generalization.",
    })

video_leakage = pd.DataFrame(
    video_rows,
    columns=["video_key_v20", "split_count", "splits", "rows", "frames", "note"]
)
safe_to_csv(video_leakage, OUT_VIDEO_LEAKAGE)

summary = pd.DataFrame([{
    "primary_split_name": "frame_stratified",
    "selection_reason": "Best class coverage for small imbalanced dataset while keeping each scan frame intact.",
    "total_rows": int(len(all_df)),
    "train_rows": int(len(train_df)),
    "val_rows": int(len(val_df)),
    "test_rows": int(len(test_df)),
    "train_percentage": round(len(train_df) / len(all_df) * 100, 2),
    "val_percentage": round(len(val_df) / len(all_df) * 100, 2),
    "test_percentage": round(len(test_df) / len(all_df) * 100, 2),
    "total_frames": int(all_df["scan_frame_id"].nunique()),
    "train_frames": int(train_df["scan_frame_id"].nunique()),
    "val_frames": int(val_df["scan_frame_id"].nunique()),
    "test_frames": int(test_df["scan_frame_id"].nunique()),
    "total_videos": int(all_df["video_key_v20"].nunique()),
    "same_frame_leakage": int(
        all_df.groupby("scan_frame_id")["split_v20"].nunique().gt(1).sum()
    ),
    "same_video_leakage": int(
        all_df.groupby("video_key_v20")["split_v20"].nunique().gt(1).sum()
    ),
    "behaviour_classes_total": int(all_df["behaviour_code"].nunique()),
    "train_classes": int(train_df["behaviour_code"].nunique()),
    "val_classes": int(val_df["behaviour_code"].nunique()),
    "test_classes": int(test_df["behaviour_code"].nunique()),
    "val_missing_classes": " | ".join(
        sorted(set(all_df["behaviour_code"].unique()) - set(val_df["behaviour_code"].unique()))
    ),
    "test_missing_classes": " | ".join(
        sorted(set(all_df["behaviour_code"].unique()) - set(test_df["behaviour_code"].unique()))
    ),
    "locked_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(summary, OUT_SUMMARY)

README = f"""# Week 7 Primary Split v21

## Primary split

The selected primary split is `frame_stratified`.

## Why this split was selected

The dataset is small and strongly imbalanced. The rarest classes are difficult to distribute across train/validation/test, especially `IA`, `BE`, and `DE`.

`frame_stratified` was selected because it provides the best behaviour-class coverage while keeping each scan frame intact. It avoids same-frame leakage.

## Important limitation

This split is **not** a strict video-level generalization split. Several videos appear across multiple splits because the split is frame-level and class-balanced.

For strict video-level generalization, use the supplementary `video_aware` split from v20.

## Summary

- Total rows: `{int(summary.iloc[0]['total_rows'])}`
- Train rows: `{int(summary.iloc[0]['train_rows'])}` ({summary.iloc[0]['train_percentage']}%)
- Validation rows: `{int(summary.iloc[0]['val_rows'])}` ({summary.iloc[0]['val_percentage']}%)
- Test rows: `{int(summary.iloc[0]['test_rows'])}` ({summary.iloc[0]['test_percentage']}%)
- Same-frame leakage: `{int(summary.iloc[0]['same_frame_leakage'])}`
- Same-video leakage: `{int(summary.iloc[0]['same_video_leakage'])}`
- Validation missing classes: `{summary.iloc[0]['val_missing_classes']}`
- Test missing classes: `{summary.iloc[0]['test_missing_classes']}`

## Files

- `week7_primary_split_v21_all.csv`
- `week7_primary_split_v21_train.csv`
- `week7_primary_split_v21_val.csv`
- `week7_primary_split_v21_test.csv`
- `week7_primary_split_v21_summary.csv`
- `week7_primary_split_v21_class_distribution.csv`
- `week7_primary_split_v21_frame_distribution.csv`
- `week7_primary_split_v21_video_leakage_report.csv`
"""

OUT_README.write_text(README)

OUT_NOTE.write_text(
    "# Week 7 Primary Split v21\n\n"
    "## Purpose\n\n"
    "This step locks the selected primary split from v20. The selected split is `frame_stratified`.\n\n"
    "## Rationale\n\n"
    "The dataset is small and imbalanced, so the primary objective is preserving behaviour-class coverage while keeping each scan frame intact. "
    "`frame_stratified` has no same-frame leakage and provides the best class coverage among the tested candidates. "
    "It does have same-video leakage, so it should not be interpreted as a strict video-level generalization split.\n\n"
    "## Summary\n\n"
    f"- Total rows: `{int(summary.iloc[0]['total_rows'])}`\n"
    f"- Train rows: `{int(summary.iloc[0]['train_rows'])}`\n"
    f"- Validation rows: `{int(summary.iloc[0]['val_rows'])}`\n"
    f"- Test rows: `{int(summary.iloc[0]['test_rows'])}`\n"
    f"- Same-frame leakage: `{int(summary.iloc[0]['same_frame_leakage'])}`\n"
    f"- Same-video leakage: `{int(summary.iloc[0]['same_video_leakage'])}`\n"
    f"- Validation missing classes: `{summary.iloc[0]['val_missing_classes']}`\n"
    f"- Test missing classes: `{summary.iloc[0]['test_missing_classes']}`\n\n"
    "## Outputs\n\n"
    f"- All rows: `{OUT_ALL}`\n"
    f"- Train: `{OUT_TRAIN}`\n"
    f"- Validation: `{OUT_VAL}`\n"
    f"- Test: `{OUT_TEST}`\n"
    f"- Summary: `{OUT_SUMMARY}`\n"
    f"- Class distribution: `{OUT_CLASS_DIST}`\n"
    f"- Frame distribution: `{OUT_FRAME_DIST}`\n"
    f"- Video leakage report: `{OUT_VIDEO_LEAKAGE}`\n"
    f"- README: `{OUT_README}`\n"
)

print("Saved:")
print(OUT_ALL)
print(OUT_TRAIN)
print(OUT_VAL)
print(OUT_TEST)
print(OUT_SUMMARY)
print(OUT_CLASS_DIST)
print(OUT_FRAME_DIST)
print(OUT_VIDEO_LEAKAGE)
print(OUT_README)
print(OUT_NOTE)

print()
print("=== v21 primary split summary ===")
print(summary.to_string(index=False))

print()
print("=== v21 class distribution ===")
print(class_dist.to_string(index=False))
