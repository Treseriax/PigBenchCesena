from pathlib import Path
from datetime import datetime
import csv
import re
from collections import defaultdict, Counter

import pandas as pd


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

V19_ROOT = W7 / "outputs" / "final_fusion_qa_dataset_stats_v19"
TRAIN_READY = V19_ROOT / "week7_final_training_ready_dataset_v19_locked.csv"

OUT_ROOT = W7 / "outputs" / "split_strategy_design_v20"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DATASET_ROOT = OUT_ROOT / "split_datasets"
DATASET_ROOT.mkdir(parents=True, exist_ok=True)

OUT_FRAME_ASSIGN = OUT_ROOT / "week7_split_strategy_design_v20_frame_stratified_assignments.csv"
OUT_VIDEO_ASSIGN = OUT_ROOT / "week7_split_strategy_design_v20_video_aware_assignments.csv"
OUT_TEMPORAL_ASSIGN = OUT_ROOT / "week7_split_strategy_design_v20_temporal_assignments.csv"

OUT_CANDIDATE_SUMMARY = OUT_ROOT / "week7_split_strategy_design_v20_candidate_summary.csv"
OUT_CLASS_DISTRIBUTION = OUT_ROOT / "week7_split_strategy_design_v20_class_distribution_by_candidate.csv"
OUT_SPLIT_LEAKAGE = OUT_ROOT / "week7_split_strategy_design_v20_leakage_report.csv"
OUT_RECOMMENDATION = OUT_ROOT / "week7_split_strategy_design_v20_recommendation.md"
OUT_NOTE = W7 / "notes" / "week7_split_strategy_design_v20_notes.md"

RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}
SPLITS = ["train", "val", "test"]


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def scan_num(scan_frame_id):
    m = re.search(r"(\d+)$", str(scan_frame_id))
    return int(m.group(1)) if m else 10**9


def prepare_df(df):
    for c in ["scan_frame_id", "final_box_id", "behaviour_code", "behaviour_label", "visual_marker_colour_v18c", "behaviour_pig_id_v18c"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str).str.strip()

    video_col = first_existing(
        df.columns,
        ["video_id", "video_id_label", "video_id_identity", "video", "source_video"]
    )
    timestamp_col = first_existing(
        df.columns,
        ["timestamp", "timestamp_label", "timestamp_identity", "datetime"]
    )

    if video_col is None:
        df["video_key_v20"] = "unknown_video"
        video_col = "video_key_v20"
    else:
        df["video_key_v20"] = df[video_col].fillna("unknown_video").astype(str).str.strip()

    if timestamp_col is None:
        df["timestamp_key_v20"] = pd.NaT
    else:
        df["timestamp_key_v20"] = pd.to_datetime(df[timestamp_col], errors="coerce")

    return df, video_col, timestamp_col


def build_units(df, unit_col, unit_type):
    rows = []
    classes = sorted(df["behaviour_code"].dropna().astype(str).unique().tolist())

    for unit_id, g in df.groupby(unit_col, sort=False):
        counts = g["behaviour_code"].value_counts().to_dict()
        row = {
            "unit_id": str(unit_id),
            "unit_type": unit_type,
            "row_count": int(len(g)),
            "frame_count": int(g["scan_frame_id"].nunique()),
            "video_count": int(g["video_key_v20"].nunique()),
            "scan_frame_ids": " | ".join(sorted(g["scan_frame_id"].astype(str).unique(), key=scan_num)),
            "video_ids": " | ".join(sorted(g["video_key_v20"].astype(str).unique())),
            "min_scan_frame_num": int(min(scan_num(x) for x in g["scan_frame_id"].astype(str).unique())),
            "min_timestamp": g["timestamp_key_v20"].min(),
        }
        for cls in classes:
            row[f"class_{cls}"] = int(counts.get(cls, 0))
        rows.append(row)

    units = pd.DataFrame(rows)
    return units, classes


def unit_vector(row, classes):
    return {cls: int(row.get(f"class_{cls}", 0)) for cls in classes}


def rarity_score(row, classes, total_class_counts):
    score = 0.0
    for cls in classes:
        n = int(row.get(f"class_{cls}", 0))
        if n > 0:
            score += n * (1.0 / max(1, total_class_counts[cls]))
    score += 0.001 * int(row["row_count"])
    return score


def greedy_assign(units, classes, strategy_name):
    total_rows = int(units["row_count"].sum())
    total_class_counts = {
        cls: int(units[f"class_{cls}"].sum()) for cls in classes
    }

    target_rows = {s: total_rows * RATIOS[s] for s in SPLITS}
    target_class_counts = {
        s: {cls: total_class_counts[cls] * RATIOS[s] for cls in classes}
        for s in SPLITS
    }

    split_rows = {s: 0 for s in SPLITS}
    split_class_counts = {
        s: {cls: 0 for cls in classes}
        for s in SPLITS
    }

    ordered = units.copy()
    ordered["rarity_score"] = ordered.apply(lambda r: rarity_score(r, classes, total_class_counts), axis=1)
    ordered = ordered.sort_values(["rarity_score", "row_count"], ascending=[False, False]).reset_index(drop=True)

    assignments = []

    def global_score(test_rows, test_counts):
        # Row target error.
        row_err = sum(abs(test_rows[s] - target_rows[s]) / max(1, total_rows) for s in SPLITS)

        # Class distribution error, class-normalized.
        class_err = 0.0
        for s in SPLITS:
            for cls in classes:
                denom = max(1, total_class_counts[cls])
                class_err += abs(test_counts[s][cls] - target_class_counts[s][cls]) / denom

        # Penalize missing classes in val/test if class has at least 3 samples.
        missing_penalty = 0.0
        for s in ["val", "test"]:
            for cls in classes:
                if total_class_counts[cls] >= 3 and target_class_counts[s][cls] >= 0.45:
                    if test_counts[s][cls] == 0:
                        missing_penalty += 0.25

        # Penalize severe row overflow.
        overflow_penalty = 0.0
        for s in SPLITS:
            if test_rows[s] > target_rows[s] * 1.25:
                overflow_penalty += (test_rows[s] - target_rows[s] * 1.25) / max(1, total_rows)

        return row_err * 2.0 + class_err + missing_penalty + overflow_penalty

    for _, row in ordered.iterrows():
        vec = unit_vector(row, classes)
        best_split = None
        best_score = None

        for s in SPLITS:
            candidate_rows = dict(split_rows)
            candidate_counts = {sp: dict(split_class_counts[sp]) for sp in SPLITS}

            candidate_rows[s] += int(row["row_count"])
            for cls in classes:
                candidate_counts[s][cls] += vec[cls]

            score = global_score(candidate_rows, candidate_counts)

            if best_score is None or score < best_score:
                best_score = score
                best_split = s

        split_rows[best_split] += int(row["row_count"])
        for cls in classes:
            split_class_counts[best_split][cls] += vec[cls]

        assignments.append({
            "candidate": strategy_name,
            "unit_id": row["unit_id"],
            "unit_type": row["unit_type"],
            "split": best_split,
            "row_count": int(row["row_count"]),
            "frame_count": int(row["frame_count"]),
            "video_count": int(row["video_count"]),
            "scan_frame_ids": row["scan_frame_ids"],
            "video_ids": row["video_ids"],
        })

    return pd.DataFrame(assignments)


def temporal_assign(frame_units):
    units = frame_units.sort_values(["min_timestamp", "min_scan_frame_num"], na_position="last").reset_index(drop=True)
    total_rows = int(units["row_count"].sum())

    train_target = total_rows * RATIOS["train"]
    val_target = total_rows * (RATIOS["train"] + RATIOS["val"])

    rows = []
    running = 0

    for _, r in units.iterrows():
        if running < train_target:
            split = "train"
        elif running < val_target:
            split = "val"
        else:
            split = "test"

        rows.append({
            "candidate": "temporal",
            "unit_id": r["unit_id"],
            "unit_type": r["unit_type"],
            "split": split,
            "row_count": int(r["row_count"]),
            "frame_count": int(r["frame_count"]),
            "video_count": int(r["video_count"]),
            "scan_frame_ids": r["scan_frame_ids"],
            "video_ids": r["video_ids"],
        })

        running += int(r["row_count"])

    return pd.DataFrame(rows)


def expand_assignments_to_rows(df, assignments, candidate_name):
    # Assignment unit can be frame or video.
    unit_type = assignments["unit_type"].iloc[0]
    assign_map = dict(zip(assignments["unit_id"].astype(str), assignments["split"].astype(str)))

    out = df.copy()
    if unit_type == "frame":
        out["split_v20"] = out["scan_frame_id"].astype(str).map(assign_map)
    elif unit_type == "video":
        out["split_v20"] = out["video_key_v20"].astype(str).map(assign_map)
    else:
        raise ValueError(f"Unknown unit_type: {unit_type}")

    out["split_candidate_v20"] = candidate_name
    out["split_ratio_policy_v20"] = "train=0.70,val=0.15,test=0.15"
    return out


def distribution_for_candidate(split_df, candidate_name):
    rows = []
    total = len(split_df)
    classes = sorted(split_df["behaviour_code"].unique().tolist())

    for split in SPLITS:
        g = split_df[split_df["split_v20"] == split]
        split_total = len(g)
        vc = g["behaviour_code"].value_counts().to_dict()

        for cls in classes:
            rows.append({
                "candidate": candidate_name,
                "split": split,
                "behaviour_code": cls,
                "count": int(vc.get(cls, 0)),
                "split_total_rows": int(split_total),
                "percentage_within_split": round((vc.get(cls, 0) / split_total * 100), 2) if split_total else 0,
                "percentage_of_total_dataset": round((vc.get(cls, 0) / total * 100), 2) if total else 0,
            })

    return pd.DataFrame(rows)


def leakage_report(split_df, candidate_name):
    rows = []

    # Same frame across multiple splits.
    frame_leak = (
        split_df.groupby("scan_frame_id")["split_v20"]
        .nunique()
        .reset_index(name="split_count")
    )
    leaking_frames = frame_leak[frame_leak["split_count"] > 1]

    rows.append({
        "candidate": candidate_name,
        "leakage_type": "same_frame_across_splits",
        "leakage_count": int(len(leaking_frames)),
        "detail": " | ".join(leaking_frames["scan_frame_id"].astype(str).head(20).tolist()),
    })

    # Same video across multiple splits.
    video_leak = (
        split_df.groupby("video_key_v20")["split_v20"]
        .nunique()
        .reset_index(name="split_count")
    )
    leaking_videos = video_leak[video_leak["split_count"] > 1]

    rows.append({
        "candidate": candidate_name,
        "leakage_type": "same_video_across_splits",
        "leakage_count": int(len(leaking_videos)),
        "detail": " | ".join(leaking_videos["video_key_v20"].astype(str).head(20).tolist()),
    })

    return pd.DataFrame(rows)


def candidate_summary(split_df, candidate_name):
    rows = []
    total_rows = len(split_df)
    total_frames = split_df["scan_frame_id"].nunique()
    total_videos = split_df["video_key_v20"].nunique()
    all_classes = sorted(split_df["behaviour_code"].unique().tolist())

    class_missing_by_split = {}

    for split in SPLITS:
        g = split_df[split_df["split_v20"] == split]
        present = set(g["behaviour_code"].unique().tolist())
        missing = [c for c in all_classes if c not in present]
        class_missing_by_split[split] = missing

        rows.append({
            "candidate": candidate_name,
            "split": split,
            "rows": int(len(g)),
            "row_percentage": round(len(g) / total_rows * 100, 2) if total_rows else 0,
            "frames": int(g["scan_frame_id"].nunique()),
            "videos": int(g["video_key_v20"].nunique()),
            "behaviour_classes_present": int(len(present)),
            "behaviour_classes_missing": " | ".join(missing),
        })

    # Overall candidate score.
    missing_count = sum(len(v) for v in class_missing_by_split.values())
    split_counts = split_df["split_v20"].value_counts().to_dict()

    row_ratio_error = 0.0
    for split in SPLITS:
        actual = split_counts.get(split, 0) / max(1, total_rows)
        row_ratio_error += abs(actual - RATIOS[split])

    video_leak_count = (
        split_df.groupby("video_key_v20")["split_v20"].nunique().gt(1).sum()
    )

    frame_leak_count = (
        split_df.groupby("scan_frame_id")["split_v20"].nunique().gt(1).sum()
    )

    overall = pd.DataFrame([{
        "candidate": candidate_name,
        "split": "OVERALL",
        "rows": int(total_rows),
        "row_percentage": 100.0,
        "frames": int(total_frames),
        "videos": int(total_videos),
        "behaviour_classes_present": int(len(all_classes)),
        "behaviour_classes_missing": "",
        "total_missing_class_slots_across_splits": int(missing_count),
        "row_ratio_error_sum": round(row_ratio_error, 4),
        "same_frame_leakage_count": int(frame_leak_count),
        "same_video_leakage_count": int(video_leak_count),
        "candidate_score_lower_is_better": round(
            missing_count * 10.0 + row_ratio_error * 10.0 + frame_leak_count * 100.0 + video_leak_count * 0.5,
            4
        ),
    }])

    per_split = pd.DataFrame(rows)
    for c in overall.columns:
        if c not in per_split.columns:
            per_split[c] = ""
    for c in per_split.columns:
        if c not in overall.columns:
            overall[c] = ""

    return pd.concat([per_split[overall.columns], overall], ignore_index=True)


df = pd.read_csv(TRAIN_READY)
df, video_col, timestamp_col = prepare_df(df)

frame_units, classes = build_units(df, "scan_frame_id", "frame")
video_units, _ = build_units(df, "video_key_v20", "video")

frame_assign = greedy_assign(frame_units, classes, "frame_stratified")
video_assign = greedy_assign(video_units, classes, "video_aware")
temporal_assign_df = temporal_assign(frame_units)

safe_to_csv(frame_assign, OUT_FRAME_ASSIGN)
safe_to_csv(video_assign, OUT_VIDEO_ASSIGN)
safe_to_csv(temporal_assign_df, OUT_TEMPORAL_ASSIGN)

candidate_datasets = {}
candidate_assignments = {
    "frame_stratified": frame_assign,
    "video_aware": video_assign,
    "temporal": temporal_assign_df,
}

summary_parts = []
dist_parts = []
leakage_parts = []

for candidate_name, assignments in candidate_assignments.items():
    split_df = expand_assignments_to_rows(df, assignments, candidate_name)

    candidate_datasets[candidate_name] = split_df

    candidate_dir = DATASET_ROOT / candidate_name
    candidate_dir.mkdir(parents=True, exist_ok=True)

    safe_to_csv(split_df, candidate_dir / f"week7_v20_{candidate_name}_all_splits.csv")

    for split in SPLITS:
        safe_to_csv(
            split_df[split_df["split_v20"] == split].copy(),
            candidate_dir / f"week7_v20_{candidate_name}_{split}.csv"
        )

    summary_parts.append(candidate_summary(split_df, candidate_name))
    dist_parts.append(distribution_for_candidate(split_df, candidate_name))
    leakage_parts.append(leakage_report(split_df, candidate_name))

candidate_summary_df = pd.concat(summary_parts, ignore_index=True)
class_distribution_df = pd.concat(dist_parts, ignore_index=True)
leakage_df = pd.concat(leakage_parts, ignore_index=True)

safe_to_csv(candidate_summary_df, OUT_CANDIDATE_SUMMARY)
safe_to_csv(class_distribution_df, OUT_CLASS_DISTRIBUTION)
safe_to_csv(leakage_df, OUT_SPLIT_LEAKAGE)

# Recommendation logic.
overall = candidate_summary_df[candidate_summary_df["split"] == "OVERALL"].copy()
overall["candidate_score_lower_is_better"] = pd.to_numeric(overall["candidate_score_lower_is_better"], errors="coerce")
overall = overall.sort_values("candidate_score_lower_is_better", ascending=True)

recommended = overall.iloc[0]["candidate"] if len(overall) else "frame_stratified"

recommendation_text = (
    "# Week 7 Split Strategy Design v20 Recommendation\n\n"
    "## Inputs\n\n"
    f"- Training-ready dataset: `{TRAIN_READY}`\n"
    f"- Rows: `{len(df)}`\n"
    f"- Frames: `{df['scan_frame_id'].nunique()}`\n"
    f"- Videos: `{df['video_key_v20'].nunique()}`\n"
    f"- Behaviour classes: `{df['behaviour_code'].nunique()}`\n\n"
    "## Candidate strategies\n\n"
    "1. `frame_stratified`: keeps each scan frame intact and greedily balances behaviour classes.\n"
    "2. `video_aware`: keeps each source video intact for stronger generalization testing.\n"
    "3. `temporal`: preserves chronological order.\n\n"
    "## Recommendation\n\n"
    f"Recommended primary split candidate: `{recommended}`.\n\n"
    "Use this recommendation as the main modelling split only after inspecting the class-distribution table. "
    "Because the dataset is small and imbalanced, especially for `IA`, `BE`, and `DE`, the split should be reported with its class coverage limitations.\n\n"
    "## Practical interpretation\n\n"
    "- Prefer `frame_stratified` when the goal is to preserve rare behaviours in train/val/test.\n"
    "- Prefer `video_aware` when the goal is stricter video-level generalization.\n"
    "- Use `temporal` mainly as an additional robustness check, not necessarily as the main split.\n\n"
    "## Outputs\n\n"
    f"- Candidate summary: `{OUT_CANDIDATE_SUMMARY}`\n"
    f"- Class distribution: `{OUT_CLASS_DISTRIBUTION}`\n"
    f"- Leakage report: `{OUT_SPLIT_LEAKAGE}`\n"
    f"- Split datasets folder: `{DATASET_ROOT}`\n"
)

OUT_RECOMMENDATION.write_text(recommendation_text)

OUT_NOTE.write_text(
    "# Week 7 Split Strategy Design v20\n\n"
    "## Purpose\n\n"
    "This step creates and evaluates candidate train/validation/test split strategies from the v19 locked training-ready dataset.\n\n"
    "## Strategies\n\n"
    "- `frame_stratified`: frame-level units, class-aware greedy assignment.\n"
    "- `video_aware`: video-level units, class-aware greedy assignment.\n"
    "- `temporal`: chronological frame-level split.\n\n"
    "## Summary\n\n"
    f"- Input rows: `{len(df)}`\n"
    f"- Input frames: `{df['scan_frame_id'].nunique()}`\n"
    f"- Input videos: `{df['video_key_v20'].nunique()}`\n"
    f"- Behaviour classes: `{df['behaviour_code'].nunique()}`\n"
    f"- Recommended candidate: `{recommended}`\n\n"
    "## Outputs\n\n"
    f"- Frame-stratified assignments: `{OUT_FRAME_ASSIGN}`\n"
    f"- Video-aware assignments: `{OUT_VIDEO_ASSIGN}`\n"
    f"- Temporal assignments: `{OUT_TEMPORAL_ASSIGN}`\n"
    f"- Candidate summary: `{OUT_CANDIDATE_SUMMARY}`\n"
    f"- Class distribution by candidate: `{OUT_CLASS_DISTRIBUTION}`\n"
    f"- Leakage report: `{OUT_SPLIT_LEAKAGE}`\n"
    f"- Recommendation: `{OUT_RECOMMENDATION}`\n"
    f"- Split datasets: `{DATASET_ROOT}`\n"
)

print("Saved:")
print(OUT_FRAME_ASSIGN)
print(OUT_VIDEO_ASSIGN)
print(OUT_TEMPORAL_ASSIGN)
print(OUT_CANDIDATE_SUMMARY)
print(OUT_CLASS_DISTRIBUTION)
print(OUT_SPLIT_LEAKAGE)
print(OUT_RECOMMENDATION)
print(OUT_NOTE)
print(DATASET_ROOT)

print()
print("=== v20 candidate summary ===")
print(candidate_summary_df.to_string(index=False))

print()
print("=== v20 leakage report ===")
print(leakage_df.to_string(index=False))

print()
print("=== v20 recommendation ===")
print(f"Recommended primary split candidate: {recommended}")
