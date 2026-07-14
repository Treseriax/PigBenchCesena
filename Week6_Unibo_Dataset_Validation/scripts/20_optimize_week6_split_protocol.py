from pathlib import Path
import itertools
import csv
import pandas as pd


PROJECT_ROOT = Path.home() / "PigBench"
W6 = PROJECT_ROOT / "Week6_Unibo_Dataset_Validation"

OUT_GT = W6 / "outputs/unified_ground_truth"
OUT_STATS = W6 / "outputs/dataset_statistics"
NOTES = W6 / "notes"

GT_BASE = OUT_GT / "week6_unified_ground_truth_v2_with_recovered_videos.csv"
GT_CURRENT_SPLIT = OUT_GT / "week6_unified_ground_truth_v2_with_split.csv"

OUT_STATS.mkdir(parents=True, exist_ok=True)
OUT_GT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


def pct(n, d):
    return round(float(n) / float(d) * 100, 2) if d else 0.0


def split_summary(gt, split_col):
    return (
        gt.groupby(split_col)
        .agg(
            video_units=("split_unit_id", "nunique"),
            label_count=("record_id", "count"),
            unique_colours=("colour_id", "nunique"),
            unique_behaviours=("behaviour_code", "nunique"),
            direct_tlc_records=("video_match_status", lambda s: int((s == "matched_tlc_hour_video").sum())),
            candidate_ctoken_records=("video_match_status", lambda s: int((s == "candidate_recovered_ctoken_video").sum())),
        )
        .reset_index()
        .rename(columns={split_col: "split"})
    )


def behaviour_coverage(gt, split_col, all_behaviours):
    rows = []
    for split_name, g in gt.groupby(split_col):
        present = set(g["behaviour_code"].astype(str))
        missing = sorted(all_behaviours - present)
        rows.append({
            "split": split_name,
            "present_behaviour_count": len(present),
            "total_behaviour_count": len(all_behaviours),
            "missing_behaviour_count": len(missing),
            "missing_behaviours": ",".join(missing),
        })
    return pd.DataFrame(rows).sort_values("split")


if not GT_BASE.exists():
    raise FileNotFoundError(GT_BASE)

gt = pd.read_csv(GT_BASE)

# Unique split unit = one hourly video.
gt["split_unit_id"] = gt["video_id"].astype(str) + "__" + gt["hour_start"].astype(str)

unit_cols = [
    "split_unit_id",
    "video_id",
    "hour_start",
    "hour_end",
    "video_match_status",
    "video_mapping_confidence",
]

units = (
    gt.groupby(unit_cols)
    .agg(
        label_count=("record_id", "count"),
        unique_colours=("colour_id", "nunique"),
        unique_behaviours=("behaviour_code", "nunique"),
        behaviours=("behaviour_code", lambda s: ",".join(sorted(set(s.astype(str))))),
    )
    .reset_index()
    .sort_values("hour_start")
)

all_behaviours = set(gt["behaviour_code"].astype(str))

behaviour_dist = (
    gt.groupby(["behaviour_code", "behaviour_label"])
    .size()
    .reset_index(name="count")
    .sort_values("count")
)

behaviour_dist["percentage"] = behaviour_dist["count"].apply(lambda x: pct(x, len(gt)))

rare_behaviours = set(
    behaviour_dist[
        (behaviour_dist["count"] < 10) | (behaviour_dist["percentage"] < 3)
    ]["behaviour_code"].astype(str)
)

unit_behaviour_map = {
    r["split_unit_id"]: set(str(r["behaviours"]).split(","))
    for _, r in units.iterrows()
}

unit_status_map = {
    r["split_unit_id"]: r["video_match_status"]
    for _, r in units.iterrows()
}

all_unit_ids = list(units["split_unit_id"])

candidate_rows = []

# We keep the same scale as current split:
# train = 8 hourly videos, val = 2, test = 2.
# Hard preference: val and test each contain 1 direct TLC + 1 candidate c-token if possible.
for val_units in itertools.combinations(all_unit_ids, 2):
    val_set = set(val_units)

    remaining_after_val = [u for u in all_unit_ids if u not in val_set]

    for test_units in itertools.combinations(remaining_after_val, 2):
        test_set = set(test_units)
        train_set = set(all_unit_ids) - val_set - test_set

        def behaviours_for(unit_set):
            b = set()
            for u in unit_set:
                b |= unit_behaviour_map[u]
            return b

        train_b = behaviours_for(train_set)
        val_b = behaviours_for(val_set)
        test_b = behaviours_for(test_set)

        def status_counts(unit_set):
            statuses = [unit_status_map[u] for u in unit_set]
            return {
                "direct": sum(s == "matched_tlc_hour_video" for s in statuses),
                "candidate": sum(s == "candidate_recovered_ctoken_video" for s in statuses),
            }

        val_status = status_counts(val_set)
        test_status = status_counts(test_set)
        train_status = status_counts(train_set)

        # Hard constraint: val and test should each contain both direct and candidate video evidence.
        if not (val_status["direct"] >= 1 and val_status["candidate"] >= 1):
            continue
        if not (test_status["direct"] >= 1 and test_status["candidate"] >= 1):
            continue

        train_missing = all_behaviours - train_b
        val_missing = all_behaviours - val_b
        test_missing = all_behaviours - test_b

        val_rare_missing = rare_behaviours - val_b
        test_rare_missing = rare_behaviours - test_b
        train_rare_missing = rare_behaviours - train_b

        # Lower score is better.
        # Critical priority: train should cover all behaviours.
        # Next priority: val/test should miss as few behaviours and rare behaviours as possible.
        score = (
            1000 * len(train_missing)
            + 200 * (len(val_missing) + len(test_missing))
            + 150 * (len(val_rare_missing) + len(test_rare_missing))
            + 50 * len(train_rare_missing)
            + 10 * abs(len(val_b) - len(test_b))
        )

        candidate_rows.append({
            "score": score,
            "train_missing_count": len(train_missing),
            "val_missing_count": len(val_missing),
            "test_missing_count": len(test_missing),
            "train_missing_behaviours": ",".join(sorted(train_missing)),
            "val_missing_behaviours": ",".join(sorted(val_missing)),
            "test_missing_behaviours": ",".join(sorted(test_missing)),
            "train_rare_missing_count": len(train_rare_missing),
            "val_rare_missing_count": len(val_rare_missing),
            "test_rare_missing_count": len(test_rare_missing),
            "train_rare_missing": ",".join(sorted(train_rare_missing)),
            "val_rare_missing": ",".join(sorted(val_rare_missing)),
            "test_rare_missing": ",".join(sorted(test_rare_missing)),
            "train_behaviour_count": len(train_b),
            "val_behaviour_count": len(val_b),
            "test_behaviour_count": len(test_b),
            "train_units": "|".join(sorted(train_set)),
            "val_units": "|".join(sorted(val_set)),
            "test_units": "|".join(sorted(test_set)),
            "train_direct_units": train_status["direct"],
            "train_candidate_units": train_status["candidate"],
            "val_direct_units": val_status["direct"],
            "val_candidate_units": val_status["candidate"],
            "test_direct_units": test_status["direct"],
            "test_candidate_units": test_status["candidate"],
        })

candidates = pd.DataFrame(candidate_rows)

if len(candidates) == 0:
    raise RuntimeError("No split candidates found under current constraints.")

candidates = candidates.sort_values(
    [
        "score",
        "train_missing_count",
        "val_missing_count",
        "test_missing_count",
        "val_rare_missing_count",
        "test_rare_missing_count",
    ]
).reset_index(drop=True)

top_candidates_path = OUT_STATS / "week6_split_optimizer_top_candidates.csv"
safe_to_csv(candidates.head(50), top_candidates_path)

best = candidates.iloc[0]

best_train = set(str(best["train_units"]).split("|"))
best_val = set(str(best["val_units"]).split("|"))
best_test = set(str(best["test_units"]).split("|"))

gt_opt = gt.copy()
gt_opt["optimized_split"] = "unassigned"

gt_opt.loc[gt_opt["split_unit_id"].isin(best_train), "optimized_split"] = "train"
gt_opt.loc[gt_opt["split_unit_id"].isin(best_val), "optimized_split"] = "val"
gt_opt.loc[gt_opt["split_unit_id"].isin(best_test), "optimized_split"] = "test"

optimized_gt_path = OUT_GT / "week6_unified_ground_truth_v2_with_optimized_split_candidate.csv"
safe_to_csv(gt_opt, optimized_gt_path)

optimized_split_summary = split_summary(gt_opt, "optimized_split")
optimized_split_summary["label_percentage"] = optimized_split_summary["label_count"].apply(lambda x: pct(x, len(gt_opt)))

optimized_split_summary_path = OUT_STATS / "week6_optimized_split_candidate_summary.csv"
safe_to_csv(optimized_split_summary, optimized_split_summary_path)

optimized_coverage = behaviour_coverage(gt_opt, "optimized_split", all_behaviours)
optimized_coverage_path = OUT_STATS / "week6_optimized_split_candidate_behaviour_coverage.csv"
safe_to_csv(optimized_coverage, optimized_coverage_path)

optimized_video_level = (
    gt_opt.groupby(
        [
            "optimized_split",
            "split_unit_id",
            "video_id",
            "hour_start",
            "hour_end",
            "video_match_status",
            "video_mapping_confidence",
        ]
    )
    .agg(
        label_count=("record_id", "count"),
        unique_colours=("colour_id", "nunique"),
        unique_behaviours=("behaviour_code", "nunique"),
        behaviours=("behaviour_code", lambda s: ",".join(sorted(set(s.astype(str))))),
    )
    .reset_index()
    .rename(columns={"optimized_split": "split"})
    .sort_values(["split", "hour_start"])
)

optimized_video_level_path = OUT_STATS / "week6_optimized_split_candidate_video_level.csv"
safe_to_csv(optimized_video_level, optimized_video_level_path)

# Compare with current split if available.
comparison_rows = []

if GT_CURRENT_SPLIT.exists():
    current = pd.read_csv(GT_CURRENT_SPLIT)

    if "split" in current.columns:
        current["split_unit_id"] = current["video_id"].astype(str) + "__" + current["hour_start"].astype(str)

        current_summary = split_summary(current, "split")
        current_coverage = behaviour_coverage(current, "split", all_behaviours)

        current_summary_path = OUT_STATS / "week6_current_split_recomputed_summary.csv"
        current_coverage_path = OUT_STATS / "week6_current_split_recomputed_behaviour_coverage.csv"

        safe_to_csv(current_summary, current_summary_path)
        safe_to_csv(current_coverage, current_coverage_path)

        for split_name in ["train", "val", "test"]:
            cur_row = current_coverage[current_coverage["split"] == split_name]
            opt_row = optimized_coverage[optimized_coverage["split"] == split_name]

            comparison_rows.append({
                "split": split_name,
                "current_missing_count": int(cur_row["missing_behaviour_count"].iloc[0]) if len(cur_row) else "",
                "current_missing_behaviours": cur_row["missing_behaviours"].iloc[0] if len(cur_row) else "",
                "optimized_missing_count": int(opt_row["missing_behaviour_count"].iloc[0]) if len(opt_row) else "",
                "optimized_missing_behaviours": opt_row["missing_behaviours"].iloc[0] if len(opt_row) else "",
            })

comparison = pd.DataFrame(comparison_rows)
comparison_path = OUT_STATS / "week6_split_current_vs_optimized_comparison.csv"
safe_to_csv(comparison, comparison_path)

note_path = NOTES / "week6_split_optimizer_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Split Optimizer Notes\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This step searches alternative train/val/test splits at video-hour level. "
        "It does not overwrite the current split. It creates an optimized candidate split that minimizes missing behaviour classes in validation/test while preserving no-leakage video-hour grouping.\n\n"
    )

    f.write("## Constraints\n\n")
    f.write("- Train/val/test split unit is the hourly video, not individual frames.\n")
    f.write("- Train has 8 hourly videos, validation has 2, and test has 2.\n")
    f.write("- Validation and test must each include at least one direct TLC video and one candidate recovered c-token video.\n")
    f.write("- Lower score is better. The score prioritizes full train behaviour coverage, then fewer missing validation/test behaviours, then fewer missing rare behaviours.\n\n")

    f.write("## Best candidate score\n\n")
    f.write(pd.DataFrame([best]).to_markdown(index=False))
    f.write("\n\n")

    f.write("## Optimized split summary\n\n")
    f.write(optimized_split_summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Optimized behaviour coverage\n\n")
    f.write(optimized_coverage.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Current vs optimized comparison\n\n")
    f.write(comparison.to_markdown(index=False) if len(comparison) else "Current split comparison not available.")
    f.write("\n\n")

    f.write("## Optimized video-level split\n\n")
    f.write(optimized_video_level.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "If the optimized candidate reduces validation/test missing behaviours without breaking train coverage and without video leakage, "
        "it can replace the initial split in the next controlled step. "
        "If improvement is minimal, the current split can be kept and the missing rare classes should be documented as a dataset limitation.\n"
    )

print("Saved:")
print(top_candidates_path)
print(optimized_gt_path)
print(optimized_split_summary_path)
print(optimized_coverage_path)
print(optimized_video_level_path)
print(comparison_path)
print(note_path)

print()
print("=== Best split candidate ===")
print(pd.DataFrame([best]).to_string(index=False))

print()
print("=== Optimized split summary ===")
print(optimized_split_summary.to_string(index=False))

print()
print("=== Optimized behaviour coverage ===")
print(optimized_coverage.to_string(index=False))

print()
print("=== Current vs optimized comparison ===")
print(comparison.to_string(index=False) if len(comparison) else "No current comparison.")
