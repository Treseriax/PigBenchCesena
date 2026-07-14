from pathlib import Path
import csv
import pandas as pd


PROJECT_ROOT = Path.home() / "PigBench"
W6 = PROJECT_ROOT / "Week6_Unibo_Dataset_Validation"

OUT_GT = W6 / "outputs/unified_ground_truth"
OUT_STATS = W6 / "outputs/dataset_statistics"
NOTES = W6 / "notes"

OPT_GT = OUT_GT / "week6_unified_ground_truth_v2_with_optimized_split_candidate.csv"

OUT_GT.mkdir(parents=True, exist_ok=True)
OUT_STATS.mkdir(parents=True, exist_ok=True)
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


if not OPT_GT.exists():
    raise FileNotFoundError(OPT_GT)

gt = pd.read_csv(OPT_GT)

if "optimized_split" not in gt.columns:
    raise RuntimeError("optimized_split column not found.")

# Keep optimized_split history, but create final recommended split column.
gt["recommended_split_v2"] = gt["optimized_split"]
gt["split_policy_v2"] = "video_hour_level_no_leakage_optimized_behaviour_coverage"
gt["split_version"] = "recommended_v2"

# Keep a standard split column too, but in a separate v2 file only.
gt["split"] = gt["recommended_split_v2"]

recommended_gt_path = OUT_GT / "week6_unified_ground_truth_v2_with_recommended_split_v2.csv"
safe_to_csv(gt, recommended_gt_path)

# Split summary.
split_summary = (
    gt.groupby("recommended_split_v2")
    .agg(
        video_units=("split_unit_id", "nunique"),
        label_count=("record_id", "count"),
        unique_colours=("colour_id", "nunique"),
        unique_behaviours=("behaviour_code", "nunique"),
        direct_tlc_records=("video_match_status", lambda s: int((s == "matched_tlc_hour_video").sum())),
        candidate_ctoken_records=("video_match_status", lambda s: int((s == "candidate_recovered_ctoken_video").sum())),
    )
    .reset_index()
    .rename(columns={"recommended_split_v2": "split"})
)

split_summary["label_percentage"] = split_summary["label_count"].apply(lambda x: pct(x, len(gt)))

summary_path = OUT_STATS / "week6_recommended_split_v2_summary.csv"
safe_to_csv(split_summary, summary_path)

# Video-level table.
video_level = (
    gt.groupby(
        [
            "recommended_split_v2",
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
    .rename(columns={"recommended_split_v2": "split"})
    .sort_values(["split", "hour_start"])
)

video_level_path = OUT_STATS / "week6_recommended_split_v2_video_level.csv"
safe_to_csv(video_level, video_level_path)

# Behaviour coverage.
all_behaviours = set(gt["behaviour_code"].astype(str))
coverage_rows = []

for split_name, g in gt.groupby("recommended_split_v2"):
    present = set(g["behaviour_code"].astype(str))
    missing = sorted(all_behaviours - present)

    coverage_rows.append({
        "split": split_name,
        "present_behaviour_count": len(present),
        "total_behaviour_count": len(all_behaviours),
        "missing_behaviour_count": len(missing),
        "missing_behaviours": ",".join(missing),
    })

coverage = pd.DataFrame(coverage_rows).sort_values("split")
coverage_path = OUT_STATS / "week6_recommended_split_v2_behaviour_coverage.csv"
safe_to_csv(coverage, coverage_path)

# Leakage check.
leakage = (
    gt.groupby("split_unit_id")["recommended_split_v2"]
    .nunique()
    .reset_index(name="num_splits")
)
leakage = leakage[leakage["num_splits"] > 1].copy()

leakage_path = OUT_STATS / "week6_recommended_split_v2_leakage_check.csv"
safe_to_csv(leakage, leakage_path)

# Behaviour distribution by split.
behaviour_by_split = (
    gt.groupby(["recommended_split_v2", "behaviour_code", "behaviour_label"])
    .size()
    .reset_index(name="count")
    .rename(columns={"recommended_split_v2": "split"})
    .sort_values(["split", "count"], ascending=[True, False])
)

behaviour_by_split_path = OUT_STATS / "week6_recommended_split_v2_behaviour_distribution.csv"
safe_to_csv(behaviour_by_split, behaviour_by_split_path)

note_path = NOTES / "week6_recommended_split_v2_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Recommended Split v2 Notes\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This file adopts the optimized split candidate as the recommended Week 6 split v2. "
        "The original split files are preserved; this v2 split is saved as a separate recommended file.\n\n"
    )

    f.write("## Rationale\n\n")
    f.write(
        "The initial split had 3 missing behaviour classes in validation and 2 missing behaviour classes in test. "
        "The optimized split reduces this to 1 missing behaviour in validation and 1 missing behaviour in test, while preserving full train behaviour coverage and video-hour level no-leakage grouping.\n\n"
    )

    f.write("## Split summary\n\n")
    f.write(split_summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Behaviour coverage\n\n")
    f.write(coverage.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Video-level split\n\n")
    f.write(video_level.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Leakage check\n\n")
    if len(leakage):
        f.write(leakage.to_markdown(index=False))
    else:
        f.write("No video-hour split leakage detected.")
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "This split is the recommended split for reporting and future experiments. "
        "Because the dataset is small and contains rare behaviours, validation/test cannot cover all behaviour classes simultaneously under the current 8/2/2 video-hour split. "
        "The remaining missing classes should be documented as a dataset limitation.\n"
    )

print("Saved:")
print(recommended_gt_path)
print(summary_path)
print(video_level_path)
print(coverage_path)
print(leakage_path)
print(behaviour_by_split_path)
print(note_path)

print()
print("=== Recommended split v2 summary ===")
print(split_summary.to_string(index=False))

print()
print("=== Recommended split v2 coverage ===")
print(coverage.to_string(index=False))

print()
print("=== Leakage rows ===")
print(leakage.to_string(index=False) if len(leakage) else "No leakage.")
