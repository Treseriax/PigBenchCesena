from pathlib import Path
import csv
import pandas as pd


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"

OUT_FEAT = W6 / "outputs/feature_extractors"
NOTES = W6 / "notes"

IN_COMP = OUT_FEAT / "week6_feature_extractor_comparison.csv"
IN_REC = OUT_FEAT / "week6_feature_extractor_recommendations.csv"

OUT_COMP = OUT_FEAT / "week6_feature_extractor_comparison_cleaned_v2.csv"
OUT_REC = OUT_FEAT / "week6_feature_extractor_recommendations_cleaned_v2.csv"
OUT_NOTE = NOTES / "week6_feature_extractor_comparison_cleaned_v2_notes.md"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


def clean_text(x):
    if pd.isna(x):
        return x

    x = str(x)

    replacements = {
        "tocandidate": "to candidate",
        "posturerepresentation": "posture representation",
        "recommendedfile": "recommended file",
        "body-shape and posturerepresentation": "body-shape and posture representation",
        "Links high/medium marker evidence tocandidate colour IDs": "Links high/medium marker evidence to candidate colour IDs",
    }

    for old, new in replacements.items():
        x = x.replace(old, new)

    return x


if not IN_COMP.exists():
    raise FileNotFoundError(IN_COMP)
if not IN_REC.exists():
    raise FileNotFoundError(IN_REC)

comp = pd.read_csv(IN_COMP)
rec = pd.read_csv(IN_REC)

for col in comp.columns:
    if comp[col].dtype == "object":
        comp[col] = comp[col].apply(clean_text)

for col in rec.columns:
    if rec[col].dtype == "object":
        rec[col] = rec[col].apply(clean_text)

safe_to_csv(comp, OUT_COMP)
safe_to_csv(rec, OUT_REC)

with open(OUT_NOTE, "w") as f:
    f.write("# Week 6 Feature Extractor Comparison Cleaned v2\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This cleaned v2 version fixes minor wording/spacing issues in the feature extractor comparison table. "
        "It does not change any counts, paths, implementation status, or technical interpretation.\n\n"
    )

    f.write("## Feature extractor comparison\n\n")
    f.write(comp.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Recommendation ranking\n\n")
    f.write(rec.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The validated Week 6 feature pipeline remains: manual scan-sampling labels, scanpoint frames, "
        "YOLOv8-s bbox geometry with QC flags, and crop colour-marker candidate features. "
        "Segmentation and embedding features remain future extensions.\n"
    )

print("Saved:")
print(OUT_COMP)
print(OUT_REC)
print(OUT_NOTE)

print()
print("=== Cleaned feature extractor comparison v2 ===")
print(comp.to_string(index=False))

print()
print("=== Cleaned recommendations v2 ===")
print(rec.to_string(index=False))
