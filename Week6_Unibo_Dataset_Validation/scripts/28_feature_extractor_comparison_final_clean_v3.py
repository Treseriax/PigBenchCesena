from pathlib import Path
import csv
import pandas as pd
import re


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"

OUT_FEAT = W6 / "outputs/feature_extractors"
NOTES = W6 / "notes"

IN_COMP = OUT_FEAT / "week6_feature_extractor_comparison.csv"
IN_REC = OUT_FEAT / "week6_feature_extractor_recommendations.csv"

OUT_COMP = OUT_FEAT / "week6_feature_extractor_comparison_final_clean_v3.csv"
OUT_REC = OUT_FEAT / "week6_feature_extractor_recommendations_final_clean_v3.csv"
OUT_NOTE = NOTES / "week6_feature_extractor_comparison_final_clean_v3_notes.md"


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
        return ""

    x = str(x)

    replacements = {
        "tocandidate": "to candidate",
        "to candidate colour IDs": "to candidate colour IDs",
        "posturerepresentation": "posture representation",
        "recommendedfile": "recommended file",
        "nan": "",
    }

    for old, new in replacements.items():
        x = x.replace(old, new)

    # Extra defensive cleanup.
    x = re.sub(r"\bto\s*candidate\b", "to candidate", x)
    x = re.sub(r"\bposture\s*representation\b", "posture representation", x)
    x = re.sub(r"\brecommended\s*file\b", "recommended file", x)

    return x.strip()


if not IN_COMP.exists():
    raise FileNotFoundError(IN_COMP)
if not IN_REC.exists():
    raise FileNotFoundError(IN_REC)

comp = pd.read_csv(IN_COMP, keep_default_na=False)
rec = pd.read_csv(IN_REC, keep_default_na=False)

for col in comp.columns:
    comp[col] = comp[col].apply(clean_text)

for col in rec.columns:
    rec[col] = rec[col].apply(clean_text)

# Make not-yet rows cleaner.
for idx, row in comp.iterrows():
    if row.get("implemented", "") == "not yet":
        comp.at[idx, "main_output"] = ""
        comp.at[idx, "record_count"] = ""

# Explicit final corrections.
comp.loc[
    comp["extractor_family"] == "Candidate bbox-to-colour assignment",
    "strength"
] = "Links high/medium marker evidence to candidate colour IDs"

comp.loc[
    comp["extractor_family"] == "Segmentation features",
    "strength"
] = "Could improve body-shape and posture representation"

safe_to_csv(comp, OUT_COMP)
safe_to_csv(rec, OUT_REC)

with open(OUT_NOTE, "w") as f:
    f.write("# Week 6 Feature Extractor Comparison Final Clean v3\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This final cleaned v3 version fixes remaining wording and formatting issues in the feature extractor comparison table. "
        "It does not change any technical result, count, implementation status, or interpretation.\n\n"
    )

    f.write("## Feature extractor comparison\n\n")
    f.write(comp.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Recommendation ranking\n\n")
    f.write(rec.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The validated Week 6 feature pipeline is: manual scan-sampling labels linked to scanpoint frames, "
        "YOLOv8-s bbox geometry with QC flags, conservative bbox subsets for optional cleaner analysis, "
        "and crop colour-marker candidate features. "
        "Segmentation and embedding features remain future extensions.\n"
    )

# Quick text audit.
note_text = OUT_NOTE.read_text()
bad_terms = ["tocandidate", "posturerepresentation", "recommendedfile", "| nan", " nan "]

bad_found = [term for term in bad_terms if term in note_text]

print("Saved:")
print(OUT_COMP)
print(OUT_REC)
print(OUT_NOTE)

print()
print("=== Remaining bad terms ===")
print(bad_found if bad_found else "None")

print()
print("=== Final cleaned v3 comparison ===")
print(comp.to_string(index=False))
