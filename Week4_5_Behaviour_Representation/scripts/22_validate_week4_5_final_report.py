from pathlib import Path
import re
import pandas as pd


ROOT = Path("Week4_5_Behaviour_Representation")
FINAL = ROOT / "final_outputs"
OUT = ROOT / "outputs/final_report_inputs"

REPORT = FINAL / "Week4_5_Final_Report_Draft.md"
VALIDATION = FINAL / "Week4_5_Final_Report_Validation.md"

if not REPORT.exists():
    raise FileNotFoundError(REPORT)

text = REPORT.read_text(errors="ignore")

expected_sections = [
    "# 1. Objective and Scope",
    "# 2. Completed Internal Dataset Preparation",
    "# 3. Trajectory-Based Feature Engineering",
    "# 4. Trajectory Visualizations",
    "# 5. ROI / Resource-Based Feature Engineering",
    "# 6. ROI Visual Quality Control",
    "# 7. Public Dataset Exploration",
    "# 8. Edinburgh Dataset Sample and Annotation Inspection",
    "# 9. Edinburgh vs Unibo Annotation Format Comparison",
    "# 10. Behaviour Representation Survey",
    "# 11. Skeleton Framework Review",
    "# 12. Mesh-Based Representation Review",
    "# 13. Embedding-Based Representation Review",
    "# 14. Final Recommendations",
    "# 15. Deliverable Inventory",
    "# Conclusion",
]

typo_patterns = [
    "finalclassifier",
    "regionfeature",
    "whenlabelled",
    "supportimage",
    "thanstill-image",
    "visualbehaviour",
    "areless",
    "beneeded",
    "drinkerROI",
    "Requiresmanual",
    "potentiallymore",
    "andsubtle",
    "overlappingbehaviour",
    "trackingoutputs",
    "candidatezones",
    "labelsare",
]

section_rows = []
for sec in expected_sections:
    section_rows.append({
        "section": sec,
        "present": sec in text
    })

section_df = pd.DataFrame(section_rows)

typo_rows = []
for pattern in typo_patterns:
    typo_rows.append({
        "pattern": pattern,
        "count": text.count(pattern)
    })

typo_df = pd.DataFrame(typo_rows)

# Check key folders and deliverables
folders = [
    ROOT / "outputs/trajectory_features",
    ROOT / "outputs/visualizations/trajectory",
    ROOT / "outputs/roi_features",
    ROOT / "outputs/visualizations/roi",
    ROOT / "outputs/dataset_comparison",
    ROOT / "outputs/representation_tables",
    ROOT / "notes",
    ROOT / "notes/public_dataset_notes",
    ROOT / "final_outputs",
]

folder_rows = []
for folder in folders:
    folder_rows.append({
        "folder": str(folder.relative_to(ROOT)),
        "exists": folder.exists(),
        "file_count": len([p for p in folder.glob("*") if p.is_file()]) if folder.exists() else 0
    })

folder_df = pd.DataFrame(folder_rows)

# Count headings
h1_count = len(re.findall(r"^# ", text, flags=re.M))
h2_count = len(re.findall(r"^## ", text, flags=re.M))
table_count = text.count("|:")

summary_rows = [
    {"metric": "report_size_bytes", "value": REPORT.stat().st_size},
    {"metric": "line_count", "value": len(text.splitlines())},
    {"metric": "h1_heading_count", "value": h1_count},
    {"metric": "h2_heading_count", "value": h2_count},
    {"metric": "markdown_table_count_estimate", "value": table_count},
    {"metric": "missing_expected_sections", "value": int((~section_df["present"]).sum())},
    {"metric": "typo_pattern_total_count", "value": int(typo_df["count"].sum())},
]

summary_df = pd.DataFrame(summary_rows)

# Save CSVs
section_csv = OUT / "final_report_section_validation.csv"
typo_csv = OUT / "final_report_typo_validation.csv"
folder_csv = OUT / "final_report_folder_validation.csv"
summary_csv = OUT / "final_report_validation_summary.csv"

section_df.to_csv(section_csv, index=False)
typo_df.to_csv(typo_csv, index=False)
folder_df.to_csv(folder_csv, index=False)
summary_df.to_csv(summary_csv, index=False)

with open(VALIDATION, "w") as f:
    f.write("# Week 4-5 Final Report Validation\n\n")

    f.write("## Summary\n\n")
    f.write(summary_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Section validation\n\n")
    f.write(section_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Typo pattern validation\n\n")
    f.write(typo_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Folder/file validation\n\n")
    f.write(folder_df.to_markdown(index=False))
    f.write("\n\n")

    if int((~section_df["present"]).sum()) == 0 and int(typo_df["count"].sum()) == 0:
        f.write("## Decision\n\n")
        f.write("The final report draft passed the basic validation checks and is ready for packaging or PDF conversion.\n")
    else:
        f.write("## Decision\n\n")
        f.write("The final report draft needs minor cleanup before packaging or PDF conversion.\n")

print("Saved:")
print(VALIDATION)
print(section_csv)
print(typo_csv)
print(folder_csv)
print(summary_csv)

print()
print("=== Summary ===")
print(summary_df.to_string(index=False))

print()
print("=== Missing sections ===")
missing = section_df[~section_df["present"]]
print(missing.to_string(index=False) if len(missing) else "No missing sections.")

print()
print("=== Typo hits ===")
hits = typo_df[typo_df["count"] > 0]
print(hits.to_string(index=False) if len(hits) else "No typo pattern hits.")
