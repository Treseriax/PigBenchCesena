from pathlib import Path


ROOT = Path("Week4_5_Behaviour_Representation")
FINAL = ROOT / "final_outputs"

REPORT = FINAL / "Week4_5_Final_Report_Draft.md"
ADDENDUM = FINAL / "Week4_5_Task_Compliance_Addendum.md"
UPDATED = FINAL / "Week4_5_Final_Report_Draft_v2.md"

if not REPORT.exists():
    raise FileNotFoundError(REPORT)

if not ADDENDUM.exists():
    raise FileNotFoundError(ADDENDUM)

report_text = REPORT.read_text(errors="ignore")
addendum_text = ADDENDUM.read_text(errors="ignore")

# Avoid duplicate append.
marker = "# Week 4-5 Task Compliance Addendum"
if marker in report_text:
    updated_text = report_text
else:
    updated_text = report_text.rstrip() + "\n\n" + addendum_text

UPDATED.write_text(updated_text)

print("Saved:", UPDATED)
print("Size bytes:", UPDATED.stat().st_size)
print("Addendum included:", marker in updated_text)
