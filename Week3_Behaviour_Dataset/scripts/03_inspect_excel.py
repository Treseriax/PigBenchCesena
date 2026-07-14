import pandas as pd
from pathlib import Path

xlsx = Path("/work/pig/datasets/Unibo/Giorno 1 - 22_7_2021 tlc1 FASCIA 9-10.xlsx")

print("Excel file:", xlsx)
print("Exists:", xlsx.exists())

xls = pd.ExcelFile(xlsx)

print("\nSheets:")
for s in xls.sheet_names:
    print("-", s)

for s in xls.sheet_names:
    print("\n" + "=" * 80)
    print("Sheet:", s)
    df = pd.read_excel(xlsx, sheet_name=s, header=None)
    print("Shape:", df.shape)
    print("\nFirst 30 rows:")
    print(df.head(30).to_string())
