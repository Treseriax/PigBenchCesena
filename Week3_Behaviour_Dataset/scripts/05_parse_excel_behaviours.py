import json
from pathlib import Path

import pandas as pd


XLSX = Path("/work/pig/datasets/Unibo/Giorno 1 - 22_7_2021 tlc1 FASCIA 9-10.xlsx")
OUT_DIR = Path("Week3_Behaviour_Dataset/outputs/behaviour_annotations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHEET = "TLC 1 big"

# Behaviour code dictionary from ethogram + Excel codes.
# Some interaction codes will be refined later after we inspect all rows.
BEHAVIOUR_CODE_MAP = {
    "PI": "standing_inactive",
    "SI": "sitting_inactive",
    "LAI": "lying_lateral",
    "STI": "lying_sternal",
    "NU": "eating",
    "BE": "drinking",
    "DE": "defecating_or_urinating",
    "AN": "exploring_or_sniffing",
    "IN": "interaction",
    "IA": "agonistic_interaction",
    "MC": "mounting_or_other_contact",
    "ARR": "other_or_unclear",
    "BOX": "out_of_view_or_box",
}

def normalize_hour_range(value):
    if pd.isna(value):
        return None
    s = str(value).strip()
    s = s.replace(",", ":")
    if "-" not in s:
        return None
    start, end = s.split("-", 1)
    return start.strip(), end.strip()

def hhmm_to_hhmmss(hhmm, minute):
    hour = int(hhmm.split(":")[0])
    return f"{hour:02d}:{int(minute):02d}:00"

def interval_end_from_start(timestamp, duration_sec=10):
    h, m, s = map(int, timestamp.split(":"))
    total = h * 3600 + m * 60 + s + duration_sec
    eh = total // 3600
    em = (total % 3600) // 60
    es = total % 60
    return f"{eh:02d}:{em:02d}:{es:02d}"

def parse_colour(pig_label):
    s = str(pig_label).strip()
    if "-" in s:
        return s.split("-")[-1].strip()
    return s

df = pd.read_excel(XLSX, sheet_name=SHEET, header=None)

records = []

# Two blocks in the sheet:
# rows 2-10: 07:00-13:00 observations, animal rows 5-10
# rows 13-21: 13:00-19:00 observations, animal rows 16-21
blocks = [
    {"hour_row": 2, "minute_row": 3, "animal_rows": range(5, 11)},
    {"hour_row": 13, "minute_row": 14, "animal_rows": range(16, 22)},
]

for block in blocks:
    hour_row = block["hour_row"]
    minute_row = block["minute_row"]
    animal_rows = block["animal_rows"]

    current_hour = None

    for col in range(1, df.shape[1]):
        hour_value = df.iat[hour_row, col]
        parsed_hour = normalize_hour_range(hour_value)

        if parsed_hour is not None:
            current_hour = parsed_hour

        minute_value = df.iat[minute_row, col]

        if current_hour is None or pd.isna(minute_value):
            continue

        try:
            minute = int(minute_value)
        except Exception:
            continue

        hour_start, hour_end = current_hour
        timestamp = hhmm_to_hhmmss(hour_start, minute)
        interval_end = interval_end_from_start(timestamp, duration_sec=10)

        for r in animal_rows:
            pig_label = df.iat[r, 0]
            behaviour_code = df.iat[r, col]

            if pd.isna(pig_label) or pd.isna(behaviour_code):
                continue

            pig_label = str(pig_label).strip()
            behaviour_code = str(behaviour_code).strip()

            if behaviour_code == "":
                continue

            records.append({
                "source_file": str(XLSX),
                "sheet": SHEET,
                "hour_start": hour_start,
                "hour_end": hour_end,
                "minute": minute,
                "timestamp": timestamp,
                "interval_start": timestamp,
                "interval_end": interval_end,
                "pig_id": pig_label,
                "colour": parse_colour(pig_label),
                "behaviour_code": behaviour_code,
                "behaviour_label": BEHAVIOUR_CODE_MAP.get(behaviour_code, "unknown_code"),
            })

out_df = pd.DataFrame(records)

csv_path = OUT_DIR / "parsed_behaviour_observations.csv"
json_path = OUT_DIR / "parsed_behaviour_observations.json"

out_df.to_csv(csv_path, index=False)

with open(json_path, "w") as f:
    json.dump(records, f, indent=2)

print("Parsed records:", len(out_df))
print("Saved:", csv_path)
print("Saved:", json_path)

print("\nFirst 30 records:")
print(out_df.head(30).to_string(index=False))

print("\n09:00-10:00 records:")
print(out_df[out_df["hour_start"] == "09:00"].to_string(index=False))

print("\nBehaviour code counts:")
print(out_df["behaviour_code"].value_counts().to_string())
