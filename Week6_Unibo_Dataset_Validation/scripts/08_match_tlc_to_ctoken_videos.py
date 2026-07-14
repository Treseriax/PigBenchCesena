from pathlib import Path
import re
import csv
import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path.home() / "PigBench"
W6 = PROJECT_ROOT / "Week6_Unibo_Dataset_Validation"

OUT_STATS = W6 / "outputs/dataset_statistics"
OUT_VIS = W6 / "outputs/visual_label_check"
NOTES = W6 / "notes"

OUT_STATS.mkdir(parents=True, exist_ok=True)
OUT_VIS.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

VIDEO_NAMING = OUT_STATS / "unibo_raw_video_naming_pattern_analysis.csv"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


def clean_text(x):
    if x is None:
        return ""
    return str(x).strip()


def read_frame(path, frame_index):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None

    frame_index = max(0, min(int(frame_index), total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ok, frame = cap.read()
    cap.release()

    if not ok:
        return None

    return frame


def preprocess(frame):
    if frame is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (176, 144), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)


def compare_frames(a, b):
    pa = preprocess(a)
    pb = preprocess(b)

    if pa is None or pb is None:
        return {
            "mad": np.nan,
            "corr": np.nan,
            "similarity_score": np.nan,
        }

    mad = float(np.mean(np.abs(pa - pb)))

    aa = pa.flatten()
    bb = pb.flatten()

    if np.std(aa) == 0 or np.std(bb) == 0:
        corr = np.nan
    else:
        corr = float(np.corrcoef(aa, bb)[0, 1])

    # Higher is better. Corr dominates; low MAD helps.
    similarity = corr - (mad / 255.0)

    return {
        "mad": mad,
        "corr": corr,
        "similarity_score": similarity,
    }


def parse_ctoken(filename):
    stem = Path(filename).stem
    m = re.search(r"^c([0-9]{4})([0-9]{6})([0-9]{6})$", stem)
    if not m:
        return None

    cam_token = "c" + m.group(1)
    date_token = m.group(2)
    time_token = m.group(3)
    start_time = f"{time_token[:2]}:{time_token[2:4]}:{time_token[4:6]}"
    return cam_token, date_token, start_time


def resize_for_contact(frame, width=240):
    if frame is None:
        blank = np.full((180, width, 3), 255, dtype=np.uint8)
        return blank

    h, w = frame.shape[:2]
    scale = width / w
    nh = int(h * scale)
    return cv2.resize(frame, (width, nh), interpolation=cv2.INTER_AREA)


def add_label(img, label):
    img = img.copy()
    h, w = img.shape[:2]
    bar = np.full((35, w, 3), 255, dtype=np.uint8)
    cv2.putText(bar, label[:40], (5, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


if not VIDEO_NAMING.exists():
    raise FileNotFoundError(VIDEO_NAMING)

videos = pd.read_csv(VIDEO_NAMING)

tlc = videos[videos["naming_family"] == "TLC_B_time_range"].copy()
ctok = videos[videos["naming_family"] == "c_token_datetime_like"].copy()

# Normalize TLC hour to HH:MM:SS.
tlc["hour_start_hms"] = tlc["parsed_start_time"].astype(str) + ":00"

# c-token already has HH:MM:SS format in parsed_start_time.
ctok["hour_start_hms"] = ctok["parsed_start_time"].astype(str)

rows = []

sample_positions = [
    ("start", 100),
    ("middle", None),
    ("late", -100),
]

for _, t in tlc.iterrows():
    tlc_hour = clean_text(t["hour_start_hms"])
    tlc_path = Path(t["absolute_path"])
    tlc_total = int(float(t["frame_count"]))

    candidates = ctok[ctok["hour_start_hms"] == tlc_hour].copy()

    for _, c in candidates.iterrows():
        c_path = Path(c["absolute_path"])
        c_total = int(float(c["frame_count"]))

        frame_scores = []

        for pos_name, pos in sample_positions:
            if pos is None:
                t_idx = tlc_total // 2
                c_idx = c_total // 2
            elif pos < 0:
                t_idx = max(0, tlc_total + pos)
                c_idx = max(0, c_total + pos)
            else:
                t_idx = min(tlc_total - 1, pos)
                c_idx = min(c_total - 1, pos)

            tf = read_frame(tlc_path, t_idx)
            cf = read_frame(c_path, c_idx)
            scores = compare_frames(tf, cf)

            frame_scores.append({
                "position": pos_name,
                "tlc_frame_index": t_idx,
                "ctoken_frame_index": c_idx,
                **scores,
            })

        valid_scores = [s for s in frame_scores if not np.isnan(s["similarity_score"])]

        mean_corr = float(np.nanmean([s["corr"] for s in frame_scores]))
        mean_mad = float(np.nanmean([s["mad"] for s in frame_scores]))
        mean_similarity = float(np.nanmean([s["similarity_score"] for s in frame_scores]))

        parsed = parse_ctoken(c["filename"])
        c_cam = parsed[0] if parsed else clean_text(c.get("parsed_camera_id", ""))

        rows.append({
            "tlc_filename": t["filename"],
            "tlc_path": str(tlc_path),
            "tlc_hour_start": tlc_hour,
            "ctoken_filename": c["filename"],
            "ctoken_path": str(c_path),
            "ctoken_camera_token": c_cam,
            "ctoken_hour_start": tlc_hour,
            "mean_corr": mean_corr,
            "mean_mad": mean_mad,
            "mean_similarity_score": mean_similarity,
            "num_positions_compared": len(valid_scores),
        })

scores_df = pd.DataFrame(rows)

scores_path = OUT_STATS / "tlc_to_ctoken_video_similarity_scores.csv"
safe_to_csv(scores_df, scores_path)

if len(scores_df):
    camera_summary = (
        scores_df.groupby("ctoken_camera_token")
        .agg(
            comparisons=("mean_similarity_score", "count"),
            mean_corr=("mean_corr", "mean"),
            mean_mad=("mean_mad", "mean"),
            mean_similarity_score=("mean_similarity_score", "mean"),
            max_similarity_score=("mean_similarity_score", "max"),
        )
        .reset_index()
        .sort_values("mean_similarity_score", ascending=False)
    )
else:
    camera_summary = pd.DataFrame()

camera_summary_path = OUT_STATS / "tlc_to_ctoken_camera_similarity_summary.csv"
safe_to_csv(camera_summary, camera_summary_path)

# Choose best camera token.
best_camera = ""
if len(camera_summary):
    best_camera = str(camera_summary.iloc[0]["ctoken_camera_token"])

# Create contact sheets for top candidate cameras.
contact_dir = OUT_VIS / "tlc_ctoken_matching_contact_sheets"
contact_dir.mkdir(parents=True, exist_ok=True)

top_cameras = camera_summary["ctoken_camera_token"].head(3).tolist() if len(camera_summary) else []

for cam in top_cameras:
    cam_scores = scores_df[scores_df["ctoken_camera_token"] == cam].copy()
    cam_scores = cam_scores.sort_values("tlc_hour_start")

    rows_imgs = []

    for _, r in cam_scores.iterrows():
        tlc_path = Path(r["tlc_path"])
        c_path = Path(r["ctoken_path"])

        tlc_frame = read_frame(tlc_path, 1000)
        c_frame = read_frame(c_path, 1000)

        tlc_img = add_label(resize_for_contact(tlc_frame), f"TLC {r['tlc_hour_start']} {Path(r['tlc_path']).name}")
        c_img = add_label(resize_for_contact(c_frame), f"{cam} {r['ctoken_hour_start']} {Path(r['ctoken_path']).name}")

        # Match heights
        h = max(tlc_img.shape[0], c_img.shape[0])

        def pad_to_h(img, h):
            if img.shape[0] == h:
                return img
            pad = np.full((h - img.shape[0], img.shape[1], 3), 255, dtype=np.uint8)
            return np.vstack([img, pad])

        combined = np.hstack([pad_to_h(tlc_img, h), pad_to_h(c_img, h)])
        rows_imgs.append(combined)

    if rows_imgs:
        width = max(img.shape[1] for img in rows_imgs)
        padded = []
        for img in rows_imgs:
            if img.shape[1] < width:
                pad = np.full((img.shape[0], width - img.shape[1], 3), 255, dtype=np.uint8)
                img = np.hstack([img, pad])
            padded.append(img)

        sheet = np.vstack(padded)
        out_path = contact_dir / f"tlc_vs_{cam}_contact_sheet.jpg"
        cv2.imwrite(str(out_path), sheet)

# Build possible recovered mappings for 15:00-19:00.
recovered_rows = []

if best_camera:
    for hour in ["15:00:00", "16:00:00", "17:00:00", "18:00:00"]:
        cand = ctok[(ctok["parsed_camera_id"] == best_camera) & (ctok["hour_start_hms"] == hour)].copy()

        # parsed_camera_id should match c0001 style from earlier script.
        # If not, parse filename fallback.
        if cand.empty:
            mask_rows = []
            for _, c in ctok.iterrows():
                parsed = parse_ctoken(c["filename"])
                if parsed and parsed[0] == best_camera and parsed[2] == hour:
                    mask_rows.append(c)
            if mask_rows:
                cand = pd.DataFrame(mask_rows)

        if not cand.empty:
            c = cand.iloc[0]
            recovered_rows.append({
                "target_hour_start": hour,
                "recommended_ctoken_camera": best_camera,
                "recommended_video_path": c["absolute_path"],
                "recommended_video_filename": c["filename"],
                "status": "candidate_recovered_video",
            })
        else:
            recovered_rows.append({
                "target_hour_start": hour,
                "recommended_ctoken_camera": best_camera,
                "recommended_video_path": "",
                "recommended_video_filename": "",
                "status": "no_candidate_video_found_for_best_camera",
            })

recovered_df = pd.DataFrame(recovered_rows)
recovered_path = OUT_STATS / "tlc_unmatched_hours_recovered_ctoken_candidates.csv"
safe_to_csv(recovered_df, recovered_path)

note_path = NOTES / "tlc_to_ctoken_video_matching_notes.md"

with open(note_path, "w") as f:
    f.write("# TLC to c-token Video Matching Notes\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "The Excel annotations cover 07:00-19:00, but only TLC-named videos were directly matched for 07:00-15:00. "
        "This step compares overlapping TLC videos with c-token videos from the same hours to infer which c-token camera corresponds to TLC1/B1.\n\n"
    )

    f.write("## Camera similarity summary\n\n")
    if len(camera_summary):
        f.write(camera_summary.to_markdown(index=False))
    else:
        f.write("No overlapping TLC/c-token videos were found for comparison.")
    f.write("\n\n")

    f.write("## Recommended c-token camera\n\n")
    f.write(f"`{best_camera}`\n\n" if best_camera else "No recommendation available.\n\n")

    f.write("## Candidate recovered videos for 15:00-19:00\n\n")
    if len(recovered_df):
        f.write(recovered_df.to_markdown(index=False))
    else:
        f.write("No recovered candidates generated.")
    f.write("\n\n")

    f.write("## Contact sheets\n\n")
    for p in sorted(contact_dir.glob("*.jpg")):
        f.write(f"- `{p}`\n")
    f.write("\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The highest-similarity c-token camera can be used as a candidate mapping for unmatched afternoon labels, "
        "but the contact sheet should be visually inspected before treating this mapping as confirmed ground truth.\n"
    )

print("Saved:")
print(scores_path)
print(camera_summary_path)
print(recovered_path)
print(note_path)
print(contact_dir)

print()
print("=== Camera similarity summary ===")
print(camera_summary.to_string(index=False) if len(camera_summary) else "No comparison rows.")

print()
print("=== Recommended camera ===")
print(best_camera if best_camera else "None")

print()
print("=== Candidate recovered videos ===")
print(recovered_df.to_string(index=False) if len(recovered_df) else "No recovered candidates.")

print()
print("=== Contact sheets ===")
for p in sorted(contact_dir.glob("*.jpg")):
    print(p)
