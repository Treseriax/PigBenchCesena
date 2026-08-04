from pathlib import Path
from datetime import datetime
import re
import csv
import json
import hashlib
import zipfile
import math

import pandas as pd
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


ROOT = Path.home() / "PigBench"
FULL = ROOT / "Full_Unibo_Behaviour_Pipeline"
UNIBO = Path("/work/pig/datasets/Unibo")

OUT = FULL / "outputs" / "v78i_visual_stream_grouping_audit"
PKG = OUT / "Full_Unibo_Visual_Stream_Grouping_Audit"
STATIC = PKG / "visual_stream_groups"
NOTES = FULL / "notes"
REPORTS = FULL / "reports"
PROGRESS = FULL / "progress"

for p in [OUT, PKG, STATIC, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_INV = PKG / "v78i_video_inventory.csv"
OUT_OBS = PKG / "v78i_human_visual_observations.csv"
OUT_FRAME_FEATURES = PKG / "v78i_frame_features.csv"
OUT_PAIRWISE = PKG / "v78i_pairwise_visual_similarity.csv"
OUT_TLC1_ANCHOR = PKG / "v78i_tlc1_anchor_similarity_by_hour.csv"
OUT_GROUP_DECISION = PKG / "v78i_visual_group_decision_table.csv"
OUT_HTML = PKG / "v78i_visual_stream_grouping_board.html"
OUT_QA = PKG / "v78i_quality_checks.csv"
OUT_README = PKG / "README_v78i_Visual_Stream_Grouping_Audit.md"
OUT_MANIFEST = PKG / "v78i_manifest.json"

OUT_DECISION = OUT / "v78i_decision_summary.csv"
OUT_ISSUES = OUT / "v78i_issues.csv"
OUT_ZIP = OUT / "Full_Unibo_Visual_Stream_Grouping_Audit.zip"
OUT_SHA = OUT / "Full_Unibo_Visual_Stream_Grouping_Audit.sha256"
OUT_NOTE = NOTES / "v78i_visual_stream_grouping_audit_notes.md"
OUT_REPORT = REPORTS / "v78i_visual_stream_grouping_audit_report.md"
OUT_PROGRESS = PROGRESS / "full_unibo_pipeline_progress_log.csv"

ENC_RE = re.compile(r"^(c\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2}).*\.mp4$", re.IGNORECASE)
TLC_RE = re.compile(r"TLC\s*([1-6])", re.IGNORECASE)


def clean(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_encoded(path):
    m = ENC_RE.match(path.name)
    if not m:
        return None

    code, yy, mm, dd, HH, MM, SS = m.groups()

    return {
        "video_path": str(path),
        "video_filename": path.name,
        "video_type": "encoded",
        "camera_code": code.lower(),
        "date": f"20{yy}-{mm}-{dd}",
        "start_hhmm": f"{HH}:{MM}",
        "start_hhmmss": f"{HH}:{MM}:{SS}",
        "tlc_camera": "",
    }


def parse_friendly(path):
    name = path.name
    if "TLC" not in name.upper():
        return None

    tlc = TLC_RE.search(name)
    if not tlc:
        return None

    start = ""
    m = re.search(r"(\d{3,4})[-_](\d{3,4})", name)
    if m:
        raw = m.group(1).zfill(4)
        start = f"{raw[:2]}:{raw[2:]}"

    return {
        "video_path": str(path),
        "video_filename": path.name,
        "video_type": "friendly",
        "camera_code": "",
        "date": "",
        "start_hhmm": start,
        "start_hhmmss": "",
        "tlc_camera": f"TLC{tlc.group(1)}",
    }


def build_inventory():
    rows = []

    for p in sorted(UNIBO.glob("*.mp4")):
        r = parse_encoded(p)
        if r:
            rows.append(r)
            continue

        r = parse_friendly(p)
        if r:
            rows.append(r)

    return pd.DataFrame(rows)


def get_frame(path, sec=5):
    if cv2 is None:
        return None

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
    else:
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return None

    return frame


def resize_gray(frame, size=(160, 90)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)
    return gray


def dhash(gray):
    small = cv2.resize(gray, (17, 16))
    diff = small[:, 1:] > small[:, :-1]
    bits = diff.flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def hamming(a, b):
    return bin(int(a) ^ int(b)).count("1")


def hist_feature(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def edge_feature(gray):
    edges = cv2.Canny(gray, 80, 160)
    return edges.astype(np.float32).flatten() / 255.0


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0:
        return 0.0
    return float(np.dot(a, b) / den)


def corr2(a, b):
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def extract_features(video_df):
    rows = []
    feature_bank = {}

    for _, r in video_df.iterrows():
        path = Path(r["video_path"])
        frame = get_frame(path, sec=5)

        if frame is None:
            rows.append({
                **r.to_dict(),
                "feature_ok": False,
                "error": "could_not_read_frame",
            })
            continue

        gray = resize_gray(frame)
        hist = hist_feature(frame)
        edge = edge_feature(gray)
        dh = dhash(gray)

        key = r["video_filename"]
        feature_bank[key] = {
            "gray": gray,
            "hist": hist,
            "edge": edge,
            "dhash": dh,
            "row": r.to_dict(),
        }

        img_path = STATIC / f"{Path(key).stem}.jpg"
        if not img_path.exists():
            display = cv2.resize(frame, (520, int(frame.shape[0] * 520 / frame.shape[1])))
            cv2.rectangle(display, (0, 0), (520, 42), (0, 0, 0), -1)
            label = f"{r['video_type']} | {r.get('camera_code','')} {r.get('tlc_camera','')} | {r.get('start_hhmm','')} | {key}"
            cv2.putText(display, label[:80], (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(img_path), display)

        rows.append({
            **r.to_dict(),
            "feature_ok": True,
            "frame_image": str(img_path),
            "dhash": str(dh),
            "gray_mean": float(gray.mean()),
            "gray_std": float(gray.std()),
            "error": "",
        })

    return pd.DataFrame(rows), feature_bank


def pairwise_similarity(feature_bank):
    rows = []
    keys = sorted(feature_bank.keys())

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = feature_bank[keys[i]]
            b = feature_bank[keys[j]]

            ra = a["row"]
            rb = b["row"]

            # Focus mostly on same hour or friendly/encoded same hour.
            same_hour = clean(ra.get("start_hhmm")) and clean(ra.get("start_hhmm")) == clean(rb.get("start_hhmm"))

            if not same_hour:
                continue

            ham = hamming(a["dhash"], b["dhash"])
            dh_sim = 1.0 - ham / 256.0
            hist_sim = cosine(a["hist"], b["hist"])
            edge_sim = cosine(a["edge"], b["edge"])
            gray_corr = corr2(a["gray"], b["gray"])

            score = 0.35 * dh_sim + 0.25 * hist_sim + 0.25 * edge_sim + 0.15 * max(0, gray_corr)

            rows.append({
                "video_a": keys[i],
                "video_b": keys[j],
                "type_a": ra.get("video_type", ""),
                "type_b": rb.get("video_type", ""),
                "code_a": ra.get("camera_code", ""),
                "code_b": rb.get("camera_code", ""),
                "tlc_a": ra.get("tlc_camera", ""),
                "tlc_b": rb.get("tlc_camera", ""),
                "hour": ra.get("start_hhmm", ""),
                "dhash_similarity": round(dh_sim, 4),
                "hist_similarity": round(hist_sim, 4),
                "edge_similarity": round(edge_sim, 4),
                "gray_correlation": round(gray_corr, 4),
                "combined_similarity_score": round(score, 4),
            })

    return pd.DataFrame(rows).sort_values("combined_similarity_score", ascending=False) if rows else pd.DataFrame()


def tlc1_anchor_table(pairwise_df):
    if pairwise_df.empty:
        return pd.DataFrame()

    rows = []

    for _, r in pairwise_df.iterrows():
        a_friendly_tlc1 = r["type_a"] == "friendly" and r["tlc_a"] == "TLC1" and r["type_b"] == "encoded"
        b_friendly_tlc1 = r["type_b"] == "friendly" and r["tlc_b"] == "TLC1" and r["type_a"] == "encoded"

        if a_friendly_tlc1:
            rows.append({
                "hour": r["hour"],
                "friendly_video": r["video_a"],
                "encoded_video": r["video_b"],
                "encoded_code": r["code_b"],
                "combined_similarity_score": r["combined_similarity_score"],
                "dhash_similarity": r["dhash_similarity"],
                "hist_similarity": r["hist_similarity"],
                "edge_similarity": r["edge_similarity"],
                "gray_correlation": r["gray_correlation"],
            })

        if b_friendly_tlc1:
            rows.append({
                "hour": r["hour"],
                "friendly_video": r["video_b"],
                "encoded_video": r["video_a"],
                "encoded_code": r["code_a"],
                "combined_similarity_score": r["combined_similarity_score"],
                "dhash_similarity": r["dhash_similarity"],
                "hist_similarity": r["hist_similarity"],
                "edge_similarity": r["edge_similarity"],
                "gray_correlation": r["gray_correlation"],
            })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.sort_values(["hour", "combined_similarity_score"], ascending=[True, False])
    df["rank_within_hour"] = df.groupby("hour").cumcount() + 1
    return df


def write_human_observations():
    rows = [
        {
            "observation_id": "obs_001",
            "source": "manual_visual_review_by_user",
            "statement": "TLC1 anchor and c0002 are visually the same.",
            "camera_codes": "c0002",
            "relation_type": "confirmed_anchor",
            "confidence": "high",
            "mapping_implication": "TLC1/c0002 remains locked as visual anchor.",
        },
        {
            "observation_id": "obs_002",
            "source": "manual_visual_review_by_user",
            "statement": "c0000 appears visually the same as c0002.",
            "camera_codes": "c0000;c0002",
            "relation_type": "same_or_duplicate_view",
            "confidence": "high",
            "mapping_implication": "Do not treat c0000 and c0002 as necessarily different TLC cameras.",
        },
        {
            "observation_id": "obs_003",
            "source": "manual_visual_review_by_user",
            "statement": "c0001 and c0003 appear visually similar / same.",
            "camera_codes": "c0001;c0003",
            "relation_type": "same_or_duplicate_view",
            "confidence": "medium_high",
            "mapping_implication": "Do not treat c0001 and c0003 as necessarily different TLC cameras without further evidence.",
        },
        {
            "observation_id": "obs_004",
            "source": "manual_visual_review_by_user",
            "statement": "c0100 appears to be a top-view camera of the c0000/c0002 visual group.",
            "camera_codes": "c0100;c0000;c0002",
            "relation_type": "top_view_of_group",
            "confidence": "medium_high",
            "mapping_implication": "c0100 may be an overhead view of the same physical area, not a separate TLC identity.",
        },
        {
            "observation_id": "obs_005",
            "source": "manual_visual_review_by_user",
            "statement": "c0101 appears to be a top-view camera of the c0001/c0003 visual group.",
            "camera_codes": "c0101;c0001;c0003",
            "relation_type": "top_view_of_group",
            "confidence": "medium_high",
            "mapping_implication": "c0101 may be an overhead view of the same physical area, not a separate TLC identity.",
        },
        {
            "observation_id": "obs_006",
            "source": "manual_visual_review_by_user",
            "statement": "No visible label or camera name was found in the reviewed frames.",
            "camera_codes": "c0000;c0001;c0002;c0003;c0100;c0101",
            "relation_type": "no_visible_label",
            "confidence": "high",
            "mapping_implication": "Visual review alone cannot assign TLC2-TLC6 by textual labels.",
        },
    ]

    df = pd.DataFrame(rows)
    to_csv(df, OUT_OBS)
    return df


def make_group_decision(pairwise_df, anchor_df, obs_df):
    rows = []

    # Use manual observations as primary, pairwise as support.
    rows.append({
        "decision_item": "one_to_one_TLC_to_c_code_assumption",
        "decision": "invalid_or_unproven",
        "reason": "Manual review indicates several encoded c-codes are duplicate/same-area or top-view relations rather than six independent TLC cameras.",
        "evidence": "c0000≈c0002; c0001≈c0003; c0100 overhead of c0000/c0002; c0101 overhead of c0001/c0003.",
        "action": "Replace fixed TLC→c-code mapping with time-aware visual-stream mapping and explicit unresolved rows.",
    })

    if not anchor_df.empty:
        best = anchor_df.sort_values("combined_similarity_score", ascending=False).head(10)
        codes = ";".join(sorted(best["encoded_code"].dropna().unique()))
        rows.append({
            "decision_item": "TLC1_anchor",
            "decision": "confirmed_but_may_have_duplicate_streams",
            "reason": f"TLC1 anchor is visually confirmed by user; automatic similarity should be reviewed. Top matching encoded codes among top rows: {codes}",
            "evidence": "See v78i_tlc1_anchor_similarity_by_hour.csv",
            "action": "Keep TLC1=c0002 as locked human-confirmed anchor; investigate c0000/c0002 duplicate relation before using either globally.",
        })
    else:
        rows.append({
            "decision_item": "TLC1_anchor",
            "decision": "manual_confirmation_only",
            "reason": "No automatic anchor table generated, but user manually confirmed TLC1/c0002.",
            "evidence": "Manual visual review.",
            "action": "Keep TLC1=c0002 locked, but do not extrapolate to TLC2-TLC6.",
        })

    rows.append({
        "decision_item": "TLC2_to_TLC6_mapping",
        "decision": "not_solved_by_current_visual_labels",
        "reason": "No labels/names visible; encoded c-codes appear to represent visual stream groups rather than a simple six-camera TLC mapping.",
        "evidence": "Manual visual observations + lack of visible labels.",
        "action": "Build v78j time-aware visual-stream resolver; map only rows whose visual stream can be tied to room/pen/time with evidence.",
    })

    return pd.DataFrame(rows)


def build_html(video_df, feature_df, pairwise_df, anchor_df, obs_df, decision_df):
    html = []
    html.append("""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>v78i Visual Stream Grouping Audit</title>
<style>
body { font-family: Arial, sans-serif; margin:0; background:#f5f5f5; }
.header { background:#111; color:white; padding:18px 24px; position:sticky; top:0; z-index:10; }
.container { padding:24px; }
.section { background:white; margin-bottom:24px; padding:18px; border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,0.08); }
table { border-collapse:collapse; width:100%; font-size:13px; }
td, th { border:1px solid #ccc; padding:5px; vertical-align:top; }
th { background:#eee; }
.grid { display:flex; flex-wrap:wrap; gap:12px; }
.card { width:540px; background:#fafafa; border:1px solid #ccc; padding:8px; border-radius:8px; }
.card img { width:100%; border-radius:4px; }
.small { font-size:12px; color:#555; }
.warn { background:#fff3cd; border:1px solid #e3c875; padding:10px; border-radius:8px; }
</style>
</head>
<body>
<div class="header">
<h1>v78i Visual Stream Grouping Audit</h1>
<p>Purpose: verify whether encoded c-codes are six independent TLC cameras or duplicate/top-view stream groups.</p>
</div>
<div class="container">
""")

    def df_table(title, df, n=50):
        html.append(f"<div class='section'><h2>{title}</h2>")
        if df is None or df.empty:
            html.append("<p>empty</p></div>")
            return
        html.append(df.head(n).to_html(index=False, escape=True))
        html.append("</div>")

    html.append("<div class='section warn'><h2>Important interpretation</h2>")
    html.append("<p>Manual review indicates c0000≈c0002, c0001≈c0003, c0100 overhead of c0000/c0002, and c0101 overhead of c0001/c0003. Therefore a strict one-to-one TLC→c-code mapping is not safe.</p>")
    html.append("</div>")

    df_table("Human visual observations", obs_df, 20)
    df_table("Group decision table", decision_df, 20)
    df_table("TLC1 anchor similarity by hour", anchor_df, 80)
    df_table("Top pairwise similarities", pairwise_df, 100)

    html.append("<div class='section'><h2>Frame thumbnails by encoded code / friendly reference</h2><div class='grid'>")

    if not feature_df.empty:
        for _, r in feature_df.iterrows():
            img = clean(r.get("frame_image"))
            if not img:
                continue
            p = Path(img)
            rel = f"visual_stream_groups/{p.name}"
            title = f"{r.get('video_type','')} | {r.get('camera_code','')} {r.get('tlc_camera','')} | {r.get('start_hhmm','')}"
            html.append(f"<div class='card'><b>{title}</b><img src='{rel}'><div class='small'>{r.get('video_filename','')}</div></div>")

    html.append("</div></div>")
    html.append("</div></body></html>")

    OUT_HTML.write_text("\n".join(html), encoding="utf-8")


issues = []

if cv2 is None:
    issues.append({
        "item": "opencv",
        "issue_type": "hard_missing_cv2",
        "issue_detail": "cv2 is required for v78i visual stream grouping.",
        "severity": "hard",
    })

video_df = build_inventory()
to_csv(video_df, OUT_INV)

obs_df = write_human_observations()

if cv2 is not None and not video_df.empty:
    feature_df, feature_bank = extract_features(video_df)
    pairwise_df = pairwise_similarity(feature_bank)
    anchor_df = tlc1_anchor_table(pairwise_df)
else:
    feature_df = pd.DataFrame()
    pairwise_df = pd.DataFrame()
    anchor_df = pd.DataFrame()

to_csv(feature_df, OUT_FRAME_FEATURES)
to_csv(pairwise_df, OUT_PAIRWISE)
to_csv(anchor_df, OUT_TLC1_ANCHOR)

decision_df = make_group_decision(pairwise_df, anchor_df, obs_df)
to_csv(decision_df, OUT_GROUP_DECISION)

build_html(video_df, feature_df, pairwise_df, anchor_df, obs_df, decision_df)

qa_rows = []

def add_qa(name, expected, actual, passed, severity, detail):
    qa_rows.append({
        "check_name": name,
        "expected": str(expected),
        "actual": str(actual),
        "passed": bool(passed),
        "severity": severity,
        "detail": detail,
    })

encoded_codes = sorted(video_df[video_df["video_type"] == "encoded"]["camera_code"].dropna().unique()) if not video_df.empty else []
encoded_count = int((video_df["video_type"] == "encoded").sum()) if not video_df.empty else 0
friendly_count = int((video_df["video_type"] == "friendly").sum()) if not video_df.empty else 0

add_qa("video_inventory_nonempty", ">0", len(video_df), len(video_df) > 0, "hard", "Video inventory should be created.")
add_qa("encoded_camera_codes_found", ">=6", len(encoded_codes), len(encoded_codes) >= 6, "hard", "Expected encoded camera codes should exist.")
add_qa("friendly_reference_found", ">0", friendly_count, friendly_count > 0, "hard", "Friendly reference videos should exist.")
add_qa("frame_features_created", ">0", len(feature_df), len(feature_df) > 0, "hard", "Frame features should be extracted.")
add_qa("pairwise_similarity_created", ">0", len(pairwise_df), len(pairwise_df) > 0, "hard", "Pairwise similarities should be created.")
add_qa("human_observations_recorded", ">=6", len(obs_df), len(obs_df) >= 6, "hard", "Human visual observations should be recorded.")
add_qa("html_board_created", "exists", OUT_HTML.exists(), OUT_HTML.exists(), "hard", "HTML board should be created.")
add_qa("one_to_one_mapping_status", "invalid_or_unproven", "invalid_or_unproven", True, "info", "Do not force six TLC cameras onto six encoded c-codes.")

qa_df = pd.DataFrame(qa_rows)
to_csv(qa_df, OUT_QA)

hard_quality_failures = int(((qa_df["severity"] == "hard") & (~qa_df["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v78i_quality_checks",
        "issue_type": "hard_visual_stream_grouping_failed",
        "issue_detail": f"{hard_quality_failures} hard QA checks failed.",
        "severity": "hard",
    })

issues.append({
    "item": "mapping_model",
    "issue_type": "info_one_to_one_tlc_to_c_code_not_safe",
    "issue_detail": "Manual observations indicate duplicate/same-view and top-view relationships among encoded c-codes. Use visual-stream/time-aware mapping instead of fixed one-to-one TLC mapping.",
    "severity": "info",
})

issues.append({
    "item": "scope",
    "issue_type": "info_audit_only",
    "issue_detail": "v78i records visual observations and computes similarity; it does not apply final mapping or run tracking.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

manifest = {
    "version": "v78i_visual_stream_grouping_audit",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "video_rows": int(len(video_df)),
    "encoded_videos": encoded_count,
    "friendly_videos": friendly_count,
    "encoded_codes": encoded_codes,
    "frame_feature_rows": int(len(feature_df)),
    "pairwise_rows": int(len(pairwise_df)),
    "tlc1_anchor_rows": int(len(anchor_df)),
    "human_observation_rows": int(len(obs_df)),
    "html_board": str(OUT_HTML),
    "hard_issue_count": int(hard_issue_count),
    "claim_boundary": "visual stream grouping audit only; no mapping applied",
}
OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

readme = f"""# v78i Visual Stream Grouping Audit

## Purpose

This stage records manual visual observations and computes same-hour visual similarity between encoded c-code videos and friendly TLC reference videos.

## Key manual observations

- TLC1 anchor and c0002 are visually the same.
- c0000 appears visually the same as c0002.
- c0001 and c0003 appear visually similar / same.
- c0100 appears to be the top-view of the c0000/c0002 visual group.
- c0101 appears to be the top-view of the c0001/c0003 visual group.
- No visible label or camera name was found.

## Interpretation

A strict one-to-one TLC→c-code mapping is not safe. The encoded c-codes appear to form visual stream groups and overhead/side-view relationships.

## Outputs

- `v78i_human_visual_observations.csv`
- `v78i_pairwise_visual_similarity.csv`
- `v78i_tlc1_anchor_similarity_by_hour.csv`
- `v78i_visual_group_decision_table.csv`
- `v78i_visual_stream_grouping_board.html`

## Next

Build v78j time-aware visual-stream resolver instead of forcing fixed TLC→c-code mapping.
"""

OUT_README.write_text(readme, encoding="utf-8")
OUT_REPORT.write_text(readme, encoding="utf-8")

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

zip_hash = sha256_file(OUT_ZIP)
OUT_SHA.write_text(f"{zip_hash}  {OUT_ZIP.name}\n")

decision = pd.DataFrame([{
    "v78i_decision": "visual_stream_grouping_audit_completed" if hard_issue_count == 0 else "visual_stream_grouping_audit_has_blocking_issues",
    "video_rows": int(len(video_df)),
    "encoded_videos": encoded_count,
    "friendly_videos": friendly_count,
    "encoded_codes": ";".join(encoded_codes),
    "frame_feature_rows": int(len(feature_df)),
    "pairwise_similarity_rows": int(len(pairwise_df)),
    "tlc1_anchor_rows": int(len(anchor_df)),
    "human_observation_rows": int(len(obs_df)),
    "html_board_path": str(OUT_HTML),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_quality_failures": int(hard_quality_failures),
    "hard_issue_count": int(hard_issue_count),
    "warning_count": int(warning_count),
    "info_count": int(info_count),
    "issue_count": int(len(issues_df)),
    "one_to_one_tlc_to_c_code_safe": False,
    "ready_for_fixed_camera_code_mapping": False,
    "ready_for_v78j_time_aware_visual_stream_resolver": bool(hard_issue_count == 0),
    "ready_for_v79_full_tracking_preparation": False,
    "claim_scope": "visual_stream_grouping_audit_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])
to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# v78i Visual Stream Grouping Audit\n\n"
    f"- v78i decision: {decision.iloc[0]['v78i_decision']}\n"
    f"- Encoded videos: {encoded_count}\n"
    f"- Friendly videos: {friendly_count}\n"
    f"- Encoded codes: {', '.join(encoded_codes)}\n"
    f"- Pairwise similarity rows: {len(pairwise_df)}\n"
    f"- TLC1 anchor rows: {len(anchor_df)}\n"
    f"- Human observations recorded: {len(obs_df)}\n"
    f"- One-to-one TLC→c-code safe: False\n"
    f"- Ready for fixed camera-code mapping: False\n"
    f"- Ready for v78j time-aware visual-stream resolver: {bool(hard_issue_count == 0)}\n"
    f"- Ready for v79 full tracking preparation: False\n\n"
    "Manual review indicates encoded c-codes are duplicate/same-view/top-view visual stream groups. Do not force fixed one-to-one mapping.\n",
    encoding="utf-8"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v78i",
    "task_name": "Visual stream grouping audit",
    "status": "PASS_REMODEL_MAPPING_NEEDED" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": "Unibo videos + manual visual observations",
    "output_summary": str(PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Build v78j time-aware visual-stream resolver; do not force fixed TLC→c-code mapping.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

to_csv(progress, OUT_PROGRESS)

print("=== v78i decision ===")
print(decision.to_string(index=False))

print("\n=== human visual observations ===")
print(obs_df.to_string(index=False))

print("\n=== group decision table ===")
print(decision_df.to_string(index=False))

print("\n=== TLC1 anchor similarity by hour ===")
print(anchor_df.head(80).to_string(index=False) if len(anchor_df) else "none")

print("\n=== top pairwise similarities ===")
print(pairwise_df.head(80).to_string(index=False) if len(pairwise_df) else "none")

print("\n=== issues ===")
print(issues_df.to_string(index=False))
