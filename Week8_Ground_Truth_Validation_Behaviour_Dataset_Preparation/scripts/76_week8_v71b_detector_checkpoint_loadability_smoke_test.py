from pathlib import Path
from datetime import datetime
import os
import csv
import json
import hashlib
import zipfile
import traceback
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V71 = W8 / "outputs" / "v71_pretrained_embedding_feasibility_audit"
V71_PKG = V71 / "Week8_Pretrained_Embedding_Feasibility_Audit"

CANDIDATES = V71_PKG / "week8_v71_embedding_source_candidates.csv"
V71_DECISION = V71 / "week8_v71_decision_summary.csv"

OUT = W8 / "outputs" / "v71b_detector_checkpoint_loadability_smoke_test"
PKG = OUT / "Week8_Detector_Checkpoint_Loadability_Smoke_Test"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_TORCH = PKG / "week8_v71b_torch_load_smoke_results.csv"
OUT_CONFIGS = PKG / "week8_v71b_candidate_config_matches.csv"
OUT_MMDET = PKG / "week8_v71b_mmdet_init_smoke_results.csv"
OUT_RECOMMEND = PKG / "week8_v71b_v72_embedding_route_recommendation.csv"
OUT_QA = PKG / "week8_v71b_quality_checks.csv"
OUT_README = PKG / "README_Week8_Detector_Checkpoint_Loadability_Smoke_Test.md"
OUT_MANIFEST = PKG / "week8_v71b_manifest.json"

OUT_DECISION = OUT / "week8_v71b_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v71b_issues.csv"
OUT_ZIP = OUT / "Week8_Detector_Checkpoint_Loadability_Smoke_Test.zip"
OUT_SHA = OUT / "Week8_Detector_Checkpoint_Loadability_Smoke_Test.sha256"
OUT_NOTE = NOTES / "week8_v71b_detector_checkpoint_loadability_smoke_test_notes.md"
OUT_REPORT = REPORTS / "week8_v71b_detector_checkpoint_loadability_smoke_test_report.md"
OUT_PROGRESS = PROGRESS / "week8_experiment_progress_log.csv"


def safe_to_csv(df, path):
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL, escapechar="\\", lineterminator="\n")


def clean(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s


def read_csv(path):
    df = pd.read_csv(path).fillna("")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(clean)
    return df


def bool_true(x):
    return str(x).strip().lower() == "true"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def short_error(e):
    s = str(e).replace("\n", " ").replace("\r", " ")
    return s[:800]


def checkpoint_size_mb(path):
    try:
        return round(Path(path).stat().st_size / (1024 * 1024), 3)
    except Exception:
        return 0.0


def classify_top_level(obj):
    if isinstance(obj, dict):
        keys = list(obj.keys())
        return "dict", ";".join([str(k) for k in keys[:30]])
    return type(obj).__name__, ""


def search_configs():
    rows = []
    search_roots = [
        ROOT / "detection",
        ROOT,
    ]

    seen = set()

    for base in search_roots:
        if not base.exists():
            continue

        for p in base.rglob("*.py"):
            sp = str(p)
            if sp in seen:
                continue
            seen.add(sp)

            low = sp.lower()
            name = p.name.lower()

            if "yolov8" not in low and "yolo" not in low:
                continue

            try:
                text = p.read_text(errors="ignore")[:20000].lower()
            except Exception:
                text = ""

            rows.append({
                "config_path": sp,
                "filename": p.name,
                "contains_yolov8": "yolov8" in low or "yolov8" in text,
                "contains_mmyolo": "mmyolo" in text,
                "contains_pig": "pig" in low or "pig" in text,
                "contains_coco": "coco" in low or "coco" in text,
                "size_hint": "s" if ("_s" in name or "-s" in name or "small" in name) else (
                    "m" if ("_m" in name or "-m" in name or "medium" in name) else (
                        "l" if ("_l" in name or "-l" in name or "large" in name) else (
                            "x" if ("_x" in name or "-x" in name or "xlarge" in name) else ""
                        )
                    )
                ),
            })

    return pd.DataFrame(rows)


def infer_size_from_checkpoint_name(path):
    name = Path(path).name.lower()

    if "yolov8-iou-n" in name or "yolov8_n" in name or "yolov8n" in name or "-n" in name:
        return "n"
    if "yolov8-iou-s" in name or "yolov8_s" in name or "yolov8s" in name or "-s" in name:
        return "s"
    if "yolov8-iou-m" in name or "yolov8_m" in name or "yolov8m" in name or "-m" in name:
        return "m"
    if "yolov8-iou-l" in name or "yolov8_l" in name or "yolov8l" in name or "-l" in name:
        return "l"
    if "yolov8-iou-x" in name or "yolov8_x" in name or "yolov8x" in name or "-x" in name:
        return "x"

    return ""


def pick_config_for_checkpoint(configs, ckpt_path):
    if len(configs) == 0:
        return ""

    size = infer_size_from_checkpoint_name(ckpt_path)

    scored = configs.copy()
    scored["score"] = 0

    if size:
        scored.loc[scored["size_hint"] == size, "score"] += 50

    scored.loc[scored["contains_pig"] == True, "score"] += 25
    scored.loc[scored["contains_yolov8"] == True, "score"] += 15
    scored.loc[scored["contains_mmyolo"] == True, "score"] += 10
    scored.loc[scored["contains_coco"] == True, "score"] += 2

    scored = scored.sort_values(["score", "config_path"], ascending=[False, True])

    if len(scored) == 0 or int(scored.iloc[0]["score"]) <= 0:
        return ""

    return clean(scored.iloc[0]["config_path"])


issues = []

for p in [CANDIDATES, V71_DECISION]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required v71 output missing.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v71b_decision": "detector_checkpoint_loadability_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v72_frozen_embedding_baseline": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


v71_decision = read_csv(V71_DECISION)
candidates = read_csv(CANDIDATES)

if len(v71_decision) == 0 or not bool_true(v71_decision.iloc[0].get("ready_for_v72_frozen_embedding_baseline", "")):
    issues.append({
        "item": str(V71_DECISION),
        "issue_type": "hard_v71_not_ready",
        "issue_detail": "v71 must be ready before v71b.",
        "severity": "hard",
    })

cand = candidates[candidates["recommended_for_v72"].astype(str).str.lower() == "true"].copy()

if len(cand) == 0:
    cand = candidates.copy()

cand = cand[cand["candidate_path"].astype(str).str.strip() != ""].copy()
cand["size_mb"] = cand["candidate_path"].map(checkpoint_size_mb)

# Safer order: prefer local .pth official PigBench checkpoint, then smaller checkpoints, then larger .pt files.
cand["priority"] = 1000
cand.loc[cand["candidate_path"].str.contains("yolov8_s.pth", case=False, regex=False), "priority"] = 0
cand.loc[cand["candidate_path"].str.endswith(".pth"), "priority"] = cand.loc[cand["candidate_path"].str.endswith(".pth"), "priority"].clip(upper=10)
cand.loc[cand["candidate_path"].str.contains("YOLOv8-IoU-s.pt", case=False, regex=False), "priority"] = 20
cand.loc[cand["candidate_path"].str.contains("YOLOv8-IoU-n.pt", case=False, regex=False), "priority"] = 21
cand.loc[cand["candidate_path"].str.contains("YOLOv8-IoU-m.pt", case=False, regex=False), "priority"] = 30
cand.loc[cand["candidate_path"].str.contains("YOLOv8-IoU-l.pt", case=False, regex=False), "priority"] = 40
cand.loc[cand["candidate_path"].str.contains("YOLOv8-IoU-x.pt", case=False, regex=False), "priority"] = 50
cand = cand.sort_values(["priority", "size_mb"], ascending=[True, True]).reset_index(drop=True)

configs = search_configs()
safe_to_csv(configs, OUT_CONFIGS)

torch_rows = []

try:
    import torch
    torch_available = True
except Exception as e:
    torch = None
    torch_available = False
    issues.append({
        "item": "torch",
        "issue_type": "hard_torch_import_failed",
        "issue_detail": short_error(e),
        "severity": "hard",
    })

if torch_available:
    for _, r in cand.iterrows():
        path = Path(clean(r["candidate_path"]))
        row = {
            "candidate_path": str(path),
            "filename": path.name,
            "size_mb": checkpoint_size_mb(path),
            "candidate_type": clean(r.get("candidate_type", "")),
            "torch_load_attempted": True,
            "torch_load_ok": False,
            "top_level_type": "",
            "top_level_keys": "",
            "state_dict_like": False,
            "load_error": "",
        }

        if not path.exists():
            row["load_error"] = "file_missing"
            torch_rows.append(row)
            continue

        try:
            obj = torch.load(str(path), map_location="cpu")
            typ, keys = classify_top_level(obj)
            row["torch_load_ok"] = True
            row["top_level_type"] = typ
            row["top_level_keys"] = keys

            if isinstance(obj, dict):
                keys_low = [str(k).lower() for k in obj.keys()]
                row["state_dict_like"] = any(k in keys_low for k in ["state_dict", "model", "ema", "optimizer", "meta"])
            else:
                row["state_dict_like"] = False

            del obj

        except Exception as e:
            row["load_error"] = short_error(e)

        torch_rows.append(row)

torch_df = pd.DataFrame(torch_rows)
safe_to_csv(torch_df, OUT_TORCH)

mmdet_rows = []

mmdet_available = False
try:
    from mmdet.apis import init_detector
    mmdet_available = True
except Exception as e:
    init_detector = None
    mmdet_available = False
    issues.append({
        "item": "mmdet_init_detector",
        "issue_type": "warning_mmdet_init_detector_unavailable",
        "issue_detail": short_error(e),
        "severity": "warning",
    })

if mmdet_available:
    # Only try the safest torch-loadable .pth first; avoid wasting time on all checkpoints.
    try_candidates = torch_df[
        (torch_df["torch_load_ok"] == True)
        & (torch_df["candidate_path"].str.endswith(".pth"))
    ].copy()

    if len(try_candidates) == 0:
        try_candidates = torch_df[torch_df["torch_load_ok"] == True].copy().head(1)

    try_candidates = try_candidates.head(2)

    for _, r in try_candidates.iterrows():
        ckpt = clean(r["candidate_path"])
        cfg = pick_config_for_checkpoint(configs, ckpt)

        row = {
            "candidate_path": ckpt,
            "config_path": cfg,
            "mmdet_init_attempted": bool(cfg),
            "mmdet_init_ok": False,
            "model_class": "",
            "has_backbone": "",
            "has_extract_feat": "",
            "init_error": "",
        }

        if not cfg:
            row["init_error"] = "no_matching_config_found"
            mmdet_rows.append(row)
            continue

        try:
            model = init_detector(cfg, ckpt, device="cpu")
            row["mmdet_init_ok"] = True
            row["model_class"] = type(model).__name__
            row["has_backbone"] = hasattr(model, "backbone")
            row["has_extract_feat"] = hasattr(model, "extract_feat")
            del model

        except Exception as e:
            row["init_error"] = short_error(e)

        mmdet_rows.append(row)

mmdet_df = pd.DataFrame(mmdet_rows)
safe_to_csv(mmdet_df, OUT_MMDET)

torch_ok_count = int(torch_df["torch_load_ok"].sum()) if len(torch_df) else 0
pth_torch_ok_count = int(((torch_df["torch_load_ok"] == True) & (torch_df["candidate_path"].str.endswith(".pth"))).sum()) if len(torch_df) else 0
mmdet_ok_count = int(mmdet_df["mmdet_init_ok"].sum()) if len(mmdet_df) else 0
config_count = int(len(configs))

recommend_rows = []

if mmdet_ok_count > 0:
    best = mmdet_df[mmdet_df["mmdet_init_ok"] == True].iloc[0]
    recommend_rows.append({
        "route": "mmdet_extract_feat_detector_embedding",
        "recommended": True,
        "checkpoint_path": clean(best["candidate_path"]),
        "config_path": clean(best["config_path"]),
        "confidence": "high",
        "reason": "mmdet init_detector succeeded and model exposes detector object.",
        "v72_plan": "Use init_detector on CPU/GPU, feed crops or anchor frames, extract backbone/neck features where possible.",
    })
elif pth_torch_ok_count > 0:
    best = torch_df[(torch_df["torch_load_ok"] == True) & (torch_df["candidate_path"].str.endswith(".pth"))].iloc[0]
    recommend_rows.append({
        "route": "checkpoint_state_dict_available_but_architecture_missing",
        "recommended": False,
        "checkpoint_path": clean(best["candidate_path"]),
        "config_path": "",
        "confidence": "medium",
        "reason": "torch.load succeeded for .pth, but mmdet architecture init did not succeed.",
        "v72_plan": "Do not force embedding extraction yet; inspect config/model architecture first.",
    })
elif torch_ok_count > 0:
    best = torch_df[torch_df["torch_load_ok"] == True].iloc[0]
    recommend_rows.append({
        "route": "generic_torch_checkpoint_available",
        "recommended": False,
        "checkpoint_path": clean(best["candidate_path"]),
        "config_path": "",
        "confidence": "low",
        "reason": "Some checkpoint loads with torch, but no reliable model architecture route.",
        "v72_plan": "Avoid claiming frozen embedding baseline until architecture route is resolved.",
    })
else:
    recommend_rows.append({
        "route": "no_loadable_checkpoint",
        "recommended": False,
        "checkpoint_path": "",
        "config_path": "",
        "confidence": "none",
        "reason": "No candidate checkpoint loaded successfully.",
        "v72_plan": "Do not run detector embedding baseline without fixing checkpoint/framework availability.",
    })

recommend = pd.DataFrame(recommend_rows)
safe_to_csv(recommend, OUT_RECOMMEND)

ready_v72 = bool(len(recommend[recommend["recommended"] == True]) > 0)

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

add_qa("candidate_rows", ">0", len(cand), len(cand) > 0, "hard", "At least one v71 candidate should exist.")
add_qa("torch_available", True, torch_available, torch_available, "hard", "Torch is required for smoke test.")
add_qa("torch_load_ok_count", ">0", torch_ok_count, torch_ok_count > 0, "hard", "At least one checkpoint should load with torch.load.")
add_qa("config_candidates_found", ">0", config_count, config_count > 0, "warning", "YOLO config files are needed for mmdet init.")
add_qa("mmdet_init_ok_count", ">0", mmdet_ok_count, mmdet_ok_count > 0, "warning", "mmdet init success is preferred for v72 embedding.")
add_qa("ready_v72_detector_embedding", True, ready_v72, ready_v72, "warning", "v72 detector embedding baseline should only proceed if route is recommended.")

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_QA)

hard_quality_failures = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())
warning_quality_failures = int(((qa["severity"] == "warning") & (~qa["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v71b_quality_checks",
        "issue_type": "hard_checkpoint_loadability_failed",
        "issue_detail": f"{hard_quality_failures} hard checks failed.",
        "severity": "hard",
    })

if warning_quality_failures:
    issues.append({
        "item": "v71b_embedding_route",
        "issue_type": "warning_detector_embedding_route_not_fully_ready",
        "issue_detail": f"{warning_quality_failures} warning checks failed.",
        "severity": "warning",
    })

issues.append({
    "item": "claim_scope",
    "issue_type": "info_smoke_test_only",
    "issue_detail": "v71b only tests loadability. It does not train or evaluate a model.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

manifest = {
    "version": "week8_v71b_detector_checkpoint_loadability_smoke_test",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "candidate_count": int(len(cand)),
    "torch_load_ok_count": torch_ok_count,
    "pth_torch_load_ok_count": pth_torch_ok_count,
    "config_count": config_count,
    "mmdet_init_ok_count": mmdet_ok_count,
    "recommended_route": recommend.to_dict(orient="records"),
    "ready_for_v72": bool(ready_v72 and hard_issue_count == 0),
    "claim_boundary": "loadability smoke test only; no classifier trained",
}
OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

OUT_README.write_text(
    "# Week8 v71b Detector Checkpoint Loadability Smoke Test\n\n"
    f"- Candidate checkpoints tested: {len(cand)}\n"
    f"- torch.load successes: {torch_ok_count}\n"
    f"- .pth torch.load successes: {pth_torch_ok_count}\n"
    f"- YOLO config candidates found: {config_count}\n"
    f"- mmdet init successes: {mmdet_ok_count}\n"
    f"- Ready for v72 detector embedding: {bool(ready_v72 and hard_issue_count == 0)}\n\n"
    "This stage does not train or evaluate a model.\n"
)

if OUT_ZIP.exists():
    OUT_ZIP.unlink()

with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(PKG.rglob("*")):
        if p.is_file():
            z.write(p, p.relative_to(OUT))

zip_hash = sha256_file(OUT_ZIP)
OUT_SHA.write_text(f"{zip_hash}  {OUT_ZIP.name}\n")

decision = pd.DataFrame([{
    "v71b_decision": "detector_checkpoint_loadability_smoke_test_completed" if hard_issue_count == 0 else "detector_checkpoint_loadability_smoke_test_has_blocking_issues",
    "candidate_count": int(len(cand)),
    "torch_load_ok_count": torch_ok_count,
    "pth_torch_load_ok_count": pth_torch_ok_count,
    "config_candidate_count": config_count,
    "mmdet_init_ok_count": mmdet_ok_count,
    "recommended_route": clean(recommend.iloc[0]["route"]),
    "recommended_checkpoint_path": clean(recommend.iloc[0]["checkpoint_path"]),
    "recommended_config_path": clean(recommend.iloc[0]["config_path"]),
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_quality_failures": hard_quality_failures,
    "warning_quality_failures": warning_quality_failures,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v72_frozen_embedding_baseline": bool(ready_v72 and hard_issue_count == 0),
    "claim_scope": "checkpoint_loadability_smoke_test_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v71b Detector Checkpoint Loadability Smoke Test\n\n"
    f"- v71b decision: {decision.iloc[0]['v71b_decision']}\n"
    f"- Candidate checkpoints: {len(cand)}\n"
    f"- torch.load successes: {torch_ok_count}\n"
    f"- .pth torch.load successes: {pth_torch_ok_count}\n"
    f"- Config candidates found: {config_count}\n"
    f"- mmdet init successes: {mmdet_ok_count}\n"
    f"- Recommended route: {clean(recommend.iloc[0]['route'])}\n"
    f"- Recommended checkpoint: {clean(recommend.iloc[0]['checkpoint_path'])}\n"
    f"- Recommended config: {clean(recommend.iloc[0]['config_path'])}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Warnings: {warning_count}\n"
    f"- Ready for v72 frozen embedding baseline: {bool(ready_v72 and hard_issue_count == 0)}\n\n"
    "v71b is only a loadability smoke test. It does not train or evaluate a model.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v71b Detector Checkpoint Loadability Smoke Test Report\n\n"
    f"Decision: {decision.iloc[0]['v71b_decision']}\n\n"
    f"Recommended route: {clean(recommend.iloc[0]['route'])}\n\n"
    f"Recommended checkpoint: {clean(recommend.iloc[0]['checkpoint_path'])}\n\n"
    f"Recommended config: {clean(recommend.iloc[0]['config_path'])}\n\n"
    f"ZIP: {OUT_ZIP}\n\n"
    f"SHA256: {zip_hash}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v71b",
    "task_name": "Detector checkpoint loadability smoke test",
    "status": "PASS_WITH_WARNINGS" if hard_issue_count == 0 and warning_count > 0 else ("PASS" if hard_issue_count == 0 else "NEEDS_FIX"),
    "input_summary": str(CANDIDATES),
    "output_summary": str(PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Run v72 frozen embedding baseline if recommended route is ready; otherwise inspect checkpoint/config route.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("=== v71b decision ===")
print(decision.to_string(index=False))
print("\n=== torch load smoke results ===")
print(torch_df.to_string(index=False))
print("\n=== mmdet init smoke results ===")
print(mmdet_df.to_string(index=False) if len(mmdet_df) else "No mmdet init attempts.")
print("\n=== recommendation ===")
print(recommend.to_string(index=False))
print("\n=== issues ===")
print(issues_df.to_string(index=False) if len(issues_df) else "No issues found.")
