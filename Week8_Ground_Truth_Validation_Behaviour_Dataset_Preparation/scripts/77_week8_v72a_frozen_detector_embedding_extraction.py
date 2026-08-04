from pathlib import Path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from datetime import datetime
import csv, json, zipfile, hashlib, traceback
import numpy as np
import pandas as pd


ROOT = Path.home() / "PigBench"
W8 = ROOT / "Week8_Ground_Truth_Validation_Behaviour_Dataset_Preparation"

V69B = W8 / "outputs" / "v69b_feature_baseline"
V71B = W8 / "outputs" / "v71b_detector_checkpoint_loadability_smoke_test"

SOURCE_FEATURE_ROWS = V69B / "Week8_Feature_Baseline" / "week8_v69b_feature_vectors.csv"
V71B_DECISION = V71B / "week8_v71b_decision_summary.csv"

OUT = W8 / "outputs" / "v72a_frozen_detector_embedding_extraction"
PKG = OUT / "Week8_Frozen_Detector_Embeddings"
NOTES = W8 / "notes"
REPORTS = W8 / "reports"
PROGRESS = W8 / "progress"

for p in [OUT, PKG, NOTES, REPORTS, PROGRESS]:
    p.mkdir(parents=True, exist_ok=True)

OUT_EMB = PKG / "week8_v72a_frozen_detector_embeddings.csv"
OUT_FAILED = PKG / "week8_v72a_failed_embedding_rows.csv"
OUT_ATTEMPTS = PKG / "week8_v72a_extraction_attempts.csv"
OUT_QA = PKG / "week8_v72a_embedding_extraction_quality_checks.csv"
OUT_README = PKG / "README_Week8_Frozen_Detector_Embeddings.md"
OUT_MANIFEST = PKG / "week8_v72a_manifest.json"

OUT_DECISION = OUT / "week8_v72a_decision_summary.csv"
OUT_ISSUES = OUT / "week8_v72a_issues.csv"
OUT_ZIP = OUT / "Week8_Frozen_Detector_Embeddings.zip"
OUT_SHA = OUT / "Week8_Frozen_Detector_Embeddings.sha256"
OUT_NOTE = NOTES / "week8_v72a_frozen_detector_embedding_extraction_notes.md"
OUT_REPORT = REPORTS / "week8_v72a_frozen_detector_embedding_extraction_report.md"
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


def read_csv_clean(path):
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
    return str(e).replace("\n", " ").replace("\r", " ")[:1200]


def load_preprocess_config(config_path):
    mean = [0.0, 0.0, 0.0]
    std = [255.0, 255.0, 255.0]
    bgr_to_rgb = True

    try:
        from mmengine import Config
        cfg = Config.fromfile(config_path)
        model_cfg = cfg.get("model", {})
        dp = model_cfg.get("data_preprocessor", {}) if isinstance(model_cfg, dict) else {}
        mean = dp.get("mean", mean)
        std = dp.get("std", std)
        bgr_to_rgb = dp.get("bgr_to_rgb", bgr_to_rgb)
    except Exception:
        pass

    return np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32), bool(bgr_to_rgb)


def prepare_image(path, mean, std, bgr_to_rgb, image_size=224):
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        return None, "image_read_failed"

    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)

    if bgr_to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)
    img = (img - mean.reshape(1, 1, 3)) / std.reshape(1, 1, 3)
    img = np.transpose(img, (2, 0, 1))

    return img, "ok"


def pool_feats(feats):
    import torch
    import torch.nn.functional as F

    if isinstance(feats, torch.Tensor):
        feats = [feats]

    if isinstance(feats, tuple):
        feats = list(feats)

    pooled = []

    for f in feats:
        if not isinstance(f, torch.Tensor):
            continue

        if f.ndim == 4:
            p = F.adaptive_avg_pool2d(f, 1).flatten(1)
            pooled.append(p)
        elif f.ndim == 3:
            pooled.append(f.mean(dim=1))
        elif f.ndim == 2:
            pooled.append(f)

    if not pooled:
        raise RuntimeError("No tensor feature maps returned by extract_feat.")

    return torch.cat(pooled, dim=1)


def run_extraction(source_rows, config_path, checkpoint_path, device):
    import torch
    from mmdet.apis import init_detector

    model = init_detector(config_path, checkpoint_path, device=device)
    model.eval()

    actual_device = next(model.parameters()).device

    mean, std, bgr_to_rgb = load_preprocess_config(config_path)

    batch_size = 2

    emb_rows = []
    failed_rows = []

    base_cols = [
        "canonical_gt_object_id",
        "scan_frame_id",
        "video_id",
        "behaviour_code",
        "current_split",
        "group_aware_split",
        "crop_type",
        "crop_path",
    ]

    n = len(source_rows)

    for start in range(0, n, batch_size):
        chunk = source_rows.iloc[start:start + batch_size].copy()

        tensors = []
        valid_meta = []

        for _, r in chunk.iterrows():
            crop_path = Path(clean(r["crop_path"]))
            img, status = prepare_image(crop_path, mean, std, bgr_to_rgb)

            if img is None:
                failed_rows.append({
                    "canonical_gt_object_id": clean(r.get("canonical_gt_object_id", "")),
                    "crop_type": clean(r.get("crop_type", "")),
                    "crop_path": str(crop_path),
                    "failure_type": status,
                })
                continue

            tensors.append(img)
            valid_meta.append(r)

        if not tensors:
            continue

        x = torch.from_numpy(np.stack(tensors, axis=0)).float().to(actual_device)

        with torch.no_grad():
            feats = model.extract_feat(x)
            emb = pool_feats(feats)
            emb = emb.detach().cpu().numpy().astype(np.float32)

        for i, r in enumerate(valid_meta):
            base = {c: clean(r.get(c, "")) for c in base_cols}
            base["embedding_source"] = "mmdet_yolov8_s_extract_feat_global_average_pool"
            base["embedding_device"] = str(actual_device)
            base["embedding_dim"] = int(emb.shape[1])
            feat_dict = {f"e{j:04d}": float(v) for j, v in enumerate(emb[i])}
            emb_rows.append({**base, **feat_dict})

        print(f"processed {min(start + batch_size, n)} / {n}")

    return pd.DataFrame(emb_rows), pd.DataFrame(failed_rows), str(actual_device)


issues = []

for p in [SOURCE_FEATURE_ROWS, V71B_DECISION]:
    if not p.exists():
        issues.append({
            "item": str(p),
            "issue_type": "hard_missing_required_input",
            "issue_detail": "Required input missing for v72a.",
            "severity": "hard",
        })

if issues:
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v72a_decision": "frozen_detector_embedding_extraction_blocked",
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v72b_embedding_baseline": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    raise SystemExit(1)


v71b = read_csv_clean(V71B_DECISION)
source = read_csv_clean(SOURCE_FEATURE_ROWS)

if len(v71b) == 0 or not bool_true(v71b.iloc[0].get("ready_for_v72_frozen_embedding_baseline", "")):
    issues.append({
        "item": str(V71B_DECISION),
        "issue_type": "hard_v71b_not_ready",
        "issue_detail": "v71b must be ready before v72a.",
        "severity": "hard",
    })

config_path = clean(v71b.iloc[0].get("recommended_config_path", ""))
checkpoint_path = clean(v71b.iloc[0].get("recommended_checkpoint_path", ""))

if not config_path or not Path(config_path).exists():
    issues.append({
        "item": "recommended_config_path",
        "issue_type": "hard_missing_recommended_config",
        "issue_detail": config_path,
        "severity": "hard",
    })

if not checkpoint_path or not Path(checkpoint_path).exists():
    issues.append({
        "item": "recommended_checkpoint_path",
        "issue_type": "hard_missing_recommended_checkpoint",
        "issue_detail": checkpoint_path,
        "severity": "hard",
    })

if len(source) != 744:
    issues.append({
        "item": "source_feature_rows",
        "issue_type": "hard_unexpected_source_rows",
        "issue_detail": f"Expected 744 source crop rows, found {len(source)}.",
        "severity": "hard",
    })

if any(i["severity"] == "hard" for i in issues):
    issues_df = pd.DataFrame(issues)
    safe_to_csv(issues_df, OUT_ISSUES)
    decision = pd.DataFrame([{
        "v72a_decision": "frozen_detector_embedding_extraction_blocked",
        "source_rows": int(len(source)),
        "hard_issue_count": int((issues_df["severity"] == "hard").sum()),
        "ready_for_v72b_embedding_baseline": False,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }])
    safe_to_csv(decision, OUT_DECISION)
    print(decision.to_string(index=False))
    print(issues_df.to_string(index=False))
    raise SystemExit(1)


attempt_rows = []
emb = pd.DataFrame()
failed = pd.DataFrame()
used_device = ""

devices = ["cpu"]

for device in devices:
    try:
        print(f"Trying extraction on {device}")
        emb, failed, used_device = run_extraction(source, config_path, checkpoint_path, device)

        attempt_rows.append({
            "device_attempted": device,
            "attempt_ok": True,
            "embedding_rows": len(emb),
            "failed_rows": len(failed),
            "error": "",
        })

        break

    except Exception as e:
        attempt_rows.append({
            "device_attempted": device,
            "attempt_ok": False,
            "embedding_rows": 0,
            "failed_rows": 0,
            "error": short_error(e),
        })
        print(f"Attempt failed on {device}: {short_error(e)}")
        print(traceback.format_exc())

attempts = pd.DataFrame(attempt_rows)
safe_to_csv(attempts, OUT_ATTEMPTS)

safe_to_csv(emb, OUT_EMB)
safe_to_csv(failed, OUT_FAILED)

embedding_cols = [c for c in emb.columns if c.startswith("e")]
embedding_dim = len(embedding_cols)

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

add_qa("source_rows", 744, len(source), len(source) == 744, "hard", "Source crop rows should be 744.")
add_qa("embedding_rows", 744, len(emb), len(emb) == 744, "hard", "One detector embedding per crop row is expected.")
add_qa("failed_rows", 0, len(failed), len(failed) == 0, "hard", "No embedding extraction failure should occur.")
add_qa("embedding_dim_positive", ">0", embedding_dim, embedding_dim > 0, "hard", "Embedding dimension should be positive.")
add_qa("crop_type_count", 2, emb["crop_type"].nunique() if len(emb) else 0, (emb["crop_type"].nunique() if len(emb) else 0) == 2, "hard", "Both tight and context10 crops should be embedded.")
add_qa("class_count", 11, emb["behaviour_code"].nunique() if len(emb) else 0, (emb["behaviour_code"].nunique() if len(emb) else 0) == 11, "hard", "All 11 behaviour classes should be present.")
add_qa("successful_device_attempt", True, bool(len(attempts[attempts["attempt_ok"] == True]) > 0), bool(len(attempts[attempts["attempt_ok"] == True]) > 0), "hard", "At least one device extraction attempt should succeed.")

qa = pd.DataFrame(qa_rows)
safe_to_csv(qa, OUT_QA)

hard_quality_failures = int(((qa["severity"] == "hard") & (~qa["passed"])).sum())

if hard_quality_failures:
    issues.append({
        "item": "v72a_quality_checks",
        "issue_type": "hard_embedding_extraction_quality_failed",
        "issue_detail": f"{hard_quality_failures} hard checks failed.",
        "severity": "hard",
    })

issues.append({
    "item": "claim_scope",
    "issue_type": "info_embedding_extraction_only",
    "issue_detail": "v72a extracts frozen detector embeddings only. It does not train or evaluate a classifier.",
    "severity": "info",
})

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail", "severity"])
safe_to_csv(issues_df, OUT_ISSUES)

hard_issue_count = int((issues_df["severity"] == "hard").sum()) if len(issues_df) else 0
warning_count = int((issues_df["severity"] == "warning").sum()) if len(issues_df) else 0
info_count = int((issues_df["severity"] == "info").sum()) if len(issues_df) else 0

manifest = {
    "version": "week8_v72a_frozen_detector_embedding_extraction",
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "source_rows": int(len(source)),
    "embedding_rows": int(len(emb)),
    "failed_rows": int(len(failed)),
    "embedding_dim": int(embedding_dim),
    "config_path": config_path,
    "checkpoint_path": checkpoint_path,
    "used_device": used_device,
    "claim_boundary": "frozen embedding extraction only; no classifier trained",
}
OUT_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

OUT_README.write_text(
    "# Week8 v72a Frozen Detector Embeddings\n\n"
    f"- Source crop rows: {len(source)}\n"
    f"- Embedding rows: {len(emb)}\n"
    f"- Failed rows: {len(failed)}\n"
    f"- Embedding dimension: {embedding_dim}\n"
    f"- Config: {config_path}\n"
    f"- Checkpoint: {checkpoint_path}\n"
    f"- Used device: {used_device}\n\n"
    "This stage extracts frozen detector embeddings only. It does not train or evaluate a classifier.\n"
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
    "v72a_decision": "frozen_detector_embedding_extraction_completed" if hard_issue_count == 0 else "frozen_detector_embedding_extraction_has_blocking_issues",
    "source_rows": int(len(source)),
    "embedding_rows": int(len(emb)),
    "failed_rows": int(len(failed)),
    "embedding_dim": int(embedding_dim),
    "crop_type_count": int(emb["crop_type"].nunique()) if len(emb) else 0,
    "behaviour_class_count": int(emb["behaviour_code"].nunique()) if len(emb) else 0,
    "used_device": used_device,
    "config_path": config_path,
    "checkpoint_path": checkpoint_path,
    "zip_path": str(OUT_ZIP),
    "zip_sha256": zip_hash,
    "hard_quality_failures": hard_quality_failures,
    "hard_issue_count": hard_issue_count,
    "warning_count": warning_count,
    "info_count": info_count,
    "issue_count": int(len(issues_df)),
    "ready_for_v72b_embedding_baseline": bool(hard_issue_count == 0),
    "claim_scope": "frozen_detector_embedding_extraction_only",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 8 v72a Frozen Detector Embedding Extraction\n\n"
    f"- v72a decision: {decision.iloc[0]['v72a_decision']}\n"
    f"- Source rows: {len(source)}\n"
    f"- Embedding rows: {len(emb)}\n"
    f"- Failed rows: {len(failed)}\n"
    f"- Embedding dimension: {embedding_dim}\n"
    f"- Used device: {used_device}\n"
    f"- Config: {config_path}\n"
    f"- Checkpoint: {checkpoint_path}\n"
    f"- Hard issues: {hard_issue_count}\n"
    f"- Ready for v72b embedding baseline: {bool(hard_issue_count == 0)}\n\n"
    "v72a extracts frozen YOLOv8 detector embeddings only. No classifier is trained here.\n"
)

OUT_REPORT.write_text(
    "# Week 8 v72a Frozen Detector Embedding Extraction Report\n\n"
    f"Decision: {decision.iloc[0]['v72a_decision']}\n\n"
    f"Embedding rows: {len(emb)}\n\n"
    f"Embedding dimension: {embedding_dim}\n\n"
    f"ZIP: {OUT_ZIP}\n\n"
    f"SHA256: {zip_hash}\n"
)

progress_row = pd.DataFrame([{
    "date": datetime.now().date().isoformat(),
    "stage": "v72a",
    "task_name": "Frozen detector embedding extraction",
    "status": "PASS" if hard_issue_count == 0 else "NEEDS_FIX",
    "input_summary": str(SOURCE_FEATURE_ROWS),
    "output_summary": str(PKG),
    "hard_issues": hard_issue_count,
    "warnings": warning_count,
    "next_action": "Run v72b embedding baseline classification." if hard_issue_count == 0 else "Fix v72a embedding extraction.",
}])

if OUT_PROGRESS.exists():
    old = pd.read_csv(OUT_PROGRESS)
    progress = pd.concat([old, progress_row], ignore_index=True)
else:
    progress = progress_row

safe_to_csv(progress, OUT_PROGRESS)

print("=== v72a decision ===")
print(decision.to_string(index=False))
print("\n=== attempts ===")
print(attempts.to_string(index=False))
print("\n=== QA ===")
print(qa.to_string(index=False))
print("\n=== issues ===")
print(issues_df.to_string(index=False) if len(issues_df) else "No issues found.")
