from pathlib import Path
import csv
import subprocess
import pandas as pd


PROJECT_ROOT = Path.home() / "PigBench"
W6 = PROJECT_ROOT / "Week6_Unibo_Dataset_Validation"

OUT = W6 / "outputs/feature_extractors"
NOTES = W6 / "notes"

OUT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as e:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"{type(e).__name__}: {e}",
        }


# 1) Python package audit.
packages = [
    "torch",
    "torchvision",
    "cv2",
    "mmdet",
    "mmcv",
    "mmengine",
    "ultralytics",
    "numpy",
    "pandas",
]

pkg_rows = []

for pkg in packages:
    code = (
        "import importlib.util\n"
        f"spec = importlib.util.find_spec('{pkg}')\n"
        "print('FOUND' if spec else 'MISSING')\n"
        f"import {pkg} as m\n"
        "print(getattr(m, '__version__', 'no_version'))\n"
        if pkg not in ["cv2"] else
        "import importlib.util\n"
        "spec = importlib.util.find_spec('cv2')\n"
        "print('FOUND' if spec else 'MISSING')\n"
        "import cv2 as m\n"
        "print(getattr(m, '__version__', 'no_version'))\n"
    )

    res = run_cmd(["python", "-c", code])
    lines = res["stdout"].splitlines()

    pkg_rows.append({
        "package": pkg,
        "status": lines[0] if lines else "ERROR",
        "version": lines[1] if len(lines) > 1 else "",
        "returncode": res["returncode"],
        "stderr": res["stderr"][:500],
    })

pkg_df = pd.DataFrame(pkg_rows)
pkg_path = OUT / "detector_python_package_audit.csv"
safe_to_csv(pkg_df, pkg_path)

# 2) Search model/config/script resources.
patterns = [
    "*.pth",
    "*.pt",
    "*.onnx",
    "*.py",
    "*.yaml",
    "*.yml",
    "*.json",
]

keyword_terms = [
    "yolo",
    "yolov8",
    "rtdetr",
    "rt-detr",
    "co-dino",
    "codino",
    "mmdet",
    "inference",
    "detect",
    "detector",
    "pig",
    "checkpoint",
    "config",
]

resource_rows = []

for pattern in patterns:
    for p in PROJECT_ROOT.rglob(pattern):
        if ".git" in str(p) or "__pycache__" in str(p):
            continue

        lower = str(p).lower()
        score = sum(1 for k in keyword_terms if k in lower)

        if score == 0 and p.suffix.lower() not in [".pth", ".pt"]:
            continue

        try:
            size_mb = p.stat().st_size / (1024 * 1024)
        except Exception:
            size_mb = None

        resource_rows.append({
            "path": str(p),
            "relative_path": str(p.relative_to(PROJECT_ROOT)),
            "suffix": p.suffix.lower(),
            "size_mb": round(size_mb, 3) if size_mb is not None else "",
            "keyword_score": score,
            "resource_guess": (
                "checkpoint" if p.suffix.lower() in [".pth", ".pt", ".onnx"]
                else "config_or_script"
            ),
        })

res_df = pd.DataFrame(resource_rows)

if len(res_df):
    res_df = res_df.sort_values(["resource_guess", "keyword_score", "size_mb"], ascending=[True, False, False])

res_path = OUT / "detector_resource_inventory.csv"
safe_to_csv(res_df, res_path)

# 3) Search /work for detector weights/configs too, but keep it targeted.
work_rows = []

for root in [Path("/work/pig/datasets"), Path("/work/pig")]:
    if not root.exists():
        continue

    for suffix in ["*.pth", "*.pt", "*.onnx", "*.yaml", "*.yml", "*.json"]:
        for p in root.rglob(suffix):
            if ".git" in str(p) or "__pycache__" in str(p):
                continue

            lower = str(p).lower()
            score = sum(1 for k in keyword_terms if k in lower)

            if score == 0 and p.suffix.lower() not in [".pth", ".pt", ".onnx"]:
                continue

            try:
                size_mb = p.stat().st_size / (1024 * 1024)
            except Exception:
                size_mb = None

            work_rows.append({
                "path": str(p),
                "suffix": p.suffix.lower(),
                "size_mb": round(size_mb, 3) if size_mb is not None else "",
                "keyword_score": score,
                "resource_guess": (
                    "checkpoint" if p.suffix.lower() in [".pth", ".pt", ".onnx"]
                    else "config_or_annotation"
                ),
            })

work_df = pd.DataFrame(work_rows)

if len(work_df):
    work_df = work_df.sort_values(["resource_guess", "keyword_score", "size_mb"], ascending=[True, False, False])

work_path = OUT / "detector_work_resource_inventory.csv"
safe_to_csv(work_df, work_path)

note_path = NOTES / "detector_bbox_resource_audit_notes.md"

with open(note_path, "w") as f:
    f.write("# Detector / Bbox Resource Audit Notes\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "This audit checks which detection packages, checkpoints, configs, and inference scripts are available for linking bounding boxes to the Week 6 scan-sampling frames.\n\n"
    )

    f.write("## Python package audit\n\n")
    f.write(pkg_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## PigBench detector resources\n\n")
    if len(res_df):
        f.write(res_df.head(80).to_markdown(index=False))
    else:
        f.write("No detector resources found in PigBench.")
    f.write("\n\n")

    f.write("## /work detector resources\n\n")
    if len(work_df):
        f.write(work_df.head(80).to_markdown(index=False))
    else:
        f.write("No detector resources found in /work targeted search.")
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write(
        "The next step is to select the most reliable detector route. "
        "If MMDetection and the PigBench YOLOv8 checkpoint/config are available, we can run inference on the 72 extracted scanpoint frames. "
        "If not, we will fall back to a simpler available detector route or document bbox extraction as pending.\n"
    )

print("Saved:")
print(pkg_path)
print(res_path)
print(work_path)
print(note_path)

print()
print("=== Python package audit ===")
print(pkg_df.to_string(index=False))

print()
print("=== PigBench detector resources top 40 ===")
print(res_df.head(40).to_string(index=False) if len(res_df) else "No resources found.")

print()
print("=== /work detector resources top 40 ===")
print(work_df.head(40).to_string(index=False) if len(work_df) else "No resources found.")
