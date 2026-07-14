from pathlib import Path
import csv
import json
import pandas as pd

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"

INTERFACE = W6 / "interface_demo"
GT = W6 / "outputs/unified_ground_truth"
VIS = W6 / "outputs/visual_label_check"
STATS = W6 / "outputs/dataset_statistics"
NOTES = W6 / "notes"

STATS.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

HTML = INTERFACE / "week6_static_visualization_viewer.html"
STREAMLIT_APP = INTERFACE / "week6_visualization_streamlit_app.py"
INDEX = INTERFACE / "week6_visualization_interface_index.csv"
NESTED_JSON = GT / "week6_scanpoint_annotations_nested_for_viewer.json"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


def resolve_path(p):
    if pd.isna(p):
        return None

    p = str(p).strip()

    if not p:
        return None

    candidates = [
        Path(p),
        W6 / p,
        ROOT / p,
        INTERFACE / p,
    ]

    for c in candidates:
        if c.exists():
            return c

    return None


def rel(path):
    try:
        return str(Path(path).relative_to(W6))
    except Exception:
        return str(path)


checks = []

def add(check, status, evidence, recommendation):
    checks.append({
        "check": check,
        "status": status,
        "evidence": evidence,
        "recommendation": recommendation,
    })


add(
    "static_html_viewer_exists",
    "PASS" if HTML.exists() else "FAIL",
    rel(HTML),
    "Regenerate static viewer." if not HTML.exists() else "No action."
)

add(
    "streamlit_app_file_exists",
    "PASS" if STREAMLIT_APP.exists() else "FAIL",
    rel(STREAMLIT_APP),
    "Regenerate Streamlit app file." if not STREAMLIT_APP.exists() else "No action."
)

add(
    "interface_index_exists",
    "PASS" if INDEX.exists() else "FAIL",
    rel(INDEX),
    "Regenerate interface index." if not INDEX.exists() else "No action."
)

add(
    "nested_json_exists",
    "PASS" if NESTED_JSON.exists() else "FAIL",
    rel(NESTED_JSON),
    "Regenerate nested viewer JSON." if not NESTED_JSON.exists() else "No action."
)

# HTML content check.
if HTML.exists():
    html_text = HTML.read_text(errors="ignore")
    expected_terms = [
        "Week",
        "frame",
        "annotation",
        "bbox",
        "behaviour",
    ]

    found = [t for t in expected_terms if t.lower() in html_text.lower()]

    add(
        "static_html_contains_expected_terms",
        "PASS" if len(found) >= 4 else "WARN",
        f"found_terms={found}",
        "Inspect or regenerate HTML viewer." if len(found) < 4 else "No action."
    )

# Index checks.
index_df = pd.DataFrame()

if INDEX.exists():
    index_df = pd.read_csv(INDEX)

    add(
        "interface_index_rows_72",
        "PASS" if len(index_df) == 72 else "WARN",
        f"rows={len(index_df)}",
        "Regenerate interface index if row count is wrong." if len(index_df) != 72 else "No action."
    )

    add(
        "interface_index_columns",
        "PASS" if len(index_df.columns) > 0 else "FAIL",
        f"columns={list(index_df.columns)}",
        "Inspect interface index." if len(index_df.columns) == 0 else "No action."
    )

# Nested JSON checks.
if NESTED_JSON.exists():
    try:
        data = json.loads(NESTED_JSON.read_text())
        if isinstance(data, dict):
            top_keys = list(data.keys())
            text_len = len(json.dumps(data))
        elif isinstance(data, list):
            top_keys = ["list"]
            text_len = len(json.dumps(data))
        else:
            top_keys = [type(data).__name__]
            text_len = len(str(data))

        add(
            "nested_json_parseable",
            "PASS",
            f"type={type(data).__name__}; top_keys={top_keys[:10]}; serialized_len={text_len}",
            "No action."
        )
    except Exception as e:
        add(
            "nested_json_parseable",
            "FAIL",
            f"{type(e).__name__}: {e}",
            "Regenerate nested viewer JSON."
        )

# Image path checks from index.
image_path_cols = []

if len(index_df):
    image_path_cols = [
        c for c in index_df.columns
        if "image" in c.lower() and "path" in c.lower()
    ]

    add(
        "interface_index_image_path_columns",
        "PASS" if image_path_cols else "WARN",
        f"image_path_cols={image_path_cols}",
        "Add image path column to index if needed." if not image_path_cols else "No action."
    )

    path_check_rows = []

    for col in image_path_cols:
        resolved = 0
        readable = 0
        total = 0

        for _, r in index_df.iterrows():
            total += 1
            p = resolve_path(r.get(col, ""))

            if p is not None:
                resolved += 1

                if CV2_AVAILABLE:
                    img = cv2.imread(str(p))
                    if img is not None:
                        readable += 1

        path_check_rows.append({
            "column": col,
            "total_rows": total,
            "resolved_paths": resolved,
            "cv2_readable_images": readable,
        })

    path_check = pd.DataFrame(path_check_rows)

    path_check_path = STATS / "week6_interface_smoke_test_image_path_check.csv"
    safe_to_csv(path_check, path_check_path)

    if len(path_check):
        best_readable = int(path_check["cv2_readable_images"].max())
        add(
            "interface_image_paths_readable",
            "PASS" if best_readable == 72 else "WARN",
            path_check.to_dict(orient="records"),
            "Inspect unresolved image paths." if best_readable != 72 else "No action."
        )

# Direct static serving instruction.
server_instruction = pd.DataFrame([
    {
        "method": "dependency_free_static_server",
        "command": "python -m http.server 8505 --directory interface_demo",
        "open_url": "http://localhost:8505/week6_static_visualization_viewer.html",
        "note": "Use Visual Studio Code port forwarding or Secure Shell port forwarding if running remotely.",
    }
])

server_instruction_path = STATS / "week6_interface_static_server_instruction.csv"
safe_to_csv(server_instruction, server_instruction_path)

audit = pd.DataFrame(checks)
audit_path = STATS / "week6_interface_dependency_free_smoke_test.csv"
safe_to_csv(audit, audit_path)

risk = audit[audit["status"].isin(["FAIL", "WARN"])].copy()
risk_path = STATS / "week6_interface_dependency_free_smoke_test_risks.csv"
safe_to_csv(risk, risk_path)

summary = pd.DataFrame([
    {
        "metric": "checks_total",
        "value": len(audit),
    },
    {
        "metric": "pass_count",
        "value": int((audit["status"] == "PASS").sum()),
    },
    {
        "metric": "warn_count",
        "value": int((audit["status"] == "WARN").sum()),
    },
    {
        "metric": "fail_count",
        "value": int((audit["status"] == "FAIL").sum()),
    },
    {
        "metric": "streamlit_runtime_required",
        "value": False,
    },
    {
        "metric": "static_html_smoke_test_ready",
        "value": int((audit["status"] == "FAIL").sum()) == 0,
    },
])

summary_path = STATS / "week6_interface_dependency_free_smoke_test_summary.csv"
safe_to_csv(summary, summary_path)

note_path = NOTES / "week6_interface_dependency_free_smoke_test_notes.md"

with open(note_path, "w") as f:
    f.write("# Week 6 Interface Dependency-Free Smoke Test\n\n")

    f.write("## Purpose\n\n")
    f.write(
        "Streamlit is not installed in the current environment. "
        "This smoke test validates the dependency-free static interface deliverable instead.\n\n"
    )

    f.write("## Result summary\n\n")
    f.write(summary.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Checks\n\n")
    f.write(audit.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Remaining risks\n\n")
    if len(risk):
        f.write(risk.to_markdown(index=False))
    else:
        f.write("No FAIL or WARN items remain for the static interface smoke test.")
    f.write("\n\n")

    f.write("## How to serve the static interface\n\n")
    f.write("```bash\n")
    f.write("cd ~/PigBench/Week6_Unibo_Dataset_Validation\n")
    f.write("python -m http.server 8505 --directory interface_demo\n")
    f.write("```\n\n")
    f.write("Then open:\n\n")
    f.write("`http://localhost:8505/week6_static_visualization_viewer.html`\n\n")

print("Saved:")
print(audit_path)
print(risk_path)
print(summary_path)
print(server_instruction_path)
print(note_path)

print()
print("=== Interface smoke test summary ===")
print(summary.to_string(index=False))

print()
print("=== Risks ===")
print(risk.to_string(index=False) if len(risk) else "None")
