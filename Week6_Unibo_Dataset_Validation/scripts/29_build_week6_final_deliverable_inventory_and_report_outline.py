from pathlib import Path
import csv
import pandas as pd


ROOT = Path.home() / "PigBench"
W6 = ROOT / "Week6_Unibo_Dataset_Validation"

OUT = W6 / "outputs"
OUT_STATS = OUT / "dataset_statistics"
OUT_GT = OUT / "unified_ground_truth"
OUT_FEAT = OUT / "feature_extractors"
OUT_VIS = OUT / "visual_label_check"
NOTES = W6 / "notes"

OUT_STATS.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n"
    )


def rel(path):
    try:
        return str(path.relative_to(W6))
    except Exception:
        return str(path)


def file_exists(path):
    return path.exists()


def file_size_mb(path):
    if not path.exists():
        return ""
    return round(path.stat().st_size / (1024 * 1024), 3)


key_deliverables = [
    {
        "category": "Unified ground truth",
        "priority": "core",
        "path": OUT_GT / "week6_unified_ground_truth_v2_with_recommended_split_v2.csv",
        "description": "Final recommended unified GT table with manual labels, recovered video paths, and recommended split v2.",
        "report_use": "Main dataset table",
    },
    {
        "category": "Split protocol",
        "priority": "core",
        "path": OUT_STATS / "week6_recommended_split_v2_summary.csv",
        "description": "Final recommended train/val/test summary.",
        "report_use": "Split protocol section",
    },
    {
        "category": "Split protocol",
        "priority": "core",
        "path": OUT_STATS / "week6_recommended_split_v2_video_level.csv",
        "description": "Video-hour level split table; prevents frame/video leakage.",
        "report_use": "Split protocol section",
    },
    {
        "category": "Split protocol",
        "priority": "core",
        "path": OUT_STATS / "week6_recommended_split_v2_behaviour_coverage.csv",
        "description": "Behaviour coverage by train/val/test.",
        "report_use": "Split limitations and coverage",
    },
    {
        "category": "Dataset statistics",
        "priority": "core",
        "path": OUT_STATS / "week6_final_dataset_overview_metrics.csv",
        "description": "Final dataset overview metrics.",
        "report_use": "Dataset statistics section",
    },
    {
        "category": "Dataset statistics",
        "priority": "core",
        "path": OUT_STATS / "week6_final_behaviour_distribution.csv",
        "description": "Behaviour class distribution.",
        "report_use": "Class distribution table",
    },
    {
        "category": "Dataset statistics",
        "priority": "core",
        "path": OUT_STATS / "week6_final_rare_behaviour_classes.csv",
        "description": "Rare behaviour classes requiring special handling.",
        "report_use": "Dataset limitations",
    },
    {
        "category": "Frame extraction",
        "priority": "core",
        "path": OUT_GT / "week6_scanpoint_frame_index.csv",
        "description": "72 extracted scanpoint frames linked to timestamps and videos.",
        "report_use": "Frame-label alignment section",
    },
    {
        "category": "Manual labels",
        "priority": "core",
        "path": OUT_GT / "week6_scanpoint_frame_labels_long.csv",
        "description": "Long-format table containing six manual colour-ID labels per scanpoint frame.",
        "report_use": "Ground-truth parsing section",
    },
    {
        "category": "Detection features",
        "priority": "core",
        "path": OUT_FEAT / "week6_yolov8s_all_scanpoint_detections_with_qc_flags.csv",
        "description": "YOLOv8-s detection table with QC flags.",
        "report_use": "BBox feature extractor section",
    },
    {
        "category": "Detection features",
        "priority": "optional",
        "path": OUT_FEAT / "week6_yolov8s_conservative_detections_score_ge_0_50.csv",
        "description": "Conservative detection subset for cleaner downstream feature extraction.",
        "report_use": "Optional conservative analysis",
    },
    {
        "category": "Marker features",
        "priority": "candidate",
        "path": OUT_FEAT / "week6_crop_colour_marker_features.csv",
        "description": "Crop-based green/blue/purple/red marker evidence for each bbox.",
        "report_use": "Candidate identity association section",
    },
    {
        "category": "Marker features",
        "priority": "candidate",
        "path": OUT_FEAT / "week6_candidate_bbox_to_colour_assignments.csv",
        "description": "High/medium-confidence candidate bbox-to-colour assignments.",
        "report_use": "Candidate identity association section",
    },
    {
        "category": "Feature extractor comparison",
        "priority": "core",
        "path": OUT_FEAT / "week6_feature_extractor_comparison_final_clean_v3.csv",
        "description": "Final cleaned comparison of implemented and future feature extractor families.",
        "report_use": "Feature extractor comparison section",
    },
    {
        "category": "Feature extractor comparison",
        "priority": "core",
        "path": OUT_FEAT / "week6_feature_extractor_recommendations_final_clean_v3.csv",
        "description": "Ranked recommendation of feature sets.",
        "report_use": "Feature extractor recommendation section",
    },
    {
        "category": "QC",
        "priority": "core",
        "path": OUT_STATS / "week6_postfix_final_consistency_audit.csv",
        "description": "Final post-fix consistency audit with all checks passing.",
        "report_use": "Validation and QC section",
    },
    {
        "category": "QC",
        "priority": "core",
        "path": OUT_STATS / "week6_detection_qc_conservative_subset_summary.csv",
        "description": "Detection QC and conservative subset summary.",
        "report_use": "Detector QC section",
    },
    {
        "category": "Visualization",
        "priority": "core",
        "path": OUT_VIS / "marker_bbox_overlay_v3_video/week6_marker_bbox_overlay_v3_slideshow.mp4",
        "description": "Slideshow video showing bbox, marker candidates, and manual labels.",
        "report_use": "Demo/video deliverable",
    },
    {
        "category": "Visualization",
        "priority": "core",
        "path": OUT_VIS / "bbox_count_warning_qc/all_bbox_count_warning_frames_contact_sheet.jpg",
        "description": "Contact sheet of low/high bbox-count QC frames.",
        "report_use": "Detector QC visual evidence",
    },
]

inventory_rows = []

for item in key_deliverables:
    p = Path(item["path"])
    inventory_rows.append({
        "category": item["category"],
        "priority": item["priority"],
        "exists": file_exists(p),
        "path": rel(p),
        "size_mb": file_size_mb(p),
        "description": item["description"],
        "report_use": item["report_use"],
    })

inventory = pd.DataFrame(inventory_rows)

inventory_path = OUT_STATS / "week6_final_deliverable_inventory.csv"
safe_to_csv(inventory, inventory_path)

missing = inventory[inventory["exists"] == False].copy()
missing_path = OUT_STATS / "week6_final_missing_deliverables.csv"
safe_to_csv(missing, missing_path)

# Full file inventory for traceability.
all_files = []

for folder in [OUT_GT, OUT_STATS, OUT_FEAT, OUT_VIS, NOTES]:
    if not folder.exists():
        continue

    for path in sorted(folder.rglob("*")):
        if path.is_file():
            all_files.append({
                "top_folder": rel(folder),
                "path": rel(path),
                "size_mb": file_size_mb(path),
                "suffix": path.suffix,
            })

all_inventory = pd.DataFrame(all_files)
all_inventory_path = OUT_STATS / "week6_full_output_file_inventory.csv"
safe_to_csv(all_inventory, all_inventory_path)

# Report outline.
outline_path = NOTES / "week6_final_report_outline.md"

with open(outline_path, "w") as f:
    f.write("# Week 6 Final Report Outline\n\n")

    f.write("## 1. Objective\n\n")
    f.write(
        "Validate the Unibo pig behaviour dataset workflow and build a reliable foundation for future behaviour classification. "
        "The pipeline links raw videos, manual Excel scan-sampling labels, scanpoint frames, pig detections, visual QC, and candidate feature extractors.\n\n"
    )

    f.write("## 2. Dataset and Raw Video Inventory\n\n")
    f.write("- Report raw video count, naming families, camera tokens, and direct TLC/c-token distinction.\n")
    f.write("- Use dataset overview metrics and raw naming analysis tables.\n\n")

    f.write("## 3. Manual Annotation Parsing\n\n")
    f.write("- Explain Excel structure: six colour IDs, 12 hourly blocks, six scan points per hour.\n")
    f.write("- Explain behaviour code mapping and final 432-label GT table.\n\n")

    f.write("## 4. Video Linkage and Recovered c-token Mapping\n\n")
    f.write("- Explain direct TLC matches for 07:00-15:00.\n")
    f.write("- Explain candidate recovered c-token videos for 15:00-19:00.\n")
    f.write("- Keep candidate recovered mappings marked as medium confidence.\n\n")

    f.write("## 5. Scanpoint Frame Extraction\n\n")
    f.write("- Explain 72 extracted scanpoint frames.\n")
    f.write("- Explain frame-to-label alignment: each scan frame has six manual labels.\n\n")

    f.write("## 6. YOLOv8-s Detection and BBox Features\n\n")
    f.write("- Explain detector setup, 540 detected bboxes, and mean bbox count per frame.\n")
    f.write("- Explain why bbox outputs are automatic detector outputs, not manual bbox ground truth.\n")
    f.write("- Include QC flags and conservative score >= 0.50 subset.\n\n")

    f.write("## 7. Crop Colour-Marker Candidate Features\n\n")
    f.write("- Explain green/blue/purple/red HSV marker features.\n")
    f.write("- Explain candidate bbox-to-colour assignment.\n")
    f.write("- State limitations: red is ambiguous between red_neck/red_tail; no_color cannot be marker-detected.\n\n")

    f.write("## 8. Visual QC Outputs\n\n")
    f.write("- Include bbox+label overlay v2.\n")
    f.write("- Include marker-bbox overlay v3.\n")
    f.write("- Include low/high bbox-count contact sheets.\n\n")

    f.write("## 9. Split Protocol\n\n")
    f.write("- Use recommended split v2.\n")
    f.write("- Explain video-hour level no-leakage design.\n")
    f.write("- Explain train/val/test sizes: 288/72/72 labels.\n")
    f.write("- Explain behaviour coverage: train 11/11, val 10/11, test 10/11.\n\n")

    f.write("## 10. Feature Extractor Comparison\n\n")
    f.write("- Compare manual labels, scanpoint frames, YOLOv8 bbox geometry, conservative bbox subset, crop marker features, and candidate bbox-to-colour assignment.\n")
    f.write("- Keep segmentation and embedding features as future extensions unless implemented later.\n\n")

    f.write("## 11. Consistency Audits and Limitations\n\n")
    f.write("- Report final post-fix audit: 0 failed checks.\n")
    f.write("- Discuss small dataset limitations, rare behaviours, candidate video mappings, automatic detector boxes, and unresolved identity mapping.\n\n")

    f.write("## 12. Conclusion\n\n")
    f.write(
        "The Week 6 pipeline is ready for reporting and future experiments. "
        "It provides a controlled dataset validation workflow, aligned GT labels, visualizations, detector-based bbox features, candidate marker features, and a no-leakage recommended split.\n\n"
    )

    f.write("## Key Deliverables\n\n")
    f.write(inventory.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Missing Deliverables\n\n")
    if len(missing):
        f.write(missing.to_markdown(index=False))
    else:
        f.write("No missing key deliverables.")
    f.write("\n")

# Executive summary.
exec_summary_path = NOTES / "week6_final_executive_summary.md"

with open(exec_summary_path, "w") as f:
    f.write("# Week 6 Final Executive Summary\n\n")
    f.write(
        "The Week 6 Unibo dataset validation pipeline successfully produced a unified and auditable dataset workflow. "
        "The final recommended ground-truth table contains 432 manual scan-sampling behaviour labels across 72 extracted scanpoint frames. "
        "Each scanpoint frame has six manual colour-ID labels. "
        "YOLOv8-s detection produced 540 pig bounding boxes across all 72 frames, and the primary detection table was preserved while QC flags and conservative subsets were added. "
        "Crop-based colour-marker analysis produced 196 medium/high candidate bbox-to-colour associations, which are useful for visual QC but not final identity-resolved ground truth. "
        "The recommended split v2 uses video-hour units to avoid leakage and contains 288 train labels, 72 validation labels, and 72 test labels. "
        "Final post-fix consistency audit passed with 0 failed checks. "
        "The dataset is ready for reporting, visualization demo, and future feature-extractor experiments.\n\n"
    )

    f.write("## Main limitations\n\n")
    f.write("- Labels come from one Excel sheet/day/camera-pen context.\n")
    f.write("- 15:00-19:00 c-token video mappings are candidate recovered mappings and should remain marked as medium confidence.\n")
    f.write("- Bboxes are automatic detector outputs, not manual bbox annotations.\n")
    f.write("- Bbox-to-colour identity is candidate-level only; red markers are ambiguous and no_color cannot be marker-detected.\n")
    f.write("- Rare behaviour classes remain limited and require careful handling in future classification experiments.\n")

print("Saved:")
print(inventory_path)
print(missing_path)
print(all_inventory_path)
print(outline_path)
print(exec_summary_path)

print()
print("=== Key deliverable inventory ===")
print(inventory.to_string(index=False))

print()
print("=== Missing key deliverables ===")
print(missing.to_string(index=False) if len(missing) else "None")
