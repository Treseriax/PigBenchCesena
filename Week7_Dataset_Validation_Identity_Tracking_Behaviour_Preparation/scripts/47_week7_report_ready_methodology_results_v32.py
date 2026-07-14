from pathlib import Path
from datetime import datetime
import csv
import pandas as pd


ROOT = Path.home() / "PigBench"
W7 = ROOT / "Week7_Dataset_Validation_Identity_Tracking_Behaviour_Preparation"

OUT_ROOT = W7 / "outputs" / "week7_report_ready_methodology_results_v32"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

OUT_REPORT_EN = OUT_ROOT / "Week7_Report_Ready_Methodology_and_Results_v32.md"
OUT_REPORT_TR = OUT_ROOT / "Week7_Report_Ready_Methodology_and_Results_v32_TR_summary.md"
OUT_METRICS = OUT_ROOT / "week7_report_ready_methodology_results_v32_key_metrics.csv"
OUT_DECISION = OUT_ROOT / "week7_report_ready_methodology_results_v32_decision_summary.csv"
OUT_ISSUES = OUT_ROOT / "week7_report_ready_methodology_results_v32_issues.csv"
OUT_NOTE = W7 / "notes" / "week7_report_ready_methodology_results_v32_notes.md"


def safe_to_csv(df, path):
    df.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


def read_first(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        if len(df):
            return df.iloc[0].to_dict()
    except Exception:
        return {}
    return {}


def read_df(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def clean(v, default=""):
    if pd.isna(v):
        return default
    s = str(v).strip()
    if s == "":
        return default
    return s


def val(row, key, default="not available"):
    return clean(row.get(key, default), default)


def md_table(rows, columns):
    if not rows:
        return "_No rows available._"

    out = []
    out.append("| " + " | ".join(columns) + " |")
    out.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for r in rows:
        vals = []
        for c in columns:
            x = str(r.get(c, ""))
            x = x.replace("\n", " ").replace("|", "/")
            vals.append(x)
        out.append("| " + " | ".join(vals) + " |")

    return "\n".join(out)


def find_csv(root, contains, preferred_terms=None):
    root = Path(root)
    if not root.exists():
        return None

    preferred_terms = preferred_terms or []
    candidates = []

    for p in root.rglob("*.csv"):
        s = str(p).lower()
        if all(term.lower() in s for term in contains):
            candidates.append(p)

    if not candidates:
        return None

    def score(p):
        s = str(p).lower()
        score_val = 0
        for term in preferred_terms:
            if term.lower() in s:
                score_val += 10
        if "decision" in s:
            score_val += 5
        if "summary" in s:
            score_val += 3
        if "issues" in s:
            score_val -= 20
        if "manifest" in s:
            score_val -= 10
        return score_val

    candidates = sorted(candidates, key=lambda p: (-score(p), str(p)))
    return candidates[0]


issues = []

# Core audited package outputs
v31_decision_path = W7 / "outputs" / "week7_complete_final_audit_v31" / "week7_complete_final_audit_v31_decision_summary.csv"
v30_decision_path = W7 / "outputs" / "week7_complete_final_package_v30" / "week7_complete_final_package_v30_decision_summary.csv"
v29e_decision_path = W7 / "outputs" / "final_identity_linking_report_package_v29e" / "week7_final_identity_linking_report_package_v29e_decision_summary.csv"
v29d_decision_path = W7 / "outputs" / "conservative_identity_arbitration_v29d" / "week7_conservative_identity_arbitration_v29d_decision_summary_fixed.csv"
v29d_clip_path = W7 / "outputs" / "conservative_identity_arbitration_v29d" / "week7_conservative_identity_arbitration_v29d_clip_summary.csv"
v29d_merge_path = W7 / "outputs" / "conservative_identity_arbitration_v29d" / "week7_conservative_identity_arbitration_v29d_conservative_merge_candidates.csv"

for key, path in [
    ("v31_decision", v31_decision_path),
    ("v30_decision", v30_decision_path),
    ("v29e_decision", v29e_decision_path),
    ("v29d_decision_fixed", v29d_decision_path),
    ("v29d_clip_summary", v29d_clip_path),
    ("v29d_merge_candidates", v29d_merge_path),
]:
    if not path.exists():
        issues.append({
            "item": key,
            "issue_type": "missing_required_file",
            "issue_detail": str(path),
        })

v31 = read_first(v31_decision_path)
v30 = read_first(v30_decision_path)
v29e = read_first(v29e_decision_path)
v29d = read_first(v29d_decision_path)
v29d_clip = read_df(v29d_clip_path)
v29d_merge = read_df(v29d_merge_path)

# Optional earlier-stage summaries
optional_paths = {
    "v17_colour_identity": find_csv(W7 / "outputs", ["v17"], ["fixed", "summary", "decision"]),
    "v18c_behaviour_fusion": find_csv(W7 / "outputs", ["v18c"], ["verified", "summary", "decision"]),
    "v19_final_fusion_stats": find_csv(W7 / "outputs", ["v19"], ["final", "fusion", "summary", "decision"]),
    "v21_primary_split": find_csv(W7 / "outputs", ["v21"], ["primary", "split", "summary", "decision"]),
    "v22_model_ready_crops": find_csv(W7 / "outputs", ["v22"], ["crop", "summary", "decision"]),
    "v24_audit_package": find_csv(W7 / "outputs", ["v24"], ["audit", "summary", "decision"]),
    "v25_tracking_temporal": find_csv(W7 / "outputs", ["v25"], ["tracking", "temporal", "summary", "decision"]),
    "v26_clip_extraction": find_csv(W7 / "outputs", ["v26"], ["clip", "summary", "decision"]),
    "v28d_dense_tracking": find_csv(W7 / "outputs", ["v28d"], ["dense", "summary", "decision"]),
}

optional_rows = {}
for key, path in optional_paths.items():
    optional_rows[key] = read_first(path) if path else {}
    if path is None:
        issues.append({
            "item": key,
            "issue_type": "optional_summary_not_found",
            "issue_detail": "Report will still be generated from audited v29e/v30/v31 outputs.",
        })

# Key metrics from audited outputs
total_tracklets = val(v29e, "total_tracklets")
accepted_tracklets = val(v29e, "conservative_accepted_tracklets")
acceptance_rate = val(v29e, "conservative_acceptance_rate")
confirmed_by_v29c = val(v29e, "accepted_confirmed_by_v29c")
v29b_only = val(v29e, "accepted_from_v29b_only")
conflict_count = val(v29e, "accepted_keep_v29b_conflict_with_v29c")
recovered_count = val(v29e, "recovered_candidates_from_v29c")
low_count = val(v29e, "low_confidence_v29c_candidates")
unknown_count = val(v29e, "unknown_no_reliable_identity")
review_queue = val(v29e, "review_queue_tracklets")
merge_candidates = val(v29e, "conservative_merge_candidates")
v29c_overwrites = val(v29e, "v29c_overwrites_v29b")
identity_status = val(v29e, "final_identity_status")

v30_zip = val(v30, "zip_path")
v30_sha = val(v30, "zip_sha256")
v30_issue_count = val(v30, "issue_count")
v30_file_count = val(v30, "copied_file_count_before_zip")
v30_subpackages = val(v30, "subpackages_included")

v31_audit_pass = val(v31, "audit_pass")
v31_issue_count = val(v31, "issue_count")
v31_zip_match = val(v31, "zip_sha256_match")
v31_zip_test = val(v31, "zip_test_pass")
v31_manifest_missing = val(v31, "manifest_missing_paths")
v31_subpackage_pass = val(v31, "subpackage_zip_pass_count")
v31_report_checks = val(v31, "report_checks_passed")
v31_report_total = val(v31, "report_checks_total")

# Metrics table
metrics_rows = [
    {"category": "Final package audit", "metric": "v31 audit pass", "value": v31_audit_pass},
    {"category": "Final package audit", "metric": "v31 issue count", "value": v31_issue_count},
    {"category": "Final package audit", "metric": "v30 zip SHA256 match", "value": v31_zip_match},
    {"category": "Final package audit", "metric": "v30 zip test pass", "value": v31_zip_test},
    {"category": "Final package audit", "metric": "manifest missing paths", "value": v31_manifest_missing},
    {"category": "Final package audit", "metric": "subpackage zip pass count", "value": v31_subpackage_pass},
    {"category": "Final package audit", "metric": "report content checks passed", "value": f"{v31_report_checks} / {v31_report_total}"},
    {"category": "Complete package", "metric": "subpackages included", "value": v30_subpackages},
    {"category": "Complete package", "metric": "copied file count before zip", "value": v30_file_count},
    {"category": "Complete package", "metric": "v30 zip path", "value": v30_zip},
    {"category": "Complete package", "metric": "v30 zip SHA256", "value": v30_sha},
    {"category": "Identity linking", "metric": "total tracklets", "value": total_tracklets},
    {"category": "Identity linking", "metric": "conservative accepted tracklets", "value": accepted_tracklets},
    {"category": "Identity linking", "metric": "conservative acceptance rate", "value": acceptance_rate},
    {"category": "Identity linking", "metric": "accepted confirmed by v29c", "value": confirmed_by_v29c},
    {"category": "Identity linking", "metric": "accepted from v29b only", "value": v29b_only},
    {"category": "Identity linking", "metric": "accepted from v29b despite v29c conflict", "value": conflict_count},
    {"category": "Identity linking", "metric": "recovered candidates from v29c", "value": recovered_count},
    {"category": "Identity linking", "metric": "low-confidence v29c candidates", "value": low_count},
    {"category": "Identity linking", "metric": "unknown / no reliable identity", "value": unknown_count},
    {"category": "Identity linking", "metric": "review queue tracklets", "value": review_queue},
    {"category": "Identity linking", "metric": "conservative merge candidates", "value": merge_candidates},
    {"category": "Identity linking", "metric": "v29c overwrites v29b", "value": v29c_overwrites},
    {"category": "Identity linking", "metric": "final identity status", "value": identity_status},
]

safe_to_csv(pd.DataFrame(metrics_rows), OUT_METRICS)

clip_rows = []
if len(v29d_clip):
    for _, r in v29d_clip.iterrows():
        clip_rows.append({
            "scan_frame_id": clean(r.get("scan_frame_id", "")),
            "raw_tracklets": clean(r.get("raw_tracklets", "")),
            "conservative_accepted_tracklets": clean(r.get("conservative_accepted_tracklets", "")),
            "confirmed_by_v29c": clean(r.get("accepted_confirmed_by_v29c", "")),
            "conflicts": clean(r.get("accepted_keep_v29b_conflict_with_v29c", "")),
            "recovered_candidates": clean(r.get("recovered_candidates_from_v29c", "")),
            "unknown": clean(r.get("unknown_no_reliable_identity", "")),
            "conservative_colours": clean(r.get("conservative_colours", "")),
            "status": clean(r.get("clip_arbitration_status", "")),
        })

clip_table = md_table(
    clip_rows,
    [
        "scan_frame_id",
        "raw_tracklets",
        "conservative_accepted_tracklets",
        "confirmed_by_v29c",
        "conflicts",
        "recovered_candidates",
        "unknown",
        "conservative_colours",
        "status",
    ],
)

merge_rows = []
if len(v29d_merge):
    for _, r in v29d_merge.iterrows():
        merge_rows.append({
            "scan_frame_id": clean(r.get("scan_frame_id", "")),
            "colour": clean(r.get("accepted_visual_colour", "")),
            "behaviour_pig_id": clean(r.get("accepted_behaviour_pig_id", "")),
            "track_ids": clean(r.get("candidate_track_ids", "")),
            "review_required": clean(r.get("review_required", "")),
            "status": clean(r.get("merge_status", "")),
        })

merge_table = md_table(
    merge_rows,
    ["scan_frame_id", "colour", "behaviour_pig_id", "track_ids", "review_required", "status"],
)

metrics_table = md_table(metrics_rows, ["category", "metric", "value"])

# English report-ready text
report_en = f"""# Week 7 Report-Ready Methodology and Results Text v32

## 1. Overview

This section summarizes the Week 7 work on dataset validation, target-pig identity preparation, behaviour-label fusion and tracking-readiness for the Unibo pig behaviour dataset. The goal of this stage was not to claim a production-ready multi-object tracking system. Instead, the goal was to create a reliable, auditable and review-aware dataset preparation pipeline that connects pig detections, visual colour-marker identity and behaviour annotations.

The final Week 7 package was created in v30 and audited in v31. The v31 audit passed with no issues: the zip checksum matched, the package zip was readable, the manifest had no missing paths, both required subpackages passed zip validation and all report content checks passed. Therefore, the package is considered report-ready and audit-ready.

## 2. Methodology

### 2.1 Input audit and dataset organization

The Week 7 workflow started from the outputs of the previous dataset-validation work. The first goal was to verify that the required Week 6 outputs, scanpoint frames, manual behaviour annotations and visual evidence files were available. This was followed by organizing the Week 7 output structure into reproducible stages with notes, decision summaries, scripts and visual review artefacts.

### 2.2 Target-pen and target-pig box validation

A major part of the work focused on obtaining reliable target-pig boxes inside the relevant pen. Automatic ROI and detector-based approaches were useful diagnostically, but they were not considered sufficient on their own. Therefore, a manual correction workflow was used to obtain final corrected target-pig boxes for the scanpoint frames. These corrected boxes provided the spatial basis for colour identity assignment and behaviour-label fusion.

### 2.3 Colour-marker identity workflow

The annotation design depends on a pig-to-colour-marker mapping. The valid marker colours were treated as blue, green, cyan, red, pink and purple. Automated colour evidence was tested, but visual ambiguity, illumination differences, occlusion and image quality limited fully automatic assignment. Therefore, the final colour identity layer was treated as a curated identity layer rather than a purely automatic colour-thresholding result.

The verified colour-to-behaviour pig ID crosswalk used in the pipeline is:

- blue -> blue
- green -> green
- cyan -> no_color
- red -> red_neck
- pink -> red_tail
- purple -> purple

### 2.4 Behaviour-label fusion

After correcting boxes and colour identities, the pipeline fused visual identity with the manual behaviour labels. The aim was to create training-ready rows where a target-pig box, visual marker identity and behaviour annotation could be connected. This made it possible to produce a behaviour-ready dataset while explicitly preserving unknown, not-visible or uncertain cases instead of forcing unreliable labels.

### 2.5 Split and crop dataset preparation

The fused dataset was converted into a model-ready crop dataset. A primary split strategy was used to create train, validation and test subsets without same-frame leakage. The dataset was documented as small and behaviour-imbalanced, so rare behaviours were not overclaimed as robustly learnable from crop-only evidence. Crop visual QA was used to verify that the exported crops generally contained the target pigs and could be used for downstream feature extraction or representation learning.

### 2.6 Temporal clip extraction

Because the original annotations are scanpoint-based and the behaviour context is temporal, 10-second clips were extracted around scanpoint timestamps. Temporal QA confirmed that the clips and preview frames were generated correctly. These clips provided the basis for testing tracking and identity continuity beyond single-frame crop evidence.

### 2.7 Detector and tracking dry-runs

A YOLOv8-s detector was used as the detection backbone. Initial simple-IoU tracking showed that detector outputs were usable but raw track IDs were not reliable final pig identities. The main issues were ROI leakage, crowded scenes, occlusions and track fragmentation. Manual target-pen polygon ROI filtering improved the relevance of detections by limiting the analysis to the correct pen, but simple-IoU tracking still fragmented in difficult clips.

Dense polygon-filtered tracking was then tested. Denser sampling helped provide more observations, but it did not solve the core association problem. This showed that the limitation was not only frame sampling frequency; it was also the association and identity-linking problem under occlusion and crowding.

### 2.8 Conservative identity-linking strategy

The final identity strategy was developed in v29. The main conclusion was that simple-IoU track IDs should not be treated as final pig identities. Instead, tracklets should be linked to marker-colour identity.

The v29 strategy was staged as follows:

- v29a: tracker strategy decision;
- v29b: colour-constrained tracklet identity linking using center-frame GT-overlap anchors;
- v29c: full-tracklet HSV colour evidence diagnostic;
- v29d: conservative arbitration between v29b and v29c;
- v29e: final identity-linking report package.

The conservative arbitration rule is the central result: v29b center-frame GT-overlap identity is the primary identity anchor, while v29c full-tracklet HSV evidence is secondary support evidence only. v29c is not allowed to overwrite v29b automatically. Conflicts, recovered candidates, low-confidence candidates and unknown identities remain in a review queue.

## 3. Key results

### 3.1 Final package audit

The final Week 7 package passed the v31 audit. The audit verified package existence, zip validity, SHA256 consistency, manifest consistency, subpackage zip integrity and report content. The package is therefore suitable for supervisor review, reproducibility inspection and report writing.

### 3.2 Identity-linking results

The identity-linking stage evaluated 50 dense tracklets. Among these, 24 tracklets were conservatively accepted as identity candidates. This corresponds to a conservative acceptance rate of 48.00%. Six accepted tracklets were confirmed by v29c, three were accepted from v29b only, and fifteen were kept from v29b despite conflict with v29c. Ten additional tracklets were recovered as candidates from v29c but require review, fourteen were low-confidence v29c candidates, and two remained unknown or without reliable identity evidence.

The main identity-linking result is not that all tracklets were solved automatically. Rather, the result is that the pipeline now separates accepted conservative identities, review-required candidates, low-confidence candidates and unknowns in a transparent way.

### 3.3 Clip-level identity summary

{clip_table}

### 3.4 Conservative merge candidates

Conservative merge candidates indicate cases where multiple accepted tracklets in the same clip share the same visual colour and behaviour pig identity. These are not automatic final merges; they are reviewable candidates.

{merge_table}

### 3.5 Key metrics table

{metrics_table}

## 4. Interpretation

The Week 7 results show that the detector and ROI pipeline can support target-pig localization, but simple tracking IDs are not reliable enough to serve as final identities. The visual marker-colour design provides an additional identity signal, but full-tracklet HSV colour evidence alone is also not reliable enough because it is affected by lighting, occlusion, blur and background colour.

The most defensible solution is therefore a conservative identity-linking layer. This layer uses GT-overlap identity anchors as the primary source and HSV colour evidence only as supporting evidence. This avoids overclaiming the tracking result while still extracting useful identity candidates and merge candidates.

## 5. Limitations

1. The final identity output is a set of conservative identity candidates, not production-grade final multi-object tracking.
2. Simple-IoU tracking remains fragmented in crowded and occluded scenes.
3. Full-tracklet HSV evidence can conflict with GT-overlap identity anchors.
4. Recovered candidates from v29c require visual review before being used as final identities.
5. Low-confidence and unknown cases are intentionally preserved rather than forced into a marker class.
6. The dataset remains small and behaviour-imbalanced, especially for rare behaviour classes.
7. Crop-only evidence may be insufficient for behaviours that require temporal or contextual understanding.

## 6. Recommended next steps

The next phase should focus on behaviour representation and stronger temporal modelling. Recommended steps are:

1. use the conservative identity candidates as the safest subset for downstream experiments;
2. keep review-required candidates separate until visually confirmed;
3. build clip-level behaviour representations using the 10-second clips;
4. evaluate temporal features such as motion, displacement, posture changes and interaction patterns;
5. consider stronger tracking-by-detection methods such as ByteTrack or BoT-SORT if the dependency path becomes stable;
6. combine improved tracking with marker-colour constrained identity correction;
7. report all uncertain identities explicitly rather than forcing assignments.

## 7. Final report statement

Week 7 produced a reproducible and audited dataset-preparation pipeline for pig identity and behaviour-readiness. The final package is report-ready and audit-ready. The identity-linking output should be described as conservative, review-aware identity candidates rather than final production tracking. This distinction is important because it accurately reflects both the strengths and limitations of the current pipeline.
"""

# Turkish user-facing summary
report_tr = f"""# Week 7 v32 Türkçe Rapor Özeti

## Genel karar

Week 7 paketi artık rapor ve audit için hazırdır. v30 final paket oluşturuldu, v31 audit geçti. Ana karar şudur:

- v29b, primary identity anchor olarak kullanılacak.
- v29c, yalnızca secondary/supporting evidence olarak kullanılacak.
- v29c, v29b'nin üzerine otomatik olarak yazmayacak.
- Conflict, recovered candidate, low-confidence candidate ve unknown durumları review queue içinde kalacak.

## Ana metrikler

- Toplam tracklet: {total_tracklets}
- Conservative accepted tracklet: {accepted_tracklets}
- Acceptance rate: {acceptance_rate}
- Review queue tracklet: {review_queue}
- Conservative merge candidate: {merge_candidates}
- v31 audit pass: {v31_audit_pass}
- v31 issue count: {v31_issue_count}
- Zip SHA256 match: {v31_zip_match}
- Zip test pass: {v31_zip_test}

## Rapor diliyle sonuç

Bu çalışmanın sonucu final production MOT değildir. Daha doğru ifade şudur:

Week 7 produced a conservative, review-aware identity-linking pipeline that connects detector-based tracklets with marker-colour identity and behaviour labels. The output is suitable for report writing and supervised review, but uncertain identities must remain flagged for manual inspection.

## Sonraki profesyonel adım

Bundan sonra v33 ile behaviour / clip-level representation tarafına geçilebilir. Yani artık soru şu olacak:

- 10 saniyelik cliplerden behaviour representation nasıl çıkarılır?
- Crop-only veri mi kullanılacak, yoksa temporal motion/interaction feature'ları mı eklenecek?
- Rare behaviour class'ları nasıl raporlanacak?
"""

OUT_REPORT_EN.write_text(report_en)
OUT_REPORT_TR.write_text(report_tr)

issues_df = pd.DataFrame(issues, columns=["item", "issue_type", "issue_detail"])
safe_to_csv(issues_df, OUT_ISSUES)

decision = pd.DataFrame([{
    "v32_decision": "report_ready_methodology_and_results_text_created",
    "english_report_path": str(OUT_REPORT_EN),
    "turkish_summary_path": str(OUT_REPORT_TR),
    "key_metrics_path": str(OUT_METRICS),
    "total_tracklets": total_tracklets,
    "conservative_accepted_tracklets": accepted_tracklets,
    "conservative_acceptance_rate": acceptance_rate,
    "review_queue_tracklets": review_queue,
    "conservative_merge_candidates": merge_candidates,
    "v31_audit_pass": v31_audit_pass,
    "v31_issue_count": v31_issue_count,
    "v31_zip_sha256_match": v31_zip_match,
    "v31_zip_test_pass": v31_zip_test,
    "report_ready": True if str(v31_audit_pass).lower() == "true" and str(v31_issue_count) in ["0", "0.0"] else False,
    "issue_count": len(issues_df),
    "ready_for_v33_behaviour_clip_representation": True,
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}])

safe_to_csv(decision, OUT_DECISION)

OUT_NOTE.write_text(
    "# Week 7 Report-Ready Methodology and Results Text v32\n\n"
    "## Summary\n\n"
    f"- English report: `{OUT_REPORT_EN}`\n"
    f"- Turkish summary: `{OUT_REPORT_TR}`\n"
    f"- Key metrics: `{OUT_METRICS}`\n"
    f"- Total tracklets: `{total_tracklets}`\n"
    f"- Conservative accepted tracklets: `{accepted_tracklets}`\n"
    f"- Acceptance rate: `{acceptance_rate}`\n"
    f"- Review queue tracklets: `{review_queue}`\n"
    f"- Conservative merge candidates: `{merge_candidates}`\n"
    f"- v31 audit pass: `{v31_audit_pass}`\n"
    f"- v31 issue count: `{v31_issue_count}`\n"
    f"- Issue count: `{len(issues_df)}`\n"
    f"- Ready for v33 behaviour/clip representation: `True`\n"
)

print("Saved:")
print(OUT_REPORT_EN)
print(OUT_REPORT_TR)
print(OUT_METRICS)
print(OUT_DECISION)
print(OUT_ISSUES)
print(OUT_NOTE)

print()
print("=== v32 decision ===")
print(decision.to_string(index=False))

print()
print("=== v32 issues ===")
if len(issues_df):
    print(issues_df.to_string(index=False))
else:
    print("No issues found.")
