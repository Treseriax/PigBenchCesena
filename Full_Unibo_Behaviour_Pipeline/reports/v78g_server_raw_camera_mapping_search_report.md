# v78g Server Raw Camera Mapping Search

## Purpose

This stage searches raw/server-accessible files for evidence that maps Excel TLC camera names to encoded camera-code videos.

It is intentionally evidence-only:
- It does not apply mappings.
- It does not run tracking.
- It does not train a model.
- It does not guess unresolved TLC cameras.

## Counts

- Files seen: 38916
- Text files scanned: 12765
- Excel files scanned: 7
- Filename hits: 711
- Text hits: 35923
- Excel hits: 55
- TLC-code pair evidence rows: 297
- Direct pair evidence sources: 7
- Possible layout/doc/image files: 370
- Hard issues: 0

## Next

Review:
1. `v78g_tlc_code_pair_evidence.csv`
2. `v78g_candidate_mapping_summary.csv`
3. `v78g_possible_layout_or_doc_files.csv`

If these files contain a reliable mapping, use it in v78f. If not, external supervisor/layout confirmation is required.
