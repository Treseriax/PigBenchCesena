# v76 Full Unibo Annotation + Video Mapping Audit

## Purpose

This is the first full-pipeline step after Week8.

It inventories all available Unibo Excel annotation files and all raw videos, inspects workbook sheets, parses video filenames, and creates candidate Excel-to-video mappings.

## Important boundary

This stage does not:
- train a model,
- run tracking,
- create final moving GT,
- assign final behaviour labels.

It only prepares the full annotation/video mapping foundation.

## Counts

- Excel files: 6
- Excel sheets: 44
- Videos: 84
- Candidate matches: 1136
- Medium/high matched videos: 7
- Medium/high matched sheets: 2
- Hard issues: 0

## Next stage

v77 should review and finalize the annotation-video mapping table, including manual resolution for ambiguous or unmatched cases.
