# Week 8 v62b2 Corrected Canonical Excel Observation Extraction

- v62b2 decision: corrected_canonical_excel_observation_extraction_completed
- Corrected canonical observation rows: 432
- Expected rows: 432
- Scanframes: 72
- All scanframes exactly six observations: True
- Canonical colour labels: blue;green;no_colour;purple;red_neck;red_tail
- Old v62b rows: 468
- Row delta corrected-minus-old: -36
- Hard issues: 0
- Warnings: 0
- Ready for v63 canonical GT v2 schema: True

Correction: v62b2 restricts extraction to exactly six canonical colour rows per time block, preventing leakage from the next Excel time block.
