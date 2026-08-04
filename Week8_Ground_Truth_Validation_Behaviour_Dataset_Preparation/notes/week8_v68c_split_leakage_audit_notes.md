# Week 8 v68c Split Leakage Audit

- v68c decision: split_leakage_audit_completed
- Metadata rows: 372
- Train objects: 255
- Val objects: 39
- Test objects: 78
- Object ID split leakage: 0
- Scanframe split leakage: 0
- Clip path split leakage: 0
- Source-video overlap count: 9
- Classes missing from val: 4
- Classes missing from test: 2
- Rare class count: 3
- Hard issues: 0
- Warnings: 3
- Ready for v69a baseline sanity: True

If source-video overlap exists, baseline metrics should be reported as exploratory clip/object-level metrics, not as production generalization evidence.
