# v77b Cleaned Annotation Window Candidates

## Purpose

v77b cleans the v77 normalized annotation candidates before final video mapping.

It:
- removes duplicate Excel file copies,
- excludes header/summary rows,
- excludes candidates without colour/identity,
- fixes observation windows to 10 seconds,
- produces clean pig-level annotation-window candidates for v78.

## Main counts

- Input v77 rows: 19278
- Clean rows: 18136
- Excluded rows: 1142
- Duplicate Excel rows excluded: 454
- Header/summary rows excluded: 559
- Missing-colour rows excluded: 129
- Behaviour classes preserved: 11
- Ambiguous generic red rows: 1360
- Hard issues: 0

## Boundary

This is not final moving GT yet.
v78 should resolve final video mapping and create the finalized annotation-window table.
