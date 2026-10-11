# Week 6 market-line restoration

The latest eligible forecast at least 60 minutes before kickoff remains the
selected model release. Missing market fields are filled from earlier immutable
releases only when season, game ID, both teams, kickoff and neutral-site status
agree. Existing selected-release lines always win. Spread and total are restored
independently.

Eligible source releases and snapshots must precede kickoff and the selected
publication; snapshots must be no older than eight days at that publication.
Available provider timestamps must not follow capture. Among eligible original
snapshots, the latest capture wins, followed by publication and stable run ID.
No final score or model performance participates in source selection. Recovery
never cascades through other recovered rows.

For 2026 College Week 6, the selected October 5 release
`20261005T200136Z-2e548c2f` had nine lines. The earlier October 4 release
`20261004T154651Z-9bbcae78` had 49 lines, eight overlapping the selected release.
The recovery adds 41 spreads and 41 totals, producing coverage of 50/58.

| Record | Original selected release | With verified recovery |
|---|---|---|
| ATS | 3–6–0 | 21–28–1 |
| Totals | 4–5–0 | 26–24–0 |

All 58 are final. The original winner record remains 35–23. Eight games have no
qualifying archived line: Miami (OH) at Massachusetts, Kent State at Western
Michigan, UConn at Temple, Rice at East Carolina, North Dakota State at UNLV,
James Madison at Georgia Southern, Louisiana at Louisiana Tech, and Hawai'i at
Arizona State. These stay excluded from market grading.

The [game-level reconciliation](reconciliation.csv) includes all 58 games,
original and restored lines, capture times, source release IDs and grades.
Forecast JSON, model projections and settlement files are unchanged. The UI
labels recovery and includes provenance in the game-log download. Other current
college weeks and NFL records have no eligible recovery changes.
