# College Football Foundation

## Current status

The existing GRIDLINE Streamlit app includes a top-level NFL / College Football selector.
The College Football path loads its foundation and aggregate historical benchmark; it does not
load or mutate NFL models, injuries, schedules, or prediction artifacts.

The foundation currently provides:

- bearer-authenticated CollegeFootballData REST v2 requests;
- a six-hour local disk cache to conserve the monthly request allowance;
- canonical FBS team IDs, game IDs, conference membership, kickoff timestamps, neutral-site
  flags, and opponent classifications;
- explicit FBS-vs-FBS identification rather than treating FCS opponents as ordinary peers;
- an aggregate public status artifact with no raw team or game records;
- a manual GitHub workflow that validates the configured repository secret.

## Historical benchmark

The fixed 2018-2025 research window contains 5,705 completed FBS-vs-FBS games and 4,470
expanding-week out-of-fold predictions. Configuration selection uses only seasons before 2025;
the 762-game 2025 season is reported as a separate confirmation.

| Target | Selected football inputs | Development MAE | 2025 MAE | 2025 baseline | 2025 listed market |
| --- | --- | ---: | ---: | ---: | ---: |
| Margin | Elo + form + advanced + preseason | 13.040 | 12.642 | 14.062 | 11.846 |
| Total | Elo + form + advanced | 13.170 | 12.731 | 12.807 | 12.366 |

The margin improvement is material and reproduced in the holdout. The total improvement is only
0.076 points in the holdout and must be treated as provisional. Preseason features improved margin
development and holdout MAE but worsened total development MAE, so they are not retained for totals.

CollegeFootballData's listed historical lines have no captured-at timestamp in this dataset. They
are not described as closing lines and are never used as football features.

Portal records are accepted only when their transfer date falls within 400 days before that
season's first kickoff. Returning-production, talent, and recruiting records are season-level
feeds rather than archived timestamp snapshots; this residual timing limitation remains disclosed
and is one reason the benchmark is not yet a production forecast bundle.

## Privacy and source boundaries

`CFBD_API_KEY`, `.env`, `data/cfb/cache/`, and `data/cfb/private/` must never be staged. Cache
files intentionally exclude authorization headers and API keys. Public artifacts must remain
derived outputs and declare whether raw provider data is published.

## Next production stage

Before publishing CFB predictions, the pipeline must:

1. fit the selected schemas on all eligible historical games;
2. save checksummed margin and total bundles with residual uncertainty;
3. construct the same point-in-time features for the 2026 schedule;
4. add an immutable college prediction ledger and postgame scoring;
5. expose weekly games, rankings, performance, and model metadata in the CFB tab;
6. keep listed market lines visually separate from independent forecasts;
7. add play-level garbage-time and coaching-change candidates only through new ablations.

Player props, live play-by-play, and GraphQL are out of scope until the game models are stable.
