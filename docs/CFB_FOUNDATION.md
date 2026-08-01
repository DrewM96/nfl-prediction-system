# College Football Foundation

## Current status

The existing GRIDLINE Streamlit app includes a top-level NFL / College Football selector.
The College Football path loads its foundation, aggregate historical benchmark, independent
model bundle, and immutable prediction ledger; it does not load or mutate NFL models, injuries,
schedules, or prediction artifacts.

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

## Production forecasts

The first schema-v1 production bundle was fitted from the fixed selected configurations on all
completed 2018-2025 FBS-vs-FBS games. Its August 1, 2026 immutable Week 1 batch contains 51
games. The app displays it only after verifying both estimator checksums and the batch's exact
manifest hash.

The production path:

1. rebuilds historical team state in kickoff order;
2. updates form, advanced efficiency, and Elo only after completed games;
3. uses known scheduled dates, but never unknown results, for future rest calculations;
4. fits the fixed Ridge schemas and records residual uncertainty;
5. excludes listed market lines from every estimator and published forecast row;
6. writes each run once and moves only a small latest-run pointer;
7. stores scores separately when postgame scoring is added.

Forecasts remain provisional. The 2025 confirmation MAE is 12.642 for margin and 12.731 for
total, with residual standard deviations near 16 points. The listed market comparison remained
more accurate. These are probabilistic estimates, not betting recommendations or evidence of an
edge.

At the August 1 cutoff, CFBD had not populated 2026 returning-production or team-talent records.
The first batch therefore neutral-imputes those inputs while retaining available recruiting and
portal context. This coverage is recorded in both the model manifest and prediction metadata and
shown as a warning in the app. Refresh the batch after those feeds and final rosters are available.

The next stage is weekly result scoring, in-season refresh automation, coaching-change and
garbage-time ablations, and a rankings/performance view built from immutable runs.

Player props, live play-by-play, and GraphQL are out of scope until the game models are stable.
