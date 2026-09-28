# Market integration audit (before implementation, 2026-09-28)

No applicable AGENTS.md was found. The working tree was clean.

## Existing provider entry points

- `nfl_prediction/odds.py`: The Odds API v4 current NFL and paid historical
  clients, environment `ODDS_API_KEY`, quota accounting, consensus, snapshot
  storage, and forecast attachment.
- `market_update.py`: current/historical CLI with credit guard and private raw
  storage. `historical_market_benchmark.py` and
  `nfl_prediction/historical_market.py`: paid historical collection/evaluation.
- `.github/workflows/market-snapshot.yml`, `weekly-update.yml`, and
  `historical-market-benchmark.yml`: credentials and scheduled/manual requests.
- `.env.example`, README market section, `docs/OPERATIONS.md`, and `NOTICE.md`:
  setup, operation, and provider attribution.
- `market_consensus.json`, `market_benchmark.json`: published derived data;
  `pipeline.py`, `results.py`, `rankings.py`, `preseason.py`, and `app.py` consume
  these summaries or frozen copies. There is no current CFB The Odds API client.

## Consensus and forecast boundaries

NFL `build_consensus` uses an unweighted median per market across books, median
prices, min/max, interpolated IQR, book count, and oldest/latest book timestamps.
Home spread is negated to get expected home margin. CFB `normalize_lines` uses
medians of CFBD provider spreads/totals, negates spread, and counts providers.
CFB production captures those lines only on a forced current-season refresh.

`PredictionLedger` creates unique immutable batches under `data/predictions/`
and `data/cfb/predictions/`. `created_at` is the batch forecast timestamp;
`predicted_home_margin` is the prediction. Each prediction's `market_consensus`
holds spread/total and `snapshot_at`; NFL also stores `market_line` and a
separate market benchmark. Latest pointers, weekly schedule, and release state
are derived publication artifacts. Settlements are separate files.

Existing NFL preseason calibration already consumes the legacy consensus
summary. This migration must not connect the new current-market cache or Owls
snapshot to that calibration, model features, fitting, or inference. Attach
new forecast context after existing projections have been finalized.

## UI, normalization, storage, tests

`app.py` loads frozen batches/release state with 60-second Streamlit data caches.
NFL featured/row cards use `nfl_market_spread_label`, `nfl_market_edge_label`,
`market_tile`, and the comparison expander. CFB cards use
`cfb_market_spread_label`. They currently label frozen lines "Vegas".
Ad hoc NFL predictions have a separate market attachment path.

NFL provider normalization is the exact 32-name `TEAM_NAME_TO_CODE` dictionary.
Logo aliases include LAR→LA and WSH→WAS. CFB uses CFBD school names and numeric
game IDs; normalized team records include school, mascot, abbreviation and ID.
No shared cross-provider CFB alias resolver exists. NFL attachment checks teams
and kickoff; historical benchmark joins teams within a selected slate.

Storage uses atomic JSON writes, immutable prediction/settlement files, ignored
`data/market_private` raw snapshots, CFBD TTL disk caches, nflreadpy caches, and
Streamlit caches. There is no application database or backend odds poller.

Relevant existing tests: `test_odds`, `test_market`, `test_historical_market`,
`test_cfb_data`, `test_cfb_production_artifacts`, `test_ui`, `test_rankings`,
`test_results`, `test_ledger`, plus forecast/pipeline/artifact regression tests.
CI runs pytest, Ruff lint, Ruff format check, and Python compilation. It does
not configure a static type checker.
