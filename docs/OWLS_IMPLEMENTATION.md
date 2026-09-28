# Owls migration completion report

## 1. Architecture

Owls normalized NFL/NCAAF odds → existing median consensus calculation →
transactional SQLite cache and append-only per-book history → read-only
Streamlit market fragments. Splits join by Owls event ID and retain individual
sportsbook attribution. Forecast market attachment runs after projections are
finalized; current edge is arithmetic against the frozen projection. No model,
feature, injury calculation, or model binary was changed. Published prediction
artifacts were not regenerated or modified.

## 2. Files changed

| Files | Change |
| --- | --- |
| `nfl_prediction/owls.py` | Environment-only authenticated adapter, normalized odds/splits parsing, team/game matching, diagnostics |
| `nfl_prediction/current_market.py` | SQLite cache/history, polling/backoff, freshness, current edge, forecast context attachment |
| `nfl_prediction/market_parity.py` | Read-only NFL/NCAAF provider comparisons |
| `nfl_prediction/market_ui.py` | Escaped, explicit frozen/current/public-action presentation |
| `current_market_update.py` | Backend worker, schedule overrides, parity CLI |
| `data/market_team_aliases.json` | Explicit CFB school/mascot alias registry |
| `app.py` | Cached market fragments, updated card labels, existing-batch timestamp display |
| `nfl_prediction/pipeline.py`, `cfb_prediction/production.py` | Attach eligible market context after predictive calculations |
| `nfl_prediction/ledger.py` | Explicit forecast timestamp and market-line defaults for new rows, including legacy CFB |
| `nfl_prediction/odds.py` | Retained legacy client with NCAAF sport selection for parity |
| `market_update.py` | Owls default with explicit legacy/historical path |
| `.env.example`, `Procfile` | Environment setup and optional backend worker command |
| `.github/workflows/weekly-update.yml`, `weekly-cfb-update.yml` | Best-effort Owls warmup; preserve legacy calibration refresh |
| `.github/workflows/market-snapshot.yml` | Explicit legacy selection during transition |
| `.github/workflows/ci.yml`, `pyproject.toml`, `requirements-dev.txt` | Reproducible pinned static type check for new market modules and entry-point compilation |
| `tests/test_owls.py`, `tests/test_app_sports.py` | New market coverage and changed card-label assertions |
| `README.md`, `NOTICE.md`, `docs/OPERATIONS.md` | Setup, attribution and operating guidance |
| `docs/OWLS_AUDIT.md`, `OWLS_MARKET_DATA.md`, `OWLS_IMPLEMENTATION.md` | Pre-change audit, full deployment/schema guide, completion inventory |
| `reports/owls-parity-live.json`, `reports/owls-live-smoke.json` | Credential-free live verification reports |

## 3. Schema

SQLite schema version 1 creates `market_cache`, `poll_state`, and
`market_observations`, an observation index, and triggers rejecting history
updates/deletes. The cache/database is ignored and private; forecasts stay in
their existing immutable JSON ledgers. First live capture persisted 545
observations, 104 with splits. No existing database migration is required.

## 4. Tests and checks

44 new parametrized test cases cover NFL/CFB parsing and identity, median/sign
parity, duplicate books, frozen/current separation, edge/movement, per-book
splits, missing/invalid percentages, stale timestamps, missing games, failed
odds/splits, HTTP/authentication/rate-limit failures, restart-safe backoff,
quota exhaustion, append-only/deduplicated history including returning prices,
provider parity, HTML escaping/labels, and immutable forecast bytes.

Final checks: full pytest suite (175 tests), `ruff check .`,
`ruff format --check .`, `python -m mypy` (six newly gated files), and compilation
of every root Python entry point and both packages. Existing model/forecast,
artifact and Streamlit tests are included. A scan of changed/new repository
files found neither configured API credential. The condensed card now exposes
a shared NFL/college summary row and equal-weight book averages for spread,
moneyline, and over/under tickets and handle. Per-metric book counts, stale
flags, missing data, and fresh-book disagreement remain explicit. Per-book
attribution and other market context are collapsed under Market details.

## 5. Live parity results

Final sample: 2026-09-28 22:02 UTC.

| Metric | NFL | NCAAF |
| --- | ---: | ---: |
| Remaining pregame games on published slate | 1 | 56 |
| Owls coverage | 1/1 | 56/56 |
| The Odds API coverage | 1/1 | 56/56 |
| Mean absolute all-book consensus difference | 0 | 0.067 |
| Maximum absolute consensus difference | 0 | 0.5 |
| Maximum common-book median difference | 0 | 0 |
| Owls book counts per game | 11 | 5–10 |
| Fresh Owls consensus under conservative rule | 1/1 | 0/56 |

The other 14 games in the published NFL slate had already started/finished and
were absent from both boards. Out-of-slate events are diagnosed, not force-joined.
The differences in CFB all-book consensus were absent when using common books.
The CFB stale labels reflect older contributing book timestamps; the first
smoke sample had two fresh games, the later parity sample had none.

Live public action joined the NFL game (DraftKings and Circa) and all 56 CFB
games (DraftKings on 50, Circa on 52). Missing books/markets remain explicit.

## 6. Limitations

- These are two samples of one slate, not a multi-week parity certification.
- Source freshness, particularly older CFB DraftKings/Bovada quotes, prevents
  treating all returned medians as fresh. No stale quote is silently promoted.
- History starts with deployment and scheduled capture; no historical API was
  used and nothing was backfilled.
- SQLite requires colocated processes and persistent local disk. Separate
  ephemeral app/worker hosts need a shared database service implementation.
- Matchups must be on the supplied GRIDLINE slate with matching kickoff and
  explicit aliases. Warm an upcoming slate before generating its forecasts.
- Existing NFL preseason calibration already uses the legacy market snapshot.
  It remains unchanged and is deliberately isolated from the new Owls data.

## 7. Exact deployment steps

Follow [OWLS_MARKET_DATA.md](OWLS_MARKET_DATA.md) for environment and host details.

1. Install `requirements.txt` on the runtime host and `requirements-dev.txt` in CI.
2. Set backend `OWLS_INSIGHT_API_KEY`, retain `ODDS_API_KEY`, and set
   `GRIDLINE_MARKET_PROVIDER=owls` and an absolute persistent `GRIDLINE_MARKET_DB`.
3. Configure the GitHub `OWLS_INSIGHT_API_KEY` secret for the weekly jobs.
4. With the deployment's current slate files in place, run
   `python current_market_update.py --sport both`. Supply `--slate-nfl` /
   `--slate-ncaaf` for upcoming unpublished schedules.
5. Supervise `python current_market_update.py --watch --sport both` and run
   `streamlit run app.py` on the same persistent disk/DB path. Do not use
   separate unshared ephemeral dynos for these two processes.
6. Verify cards and stale/error states, database backups, and worker logs. Run
   `python current_market_update.py --parity --sport both --output reports/owls-parity.json`.
7. Keep collecting/reviewing parity before turning off legacy jobs. No model
   retraining, forecast regeneration, or rewrite of published batches is needed.

## 8. Retirement recommendation

**Do not disable The Odds API yet.** Coverage and common-book spreads are
promising, but CFB source freshness needs investigation, NFL coverage needs
testing on a larger active slate, and the pre-existing preseason calibration
dependency needs a separately authorized decision. Retain the adapter and
historical benchmark path even after retiring routine legacy current polling.
