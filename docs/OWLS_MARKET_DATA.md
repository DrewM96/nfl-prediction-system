# Owls market context

The pre-change inventory is in [OWLS_AUDIT.md](OWLS_AUDIT.md). This migration
adds market context only. Model features, training, calibration calculations,
injury handling and model binaries are unchanged.

## Configuration and deployment

1. Install dependencies with `python -m pip install -r requirements-dev.txt`
   (use `requirements.txt` on the runtime host).
2. Set **process environment variables** on the backend:
   - `OWLS_INSIGHT_API_KEY`: Owls credential. This is the only credential source
     the adapter reads. No `.env` loader, config-file key, browser key, or
     constructor override is supported.
   - `GRIDLINE_MARKET_PROVIDER=owls` (default): freeze eligible cached Owls lines
     into newly generated forecasts. `legacy` preserves existing NFL The Odds
     API / CFB CollegeFootballData forecast context. This switch never rewrites
     published batches and does not change the Owls current-market worker.
   - `ODDS_API_KEY`: retained for explicit legacy snapshot jobs, historical
     benchmarks, and live parity. Keep it while parity is being evaluated.
   - `GRIDLINE_MARKET_DB`: absolute path on a **persistent local disk shared by
     the worker and Streamlit**, e.g. `/var/lib/gridline/market.sqlite3`.
     Default: ignored `data/market_private/market.sqlite3`.
3. Make the containing directory writable by the backend, restrict access, and
   back it up with SQLite's backup API. The database contains book-level data;
   do not commit or serve the database as a public file.
4. Warm the cache: `python current_market_update.py --sport both`.
5. Start one supervised backend worker:
   `python current_market_update.py --watch --sport both`.
6. Start the app: `streamlit run app.py`. Both processes must use the same DB
   path, deployment checkout, and current schedule artifacts. `Procfile` includes
   a `market` process command, but **separate ephemeral Heroku dyno filesystems
   are not shared storage**. Deploy on a host/container with a persistent local
   volume and colocated worker/app processes. Multi-host deployments need a
   shared database service implementation before rollout; do not put SQLite WAL
   on a network filesystem.
7. Configure the existing GitHub secret `OWLS_INSIGHT_API_KEY` for forecast CI.
   Weekly jobs perform a best-effort cache warmup. They intentionally retain
   the explicit legacy NFL calibration snapshot step. CI's ephemeral cache is
   only for that forecast run; it is not the production market-history store.
8. Run checks and parity below, then inspect both sports' cards, stale states,
   provider diagnostics, and worker logs before release.

On Windows, set keys in **Edit environment variables for your account → User
variables**, then open a new terminal. To load saved User variables in an
already-open PowerShell without printing them:

```powershell
$env:OWLS_INSIGHT_API_KEY = [Environment]::GetEnvironmentVariable('OWLS_INSIGHT_API_KEY', 'User')
$env:ODDS_API_KEY = [Environment]::GetEnvironmentVariable('ODDS_API_KEY', 'User')
```

The default NFL slate is `weekly_schedule.json`; CFB uses the immutable batch
referenced by `data/cfb/latest_prediction.json`. For a broader schedule or
upcoming unpublished slate, supply `--slate-nfl schedule.json` and/or
`--slate-ncaaf schedule.json`. Input is a list or a `predictions`/`games` envelope;
each row needs `game_id`, `home_team`, `away_team`, and timezone-aware
`commence_time`/`start_date` (NFL `gameday` + Eastern `gametime` also works).
Warm that slate before generation if its games are not on the published slate.
No available eligible line is stored as null, never fabricated or silently
substituted from another provider. Forecasts still complete.

## Polling and failure handling

Only the backend worker requests `/api/v1/{nfl,ncaaf}/odds` and `/splits`.
The normalized odds endpoint excludes exchanges. Each sport is polled at most
once every 300 seconds, two sequential requests per sport. At continuous maximum
usage this is 35,712 requests in 31 days, below the currently documented Rookie
75,000/month and 120/minute limits. Other uses of the account consume that same
quota. No historical endpoints, player props, or WebSocket add-ons are used.
See the [official API reference](https://owlsinsight.com/docs) for changing plan
limits, schemas and coverage (checked 2026-09-28).

Persisted SQLite poll gates and `BEGIN IMMEDIATE` prevent duplicate workers
from multiplying calls. A worker checks the gates every 30 seconds. Retries wait
at least five minutes, respect `Retry-After`, and back off an hour for 401/403.
Backoff is account-wide and survives restart. Exhausted provider minute/month
quota headers also pause requests. There is no rapid retry loop. The app reads
the local DB with a 30-second Streamlit cache and refreshes market-only fragments
every 60 seconds; it never requests Owls or re-runs inference for an odds update.

Odds and splits have separate error states. Last good odds/splits remain in the
cache after errors. Missing games retain cached lines explicitly marked stale;
missing books are reported, and the latest median uses books actually returned.
Missing split books retain their source time and an unavailable/stale marker.
Empty percentage fields remain null (zero is valid). Invalid envelopes produce
redacted errors; malformed rows, duplicate books, unknown teams, kickoff
mismatches and unmatched split IDs appear in diagnostics. Raw error bodies and
credentials are never logged. An application User-Agent is required by the
observed Owls edge service; the adapter sends `GRIDLINE/4.0 (market context)`.

Freshness is conservative: a contributing spread book older than 15 minutes,
missing/future source time, old capture time, provider stale flag, missing game,
or provider failure marks the current comparison stale. After kickoff the
comparison is marked stale/pregame-only. In-game lines are not presented as a
fresh edge against a pregame projection. Split freshness is evaluated separately
per book; stale books are excluded from cross-book agreement claims.

## Frozen versus current data

Published JSON prediction batches remain untouched. New rows include
`forecast_at`, `predicted_home_margin`, nullable `market_line`, and
`market_consensus` with provider, capture/source times, Owls event ID, spread,
total, median prices, book count, dispersion and source timestamps. Existing
batch `created_at` remains available for older forecasts. No new public splits,
movement or current-market fields enter the model feature frame.

New forecast context is attached **after** every existing predictive
calculation. The pre-existing NFL preseason calibration consumes the separate
legacy `market_consensus.json`; this migration does not route Owls into it or
change its behavior. `market_update.py` defaults to the Owls worker and never
overwrites that legacy calibration file. Explicit `--provider legacy` retains
the old publisher and paid historical functionality. This existing calibration
dependency is an additional reason not to disable The Odds API yet.

The mutable current cache is separate from the forecast ledger:

```
market movement = current home spread - frozen home spread
current home edge = frozen predicted home margin + current home spread
```

Positive edge favors home; negative edge favors away. These are point
differences, not win probabilities or expected financial returns. Cards label
the frozen projection, market at forecast, current/last cached line, movement,
edge and timestamps separately. NFL and college use the same summary layout
and show spread, moneyline, and over/under public action below it. Public action
is an equal-weight book average, not the percentage of pooled wagers or money:
Owls does not provide the sportsbook volume denominators. Tickets and handle
are averaged independently, using only books with both sides present for that
metric. Counts beside each metric identify coverage; missing data displays a
dash, and a single contributing book is labeled as such. Cached stale figures
remain visible with an "Includes stale data" badge. A disagreement badge uses
only fresh books favoring opposing sides. Per-book percentages and timestamps,
including DraftKings and Circa attribution, remain in Market details. Books may
report different lines and capture times. No opaque "sharp" score is produced.

## Database schema (SQLite user_version 1)

Tables are created idempotently on the first backend poll; no existing forecast
database needs migration.

| Table | Purpose |
| --- | --- |
| `market_cache(sport PRIMARY KEY, payload)` | Latest board, preserved last-good games, per-book odds/splits, errors, source/capture times and diagnostics |
| `poll_state(key PRIMARY KEY, next_at)` | Per-sport cadence and account-wide retry/quota gate |
| `market_observations` | Append-only per-event/per-book material changes |

Each observation has `id`, `event_id`, GRIDLINE `game_id`, `sport`, `sportsbook`,
`captured_at`, `source_timestamp`, home `spread`, home `spread_price`, home
`moneyline`, `total`, `splits_json`, complete normalized `payload`, and
`material_hash`. Payload retains away spread/moneyline prices and over/under
prices; splits retain each book's source time and nullable ticket/handle
percentages for home/away spread and moneyline and over/under total. Derived
fields include handle minus tickets in percentage points, majority ticket/money
sides, and disagreement. A tie or insufficient data yields null.

An index supports latest observations by sport/event/book. SQL triggers reject
UPDATE and DELETE of observations. Compare material hashes with the **last**
observation, ignoring timestamps: unchanged repeated polls add no rows, while
A→B→A adds three observations. The cache still advances capture/source times on
successful unchanged polls. Forecast files are never touched by this process.

Team matching reuses NFL names/codes and explicit aliases, plus the CFB registry
in `data/market_team_aliases.json`. It requires home/away orientation and exact
kickoff agreement against the slate. There is no fuzzy/prefix or mascot-only
match. Splits join by Owls event ID; `dk` is mapped to `draftkings`. An older v1
board without eventId uses Owls' documented sport/away/home/date identifier,
never a book-local numeric ID. Correct aliases only after reviewing diagnostics.

## Parity and retirement

```bash
python current_market_update.py --parity --sport both --output reports/owls-parity.json
python market_update.py --provider legacy --dry-run
```

Parity requests both providers for the same slate without changing forecasts or
the current cache. It reports coverage, missing games, per-game books/common
books, consensus and common-book spread differences, source timestamps,
freshness and raw-to-canonical team diagnostics. NFL and NCAAF The Odds API
requests use their respective sport keys. Paid historical NFL benchmarks remain
available through the existing command and explicit credit guard.

The initial live reports are `reports/owls-parity-live.json` and
`reports/owls-live-smoke.json`. The first sample found the one remaining NFL
game on both feeds with identical spreads and all 56 CFB games on both feeds,
with mean absolute consensus difference 0.067 and maximum 0.5 points. This is
one sample, not retirement approval. Many CFB Owls consensuses were stale under
the oldest-contributing-book rule (notably older DraftKings/Bovada quotes).
Live splits joined for the NFL game and all 56 CFB games; individual book and
market coverage varies.

Keep the legacy provider until several active NFL/CFB slates near kickoff show
acceptable game/book coverage, reviewed aliases, stable common-book differences,
and acceptable source freshness; test restart, 429, outage and stale UI behavior.
Investigate sustained spread differences over 0.5 points. Resolve the existing
NFL preseason calibration's legacy dependency in a separately authorized model
decision before retiring its feed. Then disable scheduled legacy **current**
snapshot requests, retain the adapter/credential for historical benchmarks as
needed, and remove secrets only after auditing all consumers. Do not delete the
integration or archived forecast context as part of this migration.

## Validation

```bash
python -m pytest -q
ruff check .
ruff format --check .
python -m mypy
```

CI additionally compiles entry points and both packages. Mypy is newly configured
for the six provider/service/CLI/UI-helper modules; the repository did not have
a static type gate before this change. Existing forecast, artifact, model and
Streamlit regression tests run in the full suite. No production forecast update
or model fitting is required to deploy the current-market layer.
