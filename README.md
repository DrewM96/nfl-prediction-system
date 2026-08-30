# GRIDLINE Football Prediction System

A point-in-time football forecasting application. The NFL side produces game margins, totals, win probabilities, player props, and timestamped market comparison. A separate College Football section has authenticated CollegeFootballData ingestion, isolated artifacts, provisional weekly forecasts, and an independent-model Top 30.

The current application is configured for the 2026 season. It loads the upcoming schedule independently from the completed seasons used for training, so preseason updates no longer fail when play-by-play for the new season does not yet exist.

## What changed in version 4

- The Odds API adapter collects NFL spreads, totals, American prices, provider timestamps, and quota headers without logging the API key.
- Book-level responses stay in ignored `data/market_private/` storage. The public `market_consensus.json` contains only median lines, prices, book counts, and dispersion.
- Sportsbook team names map to nflverse team codes, and home spreads are converted exactly once (`-6` becomes an expected home margin of `+6`).
- Snapshots after kickoff, snapshots from the future relative to a replay, and snapshots more than eight days before kickoff are rejected.
- Weeks 1-4 use a decaying preseason calibration derived from consensus-implied neutral-field
  team strength. The independent football forecast remains attached to every game for comparison
  and separate scoring; roster continuity remains limited to totals, where it improved validation.
- Power Rankings provide two explicit views: a latest market-implied neutral-field rating inferred
  from the full consensus schedule, and a football-only recent-form score frozen at the latest
  completed-game cutoff. The market ranking is not presented as an independent model.
- The immutable ledger records the market timestamp and later scores model and market errors on the same games.
- A guarded workflow collects early-week, Thursday, game-day, and prime-time snapshots for two credits per live request when using one region and two markets.

## Version 3 foundation

- Every historical feature is created before that game's result updates team or player state.
- Validation is chronological and expanding; no future season can train a model evaluated on an earlier season.
- Predictions are frozen in an immutable ledger and scored later without recomputation.
- Model files have an ordered feature schema, checksums, library versions, data cutoff, season metadata, and out-of-fold metrics.
- Game and player forecasts include residual uncertainty and 80% intervals.
- Receiving targets are counted before completions, and snap-count participation preserves active zero-opportunity games.
- Current prop menus use stable GSIS player IDs and exclude retired/cut players through the latest roster feed.
- Sportsbook signs are converted explicitly: a home favorite at `-6` corresponds to a `+6` market home margin.
- Manual injuries are session-only scenarios with visible availability assumptions; official injury-feed freshness is tracked separately.
- Dependencies are pinned, CI is required, and a scheduled updater opens a reviewable artifact pull request.

## Architecture

```mermaid
flowchart LR
    A[nflverse data] --> B[Point-in-time feature builder]
    B --> C[Expanding walk-forward validation]
    C --> D[Ridge and gradient-boosted ensemble]
    D --> E[Checksummed model bundle]
    E --> F[Pregame forecast ledger]
    E --> G[Streamlit read layer]
    F --> H[Postgame scoring and drift history]
    I[The Odds API] --> J[Private raw snapshot]
    I --> K[Derived consensus]
    K --> F
    K --> G
    L[CollegeFootballData] --> M[Ignored CFB cache]
    M --> N[Derived CFB artifacts]
    N --> G
```

Production code lives in `nfl_prediction/`:

- `config.py`: season context, paths, and division definitions.
- `data.py`: maintained `nflreadpy` ingestion with independent optional feeds.
- `features.py`: shared game/player feature construction and leakage controls.
- `modeling.py`: walk-forward validation, learned ensemble weights, uncertainty, manifests, and verified loading.
- `pipeline.py`: atomic training, forecasting, current rosters/injuries, and artifact publication.
- `ledger.py`: immutable prediction batches and separate result records.
- `market.py`: sportsbook sign and price-probability conversions.
- `odds.py`: provider client, quota accounting, team normalization, consensus, freshness, and private snapshot storage.

## Quick start

Python 3.11-3.13 is supported; CI uses Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python weekly_nfl_update.py
python weekly_cfb_update.py --season 2026
streamlit run app.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

Validate before publishing:

```bash
ruff check .
ruff format --check .
pytest -q
```

## Data and artifacts

The updater fetches schedules, play-by-play, weekly rosters, injury reports, and snap counts through `nflreadpy`. Raw downloads are not committed. Generated read artifacts remain small enough for Streamlit deployment.

`models/manifest.json` is the source of truth for model versions and results. `data/predictions/` stores each pregame run once; completed outcomes are written to a separate `.results.json` record. JSON writes and model replacement are atomic, so the app does not read half-written releases.

Never load model bundles from an untrusted source. Checksums detect corruption, but Python model serialization is still a trusted-artifact boundary.

## Market snapshots

Create `.env` from `.env.example`, set `ODDS_API_KEY`, and fetch current NFL spreads and totals:

```bash
python market_update.py --dry-run
python market_update.py
```

The dry run estimates credits without contacting the provider. Current `us` spreads plus totals are budgeted at two credits. A paid historical request is explicitly gated and estimated at 20 credits:

```bash
python market_update.py --historical-at 2025-09-07T15:00:00Z --dry-run
python market_update.py --historical-at 2025-09-07T15:00:00Z --max-credits 20
```

Historical raw and consensus files remain private and ignored. Current public consensus is a small analytical summary, not a standalone odds feed. The automated workflow uses the repository secret `ODDS_API_KEY` and opens a draft pull request containing only `market_consensus.json`.

### Historical model-versus-market benchmark

The historical benchmark rebuilds the independent model's expanding-window predictions, groups games by kickoff window, and requests consensus 30 minutes before kickoff. Both the ridge/boosting component weight and any later model/market blend use only residuals from earlier validation weeks.

Estimate the exact cost before making a paid request:

```bash
python historical_market_benchmark.py \
  --training-seasons 2022-2025 \
  --evaluation-seasons 2025 \
  --weeks 1-2 \
  --max-credits 300 \
  --dry-run
```

The guarded pilot is 14 snapshots and 280 credits for 32 games. The full matching 2023–2025 OOF period is 334 snapshots and 6,680 credits for 723 eligible games. Exact team/game records and all provider responses remain under ignored `data/market_private/`; only an aggregate report containing errors, learned weights, season splits, coverage, and disagreement buckets may be published.

The completed 723-game benchmark selected a 100% market weight for future margin and total forecasts. Margin MAE was 10.383 for the independent model, 9.618 for consensus, and 9.648 for the prequential blend. Total MAE was 10.536, 10.172, and 10.199 respectively. The result supports consensus as the tighter market-informed forecast while retaining the independent model as a research signal; it does not support claiming that disagreement is a betting edge. The production Week 1 margin uses the schedule-wide market-strength decomposition rather than copying a single game's line, then tapers that prior to 75%, 50%, 25%, and 0% in Weeks 2-5.

## Evaluation policy

Model selection uses expanding weekly splits. Production models are then fitted on every eligible row through the recorded cutoff. The manifest reports MAE, RMSE, bias, 80% interval coverage, pinball loss, latest-season holdout results, and rolling-baseline improvement. Game-margin models also report winner Brier score, log loss, and calibration error.

The app intentionally labels market analysis as paper analysis. A displayed difference is not evidence of a profitable edge. Agreement with the market usually means the independent forecast adds little actionable information; disagreement is a hypothesis to track, not a bet. Market claims require timestamped prices, no-vig probabilities, closing-line comparisons, adequate sample size, and uncertainty-aware evaluation.

### Football feature and model research

Football-only experiments do not require an Odds API key. The feature ablation derives
lagged success rate, early-down EPA, explosive-play rate, sack rate, neutral pass rate,
quarterback continuity, and performance volatility from nflverse play-by-play, then tests
all predeclared group combinations on the same expanding-window folds:

```bash
python football_feature_ablation.py --seasons 2022 2023 2024 2025
python football_model_tuning.py --seasons 2022 2023 2024 2025 --holdout-season 2025
```

The 723-game feature ablation retained the 24-feature core: every expanded feature set
worsened both margin and total MAE. The model-profile benchmark found that stronger Ridge
shrinkage was the only conservative change that improved both the 2023-2024 development
window and the separate 2025 confirmation season. Game models therefore use Ridge alpha
50 while player models retain alpha 10. The prequential all-OOF game MAE moved from 10.383
to 10.362 for margin and from 10.536 to 10.527 for totals. These are small improvements,
not evidence that the expanded statistics or a betting strategy add value.

`football_feature_benchmark.json` and `football_model_benchmark.json` preserve the full
results, including rejected configurations and season splits. The published historical
market benchmark remains a frozen comparison against the prior independent model; updating
that matched benchmark requires a separately authorized market-data run.

### Preseason roster transitions

The updater now falls back to nflverse's current seasonal roster before the weekly roster
feed opens for the new season. It compares each opening roster with prior-year regular-season
snaps and derives league-centered continuity for offense, defense, quarterback, offensive
line, skill positions, defensive front, secondary, and incoming veteran experience. Missing
PFR identifiers are matched through normalized player names, which is required for current
offensive-line coverage.

Roster inputs are weighted 100%, 75%, 50%, and 25% in Weeks 1-4, then become neutral as
current-season performance takes over. Historical opening membership is used without
game-day active/inactive status to avoid leaking late lineup information.

```bash
python roster_transition_ablation.py \
  --seasons 2022 2023 2024 2025 \
  --holdout-season 2025
```

The corrected 723-game chronological OOF evaluation retained quarterback, offensive-line, and skill-position
continuity for the total model only. Weeks 1-4 total MAE improved from 10.503 to 9.810;
the separate 2025 confirmation improved from 11.189 to 10.724; and all-OOF total MAE moved
from 10.527 to 10.387. Every roster configuration worsened margin MAE, so the margin model
remains unchanged. `roster_transition_benchmark.json` preserves all accepted and rejected
configurations.

### College Football foundation

College Football lives in the same Streamlit application but uses an independent package,
cache, artifacts, and update entry point. Configure `CFBD_API_KEY` in the environment or an
ignored local `.env`, then run:

```bash
python weekly_cfb_update.py --season 2026
```

The foundation validates stable FBS team and game IDs, conferences, regular-season calendar,
kickoff timestamps, neutral sites, and FBS/FCS classification. Raw CFBD responses are cached
under ignored `data/cfb/cache/`; the checked-in `data/cfb/foundation.json` contains aggregate
connectivity metadata only. The College Football tab verifies a separate checksummed model bundle
and matching immutable prediction batch before it displays forecasts.

The first fixed benchmark uses 5,705 completed FBS-vs-FBS games from 2018-2025. Features are
created before each result updates team state, while Ridge folds train on earlier weeks only.
The selected margin configuration combines Elo, recent form, advanced efficiency, returning
production, talent, recruiting, and portal context. The total configuration rejected preseason
inputs and retained Elo, form, and advanced efficiency only.

On the untouched 762-game 2025 season, margin MAE was 12.642 versus 14.062 for the Elo baseline;
total MAE was 12.731 versus 12.807 for the scoring-form baseline. Listed market lines were more
accurate at 11.846 and 12.366 respectively. Those lines lack snapshot timestamps, remain an
evaluation-only reference, and are never model inputs.

```bash
python cfb_historical_benchmark.py \
  --seasons 2018 2019 2020 2021 2022 2023 2024 2025 \
  --holdout-season 2025 \
  --min-train-rows 1200
```

`data/cfb/historical_benchmark.json` contains aggregate configurations, data coverage, rejected
alternatives, and development/holdout results without raw games.

The first production bundle fits those fixed schemas on all completed 2018-2025 FBS-vs-FBS
games and records 51 Week 1 forecasts from an August 1, 2026 cutoff. The August 30 refresh keeps
that immutable batch, updates the latest pointer to the 43 remaining Week 1 games, and adds the
published preseason feeds. Generate the earliest upcoming week, or intentionally select another
week, with:

```bash
python cfb_production_update.py --season 2026
python cfb_production_update.py --season 2026 --week 2
```

Each run writes checksummed estimators to `data/cfb/models/`, an immutable batch under
`data/cfb/predictions/`, a model-matched `data/cfb/power_rankings.json`, and a small latest-run
pointer used by the app. The Top 30 scores every remaining scheduled FBS-vs-FBS matchup with the
margin model, then uses a regularized schedule decomposition to express each team in points above
or below an average FBS team on a neutral field. It is an independent model rating rather than an
AP-style poll, and sportsbook lines are not ranking inputs. The independent forecast model
does not consume sportsbook lines. Forecasts remain labeled provisional because the margin
residual standard deviation is roughly 16 points and early-season roster inputs retain the
timing limitations described in the model documentation. The August 30 run covers all 138 teams
for talent and 136 for returning production; North Dakota State and Sacramento State retain
neutral returning-production values.

See [docs/CFB_FOUNDATION.md](docs/CFB_FOUNDATION.md) for boundaries and production safeguards.

## Operations

See [MODEL_CARD.md](MODEL_CARD.md) for intended use and limitations and [docs/OPERATIONS.md](docs/OPERATIONS.md) for weekly refresh, failure, rollback, and launch procedures.

## Data attribution

This project uses nflverse and CollegeFootballData. Terms and attribution details are in [NOTICE.md](NOTICE.md). Application code is MIT licensed.
