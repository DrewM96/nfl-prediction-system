# Operations Runbook

## Weekly refresh

1. Run `python weekly_nfl_update.py` after the prior week's final game is official.
2. Confirm the command exits successfully and produces the expected next-week game count.
3. Review `update_log.json` for prediction season, training seasons, data cutoff, bundle hash, and every model's metrics.
4. Confirm `models/manifest.json` checksums and `data/predictions/<run-id>.json` exist.
5. Run `ruff check .`, `ruff format --check .`, and `pytest -q`.
6. Start Streamlit and verify This Week, Custom Game, all four prop paths, Performance, and Model Card.
7. Merge/deploy only after the automated artifact pull request is reviewed.

## Market collection

The `NFL market snapshot` workflow runs at six intentional weekly checkpoints: Tuesday and Thursday afternoon, shortly before Thursday night, before the Sunday early window, between the Sunday early and late windows, and before Monday night. It uses current `us` spreads and totals, budgeted at two credits per run.

Before first merge, applying the `live-market-test` label to a same-repository pull request performs one two-credit API smoke test. That labeled-PR path never publishes a snapshot or pushes an automation branch.

1. Keep `ODDS_API_KEY` only in GitHub Actions secrets or a local ignored `.env`; never paste it into an issue, log, or artifact.
2. Review the workflow's returned remaining/used/last-request quota fields after each run.
3. Review the draft PR and confirm `snapshot_at`, game count, team mappings, spread direction, book count, and dispersion.
4. Merge the consensus PR before running the weekly model update if that immutable prediction batch should include the snapshot.
5. Never add `data/market_private/` to Git. It contains the provider's book-level response and private historical research inputs.

Paid history is opt-in only. Estimate first with `python market_update.py --historical-at <UTC> --dry-run`. The workflow's manual form accepts the same optional UTC timestamp and hard credit budget, but it is a smoke test: historical output stays private, is never published, and disappears with the ephemeral runner. Run the command locally to retain ignored research snapshots. The default guard refuses a request above 20 credits. Use targeted weekly checkpoints, deduplicate stored snapshots, and monitor the quota rather than crawling five-minute history.

## Freshness and fail-closed behavior

The application stops when the schema-v3 manifest is missing, malformed, or fails a model checksum. It never substitutes a generic score. The header displays model creation date and training cutoff. Official injuries are labeled stale and are not applied when their available season differs from the prediction season.

Market data fails open: an unavailable, future, stale, or post-kickoff snapshot is omitted while the independent forecast remains available. The app displays the provider timestamp and dispersion whenever consensus is used.

## Incident response

- **Required nflverse feed fails:** do not publish a new run. Keep the last reviewed deployment and inspect upstream availability.
- **Optional roster/injury/snap feed fails:** the updater logs the feed name. Review whether player forecasts remain safe; suppress props if membership cannot be established.
- **Metric regression:** compare latest-season holdout and baseline fields. Player models may select their transparent baseline; investigate any game model or selected player model that underperforms without a documented reason.
- **Checksum/schema failure:** rebuild from the pinned environment. Never bypass verification or copy an unknown model file into `models/`.
- **Partial write/process interruption:** atomic replacements preserve the prior file. Rerun; do not hand-edit the manifest hash.
- **Odds API/quota failure:** do not retry in a loop. Check the redacted HTTP status and quota dashboard. The core forecast remains valid without market data.
- **Team mapping failure:** the event is skipped. Add and test an explicit provider-name mapping; never fuzzy-match two NFL teams in production.

## Rollback

Artifacts and code are released in the same reviewed commit. Revert the artifact-update commit or redeploy the previous known-good commit. Prediction-ledger records are evidence and must not be deleted or rewritten during rollback.

## 2026 launch gate

Before Week 1 publication, verify current 53-player rosters, official injury reports, quarterback starters, kickoff times, neutral venues, data cutoff, market timestamps, and quota health. Keep market comparison in paper mode until multiple held-out seasons and calibration monitoring establish a credible advantage over strong baselines and timestamped market prices.
