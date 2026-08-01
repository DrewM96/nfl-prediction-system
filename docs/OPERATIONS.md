# Operations Runbook

## Weekly refresh

1. Run `python weekly_nfl_update.py` after the prior week's final game is official.
2. Confirm the command exits successfully and produces the expected next-week game count.
3. Review `update_log.json` for prediction season, training seasons, data cutoff, bundle hash, and every model's metrics.
4. Confirm `models/manifest.json` checksums and `data/predictions/<run-id>.json` exist.
5. Run `ruff check .`, `ruff format --check .`, and `pytest -q`.
6. Start Streamlit and verify This Week, Custom Game, all four prop paths, Performance, and Model Card.
7. Merge/deploy only after the automated artifact pull request is reviewed.

## Freshness and fail-closed behavior

The application stops when the schema-v3 manifest is missing, malformed, or fails a model checksum. It never substitutes a generic score. The header displays model creation date and training cutoff. Official injuries are labeled stale and are not applied when their available season differs from the prediction season.

## Incident response

- **Required nflverse feed fails:** do not publish a new run. Keep the last reviewed deployment and inspect upstream availability.
- **Optional roster/injury/snap feed fails:** the updater logs the feed name. Review whether player forecasts remain safe; suppress props if membership cannot be established.
- **Metric regression:** compare latest-season holdout and baseline fields. Player models may select their transparent baseline; investigate any game model or selected player model that underperforms without a documented reason.
- **Checksum/schema failure:** rebuild from the pinned environment. Never bypass verification or copy an unknown model file into `models/`.
- **Partial write/process interruption:** atomic replacements preserve the prior file. Rerun; do not hand-edit the manifest hash.

## Rollback

Artifacts and code are released in the same reviewed commit. Revert the artifact-update commit or redeploy the previous known-good commit. Prediction-ledger records are evidence and must not be deleted or rewritten during rollback.

## 2026 launch gate

Before Week 1 publication, verify current 53-player rosters, official injury reports, quarterback starters, kickoff times, neutral venues, and data cutoff. Keep market comparison in paper mode until multiple held-out seasons and calibration monitoring establish a credible advantage over strong baselines and timestamped market prices.
