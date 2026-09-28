**NFL input integrity and research — September 28, 2026**

This change separates source validation, recorded forecast evaluation, and experimental model improvements. Existing model bundles and frozen predictions are preserved. The new code takes effect when deployed and the updater next runs; current release artifacts have not been regenerated with the new metadata.

**Implemented behavior**

- Live team snapshots and scheduled feature rows now use the same expanding league prior. Future builder forecasts therefore use the same team inputs for equivalent contexts.
- Each new scheduled forecast records the last eight source games, seasons, QB identity, EPA/dropback counts, missing statistics, roster-transition join coverage, and direct current-season contribution to four-game features. Required missing PBP/statistics block an update before fitting models. Missing roster continuity is explicitly marked degraded.
- The production loader rejects missing required PBP columns, duplicate play IDs, and duplicate schedule IDs. This complements numerical model validation; it does not certify every provider value.
- The market calibration path records whether calibration was requested/applied and why it fell back. It does not force an unidentified market-rating solution.
- Historical `run_update(as_of=...)` calls are rejected before fetching live data. Use cached, explicitly dated inputs for historical research.
- Settlement source labels are correctly retained. `nfl_results_update.py` refreshes outcomes without training or rewriting a forecast. It waits at least eight hours after kickoff before accepting scored NFL schedules, and produces first-published, latest-before-24h, and latest-before-60m reports. The last two policies select available forecasts and do not imply issuance at precisely that horizon.
- `nfl_weather_update.py` archives hourly pregame weather with source coordinates, roof context, capture time, units, full private response, and public derived summary. Unknown or conflicting venue mappings remain unavailable. Weather is not applied to predictions. The live blended weather endpoint supplies no individual model initialization timestamp; that field remains explicitly unknown.
- The new QB research layer measures expected QB performance against the participation mixture already represented in recent form. It includes explicit reserve statuses, records identities and assumptions, and cannot change the published forecast. Starter identity remains inferred from prior usage and unavailability weights remain designation proxies. Those assumptions need prospective validation and a confirmed-lineup source before promotion.
- `nfl-observations.yml` prepares a daily results/weather update and opens a draft artifact PR after tests. It shares the existing NFL concurrency group, preserves pending weather captures, and uploads raw weather responses for 90 days. Durable public summaries are committed on merge; copy raw workflow artifacts into long-term storage before retention expires if full-response retention is required. This workflow is not scheduled until the code reaches the default branch.

**Reconciliation and current results**

`nfl_input_audit.py` reconstructed all 15 active-release game feature vectors from a fresh public nflverse download, truncated to the September 24 game-date cutoff. All 15 matched exactly within 1e-8; no recent PBP coverage failures were detected. This is a reconciliation against the latest revised source vintage, not a recreation of the original download bytes. Cache-file hashes and per-game source details are preserved in `reports/input-reconciliation.json`.

Independent result settlement added six append-only settlement documents covering 14 additional unique completed games. There are now 47 settled games; one Week 3 game remains unplayed at the refresh time. The latest-before-60m policy gives margin MAE 10.665 and total MAE 13.549. On 31 matched games, model/market margin MAE is 9.888/9.677; model/market total MAE is 12.484/11.129. These are recorded forecasts, not recomputed predictions. Full results and coverage are in `data/nfl_results_summary.json`.

The first weather collection captured 30 upcoming games and withheld one: `2026_05_PHI_JAX` has stadium ID `JAX00` but stadium name `Tottenham Hotspur Stadium`. A name-and-ID check prevented requesting Jacksonville weather for a London-labeled game. Venue coordinates come from the pinned public registry identified in `data/weather_venues.json`; coordinate/roof coverage is not a claim of official venue confirmation.

**Season-transition experiment**

Hypothesis: opponent-adjusted offensive/defensive scoring state and explicit offseason uncertainty add information beyond four-game averages.

The fixed candidates carry 50% or 75% of prior team strength into a new season, add offseason uncertainty, and update separate offensive/defensive estimates using opponent-adjusted score residuals. They supply three additional features to the existing Ridge/boosting architecture. Observation variance, process variance and home advantage were fixed before evaluation. This is a small scoring-state experiment; it does not yet estimate causal roster/coaching effects or model possessions separately.

Both core features and strength state are frozen before each historical week. Training data cover 2020–2025. For each outer evaluation season 2023–2025, configuration selection uses earlier out-of-fold seasons only, including the unchanged core as an eligible choice. Component blending remains prequential. This produces 816 evaluated games, including 192 in Weeks 1–4.

| Target | Reference MAE | Selected candidate MAE | Candidate minus reference 95% week-block interval |
|---|---:|---:|---|
| Margin, all outer seasons | 10.309 | 10.186 | [-0.262, +0.004] |
| Total, all outer seasons | 10.416 | 10.316 | [-0.197, -0.016] |
| Margin, Weeks 1–4 | 10.516 | 10.502 | [-0.290, +0.273] |
| Total, Weeks 1–4 | 10.296 | 10.165 | [-0.338, +0.066] |

Decision: retain as research. Totals merit further work, but the early-season improvement is uncertain and this is reused historical validation. Do not choose whichever single candidate looks best after viewing the outer results. Archive future shadow forecasts and examine calibration and per-season stability before considering deployment. See `reports/strength-transition/experiment.json`, `benchmark.json`, and the paired game-level CSVs.

**Reproduce**

Use Python 3.12 and `requirements-dev.txt`. On this Windows workspace the interpreter is `.venv/Scripts/python.exe`.

```text
python nfl_results_update.py
python nfl_weather_update.py
python nfl_research_data.py --seasons 2020 2021 2022 2023 2024 2025 2026
python nfl_input_audit.py --cache data/cache/research
python nfl_strength_benchmark.py --cache data/cache/research
python nfl_lineup_update.py --cache data/cache/research
ruff check .
ruff format --check .
pytest -q
```

The audit/benchmark cache contains `schedules.parquet` plus `{pbp,rosters,snaps,injuries}-{season}.parquet`. It is ignored by Git. `nfl_input_audit.py` requires the seasons named in the active manifest; the benchmark requires 2020–2025. `nfl_research_data.py` uses public nflreadpy loaders, records acquisition metadata, preserves historical cached seasons unless `--refresh` is passed, and refreshes the current season. No paid market data is requested. The benchmark uses the pinned project dependencies.

**Next evidence needed**

1. Review and deploy the integrity/observation changes so every future release records the new provenance and daily snapshots persist remotely.
2. Continue the predeclared strength experiment prospectively. Current historical totals gains are small; retain the core model as reference.
3. Add timestamped confirmed QB/backup announcements and empirically calibrated availability probabilities. The new baseline-relative layer fixes the conceptual double-counting problem but does not establish a causal points value. A separate prospective research record was captured for the remaining Week 3 game; it has no outcome yet.
4. Resolve source venue conflicts and confirm roof status. A weather effects model requires a sufficient same-horizon archived sample. Do not fit on future observed weather and label it an early-week forecast.
5. Store new experiment contracts, source/code hashes, rejected alternatives and genuinely future confirmation results. These gates prevent another cycle of selecting tiny improvements from repeated historical trials.

Weather API reference: https://open-meteo.com/en/docs. Archived issuance-time runs: https://open-meteo.com/en/docs/single-runs-api. Weather attribution: Open-Meteo.com. Stadium coordinates: https://github.com/greerreNFL/stadiums/tree/950a7cec2b577786189edfae9748a9bde58d7758.

Validation completed locally: 131 tests passed; Ruff lint and formatting checks passed; Python compilation passed. All 11 original forecast batch contents remain unchanged. Re-running the settlement command added zero documents, confirming idempotence. All 30 archived weather game windows passed completeness checks. The GitHub observation workflow has been prepared but has not been deployed or executed remotely.
