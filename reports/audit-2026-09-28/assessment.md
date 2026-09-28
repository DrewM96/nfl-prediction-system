**NFL model assessment — September 28, 2026**

Reviewed local main at `5e6f566`, the active NFL release, immutable forecast and settlement files, feature/model/research code, and existing GitHub workflow results. Scope is NFL game forecasts; CFB and player props are not comprehensively audited here. Production code and forecasts were not changed.

**Judgment**

The project has a useful forecasting and evaluation foundation, but its representation of current team strength is limited. It mostly feeds recent box-score and efficiency averages into Ridge and gradient boosting. Those averages do not explicitly separate opponent quality, roster changes, quarterback identity, and environmental conditions. Changing a few weights cannot reliably repair those omissions.

I would prioritize input traceability and consistent forecast construction, then an opponent-adjusted team-strength model with explicit season transitions, then incremental quarterback availability and weather layers. These are hypotheses to test, not promised accuracy improvements. The existing evidence specifically argues against immediately doubling current-season weights.

**What the recorded results actually establish**

The active bundle was built September 26 with a September 24 completed-game cutoff. The checked-in settlements cover 33 distinct games: 16 in Week 1, 16 in Week 2, and one in Week 3. They do not support a complete three-week assessment. The regular NFL workflow runs Tuesday; settlements currently happen during model updates rather than through a separate frequent scoring task.

I independently recalculated recorded forecast errors using PowerShell, selecting one pregame forecast per game and the latest settlement for its run. The reproducible script and CSVs are in this directory. MAE means average absolute error in points; lower is better.

| Forecast policy | Settled games | Margin MAE | Total MAE |
|---|---:|---:|---:|
| First published, the Results default | 33 | 12.103 | 13.367 |
| Latest recorded at least 60 minutes before kickoff | 33 | 12.162 | 13.900 |

The first-published policy uses August 1 forecasts for Week 1. The latest policy uses September 5 forecasts for those games. Neither represents a deliberately enforced, uniform weekly forecast horizon. The latest policy also does not imply a forecast actually made 60 minutes before kickoff: it selects the latest available qualifying run.

Only 17 settled games have timestamp-valid market comparisons. Under latest-selection, model margin MAE is 12.152 against market 12.706; model total MAE is 12.288 against market 10.676. This small sample does not establish margin superiority. It does show why a complete, consistently timed, matched sample is essential before attributing disappointment to particular features.

The active manifest's reconstructed 2026 walk-forward metrics also trail its simple rolling baseline: margin 12.162 versus 11.571; total 13.934 versus 12.710, over 33 rows. These are historical reconstructions under the current code, distinct from the frozen published scores above. The full historical margin advantage over its baseline has a reported 95% week-block interval crossing zero. The independent model's added value is modest.

Evidence: [ledger summary](ledger-summary.json), [first forecasts](first-forecasts.csv), [latest forecasts](latest-60m-forecasts.csv), [active release](../../data/nfl_release.json), [active manifest](../../models/manifest-c04fc2d5f9502de9e1cb3b415535b6b4d5bd014ad7f7fc9f4d3ca955d38c9d4f.json), [forecast selection](../../nfl_prediction/results.py).

**Verified implementation findings**

1. **The advertised preseason calibration is absent from the checked-in forecasts.** All 11 NFL batches inspected have zero positive preseason-calibration weights. The latest market snapshot contains 29 usable multi-book spread matchups across 30 teams. `build_market_power_ratings` requires at least as many games as teams and sufficient matrix rank, so it returns no ratings for this snapshot. `apply_preseason_calibration` then returns the football forecast. Thus code and documentation describe an early-season support mechanism that the current release does not receive. This is not evidence that forcing the adjustment would improve results. A release should record the requested method, applied method, and explicit fallback reason. A persistent preseason prior or a regularized market model would need its own validation. Evidence: [rankings](../../nfl_prediction/rankings.py), [calibration](../../nfl_prediction/preseason.py), [market snapshot](../../market_consensus.json).

2. **The matchup builder and frozen schedule use different priors.** Historical/upcoming game rows use expanding league means. The exported `team_snapshot` is constructed with fixed `DEFAULT_PRIORS`, and the interactive builder consumes that snapshot. For the active BUF–LAC game, BUF pressure-allowed input is 0.12655 in the frozen forecast and 0.14404 in builder state. Identical teams and conditions can therefore produce different features through different paths. The discrepancy is established; its full prediction impact has not been measured. Use the same feature builder and cutoff for both, with a parity test. Evidence: [feature construction](../../nfl_prediction/features.py), [PredictionService](../../app.py).

3. **Missing inputs can become plausible values without a release-level coverage failure.** Optional feeds can fail independently. A missing game-team PBP summary falls back to league priors; absent roster-transition data becomes zero deviation from league average. These are defensible numerical fallbacks, but missing and ordinary are different states. There is no complete per-game source-count/age/imputation contract blocking publication. This is a verified risk in the code, not a claim that a specific current game's PBP is missing. Evidence: [data loader](../../nfl_prediction/data.py), [features](../../nfl_prediction/features.py), [roster features](../../nfl_prediction/roster.py).

4. **The inputs describe observed performance more directly than underlying strength.** Scores, yards, EPA and turnovers are not explicitly adjusted for prior opponents. A hard four-game window treats adjacent games equally, then drops the oldest abruptly. EPA and yardage filters differ, and the field called pressure is a QB-hit-or-sack rate, not a complete charted pressure measure. These definitions should be explicit in the data contract. Confirm play exclusions, denominators, identity joins and opponent mappings against source records before adding more statistics.

5. **Historical replay has limits.** `run_update(as_of=...)` uses the date for season/forecast selection but does not first truncate every loaded feed to knowledge available on that date. Do not use it as a historical timestamp-faithful replay without additional filtering. Weekly model folds are chronological, but historical features may incorporate prior calendar days within the same week; that differs from a fixed Tuesday issuance policy. Final historical roster/injury data also lack the full publication history of early-week information. Evidence: [pipeline](../../nfl_prediction/pipeline.py), [modeling](../../nfl_prediction/modeling.py).

6. **A small provenance bug deserves repair.** `PredictionLedger.settle` reuses the argument name `source` as its loop variable. Settlement documents consequently store the last result object in the source field rather than the intended provider name. Outcomes remain separately stored; I found this as an auditability defect, not an explanation for prediction errors. Evidence: [ledger](../../nfl_prediction/ledger.py).

**How to improve from season to season**

Separate learning the relationship between inputs and outcomes from estimating a team's current condition. The current estimator is retrained on roughly four prior seasons plus available current-season games, without explicit training-row recency weights. Current team features, however, use only four recent games for most statistics and eight for win rate. Retraining on more history does not itself produce a better season-opening team estimate.

A candidate replacement should maintain separate offensive and defensive strength estimates, adjust for opponents, regress uncertain estimates toward a league prior, and carry a discounted team-specific estimate across the offseason. QB/coaching/roster changes should affect the prior and its uncertainty. Updating should be faster where evidence is informative or continuity is low, slower where the statistic is noisy. Different signals should not share one arbitrary decay rule.

A simple design to test is: current strength = preseason estimate + a learned update toward opponent-adjusted current-season performance. Estimate the update rate from earlier seasons. Keep uncertainty as part of the state. Evaluate separate offense/defense and pace/scoring components for totals, instead of assuming the best margin representation is automatically best for totals.

Use multiple season transitions as outer evaluations: select settings using earlier years, predict the next season, repeat. Report Weeks 1–4, 5–8, and 9–18 separately, plus all-season scores and injury-event subsets. Preserve simple scoring/Elo-style baselines and the existing model. Compare frozen forecasts and same-time market snapshots on identical games. Use paired weekly blocks for uncertainty. Repeatedly inspecting 2025 across experiments makes it a reused validation season, not a permanently untouched final test. Reserve future observations for prospective confirmation.

**Should this season count more?**

Per observation, usually that is a reasonable hypothesis about changed personnel and time. But three games should not automatically outweigh an entire prior-season estimate. Relevance, noise, opponent strength, sample size and roster continuity determine the appropriate weight.

The production four-game statistic is `(sum(last four observations) + 2 × league prior) / 6` when all four observations exist. With no missed games or byes:

| Forecast week | Current-season observations' direct share | Prior-season observations' direct share | Explicit league-prior share |
|---|---:|---:|---:|
| 1 | 0% | 66.7% | 33.3% |
| 2 | 16.7% | 50.0% | 33.3% |
| 3 | 33.3% | 33.3% | 33.3% |
| 4 | 50.0% | 16.7% | 33.3% |
| 5 | 66.7% | 0% | 33.3% |

The league prior itself contains historical games; these are direct formula shares, not the final model's prediction weights. Win percentage uses a longer window.

I recovered the completed [September 15 GitHub experiment](https://github.com/DrewM96/nfl-prediction-system/actions/runs/34978424312), which evaluated 2022–2025 after loading 2020–2025:

| Weighting | Weeks 1–4 margin MAE | Weeks 1–4 total MAE | Full-season margin MAE |
|---|---:|---:|---:|
| Existing weighting | 10.194 | 10.380 | 10.004 |
| Current season 2× | 10.270 | 10.485 | 10.041 |
| Recency plus offseason decay | 10.423 | 10.552 | 10.082 |

Do not promote either alternative on these results. There is also a design limitation: the experiment normalizes observation weights back to the same total information mass. When all observations are from the prior season, their common offseason discount cancels. It tests relative weighting, not a genuinely weaker prior-year belief or larger offseason uncertainty. It does not rule out the season-transition model proposed above. [Downloaded benchmark](season-transition/benchmark.md), [experiment implementation](../../nfl_season_transition_benchmark.py).

**Injuries**

Official injury reports and reserve roster entries are frozen as context; production predictions explicitly do not use them. Manual injury scenarios are separate, with assumed point values. A QB shadow adjustment already runs prospectively without changing published forecasts: the latest batch has 15 eligible games, two with nonzero adjustments, and none settled in that batch yet.

Broad injury counts/snap-loss features failed the project's development/2025 promotion criterion. QB replacement modeling is more promising: across 723 games, margin MAE improved 10.367 → 10.239; across 85 nonzero-adjustment games, 11.836 → 10.746. But 2025 worsened 10.062 → 10.100; gains were concentrated in 2024. This supports continued investigation, not production activation. [QB benchmark](../../qb_adjustment_benchmark.json), [injury benchmark](../../injury_feature_benchmark.json).

The first-principles quantity is the expected lineup's value relative to the lineup already represented in the team estimate. If an injured starter has missed the last three games, the rolling baseline already reflects much of the replacement's performance. Applying a full absence penalty again would double-count it. Returning starters need the reverse adjustment.

Start with QB. Record expected starter and actual replacement with source timestamps, estimate availability probabilities by designation/practice/horizon, and estimate replacement value with strong shrinkage. The current research uses fixed severity proxies (Out 1.0, Doubtful 0.75, Questionable 0.35), infers starters from past usage, and falls back toward league-average QB performance. Those are uncertain assumptions. EPA also contains teammate and opponent effects. Reconcile injury reports with IR/reserve/inactives; the QB table currently consumes injury reports rather than the merged reserve-context payload. Evaluate newly absent, already absent, returning, and uncertain starters separately. Retain the shadow comparison while improving identity and timing.

**Weather**

There is no weather ingestion or production feature. Weather is a candidate contextual signal, especially for totals, passing and kicking; its incremental value must be measured rather than assumed.

Capture stadium coordinates, actual venue and roof status, and hourly wind/gusts, temperature and precipitation forecasts over kickoff through the expected game duration. Include capture time, forecast issuance time, provider and units. Treat indoor games appropriately, retain uncertainty for retractable roofs, and estimate nonlinear effects rather than applying a universal points deduction.

Historical observed weather can support exploratory analysis, but cannot represent what a Tuesday or Saturday forecast knew. Backtests should use archived forecasts available at the same issuance horizon. Open-Meteo's [Single Runs API](https://open-meteo.com/en/docs/single-runs-api) supports individual initialization times; its documented model-dependent archive start dates mean coverage must be checked before selecting a multiseason sample. Archive live forecasts now so future evaluation has an auditable history. Start with a small weather layer on totals and compare it with the unadjusted forecast.

**Recommended sequence and acceptance criteria**

1. **Establish trustworthy measurement and inputs.** Settle existing immutable forecasts through the latest completed games without retraining or overwriting them. Report first-published and fixed-horizon results separately. Add a per-game input audit showing source games, timestamps, valid play counts, denominator definitions, missing/imputed fields, roster/QB identities, and current-versus-prior-season contribution. Validate every current matchup plus a small historical transition sample against raw source records. Fix builder/schedule parity and explicit calibration fallback reporting. Acceptance: every required input is observed or visibly marked missing, and identical forecast contexts produce identical feature vectors.

2. **Run one controlled team-strength experiment.** Test a simple opponent-adjusted offense/defense state with an explicit offseason prior against the frozen existing model and simple baselines. Predeclare a small set of update rates. Keep the prediction-time convention and evaluation samples identical. Acceptance: worthwhile paired error improvement with uncertainty reported across multiple held-out seasons and no material early-season/calibration regression. Avoid selecting on tiny all-season gains.

3. **Continue QB shadow validation.** Add lineup identity, reserve coverage, timestamped status probabilities and baseline-relative changes. Acceptance: gains persist across seasons and new-absence/return subsets, then on future frozen forecasts. Do not infer injury irrelevance from failure of the current proxy features.

4. **Archive and evaluate weather.** Begin collection immediately; evaluate after stadium/roof/timestamp QA. Acceptance: same-horizon archived or prospective inputs improve totals on held-out data, with coverage and adverse-weather sample sizes visible.

5. **Make promotion reproducible.** Preserve an experiment registry with the hypothesis, input contract, feature/model hashes, evaluation horizon, candidate count, selection rule, rejected results and next untouched evaluation period. Monitor the simpler baseline alongside the deployed model and allow it to remain champion when complexity fails to earn its place.

**Verification and limits**

All 16 model files referenced by the active manifest match their SHA-256 hashes; the release pointer matches the immutable manifest hash. The [workflow that generated the active release](https://github.com/DrewM96/nfl-prediction-system/actions/runs/36275019259) reports 118 passing tests. I reviewed relevant tests and independently recalculated the checked-in ledger metrics. A usable local Python environment was not available on PATH, so I did not rerun pytest, retrain, or re-download and reconcile the full raw PBP data. Checksums and passing tests establish artifact consistency, not correctness of every provider value. That source-to-feature reconciliation is the first recommended implementation step.
