# Model Card: NFL Prediction System v4

## Intended use

The system produces pregame analytical forecasts for NFL home margin, total points, win probability, passing yards, receiving yards, receptions, and rushing yards. It is designed for research, model monitoring, and paper-market comparison.

It is not a source of guaranteed outcomes, individualized financial advice, or a substitute for current roster, weather, lineup, and market information.

## Model design

Each target combines a standardized ridge regression with a shallow gradient-boosted regressor. Game models use Ridge alpha 50, selected on 2023-2024 and checked separately on 2025; player models retain alpha 10. The blend weight is learned from chronological out-of-fold predictions. The final production estimators are fitted on every eligible record through the manifest cutoff.

Game inputs use shrunk rolling team form, EPA/play, pressure rates on a shared rate scale, turnovers, rest, division status, week, and home field. Player inputs use shifted opportunity and efficiency features, real intended targets, stable player IDs, opponent defense, and offensive snap participation when published.

During Weeks 1-4, the total model also uses league-centered continuity for the prior season's primary quarterback, offensive line, and skill positions. Players are weighted by prior regular-season snaps; the roster effect decays from 100% to 25% and becomes neutral after Week 4. A chronological ablation reduced early-season total MAE from 10.503 to 9.810 and confirmed the direction independently in 2025 (11.189 to 10.724). Roster features worsened margins, so the margin model does not consume them.

The feature builder also derives point-in-time success rate, early-down EPA, explosive-play rate, sack rate, neutral pass tendency, quarterback continuity, and performance volatility for research. A 723-game expanding-window ablation found that every predeclared combination worsened both game targets, so these candidates are published in the team snapshot for continued monitoring but are not production model inputs. Stronger game-model Ridge shrinkage produced a smaller, consistent gain: prequential margin MAE moved from 10.383 to 10.362 and total MAE from 10.536 to 10.527.

Prediction intervals use the standard deviation of chronological residuals. They describe historical model error, not every source of real-world uncertainty. Injury availability remains a separate scenario until a timestamped report history supports safe training.

The football model is independent of sportsbook lines. When a fresh market snapshot is available, the application also shows the consensus spread and total as a separate benchmark. It does not blend that line into the model until a chronological historical test demonstrates that a learned blend improves genuinely unseen games.

Historical comparison uses a nested chronology. Candidate football models train on earlier games; their weekly component weights use only earlier out-of-fold residuals; the model/market weight then uses only earlier matched market weeks. A line captured after kickoff is rejected. This prevents a closing-line benchmark or blend weight from leaking information backward into its own evaluation.

The completed 2023–2025 benchmark matched all 723 eligible games with zero timestamp or spread-sign violations. Independent margin MAE was 10.383, 30-minute consensus MAE was 9.618, and the walk-forward blend was 9.648. Both future-production weights resolved to 100% market. The largest model/market disagreements did not reverse the conclusion: at seven or more points the independent side was 31–35 ATS before vig and its margin MAE was 12.43 versus 8.86 for consensus.

## Evaluation

All model evaluation is expanding walk-forward. A week's validation rows are predicted only by models trained on earlier seasons or earlier weeks. The manifest is authoritative for the exact metrics of the checked-in bundle.

Required release checks:

- leakage, schema, season-transition, zero-outcome, market-sign, injury-consistency, ledger, and artifact tests pass;
- each player target is selected against its declared rolling baseline and falls back to that
  baseline when added complexity does not improve chronological validation;
- latest-season holdout metrics are reviewed separately;
- interval coverage and game winner calibration remain visible;
- Streamlit starts and loads the checksummed bundle without fallback values.
- market comparisons use a snapshot known at prediction time, before kickoff, with the home-spread sign normalized once;
- independent-model and market errors are scored on the identical game subset.

## Known limitations

- The game model does not yet consume timestamped weather, travel distance, confirmed current-week quarterback announcements, or coaching continuity. Its roster signal observes whether the prior primary quarterback remains on the opening roster, not whether that player is guaranteed to start.
- The added quarterback continuity statistic observes only prior primary passers. It does not assume that a retrospective starter field or an un-timestamped depth chart was known before kickoff.
- Live market history is collected separately but is not yet a trained input. A useful market residual/blend model requires multiple seasons of paid historical snapshots and walk-forward proof.
- Current preseason player membership uses nflverse's seasonal roster until the new weekly roster feed becomes available. Offseason rosters remain fluid until final cuts, so the updater must be rerun before Week 1.
- Snap counts do not measure routes or pass-block/run-block assignments; they are participation signals.
- Residual distributions are simplified and can understate correlated injury or quarterback uncertainty.
- Consensus medians can hide book-specific limits, stale books, price differences, and line availability.
- Small apparent betting edges are dominated by model error and sportsbook vig. The UI remains paper-only.

## Monitoring and retraining

The scheduled Tuesday workflow refreshes data and models, freezes the next available week, scores previous immutable batches, runs tests, and opens a draft pull request. Human review of freshness, metrics, injuries, and model hash is required before merge and deployment.
