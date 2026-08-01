# Model Card: NFL Prediction System v3

## Intended use

The system produces pregame analytical forecasts for NFL home margin, total points, win probability, passing yards, receiving yards, receptions, and rushing yards. It is designed for research, model monitoring, and paper-market comparison.

It is not a source of guaranteed outcomes, individualized financial advice, or a substitute for current roster, weather, lineup, and market information.

## Model design

Each target combines a standardized ridge regression with a shallow gradient-boosted regressor. The blend weight is learned from chronological out-of-fold predictions. The final production estimators are fitted on every eligible record through the manifest cutoff.

Game inputs use shrunk rolling team form, EPA/play, pressure rates on a shared rate scale, turnovers, rest, division status, week, and home field. Player inputs use shifted opportunity and efficiency features, real intended targets, stable player IDs, opponent defense, and offensive snap participation when published.

Prediction intervals use the standard deviation of chronological residuals. They describe historical model error, not every source of real-world uncertainty. Injury availability remains a separate scenario until a timestamped report history supports safe training.

## Evaluation

All model evaluation is expanding walk-forward. A week's validation rows are predicted only by models trained on earlier seasons or earlier weeks. The manifest is authoritative for the exact metrics of the checked-in bundle.

Required release checks:

- leakage, schema, season-transition, zero-outcome, market-sign, injury-consistency, ledger, and artifact tests pass;
- each player target is selected against its declared rolling baseline and falls back to that
  baseline when added complexity does not improve chronological validation;
- latest-season holdout metrics are reviewed separately;
- interval coverage and game winner calibration remain visible;
- Streamlit starts and loads the checksummed bundle without fallback values.

## Known limitations

- The game model does not yet consume timestamped weather, travel distance, quarterback starter changes, coaching continuity, or live market history.
- Current preseason player membership falls back to the latest published weekly roster if the new-season roster feed is not available. The UI records the roster season.
- Snap counts do not measure routes or pass-block/run-block assignments; they are participation signals.
- Residual distributions are simplified and can understate correlated injury or quarterback uncertainty.
- Small apparent betting edges are dominated by model error and sportsbook vig. The UI remains paper-only.

## Monitoring and retraining

The scheduled Tuesday workflow refreshes data and models, freezes the next available week, scores previous immutable batches, runs tests, and opens a draft pull request. Human review of freshness, metrics, injuries, and model hash is required before merge and deployment.
