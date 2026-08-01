import numpy as np
import pandas as pd

from nfl_prediction.modeling import FittedEnsemble, fit_ensemble


class ZeroModel:
    def predict(self, values):
        return np.zeros(len(values))


def test_walk_forward_model_has_schema_and_uncertainty() -> None:
    rows = []
    for season in (2023, 2024):
        for week in range(1, 19):
            for game in range(2):
                feature = (season - 2023) * 20 + week + game
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "gameday": pd.Timestamp(season, 9, 1) + pd.Timedelta(days=week * 7),
                        "feature": float(feature),
                        "target": float(2 * feature + (game - 0.5)),
                    }
                )
    frame = pd.DataFrame(rows)
    model = fit_ensemble(
        frame,
        name="test",
        feature_names=["feature"],
        target_name="target",
        baseline=np.zeros(len(frame)),
        min_train_rows=10,
    )
    prediction = model.distribution(pd.DataFrame([{"feature": 50.0}]))[0]
    assert model.feature_names == ["feature"]
    assert prediction["std"] > 0
    assert prediction["p10"] < prediction["mean"] < prediction["p90"]
    assert 0.0 <= model.metrics["interval_80_coverage"] <= 1.0
    assert model.metrics["latest_holdout_season"] == 2024
    assert model.metrics["ridge_alpha"] == 10.0


def test_ridge_regularization_is_configurable_per_ensemble() -> None:
    rows = []
    for week in range(1, 13):
        for game in range(2):
            rows.append(
                {
                    "season": 2025,
                    "week": week,
                    "gameday": pd.Timestamp(2025, 9, 1) + pd.Timedelta(days=week * 7),
                    "feature": float(week + game),
                    "target": float(week - game),
                }
            )
    model = fit_ensemble(
        pd.DataFrame(rows),
        name="regularized",
        feature_names=["feature"],
        target_name="target",
        min_train_rows=8,
        ridge_alpha=50.0,
    )
    assert model.metrics["ridge_alpha"] == 50.0
    assert model.models[0].named_steps["ridge"].alpha == 50.0


def test_zero_projection_is_a_valid_distribution() -> None:
    model = FittedEnsemble(
        name="zero",
        feature_names=["feature"],
        target_name="target",
        models=[ZeroModel()],
        weights=[1.0],
        residual_std=1.0,
        metrics={},
    )
    result = model.distribution(pd.DataFrame([{"feature": 1.0}]))[0]
    assert result["mean"] == 0.0
    assert result["probability_above_zero"] == 0.5


def test_player_model_can_fall_back_to_a_better_feature_baseline() -> None:
    rows = []
    for week in range(1, 19):
        for player in range(3):
            target = float((week * 7 + player * 13) % 41)
            rows.append(
                {
                    "season": 2025,
                    "week": week,
                    "game_date": pd.Timestamp(2025, 9, 1) + pd.Timedelta(days=week * 7),
                    "noise": float(player),
                    "baseline": target,
                    "target": target,
                }
            )
    frame = pd.DataFrame(rows)
    model = fit_ensemble(
        frame,
        name="baseline-selected",
        feature_names=["noise", "baseline"],
        target_name="target",
        baseline=frame["baseline"],
        baseline_feature="baseline",
        min_train_rows=12,
    )
    assert model.weights[-1] == 1.0
    assert model.metrics["mae_improvement_vs_baseline"] == 0.0
