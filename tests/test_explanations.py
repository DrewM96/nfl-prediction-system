from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nfl_prediction.explanations import explain_forecast, explain_model
from nfl_prediction.explanations_ui import model_inputs_html
from nfl_prediction.modeling import FeatureBaselineRegressor, FittedEnsemble


@pytest.fixture
def fitted():
    rng = np.random.default_rng(42)
    names = ["home_off_epa_l4", "away_off_epa_l4", "home_field"]
    frame = pd.DataFrame(rng.normal(size=(100, 3)), columns=names)
    outcome = frame.iloc[:, 0] * 3 - frame.iloc[:, 1] + np.square(frame.iloc[:, 2])
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=5)).fit(frame, outcome)
    tree = GradientBoostingRegressor(n_estimators=12, max_depth=2, random_state=7).fit(
        frame, outcome
    )
    model = FittedEnsemble("game_margin", names, "home_margin", [ridge, tree], [0.3, 0.7], 12, {})
    return frame, outcome, ridge, tree, model


@pytest.mark.parametrize("weights", [[0.3, 0.7], [0, 1], [1, 0], [3, 7]])
def test_weighted_decomposition_reproduces_predictions(fitted, weights):
    frame, _, _, _, model = fitted
    model.weights = weights
    for _, row in frame.iloc[[0, 25, 99]].iterrows():
        result = explain_model(model, row.to_dict())
        assert result["reference"] + sum(result["contributions"].values()) == pytest.approx(
            model.predict(row.to_frame().T)[0], abs=1e-8
        )
        assert list(result["contributions"]) == model.feature_names


def test_cfb_ridge_is_centered_on_training_means(fitted):
    frame, _, ridge, _, _ = fitted
    model = SimpleNamespace(feature_names=list(frame), estimator=ridge, predict=ridge.predict)
    result = explain_model(model, frame.iloc[0].to_dict())
    assert result["reference"] == pytest.approx(ridge[-1].intercept_)
    assert result["reference"] + sum(result["contributions"].values()) == pytest.approx(
        ridge.predict(frame.iloc[[0]])[0]
    )
    at_mean = explain_model(model, dict(zip(frame.columns, ridge[0].mean_, strict=True)))
    assert sum(abs(value) for value in at_mean["contributions"].values()) < 1e-10


def test_tree_paths_follow_float32_threshold_routing(fitted):
    frame, _, _, tree, model = fitted
    model.models, model.weights = [tree], [1]
    split = tree.estimators_[0, 0].tree_
    row = frame.iloc[0].to_dict()
    row[frame.columns[split.feature[0]]] = float(split.threshold[0]) + 1e-12
    result = explain_model(model, row)
    assert result["reference"] + sum(result["contributions"].values()) == pytest.approx(
        result["prediction"], abs=1e-8
    )


def test_feature_baseline_component_reconciles(fitted):
    frame, _, _, _, model = fitted
    model.models, model.weights = [FeatureBaselineRegressor(0)], [1]
    result = explain_model(model, frame.iloc[0].to_dict())
    assert result["reference"] == 0
    assert result["contributions"][model.feature_names[0]] == result["prediction"]


def forecast(fitted):
    frame, _, _, _, model = fitted
    features = frame.iloc[0].to_dict()
    predicted = float(model.predict(frame.iloc[[0]])[0])
    return {
        "home_team": "ATL",
        "away_team": "SF",
        "predicted_home_margin": round(predicted, 2),
        "total": round(predicted, 1),
        "features": features,
        "forecast_at": "2026-10-20T16:56:19+00:00",
        "data_cutoff": "2026-10-19",
    }, model


@pytest.mark.parametrize("weight", [0.25, 0.75, 1])
def test_preseason_calibration_reconciles_to_displayed_margin(fitted, weight):
    game, model = forecast(fitted)
    football_margin = float(model.predict(pd.DataFrame([game["features"]]))[0])
    game["football_only"] = {"home_margin": round(football_margin, 2), "total": game["total"]}
    game["preseason_calibration"] = {"weight": weight, "prior_home_margin": -5}
    game["predicted_home_margin"] = round((1 - weight) * football_margin + weight * -5, 2)
    original = deepcopy(game)
    result = explain_forecast(game, {"game_margin": model, "game_total": model})
    margin = result["model_explanations"]["targets"]["margin"]
    total = result["model_explanations"]["targets"]["total"]
    assert margin["reference"] + sum(margin["contributions"].values()) + sum(
        item["value"] for item in margin["adjustments"]
    ) == pytest.approx(game["predicted_home_margin"], abs=0.006)
    assert margin["adjustments"][0]["weight"] == weight
    assert not total["adjustments"]
    assert game == original
    assert "Preseason strength calibration" in model_inputs_html(result, "nfl", "test")


def test_mismatched_or_missing_inputs_never_get_fabricated_contributions(fitted):
    game, model = forecast(fitted)
    game["predicted_home_margin"] += 10
    result = explain_forecast(game, {"game_margin": model, "game_total": model})
    assert "margin" not in result["model_explanations"]["targets"]
    assert "total" in result["model_explanations"]["targets"]
    game["features"] = {}
    result = explain_forecast(game, {"game_margin": model})
    assert not result["model_explanations"]["targets"]
    assert "Input snapshot unavailable" in model_inputs_html(result, "nfl", "old")


def test_target_schema_excludes_inputs_used_only_by_other_model(fitted):
    game, model = forecast(fitted)
    game["features"]["home_roster_qb_returning_delta"] = 0.25
    result = explain_forecast(game, {"game_margin": model})
    markup = model_inputs_html(result, "nfl", "test")
    assert "Returning QB continuity" not in markup
    assert "Show all 3 model inputs" in markup


def test_missing_total_feature_does_not_hide_margin_explanation(fitted):
    game, model = forecast(fitted)
    total = SimpleNamespace(feature_names=["missing"], estimator=None)
    result = explain_forecast(game, {"margin": model, "total": total})
    assert set(result["model_explanations"]["targets"]) == {"margin"}


def test_panel_escapes_values_and_has_isolated_collapsed_native_controls(fitted):
    game, model = forecast(fitted)
    game["home_team"] = '<script>alert("x")</script>'
    game["input_quality"] = {"warnings": ['Missing <img src="x"> data']}
    result = explain_forecast(game, {"margin": model, "total": model})
    markup = model_inputs_html(result, "nfl", "card")
    featured = model_inputs_html(result, "nfl", "featured")
    assert "<script>" not in markup and "<img " not in markup
    assert "&lt;script&gt;" in markup and "&lt;img " in markup
    assert '<details class="grid-inputs">' in markup
    assert " open" not in markup
    assert 'type="radio"' in markup and 'value="margin" checked' in markup
    assert "Oct 20, 12:56 PM ET" in markup
    import re

    assert set(re.findall(r'id="([^"]+)"', markup)).isdisjoint(
        re.findall(r'id="([^"]+)"', featured)
    )


def test_changed_forecast_hides_old_terms_but_keeps_saved_inputs(fitted):
    game, model = forecast(fitted)
    result = explain_forecast(game, {"margin": model})
    result["predicted_home_margin"] += 4
    markup = model_inputs_html(result, "nfl", "adjusted")
    assert "Contribution breakdown unavailable" in markup
    assert "Final predicted margin" not in markup
    assert "Show all 3 model inputs" in markup


def test_no_saved_snapshot_does_not_substitute_current_values():
    markup = model_inputs_html({"home_team": "ATL", "away_team": "SF"}, "cfb", "old")
    assert "Input snapshot unavailable" in markup
    assert "Contribution breakdown unavailable" in markup


def test_invalid_input_or_weights_fail_closed(fitted):
    frame, _, _, _, model = fitted
    features = frame.iloc[0].to_dict()
    features[model.feature_names[0]] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        explain_model(model, features)
    model.weights = [-1, 2]
    with pytest.raises(ValueError, match="weights"):
        explain_model(model, frame.iloc[0].to_dict())
