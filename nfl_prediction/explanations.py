"""Additive, per-forecast explanations for the production game models.

Ridge terms are centered on the fitted scaler's training means. Boosted trees
attribute each change along the observed decision path to its splitting feature.
These are model decompositions, not causal effects or SHAP estimates. The weighted
reference plus all feature terms must reproduce the original model prediction.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .modeling import FeatureBaselineRegressor


def _component_terms(model: Any, row: pd.DataFrame) -> tuple[float, np.ndarray]:
    values = row.to_numpy(dtype=float)[0]
    if isinstance(model, Pipeline):
        if (
            len(model.steps) != 2
            or not isinstance(model.steps[0][1], StandardScaler)
            or not isinstance(model.steps[1][1], Ridge)
        ):
            raise ValueError("Unsupported explanation pipeline")
        scaler, ridge = model.steps[0][1], model.steps[1][1]
        transformed = scaler.transform(row)[0]
        return float(ridge.intercept_), transformed * ridge.coef_
    if isinstance(model, FeatureBaselineRegressor):
        terms = np.zeros(len(values))
        terms[model.feature_index] = values[model.feature_index]
        return 0.0, terms
    if isinstance(model, GradientBoostingRegressor):
        # sklearn trees route float32 inputs. Use the identical representation at
        # threshold boundaries, rather than a hand traversal over float64 values.
        tree_values = values.astype(np.float32)
        baseline = 0.0 if model.init == "zero" else float(model.init_.predict(row.to_numpy())[0])
        terms = np.zeros(len(values))
        for estimator in model.estimators_[:, 0]:
            tree = estimator.tree_
            baseline += model.learning_rate * float(tree.value[0, 0, 0])
            node = 0
            while tree.children_left[node] != tree.children_right[node]:
                feature = tree.feature[node]
                child = (
                    tree.children_left[node]
                    if tree_values[feature] <= tree.threshold[node]
                    else tree.children_right[node]
                )
                terms[feature] += model.learning_rate * float(
                    tree.value[child, 0, 0] - tree.value[node, 0, 0]
                )
                node = child
        return baseline, terms
    raise ValueError(f"Unsupported explanation model: {type(model).__name__}")


def explain_model(model: Any, features: dict[str, float]) -> dict[str, Any]:
    """Explain one fitted NFL ensemble or CFB ridge using its ordered schema."""
    names = list(model.feature_names)
    row = pd.DataFrame([{name: float(features[name]) for name in names}], columns=names)
    if not np.isfinite(row.to_numpy()).all():
        raise ValueError("Nonfinite model inputs")
    if hasattr(model, "models"):
        components = model.models
        weights = np.asarray(model.weights, dtype=float)
    else:
        components, weights = [model.estimator], np.ones(1)
    if (
        len(components) != len(weights)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() <= 0
    ):
        raise ValueError("Invalid explanation ensemble weights")
    weights = weights / weights.sum()
    baseline = 0.0
    terms = np.zeros(len(names))
    for component, weight in zip(components, weights, strict=True):
        if weight == 0:
            continue
        reference, contribution = _component_terms(component, row)
        baseline += float(weight) * reference
        terms += float(weight) * contribution
    prediction = float(model.predict(row)[0])
    if not np.isfinite([baseline, prediction, *terms]).all() or not math.isclose(
        baseline + float(terms.sum()), prediction, rel_tol=1e-9, abs_tol=1e-7
    ):
        raise ValueError("Model explanation does not reconcile")
    return {
        "feature_names": names,
        "reference": baseline,
        "contributions": dict(zip(names, terms.tolist(), strict=True)),
        "prediction": prediction,
        "method": "Centered ridge terms and weighted tree decision-path changes"
        if hasattr(model, "models")
        else "Ridge terms centered on the fitted training means",
    }


def explain_forecast(game: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
    """Keep only decompositions that reproduce this forecast's saved outputs.

    Never rebuild features from today's team state. Independently validate the two
    targets so a missing total input does not hide an available margin explanation.
    """
    result = dict(game)
    features = game.get("features") or {}
    explanations: dict[str, Any] = {}
    schemas: dict[str, list[str]] = {}
    for target, names in (
        ("margin", ("game_margin", "margin")),
        ("total", ("game_total", "total")),
    ):
        model = next((models[name] for name in names if name in models), None)
        if model is None:
            continue
        try:
            schemas[target] = list(model.feature_names)
            explanation = explain_model(model, features)
            football = game.get("football_only") or {}
            saved = (
                football.get("home_margin", game.get("predicted_home_margin"))
                if target == "margin"
                else football.get("total", game.get("total", game.get("predicted_total")))
            )
            # NFL published margins/totals are rounded to 0.01/0.1 points.
            tolerance = 0.006 if target == "margin" else 0.051
            if saved is None or not math.isclose(
                explanation["prediction"], float(saved), rel_tol=0, abs_tol=tolerance
            ):
                continue
            explanation["model_prediction"] = explanation["prediction"]
            explanation["adjustments"] = []
            calibration = game.get("preseason_calibration") or {}
            if target == "margin" and float(calibration.get("weight", 0.0)):
                weight = float(calibration["weight"])
                if not 0 <= weight <= 1:
                    continue
                delta = weight * (
                    float(calibration["prior_home_margin"]) - explanation["prediction"]
                )
                explanation["adjustments"].append(
                    {"label": "Preseason strength calibration", "value": delta, "weight": weight}
                )
                explanation["prediction"] += delta
            published = (
                game.get("predicted_home_margin")
                if target == "margin"
                else game.get("total", game.get("predicted_total"))
            )
            if published is None or not math.isclose(
                explanation["prediction"], float(published), rel_tol=0, abs_tol=tolerance
            ):
                continue
            explanations[target] = explanation
        except (KeyError, ValueError, TypeError, AttributeError, IndexError):
            # Legacy/custom bundles can still expose their frozen inputs without
            # pretending a different model explains their published prediction.
            continue
    result["model_explanations"] = {"schema_version": 1, "targets": explanations}
    result["model_input_schema"] = schemas
    return result


def explain_forecasts(games: list[dict[str, Any]], models: dict[str, Any]) -> list[dict[str, Any]]:
    return [explain_forecast(game, models) for game in games]
