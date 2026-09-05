"""Chronological meta-model and uncertainty evaluation shared by both sports."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def prior_week_masks(frame: pd.DataFrame):
    seasons = frame["season"].to_numpy()
    weeks = frame["week"].to_numpy()
    for season, week in sorted(set(zip(seasons, weeks, strict=True))):
        yield (
            (seasons < season) | ((seasons == season) & (weeks < week)),
            ((seasons == season) & (weeks == week)),
        )


def component_weight(actual: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    difference = first - second
    denominator = float(difference @ difference)
    return (
        float(np.clip(difference @ (actual - second) / denominator, 0, 1))
        if denominator > 1e-12
        else 0.5
    )


def baseline_blend_weight(actual, predicted, baseline) -> float:
    # Prefer the transparent baseline if several weights give exactly the same error.
    return float(
        min(
            np.linspace(0, 1, 101),
            key=lambda w: np.mean(np.abs(actual - (w * predicted + (1 - w) * baseline))),
        )
    )


def prequential_predictions(frame, actual, first, second, baseline=None):
    components = np.empty(len(actual))
    output = np.empty(len(actual))
    for prior, current in prior_week_masks(frame):
        weight = (
            component_weight(actual[prior], first[prior], second[prior]) if prior.any() else 0.5
        )
        components[current] = weight * first[current] + (1 - weight) * second[current]
        if baseline is None:
            output[current] = components[current]
        else:
            blend = (
                baseline_blend_weight(actual[prior], components[prior], baseline[prior])
                if prior.any()
                else 0.0
            )
            output[current] = blend * components[current] + (1 - blend) * baseline[current]
    return output


def prior_residual_scales(frame, residuals):
    scales = np.full(len(residuals), np.nan)
    for prior, current in prior_week_masks(frame):
        if prior.sum() >= 2:
            scales[current] = max(float(np.std(residuals[prior], ddof=1)), 1e-6)
    return scales


def paired_week_block_interval(
    frame: pd.DataFrame,
    actual: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    draws: int = 2000,
) -> tuple[float, float] | None:
    """Bootstrap paired absolute-error differences by season-week blocks."""
    differences = np.abs(candidate - actual) - np.abs(reference - actual)
    blocks = (
        frame.assign(_difference=differences)
        .groupby(["season", "week"])["_difference"]
        .agg(["sum", "count"])
    )
    if len(blocks) < 4:
        return None
    samples = np.random.default_rng(42).integers(0, len(blocks), size=(draws, len(blocks)))
    estimates = blocks["sum"].to_numpy()[samples].sum(axis=1) / blocks["count"].to_numpy()[
        samples
    ].sum(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def uncertainty_metrics(frame, actual, predicted, *, winner=False):
    scales = prior_residual_scales(frame, actual - predicted)
    valid = np.isfinite(scales)
    result = {"uncertainty_oof_rows": float(valid.sum())}
    if not valid.any():
        return result
    y, mu, sigma = actual[valid], predicted[valid], scales[valid]
    low, high = mu - 1.2816 * sigma, mu + 1.2816 * sigma
    result.update(
        interval_80_coverage=float(np.mean((y >= low) & (y <= high))),
        interval_80_mean_width=float(np.mean(high - low)),
        pinball_p10=float(np.mean(np.maximum(0.1 * (y - low), -0.9 * (y - low)))),
        pinball_p90=float(np.mean(np.maximum(0.9 * (y - high), -0.1 * (y - high)))),
    )
    if winner:
        decisive = y != 0
        result["winner_oof_rows"] = float(decisive.sum())
        if decisive.any():
            probabilities = np.array(
                [0.5 * (1 + math.erf(v / math.sqrt(2))) for v in (mu / sigma)[decisive]]
            )
            outcomes = (y[decisive] > 0).astype(float)
            clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
            ece = 0.0
            for i in range(10):
                mask = (probabilities >= i / 10) & (
                    (probabilities < (i + 1) / 10) if i < 9 else probabilities <= 1
                )
                if mask.any():
                    ece += mask.mean() * abs(probabilities[mask].mean() - outcomes[mask].mean())
            result.update(
                winner_brier=float(np.mean((probabilities - outcomes) ** 2)),
                winner_log_loss=float(
                    -np.mean(outcomes * np.log(clipped) + (1 - outcomes) * np.log(1 - clipped))
                ),
                winner_calibration_error=float(ece),
            )
    return result
