from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODEL_MANIFEST_PATH, MODELS_DIR
from .io import atomic_write_json, read_json, sha256_file


def _candidate_models() -> list[Any]:
    return [
        Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))]),
        GradientBoostingRegressor(
            n_estimators=150,
            max_depth=2,
            learning_rate=0.035,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
            loss="huber",
        ),
    ]


def _clone_candidates() -> list[Any]:
    from sklearn.base import clone

    return [clone(model) for model in _candidate_models()]


def _learn_two_model_weight(y: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    difference = first - second
    denominator = float(np.dot(difference, difference))
    if denominator <= 1e-12:
        return 0.5
    return float(np.clip(np.dot(difference, y - second) / denominator, 0.0, 1.0))


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _expected_calibration_error(
    probabilities: np.ndarray, outcomes: np.ndarray, bins: int = 10
) -> float:
    error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if mask.any():
            error += float(mask.mean()) * abs(
                float(probabilities[mask].mean()) - float(outcomes[mask].mean())
            )
    return error


def chronological_oof_predictions(
    frame: pd.DataFrame,
    feature_names: list[str],
    target_name: str,
    *,
    min_train_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ordered = frame.sort_values(
        ["season", "week", "gameday" if "gameday" in frame else "game_date"]
    )
    keys = ordered[["season", "week"]].drop_duplicates().itertuples(index=False, name=None)
    actual: list[float] = []
    first_predictions: list[float] = []
    second_predictions: list[float] = []
    row_indices: list[int] = []

    for season, week in keys:
        train = ordered[
            (ordered["season"] < season)
            | ((ordered["season"] == season) & (ordered["week"] < week))
        ]
        validate = ordered[(ordered["season"] == season) & (ordered["week"] == week)]
        if len(train) < min_train_rows or validate.empty:
            continue
        candidates = _clone_candidates()
        X_train = train[feature_names].astype(float)
        y_train = train[target_name].astype(float)
        X_validate = validate[feature_names].astype(float)
        for model in candidates:
            model.fit(X_train, y_train)
        predictions = [model.predict(X_validate) for model in candidates]
        actual.extend(validate[target_name].astype(float).tolist())
        first_predictions.extend(predictions[0].tolist())
        second_predictions.extend(predictions[1].tolist())
        row_indices.extend(validate.index.tolist())

    return (
        np.asarray(actual),
        np.asarray(first_predictions),
        np.asarray(second_predictions),
        np.asarray(row_indices, dtype=int),
    )


@dataclass
class FittedEnsemble:
    name: str
    feature_names: list[str]
    target_name: str
    models: list[Any]
    weights: list[float]
    residual_std: float
    metrics: dict[str, float]

    def predict(self, frame: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(frame, pd.DataFrame):
            missing = sorted(set(self.feature_names) - set(frame.columns))
            if missing:
                raise ValueError(f"Missing features for {self.name}: {missing}")
            values = frame[self.feature_names].astype(float)
        else:
            values = np.asarray(frame, dtype=float)
            if values.ndim == 1:
                values = values.reshape(1, -1)
            if values.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"{self.name} expects {len(self.feature_names)} features, received {values.shape[1]}"
                )
        components = [model.predict(values) for model in self.models]
        return np.average(np.vstack(components), axis=0, weights=self.weights)

    def distribution(self, frame: pd.DataFrame | np.ndarray) -> list[dict[str, float]]:
        means = self.predict(frame)
        std = max(float(self.residual_std), 1e-6)
        return [
            {
                "mean": float(mean),
                "std": std,
                "p10": float(mean - 1.2816 * std),
                "p90": float(mean + 1.2816 * std),
                "probability_above_zero": float(_normal_cdf(float(mean) / std)),
            }
            for mean in means
        ]


@dataclass
class FeatureBaselineRegressor:
    feature_index: int

    def fit(self, features: Any, target: Any = None) -> FeatureBaselineRegressor:
        del features, target
        return self

    def predict(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        values = features.to_numpy() if isinstance(features, pd.DataFrame) else np.asarray(features)
        return np.asarray(values[:, self.feature_index], dtype=float)


def fit_ensemble(
    frame: pd.DataFrame,
    *,
    name: str,
    feature_names: list[str],
    target_name: str,
    baseline: pd.Series | np.ndarray | None = None,
    baseline_feature: str | None = None,
    min_train_rows: int = 200,
) -> FittedEnsemble:
    needed = feature_names + [target_name, "season", "week"]
    clean = frame.dropna(subset=needed).copy()
    if len(clean) < min_train_rows:
        raise ValueError(
            f"{name} requires at least {min_train_rows} clean rows; found {len(clean)}"
        )

    actual, first, second, validation_indices = chronological_oof_predictions(
        clean, feature_names, target_name, min_train_rows=min_train_rows
    )
    if not len(actual):
        raise ValueError(f"{name} could not produce chronological validation predictions")
    first_weight = _learn_two_model_weight(actual, first, second)
    weights = [first_weight, 1.0 - first_weight]
    ensemble_oof = first * weights[0] + second * weights[1]
    baseline_values: np.ndarray | None = None
    if baseline is not None:
        baseline_values = (
            pd.Series(baseline, index=frame.index).loc[validation_indices].astype(float).to_numpy()
        )
    baseline_weight = 0.0
    if baseline_feature is not None:
        if baseline_feature not in feature_names or baseline_values is None:
            raise ValueError("baseline_feature requires a named feature and baseline values")
        blend_candidates = np.linspace(0.0, 1.0, 101)
        blend_weight = min(
            blend_candidates,
            key=lambda weight: mean_absolute_error(
                actual, weight * ensemble_oof + (1.0 - weight) * baseline_values
            ),
        )
        ensemble_oof = blend_weight * ensemble_oof + (1.0 - blend_weight) * baseline_values
        weights = [weight * float(blend_weight) for weight in weights]
        baseline_weight = 1.0 - float(blend_weight)
    residuals = actual - ensemble_oof
    residual_std = max(float(np.std(residuals, ddof=1)), 1e-6)
    interval_low = ensemble_oof - 1.2816 * residual_std
    interval_high = ensemble_oof + 1.2816 * residual_std
    metrics = {
        "oof_rows": float(len(actual)),
        "mae": float(mean_absolute_error(actual, ensemble_oof)),
        "rmse": float(mean_squared_error(actual, ensemble_oof) ** 0.5),
        "bias": float(np.mean(ensemble_oof - actual)),
        "interval_80_coverage": float(
            np.mean((actual >= interval_low) & (actual <= interval_high))
        ),
        "pinball_p10": float(
            np.mean(np.maximum(0.10 * (actual - interval_low), -0.90 * (actual - interval_low)))
        ),
        "pinball_p90": float(
            np.mean(np.maximum(0.90 * (actual - interval_high), -0.10 * (actual - interval_high)))
        ),
    }
    validation_frame = clean.loc[validation_indices]
    latest_season = int(validation_frame["season"].max())
    latest_mask = validation_frame["season"].to_numpy() == latest_season
    metrics.update(
        {
            "latest_holdout_season": float(latest_season),
            "latest_holdout_rows": float(latest_mask.sum()),
            "latest_holdout_mae": float(
                mean_absolute_error(actual[latest_mask], ensemble_oof[latest_mask])
            ),
        }
    )
    if name == "game_margin":
        probabilities = np.asarray(
            [_normal_cdf(float(value) / residual_std) for value in ensemble_oof]
        )
        outcomes = (actual > 0).astype(float)
        clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
        metrics.update(
            {
                "winner_brier": float(np.mean((probabilities - outcomes) ** 2)),
                "winner_log_loss": float(
                    -np.mean(outcomes * np.log(clipped) + (1.0 - outcomes) * np.log(1.0 - clipped))
                ),
                "winner_calibration_error": _expected_calibration_error(probabilities, outcomes),
            }
        )
    if baseline_values is not None:
        metrics["baseline_mae"] = float(mean_absolute_error(actual, baseline_values))
        metrics["mae_improvement_vs_baseline"] = metrics["baseline_mae"] - metrics["mae"]
        metrics["latest_holdout_baseline_mae"] = float(
            mean_absolute_error(actual[latest_mask], baseline_values[latest_mask])
        )

    models = _clone_candidates()
    X = clean[feature_names].astype(float)
    y = clean[target_name].astype(float)
    for model in models:
        model.fit(X, y)
    if baseline_feature is not None:
        models.append(FeatureBaselineRegressor(feature_names.index(baseline_feature)))
        weights.append(baseline_weight)
    return FittedEnsemble(
        name=name,
        feature_names=feature_names,
        target_name=target_name,
        models=models,
        weights=weights,
        residual_std=residual_std,
        metrics=metrics,
    )


def save_model_bundle(
    ensembles: dict[str, FittedEnsemble],
    *,
    prediction_season: int,
    training_seasons: list[int],
    data_cutoff: str,
    raw_data_hash: str | None = None,
    git_commit: str | None = None,
    output_dir: str | Path = MODELS_DIR,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 3,
        "created_at": datetime.now(UTC).isoformat(),
        "prediction_season": prediction_season,
        "training_seasons": training_seasons,
        "data_cutoff": data_cutoff,
        "raw_data_hash": raw_data_hash,
        "git_commit": git_commit,
        "libraries": {"scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "models": {},
    }
    for ensemble_name, ensemble in ensembles.items():
        files = []
        for index, (model, weight) in enumerate(
            zip(ensemble.models, ensemble.weights, strict=False)
        ):
            filename = f"v3_{ensemble_name}_{index}.joblib"
            target = root / filename
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{filename}.", dir=root)
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                joblib.dump(model, temporary)
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink()
            files.append({"path": filename, "weight": weight, "sha256": sha256_file(target)})
        manifest["models"][ensemble_name] = {
            "target": ensemble.target_name,
            "features": ensemble.feature_names,
            "residual_std": ensemble.residual_std,
            "metrics": ensemble.metrics,
            "files": files,
        }
    atomic_write_json(root / "manifest.json", manifest)
    return manifest


def load_model_bundle(
    manifest_path: str | Path = MODEL_MANIFEST_PATH,
) -> tuple[dict[str, FittedEnsemble], dict[str, Any]]:
    path = Path(manifest_path)
    manifest = read_json(path)
    if not manifest or manifest.get("schema_version") != 3:
        raise ValueError("A schema-v3 model manifest is required. Run the weekly updater.")
    root = path.parent
    ensembles: dict[str, FittedEnsemble] = {}
    for name, specification in manifest["models"].items():
        models = []
        weights = []
        for model_file in specification["files"]:
            model_path = root / model_file["path"]
            if sha256_file(model_path) != model_file["sha256"]:
                raise ValueError(f"Checksum mismatch for {model_path.name}")
            models.append(joblib.load(model_path))
            weights.append(float(model_file["weight"]))
        ensembles[name] = FittedEnsemble(
            name=name,
            feature_names=list(specification["features"]),
            target_name=str(specification["target"]),
            models=models,
            weights=weights,
            residual_std=float(specification["residual_std"]),
            metrics={key: float(value) for key, value in specification["metrics"].items()},
        )
    return ensembles, manifest
