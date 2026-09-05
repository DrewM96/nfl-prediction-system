from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

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
from .io import archive_manifest, atomic_write_json, read_json, sha256_file
from .validation import paired_week_block_interval, prequential_predictions, uncertainty_metrics

GAME_RIDGE_ALPHA = 50.0


def _candidate_models(*, ridge_alpha: float = 10.0) -> list[Any]:
    return [
        Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=ridge_alpha))]),
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


def _clone_candidates(
    candidates: list[Any] | None = None, *, ridge_alpha: float = 10.0
) -> list[Any]:
    from sklearn.base import clone

    return [clone(model) for model in (candidates or _candidate_models(ridge_alpha=ridge_alpha))]


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
    candidate_models: list[Any] | None = None,
    ridge_alpha: float = 10.0,
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
        candidates = _clone_candidates(candidate_models, ridge_alpha=ridge_alpha)
        if len(candidates) != 2:
            raise ValueError("chronological validation requires exactly two candidate models")
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
    ridge_alpha: float = 10.0,
) -> FittedEnsemble:
    needed = feature_names + [target_name, "season", "week"]
    clean = frame.dropna(subset=needed).copy()
    if len(clean) < min_train_rows:
        raise ValueError(
            f"{name} requires at least {min_train_rows} clean rows; found {len(clean)}"
        )

    actual, first, second, validation_indices = chronological_oof_predictions(
        clean, feature_names, target_name, min_train_rows=min_train_rows, ridge_alpha=ridge_alpha
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
    # The globally learned weights above are for future production only.
    # Evaluate each week using weights selected before that week.
    validation_frame = clean.loc[validation_indices]
    ensemble_oof = prequential_predictions(
        validation_frame,
        actual,
        first,
        second,
        baseline_values if baseline_feature is not None else None,
    )
    residuals = actual - ensemble_oof
    residual_std = max(float(np.std(residuals, ddof=1)), 1e-6)
    metrics = {
        "ridge_alpha": float(ridge_alpha),
        "oof_rows": float(len(actual)),
        "mae": float(mean_absolute_error(actual, ensemble_oof)),
        "rmse": float(mean_squared_error(actual, ensemble_oof) ** 0.5),
        "bias": float(np.mean(ensemble_oof - actual)),
        "prequential_evaluation": 1.0,
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
    metrics.update(
        uncertainty_metrics(validation_frame, actual, ensemble_oof, winner=name == "game_margin")
    )
    if baseline_values is not None:
        metrics["baseline_mae"] = float(mean_absolute_error(actual, baseline_values))
        metrics["mae_improvement_vs_baseline"] = metrics["baseline_mae"] - metrics["mae"]
        interval = paired_week_block_interval(
            validation_frame, actual, ensemble_oof, baseline_values
        )
        if interval is not None:
            metrics["mae_difference_vs_baseline_95_low"] = interval[0]
            metrics["mae_difference_vs_baseline_95_high"] = interval[1]
        metrics["latest_holdout_baseline_mae"] = float(
            mean_absolute_error(actual[latest_mask], baseline_values[latest_mask])
        )

    models = _clone_candidates(ridge_alpha=ridge_alpha)
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
    release_id = uuid4().hex
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
            filename = f"v3_{ensemble_name}_{index}_{release_id}.joblib"
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
    archive_manifest(root / "manifest.json")
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
