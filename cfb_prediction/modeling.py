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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nfl_prediction.io import atomic_write_json, read_json, sha256_file

from .config import CFB_MODEL_MANIFEST_PATH, CFB_MODELS_DIR


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def chronological_ridge_predictions(
    games: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    min_train_rows: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean = games.dropna(subset=[*features, target, "season", "week"]).copy()
    actual: list[float] = []
    predicted: list[float] = []
    indices: list[int] = []
    groups = clean[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
    for _, group in groups.iterrows():
        before = (clean["season"] < group["season"]) | (
            clean["season"].eq(group["season"]) & clean["week"].lt(group["week"])
        )
        validate = clean["season"].eq(group["season"]) & clean["week"].eq(group["week"])
        train = clean[before]
        validation = clean[validate]
        if len(train) < min_train_rows or validation.empty:
            continue
        model = make_ridge(alpha)
        model.fit(train[features].astype(float), train[target].astype(float))
        predicted.extend(model.predict(validation[features].astype(float)).tolist())
        actual.extend(validation[target].astype(float).tolist())
        indices.extend(validation.index.astype(int).tolist())
    return np.asarray(actual), np.asarray(predicted), np.asarray(indices)


def make_ridge(alpha: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


@dataclass
class CFBFittedModel:
    name: str
    feature_names: list[str]
    target_name: str
    estimator: Pipeline
    residual_std: float
    metrics: dict[str, float]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        missing = sorted(set(self.feature_names) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing features for {self.name}: {missing}")
        return self.estimator.predict(frame[self.feature_names].astype(float))

    def distribution(self, frame: pd.DataFrame) -> list[dict[str, float]]:
        standard_deviation = max(float(self.residual_std), 1e-6)
        return [
            {
                "mean": float(mean),
                "std": standard_deviation,
                "p10": float(mean - 1.2816 * standard_deviation),
                "p90": float(mean + 1.2816 * standard_deviation),
                "probability_above_zero": _normal_cdf(float(mean) / standard_deviation),
            }
            for mean in self.predict(frame)
        ]


def fit_cfb_model(
    games: pd.DataFrame,
    *,
    name: str,
    feature_names: list[str],
    target_name: str,
    min_train_rows: int = 1200,
    alpha: float = 50.0,
) -> CFBFittedModel:
    needed = [*feature_names, target_name, "season", "week"]
    clean = games.dropna(subset=needed).copy()
    if len(clean) < min_train_rows:
        raise ValueError(f"{name} requires at least {min_train_rows} rows; found {len(clean)}")

    actual, predicted, indices = chronological_ridge_predictions(
        clean,
        feature_names,
        target_name,
        min_train_rows=min_train_rows,
        alpha=alpha,
    )
    if not len(actual):
        raise ValueError(f"{name} could not produce chronological validation predictions")
    residuals = actual - predicted
    residual_std = max(float(np.std(residuals, ddof=1)), 1e-6)
    validation = clean.loc[indices]
    latest_season = int(validation["season"].max())
    latest = validation["season"].astype(int).to_numpy() == latest_season
    metrics = {
        "ridge_alpha": float(alpha),
        "training_rows": float(len(clean)),
        "oof_rows": float(len(actual)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        "bias": float(np.mean(predicted - actual)),
        "residual_std": residual_std,
        "interval_80_coverage": float(
            np.mean(
                (actual >= predicted - 1.2816 * residual_std)
                & (actual <= predicted + 1.2816 * residual_std)
            )
        ),
        "latest_holdout_season": float(latest_season),
        "latest_holdout_rows": float(latest.sum()),
        "latest_holdout_mae": float(mean_absolute_error(actual[latest], predicted[latest])),
    }
    estimator = make_ridge(alpha)
    estimator.fit(clean[feature_names].astype(float), clean[target_name].astype(float))
    return CFBFittedModel(
        name=name,
        feature_names=list(feature_names),
        target_name=target_name,
        estimator=estimator,
        residual_std=residual_std,
        metrics=metrics,
    )


def save_cfb_model_bundle(
    models: dict[str, CFBFittedModel],
    *,
    prediction_season: int,
    training_seasons: list[int],
    data_cutoff: str,
    selected_configurations: dict[str, str],
    input_coverage: dict[str, int],
    benchmark_path: str | Path,
    git_commit: str | None = None,
    output_dir: str | Path = CFB_MODELS_DIR,
) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "sport": "college_football",
        "created_at": datetime.now(UTC).isoformat(),
        "prediction_season": int(prediction_season),
        "training_seasons": [int(season) for season in training_seasons],
        "data_cutoff": data_cutoff,
        "selected_configurations": selected_configurations,
        "input_coverage": {key: int(value) for key, value in input_coverage.items()},
        "benchmark_sha256": sha256_file(benchmark_path),
        "git_commit": git_commit,
        "libraries": {"scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "models": {},
    }
    for name, model in models.items():
        filename = f"v1_{name}.joblib"
        target = root / filename
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{filename}.", dir=root)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            joblib.dump(model.estimator, temporary)
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        manifest["models"][name] = {
            "target": model.target_name,
            "features": model.feature_names,
            "residual_std": model.residual_std,
            "metrics": model.metrics,
            "file": {"path": filename, "sha256": sha256_file(target)},
        }
    atomic_write_json(root / "manifest.json", manifest)
    return manifest


def load_cfb_model_bundle(
    manifest_path: str | Path = CFB_MODEL_MANIFEST_PATH,
) -> tuple[dict[str, CFBFittedModel], dict[str, Any]]:
    path = Path(manifest_path)
    manifest = read_json(path)
    if not manifest or manifest.get("schema_version") != 1:
        raise ValueError("A schema-v1 CFB model manifest is required")
    if manifest.get("sport") != "college_football":
        raise ValueError("The model manifest is not a College Football bundle")
    models: dict[str, CFBFittedModel] = {}
    for name, specification in manifest.get("models", {}).items():
        model_file = specification["file"]
        model_path = path.parent / model_file["path"]
        if sha256_file(model_path) != model_file["sha256"]:
            raise ValueError(f"Checksum mismatch for {model_path.name}")
        models[name] = CFBFittedModel(
            name=name,
            feature_names=list(specification["features"]),
            target_name=str(specification["target"]),
            estimator=joblib.load(model_path),
            residual_std=float(specification["residual_std"]),
            metrics={key: float(value) for key, value in specification["metrics"].items()},
        )
    if set(models) != {"margin", "total"}:
        raise ValueError("The CFB bundle must contain margin and total models")
    return models, manifest
