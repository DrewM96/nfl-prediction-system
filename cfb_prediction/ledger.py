from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from nfl_prediction.io import atomic_write_json, read_json
from nfl_prediction.ledger import PredictionLedger

from .config import CFB_LATEST_PREDICTION_PATH, CFB_PREDICTIONS_DIR


def record_cfb_prediction_batch(
    predictions: Iterable[dict[str, Any]],
    *,
    model_hash: str,
    data_cutoff: str,
    prediction_season: int,
    metadata: dict[str, Any],
    root: str | Path = CFB_PREDICTIONS_DIR,
    latest_path: str | Path = CFB_LATEST_PREDICTION_PATH,
    manifest_path: str | None = None,
    rankings_path: str | None = None,
) -> Path:
    ledger = PredictionLedger(root)
    target = ledger.record_batch(
        predictions,
        model_hash=model_hash,
        data_cutoff=data_cutoff,
        prediction_season=prediction_season,
        metadata=metadata,
    )
    atomic_write_json(
        latest_path,
        {
            "schema_version": 1,
            "run_id": target.stem,
            "path": target.name,
            "model_hash": model_hash,
            "manifest_path": manifest_path,
            "rankings_path": rankings_path,
        },
    )
    return target


def load_latest_cfb_prediction_batch(
    *,
    root: str | Path = CFB_PREDICTIONS_DIR,
    latest_path: str | Path = CFB_LATEST_PREDICTION_PATH,
) -> dict[str, Any] | None:
    pointer = read_json(latest_path)
    if not pointer:
        return None
    filename = str(pointer.get("path", ""))
    if not filename or Path(filename).name != filename:
        raise ValueError("Invalid CFB prediction pointer")
    batch = read_json(Path(root) / filename)
    if not batch or batch.get("run_id") != pointer.get("run_id"):
        raise ValueError("The latest CFB prediction batch is missing or inconsistent")
    if batch.get("model_hash") != pointer.get("model_hash"):
        raise ValueError("The latest CFB prediction model hash is inconsistent")
    return batch
