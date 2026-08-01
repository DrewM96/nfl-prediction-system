from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import PREDICTIONS_DIR
from .io import atomic_write_json, read_json


class PredictionLedger:
    """Immutable pregame prediction batches.

    Each run is a separate JSON document. Results are stored separately so
    scoring can never overwrite what the model knew before kickoff.
    """

    def __init__(self, root: str | Path = PREDICTIONS_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def record_batch(
        self,
        predictions: Iterable[dict[str, Any]],
        *,
        model_hash: str,
        data_cutoff: str,
        prediction_season: int,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        created_at = datetime.now(UTC)
        run_id = f"{created_at:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
        payload = {
            "run_id": run_id,
            "created_at": created_at.isoformat(),
            "prediction_season": prediction_season,
            "data_cutoff": data_cutoff,
            "model_hash": model_hash,
            "metadata": metadata or {},
            "predictions": list(predictions),
        }
        target = self.root / f"{run_id}.json"
        atomic_write_json(target, payload)
        return target

    def score_batch(self, run_id: str, results: Iterable[dict[str, Any]]) -> Path:
        source = self.root / f"{run_id}.json"
        if read_json(source) is None:
            raise FileNotFoundError(f"Unknown prediction run: {run_id}")
        target = self.root / f"{run_id}.results.json"
        if target.exists():
            raise FileExistsError(f"Results already recorded for {run_id}")
        atomic_write_json(
            target,
            {
                "run_id": run_id,
                "scored_at": datetime.now(UTC).isoformat(),
                "results": list(results),
            },
        )
        return target
