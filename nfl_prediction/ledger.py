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

    def result_events(self, run_id: str) -> list[dict[str, Any]]:
        """Read legacy results followed by immutable correction/settlement events."""
        if Path(run_id).name != run_id:
            raise ValueError("Invalid prediction run ID")
        legacy = read_json(self.root / f"{run_id}.results.json")
        events = [dict(legacy, revision=0)] if legacy else []
        events.extend(
            read_json(path) for path in sorted((self.root / "settlements" / run_id).glob("*.json"))
        )
        return events

    def latest_results(self, run_id: str) -> dict[str, dict[str, Any]]:
        latest = {}
        for event in self.result_events(run_id):
            for row in event["results"]:
                latest[str(row["game_id"])] = {
                    **row,
                    "settlement_revision": event["revision"],
                    "scored_at": event["scored_at"],
                }
        return latest

    def settle(
        self,
        run_id: str,
        results: Iterable[dict[str, Any]],
        *,
        source: str | None = None,
    ) -> Path | None:
        """Append changed outcomes; never rewrite predictions or earlier outcomes.

        One writer per ledger is required (workflows use a concurrency group).
        Replaying identical results is idempotent; partial weeks may grow.
        """
        if Path(run_id).name != run_id:
            raise ValueError("Invalid prediction run ID")
        batch = read_json(self.root / f"{run_id}.json")
        if batch is None:
            raise FileNotFoundError(f"Unknown prediction run: {run_id}")
        game_ids = {str(row["game_id"]) for row in batch["predictions"]}
        latest = self.latest_results(run_id)
        changed = []
        for source in results:
            row = {**source, "game_id": str(source["game_id"])}
            if row["game_id"] not in game_ids:
                raise ValueError("Settlement game is absent from prediction batch")
            previous = latest.get(row["game_id"], {})
            if any(previous.get(key) != value for key, value in row.items()):
                changed.append(row)
        if not changed:
            return None
        revision = max((e["revision"] for e in self.result_events(run_id)), default=0) + 1
        target = self.root / "settlements" / run_id / f"{revision:06d}.json"
        if target.exists():
            raise FileExistsError(f"Concurrent settlement for {run_id}; retry")
        atomic_write_json(
            target,
            {
                "run_id": run_id,
                "revision": revision,
                "scored_at": datetime.now(UTC).isoformat(),
                "source": source,
                "results": changed,
            },
        )
        return target
