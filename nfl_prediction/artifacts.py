"""Resolve complete releases and legacy immutable NFL batches."""

from __future__ import annotations

from pathlib import Path

from .io import read_json, sha256_file


def frozen_nfl_batch(root: Path, *, manifest_path: Path) -> dict:
    report = read_json(root / "weekly_report.json", {})
    update = read_json(root / "update_log.json", {})
    run_id = report.get("run_id") or update.get("ledger_run")
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("The NFL release has no valid frozen prediction run")
    batch = read_json(root / "data" / "predictions" / f"{run_id}.json")
    if (
        not batch
        or batch.get("run_id") != run_id
        or batch.get("model_hash") != sha256_file(manifest_path)
    ):
        raise ValueError("The NFL prediction batch does not match the active model bundle")
    return batch


def release_manifest(root: Path, *, sport="NFL") -> Path:
    release = (
        read_json(root / "data" / "nfl_release.json")
        if sport == "NFL"
        else read_json(root / "data" / "cfb" / "latest_prediction.json")
    )
    relative = (release or {}).get("manifest_path")
    if relative:
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Model manifest path leaves the project")
        return path
    return (
        root / "models" / "manifest.json"
        if sport == "NFL"
        else root / "data" / "cfb" / "models" / "manifest.json"
    )
