"""Read frozen forecasts and append official results for either league."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import read_json
from .ledger import PredictionLedger
from .odds import _game_kickoff, parse_timestamp


def prediction_batches(root: str | Path) -> list[dict[str, Any]]:
    batches = []
    for path in sorted(Path(root).glob("*.json")):
        if path.name.endswith(".results.json"):
            continue
        batch = read_json(path)
        if isinstance(batch, dict) and {"run_id", "created_at", "predictions"} <= batch.keys():
            batches.append(batch)
    return batches


def settle_schedule(root: str | Path, schedules: pd.DataFrame, *, sport="NFL") -> int:
    """Settle only explicit finals (CFB) or published scored schedules (NFL).

    Missing scores leave existing settlements untouched. Explicit cancelled or
    postponed states are retained without treating them as losses or zero scores.
    """
    if schedules.empty:
        return 0
    home_key, away_key = (
        ("home_points", "away_points") if sport == "CFB" else ("home_score", "away_score")
    )
    outcomes = {}
    for _, game in schedules.iterrows():
        status = str(game.get("status", "")).lower()
        if status in {"cancelled", "canceled", "postponed"}:
            outcomes[str(game["game_id"])] = dict(
                game_id=str(game["game_id"]),
                status="cancelled" if status != "postponed" else status,
                actual_home_margin=None,
                actual_total=None,
            )
            continue
        if sport == "CFB" and (pd.isna(game.get("completed")) or not bool(game.get("completed"))):
            continue
        if status and status not in {"final", "completed", "closed", "nan"}:
            continue
        home, away = game.get(home_key), game.get(away_key)
        if pd.isna(home) or pd.isna(away):
            continue
        outcomes[str(game["game_id"])] = dict(
            game_id=str(game["game_id"]),
            status="final",
            actual_home_margin=float(home - away),
            actual_total=float(home + away),
        )
    ledger = PredictionLedger(root)
    result_source = "CollegeFootballData games" if sport == "CFB" else "nflverse schedules"
    writes = 0
    for batch in prediction_batches(root):
        rows = [
            outcomes[str(p["game_id"])]
            for p in batch["predictions"]
            if str(p["game_id"]) in outcomes
        ]
        writes += ledger.settle(batch["run_id"], rows, source=result_source) is not None
    return writes


def forecast_rows(root: str | Path, *, as_of: datetime | None = None) -> pd.DataFrame:
    """Normalize both schemas. Include provenance; exclude post-kickoff runs."""
    now = as_of or datetime.now(UTC)
    if not Path(root).exists():
        return pd.DataFrame()
    ledger = PredictionLedger(root)
    rows = []
    for batch in prediction_batches(root):
        published = parse_timestamp(batch["created_at"])
        if published > now:
            continue
        settled = ledger.latest_results(batch["run_id"])
        for prediction in batch["predictions"]:
            kickoff = _game_kickoff(prediction)
            eligible = kickoff is not None and published < kickoff
            result = settled.get(str(prediction["game_id"]), {})
            football = prediction.get("football_only") or {}
            calibrated = bool((prediction.get("preseason_calibration") or {}).get("weight"))
            market = prediction.get("market_consensus") or {}
            captured = market.get("snapshot_at")
            # Old batches may contain market records: verify that they were
            # actually available at publication before including comparisons.
            try:
                market_at = parse_timestamp(captured) if captured else None
            except (TypeError, ValueError):
                market_at = None
            market_valid = bool(
                market_at
                and kickoff
                and market_at <= published < kickoff
                and published - market_at <= timedelta(days=8)
            )
            spread, total_market = market.get("spread") or {}, market.get("total") or {}
            margin = float(prediction["predicted_home_margin"])
            total = float(prediction.get("predicted_total", prediction.get("total")))
            base_margin = football.get("home_margin", margin if not calibrated else None)
            base_total = football.get("total", total)
            status = result.get(
                "status",
                "final"
                if result
                else ("awaiting result" if kickoff and kickoff <= now else "scheduled"),
            )
            row = dict(
                game_id=str(prediction["game_id"]),
                season=int(prediction.get("season", batch["prediction_season"])),
                week=int(prediction["week"]),
                home_team=prediction["home_team"],
                away_team=prediction["away_team"],
                kickoff=kickoff,
                published_at=published,
                data_cutoff=batch["data_cutoff"],
                run_id=batch["run_id"],
                model_hash=batch["model_hash"],
                eligible=eligible,
                status=status,
                revision=result.get("settlement_revision"),
                scored_at=result.get("scored_at"),
                published_margin=margin,
                published_total=total,
                independent_margin=base_margin,
                independent_total=base_total,
                market_margin=spread.get("market_home_margin") if market_valid else None,
                market_total=total_market.get("total") if market_valid else None,
                market_at=captured if market_valid else None,
                published_probability=prediction.get("home_win_probability"),
                independent_probability=football.get(
                    "home_win_probability",
                    prediction.get("home_win_probability") if not calibrated else None,
                ),
                actual_margin=result.get("actual_home_margin"),
                actual_total=result.get("actual_total"),
                forecast_method=prediction.get(
                    "forecast_method", prediction.get("forecast_type", "independent")
                ),
            )
            for source in ("published", "independent"):
                for target in ("margin", "total"):
                    mu = row[f"{source}_{target}"]
                    sigma = prediction.get(f"{target}_std")
                    low, high = prediction.get(f"{target}_p10"), prediction.get(f"{target}_p90")
                    if sigma is not None and mu is not None:
                        low, high = mu - 1.2816 * sigma, mu + 1.2816 * sigma
                    if source == "published" and target == "margin" and calibrated:
                        low, high = None, None
                    row[f"{source}_{target}_p10"] = low
                    row[f"{source}_{target}_p90"] = high
            rows.append(row)
    return pd.DataFrame(rows)


def select_forecasts(rows: pd.DataFrame, *, policy="first", horizon_minutes=60) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    selected = rows[rows["eligible"]].copy()
    if policy == "horizon":
        selected = selected[
            selected["published_at"] <= selected["kickoff"] - pd.Timedelta(minutes=horizon_minutes)
        ]
    elif policy != "first":
        raise ValueError("Unknown forecast selection policy")
    selected = selected.sort_values(["published_at", "run_id"])
    return selected.drop_duplicates(
        ["season", "game_id"], keep="first" if policy == "first" else "last"
    )


def summarize(rows: pd.DataFrame, *, source="published", target="margin") -> dict[str, Any]:
    if source not in {"published", "independent"} or target not in {"margin", "total"}:
        raise ValueError("Unknown forecast source or target")
    empty = dict(
        games=0,
        mae=None,
        rmse=None,
        bias=None,
        matched_games=0,
        matched_model_mae=None,
        market_mae=None,
        difference=None,
        difference_low=None,
        difference_high=None,
        interval_games=0,
        coverage=None,
        interval_width=None,
        winner_games=0,
        winner_accuracy=None,
        brier=None,
        log_loss=None,
    )
    if rows.empty:
        return empty
    predicted, actual = f"{source}_{target}", f"actual_{target}"
    scored = rows[rows.status.eq("final")].dropna(subset=[predicted, actual])
    if scored.empty:
        return empty
    errors = scored[predicted] - scored[actual]
    result = {
        **empty,
        "games": len(scored),
        "mae": float(errors.abs().mean()),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "bias": float(errors.mean()),
    }
    matched = scored.dropna(subset=[f"market_{target}"])
    if not matched.empty:
        model_error = (matched[predicted] - matched[actual]).abs()
        market_error = (matched[f"market_{target}"] - matched[actual]).abs()
        result.update(
            matched_games=len(matched),
            matched_model_mae=float(model_error.mean()),
            market_mae=float(market_error.mean()),
            difference=float((model_error - market_error).mean()),
        )
        blocks = (
            matched.assign(difference=model_error - market_error)
            .groupby(["season", "week"])["difference"]
            .agg(["sum", "count"])
        )
        if len(blocks) >= 4:
            samples = np.random.default_rng(42).integers(0, len(blocks), size=(1000, len(blocks)))
            draws = blocks["sum"].to_numpy()[samples].sum(axis=1) / blocks["count"].to_numpy()[
                samples
            ].sum(axis=1)
            result["difference_low"], result["difference_high"] = map(
                float, np.quantile(draws, [0.025, 0.975])
            )
    intervals = scored.dropna(subset=[f"{source}_{target}_p10", f"{source}_{target}_p90"])
    if not intervals.empty:
        low, high = intervals[f"{source}_{target}_p10"], intervals[f"{source}_{target}_p90"]
        result.update(
            interval_games=len(intervals),
            coverage=float(((intervals[actual] >= low) & (intervals[actual] <= high)).mean()),
            interval_width=float((high - low).mean()),
        )
    winners = scored[scored.actual_margin.ne(0)].dropna(
        subset=["actual_margin", f"{source}_probability", f"{source}_margin"]
    )
    if not winners.empty:
        probability = winners[f"{source}_probability"].clip(1e-6, 1 - 1e-6)
        outcome = winners.actual_margin.gt(0).astype(float)
        result.update(
            winner_games=len(winners),
            winner_accuracy=float((probability.gt(0.5) == outcome).mean()),
            brier=float(((probability - outcome) ** 2).mean()),
            log_loss=float(
                -np.mean(outcome * np.log(probability) + (1 - outcome) * np.log(1 - probability))
            ),
        )
    return result


def performance_history(root: str | Path) -> dict[str, Any]:
    rows = forecast_rows(root)
    runs = []
    if not rows.empty:
        for run_id, group in rows[rows.eligible].groupby("run_id"):
            margin, total = summarize(group), summarize(group, target="total")
            if margin["games"]:
                runs.append(
                    dict(
                        run_id=run_id,
                        games=margin["games"],
                        margin_mae=margin["mae"],
                        total_mae=total["mae"],
                        winner_accuracy=margin["winner_accuracy"],
                        market_games=margin["matched_games"],
                        matched_model_margin_mae=margin["matched_model_mae"],
                        market_margin_mae=margin["market_mae"],
                        market_total_games=total["matched_games"],
                        matched_model_total_mae=total["matched_model_mae"],
                        market_total_mae=total["market_mae"],
                    )
                )
    return {"runs": runs, "updated_at": datetime.now(UTC).isoformat(), "schema_version": 2}
