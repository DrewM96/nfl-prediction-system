#!/usr/bin/env python3
"""Retrospectively compare frozen 2026 CFB Weeks 1-2 forecasts with listed CFBD lines."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from cfb_prediction.client import CFBDClient
from cfb_prediction.data import normalize_lines
from nfl_prediction.results import forecast_rows, select_forecasts

PREDICTIONS_ROOT = Path("data/cfb/predictions")
OUTPUT_PATH = Path("data/cfb/weeks_1_2_market_check.json")


def _metrics(frame: pd.DataFrame) -> dict:
    matched = frame.dropna(subset=["published_margin", "market_margin", "actual_margin"]).copy()
    if matched.empty:
        return {
            "games": 0,
            "model_mae": None,
            "market_mae": None,
            "model_minus_market_mae": None,
            "ats_wins": 0,
            "ats_losses": 0,
            "ats_pushes": 0,
            "ats_rate": None,
        }

    model_error = (matched["published_margin"] - matched["actual_margin"]).abs()
    market_error = (matched["market_margin"] - matched["actual_margin"]).abs()
    model_edge = matched["published_margin"] - matched["market_margin"]
    ats_margin = np.sign(model_edge) * (matched["actual_margin"] - matched["market_margin"])
    wins = int(ats_margin.gt(0).sum())
    losses = int(ats_margin.lt(0).sum())
    pushes = int(ats_margin.eq(0).sum())
    decisions = wins + losses
    return {
        "games": int(len(matched)),
        "model_mae": float(model_error.mean()),
        "market_mae": float(market_error.mean()),
        "model_minus_market_mae": float(model_error.mean() - market_error.mean()),
        "ats_wins": wins,
        "ats_losses": losses,
        "ats_pushes": pushes,
        "ats_rate": float(wins / decisions) if decisions else None,
    }


def _buckets(frame: pd.DataFrame) -> list[dict]:
    matched = frame.dropna(subset=["published_margin", "market_margin", "actual_margin"]).copy()
    matched["disagreement"] = (matched["published_margin"] - matched["market_margin"]).abs()
    output = []
    for low, high, label in [
        (0.0, 2.0, "0-2"),
        (2.0, 4.0, "2-4"),
        (4.0, 6.0, "4-6"),
        (6.0, 8.0, "6-8"),
        (8.0, np.inf, "8+"),
    ]:
        if np.isinf(high):
            subset = matched[matched["disagreement"].ge(low)]
        else:
            subset = matched[matched["disagreement"].ge(low) & matched["disagreement"].lt(high)]
        item = {"bucket": label, **_metrics(subset)}
        output.append(item)
    return output


def main() -> None:
    client = CFBDClient.from_environment()
    raw_lines = client.get(
        "/lines",
        params={"year": 2026, "seasonType": "regular"},
        refresh=True,
        max_age=timedelta(0),
    )
    lines = normalize_lines(raw_lines)
    lines = lines.rename(
        columns={
            "market_home_margin": "retrospective_market_margin",
            "market_total": "retrospective_market_total",
        }
    )

    rows = select_forecasts(forecast_rows(PREDICTIONS_ROOT), policy="first")
    rows = rows[
        rows["season"].eq(2026) & rows["week"].isin([1, 2]) & rows["status"].eq("final")
    ].copy()
    rows["game_id_int"] = pd.to_numeric(rows["game_id"], errors="coerce").astype("Int64")
    matched = rows.merge(
        lines[
            [
                "game_id",
                "retrospective_market_margin",
                "retrospective_market_total",
                "market_provider_count",
            ]
        ],
        left_on="game_id_int",
        right_on="game_id",
        how="left",
        suffixes=("", "_line"),
    )
    matched["market_margin"] = matched["retrospective_market_margin"]
    matched["market_total"] = matched["retrospective_market_total"]

    game_rows = []
    for _, row in matched.sort_values(["week", "published_at", "game_id_int"]).iterrows():
        game_rows.append(
            {
                "week": int(row["week"]),
                "game_id": str(row["game_id_int"]),
                "matchup": f'{row["away_team"]} @ {row["home_team"]}',
                "published_at": row["published_at"].isoformat(),
                "model_margin": float(row["published_margin"]),
                "market_margin": float(row["market_margin"])
                if pd.notna(row["market_margin"])
                else None,
                "actual_margin": float(row["actual_margin"]),
                "model_error": float(abs(row["published_margin"] - row["actual_margin"])),
                "market_error": float(abs(row["market_margin"] - row["actual_margin"]))
                if pd.notna(row["market_margin"])
                else None,
                "model_market_disagreement": float(abs(row["published_margin"] - row["market_margin"]))
                if pd.notna(row["market_margin"])
                else None,
                "market_provider_count": int(row["market_provider_count"])
                if pd.notna(row["market_provider_count"])
                else 0,
            }
        )

    payload = {
        "schema_version": 1,
        "season": 2026,
        "weeks": [1, 2],
        "market_source": "CollegeFootballData listed-line consensus",
        "timing_warning": (
            "Retrospective listed-line comparison only. CFBD line records do not expose a historical "
            "snapshot timestamp here, so these lines must not be represented as captured at the frozen "
            "forecast publication time or as verified closing lines."
        ),
        "forecast_games": int(len(matched)),
        "matched_market_games": int(matched["market_margin"].notna().sum()),
        "combined": _metrics(matched),
        "by_week": {
            str(week): _metrics(matched[matched["week"].eq(week)]) for week in (1, 2)
        },
        "disagreement_buckets": _buckets(matched),
        "games": game_rows,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("forecast_games", "matched_market_games", "combined", "by_week", "disagreement_buckets")}, indent=2))


if __name__ == "__main__":
    main()
