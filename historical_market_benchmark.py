#!/usr/bin/env python3
"""Collect targeted historical NFL consensus and benchmark it without publishing raw odds."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from nfl_prediction.historical_market import (
    build_aggregate_report,
    build_independent_oof_games,
    build_snapshot_plan,
    collect_historical_consensus,
    save_private_records,
)
from nfl_prediction.io import atomic_write_json
from nfl_prediction.odds import OddsApiClient, estimate_historical_credits


def _integers(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        if "-" in part:
            start, end = (int(item) for item in part.split("-", maxsplit=1))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--training-seasons", default="2022-2025")
    result.add_argument("--evaluation-seasons", default="2023-2025")
    result.add_argument("--weeks", default="1-18")
    result.add_argument("--minutes-before-kickoff", type=int, default=30)
    result.add_argument("--max-credits", type=int, required=True)
    result.add_argument("--output", type=Path, default=Path("market_benchmark.json"))
    result.add_argument("--dry-run", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    training_seasons = _integers(args.training_seasons)
    evaluation_seasons = _integers(args.evaluation_seasons)
    weeks = _integers(args.weeks)
    if not set(evaluation_seasons).issubset(training_seasons):
        parser().error("evaluation seasons must be included in training seasons")
    try:
        oof = build_independent_oof_games(training_seasons=training_seasons)
        evaluation = oof[oof["season"].isin(evaluation_seasons) & oof["week"].isin(weeks)].copy()
        plan = build_snapshot_plan(evaluation, minutes_before_kickoff=args.minutes_before_kickoff)
        estimated = estimate_historical_credits(snapshots=len(plan))
        preview = {
            "training_seasons": training_seasons,
            "evaluation_seasons": evaluation_seasons,
            "weeks": weeks,
            "eligible_games": len(evaluation),
            "snapshot_requests": len(plan),
            "estimated_credits": estimated,
            "max_credits": args.max_credits,
        }
        print(json.dumps(preview, indent=2))
        if estimated > args.max_credits:
            raise ValueError(
                f"Estimated historical cost {estimated} exceeds hard budget {args.max_credits}"
            )
        if args.dry_run:
            return 0
        collected, metadata = collect_historical_consensus(
            evaluation,
            plan,
            client=OddsApiClient(),
            max_credits=args.max_credits,
        )
        save_private_records(collected)
        report = build_aggregate_report(
            collected,
            metadata,
            training_seasons=training_seasons,
            evaluation_seasons=evaluation_seasons,
            minutes_before_kickoff=args.minutes_before_kickoff,
        )
        atomic_write_json(args.output, report)
        print(json.dumps(report, indent=2))
    except Exception as exc:
        logging.error("Historical market benchmark failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
