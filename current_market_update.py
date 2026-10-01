#!/usr/bin/env python3
"""Backend Owls worker or one-shot provider parity diagnostics. Never loads a model."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nfl_prediction.config import PROJECT_ROOT
from nfl_prediction.current_market import MarketStore, poll_market
from nfl_prediction.io import atomic_write_json, read_json
from nfl_prediction.market_parity import live_parity
from nfl_prediction.money_signals import published_forecasts
from nfl_prediction.prop_cache import poll_depth, poll_props


def load_slate(sport: str, path: str | Path | None = None) -> list[dict[str, Any]]:
    if path:
        payload = read_json(path, [])
    elif sport == "nfl":
        release = read_json(PROJECT_ROOT / "data/nfl_release.json", {})
        payload = (release.get("state") or {}).get("prediction_batch") or read_json(
            PROJECT_ROOT / "weekly_schedule.json", []
        )
    else:
        pointer = read_json(PROJECT_ROOT / "data/cfb/latest_prediction.json", {})
        filename = str(pointer.get("path", ""))
        if not filename or Path(filename).name != filename:
            return []
        payload = read_json(PROJECT_ROOT / "data/cfb/predictions" / filename, {})
    return published_forecasts(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sport", choices=("nfl", "ncaaf", "both"), default="both")
    parser.add_argument("--slate-nfl", type=Path)
    parser.add_argument("--slate-ncaaf", type=Path)
    parser.add_argument("--db", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--watch", action="store_true")
    modes.add_argument("--parity", action="store_true")
    modes.add_argument(
        "--status", action="store_true", help="Read deployment status without polling"
    )
    parser.add_argument("--output", type=Path, default=Path("reports/owls-parity.json"))
    args = parser.parse_args(argv)
    sports = ("nfl", "ncaaf") if args.sport == "both" else (args.sport,)
    store = MarketStore(args.db)
    if args.status:
        boards = {sport: store.read(sport) for sport in sports}
        props, depth = store.read("nfl_props"), store.read("nfl_depth")
        print(
            json.dumps(
                {
                    "storage": "postgres" if store.database_url else "sqlite",
                    "owls_key_configured": bool(os.environ.get("OWLS_INSIGHT_API_KEY", "").strip()),
                    "configuration_error": store.configuration_error,
                    "sports": {
                        sport: {
                            "cached_games": len(board.get("games", [])),
                            "last_attempt_at": board.get("last_attempt_at"),
                            "odds_error": board.get("odds_error"),
                            "splits_error": board.get("splits_error"),
                        }
                        for sport, board in boards.items()
                    },
                    "props": {
                        "cached_quotes": len(props.get("rows", [])),
                        "last_attempt_at": props.get("last_attempt_at"),
                        "error": props.get("error") or props.get("odds_error"),
                        "depth_error": depth.get("error") or depth.get("odds_error"),
                    },
                },
                indent=2,
            ),
            flush=True,
        )
        return int(bool(store.configuration_error))
    while True:
        results = {}
        for sport in sports:
            slate = load_slate(sport, getattr(args, f"slate_{sport}"))
            if args.parity:
                results[sport] = live_parity(sport, slate)
            else:
                board = poll_market(sport, slate, store=store)
                results[sport] = {
                    "cached_games": len(board.get("games", [])),
                    "last_attempt_at": board.get("last_attempt_at"),
                    "odds_error": board.get("odds_error"),
                    "splits_error": board.get("splits_error"),
                    "diagnostics": dict(Counter(d["reason"] for d in board.get("diagnostics", []))),
                    "missing_games": board.get("missing_games", []),
                }
        if args.parity:
            atomic_write_json(args.output, results)
        elif "nfl" in sports:
            slate = load_slate("nfl", args.slate_nfl)
            props = poll_props(slate, store=store)
            season = max(
                (int(g.get("season") or datetime.now(UTC).year) for g in slate),
                default=datetime.now(UTC).year,
            )
            depth = poll_depth(season, store=store)
            results["nfl_props"] = {
                "cached_quotes": len(props.get("rows", [])),
                "error": props.get("error"),
                "last_attempt_at": props.get("last_attempt_at"),
                "diagnostics": props.get("diagnostics", {}),
                "depth_error": depth.get("error"),
            }
        summary = {
            sport: {
                key: value
                for key, value in result.items()
                if key not in {"games", "owls_unmatched", "legacy_unmatched"}
            }
            for sport, result in results.items()
        }
        print(json.dumps(summary, indent=2), flush=True)
        if not args.watch:
            return int(
                any(
                    r.get("odds_error")
                    or r.get("error")
                    or r.get("depth_error")
                    or r.get("status") == "unverified"
                    for r in results.values()
                )
            )
        # Check persisted gates frequently; cadence is enforced transactionally in the DB.
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
