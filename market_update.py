#!/usr/bin/env python3
"""Fetch a guarded NFL market snapshot and publish only derived consensus lines."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime

from nfl_prediction.config import MARKET_PRIVATE_DIR
from nfl_prediction.odds import (
    DEFAULT_MARKETS,
    MarketSnapshotStore,
    OddsApiClient,
    OddsApiError,
    build_consensus,
    estimate_historical_credits,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--historical-at", help="UTC timestamp for a paid historical snapshot")
    result.add_argument("--regions", default="us")
    result.add_argument("--markets", default=",".join(DEFAULT_MARKETS))
    result.add_argument("--max-credits", type=int, default=20)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--provider", choices=("owls", "legacy"))
    result.add_argument("--sport", choices=("nfl", "ncaaf", "both"), default="nfl")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    provider = args.provider or (
        "legacy" if args.historical_at else os.environ.get("GRIDLINE_MARKET_PROVIDER", "owls")
    )
    if provider == "owls":
        if args.historical_at:
            parser().error("Owls history is not used; pass --provider legacy for paid history")
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "provider": "Owls Insight",
                        "maximum_requests": 4 if args.sport == "both" else 2,
                        "minimum_poll_seconds": 300,
                    }
                )
            )
            return 0
        from current_market_update import main as current_main

        return current_main(["--sport", args.sport])
    if provider != "legacy":
        parser().error("GRIDLINE_MARKET_PROVIDER must be owls or legacy")
    if args.sport != "nfl":
        parser().error(
            "Legacy snapshot publishing is NFL-only; use current_market_update.py --parity for NCAAF comparison"
        )
    markets = tuple(item.strip() for item in args.markets.split(",") if item.strip())
    if not markets or not set(markets).issubset(DEFAULT_MARKETS):
        parser().error("markets must be spreads, totals, or both")
    estimated = (
        estimate_historical_credits(regions=args.regions, markets=markets)
        if args.historical_at
        else len([region for region in args.regions.split(",") if region]) * len(markets)
    )
    if estimated > args.max_credits:
        raise SystemExit(
            f"Refusing request: estimated {estimated} credits exceeds --max-credits {args.max_credits}"
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "request": "historical" if args.historical_at else "current",
                    "estimated_credits": estimated,
                }
            )
        )
        return 0
    try:
        client = OddsApiClient()
        fetch = (
            client.historical_odds(args.historical_at, regions=args.regions, markets=markets)
            if args.historical_at
            else client.current_odds(regions=args.regions, markets=markets)
        )
        consensus = build_consensus(fetch, regions=args.regions, markets=markets)
        if args.historical_at:
            stamp = consensus["snapshot_at"].replace(":", "").replace("+00:00", "Z")
            consensus_target = MARKET_PRIVATE_DIR / "consensus" / f"historical-{stamp}.json"
            store = MarketSnapshotStore(consensus_path=consensus_target)
        else:
            store = MarketSnapshotStore()
        raw_path, consensus_path = store.save(fetch, consensus)
    except (OddsApiError, FileExistsError, ValueError) as exc:
        logging.error("Market update failed: %s", exc)
        return 1
    print(
        json.dumps(
            {
                "status": "ok",
                "snapshot_at": consensus["snapshot_at"],
                "games": len(consensus["games"]),
                "credits": consensus["credits"],
                "private_raw_file": raw_path.name,
                "consensus_file": str(consensus_path),
                "completed_at": datetime.now().astimezone().isoformat(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
