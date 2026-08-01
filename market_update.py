#!/usr/bin/env python3
"""Fetch a guarded NFL market snapshot and publish only derived consensus lines."""

from __future__ import annotations

import argparse
import json
import logging
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
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
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
