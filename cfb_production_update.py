#!/usr/bin/env python3
"""Fit the selected CFB models and record a guarded weekly prediction batch."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime

from cfb_prediction.production import run_cfb_production_update


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=datetime.now().year)
    parser.add_argument("--first-training-season", type=int, default=2018)
    parser.add_argument("--week", type=int)
    parser.add_argument("--as-of", type=datetime.fromisoformat)
    parser.add_argument("--refresh-current", action="store_true")
    parser.add_argument("--git-commit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        result = run_cfb_production_update(
            prediction_season=args.season,
            historical_seasons=list(range(args.first_training_season, args.season)),
            as_of=args.as_of,
            week=args.week,
            refresh_current=args.refresh_current,
            git_commit=args.git_commit,
        )
    except Exception:
        logging.exception("College Football production update failed")
        return 1
    print(
        json.dumps(
            {
                "model_hash": result["model_hash"],
                "forecast_week": result["forecast_week"],
                "prediction_count": result["prediction_count"],
                "prediction_path": result["prediction_path"],
                "rankings_path": result["rankings_path"],
                "ranking_count": result["ranking_count"],
                "predictions": result["predictions"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
