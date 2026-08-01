#!/usr/bin/env python3
"""Refresh the derived College Football foundation artifact."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date

from cfb_prediction.pipeline import run_foundation_update


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=date.today().year)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        summary = run_foundation_update(args.season, refresh=args.refresh)
    except Exception:
        logging.exception("College Football foundation update failed")
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
