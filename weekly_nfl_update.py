#!/usr/bin/env python3
"""Run the point-in-time NFL data, training, and prediction pipeline."""

from __future__ import annotations

import json
import logging
import sys

from nfl_prediction.pipeline import run_update


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        result = run_update()
    except Exception:
        logging.exception("NFL update failed")
        return 1

    print(
        json.dumps(
            {
                "status": "ok",
                "prediction_count": len(result.predictions),
                "ledger": str(result.ledger_path),
                "metrics": result.metrics,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
