from __future__ import annotations

import pandas as pd

from cfb_historical_benchmark import _chronological_ridge_predictions


def test_ridge_folds_never_train_on_same_or_future_week() -> None:
    games = pd.DataFrame(
        [
            {"season": 2024, "week": week, "feature": float(week), "target": float(week * 2)}
            for week in range(1, 6)
        ]
    )
    actual, predicted, indices = _chronological_ridge_predictions(
        games,
        ["feature"],
        "target",
        min_train_rows=2,
        alpha=10.0,
    )

    assert indices.tolist() == [2, 3, 4]
    assert actual.tolist() == [6.0, 8.0, 10.0]
    assert len(predicted) == 3
