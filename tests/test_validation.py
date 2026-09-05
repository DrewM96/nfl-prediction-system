from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_prediction.validation import paired_week_block_interval, prequential_predictions


def test_prequential_meta_predictions_cannot_see_current_or_future_outcomes() -> None:
    frame = pd.DataFrame({"season": [2025] * 4, "week": [1, 2, 3, 4]})
    actual = np.array([0.0, 2.0, 4.0, 6.0])
    first = np.array([1.0, 3.0, 5.0, 7.0])
    second = np.array([-1.0, 1.0, 3.0, 5.0])
    baseline = np.zeros(4)

    original = prequential_predictions(frame, actual, first, second, baseline)
    changed = prequential_predictions(
        frame,
        np.array([0.0, 2.0, 4.0, -1000.0]),
        first,
        second,
        baseline,
    )

    np.testing.assert_allclose(original, changed)


def test_paired_block_interval_preserves_the_game_level_pairing() -> None:
    frame = pd.DataFrame({"season": [2025] * 4, "week": [1, 2, 3, 4]})
    actual = np.zeros(4)
    candidate = np.ones(4)
    reference = np.full(4, 2.0)

    assert paired_week_block_interval(frame, actual, candidate, reference) == (-1.0, -1.0)
