"""Research-only opponent-adjusted scoring state with explicit offseason uncertainty."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrengthConfig:
    carry: float = 0.65
    observation_variance: float = 144.0
    offseason_variance: float = 16.0
    process_variance: float = 1.0


def strength_forecasts(schedules: pd.DataFrame, config: StrengthConfig) -> pd.DataFrame:
    """Freeze a whole week's state before updating on that week's outcomes.

    Offense is points above average, defense is points prevented. Two noisy
    score observations update each side; this is a diagonal Kalman approximation,
    not a causal player-value model. No sportsbook or future roster data is used.
    """
    games = schedules.copy()
    if "game_type" in games:
        games = games[games.game_type.eq("REG")]
    games = games.dropna(subset=["home_score", "away_score"])
    states = defaultdict(lambda: np.array([0.0, 0.0, 16.0, 16.0]))
    last_season = None
    points_sum, team_games = 45.0 * 32, 64
    rows = []
    for (season, week), group in games.groupby(["season", "week"], sort=True):
        if last_season is not None and season != last_season:
            gap = int(season - last_season)
            for state in states.values():
                for _ in range(gap):
                    state[:2] *= config.carry
                    state[2:] = config.carry**2 * state[2:] + config.offseason_variance
        last_season = season
        league = points_sum / team_games
        pending = []
        for game in group.itertuples():
            home, away = states[game.home_team].copy(), states[game.away_team].copy()
            home[2:] += config.process_variance
            away[2:] += config.process_variance
            edge = 0.0 if str(getattr(game, "location", "Home")).lower() == "neutral" else 2.0
            home_mu = league + home[0] - away[1] + edge / 2
            away_mu = league + away[0] - home[1] - edge / 2
            rows.append(
                dict(
                    game_id=game.game_id,
                    season=int(season),
                    week=int(week),
                    strength_margin=home_mu - away_mu,
                    strength_total=home_mu + away_mu,
                    strength_uncertainty=float(home[2:].sum() + away[2:].sum()),
                )
            )
            home_residual, away_residual = game.home_score - home_mu, game.away_score - away_mu
            home_denominator = config.observation_variance + home[2] + away[3]
            away_denominator = config.observation_variance + away[2] + home[3]
            new_home, new_away = home.copy(), away.copy()
            new_home[0] += home[2] / home_denominator * home_residual
            new_away[1] -= away[3] / home_denominator * home_residual
            new_away[0] += away[2] / away_denominator * away_residual
            new_home[1] -= home[3] / away_denominator * away_residual
            new_home[2] *= 1 - home[2] / home_denominator
            new_away[3] *= 1 - away[3] / home_denominator
            new_away[2] *= 1 - away[2] / away_denominator
            new_home[3] *= 1 - home[3] / away_denominator
            pending.extend(((game.home_team, new_home), (game.away_team, new_away)))
        for team, state in pending:
            states[team] = state
        points_sum += float(group.home_score.sum() + group.away_score.sum())
        team_games += len(group) * 2
    return pd.DataFrame(rows)
