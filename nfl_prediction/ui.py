from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Any

TEAM_COLORS = {
    "ARI": "#97233F",
    "ATL": "#A71930",
    "BAL": "#241773",
    "BUF": "#00338D",
    "CAR": "#0085CA",
    "CHI": "#0B162A",
    "CIN": "#FB4F14",
    "CLE": "#311D00",
    "DAL": "#003594",
    "DEN": "#FB4F14",
    "DET": "#0076B6",
    "GB": "#203731",
    "HOU": "#03202F",
    "IND": "#002C5F",
    "JAX": "#101820",
    "KC": "#E31837",
    "LA": "#003594",
    "LAC": "#0080C6",
    "LV": "#A5ACAF",
    "MIA": "#008E97",
    "MIN": "#4F2683",
    "NE": "#C60C30",
    "NO": "#D3BC8D",
    "NYG": "#0B2265",
    "NYJ": "#125740",
    "PHI": "#004C54",
    "PIT": "#FFB612",
    "SEA": "#69BE28",
    "SF": "#AA0000",
    "TB": "#D50A0A",
    "TEN": "#4B92DB",
    "WAS": "#5A1414",
}

TEAM_NAMES = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LA": "Los Angeles Rams",
    "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}


def html_text(value: Any) -> str:
    return escape(str(value), quote=True)


def team_color(team: str) -> str:
    return TEAM_COLORS.get(team, "#5c6b7d")


def team_name(team: str) -> str:
    return TEAM_NAMES.get(team, team)


def format_probability(probability: float) -> str:
    return f"{min(max(float(probability), 0.0), 1.0):.1%}"


def american_moneyline(probability: float) -> int:
    probability = min(max(float(probability), 0.001), 0.999)
    if probability >= 0.5:
        odds = -100.0 * probability / (1.0 - probability)
    else:
        odds = 100.0 * (1.0 - probability) / probability
    return int(round(odds / 5.0) * 5)


def format_american(odds: int | float) -> str:
    value = int(round(float(odds)))
    return f"+{value}" if value > 0 else str(value)


def spread_label(game: dict[str, Any]) -> str:
    margin = float(game.get("predicted_home_margin", game.get("spread", 0.0)))
    if abs(margin) < 0.05:
        return "Pick"
    team = game["home_team"] if margin > 0 else game["away_team"]
    return f"{team} -{abs(margin):.1f}"


def format_game_time(game: dict[str, Any]) -> str:
    raw_date = str(game.get("gameday", ""))
    try:
        parsed = datetime.strptime(raw_date, "%Y-%m-%d")
        date_text = f"{parsed.strftime('%a')} {parsed.month}/{parsed.day}"
    except ValueError:
        date_text = raw_date
    raw_time = str(game.get("gametime") or "")
    if not raw_time:
        return date_text
    try:
        parsed_time = datetime.strptime(raw_time, "%H:%M")
        time_text = (
            parsed_time.strftime("%I:%M%p")
            .lstrip("0")
            .lower()
            .replace("am", "a")
            .replace("pm", "p")
        )
    except ValueError:
        time_text = raw_time
    result = f"{date_text} {time_text}".strip()
    neutral = game.get("neutral_site") or not float(
        (game.get("features") or {}).get("home_field", 1.0)
    )
    if neutral:
        stadium = str(game.get("stadium") or "").strip()
        return f"{result} · {stadium or 'neutral site'}"
    return result


def game_matchup_separator(game: dict[str, Any]) -> str:
    neutral = game.get("neutral_site") or not float(
        (game.get("features") or {}).get("home_field", 1.0)
    )
    return "vs" if neutral else "@"


def game_reasoning(game: dict[str, Any]) -> list[str]:
    features = game.get("features") or {}
    home = str(game.get("home_team", "Home"))
    away = str(game.get("away_team", "Away"))
    lines: list[str] = []

    calibration = game.get("preseason_calibration") or {}
    if float(calibration.get("weight", 0.0)):
        prior_margin = float(calibration["prior_home_margin"])
        leader = home if prior_margin >= 0 else away
        lines.append(
            f"Preseason consensus strength rates {leader} {abs(prior_margin):.1f} points better "
            "for this venue."
        )

    home_net_epa = float(features.get("home_off_epa_l4", 0.0)) - float(
        features.get("home_def_epa_l4", 0.0)
    )
    away_net_epa = float(features.get("away_off_epa_l4", 0.0)) - float(
        features.get("away_def_epa_l4", 0.0)
    )
    if abs(home_net_epa - away_net_epa) > 0.03:
        leader = home if home_net_epa > away_net_epa else away
        lines.append(f"{leader} carries the stronger net EPA edge over the last four games.")

    home_points = float(features.get("home_points_for_l4", 0.0)) - float(
        features.get("home_points_against_l4", 0.0)
    )
    away_points = float(features.get("away_points_for_l4", 0.0)) - float(
        features.get("away_points_against_l4", 0.0)
    )
    if abs(home_points - away_points) > 2.0:
        leader = home if home_points > away_points else away
        lines.append(
            f"{leader} has the stronger recent scoring differential (points for minus against)."
        )

    home_roster = sum(
        float(features.get(f"home_{feature}", 0.0))
        for feature in (
            "roster_qb_returning_delta",
            "roster_ol_continuity_delta",
            "roster_skill_continuity_delta",
        )
    )
    away_roster = sum(
        float(features.get(f"away_{feature}", 0.0))
        for feature in (
            "roster_qb_returning_delta",
            "roster_ol_continuity_delta",
            "roster_skill_continuity_delta",
        )
    )
    if abs(home_roster - away_roster) > 0.15:
        leader = home if home_roster > away_roster else away
        lines.append(
            f"{leader} returns more continuity across quarterback, offensive line, and skill positions."
        )

    home_pressure = float(features.get("home_pressure_generated_l4", 0.0)) - float(
        features.get("home_pressure_allowed_l4", 0.0)
    )
    away_pressure = float(features.get("away_pressure_generated_l4", 0.0)) - float(
        features.get("away_pressure_allowed_l4", 0.0)
    )
    if abs(home_pressure - away_pressure) > 0.02:
        leader = home if home_pressure > away_pressure else away
        lines.append(
            f"{leader}'s pass rush is generating more pressure relative to what its offense allows."
        )

    if float(features.get("division_game", 0.0)):
        lines.append("Division matchup: recent familiarity can narrow the usual home-field edge.")
    if features and not float(features.get("home_field", 1.0)):
        lines.append("Neutral-site designation removes the standard home-field boost.")
    if not lines:
        lines.append(
            "The teams grade out closely across recent scoring, EPA, and pressure: a near coin-flip forecast."
        )
    return lines[:3]


def market_line_label(game: dict[str, Any]) -> tuple[str, str]:
    consensus = game.get("market_consensus") or {}
    spread = consensus.get("spread") or {}
    total = consensus.get("total") or {}
    if not consensus:
        return "Sportsbook line", "— pending feed —"
    home_spread = spread.get("home_spread")
    market_total = total.get("total")
    parts: list[str] = []
    if home_spread is not None:
        parts.append(f"{game['home_team']} {float(home_spread):+.1f}")
    if market_total is not None:
        parts.append(f"O/U {float(market_total):.1f}")
    return str(consensus.get("provider", "Consensus")), " · ".join(parts) or "Feed available"


def relative_mae_improvement(metrics: dict[str, Any]) -> float | None:
    mae = metrics.get("mae")
    baseline = metrics.get("baseline_mae")
    if mae is None or baseline in (None, 0):
        return None
    return (float(baseline) - float(mae)) / float(baseline)
