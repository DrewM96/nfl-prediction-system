"""Pick records and team views derived only from selected, frozen forecasts."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def numeric(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def pick_side(projection: Any, line: Any = 0) -> int:
    projection, line = numeric(projection), numeric(line)
    if projection is None or line is None or abs(projection - line) < 0.05:
        return 0
    return 1 if projection > line else -1


def grade_pick(projection: Any, line: Any, actual: Any, status: str, *, winner=False) -> str:
    if status in {"cancelled", "postponed", "void"}:
        return "Void"
    if numeric(projection) is None:
        return "No projection"
    if numeric(line) is None:
        return "No line"
    side = pick_side(projection, line)
    if not side:
        return "No pick"
    if status != "final" or numeric(actual) is None:
        return "Pending"
    result = float(actual) - float(line)
    if abs(result) < 1e-8:
        return "Tie" if winner else "Push"
    return "Win" if side * result > 0 else "Loss"


def pick_record(outcomes) -> dict[str, Any]:
    counts = pd.Series(list(outcomes), dtype="object").value_counts()
    wins, losses = int(counts.get("Win", 0)), int(counts.get("Loss", 0))
    return {
        "wins": wins,
        "losses": losses,
        "pushes": int(counts.get("Push", 0)),
        "ties": int(counts.get("Tie", 0)),
        "decisions": wins + losses,
        "rate": wins / (wins + losses) if wins + losses else None,
        "pending": int(counts.get("Pending", 0)),
        "no_pick": int(counts.get("No pick", 0)),
        "no_line": int(counts.get("No line", 0)),
        "void": int(counts.get("Void", 0)),
    }


def score_games(rows: pd.DataFrame, *, source="published") -> pd.DataFrame:
    if source not in {"published", "independent"}:
        raise ValueError("Unknown forecast source")
    result = rows.copy()
    if result.empty:
        return result
    for kind, target, reference in (
        ("winner", "margin", None),
        ("ats", "margin", "market_margin"),
        ("total", "total", "market_total"),
    ):
        result[f"{kind}_side"] = [
            pick_side(r.get(f"{source}_{target}"), r.get(reference) if reference else 0)
            for r in result.to_dict("records")
        ]
        result[f"{kind}_result"] = [
            grade_pick(
                r.get(f"{source}_{target}"),
                r.get(reference) if reference else 0,
                r.get(f"actual_{target}"),
                r["status"],
                winner=kind == "winner",
            )
            for r in result.to_dict("records")
        ]
    for target in ("margin", "total"):
        result[f"{target}_error"] = (
            (
                pd.to_numeric(result[f"{source}_{target}"], errors="coerce")
                - pd.to_numeric(result[f"actual_{target}"], errors="coerce")
            )
            .abs()
            .where(result.status.eq("final"))
        )
    return result


def team_breakdown(scored: pd.DataFrame, *, source="published", minimum=1) -> pd.DataFrame:
    """Rank team score MAE; ATS records count only occasions the model backed that team."""
    records = []
    if scored.empty:
        return pd.DataFrame()
    finals = scored[scored.status.eq("final")]
    for team in sorted(set(finals.home_team) | set(finals.away_team)):
        games = finals[finals.home_team.eq(team) | finals.away_team.eq(team)]
        home = games.home_team.eq(team)
        sign = home.map({True: 1, False: -1})
        projected = (games[f"{source}_total"] + sign * games[f"{source}_margin"]) / 2
        actual = (games.actual_total + sign * games.actual_margin) / 2
        error = (projected - actual).dropna()
        if len(error) < minimum:
            continue
        backed = games[(home & games.ats_side.eq(1)) | (~home & games.ats_side.eq(-1))]
        ats = pick_record(backed.ats_result)
        winner = pick_record(games.winner_result)
        records.append(
            {
                "team": team,
                "games": len(error),
                "score_mae": float(error.abs().mean()),
                "score_bias": float(error.mean()),
                "margin_mae": games.margin_error.mean(),
                "winner_wins": winner["wins"],
                "winner_losses": winner["losses"],
                "winner_rate": winner["rate"],
                "ats_wins": ats["wins"],
                "ats_losses": ats["losses"],
                "ats_pushes": ats["pushes"],
                "ats_rate": ats["rate"],
                "ats_picks": ats["decisions"] + ats["pushes"],
                "form": games.sort_values(["week", "kickoff"]).winner_result.tolist()[-5:],
            }
        )
    return (
        pd.DataFrame(records)
        .sort_values(["score_mae", "games", "team"], ascending=[True, False, True])
        .reset_index(drop=True)
        if records
        else pd.DataFrame()
    )


def weekly_records(scored: pd.DataFrame, *, cumulative=False) -> pd.DataFrame:
    records = []
    if scored.empty:
        return pd.DataFrame()
    finals = scored[scored.status.eq("final")]
    for week in sorted(finals.week.unique()):
        group = finals[finals.week.le(week)] if cumulative else finals[finals.week.eq(week)]
        for kind, label in (
            ("winner", "Winners"),
            ("ats", "Against the spread"),
            ("total", "Totals"),
        ):
            record = pick_record(group[f"{kind}_result"])
            if record["decisions"]:
                records.append({"week": int(week), "series": label, **record})
    return pd.DataFrame(records)
