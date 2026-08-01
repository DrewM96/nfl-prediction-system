from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import MODEL_MANIFEST_PATH, PROJECT_ROOT, get_season_context
from .data import load_nflverse_data
from .features import (
    GAME_FEATURES,
    PLAYER_FEATURES,
    add_shifted_rolling_features,
    build_player_game_logs,
    build_point_in_time_game_features,
)
from .io import atomic_write_json, sha256_file
from .ledger import PredictionLedger
from .modeling import GAME_RIDGE_ALPHA, FittedEnsemble, fit_ensemble, save_model_bundle
from .odds import attach_market_consensus, load_market_consensus


@dataclass
class UpdateResult:
    manifest: dict[str, Any]
    predictions: list[dict[str, Any]]
    ledger_path: Path
    metrics: dict[str, Any]


def _raw_data_fingerprint(data: Any) -> str:
    """Create a reproducible fingerprint of every source that affects released artifacts."""
    digest = hashlib.sha256()
    pbp_columns = [
        "game_id",
        "play_id",
        "season",
        "week",
        "game_date",
        "season_type",
        "posteam",
        "defteam",
        "play_type",
        "yards_gained",
        "epa",
        "qb_dropback",
        "qb_hit",
        "sack",
        "interception",
        "fumble_lost",
        "pass_attempt",
        "complete_pass",
        "passing_yards",
        "pass_touchdown",
        "passer_player_id",
        "passer_player_name",
        "receiver_player_id",
        "receiver_player_name",
        "receiving_yards",
        "receiving_td",
        "rush_attempt",
        "rushing_yards",
        "rush_touchdown",
        "rusher_player_id",
        "rusher_player_name",
        "qb_kneel",
        "qb_spike",
    ]
    sources = {
        "pbp": (data.pbp, pbp_columns),
        "schedules": (data.schedules, list(data.schedules.columns)),
        "rosters": (data.rosters, list(data.rosters.columns)),
        "injuries": (data.injuries, list(data.injuries.columns)),
        "snap_counts": (data.snap_counts, list(data.snap_counts.columns)),
    }
    for name, (frame, requested_columns) in sources.items():
        digest.update(name.encode("utf-8"))
        columns = [column for column in requested_columns if column in frame]
        digest.update("\n".join(columns).encode("utf-8"))
        if frame.empty or not columns:
            continue
        hashes = pd.util.hash_pandas_object(frame[columns], index=False, categorize=True)
        digest.update(hashes.to_numpy(dtype="uint64").tobytes())
    return digest.hexdigest()


def _safe_divide(numerator: pd.Series, denominator: pd.Series, default: float) -> pd.Series:
    result = pd.to_numeric(numerator, errors="coerce") / pd.to_numeric(denominator, errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan).fillna(default)


def _attach_opponent_defense(player_frame: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    lookup = games[
        ["game_id", "home_team", "away_team", "home_def_epa_l4", "away_def_epa_l4"]
    ].drop_duplicates("game_id")
    result = player_frame.merge(lookup, on="game_id", how="left")
    result["opponent_def_epa"] = np.where(
        result["posteam"].eq(result["home_team"]),
        result["away_def_epa_l4"],
        result["home_def_epa_l4"],
    )
    return result.drop(
        columns=["home_team", "away_team", "home_def_epa_l4", "away_def_epa_l4"],
        errors="ignore",
    )


def _add_participation_spines(
    logs: dict[str, pd.DataFrame],
    snap_counts: pd.DataFrame,
    rosters: pd.DataFrame,
    pbp: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Preserve active zero-opportunity games and attach real offensive snap share."""
    required_snap = {"season", "week", "game_id", "pfr_player_id", "position", "team"}
    required_roster = {"season", "pfr_id", "gsis_id", "full_name"}
    if not required_snap.issubset(snap_counts) or not required_roster.issubset(rosters):
        return logs

    crosswalk = rosters.copy()
    crosswalk = crosswalk[
        crosswalk["pfr_id"].fillna("").ne("") & crosswalk["gsis_id"].fillna("").ne("")
    ].sort_values(["season", "week"])
    crosswalk = crosswalk.drop_duplicates(["season", "pfr_id"], keep="last")
    snaps = snap_counts.merge(
        crosswalk[["season", "pfr_id", "gsis_id", "full_name"]],
        left_on=["season", "pfr_player_id"],
        right_on=["season", "pfr_id"],
        how="inner",
    )
    if "game_type" in snaps:
        snaps = snaps[snaps["game_type"].eq("REG")]
    snaps["offense_pct"] = pd.to_numeric(snaps.get("offense_pct"), errors="coerce").fillna(0.0)
    snaps = snaps[snaps["offense_pct"].gt(0)].copy()
    game_dates = pbp[["game_id", "game_date"]].dropna().drop_duplicates("game_id")
    snaps = snaps.merge(game_dates, on="game_id", how="left")

    configurations = {
        "passing": (
            {"QB"},
            ["passing_yards", "attempts", "completions", "passing_tds", "interceptions"],
        ),
        "receiving": (
            {"WR", "TE", "RB", "FB"},
            ["receiving_yards", "targets", "receptions", "receiving_tds"],
        ),
        "rushing": ({"QB", "RB", "FB", "WR"}, ["rushing_yards", "carries", "rushing_tds"]),
    }
    enriched: dict[str, pd.DataFrame] = {}
    keys = ["season", "week", "game_id", "game_date", "posteam", "player_id"]
    for kind, (positions, stat_columns) in configurations.items():
        spine = snaps[snaps["position"].isin(positions)].copy()
        spine = spine.rename(
            columns={"team": "posteam", "gsis_id": "player_id", "full_name": "roster_name"}
        )
        spine = spine.groupby(keys, as_index=False, dropna=False).agg(
            roster_name=("roster_name", "last"), snap_share=("offense_pct", "max")
        )
        observed = logs[kind].copy()
        combined = spine.merge(observed, on=keys, how="outer", suffixes=("", "_observed"))
        combined["player_name"] = combined.get(
            "player_name", pd.Series(index=combined.index)
        ).fillna(combined.get("roster_name", ""))
        combined["snap_share"] = pd.to_numeric(combined.get("snap_share"), errors="coerce").fillna(
            0.0
        )
        for column in stat_columns:
            combined[column] = pd.to_numeric(combined.get(column), errors="coerce").fillna(0.0)
        combined = combined.drop(columns=["roster_name"], errors="ignore")
        if kind == "receiving":
            combined["team_targets"] = combined.groupby(["game_id", "posteam"])[
                "targets"
            ].transform("sum")
        enriched[kind] = combined
    return enriched


def _prepare_player_frames(
    logs: dict[str, pd.DataFrame], games: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    passing = add_shifted_rolling_features(
        logs["passing"],
        [
            "passing_yards",
            "attempts",
            "completions",
            "passing_tds",
            "interceptions",
            "snap_share",
        ],
    )
    passing["completion_pct_l4"] = _safe_divide(
        passing["completions_l4"], passing["attempts_l4"], 0.62
    )
    passing["snap_share_l4"] = passing["snap_share_l4"].fillna(1.0)
    passing = _attach_opponent_defense(passing, games)

    receiving = add_shifted_rolling_features(
        logs["receiving"],
        [
            "receiving_yards",
            "targets",
            "receptions",
            "receiving_tds",
            "team_targets",
            "snap_share",
        ],
    )
    receiving["catch_rate_l4"] = _safe_divide(
        receiving["receptions_l4"], receiving["targets_l4"], 0.65
    )
    receiving["yards_per_reception_l4"] = _safe_divide(
        receiving["receiving_yards_l4"], receiving["receptions_l4"], 10.0
    )
    receiving["target_share_l4"] = _safe_divide(
        receiving["targets_l4"], receiving["team_targets_l4"], 0.10
    )
    receiving["snap_share_l4"] = receiving["snap_share_l4"].fillna(receiving["target_share_l4"])
    receiving = _attach_opponent_defense(receiving, games)

    rushing_logs = logs["rushing"].copy()
    team_carries = rushing_logs.groupby("game_id")["carries"].transform("sum")
    carry_share = _safe_divide(rushing_logs["carries"], team_carries, 0.1)
    rushing_logs["snap_share"] = pd.to_numeric(
        rushing_logs.get("snap_share"), errors="coerce"
    ).fillna(carry_share)
    rushing = add_shifted_rolling_features(
        rushing_logs, ["rushing_yards", "carries", "rushing_tds", "snap_share"]
    )
    rushing["yards_per_carry_l4"] = _safe_divide(
        rushing["rushing_yards_l4"], rushing["carries_l4"], 4.1
    )
    rushing["snap_share_l4"] = rushing["snap_share_l4"].fillna(0.1)
    rushing = _attach_opponent_defense(rushing, games)
    return {"passing": passing, "receiving": receiving, "rushing": rushing}


def _append_prediction_rows(frame: pd.DataFrame, stat_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    latest_date = pd.to_datetime(frame["game_date"]).max() + pd.Timedelta(days=8)
    latest = frame.sort_values("game_date").groupby("player_id", as_index=False).tail(1).copy()
    latest["game_id"] = "PREDICTION"
    latest["game_date"] = latest_date
    latest["week"] = 99
    for column in stat_columns:
        if column in latest:
            latest[column] = np.nan
    combined = pd.concat([frame, latest], ignore_index=True, sort=False)
    return add_shifted_rolling_features(combined, stat_columns)


def _current_player_snapshots(
    logs: dict[str, pd.DataFrame],
    upcoming: pd.DataFrame,
    prediction_season: int,
    rosters: pd.DataFrame,
    team_snapshot: dict[str, dict[str, float]],
) -> dict[str, dict[str, dict[str, Any]]]:
    team_opponents: dict[str, tuple[str, float]] = {}
    for _, game in upcoming.iterrows():
        team_opponents[str(game["home_team"])] = (
            str(game["away_team"]),
            float(team_snapshot.get(str(game["away_team"]), {}).get("def_epa_l4", 0.0)),
        )
        team_opponents[str(game["away_team"])] = (
            str(game["home_team"]),
            float(team_snapshot.get(str(game["home_team"]), {}).get("def_epa_l4", 0.0)),
        )

    output: dict[str, dict[str, dict[str, Any]]] = {"qb": {}, "wr": {}, "rb": {}}
    current_roster: dict[str, dict[str, str]] = {}
    if {"season", "week", "gsis_id", "team", "status"}.issubset(rosters):
        roster = rosters[rosters["gsis_id"].fillna("").ne("")].copy()
        if not roster.empty:
            latest_roster_season = int(roster["season"].max())
            roster = roster[roster["season"].eq(latest_roster_season)]
            roster = roster.sort_values(["week"]).drop_duplicates("gsis_id", keep="last")
            roster = roster[roster["status"].isin(["ACT", "INA"])]
            current_roster = {
                str(row["gsis_id"]): {
                    "team": str(row["team"]),
                    "roster_season": str(latest_roster_season),
                }
                for _, row in roster.iterrows()
            }
    configurations = {
        "passing": (
            "qb",
            [
                "passing_yards",
                "attempts",
                "completions",
                "passing_tds",
                "interceptions",
                "snap_share",
            ],
        ),
        "receiving": (
            "wr",
            [
                "receiving_yards",
                "targets",
                "receptions",
                "receiving_tds",
                "team_targets",
                "snap_share",
            ],
        ),
        "rushing": ("rb", ["rushing_yards", "carries", "rushing_tds", "snap_share"]),
    }
    for kind, (position, stat_columns) in configurations.items():
        frame = logs[kind]
        if frame.empty:
            continue
        if kind == "rushing" and "snap_share" not in frame:
            team_carries = frame.groupby("game_id")["carries"].transform("sum")
            frame = frame.copy()
            frame["snap_share"] = _safe_divide(frame["carries"], team_carries, 0.1)
        extended = _append_prediction_rows(frame, stat_columns)
        current = extended[extended["game_id"].eq("PREDICTION")].copy()
        if kind == "passing":
            current["completion_pct_l4"] = _safe_divide(
                current["completions_l4"], current["attempts_l4"], 0.62
            )
            current["snap_share_l4"] = current["snap_share_l4"].fillna(1.0)
        elif kind == "receiving":
            current["catch_rate_l4"] = _safe_divide(
                current["receptions_l4"], current["targets_l4"], 0.65
            )
            current["yards_per_reception_l4"] = _safe_divide(
                current["receiving_yards_l4"], current["receptions_l4"], 10.0
            )
            current["target_share_l4"] = _safe_divide(
                current["targets_l4"], current["team_targets_l4"], 0.1
            )
            current["snap_share_l4"] = current["snap_share_l4"].fillna(current["target_share_l4"])
        else:
            current["yards_per_carry_l4"] = _safe_divide(
                current["rushing_yards_l4"], current["carries_l4"], 4.1
            )
            current["snap_share_l4"] = current["snap_share_l4"].fillna(0.1)

        for _, row in current.iterrows():
            player_id = str(row["player_id"])
            if current_roster and player_id not in current_roster:
                continue
            roster_record = current_roster.get(player_id, {})
            team = roster_record.get("team", str(row["posteam"]))
            opponent, opponent_def_epa = team_opponents.get(team, ("", 0.0))
            if not opponent:
                continue
            record = {
                "player_id": player_id,
                "player_name": str(row["player_name"]),
                "team": team,
                "opponent": opponent,
                "opponent_def_epa": opponent_def_epa,
                "prediction_season": prediction_season,
                "roster_season": int(roster_record["roster_season"]) if roster_record else None,
            }
            feature_set = PLAYER_FEATURES["receptions" if kind == "receiving" else kind]
            if kind == "receiving":
                feature_set = sorted(
                    set(PLAYER_FEATURES["receiving"] + PLAYER_FEATURES["receptions"])
                )
            for feature in feature_set:
                if feature in record:
                    continue
                record[feature] = (
                    float(row.get(feature, 0.0)) if pd.notna(row.get(feature)) else 0.0
                )
            output[position][player_id] = record
    return output


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _game_baselines(games: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    home_strength = games["home_points_for_l4"] - games["home_points_against_l4"]
    away_strength = games["away_points_for_l4"] - games["away_points_against_l4"]
    margin = home_strength - away_strength + 1.5 * games["home_field"]
    total = (
        games["home_points_for_l4"]
        + games["away_points_for_l4"]
        + games["home_points_against_l4"]
        + games["away_points_against_l4"]
    ) / 2.0
    return margin, total


def _train_player_models(frames: dict[str, pd.DataFrame]) -> dict[str, FittedEnsemble]:
    configurations = {
        "passing_yards": (
            frames["passing"],
            PLAYER_FEATURES["passing"],
            "passing_yards",
            "passing_yards_l4",
        ),
        "receiving_yards": (
            frames["receiving"],
            PLAYER_FEATURES["receiving"],
            "receiving_yards",
            "receiving_yards_l4",
        ),
        "receptions": (
            frames["receiving"],
            PLAYER_FEATURES["receptions"],
            "receptions",
            "receptions_l4",
        ),
        "rushing_yards": (
            frames["rushing"],
            PLAYER_FEATURES["rushing"],
            "rushing_yards",
            "rushing_yards_l4",
        ),
    }
    ensembles: dict[str, FittedEnsemble] = {}
    for name, (frame, features, target, baseline_column) in configurations.items():
        ensembles[name] = fit_ensemble(
            frame,
            name=name,
            feature_names=features,
            target_name=target,
            baseline=frame[baseline_column],
            baseline_feature=baseline_column,
            min_train_rows=350,
        )
    return ensembles


def _predict_upcoming_games(
    upcoming: pd.DataFrame,
    ensembles: dict[str, FittedEnsemble],
    data_cutoff: str,
) -> list[dict[str, Any]]:
    if upcoming.empty:
        return []
    margin_distributions = ensembles["game_margin"].distribution(upcoming)
    total_distributions = ensembles["game_total"].distribution(upcoming)
    predictions = []
    for position, (_, game) in enumerate(upcoming.reset_index(drop=True).iterrows()):
        margin = margin_distributions[position]
        total = total_distributions[position]
        predicted_home = (total["mean"] + margin["mean"]) / 2.0
        predicted_away = (total["mean"] - margin["mean"]) / 2.0
        predictions.append(
            {
                "game_id": str(game["game_id"]),
                "season": int(game["season"]),
                "week": int(game["week"]),
                "gameday": pd.Timestamp(game["gameday"]).strftime("%Y-%m-%d"),
                "gametime": str(game.get("gametime", "TBD")),
                "home_team": str(game["home_team"]),
                "away_team": str(game["away_team"]),
                "home_score": round(max(predicted_home, 0.0), 1),
                "away_score": round(max(predicted_away, 0.0), 1),
                "predicted_home_margin": round(margin["mean"], 2),
                "margin_std": round(margin["std"], 2),
                "home_win_probability": round(margin["probability_above_zero"], 4),
                "total": round(max(total["mean"], 0.0), 1),
                "total_std": round(total["std"], 2),
                "total_p10": round(max(total["p10"], 0.0), 1),
                "total_p90": round(max(total["p90"], 0.0), 1),
                "data_cutoff": data_cutoff,
                "features": {feature: float(game[feature]) for feature in GAME_FEATURES},
                "market_line": None,
                "injury_adjustments": [],
            }
        )
    return predictions


def _score_ledger(ledger: PredictionLedger, schedules: pd.DataFrame) -> dict[str, Any]:
    results_by_game = schedules.dropna(subset=["home_score", "away_score"]).set_index("game_id")
    summaries = []
    for path in sorted(ledger.root.glob("*.json")):
        if path.name.endswith(".results.json"):
            continue
        result_path = path.with_name(f"{path.stem}.results.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        scored = []
        for prediction in payload.get("predictions", []):
            game_id = prediction["game_id"]
            if game_id not in results_by_game.index:
                continue
            actual = results_by_game.loc[game_id]
            actual_margin = float(actual["home_score"] - actual["away_score"])
            actual_total = float(actual["home_score"] + actual["away_score"])
            scored.append(
                {
                    "game_id": game_id,
                    "actual_home_margin": actual_margin,
                    "actual_total": actual_total,
                    "margin_absolute_error": abs(
                        prediction["predicted_home_margin"] - actual_margin
                    ),
                    "total_absolute_error": abs(prediction["total"] - actual_total),
                    "winner_correct": (prediction["predicted_home_margin"] > 0)
                    == (actual_margin > 0),
                    **(
                        {
                            "market_margin_absolute_error": abs(
                                prediction["market_informed"]["home_margin"] - actual_margin
                            ),
                            "market_total_absolute_error": abs(
                                prediction["market_informed"]["total"] - actual_total
                            ),
                            "model_market_margin_difference": prediction["predicted_home_margin"]
                            - prediction["market_informed"]["home_margin"],
                        }
                        if prediction.get("market_informed")
                        else {}
                    ),
                }
            )
        if scored and not result_path.exists():
            ledger.score_batch(payload["run_id"], scored)
        if scored:
            market_rows = [row for row in scored if "market_margin_absolute_error" in row]
            summaries.append(
                {
                    "run_id": payload["run_id"],
                    "games": len(scored),
                    "margin_mae": float(np.mean([row["margin_absolute_error"] for row in scored])),
                    "total_mae": float(np.mean([row["total_absolute_error"] for row in scored])),
                    "winner_accuracy": float(np.mean([row["winner_correct"] for row in scored])),
                    "market_games": len(market_rows),
                    "market_margin_mae": (
                        float(np.mean([row["market_margin_absolute_error"] for row in market_rows]))
                        if market_rows
                        else None
                    ),
                    "market_total_mae": (
                        float(np.mean([row["market_total_absolute_error"] for row in market_rows]))
                        if market_rows
                        else None
                    ),
                }
            )
    return {"runs": summaries, "updated_at": datetime.now(UTC).isoformat()}


def _official_injury_payload(
    injuries: pd.DataFrame, prediction_season: int, generated_at: datetime
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source": "nflverse injury reports",
        "generated_at": generated_at.isoformat(),
        "prediction_season": prediction_season,
        "available_season": None,
        "available_week": None,
        "stale_for_prediction_season": True,
        "entries": [],
    }
    if injuries.empty or not {"season", "week"}.issubset(injuries):
        return payload
    season = int(injuries["season"].max())
    current = injuries[injuries["season"].eq(season)].copy()
    week = int(current["week"].max())
    current = current[current["week"].eq(week)]
    fields = [
        "team",
        "gsis_id",
        "position",
        "full_name",
        "report_primary_injury",
        "report_secondary_injury",
        "report_status",
        "practice_primary_injury",
        "practice_secondary_injury",
        "practice_status",
    ]
    fields = [field for field in fields if field in current]
    current = current[
        current.get("report_status", pd.Series(index=current.index, dtype=object)).notna()
        | current.get("practice_status", pd.Series(index=current.index, dtype=object)).notna()
    ]
    payload.update(
        {
            "available_season": season,
            "available_week": week,
            "stale_for_prediction_season": season != prediction_season,
            "entries": current[fields].where(pd.notna(current[fields]), None).to_dict("records"),
        }
    )
    return payload


def run_update(as_of: datetime | None = None) -> UpdateResult:
    now = as_of or datetime.now(UTC)
    context = get_season_context(now)
    data = load_nflverse_data(list(context.training_seasons))
    raw_data_hash = _raw_data_fingerprint(data)
    game_result = build_point_in_time_game_features(data.schedules, data.pbp, include_unplayed=True)
    games = game_result.games
    completed = games.dropna(subset=["home_margin", "total_points"]).copy()
    margin_baseline, total_baseline = _game_baselines(completed)
    ensembles = {
        "game_margin": fit_ensemble(
            completed,
            name="game_margin",
            feature_names=GAME_FEATURES,
            target_name="home_margin",
            baseline=margin_baseline,
            min_train_rows=350,
            ridge_alpha=GAME_RIDGE_ALPHA,
        ),
        "game_total": fit_ensemble(
            completed,
            name="game_total",
            feature_names=GAME_FEATURES,
            target_name="total_points",
            baseline=total_baseline,
            min_train_rows=350,
            ridge_alpha=GAME_RIDGE_ALPHA,
        ),
    }
    raw_player_logs = build_player_game_logs(data.pbp)
    raw_player_logs = _add_participation_spines(
        raw_player_logs, data.snap_counts, data.rosters, data.pbp
    )
    player_frames = _prepare_player_frames(raw_player_logs, games)
    ensembles.update(_train_player_models(player_frames))

    cutoff = pd.to_datetime(completed.get("gameday"), errors="coerce").max()
    cutoff_text = cutoff.strftime("%Y-%m-%d") if pd.notna(cutoff) else now.date().isoformat()
    actual_training_seasons = sorted(int(season) for season in completed["season"].unique())
    manifest = save_model_bundle(
        ensembles,
        prediction_season=context.prediction_season,
        training_seasons=actual_training_seasons,
        data_cutoff=cutoff_text,
        raw_data_hash=raw_data_hash,
        git_commit=_git_commit(),
    )

    unplayed = games[games["home_score"].isna()].copy()
    current_season = unplayed[unplayed["season"].eq(context.prediction_season)]
    if current_season.empty:
        upcoming = current_season
    else:
        next_week = int(current_season["week"].min())
        upcoming = current_season[current_season["week"].eq(next_week)]
    predictions = _predict_upcoming_games(upcoming, ensembles, cutoff_text)
    predictions = attach_market_consensus(predictions, load_market_consensus(), as_of=now)

    ledger = PredictionLedger()
    ledger_path = ledger.record_batch(
        predictions,
        model_hash=sha256_file(MODEL_MANIFEST_PATH),
        data_cutoff=cutoff_text,
        prediction_season=context.prediction_season,
        metadata={
            "git_commit": _git_commit(),
            "week": int(upcoming["week"].min()) if not upcoming.empty else None,
            "market_snapshot_at": next(
                (
                    prediction["market_consensus"]["snapshot_at"]
                    for prediction in predictions
                    if prediction.get("market_consensus")
                ),
                None,
            ),
        },
    )
    performance = _score_ledger(ledger, data.schedules)

    player_snapshots = _current_player_snapshots(
        raw_player_logs,
        upcoming,
        context.prediction_season,
        data.rosters,
        game_result.team_snapshot,
    )
    team_payload = {
        team: {**state, "prediction_season": context.prediction_season, "data_cutoff": cutoff_text}
        for team, state in game_result.team_snapshot.items()
    }
    atomic_write_json(PROJECT_ROOT / "team_data.json", team_payload)
    atomic_write_json(PROJECT_ROOT / "qb_data.json", player_snapshots["qb"])
    atomic_write_json(PROJECT_ROOT / "wr_data.json", player_snapshots["wr"])
    atomic_write_json(PROJECT_ROOT / "rb_data.json", player_snapshots["rb"])
    atomic_write_json(PROJECT_ROOT / "weekly_schedule.json", predictions)
    atomic_write_json(
        PROJECT_ROOT / "weekly_report.json",
        {
            "prediction_season": context.prediction_season,
            "week": int(upcoming["week"].min()) if not upcoming.empty else None,
            "generated_at": now.isoformat(),
            "data_cutoff": cutoff_text,
            "model_hash": sha256_file(MODEL_MANIFEST_PATH),
            "raw_data_hash": raw_data_hash,
            "predictions": predictions,
        },
    )
    atomic_write_json(PROJECT_ROOT / "performance_history.json", performance)
    atomic_write_json(
        PROJECT_ROOT / "official_injuries.json",
        _official_injury_payload(data.injuries, context.prediction_season, now),
    )
    atomic_write_json(
        PROJECT_ROOT / "update_log.json",
        {
            "last_updated": now.isoformat(),
            "prediction_season": context.prediction_season,
            "training_seasons": actual_training_seasons,
            "data_cutoff": cutoff_text,
            "model_manifest": str(MODEL_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
            "model_hash": sha256_file(MODEL_MANIFEST_PATH),
            "raw_data_hash": raw_data_hash,
            "ledger_run": ledger_path.stem,
            "metrics": {name: model.metrics for name, model in ensembles.items()},
        },
    )
    return UpdateResult(
        manifest=manifest,
        predictions=predictions,
        ledger_path=ledger_path,
        metrics={name: model.metrics for name, model in ensembles.items()},
    )
