from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from injury_system import (
    InjuryAdjustmentSystem,
    integrate_injuries_into_game_prediction,
    integrate_injuries_into_player_prediction,
    render_injury_manager,
)
from nfl_prediction.config import MODEL_MANIFEST_PATH, PROJECT_ROOT, is_division_game
from nfl_prediction.features import GAME_FEATURES
from nfl_prediction.io import read_json
from nfl_prediction.market import (
    american_odds_to_implied_probability,
    home_cover_probability,
    no_vig_probabilities,
    over_probability,
)
from nfl_prediction.modeling import FittedEnsemble, load_model_bundle
from nfl_prediction.odds import attach_market_consensus

LOGGER = logging.getLogger(__name__)

st.set_page_config(page_title="NFL Prediction System", page_icon="🏈", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #111827; color: #f9fafb; }
    h1, h2, h3 { color: #f9fafb; }
    [data-testid="stMetric"] { background: #1f2937; padding: 0.8rem; border-radius: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models() -> tuple[dict[str, FittedEnsemble], dict[str, Any]]:
    return load_model_bundle(MODEL_MANIFEST_PATH)


@st.cache_data
def load_state() -> dict[str, Any]:
    market = read_json(PROJECT_ROOT / "market_consensus.json")
    schedule = attach_market_consensus(read_json(PROJECT_ROOT / "weekly_schedule.json", []), market)
    return {
        "teams": read_json(PROJECT_ROOT / "team_data.json", {}),
        "qb": read_json(PROJECT_ROOT / "qb_data.json", {}),
        "wr": read_json(PROJECT_ROOT / "wr_data.json", {}),
        "rb": read_json(PROJECT_ROOT / "rb_data.json", {}),
        "schedule": schedule,
        "report": read_json(PROJECT_ROOT / "weekly_report.json", {}),
        "performance": read_json(PROJECT_ROOT / "performance_history.json", {"runs": []}),
        "official_injuries": read_json(PROJECT_ROOT / "official_injuries.json", {}),
        "update": read_json(PROJECT_ROOT / "update_log.json", {}),
        "market": market,
    }


class PredictionService:
    def __init__(
        self,
        models: dict[str, FittedEnsemble],
        manifest: dict[str, Any],
        state: dict[str, Any],
    ):
        self.models = models
        self.manifest = manifest
        self.state = state
        self.team_data = state["teams"]

    def teams(self) -> list[str]:
        return sorted(self.team_data)

    def _game_features(
        self,
        away_team: str,
        home_team: str,
        *,
        away_rest_days: int,
        home_rest_days: int,
        neutral_site: bool,
        week: int,
    ) -> pd.DataFrame:
        if away_team not in self.team_data or home_team not in self.team_data:
            raise ValueError("Current team state is unavailable. Run the updater.")
        home = self.team_data[home_team]
        away = self.team_data[away_team]
        record: dict[str, float] = {
            "home_rest_days": float(home_rest_days),
            "away_rest_days": float(away_rest_days),
            "rest_advantage": float(home_rest_days - away_rest_days),
            "division_game": float(is_division_game(home_team, away_team)),
            "week": float(week),
            "home_field": 0.0 if neutral_site else 1.0,
        }
        for feature in GAME_FEATURES:
            if feature.startswith("home_") and feature not in record:
                record[feature] = float(home[feature.removeprefix("home_")])
            elif feature.startswith("away_") and feature not in record:
                record[feature] = float(away[feature.removeprefix("away_")])
        return pd.DataFrame([record], columns=GAME_FEATURES)

    def predict_game(
        self,
        away_team: str,
        home_team: str,
        *,
        away_rest_days: int = 7,
        home_rest_days: int = 7,
        neutral_site: bool = False,
        week: int = 1,
    ) -> dict[str, Any]:
        features = self._game_features(
            away_team,
            home_team,
            away_rest_days=away_rest_days,
            home_rest_days=home_rest_days,
            neutral_site=neutral_site,
            week=week,
        )
        margin = self.models["game_margin"].distribution(features)[0]
        total = self.models["game_total"].distribution(features)[0]
        home_score = max((total["mean"] + margin["mean"]) / 2.0, 0.0)
        away_score = max((total["mean"] - margin["mean"]) / 2.0, 0.0)
        prediction = {
            "team1": away_team,
            "team1_score": round(away_score, 1),
            "team2": home_team,
            "team2_score": round(home_score, 1),
            "away_team": away_team,
            "home_team": home_team,
            "away_score": round(away_score, 1),
            "home_score": round(home_score, 1),
            "predicted_home_margin": round(margin["mean"], 2),
            "spread": round(margin["mean"], 2),
            "margin_std": round(margin["std"], 2),
            "home_win_probability": margin["probability_above_zero"],
            "total": round(total["mean"], 1),
            "total_std": round(total["std"], 2),
            "total_p10": round(max(total["p10"], 0.0), 1),
            "total_p90": round(max(total["p90"], 0.0), 1),
            "method": "schema-v3 walk-forward ensemble",
        }
        return attach_market_consensus([prediction], self.state.get("market"))[0]

    def predict_player(self, model_name: str, player: dict[str, Any]) -> dict[str, float]:
        distribution = self.models[model_name].distribution(pd.DataFrame([player]))[0]
        distribution["mean"] = max(distribution["mean"], 0.0)
        distribution["p10"] = max(distribution["p10"], 0.0)
        return distribution


def render_market_comparison(game: dict[str, Any], key: str) -> None:
    st.markdown("**Paper-market comparison**")
    consensus = game.get("market_consensus") or {}
    consensus_spread = consensus.get("spread") or {}
    consensus_total = consensus.get("total") or {}
    if consensus:
        st.caption(
            f"{consensus.get('provider', 'Market')} consensus captured "
            f"{consensus['snapshot_at'][:16].replace('T', ' ')} UTC · "
            f"{consensus_spread.get('book_count', 0)} spread books · "
            f"spread range {consensus_spread.get('line_min', 0):+.1f} to "
            f"{consensus_spread.get('line_max', 0):+.1f}"
        )
    home_line = st.number_input(
        f"Sportsbook home spread ({game['home_team']}; favorite is negative)",
        value=float(consensus_spread.get("home_spread", 0.0)),
        step=0.5,
        key=f"spread_{key}",
    )
    market_total = st.number_input(
        "Sportsbook total",
        value=float(consensus_total.get("total", game.get("total", 44.0))),
        step=0.5,
        key=f"total_{key}",
    )
    price_col1, price_col2, price_col3, price_col4 = st.columns(4)
    home_spread_price = price_col1.number_input(
        "Home spread price",
        value=float(consensus_spread.get("home_price", -110)),
        step=1.0,
        key=f"home_spread_price_{key}",
    )
    away_spread_price = price_col2.number_input(
        "Away spread price",
        value=float(consensus_spread.get("away_price", -110)),
        step=1.0,
        key=f"away_spread_price_{key}",
    )
    over_price = price_col3.number_input(
        "Over price",
        value=float(consensus_total.get("over_price", -110)),
        step=1.0,
        key=f"over_price_{key}",
    )
    under_price = price_col4.number_input(
        "Under price",
        value=float(consensus_total.get("under_price", -110)),
        step=1.0,
        key=f"under_price_{key}",
    )
    observed_at = st.text_input(
        "Line observed at (ET)",
        value=datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M"),
        key=f"line_time_{key}",
    )
    if 0 in (home_spread_price, away_spread_price, over_price, under_price):
        st.error("American odds cannot be zero. Enter a positive or negative price.")
        return
    predicted_margin = float(game["predicted_home_margin"])
    margin_std = max(float(game["margin_std"]), 0.01)
    home_cover = home_cover_probability(predicted_margin, float(home_line), margin_std)
    if home_cover >= 0.5:
        spread_side = f"{game['home_team']} {home_line:+.1f}"
        spread_probability = home_cover
        spread_price = home_spread_price
        spread_fair, _ = no_vig_probabilities(home_spread_price, away_spread_price)
    else:
        spread_side = f"{game['away_team']} {-home_line:+.1f}"
        spread_probability = 1.0 - home_cover
        spread_price = away_spread_price
        _, spread_fair = no_vig_probabilities(home_spread_price, away_spread_price)

    total_std = max(float(game["total_std"]), 0.01)
    over = over_probability(float(game["total"]), market_total, total_std)
    total_side = "Over" if over >= 0.5 else "Under"
    total_probability = max(over, 1.0 - over)
    if over >= 0.5:
        total_price = over_price
        total_fair, _ = no_vig_probabilities(over_price, under_price)
    else:
        total_price = under_price
        _, total_fair = no_vig_probabilities(over_price, under_price)
    spread_implied = american_odds_to_implied_probability(spread_price)
    total_implied = american_odds_to_implied_probability(total_price)
    col1, col2 = st.columns(2)
    col1.metric(
        "Model spread side",
        spread_side,
        f"{spread_probability - spread_fair:+.1%} vs no-vig",
    )
    col2.metric(
        "Model total side",
        f"{total_side} {market_total:.1f}",
        f"{total_probability - total_fair:+.1%} vs no-vig",
    )
    st.caption(
        f"Observed {observed_at} ET. Model probabilities: spread {spread_probability:.1%}, "
        f"total {total_probability:.1%}; raw implied: spread {spread_implied:.1%}, "
        f"total {total_implied:.1%}; no-vig: spread {spread_fair:.1%}, "
        f"total {total_fair:.1%}. Paper analysis only."
    )


st.title("🏈 NFL Prediction System")

try:
    models, manifest = load_models()
except Exception as exc:
    LOGGER.exception("Model bundle failed to load")
    st.error(f"Model bundle unavailable: {exc}")
    st.info("Run `python weekly_nfl_update.py` to create a validated schema-v3 model bundle.")
    st.stop()

state = load_state()
service = PredictionService(models, manifest, state)
injury_system = InjuryAdjustmentSystem(persist=False)

st.caption(
    f"Season {manifest['prediction_season']} · data through {manifest['data_cutoff']} · "
    f"model created {manifest['created_at'][:10]}"
)

render_injury_manager(injury_system, service.teams())
official_injuries = state["official_injuries"]
if official_injuries.get("stale_for_prediction_season"):
    st.sidebar.caption(
        "Official injury reports are not yet published for the prediction season; "
        "manual scenarios are session-only."
    )
elif official_injuries.get("entries"):
    st.sidebar.caption(
        f"Official injury feed: Week {official_injuries['available_week']} "
        f"({len(official_injuries['entries'])} reported players)."
    )

page = st.sidebar.selectbox(
    "Analysis",
    ["This Week", "Custom Game", "Player Props", "Power Rankings", "Performance", "Model Card"],
)

if page == "This Week":
    st.header("This Week")
    schedule = state["schedule"]
    if not schedule:
        st.info("No upcoming games are available for the current prediction season.")
    for game in schedule:
        with st.expander(
            f"{game['away_team']} @ {game['home_team']} — {game['gameday']} {game.get('gametime', '')}",
            expanded=True,
        ):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(game["away_team"], f"{game['away_score']:.1f}")
            col2.metric(game["home_team"], f"{game['home_score']:.1f}")
            col3.metric("Home win", f"{game['home_win_probability']:.1%}")
            col4.metric("Total", f"{game['total']:.1f}")
            st.caption(
                f"Predicted home margin: {game['predicted_home_margin']:+.1f} · "
                f"80% total interval: {game['total_p10']:.1f}–{game['total_p90']:.1f}"
            )
            market_forecast = game.get("market_informed")
            if market_forecast:
                st.info(
                    f"Market benchmark: {game['away_team']} {market_forecast['away_score']:.1f}, "
                    f"{game['home_team']} {market_forecast['home_score']:.1f} "
                    f"(home margin {market_forecast['home_margin']:+.1f}, "
                    f"total {market_forecast['total']:.1f})."
                )
            render_market_comparison(game, game["game_id"])

elif page == "Custom Game":
    st.header("Custom Game")
    teams = service.teams()
    col1, col2 = st.columns(2)
    away_team = col1.selectbox("Away team", teams)
    home_team = col2.selectbox("Home team", teams, index=1 if len(teams) > 1 else 0)
    col3, col4, col5 = st.columns(3)
    away_rest = col3.number_input("Away rest days", min_value=3, max_value=21, value=7)
    home_rest = col4.number_input("Home rest days", min_value=3, max_value=21, value=7)
    neutral = col5.checkbox("Neutral site")
    week = st.number_input("Week", min_value=1, max_value=18, value=1)
    if st.button("Predict game", type="primary"):
        if away_team == home_team:
            st.error("Select two different teams.")
        else:
            base = service.predict_game(
                away_team,
                home_team,
                away_rest_days=int(away_rest),
                home_rest_days=int(home_rest),
                neutral_site=neutral,
                week=int(week),
            )
            prediction = integrate_injuries_into_game_prediction(
                base, injury_system, away_team, home_team, home_team
            )
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(away_team, f"{prediction['away_score']:.1f}")
            col2.metric(home_team, f"{prediction['home_score']:.1f}")
            adjusted_home_win = home_cover_probability(
                float(prediction["predicted_home_margin"]), 0.0, float(base["margin_std"])
            )
            col3.metric("Home win", f"{adjusted_home_win:.1%}")
            col4.metric("Total", f"{prediction['total']:.1f}")
            st.caption(
                f"Home margin: {prediction['predicted_home_margin']:+.1f} · "
                f"model σ {base['margin_std']:.1f} points"
            )
            if prediction.get("injury_adjusted"):
                st.warning(prediction.get("adjustment_note", "Manual injury adjustment applied"))
            render_market_comparison({**base, **prediction}, "custom")

elif page == "Player Props":
    st.header("Player Props")
    prop = st.radio(
        "Prop",
        ["Passing yards", "Receiving yards", "Receptions", "Rushing yards"],
        horizontal=True,
    )
    mapping = {
        "Passing yards": ("qb", "passing_yards"),
        "Receiving yards": ("wr", "receiving_yards"),
        "Receptions": ("wr", "receptions"),
        "Rushing yards": ("rb", "rushing_yards"),
    }
    state_key, model_name = mapping[prop]
    players = state[state_key]
    roster_seasons = {
        int(player["roster_season"])
        for player in players.values()
        if player.get("roster_season") is not None
    }
    if roster_seasons and max(roster_seasons) < int(manifest["prediction_season"]):
        st.warning(
            f"The newest published weekly roster is {max(roster_seasons)}. "
            "Confirm current membership and roles before using preseason prop forecasts."
        )
    choices = sorted(
        players, key=lambda player_id: players[player_id].get("player_name", player_id)
    )
    if not choices:
        st.info("No current player records are available.")
    else:
        player_id = st.selectbox(
            "Player",
            choices,
            format_func=lambda value: (
                f"{players[value].get('player_name', value)} — {players[value].get('team', '')} "
                f"vs {players[value].get('opponent') or 'TBD'}"
            ),
        )
        line = st.number_input("Sportsbook prop line", value=0.0, step=0.5)
        if st.button("Predict prop", type="primary"):
            player = players[player_id]
            distribution = service.predict_player(model_name, player)
            adjusted, note = integrate_injuries_into_player_prediction(
                distribution["mean"],
                injury_system,
                player["player_name"],
                player["team"],
                model_name,
            )
            probability_over = over_probability(adjusted, line, max(distribution["std"], 0.01))
            col1, col2, col3 = st.columns(3)
            col1.metric("Projection", f"{adjusted:.1f}")
            col2.metric("80% interval", f"{distribution['p10']:.1f}–{distribution['p90']:.1f}")
            col3.metric("Over probability", f"{probability_over:.1%}")
            if note != "Healthy":
                st.warning(note)
            st.caption(
                "The interval reflects model residuals; availability uncertainty is shown separately."
            )

elif page == "Power Rankings":
    st.header("Power Rankings")
    rows = []
    for team, values in state["teams"].items():
        rating = (
            values["points_for_l4"]
            - values["points_against_l4"]
            + 8.0 * (values["off_epa_l4"] - values["def_epa_l4"])
            + 3.0 * (values["pressure_generated_l4"] - values["pressure_allowed_l4"])
        )
        rows.append({"Team": team, "Neutral-field rating": rating})
    ranking = pd.DataFrame(rows).sort_values("Neutral-field rating", ascending=False)
    ranking.insert(0, "Rank", range(1, len(ranking) + 1))
    st.dataframe(ranking, hide_index=True, use_container_width=True)
    st.caption("Descriptive current-state rating, not a separately trained betting model.")

elif page == "Performance":
    st.header("Immutable Prediction Performance")
    runs = state["performance"].get("runs", [])
    if not runs:
        st.info("No completed immutable prediction batches have been scored yet.")
    else:
        st.dataframe(pd.DataFrame(runs), hide_index=True, use_container_width=True)

else:
    st.header("Model Card")
    st.json(
        {
            "prediction_season": manifest["prediction_season"],
            "training_seasons": manifest["training_seasons"],
            "data_cutoff": manifest["data_cutoff"],
            "raw_data_hash": manifest.get("raw_data_hash"),
            "git_commit": manifest.get("git_commit"),
            "libraries": manifest["libraries"],
        }
    )
    for name, specification in manifest["models"].items():
        with st.expander(name, expanded=True):
            st.write("Features:", ", ".join(specification["features"]))
            st.json(specification["metrics"])
    st.warning(
        "Forecasts are estimates with uncertainty. Market comparisons are paper analysis, not financial advice."
    )
