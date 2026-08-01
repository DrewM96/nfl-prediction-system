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
from nfl_prediction.io import read_json
from nfl_prediction.market import (
    american_odds_to_implied_probability,
    home_cover_probability,
    no_vig_probabilities,
    over_probability,
)
from nfl_prediction.modeling import FittedEnsemble, load_model_bundle
from nfl_prediction.odds import attach_market_consensus
from nfl_prediction.roster import decay_roster_feature
from nfl_prediction.ui import (
    american_moneyline,
    format_american,
    format_game_time,
    format_probability,
    game_reasoning,
    html_text,
    market_line_label,
    relative_mae_improvement,
    spread_label,
    team_color,
    team_name,
)

LOGGER = logging.getLogger(__name__)
PAGE_LABELS = [
    "This Week",
    "Builder",
    "Props",
    "Rankings",
    "Performance",
    "Model",
]

st.set_page_config(
    page_title="GRIDLINE — Model-Based Forecasts",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Albert+Sans:wght@400;500;600&family=Instrument+Sans:wght@500;600;700&display=swap');
    :root {
      --grid-orange: #FF6B35;
      --grid-ink: #0f1419;
      --grid-body: #374151;
      --grid-muted: #64748b;
      --grid-faint: #94a3b8;
      --grid-border: #e5e7eb;
      --grid-recessed: #f3f4f6;
    }
    html, body, [class*="css"], .stApp {
      font-family: 'Albert Sans', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif;
      color: var(--grid-ink);
    }
    .stApp { background: #ffffff; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { top: 0.35rem; }
    .block-container {
      max-width: 1120px;
      padding: 0 2rem 7rem;
    }
    h1, h2, h3, .grid-display {
      font-family: 'Instrument Sans', sans-serif;
      letter-spacing: -0.01em;
    }
    .grid-topbar {
      position: sticky;
      top: 0;
      z-index: 50;
      width: 100vw;
      margin-left: calc(50% - 50vw);
      padding: 18px 32px;
      background: rgba(255,255,255,.97);
      border-bottom: 1px solid var(--grid-border);
    }
    .grid-topbar-inner {
      max-width: 1120px;
      margin: 0 auto;
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 20px;
    }
    .grid-brand { display: flex; align-items: baseline; gap: 12px; }
    .grid-wordmark { font: 500 20px 'Instrument Sans', sans-serif; letter-spacing: .3px; }
    .grid-eyebrow, .grid-kicker {
      font: 600 11px 'Instrument Sans', sans-serif;
      color: var(--grid-faint);
      letter-spacing: .6px;
      text-transform: uppercase;
    }
    .grid-meta { color: var(--grid-muted); font-size: 13px; text-align: right; }
    .grid-page-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding: 28px 0 20px;
    }
    .grid-page-title { margin: 0; font: 500 26px 'Instrument Sans', sans-serif; }
    .grid-badge {
      padding: 6px 12px;
      border-radius: 20px;
      border: 1px solid var(--grid-border);
      background: var(--grid-recessed);
      color: var(--grid-faint);
      font-size: 12px;
      white-space: nowrap;
    }
    .grid-hero {
      padding: 24px 28px;
      margin-bottom: 20px;
      border: 1px solid var(--grid-border);
      border-radius: 16px;
      background: linear-gradient(135deg,#f8f9fb,#ffffff);
    }
    .grid-hero-main {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
      flex-wrap: wrap;
    }
    .grid-matchup { display: flex; align-items: center; gap: 18px; }
    .grid-team { text-align: center; min-width: 44px; }
    .grid-team-chip { width: 14px; height: 44px; border-radius: 4px; margin: 0 auto 8px; }
    .grid-team-abbr { font: 500 22px 'Instrument Sans', sans-serif; }
    .grid-score { color: var(--grid-muted); font-size: 13px; }
    .grid-at { color: var(--grid-faint); font-weight: 600; }
    .grid-date { margin-left: 8px; color: var(--grid-faint); font-size: 13px; }
    .grid-tiles { display: grid; grid-template-columns: repeat(4,minmax(88px,1fr)); gap: 12px; }
    .grid-tile {
      min-width: 88px;
      padding: 10px 14px;
      text-align: center;
      background: var(--grid-recessed);
      border: 1px solid var(--grid-border);
      border-radius: 10px;
    }
    .grid-tile.dashed { border-style: dashed; border-color: #cbd5e1; }
    .grid-tile-label, .grid-mini-label {
      color: var(--grid-faint);
      font: 500 9px 'Instrument Sans', sans-serif;
      letter-spacing: .5px;
      text-transform: uppercase;
    }
    .grid-tile-value { margin-top: 2px; font: 500 19px 'Instrument Sans', sans-serif; }
    .grid-tile-value.pending { color: #9ca3af; font-size: 12px; }
    .grid-probability { margin-top: 18px; }
    .grid-prob-labels { display: flex; justify-content: space-between; color: var(--grid-muted); font-size: 11px; margin-bottom: 5px; }
    .grid-prob-track { display: flex; height: 8px; overflow: hidden; border-radius: 4px; background: var(--grid-border); }
    .grid-prob-away { height: 100%; background: #94a3b8; }
    .grid-prob-home { height: 100%; background: var(--grid-orange); }
    .grid-team-inline { display: flex; align-items: center; gap: 9px; }
    .grid-team-rail { width: 6px; height: 34px; border-radius: 2px; flex: 0 0 auto; }
    .grid-team-pair { display: flex; align-items: center; gap: 8px; }
    .grid-row-value { text-align: center; font-size: 14px; font-weight: 600; }
    .grid-row-date { color: var(--grid-muted); font-size: 12px; }
    .grid-reason { display: flex; gap: 8px; padding: 5px 0; color: var(--grid-body); font-size: 13px; }
    .grid-reason > span { color: var(--grid-orange); }
    .grid-placeholder {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      padding: 9px 14px;
      border: 1px dashed #cbd5e1;
      border-radius: 8px;
      color: #9ca3af;
      font-size: 12px;
    }
    .grid-card {
      padding: 22px 24px;
      margin-bottom: 16px;
      border: 1px solid var(--grid-border);
      border-radius: 14px;
      background: #fff;
    }
    .grid-results { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; }
    .grid-result {
      padding: 12px;
      text-align: center;
      background: var(--grid-recessed);
      border: 1px solid var(--grid-border);
      border-radius: 10px;
    }
    .grid-result-value { margin-top: 2px; font: 500 24px 'Instrument Sans', sans-serif; }
    .grid-market-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .grid-market-card {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 14px 18px;
      background: var(--grid-recessed);
      border: 1px solid var(--grid-border);
      border-radius: 10px;
    }
    .grid-market-value { font: 500 17px 'Instrument Sans', sans-serif; }
    .grid-edge-spread { color: #16a34a; font-weight: 600; font-size: 14px; }
    .grid-edge-total { color: #0284c7; font-weight: 600; font-size: 14px; }
    .grid-player-chip {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      padding: 10px 16px;
      margin: 4px 0 18px;
      border: 1px solid var(--grid-orange);
      border-radius: 10px;
      background: #fff4ee;
    }
    .grid-prop-panel { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; align-items: center; padding-top: 18px; border-top: 1px solid var(--grid-border); }
    .grid-projection { font: 500 40px 'Instrument Sans', sans-serif; }
    .grid-prop-bar { display: flex; height: 14px; overflow: hidden; border-radius: 7px; background: var(--grid-border); }
    .grid-under { background: #dc2626; }
    .grid-over { background: #16a34a; }
    .grid-rank-row { display: grid; grid-template-columns: 34px 72px 1fr 52px; align-items: center; gap: 12px; padding: 10px 0; border-bottom: 1px solid #f1f3f5; }
    .grid-rank-row:last-child { border-bottom: none; }
    .grid-rank { color: var(--grid-faint); font-size: 14px; }
    .grid-rank-team { display: flex; align-items: center; gap: 9px; font-weight: 600; }
    .grid-rank-chip { width: 5px; height: 20px; border-radius: 2px; }
    .grid-rank-track { height: 10px; overflow: hidden; border-radius: 5px; background: #f1f3f5; }
    .grid-rank-fill { height: 100%; border-radius: 5px; }
    .grid-rank-value { text-align: right; font: 600 13px 'Instrument Sans', sans-serif; }
    .grid-model-summary { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 20px; }
    .grid-model-card { padding: 18px 20px; margin-bottom: 10px; border: 1px solid var(--grid-border); border-radius: 12px; }
    .grid-model-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 12px; }
    .grid-model-metrics { display: grid; grid-template-columns: repeat(5,1fr); gap: 10px; }
    .grid-model-metric { padding: 10px; text-align: center; background: var(--grid-recessed); border: 1px solid var(--grid-border); border-radius: 8px; }
    .grid-model-value { margin-top: 2px; font: 500 16px 'Instrument Sans', sans-serif; }
    .grid-ghost-chart { opacity: .35; pointer-events: none; padding-top: 14px; }
    .grid-ghost-row { display: grid; grid-template-columns: 54px 1fr; align-items: center; gap: 12px; margin: 12px 0; color: var(--grid-muted); font-size: 11px; }
    .grid-ghost-bar { height: 14px; border-radius: 7px; background: #cbd5e1; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
      border-color: var(--grid-border);
      border-radius: 12px;
      box-shadow: none;
    }
    div[data-testid="stButton"] button, div[data-testid="stFormSubmitButton"] button {
      border-radius: 8px;
      border-color: var(--grid-border);
      font-family: 'Instrument Sans', sans-serif;
      font-weight: 600;
    }
    div[data-testid="stFormSubmitButton"] button[kind="primary"], button[kind="primary"] {
      background: var(--grid-orange);
      border-color: var(--grid-orange);
      color: white;
    }
    div[data-testid="stRadio"] [role="radiogroup"] { gap: 8px; }
    div[data-testid="stRadio"] label {
      padding: 7px 12px;
      border: 1px solid var(--grid-border);
      border-radius: 20px;
      background: #fff;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
      border-color: var(--grid-orange);
      background: var(--grid-orange);
      color: white;
    }
    div[data-testid="stRadio"] label > div:first-child { display: none; }
    .st-key-bottom_nav {
      position: fixed;
      z-index: 1000;
      left: 0;
      right: 0;
      bottom: 0;
      padding: 7px max(12px, calc((100vw - 1120px)/2));
      background: rgba(255,255,255,.98);
      border-top: 1px solid var(--grid-border);
    }
    .st-key-bottom_nav [role="radiogroup"] { display: grid; grid-template-columns: repeat(6,1fr); width: 100%; gap: 0 !important; }
    .st-key-bottom_nav label { justify-content: center; border: 0 !important; border-radius: 0 !important; background: transparent !important; color: var(--grid-faint); font-size: 11px; }
    .st-key-bottom_nav label:has(input:checked) { color: var(--grid-ink); }
    .st-key-bottom_nav label:has(input:checked)::before { content: ''; width: 6px; height: 6px; margin-right: 7px; border-radius: 50%; background: var(--grid-orange); }
    [data-testid="stSidebar"] { background: #f8f9fb; border-right: 1px solid var(--grid-border); }
    .grid-muted { color: var(--grid-faint); font-size: 12px; }
    .grid-positive { color: #16a34a; }
    @media (max-width: 760px) {
      .block-container { padding-left: 1rem; padding-right: 1rem; }
      .grid-topbar { padding: 14px 16px; }
      .grid-topbar-inner { align-items: flex-start; }
      .grid-eyebrow, .grid-meta { display: none; }
      .grid-page-head { padding-top: 20px; align-items: flex-start; }
      .grid-badge { white-space: normal; text-align: right; }
      .grid-hero { padding: 20px; }
      .grid-tiles, .grid-results, .grid-model-summary { grid-template-columns: repeat(2,1fr); }
      .grid-market-grid, .grid-prop-panel { grid-template-columns: 1fr; }
      .grid-model-metrics { grid-template-columns: repeat(2,1fr); }
      .grid-rank-row { grid-template-columns: 26px 64px 1fr 44px; gap: 8px; }
      .st-key-bottom_nav label { padding: 8px 2px !important; font-size: 9px; }
    }
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
        "market_benchmark": read_json(PROJECT_ROOT / "market_benchmark.json"),
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
        feature_names = list(
            dict.fromkeys(
                feature
                for model_name in ("game_margin", "game_total")
                for feature in self.models[model_name].feature_names
            )
        )
        for feature in feature_names:
            if feature.startswith("home_") and feature not in record:
                record[feature] = decay_roster_feature(
                    feature,
                    float(home[feature.removeprefix("home_")]),
                    week,
                )
            elif feature.startswith("away_") and feature not in record:
                record[feature] = decay_roster_feature(
                    feature,
                    float(away[feature.removeprefix("away_")]),
                    week,
                )
        return pd.DataFrame([record], columns=feature_names)

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


def page_header(title: str, badge: str | None = None) -> None:
    badge_html = f'<div class="grid-badge">{html_text(badge)}</div>' if badge else ""
    st.markdown(
        f'<div class="grid-page-head"><h1 class="grid-page-title">{html_text(title)}</h1>{badge_html}</div>',
        unsafe_allow_html=True,
    )


def top_bar(manifest: dict[str, Any]) -> None:
    st.markdown(
        f"""
        <div class="grid-topbar"><div class="grid-topbar-inner">
          <div class="grid-brand">
            <span class="grid-wordmark">GRIDLINE</span>
            <span class="grid-eyebrow">Model-Based Forecasts</span>
          </div>
          <div class="grid-meta">Season {html_text(manifest["prediction_season"])} · data through {html_text(manifest["data_cutoff"])} · model built {html_text(str(manifest["created_at"])[:10])}</div>
        </div></div>
        """,
        unsafe_allow_html=True,
    )


def probability_bar(game: dict[str, Any]) -> str:
    home_probability = min(max(float(game["home_win_probability"]), 0.0), 1.0)
    away_probability = 1.0 - home_probability
    away = html_text(game["away_team"])
    home = html_text(game["home_team"])
    return f"""
      <div class="grid-probability">
        <div class="grid-prob-labels"><span>{away} win prob {away_probability:.1%}</span><span>{home} win prob {home_probability:.1%}</span></div>
        <div class="grid-prob-track"><div class="grid-prob-away" style="width:{away_probability:.2%}"></div><div class="grid-prob-home" style="width:{home_probability:.2%}"></div></div>
      </div>
    """


def market_tile(game: dict[str, Any]) -> str:
    label, value = market_line_label(game)
    pending = not bool(game.get("market_consensus"))
    classes = "grid-tile dashed" if pending else "grid-tile"
    value_class = "grid-tile-value pending" if pending else "grid-tile-value"
    return f'<div class="{classes}"><div class="grid-tile-label">{html_text(label)}</div><div class="{value_class}">{html_text(value)}</div></div>'


def render_featured_game(game: dict[str, Any]) -> None:
    away = html_text(game["away_team"])
    home = html_text(game["home_team"])
    st.markdown(
        f"""
        <div class="grid-hero">
          <div class="grid-kicker" style="color:#FF6B35;margin-bottom:14px">Featured matchup</div>
          <div class="grid-hero-main">
            <div class="grid-matchup">
              <div class="grid-team"><div class="grid-team-chip" style="background:{team_color(game["away_team"])}"></div><div class="grid-team-abbr">{away}</div><div class="grid-score">{float(game["away_score"]):.1f}</div></div>
              <div class="grid-at">@</div>
              <div class="grid-team"><div class="grid-team-chip" style="background:{team_color(game["home_team"])}"></div><div class="grid-team-abbr">{home}</div><div class="grid-score">{float(game["home_score"]):.1f}</div></div>
              <div class="grid-date">{html_text(format_game_time(game))}</div>
            </div>
            <div class="grid-tiles">
              <div class="grid-tile"><div class="grid-tile-label">Spread</div><div class="grid-tile-value">{html_text(spread_label(game))}</div></div>
              <div class="grid-tile"><div class="grid-tile-label">Home ML</div><div class="grid-tile-value">{format_american(american_moneyline(game["home_win_probability"]))}</div></div>
              <div class="grid-tile"><div class="grid-tile-label">Total O/U</div><div class="grid-tile-value">{float(game["total"]):.1f}</div></div>
              {market_tile(game)}
            </div>
          </div>
          {probability_bar(game)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_market_comparison(game: dict[str, Any], key: str) -> None:
    consensus = game.get("market_consensus") or {}
    consensus_spread = consensus.get("spread") or {}
    consensus_total = consensus.get("total") or {}
    if consensus:
        snapshot = str(consensus.get("snapshot_at", ""))[:16].replace("T", " ")
        st.caption(
            f"{consensus.get('provider', 'Market')} consensus captured {snapshot} UTC · "
            f"{consensus_spread.get('book_count', 0)} spread books"
        )
    input_columns = st.columns(2)
    home_line = input_columns[0].number_input(
        f"Sportsbook home spread ({game['home_team']}; favorite is negative)",
        value=float(consensus_spread.get("home_spread", 0.0)),
        step=0.5,
        key=f"spread_{key}",
    )
    market_total = input_columns[1].number_input(
        "Sportsbook total",
        value=float(consensus_total.get("total", game.get("total", 44.0))),
        step=0.5,
        key=f"total_{key}",
    )
    price_columns = st.columns(4)
    home_spread_price = price_columns[0].number_input(
        "Home price",
        value=float(consensus_spread.get("home_price", -110)),
        step=1.0,
        key=f"home_spread_price_{key}",
    )
    away_spread_price = price_columns[1].number_input(
        "Away price",
        value=float(consensus_spread.get("away_price", -110)),
        step=1.0,
        key=f"away_spread_price_{key}",
    )
    over_price = price_columns[2].number_input(
        "Over price",
        value=float(consensus_total.get("over_price", -110)),
        step=1.0,
        key=f"over_price_{key}",
    )
    under_price = price_columns[3].number_input(
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
    st.markdown(
        f"""
        <div class="grid-market-grid">
          <div class="grid-market-card"><div><div class="grid-muted">Model spread side</div><div class="grid-market-value">{html_text(spread_side)}</div></div><div class="grid-edge-spread">{spread_probability - spread_fair:+.1%} vs no-vig</div></div>
          <div class="grid-market-card"><div><div class="grid-muted">Model total side</div><div class="grid-market-value">{html_text(total_side)} {float(market_total):.1f}</div></div><div class="grid-edge-total">{total_probability - total_fair:+.1%} vs no-vig</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Observed {observed_at} ET · model probability {spread_probability:.1%} spread / "
        f"{total_probability:.1%} total · raw implied {spread_implied:.1%} / {total_implied:.1%} · "
        "paper analysis only, not financial advice"
    )


def render_game_row(game: dict[str, Any], index: int) -> None:
    game_id = str(game.get("game_id", index))
    expanded = st.session_state.get("expanded_game_id") == game_id
    with st.container(border=True, key=f"game_card_{index}"):
        columns = st.columns([1.45, 2.0, 0.8, 0.75, 0.8, 0.85], vertical_alignment="center")
        columns[0].markdown(
            f'<div class="grid-row-date">{html_text(format_game_time(game))}</div>',
            unsafe_allow_html=True,
        )
        columns[1].markdown(
            f"""
            <div class="grid-team-pair">
              <div class="grid-team-inline"><span class="grid-team-rail" style="background:{team_color(game["away_team"])}"></span><span><b>{html_text(game["away_team"])}</b><br><span class="grid-muted">{float(game["away_score"]):.1f}</span></span></div>
              <span class="grid-at">@</span>
              <div class="grid-team-inline"><span class="grid-team-rail" style="background:{team_color(game["home_team"])}"></span><span><b>{html_text(game["home_team"])}</b><br><span class="grid-muted">{float(game["home_score"]):.1f}</span></span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        columns[2].markdown(
            f'<div class="grid-row-value"><div class="grid-mini-label">Spread</div>{html_text(spread_label(game))}</div>',
            unsafe_allow_html=True,
        )
        columns[3].markdown(
            f'<div class="grid-row-value"><div class="grid-mini-label">Home ML</div>{format_american(american_moneyline(game["home_win_probability"]))}</div>',
            unsafe_allow_html=True,
        )
        columns[4].markdown(
            f'<div class="grid-row-value"><div class="grid-mini-label">Total</div>{float(game["total"]):.1f}</div>',
            unsafe_allow_html=True,
        )
        if columns[5].button(
            "Hide ▲" if expanded else "Why? ▼",
            key=f"toggle_game_{index}",
            width="stretch",
        ):
            st.session_state.expanded_game_id = None if expanded else game_id
            st.rerun()
        if expanded:
            st.divider()
            detail_columns = st.columns(2)
            with detail_columns[0]:
                st.markdown(
                    '<div class="grid-kicker">Why the model leans this way</div>',
                    unsafe_allow_html=True,
                )
                reasons = "".join(
                    f'<div class="grid-reason"><span>›</span>{html_text(reason)}</div>'
                    for reason in game_reasoning(game)
                )
                st.markdown(reasons, unsafe_allow_html=True)
            with detail_columns[1]:
                st.markdown(
                    '<div class="grid-kicker">Win probability</div>', unsafe_allow_html=True
                )
                st.markdown(probability_bar(game), unsafe_allow_html=True)
                st.markdown(
                    f'<div style="font-size:12px;color:#64748b;margin:12px 0">80% total range: <b style="color:#0f1419">{float(game["total_p10"]):.1f}–{float(game["total_p90"]):.1f}</b></div>{market_tile(game).replace("grid-tile", "grid-placeholder", 1)}',
                    unsafe_allow_html=True,
                )
            with st.expander("Compare a sportsbook line"):
                render_market_comparison(game, game_id)


def render_this_week(state: dict[str, Any]) -> None:
    schedule = state["schedule"]
    week = state.get("report", {}).get("week")
    title = f"This Week — Week {week}" if week is not None else "This Week"
    page_header(title, "Paper analysis · not betting advice")
    if not schedule:
        st.info("No upcoming games are available for the current prediction season.")
        return
    featured = max(schedule, key=lambda game: abs(float(game["predicted_home_margin"])))
    render_featured_game(featured)
    with st.expander("Featured matchup market comparison"):
        render_market_comparison(featured, f"featured_{featured['game_id']}")
    for index, game in enumerate(schedule):
        if game.get("game_id") == featured.get("game_id"):
            continue
        render_game_row(game, index)


def render_prediction_results(prediction: dict[str, Any] | None) -> None:
    if prediction is None:
        values = ("—", "—", "—", "—")
        away = "Away"
        home = "Home"
    else:
        away = str(prediction["away_team"])
        home = str(prediction["home_team"])
        values = (
            f"{float(prediction['away_score']):.1f}",
            f"{float(prediction['home_score']):.1f}",
            format_probability(prediction["home_win_probability"]),
            f"{float(prediction['total']):.1f}",
        )
    st.markdown(
        f"""
        <div class="grid-results">
          <div class="grid-result"><div class="grid-tile-label">{html_text(away)}</div><div class="grid-result-value">{values[0]}</div></div>
          <div class="grid-result"><div class="grid-tile-label">{html_text(home)}</div><div class="grid-result-value">{values[1]}</div></div>
          <div class="grid-result"><div class="grid-tile-label">Home win</div><div class="grid-result-value" style="color:#FF6B35">{values[2]}</div></div>
          <div class="grid-result"><div class="grid-tile-label">Total</div><div class="grid-result-value">{values[3]}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_builder(service: PredictionService, injury_system: InjuryAdjustmentSystem) -> None:
    page_header("Custom Game Builder")
    teams = service.teams()
    with st.container(border=True):
        with st.form("custom_game_form"):
            team_columns = st.columns(2)
            away_team = team_columns[0].selectbox(
                "Away team",
                teams,
                format_func=lambda team: f"{team} — {team_name(team)}",
            )
            home_team = team_columns[1].selectbox(
                "Home team",
                teams,
                index=1 if len(teams) > 1 else 0,
                format_func=lambda team: f"{team} — {team_name(team)}",
            )
            context_columns = st.columns(3)
            away_rest = context_columns[0].number_input(
                "Away rest days", min_value=3, max_value=21, value=7
            )
            home_rest = context_columns[1].number_input(
                "Home rest days", min_value=3, max_value=21, value=7
            )
            week = context_columns[2].number_input("Week", min_value=1, max_value=18, value=1)
            neutral = st.checkbox("Neutral site")
            submitted = st.form_submit_button("Predict Game", type="primary", width="stretch")
        if submitted:
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
                adjusted_home_win = home_cover_probability(
                    float(prediction["predicted_home_margin"]),
                    0.0,
                    float(base["margin_std"]),
                )
                st.session_state.custom_prediction = {
                    **base,
                    **prediction,
                    "home_win_probability": adjusted_home_win,
                }
        st.divider()
        prediction = st.session_state.get("custom_prediction")
        render_prediction_results(prediction)
        if prediction and prediction.get("injury_adjusted"):
            st.warning(prediction.get("adjustment_note", "Manual injury adjustment applied"))
    if prediction:
        with st.container(border=True):
            st.markdown(
                '<div class="grid-kicker" style="margin-bottom:14px">Paper-market comparison</div>',
                unsafe_allow_html=True,
            )
            render_market_comparison(prediction, "custom")
    else:
        st.caption("Choose a matchup and run the model to unlock the paper-market comparison.")


def render_props(
    state: dict[str, Any], service: PredictionService, injury_system: InjuryAdjustmentSystem
) -> None:
    page_header("Player Props")
    mapping = {
        "Passing yards": ("qb", "passing_yards"),
        "Receiving yards": ("wr", "receiving_yards"),
        "Receptions": ("wr", "receptions"),
        "Rushing yards": ("rb", "rushing_yards"),
    }
    prop = st.radio(
        "Prop type",
        list(mapping),
        horizontal=True,
        label_visibility="collapsed",
        key="prop_type",
    )
    state_key, model_name = mapping[prop]
    players = state[state_key]
    roster_seasons = {
        int(player["roster_season"])
        for player in players.values()
        if player.get("roster_season") is not None
    }
    if roster_seasons and max(roster_seasons) < int(service.manifest["prediction_season"]):
        st.warning(
            f"The newest published weekly roster is {max(roster_seasons)}. "
            "Confirm current membership and roles before using preseason prop forecasts."
        )
    choices = sorted(
        players, key=lambda player_id: players[player_id].get("player_name", player_id)
    )
    if not choices:
        st.info("No current player records are available.")
        return
    with st.container(border=True):
        with st.form(f"prop_form_{model_name}"):
            player_id = st.selectbox(
                "Player",
                choices,
                format_func=lambda value: (
                    f"{players[value].get('player_name', value)} — {players[value].get('team', '')} "
                    f"vs {players[value].get('opponent') or 'TBD'}"
                ),
            )
            player = players[player_id]
            st.markdown(
                f"""
                <div class="grid-player-chip"><span class="grid-team-rail" style="height:18px;background:{team_color(str(player.get("team", "")))}"></span><span><b>{html_text(player.get("player_name", player_id))}</b><br><span class="grid-muted">{html_text(player.get("team", ""))} · {html_text(state_key.upper())}</span></span></div>
                """,
                unsafe_allow_html=True,
            )
            line = st.number_input("Sportsbook prop line", value=0.0, step=0.5)
            submitted = st.form_submit_button("Project Prop", type="primary", width="stretch")
        if submitted:
            distribution = service.predict_player(model_name, player)
            adjusted, note = integrate_injuries_into_player_prediction(
                distribution["mean"],
                injury_system,
                player["player_name"],
                player["team"],
                model_name,
            )
            probability_over = over_probability(adjusted, line, max(distribution["std"], 0.01))
            st.session_state.prop_prediction = {
                "model": model_name,
                "player_id": player_id,
                "name": player["player_name"],
                "projection": adjusted,
                "p10": distribution["p10"],
                "p90": distribution["p90"],
                "line": line,
                "over": probability_over,
                "note": note,
            }
        result = st.session_state.get("prop_prediction")
        if result and result.get("model") == model_name:
            over = min(max(float(result["over"]), 0.0), 1.0)
            under = 1.0 - over
            st.markdown(
                f"""
                <div class="grid-prop-panel">
                  <div><div class="grid-kicker">Model projection · {html_text(result["name"])}</div><div class="grid-projection">{float(result["projection"]):.1f}</div><div class="grid-muted">80% interval {float(result["p10"]):.1f}–{float(result["p90"]):.1f} · sportsbook line {float(result["line"]):.1f}</div></div>
                  <div><div class="grid-prob-labels"><span>Under {under:.1%}</span><span>Over {over:.1%}</span></div><div class="grid-prop-bar"><div class="grid-under" style="width:{under:.2%}"></div><div class="grid-over" style="width:{over:.2%}"></div></div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if result["note"] != "Healthy":
                st.warning(result["note"])
            st.caption(
                "The interval reflects model residuals; availability uncertainty is shown separately."
            )


def render_rankings(state: dict[str, Any]) -> None:
    page_header("Power Rankings")
    st.markdown(
        '<div class="grid-muted" style="margin-bottom:18px">Neutral-field rating · descriptive current-state, not a separately trained betting model</div>',
        unsafe_allow_html=True,
    )
    rows: list[tuple[str, float]] = []
    for team, values in state["teams"].items():
        rating = (
            values["points_for_l4"]
            - values["points_against_l4"]
            + 8.0 * (values["off_epa_l4"] - values["def_epa_l4"])
            + 3.0 * (values["pressure_generated_l4"] - values["pressure_allowed_l4"])
        )
        rows.append((team, rating))
    rows.sort(key=lambda item: item[1], reverse=True)
    max_abs = max((abs(rating) for _, rating in rows), default=1.0)
    row_html = "".join(
        f"""
        <div class="grid-rank-row"><div class="grid-rank">{rank}</div><div class="grid-rank-team"><span class="grid-rank-chip" style="background:{team_color(team)}"></span>{html_text(team)}</div><div class="grid-rank-track"><div class="grid-rank-fill" style="width:{max(4.0, abs(rating) / max_abs * 100):.1f}%;background:{"#16a34a" if rating >= 0 else "#dc2626"}"></div></div><div class="grid-rank-value">{rating:+.1f}</div></div>
        """
        for rank, (team, rating) in enumerate(rows, start=1)
    )
    st.markdown(f'<div class="grid-card">{row_html}</div>', unsafe_allow_html=True)


def render_performance(state: dict[str, Any]) -> None:
    page_header("Immutable Prediction Performance")
    runs = state["performance"].get("runs", [])
    if runs:
        st.dataframe(pd.DataFrame(runs), hide_index=True, width="stretch")
    else:
        st.info("No completed immutable prediction batches have been scored yet.")
        st.markdown(
            """
            <div class="grid-card grid-ghost-chart">
              <div class="grid-kicker">Illustrative — MAE by week, populates after scoring</div>
              <div class="grid-ghost-row"><span>Week 1</span><div class="grid-ghost-bar" style="width:76%"></div></div>
              <div class="grid-ghost-row"><span>Week 2</span><div class="grid-ghost-bar" style="width:62%"></div></div>
              <div class="grid-ghost-row"><span>Week 3</span><div class="grid-ghost-bar" style="width:69%"></div></div>
              <div class="grid-ghost-row"><span>Week 4</span><div class="grid-ghost-bar" style="width:54%"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    benchmark = state.get("market_benchmark")
    if not benchmark:
        return
    variants = benchmark["variants"]
    st.markdown(
        f"""
        <div class="grid-page-head" style="padding-bottom:12px"><h2 class="grid-page-title" style="font-size:20px">Historical Model vs Market</h2></div>
        <div class="grid-results">
          <div class="grid-result"><div class="grid-tile-label">Matched games</div><div class="grid-result-value">{int(benchmark["games"]):,}</div></div>
          <div class="grid-result"><div class="grid-tile-label">Independent margin MAE</div><div class="grid-result-value">{float(variants["independent_margin"]["mae"]):.2f}</div></div>
          <div class="grid-result"><div class="grid-tile-label">Market margin MAE</div><div class="grid-result-value">{float(variants["market_margin"]["mae"]):.2f}</div></div>
          <div class="grid-result"><div class="grid-tile-label">Validated market weight</div><div class="grid-result-value" style="color:#16a34a">{float(benchmark["future_production_market_margin_weight"]):.0%}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Consensus captured about {benchmark['methodology']['snapshot_minutes_before_kickoff']} "
        f"minutes before kickoff across {benchmark['methodology']['evaluation_seasons']}. "
        "The independent forecast remains visible for research; historical disagreement did not "
        "establish an ATS edge."
    )
    st.dataframe(
        pd.DataFrame(benchmark["disagreement_buckets"]),
        hide_index=True,
        width="stretch",
    )


def render_model_card(manifest: dict[str, Any]) -> None:
    page_header("Model Card")
    training = "–".join(str(season) for season in manifest["training_seasons"])
    commit = str(manifest.get("git_commit") or "unavailable")[:10]
    st.markdown(
        f"""
        <div class="grid-model-summary">
          <div class="grid-model-card"><div class="grid-tile-label">Prediction season</div><div class="grid-result-value" style="font-size:18px">{html_text(manifest["prediction_season"])}</div></div>
          <div class="grid-model-card"><div class="grid-tile-label">Training seasons</div><div class="grid-result-value" style="font-size:18px">{html_text(training)}</div></div>
          <div class="grid-model-card"><div class="grid-tile-label">Data cutoff</div><div class="grid-result-value" style="font-size:18px">{html_text(manifest["data_cutoff"])}</div></div>
          <div class="grid-model-card"><div class="grid-tile-label">Git commit</div><div style="margin-top:6px;font:600 14px monospace;color:#64748b">{html_text(commit)}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for name, specification in manifest["models"].items():
        metrics = specification["metrics"]
        improvement = relative_mae_improvement(metrics)
        improvement_text = "—" if improvement is None else f"{improvement:+.1%}"
        title = name.replace("_", " ").title()
        st.markdown(
            f"""
            <div class="grid-model-card">
              <div class="grid-model-head"><div style="font:500 15px 'Instrument Sans',sans-serif">{html_text(title)}</div><div class="grid-muted">{int(metrics.get("oof_rows", 0)):,} out-of-fold rows · {len(specification.get("features", []))} features</div></div>
              <div class="grid-model-metrics">
                <div class="grid-model-metric"><div class="grid-mini-label">MAE</div><div class="grid-model-value">{float(metrics.get("mae", 0)):.2f}</div></div>
                <div class="grid-model-metric"><div class="grid-mini-label">RMSE</div><div class="grid-model-value">{float(metrics.get("rmse", 0)):.2f}</div></div>
                <div class="grid-model-metric"><div class="grid-mini-label">Bias</div><div class="grid-model-value">{float(metrics.get("bias", 0)):+.2f}</div></div>
                <div class="grid-model-metric"><div class="grid-mini-label">80% coverage</div><div class="grid-model-value">{float(metrics.get("interval_80_coverage", 0)):.1%}</div></div>
                <div class="grid-model-metric"><div class="grid-mini-label">vs baseline</div><div class="grid-model-value grid-positive">{improvement_text}</div></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander(f"{title} feature schema"):
            st.write(", ".join(specification.get("features", [])))
    st.caption(
        "Forecasts are estimates with uncertainty. Market comparisons are paper analysis, not financial advice."
    )


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
top_bar(manifest)

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

if "active_screen" not in st.session_state:
    st.session_state.active_screen = PAGE_LABELS[0]

page = st.session_state.active_screen
if page == "This Week":
    render_this_week(state)
elif page == "Builder":
    render_builder(service, injury_system)
elif page == "Props":
    render_props(state, service, injury_system)
elif page == "Rankings":
    render_rankings(state)
elif page == "Performance":
    render_performance(state)
else:
    render_model_card(manifest)

st.radio(
    "Navigate",
    PAGE_LABELS,
    horizontal=True,
    label_visibility="collapsed",
    key="active_screen",
)
