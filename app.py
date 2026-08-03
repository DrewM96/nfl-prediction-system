from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from cfb_prediction.config import (
    CFB_FOUNDATION_PATH,
    CFB_HISTORICAL_BENCHMARK_PATH,
    CFB_MODEL_MANIFEST_PATH,
)
from cfb_prediction.ledger import load_latest_cfb_prediction_batch
from cfb_prediction.modeling import load_cfb_model_bundle
from injury_system import (
    InjuryAdjustmentSystem,
    integrate_injuries_into_game_prediction,
    integrate_injuries_into_player_prediction,
    render_injury_manager,
)
from nfl_prediction.config import MODEL_MANIFEST_PATH, PROJECT_ROOT, is_division_game
from nfl_prediction.io import read_json, sha256_file
from nfl_prediction.market import (
    american_odds_to_implied_probability,
    home_cover_probability,
    no_vig_probabilities,
    over_probability,
)
from nfl_prediction.modeling import FittedEnsemble, load_model_bundle
from nfl_prediction.odds import attach_market_consensus
from nfl_prediction.rankings import build_football_form_ratings, build_market_power_ratings
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
SPORT_LABELS = ["NFL", "College Football"]

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
    [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stSidebarCollapsedControl"] {
      display: none !important;
    }
    [data-testid="stMain"] { width: 100% !important; }
    [data-testid="stMainBlockContainer"], .block-container {
      max-width: 1120px !important;
      padding: 0 2rem 6rem !important;
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
      margin: -16px 0 0 calc(50% - 50vw);
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
    .grid-meta {
      max-width: 520px;
      overflow: hidden;
      color: var(--grid-muted);
      font-size: 13px;
      text-align: right;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .grid-page-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding: 28px 0 20px;
    }
    h1.grid-page-title, h2.grid-page-title {
      margin: 0 !important;
      padding: 0 !important;
      font-family: 'Instrument Sans', sans-serif !important;
      font-size: 26px !important;
      font-weight: 500 !important;
      line-height: 1.22 !important;
      letter-spacing: -0.01em !important;
    }
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
    .grid-rank-row { display: grid; grid-template-columns: 34px 240px 1fr 52px; align-items: center; gap: 12px; padding: 10px 0; border-bottom: 1px solid #f1f3f5; }
    .grid-rank-row:last-child { border-bottom: none; }
    .grid-rank { color: var(--grid-faint); font-size: 14px; }
    .grid-rank-team { display: flex; align-items: center; gap: 9px; font-weight: 600; }
    .grid-rank-team-copy { min-width: 0; }
    .grid-rank-roster { margin-top: 3px; color: var(--grid-faint); font-size: 10px; font-weight: 400; white-space: nowrap; }
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
    .grid-topbar, .grid-page-head, .grid-hero, .grid-card, .grid-results,
    .grid-model-summary, .grid-model-card, .grid-market-grid, .grid-prop-panel {
      line-height: normal;
      box-sizing: border-box;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
      border: 1px solid var(--grid-border) !important;
      border-radius: 12px !important;
      box-shadow: none;
    }
    [class*="st-key-game_card_"] [data-testid="stVerticalBlockBorderWrapper"] {
      padding: 10px 18px !important;
    }
    [class*="st-key-game_card_"] [data-testid="stVerticalBlock"] {
      gap: 0 !important;
    }
    [class*="st-key-game_card_"] .stMarkdown p { margin: 0 !important; }
    [class*="st-key-toggle_game_"] button {
      min-height: 24px !important;
      height: 24px !important;
      padding: 0 !important;
      border: 0 !important;
      background: transparent !important;
      color: var(--grid-orange) !important;
      font-size: 12px !important;
      box-shadow: none !important;
    }
    .st-key-custom_game_shell [data-testid="stVerticalBlockBorderWrapper"],
    .st-key-prop_shell [data-testid="stVerticalBlockBorderWrapper"] {
      padding: 24px !important;
      border-radius: 14px !important;
    }
    .st-key-custom_game_shell [data-testid="stSelectbox"] label p,
    .st-key-custom_game_shell [data-testid="stNumberInput"] label p,
    .st-key-prop_shell [data-testid="stSelectbox"] label p,
    .st-key-prop_shell [data-testid="stNumberInput"] label p {
      color: var(--grid-faint) !important;
      font: 500 11px 'Instrument Sans', sans-serif !important;
      letter-spacing: .3px;
      text-transform: uppercase;
    }
    .st-key-custom_game_shell [data-baseweb="select"] > div,
    .st-key-prop_shell [data-baseweb="select"] > div,
    .st-key-custom_game_shell [data-testid="stNumberInput"] input,
    .st-key-prop_shell [data-testid="stNumberInput"] input {
      min-height: 44px !important;
      background: var(--grid-recessed) !important;
      border-color: var(--grid-border) !important;
    }
    .st-key-custom_game_shell [data-testid="stFormSubmitButton"] button {
      min-height: 44px !important;
    }
    .st-key-prop_shell [data-testid="stPopover"] button {
      min-height: 30px !important;
      padding: 4px 10px !important;
      border-color: var(--grid-border) !important;
      background: #fff !important;
      color: var(--grid-muted) !important;
      font-size: 12px !important;
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
    .st-key-active_screen {
      position: fixed;
      z-index: 1000;
      left: 0;
      right: 0;
      bottom: 0;
      width: 100vw !important;
      max-width: none !important;
      margin: 0 !important;
      box-sizing: border-box;
      padding: 5px max(12px, calc((100vw - 1120px)/2));
      background: rgba(255,255,255,.98);
      border-top: 1px solid var(--grid-border);
    }
    .st-key-active_screen > div { width: 100% !important; }
    .st-key-active_screen [role="radiogroup"] { display: grid; grid-template-columns: repeat(6,1fr); width: 100%; gap: 0 !important; }
    .st-key-active_screen label {
      min-height: 46px;
      padding: 5px 4px !important;
      justify-content: center;
      flex-direction: column;
      gap: 5px;
      border: 0 !important;
      border-radius: 0 !important;
      background: transparent !important;
      color: var(--grid-faint) !important;
      font-size: 11px;
    }
    .st-key-active_screen label::before {
      content: '';
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: #cbd5e1;
    }
    .st-key-active_screen label:has(input:checked) { color: var(--grid-ink) !important; }
    .st-key-active_screen label:has(input:checked)::before { background: var(--grid-orange); }
    .st-key-active_screen label p {
      margin: 0 !important;
      color: inherit !important;
      font-size: 11px !important;
      line-height: 1 !important;
    }
    [data-testid="stSidebar"] { background: #f8f9fb; border-right: 1px solid var(--grid-border); }
    .grid-muted { color: var(--grid-faint); font-size: 12px; }
    .grid-positive { color: #16a34a; }
    @media (max-width: 760px) {
      [data-testid="stMainBlockContainer"], .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
      .grid-topbar { padding: 14px 16px; }
      .grid-topbar-inner { align-items: flex-start; }
      .grid-eyebrow, .grid-meta { display: none; }
      .grid-page-head { padding-top: 20px; align-items: flex-start; }
      .grid-badge { white-space: normal; text-align: right; }
      .grid-hero { padding: 20px; }
      .grid-tiles, .grid-results, .grid-model-summary { grid-template-columns: repeat(2,1fr); }
      .grid-market-grid, .grid-prop-panel { grid-template-columns: 1fr; }
      .grid-model-metrics { grid-template-columns: repeat(2,1fr); }
      .grid-rank-row { grid-template-columns: 26px 190px 1fr 44px; gap: 8px; }
      .st-key-active_screen label { padding: 6px 2px !important; font-size: 9px; }
      .st-key-active_screen label p { font-size: 9px !important; white-space: nowrap; }
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


@st.cache_data
def load_cfb_state() -> dict[str, Any]:
    state = read_json(
        CFB_FOUNDATION_PATH,
        {
            "status": "not_built",
            "prediction_season": datetime.now().year,
            "created_at": "pending",
            "data_cutoff": "not built",
            "source": "CollegeFootballData REST API v2",
            "team_count": 0,
            "scheduled_game_count": 0,
            "fbs_vs_fbs_game_count": 0,
            "completed_game_count": 0,
            "calendar_week_count": 0,
            "models": {},
        },
    )
    state["historical_benchmark"] = read_json(CFB_HISTORICAL_BENCHMARK_PATH)
    try:
        _, model_manifest = load_cfb_model_bundle()
        prediction_batch = load_latest_cfb_prediction_batch()
        if prediction_batch is None:
            raise ValueError("No immutable CFB prediction batch is available")
        model_hash = sha256_file(CFB_MODEL_MANIFEST_PATH)
        if prediction_batch.get("model_hash") != model_hash:
            raise ValueError("The CFB prediction batch does not match the active model bundle")
        state["model_manifest"] = model_manifest
        state["prediction_batch"] = prediction_batch
        state["production_status"] = "forecast_ready"
    except Exception as exc:
        state["production_status"] = "not_built"
        state["production_error"] = str(exc)
    return state


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


def top_bar(manifest: dict[str, Any], *, league: str = "NFL") -> None:
    st.markdown(
        f"""
        <div class="grid-topbar"><div class="grid-topbar-inner">
          <div class="grid-brand">
            <span class="grid-wordmark">GRIDLINE</span>
            <span class="grid-eyebrow">{html_text(league)} Forecasts</span>
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
              <div class="grid-tile"><div class="grid-tile-label">Moneyline</div><div class="grid-tile-value">{format_american(american_moneyline(game["home_win_probability"]))}</div></div>
              <div class="grid-tile"><div class="grid-tile-label">Total O/U</div><div class="grid-tile-value">{float(game["total"]):.1f}</div></div>
              {market_tile(game)}
            </div>
          </div>
          {probability_bar(game)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def market_inputs(
    game: dict[str, Any],
    key: str,
    consensus_spread: dict[str, Any],
    consensus_total: dict[str, Any],
) -> tuple[float, float, float, float, float, float, str]:
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
    return (
        float(home_line),
        float(market_total),
        float(home_spread_price),
        float(away_spread_price),
        float(over_price),
        float(under_price),
        observed_at,
    )


def render_market_comparison(
    game: dict[str, Any], key: str, *, compact_controls: bool = False
) -> None:
    consensus = game.get("market_consensus") or {}
    consensus_spread = consensus.get("spread") or {}
    consensus_total = consensus.get("total") or {}
    if consensus:
        snapshot = str(consensus.get("snapshot_at", ""))[:16].replace("T", " ")
        st.caption(
            f"{consensus.get('provider', 'Market')} consensus captured {snapshot} UTC · "
            f"{consensus_spread.get('book_count', 0)} spread books"
        )
    if compact_controls:
        with st.popover("Adjust sportsbook line"):
            inputs = market_inputs(game, key, consensus_spread, consensus_total)
    else:
        inputs = market_inputs(game, key, consensus_spread, consensus_total)
    (
        home_line,
        market_total,
        home_spread_price,
        away_spread_price,
        over_price,
        under_price,
        observed_at,
    ) = inputs
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
        columns = st.columns([1.5, 4.8, 0.9, 0.9, 1.0, 0.9], vertical_alignment="center")
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
            f'<div class="grid-row-value"><div class="grid-mini-label">ML</div>{format_american(american_moneyline(game["home_win_probability"]))}</div>',
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
    if "expanded_game_id" not in st.session_state:
        st.session_state.expanded_game_id = str(featured.get("game_id"))
    render_featured_game(featured)
    for index, game in enumerate(schedule):
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
    away_default = teams.index("BUF") if "BUF" in teams else 0
    home_default = teams.index("HOU") if "HOU" in teams else (1 if len(teams) > 1 else 0)
    with st.container(border=True, key="custom_game_shell"):
        with st.form("custom_game_form", border=False):
            team_columns = st.columns(2)
            away_team = team_columns[0].selectbox(
                "Away team",
                teams,
                index=away_default,
                format_func=lambda team: f"{team} — {team_name(team)}",
            )
            home_team = team_columns[1].selectbox(
                "Home team",
                teams,
                index=home_default,
                format_func=lambda team: f"{team} — {team_name(team)}",
            )
            context_columns = st.columns([1, 1, 1, 0.8], vertical_alignment="bottom")
            away_rest = context_columns[0].number_input(
                "Away rest days", min_value=3, max_value=21, value=7
            )
            home_rest = context_columns[1].number_input(
                "Home rest days", min_value=3, max_value=21, value=7
            )
            week = context_columns[2].number_input("Week", min_value=1, max_value=18, value=1)
            neutral = context_columns[3].checkbox("Neutral site")
            submitted = st.form_submit_button("Predict Game", type="primary", width="stretch")
        if submitted or "custom_prediction" not in st.session_state:
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
            render_market_comparison(prediction, "custom", compact_controls=True)
    else:
        st.caption("Choose a matchup and run the model to unlock the paper-market comparison.")
    render_injury_manager(injury_system, service.teams(), container=st)
    official_injuries = service.state["official_injuries"]
    if official_injuries.get("stale_for_prediction_season"):
        st.caption(
            "Official injury reports are not yet published for the prediction season; "
            "manual scenarios are session-only."
        )
    elif official_injuries.get("entries"):
        st.caption(
            f"Official injury feed: Week {official_injuries['available_week']} "
            f"({len(official_injuries['entries'])} reported players)."
        )


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
    roster_notice = None
    if roster_seasons and max(roster_seasons) < int(service.manifest["prediction_season"]):
        roster_notice = (
            f"Newest published weekly roster: {max(roster_seasons)}. "
            "Confirm current membership and roles for preseason forecasts."
        )
    choices = sorted(
        players, key=lambda player_id: players[player_id].get("player_name", player_id)
    )
    if not choices:
        st.info("No current player records are available.")
        return
    default_player_id = next(
        (player_id for player_id in choices if players[player_id].get("player_name") == "D.Maye"),
        choices[0],
    )
    with st.container(border=True, key="prop_shell"):
        control_columns = st.columns([4, 1])
        control_columns[0].markdown(
            f'<div class="grid-kicker" style="padding-top:7px">Select player · {html_text(prop)}</div>',
            unsafe_allow_html=True,
        )
        with control_columns[1].popover("Change player / line", use_container_width=True):
            player_id = st.selectbox(
                "Player",
                choices,
                index=choices.index(default_player_id),
                format_func=lambda value: (
                    f"{players[value].get('player_name', value)} — "
                    f"{players[value].get('team', '')} vs "
                    f"{players[value].get('opponent') or 'TBD'}"
                ),
            )
            selected_player = players[player_id]
            selected_distribution = service.predict_player(model_name, selected_player)
            selected_adjusted, _ = integrate_injuries_into_player_prediction(
                selected_distribution["mean"],
                injury_system,
                selected_player["player_name"],
                selected_player["team"],
                model_name,
            )
            default_line = round(selected_adjusted * 2.0) / 2.0
            line = st.number_input(
                "Sportsbook prop line",
                value=float(default_line),
                step=0.5,
                key=f"prop_line_{model_name}_{player_id}",
            )
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
        over = min(max(float(probability_over), 0.0), 1.0)
        under = 1.0 - over
        st.markdown(
            f"""
            <div class="grid-player-chip"><span class="grid-team-rail" style="height:18px;background:{team_color(str(player.get("team", "")))}"></span><span><b>{html_text(player.get("player_name", player_id))}</b><br><span class="grid-muted">{html_text(player.get("team", ""))} · {html_text(state_key.upper())}</span></span></div>
            <div class="grid-prop-panel">
              <div><div class="grid-kicker">Projection — {html_text(player["player_name"])}</div><div class="grid-projection">{float(adjusted):.1f}</div><div class="grid-muted">80% interval {float(distribution["p10"]):.1f}–{float(distribution["p90"]):.1f}<br>Sportsbook line: <b style="color:#0f1419">{float(line):.1f}</b></div></div>
              <div><div class="grid-prob-labels"><span>Under {under:.1%}</span><span>Over {over:.1%}</span></div><div class="grid-prop-bar"><div class="grid-under" style="width:{under:.2%}"></div><div class="grid-over" style="width:{over:.2%}"></div></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if note != "Healthy":
            st.warning(note)
    if roster_notice:
        st.caption(roster_notice)
    st.caption(
        "The interval reflects model residuals; availability uncertainty is shown separately."
    )


def render_rankings(state: dict[str, Any]) -> None:
    page_header("Power Rankings")
    market = build_market_power_ratings(state.get("market"))
    football = build_football_form_ratings(state["teams"])
    choices = (
        ["Latest market consensus", "2025 football form"] if market else ["2025 football form"]
    )
    source = st.radio(
        "Ranking source",
        choices,
        horizontal=True,
        key="power_ranking_source",
    )
    if source == "Latest market consensus" and market:
        raw_snapshot = str(market.get("snapshot_at") or "")
        try:
            snapshot = datetime.fromisoformat(raw_snapshot.replace("Z", "+00:00")).astimezone(
                ZoneInfo("America/New_York")
            )
            snapshot_text = (
                f"{snapshot.strftime('%b %d, %Y')} · {snapshot.strftime('%I:%M%p').lstrip('0')} ET"
            )
        except (TypeError, ValueError):
            snapshot_text = raw_snapshot or "unknown"
        st.markdown(
            f"""
            <div class="grid-muted" style="margin-bottom:14px">Market-implied points above or below an average NFL team on a neutral field · snapshot {html_text(snapshot_text)}</div>
            <div class="grid-results">
              <div class="grid-result"><div class="grid-tile-label">Schedule lines</div><div class="grid-result-value">{int(market["game_count"])}</div></div>
              <div class="grid-result"><div class="grid-tile-label">Median books</div><div class="grid-result-value">{float(market["median_book_count"]):.0f}</div></div>
              <div class="grid-result"><div class="grid-tile-label">Implied home field</div><div class="grid-result-value">{float(market["home_field_points"]):.2f}</div></div>
              <div class="grid-result"><div class="grid-tile-label">Line reconstruction MAE</div><div class="grid-result-value">{float(market["line_fit_mae"]):.2f}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        ranking_rows = market["ratings"]
        st.caption(
            "This view decomposes the latest consensus spreads across the full schedule. It is "
            f"market information, not the independent GRIDLINE model and not evidence of betting value. "
            f"Single-book games are excluded ({int(market['excluded_single_book_games'])}); many "
            "later look-ahead lines currently have only two or three books and will move."
        )
    else:
        st.markdown(
            f'<div class="grid-muted" style="margin-bottom:18px">Completed-game form using points, EPA, and pressure · data through {html_text(football.get("data_cutoff") or "unknown")} · no offseason market input</div>',
            unsafe_allow_html=True,
        )
        ranking_rows = football["ratings"]
        st.caption(
            "No new NFL games have been played since this cutoff. This descriptive view updates "
            "after completed games; roster features remain excluded because they worsened margin validation."
        )
    roster_context = state["teams"]

    def roster_label(team: str) -> str:
        values = roster_context.get(team, {})
        required = (
            "roster_qb_returning",
            "roster_ol_continuity",
            "roster_skill_continuity",
        )
        if not all(key in values for key in required):
            return "Roster context unavailable"
        qb = "QB returns" if float(values["roster_qb_returning"]) >= 0.5 else "New QB room"
        return (
            f"{qb} | OL {float(values['roster_ol_continuity']):.0%} | "
            f"Skill {float(values['roster_skill_continuity']):.0%}"
        )

    max_abs = max((abs(float(row["rating"])) for row in ranking_rows), default=1.0)
    row_html = "".join(
        f'<div class="grid-rank-row"><div class="grid-rank">{rank}</div>'
        f'<div class="grid-rank-team"><span class="grid-rank-chip" '
        f'style="background:{team_color(str(row["team"]))}"></span>'
        f'<div class="grid-rank-team-copy">{html_text(team_name(str(row["team"])))}'
        f'<div class="grid-rank-roster">{html_text(roster_label(str(row["team"])))}</div></div></div>'
        f'<div class="grid-rank-track"><div class="grid-rank-fill" '
        f'style="width:{max(4.0, abs(float(row["rating"])) / max_abs * 100):.1f}%;'
        f'background:{"#16a34a" if float(row["rating"]) >= 0 else "#dc2626"}"></div></div>'
        f'<div class="grid-rank-value">{float(row["rating"]):+.1f}</div></div>'
        for rank, row in enumerate(ranking_rows, start=1)
    )
    st.markdown(f'<div class="grid-card">{row_html}</div>', unsafe_allow_html=True)
    st.caption(
        "Roster context compares the current 2026 roster with each team's 2025 snap distribution. "
        "It informs the Weeks 1-4 totals model, with a weekly decay, but does not change the market "
        "ranking order. QB status means the prior season's primary quarterback remains on the roster; "
        "it is not a confirmed Week 1 starter designation."
    )


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


def render_cfb_foundation(state: dict[str, Any]) -> None:
    ready = state.get("status") == "data_ready"
    forecast_ready = state.get("production_status") == "forecast_ready"
    badge = (
        "Provisional forecasts ready"
        if forecast_ready
        else ("Data foundation ready" if ready else "Run weekly_cfb_update.py")
    )
    page_header("College Football", badge)
    st.markdown(
        f"""
        <div class="grid-hero">
          <div class="grid-kicker">FBS · {html_text(state.get("prediction_season", "—"))}</div>
          <h2 style="margin:8px 0 8px;font:500 28px 'Instrument Sans',sans-serif">A separate college model, inside the same GRIDLINE app.</h2>
          <div class="grid-muted">The authenticated CollegeFootballData feed is isolated from the NFL pipeline. Raw API responses remain in a local ignored cache; only derived artifacts are eligible for publication.</div>
        </div>
        <div class="grid-results">
          <div class="grid-result"><div class="grid-tile-label">FBS teams</div><div class="grid-result-value">{int(state.get("team_count", 0)):,}</div></div>
          <div class="grid-result"><div class="grid-tile-label">Scheduled games</div><div class="grid-result-value">{int(state.get("scheduled_game_count", 0)):,}</div></div>
          <div class="grid-result"><div class="grid-tile-label">FBS vs FBS</div><div class="grid-result-value">{int(state.get("fbs_vs_fbs_game_count", 0)):,}</div></div>
          <div class="grid-result"><div class="grid-tile-label">Regular-season weeks</div><div class="grid-result-value">{int(state.get("calendar_week_count", 0)):,}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not ready:
        st.warning(
            "The College Football artifact has not been built in this environment. "
            "Run `python weekly_cfb_update.py --season 2026`."
        )
        return
    prediction_batch = state.get("prediction_batch")
    model_manifest = state.get("model_manifest")
    if prediction_batch and model_manifest:
        forecast_week = prediction_batch.get("metadata", {}).get("forecast_week", "—")
        predictions = prediction_batch.get("predictions", [])
        rows = []
        for prediction in predictions:
            margin = float(prediction["predicted_home_margin"])
            favorite = prediction["home_team"] if margin >= 0 else prediction["away_team"]
            kickoff = datetime.fromisoformat(str(prediction["start_date"])).astimezone(
                ZoneInfo("America/New_York")
            )
            rows.append(
                {
                    "Kickoff (ET)": (
                        f"{kickoff.strftime('%a')} {kickoff.month}/{kickoff.day} "
                        f"{kickoff.strftime('%I:%M%p').lstrip('0').lower()}"
                    ),
                    "Matchup": f"{prediction['away_team']} at {prediction['home_team']}",
                    "Model score": (
                        f"{prediction['away_team']} {float(prediction['predicted_away_score']):.1f} · "
                        f"{prediction['home_team']} {float(prediction['predicted_home_score']):.1f}"
                    ),
                    "Projected line": f"{favorite} -{abs(margin):.1f}",
                    "Total": f"{float(prediction['predicted_total']):.1f}",
                    "Home win": format_probability(prediction["home_win_probability"]),
                }
            )
        st.markdown(
            f"""
            <div class="grid-page-head" style="padding-bottom:12px"><div><div class="grid-kicker">First immutable production run</div><h2 class="grid-page-title" style="font-size:22px">2026 Week {html_text(forecast_week)} Forecasts</h2></div></div>
            <div class="grid-results">
              <div class="grid-result"><div class="grid-tile-label">Games</div><div class="grid-result-value">{len(predictions):,}</div></div>
              <div class="grid-result"><div class="grid-tile-label">Margin model · 2025 MAE</div><div class="grid-result-value">{float(model_manifest["models"]["margin"]["metrics"]["latest_holdout_mae"]):.2f}</div></div>
              <div class="grid-result"><div class="grid-tile-label">Total model · 2025 MAE</div><div class="grid-result-value">{float(model_manifest["models"]["total"]["metrics"]["latest_holdout_mae"]):.2f}</div></div>
              <div class="grid-result"><div class="grid-tile-label">Status</div><div class="grid-result-value" style="font-size:18px">Provisional</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "Independent football-model forecasts recorded before kickoff. Sportsbook lines are "
            "not model inputs. Wide residual uncertainty remains, especially before teams play."
        )
        coverage = model_manifest.get("input_coverage", {})
        covered_teams = int(coverage.get("scheduled_fbs_teams", 0))
        returning_teams = int(coverage.get("returning_production_teams", 0))
        talent_teams = int(coverage.get("talent_teams", 0))
        if covered_teams and (returning_teams < covered_teams or talent_teams < covered_teams):
            st.warning(
                "2026 preseason feed coverage is not final: returning production is available "
                f"for {returning_teams}/{covered_teams} scheduled FBS teams and talent for "
                f"{talent_teams}/{covered_teams}. Missing values are neutral-imputed. Regenerate "
                "this batch when CFBD publishes those feeds and final rosters settle."
            )
        with st.expander("Model and forecast audit"):
            st.write(
                f"Run {prediction_batch.get('run_id')} · model {prediction_batch.get('model_hash', '')[:12]}… · "
                f"data cutoff {prediction_batch.get('data_cutoff')}"
            )
            st.write(
                "Margin uses Elo, recent form, advanced efficiency, and preseason roster/talent "
                "context where available. Total uses Elo, form, and advanced efficiency."
            )
    else:
        st.info(
            "The historical benchmark passed, but no checksummed production forecast is installed. "
            "Run `python cfb_production_update.py --season 2026`."
        )
    benchmark = state.get("historical_benchmark")
    if benchmark:
        selected = benchmark["selected_by_development"]
        margin = benchmark["results"][selected["margin"]]["margin"]
        total = benchmark["results"][selected["total"]]["total"]
        st.markdown(
            f"""
            <div class="grid-page-head" style="padding-bottom:12px"><h2 class="grid-page-title" style="font-size:20px">Historical Football Benchmark</h2></div>
            <div class="grid-results">
              <div class="grid-result"><div class="grid-tile-label">Evaluated games</div><div class="grid-result-value">{int(benchmark["completed_fbs_games"]):,}</div></div>
              <div class="grid-result"><div class="grid-tile-label">2025 margin MAE</div><div class="grid-result-value">{float(margin["model"]["holdout"]["mae"]):.2f}</div></div>
              <div class="grid-result"><div class="grid-tile-label">2025 total MAE</div><div class="grid-result-value">{float(total["model"]["holdout"]["mae"]):.2f}</div></div>
              <div class="grid-result"><div class="grid-tile-label">Status</div><div class="grid-result-value" style="font-size:18px">Research</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Configurations were selected before the untouched 2025 season. Listed historical "
            "market lines remained more accurate and are comparison-only; CFBD does not provide "
            "a timestamp proving that these were closing lines."
        )
    with st.expander("Production safeguards"):
        st.write(
            "The app shows a forecast only when both model files pass their manifest checksums and "
            "the immutable prediction batch matches that exact manifest. Future results never update "
            "pregame features, and each weekly run is stored separately for honest scoring."
        )
    st.caption(
        f"Source: {state.get('source', 'CollegeFootballData')} · derived through "
        f"{state.get('data_cutoff', 'unknown')} · no raw CFBD responses published."
    )


if "active_sport" not in st.session_state:
    st.session_state.active_sport = SPORT_LABELS[0]

active_sport = st.session_state.active_sport
if active_sport == "College Football":
    cfb_state = load_cfb_state()
    top_bar(cfb_state, league="College Football")
    st.radio(
        "Sport",
        SPORT_LABELS,
        horizontal=True,
        label_visibility="collapsed",
        key="active_sport",
    )
    render_cfb_foundation(cfb_state)
    st.stop()

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
top_bar(manifest, league="NFL")
st.radio(
    "Sport",
    SPORT_LABELS,
    horizontal=True,
    label_visibility="collapsed",
    key="active_sport",
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
