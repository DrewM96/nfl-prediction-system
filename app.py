# nfl_app.py - NFL Themed v2.1 with Weekly Report

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os
from datetime import datetime
from injury_system import (
    InjuryAdjustmentSystem, 
    render_injury_manager, 
    integrate_injuries_into_game_prediction,
    integrate_injuries_into_player_prediction
)

# Page configuration
st.set_page_config(
    page_title="NFL Prediction System v2.1",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# NFL-themed custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Condensed:wght@700;900&display=swap');
    
    /* Main background - dark gray */
    .stApp {
        background-color: #1a1a1a !important;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #1a1a1a !important;
        padding-top: 2rem !important;
    }
    
    /* Headers - rigid font */
    h1, h2, h3 {
        font-family: 'Roboto Condensed', sans-serif !important;
    }
    
    h1 {
        color: #FFFFFF !important;
        font-weight: 900 !important;
        font-size: 3.5rem !important;
        text-transform: uppercase !important;
        letter-spacing: 3px !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        border-bottom: 5px solid #CC0000;
        padding-bottom: 20px;
        margin-bottom: 30px;
    }
    
    h2 {
        color: #FFFFFF !important;
        font-weight: 900 !important;
        font-size: 2.2rem !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        border-left: 6px solid #CC0000;
        padding-left: 15px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    
    h3 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        font-size: 1.3rem !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2.8rem !important;
        font-weight: 900 !important;
        color: #FFFFFF !important;
        font-family: 'Roboto Condensed', sans-serif !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        color: #CCCCCC !important;
        font-family: 'Roboto Condensed', sans-serif !important;
    }
    
    /* Primary buttons - RED */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #CC0000 0%, #990000 100%) !important;
        color: white !important;
        font-weight: 900 !important;
        font-size: 1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 16px 32px !important;
        box-shadow: 0 4px 15px rgba(204, 0, 0, 0.5) !important;
        transition: all 0.2s ease !important;
        font-family: 'Roboto Condensed', sans-serif !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(204, 0, 0, 0.7) !important;
        background: linear-gradient(135deg, #E60000 0%, #CC0000 100%) !important;
    }
    
    /* Secondary buttons - BLUE */
    .stButton > button {
        background: linear-gradient(135deg, #0047AB 0%, #003380 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 12px 24px !important;
        box-shadow: 0 3px 10px rgba(0, 71, 171, 0.4) !important;
        transition: all 0.2s ease !important;
        font-family: 'Roboto Condensed', sans-serif !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0, 71, 171, 0.6) !important;
        background: linear-gradient(135deg, #0052CC 0%, #0047AB 100%) !important;
    }
    
    /* Info boxes - BLUE accent */
    div[data-baseweb="notification"] {
        background-color: rgba(0, 71, 171, 0.15) !important;
        border-left: 5px solid #0047AB !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    
    /* Success boxes */
    .stSuccess {
        background-color: rgba(0, 200, 0, 0.15) !important;
        border-left: 5px solid #00C800 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    
    /* Warning boxes */
    .stWarning {
        background-color: rgba(255, 165, 0, 0.15) !important;
        border-left: 5px solid #FFA500 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    
    /* Error boxes - RED accent */
    .stError {
        background-color: rgba(204, 0, 0, 0.15) !important;
        border-left: 5px solid #CC0000 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar - darker with RED border */
    [data-testid="stSidebar"] {
        background-color: #0d0d0d !important;
        border-right: 3px solid #CC0000 !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #FFFFFF !important;
    }
    
    /* Sidebar headings */
    [data-testid="stSidebar"] h1 {
        font-size: 1.8rem !important;
        border-bottom: 3px solid #CC0000 !important;
        padding-bottom: 10px !important;
    }
    
    [data-testid="stSidebar"] h2 {
        font-size: 1.3rem !important;
        color: #FFFFFF !important;
        border-left: 4px solid #0047AB !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background-color: #2a2a2a !important;
        border: 2px solid #0047AB !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-radius: 6px !important;
    }
    
    .stSelectbox label {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        font-family: 'Roboto Condensed', sans-serif !important;
    }
    
    /* Dropdown menus */
    [data-baseweb="select"] > div {
        background-color: #2a2a2a !important;
        color: #FFFFFF !important;
    }
    
    [data-baseweb="popover"] {
        background-color: #2a2a2a !important;
    }
    
    [role="listbox"] {
        background-color: #2a2a2a !important;
    }
    
    [role="option"] {
        background-color: #2a2a2a !important;
        color: #FFFFFF !important;
    }
    
    [role="option"]:hover {
        background-color: #0047AB !important;
        color: #FFFFFF !important;
    }
    
    /* Input fields */
    input, textarea {
        background-color: #2a2a2a !important;
        color: #FFFFFF !important;
        border: 2px solid #0047AB !important;
        border-radius: 6px !important;
    }
    
    input::placeholder, textarea::placeholder {
        color: #999999 !important;
    }
    
    /* Dividers */
    hr {
        border-color: #CC0000 !important;
        border-width: 2px !important;
        margin: 2rem 0 !important;
    }
    
    /* Captions */
    .caption, [data-testid="stCaptionContainer"] {
        color: #999999 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    
    /* General text */
    p, span, div {
        color: #FFFFFF !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #2a2a2a !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-radius: 6px !important;
    }
    
    /* Code blocks */
    code {
        background-color: #2a2a2a !important;
        color: #FFFFFF !important;
        padding: 4px 8px !important;
        border-radius: 4px !important;
    }
    
    /* Markdown */
    .stMarkdown {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

MODEL_UPDATE_DATE = "2025-10-08"

class ProductionGamePredictor:
    def __init__(self, team_data_file='team_data.json'):
        try:
            with open(team_data_file, 'r') as f:
                self.team_data = json.load(f)
            all_points = [team['points_L4'] for team in self.team_data.values()]
            all_allowed = [team['opp_points_L4'] for team in self.team_data.values()]
            self.league_avg_points = np.mean(all_points)
            self.league_avg_allowed = np.mean(all_allowed)
        except FileNotFoundError:
            st.error("Team data not found. Please run weekly update.")
            self.team_data = {}
            self.league_avg_points = 22.0
            self.league_avg_allowed = 22.0
        
        self.ensemble_models = self._load_ensemble_models()
        
        try:
            with open('defense_rankings.json', 'r') as f:
                self.defense_rankings = json.load(f)
        except FileNotFoundError:
            self.defense_rankings = {}
    
    def _load_ensemble_models(self):
        ensemble_models = {}
        model_types = ['passing', 'receiving', 'receptions', 'rushing', 'game_total', 'game_spread']
        for model_name in model_types:
            models = []
            for idx in range(3):
                path = f'models/{model_name}_model_{idx}.pkl'
                if os.path.exists(path):
                    try:
                        model = joblib.load(path)
                        models.append(model)
                    except:
                        pass
            if len(models) > 0:
                ensemble_models[model_name] = models
            else:
                ensemble_models[model_name] = None
        return ensemble_models
    
    def predict_with_ensemble(self, model_name, features):
        if not self.ensemble_models.get(model_name):
            return None
        models = self.ensemble_models[model_name]
        predictions = []
        for model in models:
            try:
                pred = model.predict(features)[0]
                predictions.append(pred)
            except:
                continue
        if len(predictions) == 0:
            return None
        return np.mean(predictions)
    
    def predict_game(self, team1, team2, home_team=None):
        if self.ensemble_models.get('game_total') and self.ensemble_models.get('game_spread'):
            try:
                if home_team == team1:
                    away_team = team2
                else:
                    away_team = team1
                home_stats = self.team_data.get(home_team, {})
                away_stats = self.team_data.get(away_team, {})
                if not home_stats or not away_stats:
                    raise ValueError("Missing team stats")
                features = np.array([[
                    home_stats.get('points_L4', 22),
                    home_stats.get('opp_points_L4', 22),
                    home_stats.get('yards_L4', 350),
                    home_stats.get('points_L8', 22),
                    home_stats.get('opp_points_L8', 22),
                    home_stats.get('win_pct_L8', 0.5),
                    home_stats.get('turnovers_L4', 1.0),
                    away_stats.get('points_L4', 22),
                    away_stats.get('opp_points_L4', 22),
                    away_stats.get('yards_L4', 350),
                    away_stats.get('points_L8', 22),
                    away_stats.get('opp_points_L8', 22),
                    away_stats.get('win_pct_L8', 0.5),
                    away_stats.get('turnovers_L4', 1.0),
                    7, 7, 0, 0
                ]])
                total_pred = self.predict_with_ensemble('game_total', features)
                spread_pred_raw = self.predict_with_ensemble('game_spread', features)
                if total_pred is not None and spread_pred_raw is not None:
                    try:
                        with open('calibration_params.json', 'r') as f:
                            calib = json.load(f)
                            spread_factor = calib.get('spread_factor', 0.7)
                    except:
                        spread_factor = 0.7
                    spread_pred = spread_pred_raw * spread_factor
                    home_score = (total_pred + spread_pred) / 2
                    away_score = (total_pred - spread_pred) / 2
                    return {
                        'team1': team1,
                        'team1_score': round(away_score if team1 == away_team else home_score, 1),
                        'team2': team2,
                        'team2_score': round(home_score if team2 == home_team else away_score, 1),
                        'total': round(total_pred, 1),
                        'spread': round(spread_pred, 1),
                        'home_team': home_team,
                        'away_team': away_team,
                        'confidence': f'Calibrated ({spread_factor:.2f})',
                        'method': 'ensemble'
                    }
            except:
                pass
        return {
            'team1': team1,
            'team1_score': 22.0,
            'team2': team2,
            'team2_score': 22.0,
            'total': 44.0,
            'spread': 0.0,
            'home_team': home_team,
            'away_team': team1 if team2 == home_team else team2,
            'confidence': 'Fallback',
            'method': 'fallback'
        }
    
    def predict_player_passing(self, player_stats, opponent_team=None):
        if not self.ensemble_models.get('passing'):
            base = player_stats.get('passing_yards_L4', 250)
            return round(base, 1), "Statistical fallback"
        try:
            defense_rank = 16
            if opponent_team and opponent_team in self.defense_rankings:
                defense_rank = self.defense_rankings[opponent_team].get('pass_def_rank', 16)
            features = np.array([[
                player_stats.get('passing_yards_L4', 250),
                player_stats.get('passing_yards_L8', 250),
                player_stats.get('passing_yards_L16', 250),
                player_stats.get('completion_pct_L4', 0.65),
                player_stats.get('completion_pct_L8', 0.65),
                player_stats.get('attempts_L4', 35),
                player_stats.get('attempts_L8', 35),
                player_stats.get('passing_tds_L4', 1.5),
                player_stats.get('passing_tds_L8', 1.5),
                player_stats.get('interception_L4', 0.5),
                defense_rank
            ]])
            prediction = self.predict_with_ensemble('passing', features)
            if prediction:
                return round(prediction, 1), f"Ensemble vs #{int(defense_rank)} pass D"
            else:
                return None, "Failed"
        except Exception as e:
            return None, f"Error: {str(e)[:50]}"
    
    def predict_player_receiving(self, player_stats, opponent_team=None):
        if not self.ensemble_models.get('receiving'):
            base = player_stats.get('receiving_yards_L4', 50)
            return round(base, 1), "Statistical fallback"
        try:
            defense_rank = 16
            if opponent_team and opponent_team in self.defense_rankings:
                defense_rank = self.defense_rankings[opponent_team].get('pass_def_rank', 16)
            features = np.array([[
                player_stats.get('receiving_yards_L4', 50),
                player_stats.get('receiving_yards_L8', 50),
                player_stats.get('receiving_yards_L16', 50),
                player_stats.get('receptions_L4', 4),
                player_stats.get('receptions_L8', 4),
                player_stats.get('yards_per_rec_L4', 12),
                player_stats.get('yards_per_rec_L8', 12),
                player_stats.get('receiving_tds_L4', 0.5),
                player_stats.get('receiving_tds_L8', 0.5),
                player_stats.get('target_share_L4', 0.15),
                player_stats.get('target_share_L8', 0.15),
                defense_rank
            ]])
            prediction = self.predict_with_ensemble('receiving', features)
            if prediction:
                return round(prediction, 1), f"Ensemble vs #{int(defense_rank)} pass D"
            else:
                return None, "Failed"
        except Exception as e:
            return None, f"Error: {str(e)[:50]}"
    
    def predict_player_receptions(self, player_stats, opponent_team=None):
        if not self.ensemble_models.get('receptions'):
            base = player_stats.get('receptions_L4', 4)
            return round(base, 1), "Statistical fallback"
        try:
            defense_rank = 16
            if opponent_team and opponent_team in self.defense_rankings:
                defense_rank = self.defense_rankings[opponent_team].get('pass_def_rank', 16)
            features = np.array([[
                player_stats.get('receptions_L4', 4),
                player_stats.get('receptions_L8', 4),
                player_stats.get('receptions_L16', 4),
                player_stats.get('receiving_yards_L4', 50),
                player_stats.get('yards_per_rec_L4', 12),
                player_stats.get('target_share_L4', 0.15),
                defense_rank
            ]])
            prediction = self.predict_with_ensemble('receptions', features)
            if prediction:
                return round(prediction, 1), f"Ensemble vs #{int(defense_rank)} pass D"
            else:
                return None, "Failed"
        except Exception as e:
            return None, f"Error: {str(e)[:50]}"
    
    def predict_player_rushing(self, player_stats, opponent_team=None):
        if not self.ensemble_models.get('rushing'):
            base = player_stats.get('rushing_yards_L4', 80)
            return round(base, 1), "Statistical fallback"
        try:
            defense_rank = 16
            if opponent_team and opponent_team in self.defense_rankings:
                defense_rank = self.defense_rankings[opponent_team].get('rush_def_rank', 16)
            features = np.array([[
                player_stats.get('rushing_yards_L4', 80),
                player_stats.get('rushing_yards_L8', 80),
                player_stats.get('rushing_yards_L16', 80),
                player_stats.get('attempts_L4', 18),
                player_stats.get('attempts_L8', 18),
                player_stats.get('yards_per_carry_L4', 4.5),
                player_stats.get('yards_per_carry_L8', 4.5),
                player_stats.get('rushing_tds_L4', 0.5),
                player_stats.get('rushing_tds_L8', 0.5),
                defense_rank
            ]])
            prediction = self.predict_with_ensemble('rushing', features)
            if prediction:
                return round(prediction, 1), f"Ensemble vs #{int(defense_rank)} rush D"
            else:
                return None, "Failed"
        except Exception as e:
            return None, f"Error: {str(e)[:50]}"
    
    def get_system_status(self):
        total_models = sum([len(models) if models else 0 for models in self.ensemble_models.values()])
        return {
            'teams_loaded': len(self.team_data),
            'ensemble_models': len([m for m in self.ensemble_models.values() if m]),
            'total_sub_models': total_models,
            'defense_rankings': len(self.defense_rankings),
            'last_update': MODEL_UPDATE_DATE,
            'version': '2.1 - With Injuries'
        }
    
    def list_available_teams(self):
        return sorted(list(self.team_data.keys()))

def load_weekly_report():
    try:
        with open('weekly_report.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_resource
def load_prediction_system():
    return ProductionGamePredictor()

@st.cache_data
def load_player_data():
    data = {}
    files = ['qb_data.json', 'wr_data.json', 'rb_data.json']
    for file in files:
        try:
            if os.path.exists(file):
                with open(file, 'r') as f:
                    data[file.replace('_data.json', '')] = json.load(f)
            else:
                data[file.replace('_data.json', '')] = {}
        except:
            data[file.replace('_data.json', '')] = {}
    return data

st.title("🏈 NFL PREDICTION SYSTEM v2.1")
st.caption(f"INJURY-ADJUSTED ANALYTICS | UPDATED: {MODEL_UPDATE_DATE}")

prediction_system = load_prediction_system()
player_data = load_player_data()
injury_system = InjuryAdjustmentSystem()

system_status = prediction_system.get_system_status()
available_teams = prediction_system.list_available_teams()

st.sidebar.title("⚙️ SYSTEM STATUS")
st.sidebar.success(f"✅ **TEAMS:** {system_status['teams_loaded']}")
st.sidebar.info(f"🤖 **MODELS:** {system_status['ensemble_models']}")
st.sidebar.info(f"📦 **SUB-MODELS:** {system_status['total_sub_models']}")
st.sidebar.caption(f"**VERSION:** {system_status['version']}")

render_injury_manager(injury_system, available_teams)

if system_status['teams_loaded'] == 0:
    st.error("⚠️ SYSTEM REQUIRES DATA. RUN weekly_nfl_update.py")
    st.stop()

page = st.sidebar.selectbox("📊 CHOOSE ANALYSIS", [
    "🎯 Game Predictions",
    "🎲 Player Props",
    "📈 Weekly Report",
    "ℹ️ System Info"
])

if page == "🎯 Game Predictions":
    st.header("🎯 GAME PREDICTIONS")
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("SELECT MATCHUP")
        col_away, col_home = st.columns(2)
        with col_away:
            away_team = st.selectbox("AWAY TEAM", available_teams, key="away")
        with col_home:
            home_team = st.selectbox("HOME TEAM", available_teams, key="home")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🏈 PREDICT GAME", type="primary", use_container_width=True):
            if away_team != home_team:
                with st.spinner("ANALYZING MATCHUP..."):
                    prediction = integrate_injuries_into_game_prediction(
                        prediction_system, injury_system, away_team, home_team, home_team
                    )
                st.markdown("---")
                st.success(f"**{prediction['away_team']} @ {prediction['home_team']}**")
                st.caption(f"METHOD: {prediction.get('method', 'unknown').upper()}")
                col_pred1, col_pred2 = st.columns(2)
                with col_pred1:
                    away_label = prediction['team1'] if prediction['team1'] == prediction['away_team'] else prediction['team2']
                    away_score = prediction['team1_score'] if prediction['team1'] == prediction['away_team'] else prediction['team2_score']
                    st.metric(f"**{away_label}** (AWAY)", f"{away_score} PTS")
                with col_pred2:
                    home_label = prediction['team1'] if prediction['team1'] == prediction['home_team'] else prediction['team2']
                    home_score = prediction['team1_score'] if prediction['team1'] == prediction['home_team'] else prediction['team2_score']
                    st.metric(f"**{home_label}** (HOME)", f"{home_score} PTS")
                st.markdown("<br>", unsafe_allow_html=True)
                col_total1, col_total2 = st.columns(2)
                with col_total1:
                    st.metric("TOTAL POINTS", f"{prediction['total']}")
                with col_total2:
                    if prediction['spread'] > 0:
                        spread_text = f"{prediction['home_team']} by {abs(prediction['spread']):.1f}"
                    else:
                        spread_text = f"{prediction['away_team']} by {abs(prediction['spread']):.1f}"
                    st.metric("SPREAD", spread_text)
                if prediction.get('injury_adjusted'):
                    st.warning(f"⚠️ {prediction['adjustment_note']}")
                    if prediction.get('original_spread'):
                        st.caption(f"ORIGINAL: {prediction['original_spread']:+.1f} → ADJUSTED: {prediction['spread']:+.1f}")
                st.session_state.last_prediction = prediction
            else:
                st.error("⚠️ SELECT DIFFERENT TEAMS")
    with col2:
        st.subheader("BETTING CONTEXT")
        if 'last_prediction' in st.session_state:
            pred = st.session_state.last_prediction
            st.markdown("**ANALYSIS:**")
            st.markdown("<br>", unsafe_allow_html=True)
            total = pred['total']
            if total >= 50:
                st.info("🔥 **HIGH-SCORING** expected")
            elif total <= 38:
                st.info("🛡️ **DEFENSIVE BATTLE** expected")
            else:
                st.info("📊 **AVERAGE SCORING** expected")
            spread = abs(pred['spread'])
            if spread >= 10:
                st.info("💪 **SIGNIFICANT FAVORITE**")
            elif spread <= 3:
                st.info("⚖️ **TOSS-UP GAME**")
            else:
                st.info("📈 **MODERATE FAVORITE**")

elif page == "🎲 Player Props":
    st.header("🎲 PLAYER PROPS")
    st.markdown("---")
    position = st.selectbox("**POSITION**", ["Quarterback", "Wide Receiver/TE", "Running Back"])
    if position == "Quarterback":
        if 'qb' in player_data and player_data['qb']:
            qb_names = sorted(list(player_data['qb'].keys()))
            selected_qb = st.selectbox("**SELECT QB**", qb_names)
            if selected_qb:
                qb_stats = player_data['qb'][selected_qb]
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.subheader(f"📊 {selected_qb}")
                    st.metric("YARDS (L4)", f"{qb_stats['passing_yards_L4']:.0f}")
                    st.metric("YARDS (L8)", f"{qb_stats['passing_yards_L8']:.0f}")
                    st.metric("COMPLETION %", f"{qb_stats['completion_pct_L4']:.1%}")
                    st.caption(f"**TEAM:** {qb_stats['team']}")
                with col2:
                    st.subheader("🎯 PREDICTION")
                    opponent = st.selectbox("**OPPONENT**", available_teams, key="qb_opp")
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🏈 PREDICT PASSING YARDS", type="primary", use_container_width=True):
                        with st.spinner("CALCULATING..."):
                            base_prediction, status = prediction_system.predict_player_passing(qb_stats, opponent)
                            prediction, injury_status = integrate_injuries_into_player_prediction(
                                injury_system, base_prediction, selected_qb, qb_stats['team']
                            )
                        if prediction is not None:
                            st.success(f"## 🎯 {prediction} YARDS")
                            st.caption(status.upper())
                            if "Out" in injury_status or "Doubtful" in injury_status:
                                st.error(f"🚨 {injury_status}")
                            elif "Questionable" in injury_status:
                                st.warning(f"⚠️ {injury_status}")
                            if prediction >= qb_stats['passing_yards_L4'] * 1.1:
                                st.info("📈 **STRONG MATCHUP**")
                            elif prediction <= qb_stats['passing_yards_L4'] * 0.9:
                                st.info("📉 **TOUGH MATCHUP**")
                        else:
                            st.error(f"⚠️ FAILED: {status}")
        else:
            st.warning("⚠️ NO QB DATA AVAILABLE")
    elif position == "Wide Receiver/TE":
        if 'wr' in player_data and player_data['wr']:
            wr_names = sorted(list(player_data['wr'].keys()))
            selected_wr = st.selectbox("**SELECT WR/TE**", wr_names)
            if selected_wr:
                wr_stats = player_data['wr'][selected_wr]
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.subheader(f"📊 {selected_wr}")
                    st.metric("REC YARDS (L4)", f"{wr_stats['receiving_yards_L4']:.0f}")
                    st.metric("RECEPTIONS (L4)", f"{wr_stats['receptions_L4']:.1f}")
                    st.caption(f"**TEAM:** {wr_stats['team']}")
                with col2:
                    st.subheader("🎯 PREDICTIONS")
                    opponent = st.selectbox("**OPPONENT**", available_teams, key="wr_opp")
                    st.markdown("<br>", unsafe_allow_html=True)
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button("REC YARDS", use_container_width=True):
                            base_prediction, status = prediction_system.predict_player_receiving(wr_stats, opponent)
                            prediction, injury_status = integrate_injuries_into_player_prediction(
                                injury_system, base_prediction, selected_wr, wr_stats['team']
                            )
                            if prediction:
                                st.success(f"**{prediction} YDS**")
                                if "Out" in injury_status or "Doubtful" in injury_status:
                                    st.error(f"🚨 {injury_status}")
                            else:
                                st.error(status)
                    with col_btn2:
                        if st.button("RECEPTIONS", use_container_width=True):
                            base_prediction, status = prediction_system.predict_player_receptions(wr_stats, opponent)
                            prediction, injury_status = integrate_injuries_into_player_prediction(
                                injury_system, base_prediction, selected_wr, wr_stats['team']
                            )
                            if prediction:
                                st.success(f"**{prediction} REC**")
                                if "Out" in injury_status or "Doubtful" in injury_status:
                                    st.error(f"🚨 {injury_status}")
                            else:
                                st.error(status)
        else:
            st.warning("⚠️ NO WR DATA AVAILABLE")
    elif position == "Running Back":
        if 'rb' in player_data and player_data['rb']:
            rb_names = sorted(list(player_data['rb'].keys()))
            selected_rb = st.selectbox("**SELECT RB**", rb_names)
            if selected_rb:
                rb_stats = player_data['rb'][selected_rb]
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.subheader(f"📊 {selected_rb}")
                    st.metric("RUSH YARDS (L4)", f"{rb_stats['rushing_yards_L4']:.0f}")
                    st.metric("ATTEMPTS (L4)", f"{rb_stats['attempts_L4']:.1f}")
                    st.caption(f"**TEAM:** {rb_stats['team']}")
                with col2:
                    st.subheader("🎯 PREDICTION")
                    opponent = st.selectbox("**OPPONENT**", available_teams, key="rb_opp")
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🏃 PREDICT RUSHING YARDS", type="primary", use_container_width=True):
                        with st.spinner("CALCULATING..."):
                            base_prediction, status = prediction_system.predict_player_rushing(rb_stats, opponent)
                            prediction, injury_status = integrate_injuries_into_player_prediction(
                                injury_system, base_prediction, selected_rb, rb_stats['team']
                            )
                        if prediction is not None:
                            st.success(f"## 🎯 {prediction} YARDS")
                            st.caption(status.upper())
                            if "Out" in injury_status or "Doubtful" in injury_status:
                                st.error(f"🚨 {injury_status}")
                            elif "Questionable" in injury_status:
                                st.warning(f"⚠️ {injury_status}")
                            if prediction >= rb_stats['rushing_yards_L4'] * 1.15:
                                st.info("📈 **GREAT MATCHUP**")
                            elif prediction <= rb_stats['rushing_yards_L4'] * 0.85:
                                st.info("📉 **TOUGH MATCHUP**")
                        else:
                            st.error(f"⚠️ FAILED: {status}")
        else:
            st.warning("⚠️ NO RB DATA AVAILABLE")

elif page == "📈 Weekly Report":
    st.header("📈 WEEKLY ACCURACY REPORT")
    st.markdown("---")
    weekly_report = load_weekly_report()
    if weekly_report:
        week_num = weekly_report.get('week', 'N/A')
        report_date = weekly_report.get('date', 'N/A')
        st.subheader(f"WEEK {week_num} RESULTS")
        st.caption(f"Generated: {report_date}")
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            games_predicted = weekly_report.get('games_predicted', 0)
            st.metric("GAMES PREDICTED", games_predicted)
        with col2:
            correct_winners = weekly_report.get('correct_winners', 0)
            win_accuracy = (correct_winners / games_predicted * 100) if games_predicted > 0 else 0
            st.metric("WIN ACCURACY", f"{win_accuracy:.1f}%")
        with col3:
            avg_spread_error = weekly_report.get('avg_spread_error', 0)
            st.metric("AVG SPREAD ERROR", f"{avg_spread_error:.1f} pts")
        with col4:
            avg_total_error = weekly_report.get('avg_total_error', 0)
            st.metric("AVG TOTAL ERROR", f"{avg_total_error:.1f} pts")
        st.markdown("---")
        st.subheader("GAME-BY-GAME BREAKDOWN")
        games = weekly_report.get('games', [])
        if games:
            for game in games:
                correct_str = '✅ CORRECT' if game.get('correct_winner') else '❌ INCORRECT'
                with st.expander(f"{game['away_team']} @ {game['home_team']} - {correct_str}"):
                    col_pred, col_actual = st.columns(2)
                    with col_pred:
                        st.markdown("**PREDICTED**")
                        st.write(f"{game['away_team']}: {game['predicted_away_score']} pts")
                        st.write(f"{game['home_team']}: {game['predicted_home_score']} pts")
                        st.write(f"Total: {game['predicted_total']}")
                        st.write(f"Spread: {game['predicted_spread']}")
                    with col_actual:
                        st.markdown("**ACTUAL**")
                        st.write(f"{game['away_team']}: {game['actual_away_score']} pts")
                        st.write(f"{game['home_team']}: {game['actual_home_score']} pts")
                        st.write(f"Total: {game['actual_total']}")
                        st.write(f"Spread: {game['actual_spread']}")
                    st.markdown("**ERRORS**")
                    st.write(f"Spread Error: {abs(game['spread_error']):.1f} pts")
                    st.write(f"Total Error: {abs(game['total_error']):.1f} pts")
        else:
            st.info("No game results available for this week")
        st.markdown("---")
        st.caption("💡 Weekly reports are generated automatically after running weekly_nfl_update.py")
    else:
        st.info("📊 NO WEEKLY REPORT AVAILABLE")
        st.write("Weekly reports are generated automatically when you run the weekly update script.")
        st.write("After Week 1 games are completed, run:")
        st.code("python weekly_nfl_update.py", language="bash")

elif page == "ℹ️ System Info":
    st.header("ℹ️ SYSTEM INFORMATION")
    st.markdown("---")
    
    # Weekly Update Button
    st.subheader("🔄 DATA MANAGEMENT")
    col_update1, col_update2 = st.columns([1, 2])
    
    with col_update1:
        if st.button("🔄 RUN WEEKLY UPDATE", type="primary", use_container_width=True):
            with st.spinner("Running weekly update... This may take 2-3 minutes..."):
                try:
                    import subprocess
                    result = subprocess.run(
                        ['python', 'weekly_nfl_update.py'],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    
                    if result.returncode == 0:
                        st.success("✅ WEEKLY UPDATE COMPLETE!")
                        st.info("Data and models have been updated. Refresh the page to see changes.")
                        
                        # Show update log if available
                        try:
                            with open('update_log.json', 'r') as f:
                                log = json.load(f)
                                st.caption(f"Updated: {log.get('timestamp', 'N/A')}")
                        except:
                            pass
                        
                        # Clear cache to reload new data
                        st.cache_data.clear()
                        st.cache_resource.clear()
                    else:
                        st.error(f"⚠️ UPDATE FAILED")
                        st.code(result.stderr)
                        
                except subprocess.TimeoutExpired:
                    st.error("⚠️ UPDATE TIMED OUT (>5 minutes)")
                except Exception as e:
                    st.error(f"⚠️ ERROR: {str(e)}")
    
    with col_update2:
        st.markdown("""
        **What this does:**
        - Downloads latest NFL data
        - Retrains all prediction models
        - Updates player statistics
        - Generates weekly accuracy report
        - Typically takes 2-3 minutes
        """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("FEATURES")
        st.markdown("""
        ✅ **ENSEMBLE MODELS** (3 per prediction)  
        ✅ **CALIBRATED GAME PREDICTIONS**  
        ✅ **SPLIT DEFENSE RANKINGS**  
        ✅ **INJURY ADJUSTMENTS**  
        ✅ **POINT-IN-TIME STATS** (no leakage)  
        ✅ **WEEKLY ACCURACY REPORTS**
        """)
    
    with col2:
        st.subheader("DATA COVERAGE")
        st.metric("TEAMS", len(available_teams))
        st.metric("QBS", len(player_data.get('qb', {})))
        st.metric("WRS/TES", len(player_data.get('wr', {})))
        st.metric("RBS", len(player_data.get('rb', {})))
    
    st.markdown("---")
    st.subheader("PERFORMANCE TARGETS")
    st.markdown("""
    • **GAME PREDICTIONS:** 55-58% win accuracy  
    • **SPREAD MAE:** ~10 points  
    • **PLAYER PROPS:** Highly accurate vs closing lines
    """)
    st.markdown("---")
    st.subheader("METHODOLOGY")
    st.markdown("""
    **ENSEMBLE APPROACH:** Each prediction uses 3 separate models trained on different data splits  
    **CALIBRATION:** Spread predictions are calibrated to historical performance  
    **INJURY SYSTEM:** Manual tracking with automatic adjustments based on position importance  
    **CROSS-SEASON STATS:** Rolling averages across multiple seasons for stability
    """)
    st.markdown("---")
    st.caption("💡 Last updated: " + MODEL_UPDATE_DATE)

st.sidebar.markdown("---")
st.sidebar.markdown("**🚀 PRODUCTION v2.1**")
st.sidebar.markdown("**INJURY SYSTEM ACTIVE**")
st.sidebar.markdown(f"**UPDATED:** {MODEL_UPDATE_DATE}")