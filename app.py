# app.py - NFL Themed v2.2 with EPA & OL/DL Integration

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
    page_title="NFL Prediction System v2.2",
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

MODEL_UPDATE_DATE = "2025-10-17"

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
        
        # NEW: Load EPA metrics
        try:
            with open('epa_metrics.json', 'r') as f:
                self.epa_metrics = json.load(f)
        except FileNotFoundError:
            self.epa_metrics = {}
        
        # NEW: Load OL/DL rankings
        try:
            with open('ol_dl_rankings.json', 'r') as f:
                self.ol_dl_rankings = json.load(f)
        except FileNotFoundError:
            self.ol_dl_rankings = {}
    
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
    
    def get_epa_for_team(self, team, season=2025):
        """Get EPA metrics for a team"""
        if team not in self.epa_metrics or str(season) not in self.epa_metrics[team]:
            return 0.0, 0.0
        
        epa_data = self.epa_metrics[team][str(season)]
        return epa_data.get('epa_per_play', 0.0), epa_data.get('def_epa', 0.0)
    
    def get_ol_dl_matchup(self, offense_team, defense_team, season=2025):
        """Get OL vs DL matchup info"""
        if offense_team not in self.ol_dl_rankings or defense_team not in self.ol_dl_rankings:
            return None
        
        if str(season) not in self.ol_dl_rankings[offense_team] or str(season) not in self.ol_dl_rankings[defense_team]:
            return None
        
        ol_data = self.ol_dl_rankings[offense_team][str(season)].get('ol', {})
        dl_data = self.ol_dl_rankings[defense_team][str(season)].get('dl', {})
        
        ol_score = ol_data.get('score', 50.0)
        dl_score = dl_data.get('score', 50.0)
        matchup_score = ol_score - dl_score
        
        # Determine advantage
        if matchup_score > 15:
            advantage = 'strong_offense'
            explanation = f"{offense_team} OL dominates {defense_team} DL"
        elif matchup_score > 5:
            advantage = 'offense'
            explanation = f"{offense_team} OL has edge"
        elif matchup_score < -15:
            advantage = 'strong_defense'
            explanation = f"{defense_team} DL dominates - expect pressure"
        elif matchup_score < -5:
            advantage = 'defense'
            explanation = f"{defense_team} DL has edge"
        else:
            advantage = 'neutral'
            explanation = "Evenly matched trenches"
        
        return {
            'matchup_score': matchup_score,
            'advantage': advantage,
            'explanation': explanation,
            'ol_rank': ol_data.get('rank', 16),
            'ol_score': ol_score,
            'dl_rank': dl_data.get('rank', 16),
            'dl_score': dl_score
        }
    
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
                
                # Get EPA metrics
                home_epa, _ = self.get_epa_for_team(home_team)
                
                # Get OL/DL matchup
                ol_dl_info = self.get_ol_dl_matchup(home_team, away_team)
                ol_dl_score = ol_dl_info['matchup_score'] if ol_dl_info else 0.0
                
                # UPDATED: Now 20 features including EPA and OL/DL
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
                    7, 7, 0, 0,  # rest, division
                    home_epa, ol_dl_score  # NEW: EPA and OL/DL
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
                        'method': 'ensemble',
                        'ol_dl_matchup': ol_dl_info  # NEW
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
            'method': 'fallback',
            'ol_dl_matchup': None
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
            'epa_teams': len(self.epa_metrics),
            'ol_dl_teams': len(self.ol_dl_rankings),
            'last_update': MODEL_UPDATE_DATE,
            'version': '2.2 - EPA & OL/DL'
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

st.title("🏈 NFL PREDICTION SYSTEM v2.2")
st.caption(f"WITH EPA & OL/DL ANALYTICS | UPDATED: {MODEL_UPDATE_DATE}")

prediction_system = load_prediction_system()
player_data = load_player_data()
injury_system = InjuryAdjustmentSystem()

system_status = prediction_system.get_system_status()
available_teams = prediction_system.list_available_teams()

st.sidebar.title("⚙️ SYSTEM STATUS")
st.sidebar.success(f"✅ **TEAMS:** {system_status['teams_loaded']}")
st.sidebar.info(f"🤖 **MODELS:** {system_status['ensemble_models']}")
st.sidebar.info(f"📦 **SUB-MODELS:** {system_status['total_sub_models']}")
# NEW: Show EPA and OL/DL status
if system_status['epa_teams'] > 0:
    st.sidebar.success(f"⚡ **EPA METRICS:** {system_status['epa_teams']} teams")
if system_status['ol_dl_teams'] > 0:
    st.sidebar.success(f"🏈 **OL/DL RANKINGS:** {system_status['ol_dl_teams']} teams")
st.sidebar.caption(f"**VERSION:** {system_status['version']}")

render_injury_manager(injury_system, available_teams)

if system_status['teams_loaded'] == 0:
    st.error("⚠️ SYSTEM REQUIRES DATA. RUN weekly_nfl_update.py")
    st.stop()

page = st.sidebar.selectbox("📊 CHOOSE ANALYSIS", [
    "🏈 This Week's Games",
    "🎯 Game Predictions",
    "🎲 Player Props",
    "📈 Weekly Report",
    "🏆 Power Rankings",  # NEW
    "📊 Season History",
    "ℹ️ System Info"
])

if page == "🏈 This Week's Games":
    st.header("🏈 THIS WEEK'S GAMES")
    st.markdown("---")
    
    # Load this week's predictions
    try:
        with open('weekly_schedule.json', 'r') as f:
            schedule = json.load(f)
        
        if not schedule or len(schedule) == 0:
            st.warning("⚠️ No games in schedule")
            schedule = []
    except FileNotFoundError:
        st.error("⚠️ NO SCHEDULE FOUND")
        st.write("Run weekly update to generate predictions:")
        st.code("python weekly_nfl_update.py", language="bash")
        schedule = []
    except Exception as e:
        st.error(f"⚠️ Error loading schedule: {e}")
        schedule = []
    
    # Load Vegas lines (if they exist)
    try:
        with open('vegas_lines.json', 'r') as f:
            vegas_lines = json.load(f)
    except FileNotFoundError:
        vegas_lines = {}
    except Exception as e:
        st.warning(f"Could not load Vegas lines: {e}")
        vegas_lines = {}
    
    if len(schedule) == 0:
        st.info("📊 No games scheduled. Run weekly update script.")
    else:
        # Get current week from schedule
        current_week = schedule[0].get('week', 'N/A')
        st.subheader(f"WEEK {current_week} PREDICTIONS")
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("GAMES", len(schedule))
        with col2:
            avg_total = np.mean([g['total'] for g in schedule])
            st.metric("AVG TOTAL", f"{avg_total:.1f}")
        with col3:
            games_with_lines = sum(1 for g in schedule if g['game_id'] in vegas_lines)
            st.metric("VEGAS LINES", f"{games_with_lines}/{len(schedule)}")
        
        st.markdown("---")
        
        # Add Vegas lines interface
        with st.expander("➕ ADD/UPDATE VEGAS LINES"):
            st.write("**Enter Vegas lines to compare with model predictions**")
            
            game_options = [f"{g['away_team']} @ {g['home_team']}" for g in schedule]
            selected_game_idx = st.selectbox("Select Game", range(len(game_options)), format_func=lambda x: game_options[x])
            
            selected_game = schedule[selected_game_idx]
            game_id = selected_game['game_id']
            
            col_vegas1, col_vegas2 = st.columns(2)
            
            # Get existing Vegas line if available
            existing_line = vegas_lines.get(game_id, {})
            
            with col_vegas1:
                vegas_spread = st.number_input(
                    f"Vegas Spread ({selected_game['home_team']})", 
                    value=existing_line.get('spread', 0.0),
                    step=0.5,
                    help="Positive = home favored, Negative = away favored"
                )
            
            with col_vegas2:
                vegas_total = st.number_input(
                    "Vegas Total (O/U)", 
                    value=existing_line.get('total', 44.0),
                    step=0.5
                )
            
            if st.button("💾 SAVE VEGAS LINE", type="primary"):
                vegas_lines[game_id] = {
                    'spread': float(vegas_spread),
                    'total': float(vegas_total),
                    'updated': datetime.now().isoformat()
                }
                
                with open('vegas_lines.json', 'w') as f:
                    json.dump(vegas_lines, f, indent=2)
                
                st.success("✅ Vegas line saved!")
                st.rerun()
        
        st.markdown("---")
        
        # Display games
        st.subheader("GAMES & PREDICTIONS")
        
        # Sort games by game time
        sorted_schedule = sorted(schedule, key=lambda x: x.get('gameday', ''))
        
        for game in sorted_schedule:
            game_id = game['game_id']
            away = game['away_team']
            home = game['home_team']
            
            # Model predictions
            pred_away = game['away_score']
            pred_home = game['home_score']
            pred_total = game['total']
            pred_spread = game['spread']
            
            # Vegas lines
            vegas_line = vegas_lines.get(game_id, {})
            has_vegas = len(vegas_line) > 0;
            
            with st.container():
                # Game header
                col_header, col_time = st.columns([3, 1])
                with col_header:
                    st.markdown(f"### {away} @ {home}")
                with col_time:
                    game_time = game.get('gametime', 'TBD')
                    game_day = game.get('gameday', '')
                    if game_day:
                        st.caption(f"{game_day} {game_time}")
                
                # Predictions vs Vegas
                col_pred, col_vegas, col_edge = st.columns([2, 2, 1])
                
                with col_pred:
                    st.markdown("**MODEL PREDICTION**")
                    st.write(f"**{away}:** {pred_away}")
                    st.write(f"**{home}:** {pred_home}")
                    # Fix the caption to show correct team with minus sign
                    if pred_spread > 0:
                        spread_display = f"{home} -{pred_spread:.1f}"
                    else:
                        spread_display = f"{away} -{abs(pred_spread):.1f}"
                    st.caption(f"Total: {pred_total} | Spread: {spread_display}")
                
                with col_vegas:
                    if has_vegas:
                        st.markdown("**VEGAS LINE**")
                        vegas_spread_val = vegas_line['spread']
                        vegas_total_val = vegas_line['total']
                        
                        st.write(f"**Spread:** {home} {vegas_spread_val:+.1f}")
                        st.write(f"**Total:** {vegas_total_val}")
                        st.caption(f"Updated: {vegas_line.get('updated', 'N/A')[:10]}")
                    else:
                        st.markdown("**VEGAS LINE**")
                        st.info("No Vegas line entered")
                
                with col_edge:
                    if has_vegas:
                        st.markdown("**EDGE**")
                        
                        # Calculate edges
                        spread_diff = abs(pred_spread - vegas_line['spread'])
                        total_diff = abs(pred_total - vegas_line['total'])
                        
                        # Highlight significant edges
                        if spread_diff >= 3:
                            st.error(f"🔥 {spread_diff:.1f} pts")
                            st.caption("Spread edge")
                        elif spread_diff >= 2:
                            st.warning(f"⚠️ {spread_diff:.1f} pts")
                            st.caption("Spread edge")
                        else:
                            st.success(f"✓ {spread_diff:.1f} pts")
                            st.caption("Spread edge")
                        
                        if total_diff >= 3:
                            st.info(f"📊 {total_diff:.1f} pts")
                            st.caption("Total edge")
                    else:
                        st.write("")
                
                # NEW: Display OL/DL matchup if available
                if 'ol_dl_matchup' in game and game['ol_dl_matchup']:
                    ol_dl = game['ol_dl_matchup']
                    st.markdown("---")
                    st.markdown("**🏈 TRENCH BATTLE**")
                    
                    matchup_col1, matchup_col2 = st.columns(2)
                    with matchup_col1:
                        st.caption(f"**{home} OL:** Rank #{ol_dl.get('ol_rank', '?')} (Score: {ol_dl.get('ol_score', 0):.1f})")
                    with matchup_col2:
                        st.caption(f"**{away} DL:** Rank #{ol_dl.get('dl_rank', '?')} (Score: {ol_dl.get('dl_score', 0):.1f})")
                    
                    advantage = ol_dl.get('advantage', 'neutral')
                    if advantage == 'strong_offense':
                        st.success(f"✅ {ol_dl.get('explanation', '')}")
                    elif advantage == 'offense':
                        st.info(f"ℹ️ {ol_dl.get('explanation', '')}")
                    elif advantage == 'strong_defense':
                        st.error(f"⚠️ {ol_dl.get('explanation', '')}")
                    elif advantage == 'defense':
                        st.warning(f"⚠️ {ol_dl.get('explanation', '')}")
                    else:
                        st.caption(f"⚖️ {ol_dl.get('explanation', '')}")
                
                st.markdown("---")
        
        # Best bets section
        st.markdown("---")
        st.subheader("💰 BEST BETS (3+ POINT EDGE)")
        
        best_bets = []
        for game in sorted_schedule:
            game_id = game['game_id']
            if game_id in vegas_lines:
                vegas = vegas_lines[game_id]
                spread_diff = abs(game['spread'] - vegas['spread'])
                total_diff = abs(game['total'] - vegas['total'])
                
                if spread_diff >= 3:
                    best_bets.append({
                        'game': f"{game['away_team']} @ {game['home_team']}",
                        'type': 'SPREAD',
                        'model': f"{game['home_team']} {game['spread']:+.1f}",
                        'vegas': f"{game['home_team']} {vegas['spread']:+.1f}",
                        'edge': spread_diff
                    })
                
                if total_diff >= 3:
                    best_bets.append({
                        'game': f"{game['away_team']} @ {game['home_team']}",
                        'type': 'TOTAL',
                        'model': f"{game['total']:.1f}",
                        'vegas': f"{vegas['total']:.1f}",
                        'edge': total_diff
                    })
        
        if len(best_bets) > 0:
            for bet in sorted(best_bets, key=lambda x: x['edge'], reverse=True):
                col1, col2, col3, col4, col5 = st.columns([3, 1, 2, 2, 1])
                
                with col1:
                    st.write(f"**{bet['game']}**")
                with col2:
                    st.write(bet['type'])
                with col3:
                    st.write(f"Model: {bet['model']}")
                with col4:
                    st.write(f"Vegas: {bet['vegas']}")
                with col5:
                    st.error(f"🔥 {bet['edge']:.1f}")
        else:
            st.info("No significant edges found. Add Vegas lines to see opportunities!")

elif page == "🎯 Game Predictions":
    st.header("🎯 CUSTOM GAME PREDICTION")
    
    col1, col2 = st.columns(2)
    
    with col1:
        away_team = st.selectbox("🏈 AWAY TEAM", available_teams, key='away')
    
    with col2:
        home_team = st.selectbox("🏠 HOME TEAM", available_teams, key='home')
    
    if st.button("🔮 PREDICT GAME", type="primary"):
        if away_team == home_team:
            st.error("❌ TEAMS MUST BE DIFFERENT")
        else:
            with st.spinner("⚙️ RUNNING PREDICTION..."):
                base_prediction = prediction_system.predict_game(away_team, home_team, home_team=home_team)
                
                # Apply injury adjustments
                prediction = integrate_injuries_into_game_prediction(
                    base_prediction, 
                    injury_system, 
                    away_team, 
                    home_team
                )
                
                st.success("✅ PREDICTION COMPLETE")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(f"{away_team} Score", f"{prediction['team1_score']:.1f}")
                
                with col2:
                    st.metric(f"{home_team} Score", f"{prediction['team2_score']:.1f}")
                
                with col3:
                    st.metric("Total Points", f"{prediction['total']:.1f}")
                
                with col4:
                    spread_value = prediction['spread']
                    spread_display = f"{home_team} {spread_value:+.1f}"
                    st.metric("Spread", spread_display)
                
                st.caption(f"**METHOD:** {prediction.get('confidence', 'N/A')} | {prediction.get('method', 'ensemble').upper()}")
                
                # NEW: Display OL/DL matchup analysis
                if prediction.get('ol_dl_matchup'):
                    st.markdown("---")
                    st.markdown("### 🏈 TRENCH WARFARE ANALYSIS")
                    
                    ol_dl = prediction['ol_dl_matchup']
                    
                    trench_col1, trench_col2, trench_col3 = st.columns(3)
                    
                    with trench_col1:
                        st.metric(f"{home_team} OL Rank", f"#{ol_dl['ol_rank']}")
                        st.caption(f"Score: {ol_dl['ol_score']:.1f}")
                    
                    with trench_col2:
                        st.metric("Matchup Score", f"{ol_dl['matchup_score']:+.1f}")
                        advantage = ol_dl['advantage']
                        if advantage in ['strong_offense', 'offense']:
                            st.success("Offense has edge")
                        elif advantage in ['strong_defense', 'defense']:
                            st.error("Defense has edge")
                        else:
                            st.info("Even matchup")
                    
                    with trench_col3:
                        st.metric(f"{away_team} DL Rank", f"#{ol_dl['dl_rank']}")
                        st.caption(f"Score: {ol_dl['dl_score']:.1f}")
                    
                    st.info(f"**📊 ANALYSIS:** {ol_dl['explanation']}")
                
                # Show injury adjustments if any
                if prediction.get('injury_adjusted'):
                    st.markdown("---")
                    st.warning("⚠️ **INJURY ADJUSTMENTS APPLIED**")
                    
                    if prediction.get('away_adjustments'):
                        st.markdown(f"**{away_team}:**")
                        for adj in prediction['away_adjustments']:
                            st.caption(f"• {adj}")
                    
                    if prediction.get('home_adjustments'):
                        st.markdown(f"**{home_team}:**")
                        for adj in prediction['home_adjustments']:
                            st.caption(f"• {adj}")

elif page == "🎲 Player Props":
    st.header("🎲 PLAYER PROP PREDICTIONS")
    
    prop_type = st.radio(
        "📋 SELECT PROP TYPE",
        ["Passing Yards", "Receiving Yards", "Receptions", "Rushing Yards"],
        horizontal=True
    )
    
    if prop_type == "Passing Yards":
        player_list = list(player_data.get('qb', {}).keys())
        position = 'qb'
        pred_func = prediction_system.predict_player_passing
    elif prop_type == "Receiving Yards":
        player_list = list(player_data.get('wr', {}).keys())
        position = 'wr'
        pred_func = prediction_system.predict_player_receiving
    elif prop_type == "Receptions":
        player_list = list(player_data.get('wr', {}).keys())
        position = 'wr'
        pred_func = prediction_system.predict_player_receptions
    else:
        player_list = list(player_data.get('rb', {}).keys())
        position = 'rb'
        pred_func = prediction_system.predict_player_rushing
    
    if not player_list:
        st.warning(f"⚠️ NO {position.upper()} DATA AVAILABLE")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        player_name = st.selectbox("🏃 SELECT PLAYER", sorted(player_list))
    
    with col2:
        opponent = st.selectbox("🏈 OPPONENT", ['League Average'] + available_teams)
    
    if st.button("🔮 PREDICT PERFORMANCE", type="primary"):
        player_stats = player_data[position][player_name]
        opp_team = None if opponent == 'League Average' else opponent
        
        with st.spinner("⚙️ CALCULATING..."):
            base_prediction, method = pred_func(player_stats, opp_team)
            
            if base_prediction is None:
                st.error(f"❌ PREDICTION FAILED: {method}")
            else:
                # Apply injury adjustments
                prediction, injury_note = integrate_injuries_into_player_prediction(
                    base_prediction,
                    injury_system,
                    player_name,
                    player_stats.get('team', ''),
                   
                )
                
                st.success("✅ PREDICTION COMPLETE")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("PREDICTION", f"{prediction:.1f}")
                
                with col2:
                    recent_avg = player_stats.get(f"{prop_type.lower().replace(' ', '_')}_L4", 0)
                    if recent_avg > 0:
                        diff = prediction - recent_avg
                        st.metric("vs L4 AVG", f"{diff:+.1f}", delta=f"{diff:+.1f}")
                
                with col3:
                    if opp_team and opp_team in prediction_system.defense_rankings:
                        def_rank = prediction_system.defense_rankings[opp_team].get('pass_def_rank' if prop_type != "Rushing Yards" else 'rush_def_rank', 16)
                        st.metric("Opp Def Rank", f"#{int(def_rank)}")
                
                st.caption(f"**METHOD:** {method}")
                
                if injury_note:
                    st.warning(f"⚠️ **INJURY ADJUSTMENT:** {injury_note}")
                
                # Show recent performance
                st.markdown("---")
                st.markdown("### 📊 RECENT PERFORMANCE")
                
                perf_col1, perf_col2, perf_col3 = st.columns(3)
                
                with perf_col1:
                    l4_key = f"{prop_type.lower().replace(' ', '_')}_L4"
                    st.metric("Last 4 Games", f"{player_stats.get(l4_key, 0):.1f}")
                
                with perf_col2:
                    l8_key = f"{prop_type.lower().replace(' ', '_')}_L8"
                    st.metric("Last 8 Games", f"{player_stats.get(l8_key, 0):.1f}")
                
                with perf_col3:
                    l16_key = f"{prop_type.lower().replace(' ', '_')}_L16"
                    if l16_key in player_stats:
                        st.metric("Last 16 Games", f"{player_stats.get(l16_key, 0):.1f}")

elif page == "📈 Weekly Report":
    st.header("📈 WEEKLY ANALYSIS REPORT")
    
    weekly_report = load_weekly_report()
    
    if not weekly_report:
        st.warning("⚠️ NO WEEKLY REPORT AVAILABLE. RUN weekly_nfl_update.py")
        st.stop()
    
    st.info(f"📅 **WEEK {weekly_report.get('week', 'N/A')}** | Generated: {weekly_report.get('generated_date', 'N/A')}")
    
    if 'betting_opportunities' in weekly_report:
        st.markdown("## 💰 BETTING OPPORTUNITIES")
        
        for opp in weekly_report['betting_opportunities']:
            with st.expander(f"🎯 {opp['game']}", expanded=True):
                st.markdown(f"**TYPE:** {opp['type']}")
                st.markdown(f"**ANGLE:** {opp['angle']}")
                st.markdown(f"**CONFIDENCE:** {opp['confidence']}")
                
                if 'prediction' in opp:
                    pred_col1, pred_col2 = st.columns(2)
                    with pred_col1:
                        st.metric("Predicted Total", f"{opp['prediction']:.1f}")
                    with pred_col2:
                        if 'vegas_line' in opp:
                            st.metric("Vegas Line", f"{opp['vegas_line']:.1f}")
    
    if 'top_performers' in weekly_report:
        st.markdown("---")
        st.markdown("## ⭐ PROJECTED TOP PERFORMERS")
        
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        with perf_col1:
            st.markdown("### 🎯 QUARTERBACKS")
            for qb in weekly_report['top_performers'].get('qbs', [])[:5]:
                st.metric(qb['name'], f"{qb['prediction']:.1f} yds", f"vs {qb['opponent']}")
        
        with perf_col2:
            st.markdown("### 🏃 RECEIVERS")
            for wr in weekly_report['top_performers'].get('wrs', [])[:5]:
                st.metric(wr['name'], f"{wr['prediction']:.1f} yds", f"vs {wr['opponent']}")
        
        with perf_col3:
            st.markdown("### 💨 RUNNING BACKS")
            for rb in weekly_report['top_performers'].get('rbs', [])[:5]:
                st.metric(rb['name'], f"{rb['prediction']:.1f} yds", f"vs {rb['opponent']}")

elif page == "🏆 Power Rankings":
    st.header("🏆 NFL POWER RANKINGS")
    st.caption("Teams ranked by predicted spread vs average NFL team on neutral field")
    st.markdown("---")
    
    # Calculate power rankings for each team
    power_rankings = []
    
    for team in available_teams:
        team_stats = prediction_system.team_data.get(team, {})
        
        if not team_stats:
            continue
        
        # Get EPA metrics
        epa_off, epa_def = prediction_system.get_epa_for_team(team, 2025)
        
        # Get OL/DL rankings
        ol_rank = 16
        ol_score = 50.0
        dl_rank = 16
        dl_score = 50.0
        
        if team in prediction_system.ol_dl_rankings and '2025' in prediction_system.ol_dl_rankings[team]:
            ol_data = prediction_system.ol_dl_rankings[team]['2025'].get('ol', {})
            dl_data = prediction_system.ol_dl_rankings[team]['2025'].get('dl', {})
            ol_rank = ol_data.get('rank', 16)
            ol_score = ol_data.get('score', 50.0)
            dl_rank = dl_data.get('rank', 16)
            dl_score = dl_data.get('score', 50.0)
        
        # Calculate power rating using offensive/defensive efficiency
        points_scored = team_stats.get('points_L4', 22.0)
        points_allowed = team_stats.get('opp_points_L4', 22.0)
        win_pct = team_stats.get('win_pct_L8', 0.5)
        
        # Calculate how much better/worse than league average (22 ppg)
        offensive_advantage = (points_scored - 22.0) * 0.4
        defensive_advantage = (22.0 - points_allowed) * 0.4
        
        # Add EPA contribution (scaled) - BOTH offense AND defense
        epa_off_contribution = epa_off * 8  # Offensive EPA
        epa_def_contribution = -epa_def * 8  # Defensive EPA (negative is good, so flip it)
        
        # Add OL/DL contribution (scaled)
        ol_contribution = (ol_score - 50.0) * 0.05
        dl_contribution = (dl_score - 50.0) * 0.05
        
        # Add win percentage adjustment
        win_adjustment = (win_pct - 0.5) * 8  # 0.25 difference = 2 points
        
        # Combine all factors
        power_rating = (offensive_advantage + defensive_advantage + 
                       epa_off_contribution + epa_def_contribution + 
                       ol_contribution + dl_contribution + 
                       win_adjustment)
        
        
        power_rankings.append({
            'team': team,
            'power_rating': power_rating,
            'record': f"{int(team_stats.get('win_pct_L8', 0.5) * 8)}-{8 - int(team_stats.get('win_pct_L8', 0.5) * 8)}",
            'points_L4': team_stats.get('points_L4', 22.0),
            'opp_points_L4': team_stats.get('opp_points_L4', 22.0),
            'epa_off': epa_off,
            'epa_def': epa_def,
            'ol_rank': ol_rank,
            'ol_score': ol_score,
            'dl_rank': dl_rank,
            'dl_score': dl_score
        })
    
    # Sort by power rating (descending)
    power_rankings.sort(key=lambda x: x['power_rating'], reverse=True)
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_team = power_rankings[0]
        st.metric("🥇 BEST TEAM", best_team['team'])
        st.caption(f"Would be favored by {best_team['power_rating']:.1f} pts vs average team")
    
    with col2:
        worst_team = power_rankings[-1]
        st.metric("📉 WORST TEAM", worst_team['team'])
        st.caption(f"Would be underdog by {abs(worst_team['power_rating']):.1f} pts vs average team")
    
    with col3:
        avg_rating = np.mean([t['power_rating'] for t in power_rankings])
        st.metric("SPREAD RANGE", f"{best_team['power_rating'] - worst_team['power_rating']:.1f} pts")
        st.caption(f"Difference between best and worst")
    
    st.markdown("---")
    
    # Tier breakdown
    st.subheader("📊 TIER BREAKDOWN")
    
    tier_col1, tier_col2, tier_col3, tier_col4 = st.columns(4)
    
    elite = [t for t in power_rankings if t['power_rating'] >= 7]
    good = [t for t in power_rankings if 3 <= t['power_rating'] < 7]
    average = [t for t in power_rankings if -3 <= t['power_rating'] < 3]
    poor = [t for t in power_rankings if t['power_rating'] < -3]
    
    with tier_col1:
        st.metric("🔥 ELITE", len(elite))
        st.caption("≥ 7 pt favorites")
    
    with tier_col2:
        st.metric("💪 GOOD", len(good))
        st.caption("3-7 pt favorites")
    
    with tier_col3:
        st.metric("📊 AVERAGE", len(average))
        st.caption("±3 pts")
    
    with tier_col4:
        st.metric("📉 POOR", len(poor))
        st.caption("< -3 pts")
    
    st.markdown("---")
    
    # Display full rankings table
    st.subheader("🏅 FULL RANKINGS")
    
    # Create rankings display
    for idx, team_data in enumerate(power_rankings, 1):
        # Determine tier badge and color
        rating = team_data['power_rating']
        if rating >= 7:
            tier_badge = "🔥"
            header_color = "#00C800"  # Green
        elif rating >= 3:
            tier_badge = "💪"
            header_color = "#0047AB"  # Blue
        elif rating >= -3:
            tier_badge = "📊"
            header_color = "#FFA500"  # Orange
        else:
            tier_badge = "📉"
            header_color = "#CC0000"  # Red
        
        # Add visual separator between teams
        if idx > 1:
            st.markdown("<br>", unsafe_allow_html=True)
        
        # Team header with colored border
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, {header_color}22 0%, #1a1a1a 100%); 
                    border-left: 6px solid {header_color}; 
                    padding: 20px; 
                    margin: 15px 0; 
                    border-radius: 8px;">
            <h2 style="margin: 0; color: white; font-family: 'Roboto Condensed', sans-serif;">
                <span style="color: {header_color}; font-size: 1.2em;">#{idx}</span> 
                {tier_badge} 
                <span style="font-weight: 900;">{team_data['team']}</span>
                <span style="color: {header_color}; font-size: 0.9em; margin-left: 15px;">({team_data['power_rating']:+.1f})</span>
            </h2>
            <p style="margin: 8px 0 0 0; color: #CCCCCC; font-size: 0.9em;">
                Record: {team_data['record']} | Power Rating vs Average Team: {team_data['power_rating']:+.1f} pts
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Main metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Power Rating", f"{team_data['power_rating']:+.1f}")
            st.caption("vs Avg Team")
        
        with metric_col2:
            st.metric("Record (L8)", team_data['record'])
        
        with metric_col3:
            st.metric("PPG (L4)", f"{team_data['points_L4']:.1f}")
            st.caption(f"Allow: {team_data['opp_points_L4']:.1f}")
        
        with metric_col4:
            net_points = team_data['points_L4'] - team_data['opp_points_L4']
            st.metric("Point Diff", f"{net_points:+.1f}")
        
        # Expandable advanced metrics
        with st.expander("⚡ VIEW ADVANCED METRICS", expanded=False):
            adv_col1, adv_col2, adv_col3 = st.columns(3)
            
            with adv_col1:
                st.markdown("**EPA METRICS**")
                st.markdown("---")
                
                st.markdown("**Offensive EPA**")
                epa_off = team_data['epa_off']
                if epa_off > 0.1:
                    st.success(f"{epa_off:+.3f} EPA/Play")
                    st.caption("✅ Above average offense")
                elif epa_off < -0.1:
                    st.error(f"{epa_off:+.3f} EPA/Play")
                    st.caption("❌ Below average offense")
                else:
                    st.info(f"{epa_off:+.3f} EPA/Play")
                    st.caption("📊 Average offense")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown("**Defensive EPA**")
                epa_def = team_data['epa_def']
                if epa_def < -0.1:
                    st.success(f"{epa_def:+.3f} EPA/Play")
                    st.caption("✅ Above average defense")
                elif epa_def > 0.1:
                    st.error(f"{epa_def:+.3f} EPA/Play")
                    st.caption("❌ Below average defense")
                else:
                    st.info(f"{epa_def:+.3f} EPA/Play")
                    st.caption("📊 Average defense")
            
            with adv_col2:
                st.markdown("**OFFENSIVE LINE**")
                st.markdown("---")
                st.metric("National Rank", f"#{int(team_data['ol_rank'])}")
                st.metric("OL Score", f"{team_data['ol_score']:.1f}")
                
                if team_data['ol_rank'] <= 10:
                    st.caption("✅ Elite pass protection")
                elif team_data['ol_rank'] <= 20:
                    st.caption("📊 Average pass protection")
                else:
                    st.caption("❌ Struggles in pass pro")
            
            with adv_col3:
                st.markdown("**DEFENSIVE LINE**")
                st.markdown("---")
                st.metric("National Rank", f"#{int(team_data['dl_rank'])}")
                st.metric("DL Score", f"{team_data['dl_score']:.1f}")
                
                if team_data['dl_rank'] <= 10:
                    st.caption("✅ Elite pass rush")
                elif team_data['dl_rank'] <= 20:
                    st.caption("📊 Average pass rush")
                else:
                    st.caption("❌ Struggles to pressure QB")
        
        # Divider between teams
        st.markdown("---")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 **Power Rating:** Predicted point spread if this team played an average NFL team on a neutral field. Positive = team would be favored, Negative = team would be underdog")


elif page == "📊 Season History":
    st.header("📊 SEASON PERFORMANCE HISTORY")
    
    try:
        with open('prediction_history.json', 'r') as f:
            history = json.load(f)
        
        if not history or 'games' not in history:
            st.warning("⚠️ NO HISTORICAL DATA AVAILABLE")
            st.stop()
        
        games = history['games']
        
        # Calculate statistics
        total_games = len(games)
        
        spread_correct = sum(1 for g in games if g.get('spread_correct', False))
        total_correct = sum(1 for g in games if g.get('total_correct', False))
        winner_correct = sum(1 for g in games if g.get('winner_correct', False))
        
        spread_acc = (spread_correct / total_games * 100) if total_games > 0 else 0
        total_acc = (total_correct / total_games * 100) if total_games > 0 else 0
        winner_acc = (winner_correct / total_games * 100) if total_games > 0 else 0
        
        # Display overall stats
        st.markdown("## 🎯 OVERALL ACCURACY")
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            st.metric("Total Games", total_games)
        
        with stat_col2:
            st.metric("Spread Accuracy", f"{spread_acc:.1f}%", f"{spread_correct}/{total_games}")
        
        with stat_col3:
            st.metric("Total Accuracy", f"{total_acc:.1f}%", f"{total_correct}/{total_games}")
        
        with stat_col4:
            st.metric("Winner Accuracy", f"{winner_acc:.1f}%", f"{winner_correct}/{total_games}")
        
        # Detailed game results
        st.markdown("---")
        st.markdown("## 📋 DETAILED RESULTS")
        
        for game in sorted(games, key=lambda x: x.get('date', ''), reverse=True)[:20]:
            with st.expander(f"{game.get('date', 'N/A')} - {game['away_team']} @ {game['home_team']}"):
                result_col1, result_col2, result_col3 = st.columns(3)
                
                with result_col1:
                    st.markdown("**PREDICTED**")
                    st.caption(f"{game['away_team']}: {game['predicted_away_score']:.1f}")
                    st.caption(f"{game['home_team']}: {game['predicted_home_score']:.1f}")
                    st.caption(f"Total: {game['predicted_total']:.1f}")
                    st.caption(f"Spread: {game['predicted_spread']:+.1f}")
                
                with result_col2:
                    st.markdown("**ACTUAL**")
                    st.caption(f"{game['away_team']}: {game.get('actual_away_score', 'N/A')}")
                    st.caption(f"{game['home_team']}: {game.get('actual_home_score', 'N/A')}")
                    st.caption(f"Total: {game.get('actual_total', 'N/A')}")
                    st.caption(f"Spread: {game.get('actual_spread', 'N/A')}")
                
                with result_col3:
                    st.markdown("**RESULTS**")
                    if game.get('spread_correct'):
                        st.success("✅ Spread")
                    else:
                        st.error("❌ Spread")
                    
                    if game.get('total_correct'):
                        st.success("✅ Total")
                    else:
                        st.error("❌ Total")
                    
                    if game.get('winner_correct'):
                        st.success("✅ Winner")
                    else:
                        st.error("❌ Winner")
    
    except FileNotFoundError:
        st.warning("⚠️ NO PREDICTION HISTORY FILE FOUND")

elif page == "ℹ️ System Info":
    st.header("ℹ️ SYSTEM INFORMATION")
    
    st.markdown("## 🤖 MODEL ARCHITECTURE")
    
    st.markdown("""
    **VERSION 2.2** introduces advanced analytics:
    - ⚡ **EPA (Expected Points Added)** metrics for offensive efficiency
    - 🏈 **OL/DL Rankings** for trench warfare analysis
    - 🎯 **Enhanced game predictions** with line battle context
    
    The system uses **ensemble machine learning** with 3 models per prediction type:
    - 🎯 **Game Totals & Spreads:** 20-feature models with EPA and OL/DL data
    - 🏈 **Player Props:** Position-specific models with matchup analysis
    - 🛡️ **Defense Rankings:** Dynamic opponent strength ratings
    - 🤕 **Injury System:** Real-time adjustments for player availability
    """)
    
    st.markdown("---")
    st.markdown("## 📊 CURRENT STATUS")
    
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        st.metric("Teams Loaded", system_status['teams_loaded'])
        st.metric("Ensemble Models", system_status['ensemble_models'])
        st.metric("Total Sub-Models", system_status['total_sub_models'])
    
    with status_col2:
        st.metric("Defense Rankings", system_status['defense_rankings'])
        st.metric("EPA Teams", system_status['epa_teams'])
        st.metric("OL/DL Teams", system_status['ol_dl_teams'])
    
    st.caption(f"Last Update: {system_status['last_update']}")
    
    st.markdown("---")
    st.markdown("## 📚 FEATURE DETAILS")
    
    with st.expander("⚡ EPA METRICS"):
        st.markdown("""
        **Expected Points Added (EPA)** measures offensive efficiency:
        - Positive EPA = efficient offense
        - Negative EPA = struggling offense
        - Integrated into game prediction models
        - Helps identify high-powered vs low-powered offenses
        """)
    
    with st.expander("🏈 OL/DL RANKINGS"):
        st.markdown("""
        **Offensive Line vs Defensive Line** matchup analysis:
        - OL Score: Pass/run blocking effectiveness
        - DL Score: Pressure and run stopping ability
        - Matchup Score: OL - DL difference
        - Helps predict QB protection and running game success
        
        **Matchup Interpretation:**
        - +15 or more: Strong offensive line advantage
        - +5 to +15: Moderate offensive edge
        - -5 to +5: Even matchup
        - -15 to -5: Moderate defensive edge  
        - -15 or worse: Strong defensive line dominance
        """)
    
    with st.expander("🎯 GAME PREDICTION MODEL"):
        st.markdown("""
        **20-Feature Ensemble Model:**
        - Home/Away offensive stats (points, yards, turnovers)
        - Recent form (L4 and L8 game averages)
        - Win percentages and momentum
        - Rest days and division rivalry factors
        - ⚡ EPA offensive efficiency (NEW)
        - 🏈 OL/DL matchup score (NEW)
        
        **Output:** Total points and point spread with calibration
        """)
    
    with st.expander("🏃 PLAYER PREDICTION MODELS"):
        st.markdown("""
        **Position-Specific Models:**
        
        **QB Model (11 features):**
        - Passing yards (L4, L8, L16)
        - Completion percentage trends
        - Attempt volume
        - TD and INT rates
        - Opponent pass defense rank
        
        **WR Model (12 features):**
        - Receiving yards (L4, L8, L16)
        - Reception totals and trends
        - Yards per reception
        - Target share
        - TD rate
        - Opponent pass defense rank
        
        **RB Model (10 features):**
        - Rushing yards (L4, L8, L16)
        - Attempt volume
        - Yards per carry
        - TD rate
        - Opponent rush defense rank
        """)
    
    with st.expander("🤕 INJURY ADJUSTMENT SYSTEM"):
        st.markdown("""
        **Dynamic Injury Management:**
        - Manual injury tracking by position
        - Severity levels: Minor (5%), Moderate (15%), Severe (30%)
        - Automatic prediction adjustments
        - Team-level game score impacts
        - Player-level prop adjustments
        
        **Integration:**
        - Game predictions account for key player absences
        - Player props automatically adjusted for injury status
        - Visual warnings when adjustments applied
        """)
    
    st.markdown("---")
    st.markdown("## 🔄 UPDATE WORKFLOW")
    
    st.markdown("""
    **Weekly Update Process:**
    1. 📥 Fetch latest NFL data (nfl_data_fetcher.py)
    2. 🔄 Update team statistics and rankings
    3. ⚡ Calculate EPA metrics
    4. 🏈 Update OL/DL rankings
    5. 🤖 Retrain all ensemble models
    6. 📊 Generate weekly predictions
    7. 💾 Save to JSON files
    8. 🚀 Deploy to Streamlit
    
    **Run:** `python weekly_nfl_update.py`
    """)
    
    st.markdown("---")
    st.markdown("## ⚙️ TECHNICAL STACK")
    
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.markdown("""
        **Machine Learning:**
        - Scikit-learn ensemble models
        - 3-model voting for robustness
        - Feature engineering pipeline
        - Model calibration system
        """)
    
    with tech_col2:
        st.markdown("""
        **Data & Deployment:**
        - JSON data storage
        - Streamlit web interface
        - Real-time prediction engine
        - Modular Python architecture
        """)
    
    st.markdown("---")
    st.markdown("## 📈 PERFORMANCE TARGETS")
    
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    
    with perf_col1:
        st.markdown("### 🎯 SPREAD")
        st.markdown("**Target:** 52.8%+")
        st.caption("Break-even for -110 odds")
    
    with perf_col2:
        st.markdown("### 📊 TOTAL")
        st.markdown("**Target:** 52.8%+")
        st.caption("Break-even for -110 odds")
    
    with perf_col3:
        st.markdown("### 🏆 WINNER")
        st.markdown("**Target:** 65%+")
        st.caption("Straight up accuracy")
    
    st.markdown("---")
    st.info("💡 **TIP:** Run `python weekly_nfl_update.py` every Monday to refresh all data and predictions for the upcoming week.")