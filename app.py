# nfl_app.py - Clean v2.1 with Injury System

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

MODEL_UPDATE_DATE = "2025-10-08"

class ProductionGamePredictor:
    def __init__(self, team_data_file='team_data.json'):
        """Production-ready NFL prediction system with ensemble models"""
        
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
        """Load ensemble models (3 sub-models each)"""
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
                    except Exception as e:
                        pass
            
            if len(models) > 0:
                ensemble_models[model_name] = models
            else:
                ensemble_models[model_name] = None
        
        return ensemble_models
    
    def predict_with_ensemble(self, model_name, features):
        """Make prediction using ensemble averaging"""
        if not self.ensemble_models.get(model_name):
            return None
        
        models = self.ensemble_models[model_name]
        predictions = []
        
        for model in models:
            try:
                pred = model.predict(features)[0]
                predictions.append(pred)
            except Exception as e:
                continue
        
        if len(predictions) == 0:
            return None
        
        return np.mean(predictions)
    
    def predict_game(self, team1, team2, home_team=None):
        """Game outcome prediction using ENSEMBLE models with calibration"""
        
        if self.ensemble_models.get('game_total') and self.ensemble_models.get('game_spread'):
            try:
                home_stats = self.team_data.get(home_team, {}) if home_team else {}
                away_team = team1 if team2 == home_team else team2
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
                        'confidence': f'Calibrated ({spread_factor:.2f})',
                        'method': 'ensemble'
                    }
            except Exception as e:
                pass
        
        # Fallback
        return {
            'team1': team1,
            'team1_score': 22.0,
            'team2': team2,
            'team2_score': 22.0,
            'total': 44.0,
            'spread': 0.0,
            'confidence': 'Fallback',
            'method': 'fallback'
        }
    
    def predict_player_passing(self, player_stats, opponent_team=None):
        """QB prediction with ENSEMBLE models"""
        
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
        """WR prediction with ENSEMBLE models"""
        
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
        """Receptions prediction with ENSEMBLE models"""
        
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
        """RB prediction with ENSEMBLE models"""
        
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
        """System health status"""
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
        """List all available teams"""
        return sorted(list(self.team_data.keys()))

# Load system
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

# Initialize
st.title("🏈 NFL Prediction System v2.1")
st.caption(f"With Injury Adjustments | Updated: {MODEL_UPDATE_DATE}")

prediction_system = load_prediction_system()
player_data = load_player_data()
injury_system = InjuryAdjustmentSystem()

system_status = prediction_system.get_system_status()
available_teams = prediction_system.list_available_teams()

# Sidebar
st.sidebar.title("System Status v2.1")
st.sidebar.success(f"✅ Teams: {system_status['teams_loaded']}")
st.sidebar.info(f"🤖 Models: {system_status['ensemble_models']}")
st.sidebar.info(f"📦 Sub-Models: {system_status['total_sub_models']}")
st.sidebar.caption(system_status['version'])

# Add injury manager to sidebar
render_injury_manager(injury_system, available_teams)

if system_status['teams_loaded'] == 0:
    st.error("System requires data. Run weekly_nfl_update.py")
    st.stop()

# Navigation
page = st.sidebar.selectbox("Choose Analysis", [
    "Game Predictions",
    "Player Props",
    "System Info"
])

if page == "Game Predictions":
    st.header("🎯 Game Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Select Matchup")
        away_team = st.selectbox("Away Team", available_teams, key="away")
        home_team = st.selectbox("Home Team", available_teams, key="home")
        
        if st.button("Predict Game", type="primary"):
            if away_team != home_team:
                prediction = integrate_injuries_into_game_prediction(
                    prediction_system,
                    injury_system, 
                    away_team,
                    home_team,
                    home_team
                )
                
                st.success(f"**{prediction['team1']} @ {prediction['team2']}**")
                st.caption(f"Method: {prediction.get('method', 'unknown')}")
                
                col_pred1, col_pred2 = st.columns(2)
                with col_pred1:
                    st.metric(f"{prediction['team1']} (Away)", f"{prediction['team1_score']} pts")
                with col_pred2:
                    st.metric(f"{prediction['team2']} (Home)", f"{prediction['team2_score']} pts")
                
                col_total1, col_total2 = st.columns(2)
                with col_total1:
                    st.metric("Total", f"{prediction['total']}")
                with col_total2:
                    spread_text = f"{prediction['team2']} by {abs(prediction['spread']):.1f}" if prediction['spread'] < 0 else f"{prediction['team1']} by {prediction['spread']:.1f}"
                    st.metric("Spread", spread_text)
                
                if prediction.get('injury_adjusted'):
                    st.warning(f"⚠️ {prediction['adjustment_note']}")
                    if prediction.get('original_spread'):
                        st.caption(f"Original: {prediction['original_spread']:+.1f} → Adjusted: {prediction['spread']:+.1f}")
                
                st.session_state.last_prediction = prediction
            else:
                st.error("Select different teams")
    
    with col2:
        st.subheader("Betting Context")
        if 'last_prediction' in st.session_state:
            pred = st.session_state.last_prediction
            st.write("**Analysis:**")
            
            total = pred['total']
            if total >= 50:
                st.write("🔥 High-scoring expected")
            elif total <= 38:
                st.write("🛡️ Defensive battle")
            else:
                st.write("📊 Average scoring")
            
            spread = abs(pred['spread'])
            if spread >= 10:
                st.write("💪 Significant favorite")
            elif spread <= 3:
                st.write("⚖️ Toss-up game")
            else:
                st.write("📈 Moderate favorite")

elif page == "Player Props":
    st.header("🎲 Player Props")
    
    position = st.selectbox("Position", ["Quarterback", "Wide Receiver/TE", "Running Back"])
    
    if position == "Quarterback":
        if 'qb' in player_data and player_data['qb']:
            qb_names = sorted(list(player_data['qb'].keys()))
            selected_qb = st.selectbox("Select QB", qb_names)
            
            if selected_qb:
                qb_stats = player_data['qb'][selected_qb]
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader(f"📊 {selected_qb}")
                    st.metric("Yards (L4)", f"{qb_stats['passing_yards_L4']:.0f}")
                    st.metric("Yards (L8)", f"{qb_stats['passing_yards_L8']:.0f}")
                    st.metric("Completion %", f"{qb_stats['completion_pct_L4']:.1%}")
                    st.write(f"Team: {qb_stats['team']}")
                
                with col2:
                    st.subheader("🎯 Prediction")
                    opponent = st.selectbox("Opponent", available_teams, key="qb_opp")
                    
                    if st.button("🏈 PREDICT PASSING YARDS", type="primary"):
                        with st.spinner("Predicting..."):
                            base_prediction, status = prediction_system.predict_player_passing(qb_stats, opponent)
                            
                            prediction, injury_status = integrate_injuries_into_player_prediction(
                                injury_system,
                                base_prediction,
                                selected_qb,
                                qb_stats['team']
                            )
                        
                        if prediction is not None:
                            st.success(f"## 🎯 {prediction} yards")
                            st.caption(status)
                            
                            if "Out" in injury_status or "Doubtful" in injury_status:
                                st.error(f"🚨 {injury_status}")
                            elif "Questionable" in injury_status:
                                st.warning(f"⚠️ {injury_status}")
                            
                            if prediction >= qb_stats['passing_yards_L4'] * 1.1:
                                st.write("📈 Strong matchup")
                            elif prediction <= qb_stats['passing_yards_L4'] * 0.9:
                                st.write("📉 Tough matchup")
                        else:
                            st.error(f"Failed: {status}")
        else:
            st.warning("No QB data available")
    
    elif position == "Wide Receiver/TE":
        if 'wr' in player_data and player_data['wr']:
            wr_names = sorted(list(player_data['wr'].keys()))
            selected_wr = st.selectbox("Select WR/TE", wr_names)
            
            if selected_wr:
                wr_stats = player_data['wr'][selected_wr]
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader(f"📊 {selected_wr}")
                    st.metric("Rec Yards (L4)", f"{wr_stats['receiving_yards_L4']:.0f}")
                    st.metric("Receptions (L4)", f"{wr_stats['receptions_L4']:.1f}")
                    st.write(f"Team: {wr_stats['team']}")
                
                with col2:
                    st.subheader("🎯 Predictions")
                    opponent = st.selectbox("Opponent", available_teams, key="wr_opp")
                    
                    col_btn1, col_btn2 = st.columns(2)
                    
                    with col_btn1:
                        if st.button("Rec Yards"):
                            base_prediction, status = prediction_system.predict_player_receiving(wr_stats, opponent)
                            prediction, injury_status = integrate_injuries_into_player_prediction(
                                injury_system, base_prediction, selected_wr, wr_stats['team']
                            )
                            if prediction:
                                st.success(f"**{prediction} yds**")
                                if "Out" in injury_status or "Doubtful" in injury_status:
                                    st.error(f"🚨 {injury_status}")
                            else:
                                st.error(status)
                    
                    with col_btn2:
                        if st.button("Receptions"):
                            base_prediction, status = prediction_system.predict_player_receptions(wr_stats, opponent)
                            prediction, injury_status = integrate_injuries_into_player_prediction(
                                injury_system, base_prediction, selected_wr, wr_stats['team']
                            )
                            if prediction:
                                st.success(f"**{prediction} rec**")
                                if "Out" in injury_status or "Doubtful" in injury_status:
                                    st.error(f"🚨 {injury_status}")
                            else:
                                st.error(status)
        else:
            st.warning("No WR data available")
    
    elif position == "Running Back":
        if 'rb' in player_data and player_data['rb']:
            rb_names = sorted(list(player_data['rb'].keys()))
            selected_rb = st.selectbox("Select RB", rb_names)
            
            if selected_rb:
                rb_stats = player_data['rb'][selected_rb]
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader(f"📊 {selected_rb}")
                    st.metric("Rush Yards (L4)", f"{rb_stats['rushing_yards_L4']:.0f}")
                    st.metric("Attempts (L4)", f"{rb_stats['attempts_L4']:.1f}")
                    st.write(f"Team: {rb_stats['team']}")
                
                with col2:
                    st.subheader("🎯 Prediction")
                    opponent = st.selectbox("Opponent", available_teams, key="rb_opp")
                    
                    if st.button("🏃 PREDICT RUSHING YARDS", type="primary"):
                        with st.spinner("Predicting..."):
                            base_prediction, status = prediction_system.predict_player_rushing(rb_stats, opponent)
                            
                            prediction, injury_status = integrate_injuries_into_player_prediction(
                                injury_system,
                                base_prediction,
                                selected_rb,
                                rb_stats['team']
                            )
                        
                        if prediction is not None:
                            st.success(f"## 🎯 {prediction} yards")
                            st.caption(status)
                            
                            if "Out" in injury_status or "Doubtful" in injury_status:
                                st.error(f"🚨 {injury_status}")
                            elif "Questionable" in injury_status:
                                st.warning(f"⚠️ {injury_status}")
                            
                            if prediction >= rb_stats['rushing_yards_L4'] * 1.15:
                                st.write("📈 Great matchup")
                            elif prediction <= rb_stats['rushing_yards_L4'] * 0.85:
                                st.write("📉 Tough matchup")
                        else:
                            st.error(f"Failed: {status}")
        else:
            st.warning("No RB data available")

elif page == "System Info":
    st.header("📊 System Information v2.1")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Features")
        st.write("✅ Ensemble models (3 per prediction)")
        st.write("✅ Calibrated game predictions")
        st.write("✅ Split defense rankings")
        st.write("✅ Injury adjustments")
        st.write("✅ Point-in-time stats (no leakage)")
    
    with col2:
        st.subheader("Data Coverage")
        st.metric("Teams", len(available_teams))
        st.metric("QBs", len(player_data.get('qb', {})))
        st.metric("WRs/TEs", len(player_data.get('wr', {})))
        st.metric("RBs", len(player_data.get('rb', {})))
    
    st.subheader("Performance Targets")
    st.write("• Game Predictions: 55-58% win accuracy")
    st.write("• Spread MAE: ~10 points")
    st.write("• Player Props: Highly accurate")
    
    st.info("💡 Run `python weekly_nfl_update.py` every Monday to update data")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**🚀 Production v2.1**")
st.sidebar.markdown("Injury System Active")
st.sidebar.markdown(f"Updated: {MODEL_UPDATE_DATE}")