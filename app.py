# nfl_app.py - Production-Ready NFL Prediction System

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os
from datetime import datetime


# Page configuration
st.set_page_config(
    page_title="NFL Prediction System - Pro Edition",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Production configuration
PRODUCTION_MODE = True  # Set to False for development
MODEL_UPDATE_DATE = "2024-12-XX"  # Update this when models are retrained

class ProductionGamePredictor:
    def __init__(self, team_data_file='team_data.json', use_enhanced=True):
        """Production-ready NFL prediction system"""
        
        # Load team data
        try:
            with open(team_data_file, 'r') as f:
                self.team_data = json.load(f)
            
            all_points = [team['points_L4'] for team in self.team_data.values()]
            all_allowed = [team['opp_points_L4'] for team in self.team_data.values()]
            
            self.league_avg_points = np.mean(all_points)
            self.league_avg_allowed = np.mean(all_allowed)
            
        except FileNotFoundError:
            st.error("Team data not found. Please update team statistics.")
            self.team_data = {}
            self.league_avg_points = 22.0
            self.league_avg_allowed = 22.0
        
        # Load models (enhanced if available, fallback to original)
        self.models = self._load_models(use_enhanced)
        
        # Load defense rankings
        try:
            with open('defense_rankings.json', 'r') as f:
                self.defense_rankings = json.load(f)
        except FileNotFoundError:
            self.defense_rankings = {}
    
    def _load_models(self, use_enhanced=True):
        """Load models with fallback strategy"""
        models = {}
        
        # Try enhanced models first
        if use_enhanced:
            enhanced_paths = {
                'passing': 'models/enhanced_passing_model.pkl',
                'receiving': 'models/enhanced_receiving_model.pkl',
                'receptions': 'models/enhanced_receptions_model.pkl',
                'rushing': 'models/enhanced_rushing_model.pkl'
            }
            
            for model_name, path in enhanced_paths.items():
                if os.path.exists(path):
                    try:
                        models[model_name] = joblib.load(path)
                        models[f'{model_name}_type'] = 'enhanced'
                    except:
                        models[model_name] = None
                        models[f'{model_name}_type'] = 'failed'
        
        # Fallback to original models
        original_paths = {
            'passing': 'models/passing_model.pkl',
            'receiving': 'models/receiving_model.pkl',
            'receptions': 'models/receptions_model.pkl',
            'rushing': 'models/rushing_model.pkl'
        }
        
        for model_name, path in original_paths.items():
            if model_name not in models or models[model_name] is None:
                if os.path.exists(path):
                    try:
                        models[model_name] = joblib.load(path)
                        models[f'{model_name}_type'] = 'original'
                    except:
                        models[model_name] = None
                        models[f'{model_name}_type'] = 'failed'
        
        return models
    
    def predict_team_score(self, offense_team, defense_team, is_home=False, recency_weight=0.7):
        """Production team scoring prediction"""
        off_data = self.team_data.get(offense_team, {})
        def_data = self.team_data.get(defense_team, {})
        
        if not off_data or not def_data:
            return self.league_avg_points
        
        off_recent = off_data.get('points_L4', self.league_avg_points)
        off_season = off_data.get('points_L8', self.league_avg_points)
        offensive_scoring = (recency_weight * off_recent + (1 - recency_weight) * off_season)
        
        def_recent = def_data.get('opp_points_L4', self.league_avg_allowed)
        def_season = def_data.get('opp_points_L8', self.league_avg_allowed)
        defensive_allowing = (recency_weight * def_recent + (1 - recency_weight) * def_season)
        
        # Matchup calculation
        offensive_efficiency = offensive_scoring / self.league_avg_points
        defensive_efficiency = defensive_allowing / self.league_avg_allowed
        base_prediction = self.league_avg_points * offensive_efficiency * defensive_efficiency
        
        # Adjustments
        home_bonus = 2.8 if is_home else 0
        regression_factor = 0.85
        
        final_prediction = (regression_factor * base_prediction + 
                          (1 - regression_factor) * self.league_avg_points + 
                          home_bonus)
        
        return max(final_prediction, 10.0)
    
    def predict_game(self, team1, team2, home_team=None):
        """Game outcome prediction"""
        team1_home = (home_team == team1) if home_team else False
        team2_home = (home_team == team2) if home_team else False
        
        team1_score = self.predict_team_score(team1, team2, team1_home)
        team2_score = self.predict_team_score(team2, team1, team2_home)
        
        return {
            'team1': team1,
            'team1_score': round(team1_score, 1),
            'team2': team2,
            'team2_score': round(team2_score, 1), 
            'total': round(team1_score + team2_score, 1),
            'spread': round(team1_score - team2_score, 1),
            'confidence': 'Medium'  # Could be enhanced with prediction intervals
        }
    
    def predict_player_passing(self, player_stats, opponent_team=None):
        """Production QB prediction with enhanced defense adjustment"""
        if not self.models.get('passing'):
            return None, "Passing model not loaded"

        try:
            defense_rank = 16
            defense_multiplier = 1.0
            
            if opponent_team and opponent_team in self.defense_rankings:
                defense_rank = self.defense_rankings[opponent_team]['rank']
                
                # Enhanced defense adjustment based on analysis
                if defense_rank <= 5:     # Elite defense
                    defense_multiplier = 0.80
                elif defense_rank <= 12:  # Good defense
                    defense_multiplier = 0.90
                elif defense_rank <= 20:  # Average defense
                    defense_multiplier = 1.0
                elif defense_rank <= 28:  # Poor defense
                    defense_multiplier = 1.15
                else:                     # Terrible defense
                    defense_multiplier = 1.25
            
            model = self.models['passing']
            
            # Multiple feature combinations for robustness
            feature_combinations = [
                [
                    player_stats.get('passing_yards_L4', 250),
                    player_stats.get('completion_pct_L4', 0.65),
                    player_stats.get('attempts_L4', 35),
                    defense_rank
                ],
                [
                    player_stats.get('passing_yards_L4', 250),
                    player_stats.get('passing_yards_L8', 250),
                    player_stats.get('completion_pct_L4', 0.65),
                    player_stats.get('attempts_L4', 35),
                    player_stats.get('passing_tds_L4', 1.5),
                    defense_rank
                ]
            ]
            
            for features in feature_combinations:
                try:
                    feature_array = np.array([features])
                    base_prediction = model.predict(feature_array)[0]
                    adjusted_prediction = base_prediction * defense_multiplier
                    
                    model_type = self.models.get('passing_type', 'unknown')
                    return max(0, round(adjusted_prediction, 1)), f"Success ({model_type}) vs #{defense_rank} defense"
                except:
                    continue
            
            # Fallback
            base_avg = player_stats.get('passing_yards_L4', 250)
            adjusted_avg = base_avg * defense_multiplier
            return round(adjusted_avg, 1), f"Fallback vs #{defense_rank} defense"
            
        except Exception as e:
            return None, f"Error: {str(e)[:50]}"
    
    def predict_player_receiving(self, player_stats, opponent_team=None):
        """Production WR prediction"""
        if not self.models.get('receiving'):
            return None, "Receiving model not loaded"

        try:
            defense_rank = 16
            defense_multiplier = 1.0
            
            if opponent_team and opponent_team in self.defense_rankings:
                defense_rank = self.defense_rankings[opponent_team]['rank']
                
                if defense_rank <= 8:
                    defense_multiplier = 0.80
                elif defense_rank <= 16:
                    defense_multiplier = 1.0
                elif defense_rank <= 24:
                    defense_multiplier = 1.15
                else:
                    defense_multiplier = 1.25
            
            model = self.models['receiving']
            
            feature_combinations = [
                [
                    player_stats.get('receiving_yards_L4', 50),
                    player_stats.get('receptions_L4', 4),
                    player_stats.get('yards_per_rec_L4', 12),
                    defense_rank
                ]
            ]
            
            for features in feature_combinations:
                try:
                    feature_array = np.array([features])
                    base_prediction = model.predict(feature_array)[0]
                    adjusted_prediction = base_prediction * defense_multiplier
                    
                    model_type = self.models.get('receiving_type', 'unknown')
                    return max(0, round(adjusted_prediction, 1)), f"Success ({model_type}) vs #{defense_rank} defense"
                except:
                    continue
            
            base_avg = player_stats.get('receiving_yards_L4', 50)
            adjusted_avg = base_avg * defense_multiplier
            return round(adjusted_avg, 1), f"Fallback vs #{defense_rank} defense"
            
        except Exception as e:
            return None, f"Error: {str(e)[:50]}"
    
    def predict_player_receptions(self, player_stats, opponent_team=None):
        """Production receptions prediction"""
        if not self.models.get('receptions'):
            return None, "Receptions model not loaded"

        try:
            defense_rank = 16
            defense_multiplier = 1.0
            
            if opponent_team and opponent_team in self.defense_rankings:
                defense_rank = self.defense_rankings[opponent_team]['rank']
                
                if defense_rank <= 8:
                    defense_multiplier = 0.85
                elif defense_rank <= 16:
                    defense_multiplier = 1.0
                elif defense_rank <= 24:
                    defense_multiplier = 1.1
                else:
                    defense_multiplier = 1.2
            
            model = self.models['receptions']
            
            feature_combinations = [
                [
                    player_stats.get('receptions_L4', 4),
                    player_stats.get('receiving_yards_L4', 50),
                    defense_rank
                ]
            ]
            
            for features in feature_combinations:
                try:
                    feature_array = np.array([features])
                    base_prediction = model.predict(feature_array)[0]
                    adjusted_prediction = base_prediction * defense_multiplier
                    
                    model_type = self.models.get('receptions_type', 'unknown')
                    return max(0, round(adjusted_prediction, 1)), f"Success ({model_type}) vs #{defense_rank} defense"
                except:
                    continue
            
            base_avg = player_stats.get('receptions_L4', 4)
            adjusted_avg = base_avg * defense_multiplier
            return round(adjusted_avg, 1), f"Fallback vs #{defense_rank} defense"
            
        except Exception as e:
            return None, f"Error: {str(e)[:50]}"
    
    def predict_player_rushing(self, player_stats, opponent_team=None):
        """Production RB prediction"""
        if not self.models.get('rushing'):
            return None, "Rushing model not loaded"

        try:
            defense_rank = 16
            defense_multiplier = 1.0
            
            if opponent_team and opponent_team in self.defense_rankings:
                defense_rank = self.defense_rankings[opponent_team]['rank']
                
                if defense_rank <= 8:
                    defense_multiplier = 0.75
                elif defense_rank <= 16:
                    defense_multiplier = 1.0
                elif defense_rank <= 24:
                    defense_multiplier = 1.2
                else:
                    defense_multiplier = 1.3
            
            model = self.models['rushing']
            
            feature_combinations = [
                [
                    player_stats.get('rushing_yards_L4', 80),
                    player_stats.get('attempts_L4', 18),
                    player_stats.get('yards_per_carry_L4', 4.5),
                    defense_rank
                ]
            ]
            
            for features in feature_combinations:
                try:
                    feature_array = np.array([features])
                    base_prediction = model.predict(feature_array)[0]
                    adjusted_prediction = base_prediction * defense_multiplier
                    
                    model_type = self.models.get('rushing_type', 'unknown')
                    return max(0, round(adjusted_prediction, 1)), f"Success ({model_type}) vs #{defense_rank} defense"
                except:
                    continue
            
            base_avg = player_stats.get('rushing_yards_L4', 80)
            adjusted_avg = base_avg * defense_multiplier
            return round(adjusted_avg, 1), f"Fallback vs #{defense_rank} defense"
            
        except Exception as e:
            return None, f"Error: {str(e)[:50]}"
    
    def get_system_status(self):
        """Get system health status for monitoring"""
        status = {
            'teams_loaded': len(self.team_data),
            'models_loaded': len([m for m in self.models.values() if m is not None and not isinstance(m, str)]),
            'defense_rankings': len(self.defense_rankings),
            'last_update': MODEL_UPDATE_DATE
        }
        return status
    
    def list_available_teams(self):
        """List all available teams"""
        return sorted(list(self.team_data.keys()))

# Load models and data with caching
@st.cache_resource
def load_prediction_system():
    """Load the prediction system with caching"""
    return ProductionGamePredictor()

@st.cache_data
def load_player_data():
    """Load player data with caching"""
    data = {}
    files = ['qb_data.json', 'wr_data.json', 'rb_data.json']
    
    for file in files:
        try:
            if os.path.exists(file):
                with open(file, 'r') as f:
                    data[file.replace('_data.json', '')] = json.load(f)
            else:
                data[file.replace('_data.json', '')] = {}
        except Exception as e:
            st.sidebar.warning(f"Error loading {file}")
            data[file.replace('_data.json', '')] = {}
    
    return data

# Initialize the app
st.title("🏈 NFL Prediction System - Pro Edition")
st.caption(f"Last Updated: {MODEL_UPDATE_DATE}")

# Load system
prediction_system = load_prediction_system()
player_data = load_player_data()

# System status
system_status = prediction_system.get_system_status()
available_teams = prediction_system.list_available_teams()

# Sidebar
st.sidebar.title("System Status")
st.sidebar.success(f"Teams: {system_status['teams_loaded']}")
st.sidebar.info(f"Models: {system_status['models_loaded']}/4")
st.sidebar.info(f"Defense Rankings: {system_status['defense_rankings']}")

if system_status['teams_loaded'] == 0:
    st.sidebar.error("⚠️ Data needs update")
    st.error("System requires data update. Please refresh team and player statistics.")
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
                prediction = prediction_system.predict_game(away_team, home_team, home_team=home_team)
                
                st.success(f"**{prediction['team1']} @ {prediction['team2']}**")
                
                col_pred1, col_pred2 = st.columns(2)
                with col_pred1:
                    st.metric(f"{prediction['team1']} (Away)", f"{prediction['team1_score']} points")
                with col_pred2:
                    st.metric(f"{prediction['team2']} (Home)", f"{prediction['team2_score']} points")
                
                col_total1, col_total2 = st.columns(2)
                with col_total1:
                    st.metric("Total Points", f"{prediction['total']}")
                with col_total2:
                    spread_text = f"{prediction['team2']} by {abs(prediction['spread']):.1f}" if prediction['spread'] < 0 else f"{prediction['team1']} by {prediction['spread']:.1f}"
                    st.metric("Spread", spread_text)
                
                st.session_state.last_prediction = prediction
            else:
                st.error("Select different teams")
    
    with col2:
        st.subheader("Betting Context")
        if 'last_prediction' in st.session_state:
            pred = st.session_state.last_prediction
            st.write("**Key Insights:**")
            
            total = pred['total']
            if total >= 50:
                st.write("🔥 High-scoring game expected")
            elif total <= 38:
                st.write("🛡️ Defensive battle expected")
            else:
                st.write("📊 Average scoring expected")
            
            spread = abs(pred['spread'])
            if spread >= 10:
                st.write("💪 Significant favorite")
            elif spread <= 3:
                st.write("⚖️ Close game predicted")
            else:
                st.write("📈 Moderate favorite")

elif page == "Player Props":
    st.header("🎲 Player Prop Predictions")
    st.write("Player predictions using production models with opponent adjustments")
    
    position = st.selectbox("Select Position", ["Quarterback", "Wide Receiver/TE", "Running Back"])
    
    # QUARTERBACK SECTION
    if position == "Quarterback":
        if 'qb' in player_data and player_data['qb']:
            qb_names = sorted(list(player_data['qb'].keys()))
            selected_qb = st.selectbox("Select QB", qb_names)
            
            if selected_qb:
                qb_stats = player_data['qb'][selected_qb]
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader(f"📊 {selected_qb} Stats")
                    
                    st.metric("Last 4 Games Avg", f"{qb_stats['passing_yards_L4']:.0f} yards")
                    st.metric("Season Avg (L8)", f"{qb_stats['passing_yards_L8']:.0f} yards")
                    st.metric("Completion %", f"{qb_stats['completion_pct_L4']:.1%}")
                    st.metric("Attempts/Game", f"{qb_stats['attempts_L4']:.0f}")
                    st.metric("TDs/Game", f"{qb_stats['passing_tds_L4']:.1f}")
                    
                    st.write(f"**Team:** {qb_stats['team']}")
                    st.write(f"**Last vs:** {qb_stats['last_opponent']}")
                
                with col2:
                    st.subheader("🎯 Make Prediction")
                    
                    opponent_team = st.selectbox("Select Opponent Defense", available_teams, key="qb_opp")
                    
                    if opponent_team in prediction_system.defense_rankings:
                        def_info = prediction_system.defense_rankings[opponent_team]
                        def_rank = def_info['rank']
                        def_yards = def_info['yards_allowed']
                        
                        if def_rank <= 8:
                            st.error(f"🛡️ {opponent_team}: Elite Defense (#{def_rank}) - {def_yards:.0f} yds/game")
                        elif def_rank <= 16:
                            st.warning(f"🟡 {opponent_team}: Average Defense (#{def_rank}) - {def_yards:.0f} yds/game")
                        else:
                            st.success(f"🎯 {opponent_team}: Poor Defense (#{def_rank}) - {def_yards:.0f} yds/game")
                    
                    if st.button("🏈 PREDICT PASSING YARDS", type="primary", key="predict_qb"):
                        with st.spinner("Making prediction..."):
                            prediction, status = prediction_system.predict_player_passing(qb_stats, opponent_team)
                        
                        if prediction is not None:
                            st.success(f"## 🎯 Predicted: {prediction} yards")
                            
                            lower = max(0, prediction - 30)
                            upper = prediction + 30
                            st.info(f"**Confidence Range:** {lower:.0f} - {upper:.0f} yards")
                            
                            st.write(f"**Status:** {status}")
                            
                            if prediction >= qb_stats['passing_yards_L4'] * 1.1:
                                st.write("📈 Prediction is 10%+ above recent average - consider Over")
                            elif prediction <= qb_stats['passing_yards_L4'] * 0.9:
                                st.write("📉 Prediction is 10%+ below recent average - consider Under")
                            else:
                                st.write("📊 Prediction aligns with recent performance")
                        else:
                            st.error(f"Prediction failed: {status}")
        else:
            st.warning("⚠️ No QB data available.")
    
    # WIDE RECEIVER SECTION  
    elif position == "Wide Receiver/TE":
        if 'wr' in player_data and player_data['wr']:
            wr_names = sorted(list(player_data['wr'].keys()))
            selected_wr = st.selectbox("Select WR/TE", wr_names)
            
            if selected_wr:
                wr_stats = player_data['wr'][selected_wr]
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader(f"📊 {selected_wr} Stats")
                    
                    st.metric("Receiving Yards (L4)", f"{wr_stats['receiving_yards_L4']:.0f}")
                    st.metric("Receptions (L4)", f"{wr_stats['receptions_L4']:.1f}")
                    st.metric("Yards/Reception", f"{wr_stats['yards_per_rec_L4']:.1f}")
                    st.metric("TDs (L4)", f"{wr_stats['receiving_tds_L4']:.1f}")
                    
                    st.write(f"**Team:** {wr_stats['team']}")
                    st.write(f"**Last vs:** {wr_stats['last_opponent']}")
                
                with col2:
                    st.subheader("🎯 Make Predictions")
                    
                    opponent_team = st.selectbox("Select Opponent Defense", available_teams, key="wr_opp")
                    
                    if opponent_team in prediction_system.defense_rankings:
                        def_rank = prediction_system.defense_rankings[opponent_team]['rank']
                        if def_rank <= 10:
                            st.error(f"🛡️ Strong Defense (#{def_rank})")
                        elif def_rank >= 25:
                            st.success(f"🎯 Weak Defense (#{def_rank})")
                        else:
                            st.warning(f"🟡 Average Defense (#{def_rank})")
                    
                    col_btn1, col_btn2 = st.columns(2)
                    
                    with col_btn1:
                        if st.button("Predict Receiving Yards", key="pred_rec_yards"):
                            prediction, status = prediction_system.predict_player_receiving(wr_stats, opponent_team)
                            if prediction:
                                st.success(f"**{prediction} yards**")
                                st.write(f"Status: {status}")
                                lower = max(0, prediction - 15)
                                upper = prediction + 15
                                st.write(f"Range: {lower:.0f}-{upper:.0f}")
                            else:
                                st.error(f"Failed: {status}")
                    
                    with col_btn2:
                        if st.button("Predict Receptions", key="pred_recs"):
                            prediction, status = prediction_system.predict_player_receptions(wr_stats, opponent_team)
                            if prediction:
                                st.success(f"**{prediction} catches**")
                                st.write(f"Status: {status}")
                                lower = max(0, prediction - 2)
                                upper = prediction + 2
                                st.write(f"Range: {lower:.1f}-{upper:.1f}")
                            else:
                                st.error(f"Failed: {status}")
        else:
            st.warning("⚠️ No WR data available.")
    
    # RUNNING BACK SECTION
    elif position == "Running Back":
        if 'rb' in player_data and player_data['rb']:
            rb_names = sorted(list(player_data['rb'].keys()))
            selected_rb = st.selectbox("Select RB", rb_names)
            
            if selected_rb:
                rb_stats = player_data['rb'][selected_rb]
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader(f"📊 {selected_rb} Stats")
                    
                    st.metric("Rushing Yards (L4)", f"{rb_stats['rushing_yards_L4']:.0f}")
                    st.metric("Attempts (L4)", f"{rb_stats['attempts_L4']:.1f}")
                    st.metric("Yards/Carry", f"{rb_stats['yards_per_carry_L4']:.1f}")
                    st.metric("TDs (L4)", f"{rb_stats['rushing_tds_L4']:.1f}")
                    
                    st.write(f"**Team:** {rb_stats['team']}")
                    st.write(f"**Last vs:** {rb_stats['last_opponent']}")
                
                with col2:
                    st.subheader("🎯 Make Prediction")
                    
                    opponent_team = st.selectbox("Select Opponent Defense", available_teams, key="rb_opp")
                    
                    if opponent_team in prediction_system.defense_rankings:
                        def_rank = prediction_system.defense_rankings[opponent_team]['rank']
                        if def_rank <= 8:
                            st.error(f"🛡️ Elite Run Defense (#{def_rank})")
                        elif def_rank >= 24:
                            st.success(f"🎯 Weak Run Defense (#{def_rank})")
                        else:
                            st.warning(f"🟡 Average Run Defense (#{def_rank})")
                    
                    if st.button("🏃 PREDICT RUSHING YARDS", type="primary", key="predict_rb"):
                        with st.spinner("Making prediction..."):
                            prediction, status = prediction_system.predict_player_rushing(rb_stats, opponent_team)
                        
                        if prediction is not None:
                            st.success(f"## 🎯 Predicted: {prediction} yards")
                            
                            lower = max(0, prediction - 20)
                            upper = prediction + 20
                            st.info(f"**Confidence Range:** {lower:.0f} - {upper:.0f} yards")
                            
                            st.write(f"**Status:** {status}")
                            
                            if prediction >= rb_stats['rushing_yards_L4'] * 1.15:
                                st.write("📈 Strong matchup - consider Over")
                            elif prediction <= rb_stats['rushing_yards_L4'] * 0.85:
                                st.write("📉 Tough matchup - consider Under")
                            else:
                                st.write("📊 Neutral prediction")
                        else:
                            st.error(f"Prediction failed: {status}")
        else:
            st.warning("⚠️ No RB data available.")

elif page == "System Info":
    st.header("📊 System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance")
        performance_data = {
            'Model': ['Passing Yards', 'Receiving Yards', 'Receptions', 'Rushing Yards'],
            'MAE': ['66.7 yards', '19.8 yards', '1.4 catches', '18.8 yards'],
            'Type': [prediction_system.models.get(f'{m}_type', 'unknown') for m in ['passing', 'receiving', 'receptions', 'rushing']]
        }
        st.dataframe(pd.DataFrame(performance_data))
    
    with col2:
        st.subheader("Data Coverage")
        st.metric("NFL Teams", len(available_teams))
        st.metric("QBs", len(player_data.get('qb', {})))
        st.metric("WRs/TEs", len(player_data.get('wr', {})))
        st.metric("RBs", len(player_data.get('rb', {})))
    
    st.subheader("Update Schedule")
    st.write("- **Team Stats**: Updated after each week's games")
    st.write("- **Player Stats**: Updated after each week's games")
    st.write("- **Models**: Retrained monthly with new data")
    st.write("- **Defense Rankings**: Updated weekly")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Production System**")
st.sidebar.markdown("Ready for deployment")
st.sidebar.markdown("Automated updates: Weekly")
