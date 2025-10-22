#!/usr/bin/env python3
"""
NFL Model Update Script v2.2 - WITH EPA AND OL/DL INTEGRATION
Generates accuracy report BEFORE updating models, then updates everything
INCLUDING new EPA and OL/DL matchup metrics

Key Flow:
1. Generate accuracy report using OLD models (true out-of-sample test)
2. Calculate EPA metrics and OL/DL rankings
3. Update all data and retrain models with new EPA/OL-DL features
"""

import sys
import subprocess
import pandas as pd
import numpy as np
import json
import os
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def install_packages():
    """Install required packages if they're not available"""
    try:
        import nfl_data_py
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.neural_network import MLPRegressor
        import joblib
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"Installing missing packages...")
        packages = ['nfl-data-py', 'pandas', 'numpy', 'scikit-learn', 'joblib']
        for package in packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            except:
                print(f"❌ Failed to install {package}")
                return False
        print("✅ Packages installed successfully")
        return True



class NFLWeeklyUpdater:
    """Enhanced NFL prediction system with EPA and OL/DL integration"""
    
    def __init__(self):
        self.current_year = datetime.now().year
        if datetime.now().month >= 9:
            self.data_years = [self.current_year - 2, self.current_year - 1, self.current_year]
        else:
            self.data_years = [self.current_year - 3, self.current_year - 2, self.current_year - 1]
        
        self.models = {}
        self.ensemble_models = {}
        self.calibration_params = {}
        self.team_data = {}
        self.player_data = {'qb': {}, 'wr': {}, 'rb': {}}
        self.defense_rankings = {}
        
        # NEW: EPA and OL/DL data structures
        self.epa_metrics = {}
        self.ol_dl_rankings = {}

        # Dynamic league average (calculated from actual data)
        self.league_avg_ppg = 22.0  # Default, will be updated with actual data

        os.makedirs("models", exist_ok=True)
        os.makedirs("data", exist_ok=True)
    
    def generate_weekly_accuracy_report(self):
        """Generate accuracy report for the most recent completed week"""
        print("\n" + "="*60)
        print("🎯 GENERATING WEEKLY ACCURACY REPORT")
        print("Using OLD models to test predictions (out-of-sample)")
        print("="*60)
        
        # Load data if not already loaded
        if not hasattr(self, 'schedule_regular'):
            print("   Loading NFL data for accuracy report...")
            if not self.load_nfl_data():
                print("   ⚠️ Could not load data - skipping accuracy report")
                return
            
            # Also need team_data and EPA/OL-DL metrics for the report
            print("   Calculating EPA metrics...")
            self.calculate_epa_metrics()
            print("   Calculating OL/DL rankings...")
            self.calculate_ol_dl_strength()
            print("   Creating team data...")
            self.create_team_data()
        
        current_season = max(self.data_years)
        print(f"   Current NFL Season: {current_season}")
        
        # Find most recent completed week with >12 games
        completed_games = self.schedule_regular[
            (self.schedule_regular['season'] == current_season) &
            (self.schedule_regular['home_score'].notna())
        ]
        
        if len(completed_games) == 0:
            print("   ⚠️ No completed games found for current season")
            return
        
        # Count games per week
        games_per_week = completed_games.groupby('week').size()
        
        # Find most recent week with >12 games
        full_weeks = games_per_week[games_per_week > 12]
        
        if len(full_weeks) == 0:
            print("   ⚠️ No full weeks (>12 games) found yet this season")
            return
        
        latest_week = full_weeks.index.max()
        week_game_count = full_weeks[latest_week]
        print(f"📊 Analyzing Week {latest_week} of {current_season} season ({week_game_count} games)...")
        
        week_games = completed_games[completed_games['week'] == latest_week]
        print(f"   Found {len(week_games)} completed games in Week {latest_week}")
        
        # Try to load OLD models
        try:
            game_total_models = []
            game_spread_models = []
            
            for idx in range(3):
                total_path = f'models/game_total_model_{idx}.pkl'
                spread_path = f'models/game_spread_model_{idx}.pkl'
                
                if os.path.exists(total_path):
                    game_total_models.append(joblib.load(total_path))
                if os.path.exists(spread_path):
                    game_spread_models.append(joblib.load(spread_path))
            
            if len(game_total_models) == 0 or len(game_spread_models) == 0:
                print("   ⚠️ No models found - skipping accuracy report")
                return
            
            print(f"✅ Loaded {len(game_total_models)} ensemble models")
            
            # Determine expected feature count from first model
            expected_features = game_total_models[0].n_features_in_
            print(f"   Models expect {expected_features} features")
            
        except Exception as e:
            print(f"   ⚠️ Error loading models: {str(e)[:100]}")
            return
        
        # Test predictions on completed games
        predictions = []
        print(f"\nTesting predictions on {len(week_games)} completed games...")
        print("-" * 60)
        
        for _, game in week_games.iterrows():
            try:
                home_team = game['home_team']
                away_team = game['away_team']
                actual_home = float(game['home_score'])
                actual_away = float(game['away_score'])
                
                print(f"\n{away_team} @ {home_team}")
                print(f"  Actual: {away_team} {actual_away} - {home_team} {actual_home}")
                
                # Get team stats
                home_stats = self.team_data.get(home_team, {})
                away_stats = self.team_data.get(away_team, {})
                
                if not home_stats or not away_stats:
                    print(f"  ⚠️ Missing team stats")
                    continue
                
                # Build features based on what the model expects
                if expected_features == 18:
                    # OLD model format (no EPA/OL-DL)
                    features = np.array([[
                        float(home_stats.get('points_L4', self.league_avg_ppg)),
                        float(home_stats.get('opp_points_L4', self.league_avg_ppg)),
                        float(home_stats.get('yards_L4', 350.0)),
                        float(home_stats.get('points_L8', self.league_avg_ppg)),
                        float(home_stats.get('opp_points_L8', self.league_avg_ppg)),
                        float(home_stats.get('win_pct_L8', 0.5)),
                        float(home_stats.get('turnovers_L4', 1.0)),
                        float(away_stats.get('points_L4', self.league_avg_ppg)),
                        float(away_stats.get('opp_points_L4', self.league_avg_ppg)),
                        float(away_stats.get('yards_L4', 350.0)),
                        float(away_stats.get('points_L8', self.league_avg_ppg)),
                        float(away_stats.get('opp_points_L8', self.league_avg_ppg)),
                        float(away_stats.get('win_pct_L8', 0.5)),
                        float(away_stats.get('turnovers_L4', 1.0)),
                        7.0, 7.0, 0.0, 0.0  # rest and division
                    ]])
                elif expected_features == 20:
                    # NEW model format (with EPA/OL-DL)
                    home_epa = 0.0
                    if home_team in self.epa_metrics and str(current_season) in self.epa_metrics[home_team]:
                        home_epa = self.epa_metrics[home_team][str(current_season)].get('epa_per_play', 0.0)
                    
                    ol_dl_score = 0.0
                    if (home_team in self.ol_dl_rankings and away_team in self.ol_dl_rankings and
                        str(current_season) in self.ol_dl_rankings[home_team] and 
                        str(current_season) in self.ol_dl_rankings[away_team]):
                        
                        ol_score = self.ol_dl_rankings[home_team][str(current_season)].get('ol', {}).get('score', 50.0)
                        dl_score = self.ol_dl_rankings[away_team][str(current_season)].get('dl', {}).get('score', 50.0)
                        ol_dl_score = ol_score - dl_score
                    
                    features = np.array([[
                        float(home_stats.get('points_L4', self.league_avg_ppg)),
                        float(home_stats.get('opp_points_L4', self.league_avg_ppg)),
                        float(home_stats.get('yards_L4', 350.0)),
                        float(home_stats.get('points_L8', self.league_avg_ppg)),
                        float(home_stats.get('opp_points_L8', self.league_avg_ppg)),
                        float(home_stats.get('win_pct_L8', 0.5)),
                        float(home_stats.get('turnovers_L4', 1.0)),
                        float(away_stats.get('points_L4', self.league_avg_ppg)),
                        float(away_stats.get('opp_points_L4', self.league_avg_ppg)),
                        float(away_stats.get('yards_L4', 350.0)),
                        float(away_stats.get('points_L8', self.league_avg_ppg)),
                        float(away_stats.get('opp_points_L8', self.league_avg_ppg)),
                        float(home_stats.get('win_pct_L8', 0.5)),
                        float(away_stats.get('turnovers_L4', 1.0)),
                        7.0, 7.0, 0.0, 0.0,  # rest and division
                        float(home_epa),
                        float(ol_dl_score)
                    ]])
                else:
                    print(f"  ⚠️ Unexpected feature count: {expected_features}")
                    continue
                
                # Handle NaN
                if np.isnan(features).any():
                    features = np.nan_to_num(features, nan=0.0)
                
                # Predict
                total_preds = [m.predict(features)[0] for m in game_total_models]
                spread_preds = [m.predict(features)[0] for m in game_spread_models]
                
                pred_total = np.mean(total_preds)
                pred_spread_raw = np.mean(spread_preds)
                
                # Apply calibration
                try:
                    with open('calibration_params.json', 'r') as f:
                        calib = json.load(f)
                        spread_factor = calib.get('spread_factor', 0.7)
                except:
                    spread_factor = 0.7
                
                pred_spread = pred_spread_raw * spread_factor
                pred_home = (pred_total + pred_spread) / 2
                pred_away = (pred_total - pred_spread) / 2
                
                # Calculate accuracy
                actual_total = actual_home + actual_away
                actual_spread = actual_home - actual_away
                
                spread_error = abs(pred_spread - actual_spread)
                total_error = abs(pred_total - actual_total)
                
                pred_winner = home_team if pred_home > pred_away else away_team
                actual_winner = home_team if actual_home > actual_away else away_team
                winner_correct = (pred_winner == actual_winner)
                
                print(f"  Predicted: {away_team} {pred_away:.1f} - {home_team} {pred_home:.1f}")
                print(f"  Total Error: {total_error:.1f} | Spread Error: {spread_error:.1f} | Winner: {'✓' if winner_correct else '✗'}")
                
                predictions.append({
                    'game': f"{away_team} @ {home_team}",
                    'pred_total': pred_total,
                    'actual_total': actual_total,
                    'pred_spread': pred_spread,
                    'actual_spread': actual_spread,
                    'pred_home': pred_home,
                    'pred_away': pred_away,
                    'actual_home': actual_home,
                    'actual_away': actual_away,
                    'spread_error': spread_error,
                    'total_error': total_error,
                    'winner_correct': winner_correct
                })
                
            except Exception as e:
                print(f"  ⚠️ Error: {str(e)[:100]}")
                continue
        
        # Calculate and save summary stats
        if len(predictions) > 0:
            avg_spread_error = np.mean([p['spread_error'] for p in predictions])
            avg_total_error = np.mean([p['total_error'] for p in predictions])
            winner_accuracy = np.mean([p['winner_correct'] for p in predictions]) * 100
            
            print("\n" + "="*60)
            print(f"📊 WEEK {latest_week} ACCURACY REPORT")
            print("="*60)
            print(f"Games Predicted: {len(predictions)}")
            print(f"Correct Winners: {sum(p['winner_correct'] for p in predictions)}/{len(predictions)} ({winner_accuracy:.1f}%)")
            print(f"Avg Spread Error: {avg_spread_error:.1f} points")
            print(f"Avg Total Error: {avg_total_error:.1f} points")
            print("="*60)
            
            # Save report
            report = {
                'week': int(latest_week),
                'season': int(current_season),
                'games_predicted': len(predictions),
                'winner_accuracy': round(winner_accuracy, 1),
                'avg_spread_error': round(avg_spread_error, 1),
                'avg_total_error': round(avg_total_error, 1),
                'predictions': predictions,
                'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open('weekly_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            print("✅ Report saved to weekly_report.json")
            
            # Update season history
            try:
                with open('season_history.json', 'r') as f:
                    history = json.load(f)
            except FileNotFoundError:
                history = {'weeks': []}
            
            # Add or update this week
            week_entry = {
                'week': int(latest_week),
                'season': int(current_season),
                'games': len(predictions),
                'winner_accuracy': round(winner_accuracy, 1),
                'spread_error': round(avg_spread_error, 1),
                'total_error': round(avg_total_error, 1)
            }
            
            # Remove existing entry for this week if present
            history['weeks'] = [w for w in history['weeks'] if not (w['week'] == latest_week and w['season'] == current_season)]
            history['weeks'].append(week_entry)
            
            with open('season_history.json', 'w') as f:
                json.dump(history, f, indent=2)
            
            print(f"   ✅ Added Week {latest_week} to season history")
            print(f"   📚 Season history saved ({len(history['weeks'])} weeks tracked)")
            
        else:
            print("\n" + "="*60)
            print(f"📊 WEEK {latest_week} ACCURACY REPORT")
            print("="*60)
            print("Games Predicted: 0")
            print("Correct Winners: 0/0 (0.0%)")
            print("Avg Spread Error: 0.0 points")
            print("Avg Total Error: 0.0 points")
            print("="*60)
        
        print("✅ Weekly report generated successfully")

    def load_nfl_data(self):
        """Load the latest NFL data"""
        print(f"\n📊 Loading NFL data for {self.data_years}...")
        
        try:
            import nfl_data_py as nfl
            
            self.pbp_data = nfl.import_pbp_data(self.data_years)
            
            float_cols = self.pbp_data.select_dtypes(include=['float64']).columns
            self.pbp_data[float_cols] = self.pbp_data[float_cols].astype('float32')
            print("Downcasting floats.")
            
            print(f"✅ Loaded {len(self.pbp_data):,} plays")
            
            self.schedule_data = nfl.import_schedules(self.data_years)
            print(f"✅ Loaded {len(self.schedule_data):,} games")
            
            self.pbp_regular = self.pbp_data[self.pbp_data['week'] <= 18].copy()
            self.schedule_regular = self.schedule_data[self.schedule_data['week'] <= 18].copy()
            
            print(f"✅ Regular season: {len(self.pbp_regular):,} plays, {len(self.schedule_regular):,} games")
            return True
            
        except Exception as e:
            print(f"❌ Error loading NFL data: {e}")
            return False

    def calculate_league_average(self):
        """Calculate league average PPG from recent games"""
        print(f"\n📊 Calculating league average PPG...")

        current_season = max(self.data_years)

        # Get completed games from current season
        completed_games = self.schedule_regular[
            (self.schedule_regular['season'] == current_season) &
            (self.schedule_regular['home_score'].notna())
        ]

        if len(completed_games) == 0:
            print("   ⚠️ No completed games found, using default 22.0 PPG")
            self.league_avg_ppg = 22.0
            return

        # Calculate average points per team across all games
        all_scores = []
        for _, game in completed_games.iterrows():
            all_scores.append(float(game['home_score']))
            all_scores.append(float(game['away_score']))

        self.league_avg_ppg = np.mean(all_scores)
        print(f"   ✅ League average: {self.league_avg_ppg:.2f} PPG (from {len(completed_games)} games)")

    def calculate_epa_metrics(self):
        """Calculate EPA (Expected Points Added) metrics for all teams"""
        print(f"\n⚡ Calculating EPA metrics...")
        
        # Filter to plays with EPA data
        valid_plays = self.pbp_regular[
            (self.pbp_regular['epa'].notna()) & 
            (self.pbp_regular['play_type'].isin(['pass', 'run']))
        ].copy()
        
        print(f"   Analyzing {len(valid_plays):,} plays with EPA data")
        
        for season in self.data_years:
            season_plays = valid_plays[valid_plays['season'] == season]
            
            # Offensive EPA
            offense_epa = season_plays.groupby('posteam').agg({
                'epa': ['mean', 'sum', 'count']
            }).reset_index()
            offense_epa.columns = ['team', 'epa_per_play', 'total_epa', 'play_count']
            
            # Pass EPA
            pass_plays = season_plays[season_plays['pass'] == 1]
            pass_epa = pass_plays.groupby('posteam')['epa'].mean().reset_index()
            pass_epa.columns = ['team', 'pass_epa_per_play']
            
            # Rush EPA
            rush_plays = season_plays[season_plays['rush'] == 1]
            rush_epa = rush_plays.groupby('posteam')['epa'].mean().reset_index()
            rush_epa.columns = ['team', 'rush_epa_per_play']
            
            # Defensive EPA (lower is better)
            defense_epa = season_plays.groupby('defteam')['epa'].mean().reset_index()
            defense_epa.columns = ['team', 'def_epa_per_play']
            
            # Merge all
            team_epa = offense_epa.merge(pass_epa, on='team', how='left')
            team_epa = team_epa.merge(rush_epa, on='team', how='left')
            team_epa = team_epa.merge(defense_epa, on='team', how='left')
            
            # Success rate (plays with positive EPA)
            team_epa['epa_success_rate'] = season_plays.groupby('posteam').apply(
                lambda x: (x['epa'] > 0).mean()
            ).values
            
            # Store by team and season
            for _, row in team_epa.iterrows():
                team = row['team']
                if team not in self.epa_metrics:
                    self.epa_metrics[team] = {}
                self.epa_metrics[team][season] = {
                    'epa_per_play': float(row['epa_per_play']),
                    'pass_epa': float(row['pass_epa_per_play']),
                    'rush_epa': float(row['rush_epa_per_play']),
                    'def_epa': float(row['def_epa_per_play']),
                    'success_rate': float(row['epa_success_rate'])
                }
        
        print(f"✅ EPA metrics calculated for {len(self.epa_metrics)} teams")
    
    def calculate_ol_dl_strength(self):
        """Calculate OL and DL strength from pressure/sack metrics"""
        print(f"\n🏈 Calculating OL/DL matchup strength...")
        
        # Get dropback plays
        pass_plays = self.pbp_regular[
            (self.pbp_regular['pass'] == 1) & 
            (self.pbp_regular['qb_dropback'] == 1)
        ].copy()
        
        print(f"   Analyzing {len(pass_plays):,} dropback plays")
        
        for season in self.data_years:
            season_passes = pass_plays[pass_plays['season'] == season]
            
            # OFFENSIVE LINE STRENGTH (protecting QB)
            ol_stats = season_passes.groupby('posteam').agg({
                'sack': ['sum', 'count'],
                'qb_hit': 'sum',
                'epa': 'mean'
            }).reset_index()
            ol_stats.columns = ['team', 'sacks_allowed', 'dropbacks', 'qb_hits', 'pass_epa']
            
            # Calculate rates
            ol_stats['sack_rate'] = ol_stats['sacks_allowed'] / ol_stats['dropbacks']
            ol_stats['pressure_rate'] = (ol_stats['sacks_allowed'] + ol_stats['qb_hits']) / ol_stats['dropbacks']
            
            # OL Score: Lower pressure = better (invert for scoring)
            ol_stats['ol_pass_block_score'] = 100 - (ol_stats['pressure_rate'] * 100)
            ol_stats['ol_rank'] = ol_stats['ol_pass_block_score'].rank(ascending=False)
            
            # DEFENSIVE LINE STRENGTH (pressuring QB)
            dl_stats = season_passes.groupby('defteam').agg({
                'sack': ['sum', 'count'],
                'qb_hit': 'sum',
                'epa': 'mean'
            }).reset_index()
            dl_stats.columns = ['team', 'sacks_generated', 'opponent_dropbacks', 'qb_hits_gen', 'pass_epa_allowed']
            
            # Calculate rates
            dl_stats['sack_rate_gen'] = dl_stats['sacks_generated'] / dl_stats['opponent_dropbacks']
            dl_stats['pressure_rate_gen'] = (dl_stats['sacks_generated'] + dl_stats['qb_hits_gen']) / dl_stats['opponent_dropbacks']
            
            # DL Score: Higher pressure = better
            dl_stats['dl_pass_rush_score'] = dl_stats['pressure_rate_gen'] * 100
            dl_stats['dl_rank'] = dl_stats['dl_pass_rush_score'].rank(ascending=False)
            
            # Store by team and season
            for _, ol_row in ol_stats.iterrows():
                team = ol_row['team']
                if team not in self.ol_dl_rankings:
                    self.ol_dl_rankings[team] = {}
                if season not in self.ol_dl_rankings[team]:
                    self.ol_dl_rankings[team][season] = {}
                
                self.ol_dl_rankings[team][season]['ol'] = {
                    'rank': float(ol_row['ol_rank']),
                    'score': float(ol_row['ol_pass_block_score']),
                    'sack_rate': float(ol_row['sack_rate']),
                    'pressure_rate': float(ol_row['pressure_rate'])
                }
            
            for _, dl_row in dl_stats.iterrows():
                team = dl_row['team']
                if team not in self.ol_dl_rankings:
                    self.ol_dl_rankings[team] = {}
                if season not in self.ol_dl_rankings[team]:
                    self.ol_dl_rankings[team][season] = {}
                
                self.ol_dl_rankings[team][season]['dl'] = {
                    'rank': float(dl_row['dl_rank']),
                    'score': float(dl_row['dl_pass_rush_score']),
                    'sack_rate': float(dl_row['sack_rate_gen']),
                    'pressure_rate': float(dl_row['pressure_rate_gen'])
                }
        
        print(f"✅ OL/DL rankings calculated for {len(self.ol_dl_rankings)} teams")
    
    def get_epa_for_team(self, team, season):
        """Get EPA metrics for a team in a season"""
        if team not in self.epa_metrics or season not in self.epa_metrics[team]:
            return 0.0, 0.0
        
        epa_data = self.epa_metrics[team][season]
        return epa_data.get('epa_per_play', 0.0), epa_data.get('def_epa', 0.0)
    
    def get_ol_dl_matchup_score(self, offense_team, defense_team, season):
        """Calculate OL vs DL matchup advantage"""
        if offense_team not in self.ol_dl_rankings or defense_team not in self.ol_dl_rankings:
            return 0.0
        
        if season not in self.ol_dl_rankings[offense_team] or season not in self.ol_dl_rankings[defense_team]:
            return 0.0
        
        ol_score = self.ol_dl_rankings[offense_team][season].get('ol', {}).get('score', 50.0)
        dl_score = self.ol_dl_rankings[defense_team][season].get('dl', {}).get('score', 50.0)
        
        # Positive = OL advantage, Negative = DL advantage
        return ol_score - dl_score
    
    def create_player_logs(self):
        """Create game logs for all players"""
        print(f"\n🏈 Creating player game logs...")
        
        # Passing logs
        passing_plays = self.pbp_regular[
            (self.pbp_regular['play_type'] == 'pass') & 
            (self.pbp_regular['passer_player_name'].notna())
        ].copy()
        
        self.passing_logs = passing_plays.groupby([
            'season', 'game_id', 'passer_player_name', 'posteam', 'defteam', 'week', 'game_date'
        ]).agg({
            'yards_gained': 'sum',
            'complete_pass': 'sum',
            'pass_attempt': 'sum',
            'pass_touchdown': 'sum',
            'interception': 'sum'
        }).reset_index()
        
        self.passing_logs = self.passing_logs.rename(columns={
            'passer_player_name': 'player_name',
            'yards_gained': 'passing_yards',
            'complete_pass': 'completions',
            'pass_attempt': 'attempts',
            'pass_touchdown': 'passing_tds'
        })
        
        # Receiving logs
        receiving_plays = self.pbp_regular[
            (self.pbp_regular['play_type'] == 'pass') & 
            (self.pbp_regular['receiver_player_name'].notna()) &
            (self.pbp_regular['complete_pass'] == 1)
        ].copy()
        
        self.receiving_logs = receiving_plays.groupby([
            'season', 'game_id', 'receiver_player_name', 'posteam', 'defteam', 'week', 'game_date'
        ]).agg({
            'yards_gained': 'sum',
            'complete_pass': 'sum',
            'pass_touchdown': 'sum'
        }).reset_index()
        
        self.receiving_logs = self.receiving_logs.rename(columns={
            'receiver_player_name': 'player_name',
            'yards_gained': 'receiving_yards',
            'complete_pass': 'receptions',
            'pass_touchdown': 'receiving_tds'
        })
        
        # Add target share
        team_targets = receiving_plays.groupby(['season', 'game_id', 'posteam']).size().reset_index(name='team_targets')
        player_targets = receiving_plays.groupby(['season', 'game_id', 'receiver_player_name', 'posteam']).size().reset_index(name='player_targets')
        
        targets_merged = player_targets.merge(team_targets, on=['season', 'game_id', 'posteam'])
        targets_merged['target_share'] = targets_merged['player_targets'] / targets_merged['team_targets']
        
        self.receiving_logs = self.receiving_logs.merge(
            targets_merged[['season', 'game_id', 'receiver_player_name', 'target_share']],
            left_on=['season', 'game_id', 'player_name'],
            right_on=['season', 'game_id', 'receiver_player_name'],
            how='left'
        ).drop('receiver_player_name', axis=1)
        
        # Rushing logs
        rushing_plays = self.pbp_regular[
            (self.pbp_regular['play_type'] == 'run') & 
            (self.pbp_regular['rusher_player_name'].notna())
        ].copy()
        
        self.rushing_logs = rushing_plays.groupby([
            'season', 'game_id', 'rusher_player_name', 'posteam', 'defteam', 'week', 'game_date'
        ]).agg({
            'yards_gained': 'sum',
            'play_type': 'count',
            'rush_touchdown': 'sum'
        }).reset_index()
        
        self.rushing_logs = self.rushing_logs.rename(columns={
            'rusher_player_name': 'player_name',
            'yards_gained': 'rushing_yards',
            'play_type': 'attempts',
            'rush_touchdown': 'rushing_tds'
        })
        
        print(f"✅ Created logs: {len(self.passing_logs)} passing, {len(self.receiving_logs)} receiving, {len(self.rushing_logs)} rushing")
    
    def create_rolling_features(self, df, stat_cols, window_sizes=[4, 8, 16]):
        """Create cross-season rolling averages with exponential recency weighting"""
        df = df.sort_values(['player_name', 'season', 'game_date']).copy()
        
        for window in window_sizes:
            for stat in stat_cols:
                col_name = f'{stat}_L{window}'
                
                def exponential_rolling_mean(series):
                    result = []
                    for i in range(len(series)):
                        if i == 0:
                            result.append(np.nan)
                        else:
                            start_idx = max(0, i - window)
                            values = series.iloc[start_idx:i]
                            
                            if len(values) == 0:
                                result.append(np.nan)
                            else:
                                weights = np.exp(np.linspace(-1, 0, len(values)))
                                weighted_mean = np.average(values, weights=weights)
                                result.append(weighted_mean)
                    
                    return pd.Series(result, index=series.index)
                
                df[col_name] = df.groupby('player_name')[stat].transform(exponential_rolling_mean)
        
        return df
    
    def build_player_features(self):
        """Build enhanced features for all player types"""
        print(f"\n📈 Building player features...")
        
        self.passing_features = self.create_rolling_features(
            self.passing_logs,
            ['passing_yards', 'completions', 'attempts', 'passing_tds', 'interception']
        )
        
        self.passing_features['completion_pct'] = self.passing_features['completions'] / self.passing_features['attempts']
        self.passing_features['completion_pct_L4'] = self.passing_features['completions_L4'] / self.passing_features['attempts_L4']
        self.passing_features['completion_pct_L8'] = self.passing_features['completions_L8'] / self.passing_features['attempts_L8']
        
        self.receiving_features = self.create_rolling_features(
            self.receiving_logs,
            ['receiving_yards', 'receptions', 'receiving_tds', 'target_share']
        )
        
        self.receiving_features['yards_per_rec'] = self.receiving_features['receiving_yards'] / self.receiving_features['receptions']
        self.receiving_features['yards_per_rec_L4'] = self.receiving_features['receiving_yards_L4'] / self.receiving_features['receptions_L4']
        self.receiving_features['yards_per_rec_L8'] = self.receiving_features['receiving_yards_L8'] / self.receiving_features['receptions_L8']
        
        self.rushing_features = self.create_rolling_features(
            self.rushing_logs,
            ['rushing_yards', 'attempts', 'rushing_tds']
        )
        
        self.rushing_features['yards_per_carry'] = self.rushing_features['rushing_yards'] / self.rushing_features['attempts']
        self.rushing_features['yards_per_carry_L4'] = self.rushing_features['rushing_yards_L4'] / self.rushing_features['attempts_L4']
        self.rushing_features['yards_per_carry_L8'] = self.rushing_features['rushing_yards_L8'] / self.rushing_features['attempts_L8']
        
        print("✅ Player features created")
    
    def create_defense_rankings(self):
        """Create split defensive rankings (pass/rush) by season"""
        print(f"\n🛡️ Creating enhanced defense rankings...")
        
        pass_plays = self.pbp_regular[self.pbp_regular['play_type'] == 'pass']
        pass_defense = pass_plays.groupby(['season', 'defteam']).agg({
            'yards_gained': 'sum',
            'play_type': 'count'
        }).reset_index()
        pass_defense['yards_per_play'] = pass_defense['yards_gained'] / pass_defense['play_type']
        pass_defense['pass_def_rank'] = pass_defense.groupby('season')['yards_per_play'].rank(ascending=True)
        
        rush_plays = self.pbp_regular[self.pbp_regular['play_type'] == 'run']
        rush_defense = rush_plays.groupby(['season', 'defteam']).agg({
            'yards_gained': 'sum',
            'play_type': 'count'
        }).reset_index()
        rush_defense['yards_per_play'] = rush_defense['yards_gained'] / rush_defense['play_type']
        rush_defense['rush_def_rank'] = rush_defense.groupby('season')['yards_per_play'].rank(ascending=True)
        
        self.defense_rankings_data = pass_defense[['season', 'defteam', 'pass_def_rank']].merge(
            rush_defense[['season', 'defteam', 'rush_def_rank']],
            on=['season', 'defteam']
        )
        
        self.passing_features = self.passing_features.merge(
            self.defense_rankings_data[['season', 'defteam', 'pass_def_rank']],
            on=['season', 'defteam'],
            how='left'
        )
        self.passing_features['pass_def_rank'] = self.passing_features['pass_def_rank'].fillna(16)
        
        self.receiving_features = self.receiving_features.merge(
            self.defense_rankings_data[['season', 'defteam', 'pass_def_rank']],
            on=['season', 'defteam'],
            how='left'
        )
        self.receiving_features['pass_def_rank'] = self.receiving_features['pass_def_rank'].fillna(16)
        
        self.rushing_features = self.rushing_features.merge(
            self.defense_rankings_data[['season', 'defteam', 'rush_def_rank']],
            on=['season', 'defteam'],
            how='left'
        )
        self.rushing_features['rush_def_rank'] = self.rushing_features['rush_def_rank'].fillna(16)
        
        print("✅ Split defense rankings created and merged")
    
    def train_models(self):
        """Train all prediction models with ensemble approach"""
        print(f"\n🎯 Training prediction models with ensemble...")
        
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.metrics import mean_absolute_error
        
        def build_ensemble_model(data, feature_cols, target_col, model_name, min_games=3):
            clean_data = data[feature_cols + [target_col, 'season', 'week', 'game_date', 'player_name']].dropna()
            
            game_counts = clean_data.groupby('player_name')['game_date'].nunique()
            valid_players = game_counts[game_counts >= min_games].index
            clean_data = clean_data[clean_data['player_name'].isin(valid_players)]
            
            clean_data = clean_data[clean_data['week'] >= 3]
            
            if len(clean_data) == 0:
                print(f"⚠️ {model_name}: No data after filtering")
                return None, None, None
            
            clean_data = clean_data.sort_values(['season', 'week', 'game_date'])
            
            seasons_available = sorted(clean_data['season'].unique())
            
            if len(seasons_available) >= 2:
                current_season = max(seasons_available)
                current_week_count = clean_data[clean_data['season'] == current_season]['week'].nunique()
                
                if current_week_count >= 12:
                    train_mask = ((clean_data['season'] < current_season) | 
                                 ((clean_data['season'] == current_season) & 
                                  (clean_data['week'] <= 12)))
                    test_mask = ((clean_data['season'] == current_season) & 
                                (clean_data['week'] > 12))
                else:
                    prev_season = seasons_available[-2]
                    train_mask = clean_data['season'] < prev_season
                    test_mask = clean_data['season'] == prev_season
            else:
                split_idx = int(len(clean_data) * 0.8)
                train_mask = clean_data.index.isin(clean_data.index[:split_idx])
                test_mask = clean_data.index.isin(clean_data.index[split_idx:])
            
            X_train = clean_data[train_mask][feature_cols]
            y_train = clean_data[train_mask][target_col]
            X_test = clean_data[test_mask][feature_cols]
            y_test = clean_data[test_mask][target_col]
            
            if len(X_train) == 0:
                return None, None, None
            
            models = []
            
            gb_model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
            gb_model.fit(X_train, y_train)
            models.append(gb_model)
            
            nn_model = MLPRegressor(
                hidden_layer_sizes=(50, 25),
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
            nn_model.fit(X_train, y_train)
            models.append(nn_model)
            
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            models.append(lr_model)
            
            if len(X_test) > 0:
                ensemble_pred = np.mean([m.predict(X_test) for m in models], axis=0)
                mae = mean_absolute_error(y_test, ensemble_pred)
                
                train_seasons = sorted(clean_data[train_mask]['season'].unique())
                test_seasons = sorted(clean_data[test_mask]['season'].unique())
                print(f"✅ {model_name}: MAE {mae:.1f} (ensemble of 3 models)")
                print(f"   Trained on: {train_seasons} ({len(X_train)} games, {len(valid_players)} players)")
                print(f"   Tested on: {test_seasons} ({len(X_test)} games)")
            
            return models[0], models[1], models[2]
        
        passing_features = [
            'passing_yards_L4', 'passing_yards_L8', 'passing_yards_L16',
            'completion_pct_L4', 'completion_pct_L8',
            'attempts_L4', 'attempts_L8',
            'passing_tds_L4', 'passing_tds_L8',
            'interception_L4',
            'pass_def_rank'
        ]
        
        receiving_features = [
            'receiving_yards_L4', 'receiving_yards_L8', 'receiving_yards_L16',
            'receptions_L4', 'receptions_L8',
            'yards_per_rec_L4', 'yards_per_rec_L8',
            'receiving_tds_L4', 'receiving_tds_L8',
            'target_share_L4', 'target_share_L8',
            'pass_def_rank'
        ]
        
        rushing_features = [
            'rushing_yards_L4', 'rushing_yards_L8', 'rushing_yards_L16',
            'attempts_L4', 'attempts_L8',
            'yards_per_carry_L4', 'yards_per_carry_L8',
            'rushing_tds_L4', 'rushing_tds_L8',
            'rush_def_rank'
        ]
        
        gb, nn, lr = build_ensemble_model(
            self.passing_features, passing_features, 'passing_yards', 'Passing Yards'
        )
        if gb: self.ensemble_models['passing'] = [gb, nn, lr]
        
        gb, nn, lr = build_ensemble_model(
            self.receiving_features, receiving_features, 'receiving_yards', 'Receiving Yards'
        )
        if gb: self.ensemble_models['receiving'] = [gb, nn, lr]
        
        gb, nn, lr = build_ensemble_model(
            self.receiving_features,
            ['receptions_L4', 'receptions_L8', 'receptions_L16', 'receiving_yards_L4', 
             'yards_per_rec_L4', 'target_share_L4', 'pass_def_rank'],
            'receptions', 'Receptions'
        )
        if gb: self.ensemble_models['receptions'] = [gb, nn, lr]
        
        gb, nn, lr = build_ensemble_model(
            self.rushing_features, rushing_features, 'rushing_yards', 'Rushing Yards'
        )
        if gb: self.ensemble_models['rushing'] = [gb, nn, lr]
        
        print(f"✅ Trained {len(self.ensemble_models)} ensemble models (3 models each)")
    
    def get_team_stats_before_game(self, team, season, week, game_date):
        """Get team stats using ONLY games before this date (NO DATA LEAKAGE)"""
        
        prior_games = self.schedule_regular[
            (
                ((self.schedule_regular['home_team'] == team) | 
                 (self.schedule_regular['away_team'] == team))
            ) &
            (
                (self.schedule_regular['season'] < season) |
                ((self.schedule_regular['season'] == season) & 
                 (pd.to_datetime(self.schedule_regular['gameday']) < pd.to_datetime(game_date)))
            )
        ].copy()
        
        if len(prior_games) == 0:
            return {
                'points_L4': self.league_avg_ppg, 'points_L8': self.league_avg_ppg,
                'opp_points_L4': self.league_avg_ppg, 'opp_points_L8': self.league_avg_ppg,
                'yards_L4': 350.0, 'win_pct_L8': 0.5,
                'turnovers_L4': 1.0
            }
        
        team_scores = []
        opp_scores = []
        total_yards = []
        turnovers = []
        
        for _, game in prior_games.iterrows():
            if game['home_team'] == team:
                team_scores.append(game['home_score'])
                opp_scores.append(game['away_score'])
            else:
                team_scores.append(game['away_score'])
                opp_scores.append(game['home_score'])
            
            game_plays = self.pbp_regular[
                (self.pbp_regular['game_id'] == game['game_id']) &
                (self.pbp_regular['posteam'] == team)
            ]
            total_yards.append(game_plays['yards_gained'].sum() if len(game_plays) > 0 else 350)
            
            game_turnovers = game_plays['interception'].sum() + game_plays['fumble_lost'].sum()
            turnovers.append(game_turnovers if len(game_plays) > 0 else 1)
        
        stats = {
            'points_L4': np.mean(team_scores[-4:]) if len(team_scores) >= 1 else self.league_avg_ppg,
            'points_L8': np.mean(team_scores[-8:]) if len(team_scores) >= 1 else self.league_avg_ppg,
            'opp_points_L4': np.mean(opp_scores[-4:]) if len(opp_scores) >= 1 else self.league_avg_ppg,
            'opp_points_L8': np.mean(opp_scores[-8:]) if len(opp_scores) >= 1 else self.league_avg_ppg,
            'yards_L4': np.mean(total_yards[-4:]) if len(total_yards) >= 1 else 350.0,
            'win_pct_L8': np.mean([1 if ts > os else 0 for ts, os in zip(team_scores[-8:], opp_scores[-8:])]) if len(team_scores) >= 1 else 0.5,
            'turnovers_L4': np.mean(turnovers[-4:]) if len(turnovers) >= 1 else 1.0
        }
        
        return stats
    
    def calculate_rest_days(self, team, game_date):
        """Calculate days of rest before this game"""
        prior_game = self.schedule_regular[
            (
                ((self.schedule_regular['home_team'] == team) | 
                 (self.schedule_regular['away_team'] == team))
            ) &
            (pd.to_datetime(self.schedule_regular['gameday']) < pd.to_datetime(game_date))
        ].sort_values('gameday', ascending=False)
        
        if len(prior_game) == 0:
            return 7
        
        last_game_date = pd.to_datetime(prior_game.iloc[0]['gameday'])
        current_game_date = pd.to_datetime(game_date)
        rest_days = (current_game_date - last_game_date).days
        
        return rest_days
    
    def is_division_game(self, team1, team2):
        """Check if two teams are in the same division"""
        divisions = {
            'AFC East': ['BUF', 'MIA', 'NE', 'NYJ'],
            'AFC North': ['BAL', 'CIN', 'CLE', 'PIT'],
            'AFC South': ['HOU', 'IND', 'JAX', 'TEN'],
            'AFC West': ['DEN', 'KC', 'LV', 'LAC'],
            'NFC East': ['DAL', 'NYG', 'PHI', 'WAS'],
            'NFC North': ['CHI', 'DET', 'GB', 'MIN'],
            'NFC South': ['ATL', 'CAR', 'NO', 'TB'],
            'NFC West': ['ARI', 'LAR', 'SF', 'SEA']
        }
        
        for division, teams in divisions.items():
            if team1 in teams and team2 in teams:
                return True
        return False
    
    def train_game_models(self):
        """Train game outcome models with EPA and OL/DL features"""
        print(f"\n🏟️ Training enhanced game outcome models WITH EPA & OL/DL...")
        
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, accuracy_score
        
        matchups = []
        
        print("   Building point-in-time game features with EPA & OL/DL...")
        
        for idx, game in self.schedule_regular.iterrows():
            if idx % 100 == 0:
                print(f"   Processing game {idx}/{len(self.schedule_regular)}...")
            
            home_team = game['home_team']
            away_team = game['away_team']
            game_date = game['gameday']
            season = game['season']
            week = game['week']
            
            home_stats = self.get_team_stats_before_game(home_team, season, week, game_date)
            away_stats = self.get_team_stats_before_game(away_team, season, week, game_date)
            
            home_rest = self.calculate_rest_days(home_team, game_date)
            away_rest = self.calculate_rest_days(away_team, game_date)
            
            division_game = self.is_division_game(home_team, away_team)
            
            # NEW: Get EPA metrics
            home_epa, home_def_epa = self.get_epa_for_team(home_team, season)
            away_epa, away_def_epa = self.get_epa_for_team(away_team, season)
            
            # NEW: Get OL/DL matchup score
            ol_dl_matchup = self.get_ol_dl_matchup_score(home_team, away_team, season)
            
            matchup = {
                'season': season,
                'week': week,
                'game_date': game_date,
                'home_team': home_team,
                'away_team': away_team,
                'home_score': game['home_score'],
                'away_score': game['away_score'],
                'total_points': game['home_score'] + game['away_score'],
                'point_diff': game['home_score'] - game['away_score'],
                
                'home_points_L4': home_stats['points_L4'],
                'home_opp_points_L4': home_stats['opp_points_L4'],
                'home_yards_L4': home_stats['yards_L4'],
                'home_points_L8': home_stats['points_L8'],
                'home_opp_points_L8': home_stats['opp_points_L8'],
                'home_win_pct_L8': home_stats['win_pct_L8'],
                'home_turnovers_L4': home_stats['turnovers_L4'],
                
                'away_points_L4': away_stats['points_L4'],
                'away_opp_points_L4': away_stats['opp_points_L4'],
                'away_yards_L4': away_stats['yards_L4'],
                'away_points_L8': away_stats['points_L8'],
                'away_opp_points_L8': away_stats['opp_points_L8'],
                'away_win_pct_L8': away_stats['win_pct_L8'],
                'away_turnovers_L4': away_stats['turnovers_L4'],
                
                'home_rest_days': home_rest,
                'away_rest_days': away_rest,
                'rest_advantage': home_rest - away_rest,
                'division_game': 1 if division_game else 0,
                
                # NEW EPA features
                'home_epa': home_epa,
                'away_epa': away_epa,
                
                # NEW OL/DL matchup feature
                'ol_dl_matchup': ol_dl_matchup
            }
            matchups.append(matchup)
        
        matchups_df = pd.DataFrame(matchups)
        print(f"   ✅ Created {len(matchups_df)} game matchups with EPA & OL/DL")
        
        model_games = matchups_df[matchups_df['week'] >= 5].copy()
        print(f"   After filtering: {len(model_games)} games")
        
        # UPDATED: Game features now include EPA and OL/DL (20 features total)
        game_features = [
            'home_points_L4', 'home_opp_points_L4', 'home_yards_L4',
            'home_points_L8', 'home_opp_points_L8',
            'home_win_pct_L8', 'home_turnovers_L4',
            'away_points_L4', 'away_opp_points_L4', 'away_yards_L4', 
            'away_points_L8', 'away_opp_points_L8',
            'away_win_pct_L8', 'away_turnovers_L4',
            'home_rest_days', 'away_rest_days', 'rest_advantage',
            'division_game',
            'home_epa', 'ol_dl_matchup'  # NEW: EPA and OL/DL features
        ]
        
        clean_games = model_games[game_features + ['total_points', 'point_diff', 'season']].dropna()
        
        if len(clean_games) == 0:
            print("⚠️ No clean game data for modeling")
            return
        
        print(f"   Available seasons: {sorted(clean_games['season'].unique())}")
        
        if 2024 in clean_games['season'].unique():
            train_mask = clean_games['season'] != 2024
            test_mask = clean_games['season'] == 2024
        else:
            split_idx = int(len(clean_games) * 0.8)
            train_mask = clean_games.index.isin(clean_games.index[:split_idx])
            test_mask = clean_games.index.isin(clean_games.index[split_idx:])
        
        X_train = clean_games[train_mask][game_features]
        X_test = clean_games[test_mask][game_features]
        
        print(f"   Evaluation sets: Train={len(X_train)} games, Test={len(X_test)} games")
        
        # Train ensemble for total points
        y_train_total = clean_games[train_mask]['total_points']
        y_test_total = clean_games[test_mask]['total_points']
        
        total_gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
        total_nn = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42, early_stopping=True)
        total_lr = LinearRegression()
        
        total_gb.fit(X_train, y_train_total)
        total_nn.fit(X_train, y_train_total)
        total_lr.fit(X_train, y_train_total)
        
        if len(X_test) > 0:
            total_pred = np.mean([
                total_gb.predict(X_test),
                total_nn.predict(X_test),
                total_lr.predict(X_test)
            ], axis=0)
            total_mae = mean_absolute_error(y_test_total, total_pred)
            print(f"✅ Total Points Model: MAE {total_mae:.1f} points (ensemble WITH EPA/OL-DL)")
        
        # Train ensemble for point spread
        y_train_spread = clean_games[train_mask]['point_diff']
        y_test_spread = clean_games[test_mask]['point_diff']
        
        spread_gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
        spread_nn = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42, early_stopping=True)
        spread_lr = LinearRegression()
        
        spread_gb.fit(X_train, y_train_spread)
        spread_nn.fit(X_train, y_train_spread)
        spread_lr.fit(X_train, y_train_spread)
        
        if len(X_test) > 0:
            spread_pred_raw = np.mean([
                spread_gb.predict(X_test),
                spread_nn.predict(X_test),
                spread_lr.predict(X_test)
            ], axis=0)
            spread_mae = mean_absolute_error(y_test_spread, spread_pred_raw)
            win_accuracy = accuracy_score(y_test_spread > 0, spread_pred_raw > 0)
            print(f"✅ Point Spread Model: MAE {spread_mae:.1f} points, Win Accuracy {win_accuracy:.1%} (ensemble WITH EPA/OL-DL)")
            
            # Calibration
            print(f"\n   🎯 Calibrating spread predictions...")
            avg_pred_spread = np.mean(np.abs(spread_pred_raw))
            avg_actual_spread = np.mean(np.abs(y_test_spread))
            
            if avg_pred_spread > 0:
                spread_calibration_factor = avg_actual_spread / avg_pred_spread
            else:
                spread_calibration_factor = 0.7
            
            spread_calibration_factor *= 0.85
            
            self.calibration_params['spread_factor'] = spread_calibration_factor
            
            spread_pred_calibrated = spread_pred_raw * spread_calibration_factor
            calibrated_mae = mean_absolute_error(y_test_spread, spread_pred_calibrated)
            calibrated_accuracy = accuracy_score(y_test_spread > 0, spread_pred_calibrated > 0)
            
            print(f"   Calibration factor: {spread_calibration_factor:.3f}")
            print(f"   Calibrated MAE: {calibrated_mae:.1f} points")
            print(f"   Calibrated Win Accuracy: {calibrated_accuracy:.1%}")
        else:
            self.calibration_params['spread_factor'] = 0.7
        
        # Retrain on ALL data for production
        print(f"\n   📊 Retraining on ALL data for production predictions...")
        X_all = clean_games[game_features]
        y_total_all = clean_games['total_points']
        y_spread_all = clean_games['point_diff']
        
        total_gb_prod = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
        total_nn_prod = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42, early_stopping=True)
        total_lr_prod = LinearRegression()
        
        total_gb_prod.fit(X_all, y_total_all)
        total_nn_prod.fit(X_all, y_total_all)
        total_lr_prod.fit(X_all, y_total_all)
        
        spread_gb_prod = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)
        spread_nn_prod = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42, early_stopping=True)
        spread_lr_prod = LinearRegression()
        
        spread_gb_prod.fit(X_all, y_spread_all)
        spread_nn_prod.fit(X_all, y_spread_all)
        spread_lr_prod.fit(X_all, y_spread_all)
        
        print(f"   ✅ Production models trained on {len(X_all)} games from {sorted(clean_games['season'].unique())}")
        
        self.ensemble_models['game_total'] = [total_gb_prod, total_nn_prod, total_lr_prod]
        self.ensemble_models['game_spread'] = [spread_gb_prod, spread_nn_prod, spread_lr_prod]
        
        print(f"✅ Game outcome ensemble models trained with EPA & OL/DL features")
    
    def create_team_data(self):
        """Create team performance data for app"""
        print(f"\n🏟️ Creating team performance data...")
        
        current_season = max(self.data_years)
        
        # Get the most recent week with completed games
        completed_games = self.schedule_regular[
            (self.schedule_regular['season'] == current_season) &
            (self.schedule_regular['home_score'].notna())
        ]
        
        if len(completed_games) > 0:
            latest_week = int(completed_games['week'].max())
            latest_date = completed_games['gameday'].max()
            print(f"   Using data through Week {latest_week} of {current_season}")
        else:
            # No completed games this season, use last game of previous season
            latest_week = 18
            latest_date = f"{current_season}-01-15"
            print(f"   No completed games in {current_season}, using {current_season-1} data")
        
        team_stats = {}
        for team in self.schedule_regular['home_team'].unique():
            # Get stats up through the most recent completed week
            stats = self.get_team_stats_before_game(
                team, 
                current_season,
                latest_week + 1,  # Next week after latest completed
                latest_date
            )
            team_stats[team] = stats
        
        self.team_data = team_stats

        # Calculate and save actual league average
        all_points = [stats['points_L4'] for stats in team_stats.values() if 'points_L4' in stats]
        league_avg_ppg = np.mean(all_points) if all_points else 22.0

        with open('league_average.json', 'w') as f:
            json.dump({'ppg': float(league_avg_ppg), 'season': int(current_season)}, f, indent=2)

        print(f"   League average PPG: {league_avg_ppg:.1f}")

        # Verify data quality
        valid_teams = sum(1 for stats in team_stats.values() if stats['points_L4'] > 0)
        print(f"✅ Created team data for {len(team_stats)} teams ({valid_teams} with valid stats)")
        
        if valid_teams == 0:
            print(f"⚠️ WARNING: No teams have valid stats! Check data availability.")
    
    def create_weekly_schedule_predictions(self):
        """Create predictions for upcoming week WITH EPA & OL/DL"""
        print(f"\n📅 Creating weekly schedule predictions WITH EPA & OL/DL...")
        
        # Determine target week
        current_season = max(self.data_years)
        completed_weeks = self.schedule_regular[
            (self.schedule_regular['season'] == current_season) &
            (self.schedule_regular['home_score'].notna())
        ]['week'].max()
        
        target_week = completed_weeks + 1 if pd.notna(completed_weeks) else 1
        print(f"   Generating predictions for Week {target_week}")
        
        # Get games for target week
        upcoming_games = self.schedule_regular[
            (self.schedule_regular['season'] == current_season) &
            (self.schedule_regular['week'] == target_week)
        ].copy()
        
        if len(upcoming_games) == 0:
            print(f"   ⚠️ No games found for Week {target_week}")
            with open('weekly_schedule.json', 'w') as f:
                json.dump([], f, indent=2)
            return []
        
        weekly_predictions = []
        
        for _, game in upcoming_games.iterrows():
            try:
                home_team = game['home_team']
                away_team = game['away_team']
                game_date = game.get('gameday', '')
                
                # Get team stats
                home_stats = self.team_data.get(home_team, {})
                away_stats = self.team_data.get(away_team, {})
                
                if not home_stats or not away_stats:
                    print(f"   ⚠️ Missing stats for {away_team} @ {home_team}")
                    continue
                
                # Get EPA metrics with fallback to 0.0
                home_epa = 0.0
                away_epa = 0.0
                
                if home_team in self.epa_metrics and str(current_season) in self.epa_metrics[home_team]:
                    home_epa = self.epa_metrics[home_team][str(current_season)].get('epa_per_play', 0.0)
                
                if away_team in self.epa_metrics and str(current_season) in self.epa_metrics[away_team]:
                    away_epa = self.epa_metrics[away_team][str(current_season)].get('epa_per_play', 0.0)
                
                # Get OL/DL matchup score with fallback to 0.0
                ol_dl_score = 0.0
                ol_dl_info = None
                
                if (home_team in self.ol_dl_rankings and away_team in self.ol_dl_rankings and
                    str(current_season) in self.ol_dl_rankings[home_team] and 
                    str(current_season) in self.ol_dl_rankings[away_team]):
                    
                    ol_data = self.ol_dl_rankings[home_team][str(current_season)].get('ol', {})
                    dl_data = self.ol_dl_rankings[away_team][str(current_season)].get('dl', {})
                    
                    ol_score = ol_data.get('score', 50.0)
                    dl_score = dl_data.get('score', 50.0)
                    ol_dl_score = ol_score - dl_score
                    
                    # Determine advantage
                    if ol_dl_score > 15:
                        advantage = 'strong_offense'
                        explanation = f"{home_team} OL dominates {away_team} DL"
                    elif ol_dl_score > 5:
                        advantage = 'offense'
                        explanation = f"{home_team} OL has edge"
                    elif ol_dl_score < -15:
                        advantage = 'strong_defense'
                        explanation = f"{away_team} DL dominates - expect pressure"
                    elif ol_dl_score < -5:
                        advantage = 'defense'
                        explanation = f"{away_team} DL has edge"
                    else:
                        advantage = 'neutral'
                        explanation = "Evenly matched trenches"
                    
                    ol_dl_info = {
                        'matchup_score': round(ol_dl_score, 1),
                        'advantage': advantage,
                        'explanation': explanation,
                        'ol_rank': ol_data.get('rank', 16),
                        'ol_score': round(ol_score, 1),
                        'dl_rank': dl_data.get('rank', 16),
                        'dl_score': round(dl_score, 1)
                    }
                
                # Build 20-feature array with NaN protection
                features = np.array([[
                    float(home_stats.get('points_L4', self.league_avg_ppg)),
                    float(home_stats.get('opp_points_L4', self.league_avg_ppg)),
                    float(home_stats.get('yards_L4', 350.0)),
                    float(home_stats.get('points_L8', self.league_avg_ppg)),
                    float(home_stats.get('opp_points_L8', self.league_avg_ppg)),
                    float(home_stats.get('win_pct_L8', 0.5)),
                    float(home_stats.get('turnovers_L4', 1.0)),
                    float(away_stats.get('points_L4', self.league_avg_ppg)),
                    float(away_stats.get('opp_points_L4', self.league_avg_ppg)),
                    float(away_stats.get('yards_L4', 350.0)),
                    float(away_stats.get('points_L8', self.league_avg_ppg)),
                    float(away_stats.get('opp_points_L8', self.league_avg_ppg)),
                    float(away_stats.get('win_pct_L8', 0.5)),
                    float(away_stats.get('turnovers_L4', 1.0)),
                    7.0, 7.0, 0.0, 0.0,  # rest days and division
                    float(home_epa),
                    float(ol_dl_score)
                ]])
                
                # Check for NaN values and replace with defaults
                if np.isnan(features).any():
                    print(f"   ⚠️ NaN detected in features for {away_team} @ {home_team}, using fallbacks")
                    features = np.nan_to_num(features, nan=0.0)
                
                # Make predictions
                total_preds = [m.predict(features)[0] for m in self.ensemble_models['game_total']]
                spread_preds = [m.predict(features)[0] for m in self.ensemble_models['game_spread']]
                
                total_pred = np.mean(total_preds)
                spread_pred_raw = np.mean(spread_preds)
                
                # Apply calibration
                try:
                    with open('calibration_params.json', 'r') as f:
                        calib = json.load(f)
                        spread_factor = calib.get('spread_factor', 0.7)
                except:
                    spread_factor = 0.7
                
                spread_pred = spread_pred_raw * spread_factor
                
                # Calculate scores
                home_score = (total_pred + spread_pred) / 2
                away_score = (total_pred - spread_pred) / 2
                
                # Determine betting context
                if abs(spread_pred) > 10:
                    context = "⚠️ Significant favorite"
                elif total_pred > 50:
                    context = "🔥 High-scoring game expected"
                elif abs(spread_pred) < 3:
                    context = "⚖️ Toss-up game"
                else:
                    context = "📊 Standard matchup"
                
                weekly_predictions.append({
                    'game_id': str(game.get('game_id', '')),
                    'week': int(target_week),
                    'home_team': str(home_team),
                    'away_team': str(away_team),
                    'gameday': pd.to_datetime(game_date).strftime('%Y-%m-%d') if pd.notna(game_date) else 'TBD',
                    'gametime': str(game.get('gametime', 'TBD')),
                    'home_score': float(round(home_score, 1)),
                    'away_score': float(round(away_score, 1)),
                    'total': float(round(total_pred, 1)),
                    'spread': float(round(spread_pred, 1)),
                    'betting_angle': str(context),
                    'ol_dl_matchup': ol_dl_info
                })
                
            except Exception as e:
                print(f"   ❌ Error predicting {away_team} @ {home_team}: {str(e)[:100]}")
                continue
        
        # Save predictions
        with open('weekly_schedule.json', 'w') as f:
            json.dump(weekly_predictions, f, indent=2)
        
        print(f"✅ Created predictions for {len(weekly_predictions)} games in Week {target_week}")
        
        # Also create weekly_report.json for the app
        weekly_report = {
            'week': int(target_week),
            'season': int(current_season),
            'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'games': weekly_predictions,
            'betting_opportunities': [],
            'top_performers': {'qbs': [], 'wrs': [], 'rbs': []}
        }
        
        with open('weekly_report.json', 'w') as f:
            json.dump(weekly_report, f, indent=2)
        
        return weekly_predictions
    
    def extract_current_players(self):
        """Extract current player data for predictions"""
        print(f"\n👥 Extracting current player data...")
        
        current_season = max(self.data_years)
        current_season_data = self.pbp_regular[self.pbp_regular['season'] == current_season]
        
        if len(current_season_data) == 0:
            print(f"⚠️ No data found for {current_season}, using {min(self.data_years)}")
            current_season = min(self.data_years)
        
        current_week = max(self.pbp_regular[self.pbp_regular['season'] == current_season]['week'])
        min_week = max(1, current_week - 6)
        
        print(f"   Using {current_season} season data, weeks {min_week}-{current_week}")
        
        # QB data
        qb_recent = self.passing_features[
            (self.passing_features['season'] == current_season) &
            (self.passing_features['week'] >= min_week) &
            (self.passing_features['passing_yards_L4'].notna())
        ]
        
        if len(qb_recent) > 0:
            qb_game_counts = qb_recent.groupby('player_name')['game_id'].nunique()
            valid_qbs = qb_game_counts[qb_game_counts >= 3].index
            qb_recent = qb_recent[qb_recent['player_name'].isin(valid_qbs)]
            
            qb_latest = qb_recent.sort_values(['player_name', 'game_date']).groupby('player_name').last()
            
            for player, stats in qb_latest.iterrows():
                self.player_data['qb'][player] = {
                    'passing_yards_L4': float(stats['passing_yards_L4']),
                    'passing_yards_L8': float(stats['passing_yards_L8']),
                    'passing_yards_L16': float(stats.get('passing_yards_L16', stats['passing_yards_L8'])),
                    'completion_pct_L4': float(stats['completion_pct_L4']),
                    'completion_pct_L8': float(stats.get('completion_pct_L8', stats['completion_pct_L4'])),
                    'attempts_L4': float(stats['attempts_L4']),
                    'attempts_L8': float(stats.get('attempts_L8', stats['attempts_L4'])),
                    'passing_tds_L4': float(stats['passing_tds_L4']),
                    'passing_tds_L8': float(stats.get('passing_tds_L8', stats['passing_tds_L4'])),
                    'interception_L4': float(stats.get('interception_L4', 0.5)),
                    'team': stats['posteam'],
                    'last_opponent': stats['defteam']
                }
        
        # WR data
        wr_recent = self.receiving_features[
            (self.receiving_features['season'] == current_season) &
            (self.receiving_features['week'] >= min_week) &
            (self.receiving_features['receiving_yards_L4'].notna())
        ]
        
        if len(wr_recent) > 0:
            wr_game_counts = wr_recent.groupby('player_name')['game_id'].nunique()
            valid_wrs = wr_game_counts[wr_game_counts >= 3].index
            wr_recent = wr_recent[wr_recent['player_name'].isin(valid_wrs)]
            
            wr_latest = wr_recent.sort_values(['player_name', 'game_date']).groupby('player_name').last()
            
            for player, stats in wr_latest.iterrows():
                self.player_data['wr'][player] = {
                    'receiving_yards_L4': float(stats['receiving_yards_L4']),
                    'receiving_yards_L8': float(stats['receiving_yards_L8']),
                    'receiving_yards_L16': float(stats.get('receiving_yards_L16', stats['receiving_yards_L8'])),
                    'receptions_L4': float(stats['receptions_L4']),
                    'receptions_L8': float(stats.get('receptions_L8', stats['receptions_L4'])),
                    'receptions_L16': float(stats.get('receptions_L16', stats['receptions_L8'])),
                    'yards_per_rec_L4': float(stats['yards_per_rec_L4']),
                    'yards_per_rec_L8': float(stats.get('yards_per_rec_L8', stats['yards_per_rec_L4'])),
                    'receiving_tds_L4': float(stats['receiving_tds_L4']),
                    'receiving_tds_L8': float(stats.get('receiving_tds_L8', stats['receiving_tds_L4'])),
                    'target_share_L4': float(stats.get('target_share_L4', 0.15)),
                    'target_share_L8': float(stats.get('target_share_L8', 0.15)),
                    'team': stats['posteam'],
                    'last_opponent': stats['defteam']
                }
        
        # RB data
        rb_recent = self.rushing_features[
            (self.rushing_features['season'] == current_season) &
            (self.rushing_features['week'] >= min_week) &
            (self.rushing_features['rushing_yards_L4'].notna())
        ]
        
        if len(rb_recent) > 0:
            rb_game_counts = rb_recent.groupby('player_name')['game_id'].nunique()
            valid_rbs = rb_game_counts[rb_game_counts >= 3].index
            rb_recent = rb_recent[rb_recent['player_name'].isin(valid_rbs)]
            
            rb_latest = rb_recent.sort_values(['player_name', 'game_date']).groupby('player_name').last()
            
            for player, stats in rb_latest.iterrows():
                self.player_data['rb'][player] = {
                    'rushing_yards_L4': float(stats['rushing_yards_L4']),
                    'rushing_yards_L8': float(stats['rushing_yards_L8']),
                    'rushing_yards_L16': float(stats.get('rushing_yards_L16', stats['rushing_yards_L8'])),
                    'attempts_L4': float(stats['attempts_L4']),
                    'attempts_L8': float(stats.get('attempts_L8', stats['attempts_L4'])),
                    'yards_per_carry_L4': float(stats['yards_per_carry_L4']),
                    'yards_per_carry_L8': float(stats.get('yards_per_carry_L8', stats['yards_per_carry_L4'])),
                    'rushing_tds_L4': float(stats['rushing_tds_L4']),
                    'rushing_tds_L8': float(stats.get('rushing_tds_L8', stats['rushing_tds_L4'])),
                    'team': stats['posteam'],
                    'last_opponent': stats['defteam']
                }
        
        # Defense rankings
        current_defense = self.defense_rankings_data[
            self.defense_rankings_data['season'] == current_season
        ]
        
        for _, row in current_defense.iterrows():
            self.defense_rankings[row['defteam']] = {
                'pass_def_rank': int(row['pass_def_rank']),
                'rush_def_rank': int(row['rush_def_rank'])
            }
        
        print(f"✅ Extracted: {len(self.player_data['qb'])} QBs, {len(self.player_data['wr'])} WRs, {len(self.player_data['rb'])} RBs")
    
    def save_all_data(self):
        """Save ensemble models and data files INCLUDING EPA & OL/DL"""
        print(f"\n💾 Saving all data and models WITH EPA & OL/DL...")
        
        import joblib
        
        # Save ensemble models
        model_count = 0
        for model_name, models in self.ensemble_models.items():
            if models is not None:
                for idx, model in enumerate(models):
                    joblib.dump(model, f'models/{model_name}_model_{idx}.pkl')
                model_count += 1
        
        # Save calibration parameters
        with open('calibration_params.json', 'w') as f:
            json.dump(self.calibration_params, f, indent=2)
        print(f"✅ Saved calibration parameters")
        
        # Save player data
        with open('qb_data.json', 'w') as f:
            json.dump(self.player_data['qb'], f, indent=2)
        
        with open('wr_data.json', 'w') as f:
            json.dump(self.player_data['wr'], f, indent=2)
        
        with open('rb_data.json', 'w') as f:
            json.dump(self.player_data['rb'], f, indent=2)
        
        # Save team data
        with open('team_data.json', 'w') as f:
            # Convert numpy types to native Python types
            team_data_serializable = {}
            for team, stats in self.team_data.items():
                team_data_serializable[team] = {
                    k: float(v) if hasattr(v, 'item') else v 
                    for k, v in stats.items()
                }
            json.dump(team_data_serializable, f, indent=2)
        
        # Save defense rankings
        with open('defense_rankings.json', 'w') as f:
            json.dump(self.defense_rankings, f, indent=2)
        
        # NEW: Save EPA metrics
        with open('epa_metrics.json', 'w') as f:
            json.dump(self.epa_metrics, f, indent=2)
        print(f"✅ Saved EPA metrics")
        
        # NEW: Save OL/DL rankings
        with open('ol_dl_rankings.json', 'w') as f:
            json.dump(self.ol_dl_rankings, f, indent=2)
        print(f"✅ Saved OL/DL rankings")
        
        print(f"✅ Saved {model_count} ensemble models (3 sub-models each) and all data files")
        
        # Create update log
        update_info = {
            'last_updated': datetime.now().isoformat(),
            'version': '2.2 - WITH EPA & OL/DL',
            'data_years': self.data_years,
            'ensemble_models_trained': list(self.ensemble_models.keys()),
            'calibration_applied': True,
            'spread_calibration_factor': self.calibration_params.get('spread_factor', 'N/A'),
            'weekly_report_generated': True,
            'epa_metrics_calculated': True,
            'ol_dl_rankings_calculated': True,
            'players': {
                'qb_count': len(self.player_data['qb']),
                'wr_count': len(self.player_data['wr']),
                'rb_count': len(self.player_data['rb'])
            }
        }
        
        with open('update_log.json', 'w') as f:
            json.dump(update_info, f, indent=2)
        
        print(f"✅ Update log created")
    
    def run_full_update(self):
        """Run the complete weekly update process WITH EPA & OL/DL"""
        print("=" * 60)
        print("NFL PREDICTION SYSTEM v2.2 - WEEKLY UPDATE")
        print("WITH EPA AND OL/DL INTEGRATION")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # STEP 0: Generate accuracy report BEFORE updating models
        print("\n🎯 STEP 0: Generate Weekly Accuracy Report (using OLD models)")
        print("="*60)
        try:
            self.generate_weekly_accuracy_report()
            print("✅ Weekly report generated successfully\n")
        except Exception as e:
            print(f"⚠️ Could not generate weekly report: {e}")
            print("   Continuing with model update...\n")
        
        print("=" * 60)
        print("NOW UPDATING MODELS WITH NEW DATA + EPA + OL/DL")
        print("=" * 60)
        
        steps = [
            ("Loading NFL Data", self.load_nfl_data),
            ("Calculating League Average", self.calculate_league_average),
            ("Calculating EPA Metrics", self.calculate_epa_metrics),
            ("Calculating OL/DL Strength", self.calculate_ol_dl_strength),
            ("Creating Player Logs", self.create_player_logs),
            ("Building Player Features", self.build_player_features),
            ("Creating Split Defense Rankings", self.create_defense_rankings),
            ("Training Ensemble Player Models", self.train_models),
            ("Training Game Models WITH EPA & OL/DL", self.train_game_models),
            ("Creating Team Data", self.create_team_data),
            ("Creating Weekly Schedule WITH EPA & OL/DL", self.create_weekly_schedule_predictions),
            ("Extracting Current Players", self.extract_current_players),
            ("Saving All Data", self.save_all_data)
        ]
        
        for step_name, step_func in steps:
            try:
                step_func()
            except Exception as e:
                print(f"❌ Error in {step_name}: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        print("\n" + "=" * 60)
        print("✅ WEEKLY UPDATE COMPLETE!")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        print(f"\n📊 Update Summary:")
        print(f"Ensemble models: {len(self.ensemble_models)} (3 sub-models each)")
        print(f"Calibration factor: {self.calibration_params.get('spread_factor', 'N/A'):.3f}")
        print(f"QBs: {len(self.player_data['qb'])}")
        print(f"WRs: {len(self.player_data['wr'])}")
        print(f"RBs: {len(self.player_data['rb'])}")
        print(f"EPA metrics: {len(self.epa_metrics)} teams")
        print(f"OL/DL rankings: {len(self.ol_dl_rankings)} teams")
        print(f"\n🚀 Your NFL prediction system is now updated WITH EPA & OL/DL!")
        print("\n✨ This Week's Update:")
        print("  • Generated accuracy report using old models (true out-of-sample test)")
        print("  • Calculated EPA metrics for all teams")
        print("  • Calculated OL/DL matchup strength")
        print("  • Updated all models with latest game data + EPA + OL/DL features")
        print("  • Check Weekly Report tab in app for results")
        
        return True

def main():
    """Main function to run the weekly update"""
    if not install_packages():
        print("❌ Failed to install required packages")
        return
    
    updater = NFLWeeklyUpdater()
    success = updater.run_full_update()
    
    if success:
        print("\n🎯 Next steps:")
        print("1. Check weekly_report.json for accuracy results")
        print("2. Restart your Streamlit app to see updated predictions")
        print("3. View Weekly Report tab in app for detailed analysis")
        print("4. EPA and OL/DL metrics are now integrated into predictions!")
    else:
        print("\n❌ Update failed - check error messages above")

if __name__ == "__main__":
    main()