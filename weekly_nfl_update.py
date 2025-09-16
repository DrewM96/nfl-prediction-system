#!/usr/bin/env python3
"""
Weekly NFL Model Update Script
Run this every Monday to update your NFL prediction system with the latest data.

Requirements:
- pip install nfl-data-py pandas numpy scikit-learn joblib

Usage:
python weekly_nfl_update.py
"""

import sys
import subprocess
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Install required packages if not available
def install_packages():
    """Install required packages if they're not available"""
    try:
        import nfl_data_py
        from sklearn.linear_model import LinearRegression
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
    """Complete NFL prediction system updater"""
    
    def __init__(self):
        self.current_year = datetime.now().year
        # Load 3 seasons for cross-season rolling averages
        if datetime.now().month >= 9:  # September or later
            # Current season has started, use it + previous 2 years  
            self.data_years = [self.current_year - 2, self.current_year - 1, self.current_year]
        else:
            # Off-season, use previous 3 complete seasons
            self.data_years = [self.current_year - 3, self.current_year - 2, self.current_year - 1]
        self.models = {}
        self.team_data = {}
        self.player_data = {'qb': {}, 'wr': {}, 'rb': {}}
        self.defense_rankings = {}
        
        # Create necessary directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        
    def load_nfl_data(self):
        """Load the latest NFL data"""
        print(f"\n📊 Loading NFL data for {self.data_years}...")
        
        try:
            import nfl_data_py as nfl
            
            # Load play-by-play data
            self.pbp_data = nfl.import_pbp_data(self.data_years)
            print(f"✅ Loaded {len(self.pbp_data):,} plays")
            
            # Load schedule data
            self.schedule_data = nfl.import_schedules(self.data_years)
            print(f"✅ Loaded {len(self.schedule_data):,} games")
            
            # Filter to regular season only
            self.pbp_regular = self.pbp_data[self.pbp_data['week'] <= 18].copy()
            self.schedule_regular = self.schedule_data[self.schedule_data['week'] <= 18].copy()
            
            print(f"✅ Regular season: {len(self.pbp_regular):,} plays, {len(self.schedule_regular):,} games")
            return True
            
        except Exception as e:
            print(f"❌ Error loading NFL data: {e}")
            return False
    
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
        """Create cross-season rolling averages with recency weighting"""
        df = df.sort_values(['player_name', 'season', 'game_date']).copy()
        
        for window in window_sizes:
            for stat in stat_cols:
                col_name = f'{stat}_L{window}'
                
                # Create weighted cross-season rolling averages
                def weighted_rolling_mean(series):
                    result = []
                    for i in range(len(series)):
                        if i == 0:
                            result.append(np.nan)  # No previous data
                        else:
                            # Get the last 'window' games before current game
                            start_idx = max(0, i - window)
                            values = series.iloc[start_idx:i]
                            
                            if len(values) == 0:
                                result.append(np.nan)
                            else:
                                # Apply recency weights (more recent games weighted higher)
                                weights = np.linspace(0.6, 1.0, len(values))  # 0.6 to 1.0 weight range
                                weighted_mean = np.average(values, weights=weights)
                                result.append(weighted_mean)
                    
                    return pd.Series(result, index=series.index)
                
                df[col_name] = df.groupby('player_name')[stat].transform(weighted_rolling_mean)
        
        return df
    
    def build_player_features(self):
        """Build features for all player types"""
        print(f"\n📈 Building player features...")
        
        # Passing features
        self.passing_features = self.create_rolling_features(
            self.passing_logs,
            ['passing_yards', 'completions', 'attempts', 'passing_tds']
        )
        
        # Add completion percentage
        self.passing_features['completion_pct'] = self.passing_features['completions'] / self.passing_features['attempts']
        self.passing_features['completion_pct_L4'] = self.passing_features['completions_L4'] / self.passing_features['attempts_L4']
        self.passing_features['completion_pct_L8'] = self.passing_features['completions_L8'] / self.passing_features['attempts_L8']
        
        # Receiving features
        self.receiving_features = self.create_rolling_features(
            self.receiving_logs,
            ['receiving_yards', 'receptions', 'receiving_tds']
        )
        
        # Add yards per reception
        self.receiving_features['yards_per_rec'] = self.receiving_features['receiving_yards'] / self.receiving_features['receptions']
        self.receiving_features['yards_per_rec_L4'] = self.receiving_features['receiving_yards_L4'] / self.receiving_features['receptions_L4']
        self.receiving_features['yards_per_rec_L8'] = self.receiving_features['receiving_yards_L8'] / self.receiving_features['receptions_L8']
        
        # Rushing features
        self.rushing_features = self.create_rolling_features(
            self.rushing_logs,
            ['rushing_yards', 'attempts', 'rushing_tds']
        )
        
        # Add yards per carry
        self.rushing_features['yards_per_carry'] = self.rushing_features['rushing_yards'] / self.rushing_features['attempts']
        self.rushing_features['yards_per_carry_L4'] = self.rushing_features['rushing_yards_L4'] / self.rushing_features['attempts_L4']
        self.rushing_features['yards_per_carry_L8'] = self.rushing_features['rushing_yards_L8'] / self.rushing_features['attempts_L8']
        
        print("✅ Player features created")
    
    def create_defense_rankings(self):
        """Create defensive rankings by season"""
        print(f"\n🛡️ Creating defense rankings...")
        
        defense_stats = self.pbp_regular.groupby(['season', 'defteam', 'week']).agg({
            'yards_gained': 'sum'
        }).reset_index()
        
        # Calculate season averages
        defense_season = defense_stats.groupby(['season', 'defteam']).agg({
            'yards_gained': 'mean'
        }).reset_index()
        
        # Rank defenses within each season
        defense_season['defense_rank'] = defense_season.groupby('season')['yards_gained'].rank(ascending=True)
        
        self.defense_rankings_data = defense_season.rename(columns={
            'yards_gained': 'avg_yards_allowed'
        })
        
        # Add to player features - FIXED VERSION
        self.passing_features = self.passing_features.merge(
            self.defense_rankings_data[['season', 'defteam', 'defense_rank']],
            on=['season', 'defteam'],
            how='left'
        )
        
        self.receiving_features = self.receiving_features.merge(
            self.defense_rankings_data[['season', 'defteam', 'defense_rank']],
            on=['season', 'defteam'],
            how='left'
        )
        
        self.rushing_features = self.rushing_features.merge(
            self.defense_rankings_data[['season', 'defteam', 'defense_rank']],
            on=['season', 'defteam'],
            how='left'
        )
        
        # Fill any missing defense ranks with average (16)
        for df in [self.passing_features, self.receiving_features, self.rushing_features]:
            df['defense_rank'] = df['defense_rank'].fillna(16)
        
        print("✅ Defense rankings created and merged")
    
    def train_models(self):
        """Train all prediction models"""
        print(f"\n🎯 Training prediction models...")
        
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error
        
        def build_model(data, feature_cols, target_col, model_name):
            # Clean data
            clean_data = data[feature_cols + [target_col, 'season', 'week', 'game_date']].dropna()
            
            # Filter to games with sufficient history (week 3+ to allow cross-season rolling)
            clean_data = clean_data[clean_data['week'] >= 3]
            
            if len(clean_data) == 0:
                print(f"⚠️ {model_name}: No data after filtering")
                return None
            
            # Sort chronologically across all seasons
            clean_data = clean_data.sort_values(['season', 'week', 'game_date'])
            
            # Smart training/testing split
            seasons_available = sorted(clean_data['season'].unique())
            
            if len(seasons_available) >= 2:
                # Multi-season approach - use most recent complete season for testing
                current_season = max(seasons_available)
                current_season_data = clean_data[clean_data['season'] == current_season]
                current_week_count = current_season_data['week'].nunique()
                
                if current_week_count >= 12:  # Most of season complete
                    # Use all previous seasons + early current season for training
                    # Late current season for testing
                    train_mask = ((clean_data['season'] < current_season) | 
                                 ((clean_data['season'] == current_season) & 
                                  (clean_data['week'] <= 12)))
                    test_mask = ((clean_data['season'] == current_season) & 
                                (clean_data['week'] > 12))
                else:
                    # Current season not complete enough - use previous season for testing
                    prev_season = seasons_available[-2]
                    train_mask = clean_data['season'] < prev_season
                    test_mask = clean_data['season'] == prev_season
            else:
                # Single season - chronological split
                split_idx = int(len(clean_data) * 0.8)
                train_mask = clean_data.index.isin(clean_data.index[:split_idx])
                test_mask = clean_data.index.isin(clean_data.index[split_idx:])
            
            train_count = sum(train_mask)
            test_count = sum(test_mask)
            
            if train_count == 0:
                print(f"⚠️ {model_name}: No training data")
                return None
            
            X_train = clean_data[train_mask][feature_cols]
            y_train = clean_data[train_mask][target_col]
            
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Evaluate if test data exists
            if test_count > 0:
                X_test = clean_data[test_mask][feature_cols]
                y_test = clean_data[test_mask][target_col]
                pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, pred)
                
                # Show training periods for context
                train_seasons = sorted(clean_data[train_mask]['season'].unique())
                test_seasons = sorted(clean_data[test_mask]['season'].unique())
                print(f"✅ {model_name}: MAE {mae:.1f}")
                print(f"   Trained on: {train_seasons} ({train_count} games)")
                print(f"   Tested on: {test_seasons} ({test_count} games)")
            else:
                train_seasons = sorted(clean_data[train_mask]['season'].unique())
                print(f"✅ {model_name}: Trained on {train_seasons} ({train_count} games)")
            
            return model
        
        # Define features for each model
        passing_features = [
            'passing_yards_L4', 'passing_yards_L8', 'passing_yards_L16',
            'completion_pct_L4', 'completion_pct_L8',
            'attempts_L4', 'attempts_L8',
            'passing_tds_L4', 'passing_tds_L8',
            'defense_rank'
        ]
        
        receiving_features = [
            'receiving_yards_L4', 'receiving_yards_L8', 'receiving_yards_L16',
            'receptions_L4', 'receptions_L8',
            'yards_per_rec_L4', 'yards_per_rec_L8',
            'receiving_tds_L4', 'receiving_tds_L8',
            'defense_rank'
        ]
        
        rushing_features = [
            'rushing_yards_L4', 'rushing_yards_L8', 'rushing_yards_L16',
            'attempts_L4', 'attempts_L8',
            'yards_per_carry_L4', 'yards_per_carry_L8',
            'rushing_tds_L4', 'rushing_tds_L8',
            'defense_rank'
        ]
        
        # Train all models
        self.models['passing'] = build_model(
            self.passing_features, passing_features, 'passing_yards', 'Passing Yards'
        )
        
        self.models['receiving'] = build_model(
            self.receiving_features, receiving_features, 'receiving_yards', 'Receiving Yards'
        )
        
        self.models['receptions'] = build_model(
            self.receiving_features,
            ['receptions_L4', 'receptions_L8', 'receptions_L16', 'receiving_yards_L4', 'yards_per_rec_L4', 'defense_rank'],
            'receptions', 'Receptions'
        )
        
        self.models['rushing'] = build_model(
            self.rushing_features, rushing_features, 'rushing_yards', 'Rushing Yards'
        )
        
        print(f"✅ Trained {len([m for m in self.models.values() if m is not None])} models")
    
    def create_team_data(self):
        """Create team performance data for game predictions"""
        print(f"\n🏟️ Creating team performance data...")
        
        # Team offensive stats per game
        offense_stats = self.pbp_regular.groupby(['season', 'game_id', 'posteam']).agg({
            'yards_gained': 'sum',
            'touchdown': 'sum'
        }).reset_index()
        
        # Get actual points from schedule
        game_points = []
        for _, game in self.schedule_regular.iterrows():
            # Home team
            game_points.append({
                'season': game['season'], 'game_id': game['game_id'],
                'team': game['home_team'], 'points': game['home_score'],
                'opp_points': game['away_score']
            })
            # Away team
            game_points.append({
                'season': game['season'], 'game_id': game['game_id'],
                'team': game['away_team'], 'points': game['away_score'],
                'opp_points': game['home_score']
            })
        
        points_df = pd.DataFrame(game_points)
        
        # Merge with offensive stats
        team_games = offense_stats.merge(
            points_df,
            left_on=['season', 'game_id', 'posteam'],
            right_on=['season', 'game_id', 'team'],
            how='inner'
        )
        
        # Add game date
        team_games = team_games.merge(
            self.schedule_regular[['season', 'game_id', 'gameday']],
            on=['season', 'game_id']
        )
        
        # Create rolling team stats
        team_games = team_games.sort_values(['team', 'season', 'gameday'])
        
        for window in [4, 8]:
            for stat in ['points', 'opp_points', 'yards_gained']:
                col_name = f'{stat}_L{window}'
                team_games[col_name] = team_games.groupby('team')[stat].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )
        
        # Get current season stats for each team
        current_season_for_teams = max(self.data_years)
        current_teams = team_games[team_games['season'] == current_season_for_teams]
        
        if len(current_teams) == 0:
            # Fallback to previous season if current has no data
            current_season_for_teams = min(self.data_years)
            current_teams = team_games[team_games['season'] == current_season_for_teams]
        
        latest_team_stats = current_teams.sort_values(['team', 'gameday']).groupby('team').last()
        
        # Create team data dictionary
        for team_name, stats in latest_team_stats.iterrows():
            self.team_data[team_name] = {
                'points_L4': float(stats['points_L4']),
                'points_L8': float(stats['points_L8']),
                'opp_points_L4': float(stats['opp_points_L4']),
                'opp_points_L8': float(stats['opp_points_L8']),
                'yards_L4': float(stats['yards_gained_L4'])
            }
        
        print(f"✅ Created team data for {len(self.team_data)} teams")
    
    def extract_current_players(self):
        """Extract current player data for predictions"""
        print(f"\n👥 Extracting current player data...")
        
        # Use the most recent season with data
        current_season = max(self.data_years)
        
        # Check if we have recent data from the current season
        current_season_data = self.pbp_regular[self.pbp_regular['season'] == current_season]
        if len(current_season_data) == 0:
            print(f"⚠️ No data found for {current_season}, using {min(self.data_years)}")
            current_season = min(self.data_years)
        
        current_week = max(self.pbp_regular[self.pbp_regular['season'] == current_season]['week'])
        min_week = max(1, current_week - 6)  # Last 6 weeks
        
        print(f"   Using {current_season} season data, weeks {min_week}-{current_week}")
        
        # QB data
        qb_recent = self.passing_features[
            (self.passing_features['season'] == current_season) &
            (self.passing_features['week'] >= min_week) &
            (self.passing_features['passing_yards_L4'].notna())
        ]
        
        if len(qb_recent) > 0:
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
                    'team': stats['posteam'],
                    'last_opponent': stats['defteam']  # Added missing field
                }
        
        # WR data
        wr_recent = self.receiving_features[
            (self.receiving_features['season'] == current_season) &
            (self.receiving_features['week'] >= min_week) &
            (self.receiving_features['receiving_yards_L4'].notna())
        ]
        
        if len(wr_recent) > 0:
            wr_latest = wr_recent.sort_values(['player_name', 'game_date']).groupby('player_name').last()
            
            for player, stats in wr_latest.iterrows():
                self.player_data['wr'][player] = {
                    'receiving_yards_L4': float(stats['receiving_yards_L4']),
                    'receiving_yards_L8': float(stats['receiving_yards_L8']),
                    'receiving_yards_L16': float(stats.get('receiving_yards_L16', stats['receiving_yards_L8'])),
                    'receptions_L4': float(stats['receptions_L4']),
                    'receptions_L8': float(stats.get('receptions_L8', stats['receptions_L4'])),
                    'yards_per_rec_L4': float(stats['yards_per_rec_L4']),
                    'yards_per_rec_L8': float(stats.get('yards_per_rec_L8', stats['yards_per_rec_L4'])),
                    'receiving_tds_L4': float(stats['receiving_tds_L4']),
                    'receiving_tds_L8': float(stats.get('receiving_tds_L8', stats['receiving_tds_L4'])),
                    'team': stats['posteam'],
                    'last_opponent': stats['defteam']  # Added missing field
                }
        
        # RB data
        rb_recent = self.rushing_features[
            (self.rushing_features['season'] == current_season) &
            (self.rushing_features['week'] >= min_week) &
            (self.rushing_features['rushing_yards_L4'].notna())
        ]
        
        if len(rb_recent) > 0:
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
                    'last_opponent': stats['defteam']  # Added missing field
                }
        
        # Defense rankings for current season
        current_defense = self.defense_rankings_data[
            self.defense_rankings_data['season'] == current_season
        ]
        
        for _, row in current_defense.iterrows():
            self.defense_rankings[row['defteam']] = {
                'rank': int(row['defense_rank']),
                'yards_allowed': float(row['avg_yards_allowed'])
            }
        
        print(f"✅ Extracted: {len(self.player_data['qb'])} QBs, {len(self.player_data['wr'])} WRs, {len(self.player_data['rb'])} RBs")
    
    def save_all_data(self):
        """Save models and data files"""
        print(f"\n💾 Saving all data and models...")
        
        import joblib
        
        # Save models
        model_count = 0
        for model_name, model in self.models.items():
            if model is not None:
                joblib.dump(model, f'models/{model_name}_model.pkl')
                model_count += 1
        
        # Save player data
        with open('qb_data.json', 'w') as f:
            json.dump(self.player_data['qb'], f, indent=2)
        
        with open('wr_data.json', 'w') as f:
            json.dump(self.player_data['wr'], f, indent=2)
        
        with open('rb_data.json', 'w') as f:
            json.dump(self.player_data['rb'], f, indent=2)
        
        # Save team data
        with open('team_data.json', 'w') as f:
            json.dump(self.team_data, f, indent=2)
        
        # Save defense rankings
        with open('defense_rankings.json', 'w') as f:
            json.dump(self.defense_rankings, f, indent=2)
        
        print(f"✅ Saved {model_count} models and all data files")
        
        # Create update log
        update_info = {
            'last_updated': datetime.now().isoformat(),
            'data_years': self.data_years,
            'models_trained': list(self.models.keys()),
            'players': {
                'qb_count': len(self.player_data['qb']),
                'wr_count': len(self.player_data['wr']),
                'rb_count': len(self.player_data['rb'])
            },
            'teams': len(self.team_data)
        }
        
        with open('update_log.json', 'w') as f:
            json.dump(update_info, f, indent=2)
        
        print(f"✅ Update log created")
    
    def run_full_update(self):
        """Run the complete weekly update process"""
        print("=" * 60)
        print("NFL PREDICTION SYSTEM - WEEKLY UPDATE")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        steps = [
            ("Loading NFL Data", self.load_nfl_data),
            ("Creating Player Logs", self.create_player_logs),
            ("Building Player Features", self.build_player_features),
            ("Creating Defense Rankings", self.create_defense_rankings),
            ("Training Models", self.train_models),
            ("Creating Team Data", self.create_team_data),
            ("Extracting Current Players", self.extract_current_players),
            ("Saving All Data", self.save_all_data)
        ]
        
        for step_name, step_func in steps:
            try:
                step_func()
            except Exception as e:
                print(f"❌ Error in {step_name}: {e}")
                return False
        
        print("\n" + "=" * 60)
        print("✅ WEEKLY UPDATE COMPLETE!")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        print(f"\n📊 Update Summary:")
        print(f"Models trained: {len([m for m in self.models.values() if m is not None])}")
        print(f"Teams: {len(self.team_data)}")
        print(f"QBs: {len(self.player_data['qb'])}")
        print(f"WRs: {len(self.player_data['wr'])}")
        print(f"RBs: {len(self.player_data['rb'])}")
        print(f"\n🚀 Your NFL prediction system is now updated!")
        print("Restart your Streamlit app to use the latest data.")
        
        return True

def main():
    """Main function to run the weekly update"""
    # Check and install packages
    if not install_packages():
        print("❌ Failed to install required packages")
        return
    
    # Run the update
    updater = NFLWeeklyUpdater()
    success = updater.run_full_update()
    
    if success:
        print("\n🎯 Next steps:")
        print("1. Restart your Streamlit app")
        print("2. Test predictions with updated data")
        print("3. Check update_log.json for details")
    else:
        print("\n❌ Update failed - check error messages above")

if __name__ == "__main__":
    main()
