#!/usr/bin/env python3
"""
NFL Model Diagnostic Script
Checks what the model actually sees vs reality
"""

import json
import pandas as pd
from datetime import datetime

def diagnose_model():
    """Run comprehensive diagnostics on the prediction system"""
    
    print("=" * 70)
    print("NFL PREDICTION MODEL DIAGNOSTICS")
    print("=" * 70)
    
    # 1. Check team data freshness
    print("\n1️⃣ CHECKING TEAM DATA FRESHNESS...")
    try:
        with open('team_data.json', 'r') as f:
            team_data = json.load(f)
        
        print(f"✅ Loaded data for {len(team_data)} teams")
        
        # Sample a few teams
        sample_teams = ['ARI', 'IND', 'KC', 'SF']
        for team in sample_teams:
            if team in team_data:
                data = team_data[team]
                print(f"\n{team}:")
                print(f"  Points Scored (L4): {data.get('points_L4', 'N/A'):.1f}")
                print(f"  Points Allowed (L4): {data.get('opp_points_L4', 'N/A'):.1f}")
                print(f"  Points Scored (L8): {data.get('points_L8', 'N/A'):.1f}")
                print(f"  Points Allowed (L8): {data.get('opp_points_L8', 'N/A'):.1f}")
                
                # Check if they look reasonable
                pts_scored = data.get('points_L4', 0)
                if pts_scored < 10 or pts_scored > 40:
                    print(f"  ⚠️ WARNING: Points scored seems unrealistic!")
            else:
                print(f"\n{team}: ❌ NOT FOUND IN DATA")
    
    except FileNotFoundError:
        print("❌ team_data.json not found!")
        return
    
    # 2. Check update log
    print("\n2️⃣ CHECKING LAST UPDATE...")
    try:
        with open('update_log.json', 'r') as f:
            log = json.load(f)
        
        last_update = log.get('last_updated', 'Unknown')
        data_years = log.get('data_years', [])
        
        print(f"Last Update: {last_update}")
        print(f"Data Years: {data_years}")
        
        # Parse the date
        if last_update != 'Unknown':
            update_date = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
            days_old = (datetime.now() - update_date.replace(tzinfo=None)).days
            
            if days_old > 7:
                print(f"⚠️ WARNING: Data is {days_old} days old! Run weekly update.")
            else:
                print(f"✅ Data is fresh ({days_old} days old)")
    
    except FileNotFoundError:
        print("❌ update_log.json not found!")
    
    # 3. Check weekly schedule predictions
    print("\n3️⃣ CHECKING WEEKLY PREDICTIONS...")
    try:
        with open('weekly_schedule.json', 'r') as f:
            games = json.load(f)
        
        print(f"✅ Found {len(games)} games in schedule")
        
        # Find the ARI @ IND game
        ari_ind_game = None
        for game in games:
            if (game.get('away_team') == 'ARI' and game.get('home_team') == 'IND') or \
               (game.get('home_team') == 'ARI' and game.get('away_team') == 'IND'):
                ari_ind_game = game
                break
        
        if ari_ind_game:
            print(f"\n🏈 Found ARI vs IND game:")
            print(f"   Away: {ari_ind_game['away_team']} - Predicted: {ari_ind_game['predicted_away_score']}")
            print(f"   Home: {ari_ind_game['home_team']} - Predicted: {ari_ind_game['predicted_home_score']}")
            print(f"   Predicted Spread: {ari_ind_game['predicted_spread']:.1f}")
            print(f"   Predicted Total: {ari_ind_game['predicted_total']:.1f}")
            
            # Compare to typical Vegas line
            print(f"\n   📊 REALITY CHECK:")
            print(f"   Vegas typically: IND -7")
            print(f"   Your model: {ari_ind_game['home_team']} {ari_ind_game['predicted_spread']:+.1f}")
            print(f"   Discrepancy: {abs(-7 - ari_ind_game['predicted_spread']):.1f} points")
            
            if abs(-7 - ari_ind_game['predicted_spread']) > 5:
                print(f"   ⚠️ LARGE DISCREPANCY! Model may be missing key info.")
        else:
            print("\n⚠️ ARI vs IND game not found in schedule")
    
    except FileNotFoundError:
        print("❌ weekly_schedule.json not found!")
    
    # 4. Check model files
    print("\n4️⃣ CHECKING MODEL FILES...")
    import os
    model_files = [
        'models/game_total_model_0.pkl',
        'models/game_total_model_1.pkl',
        'models/game_total_model_2.pkl',
        'models/game_spread_model_0.pkl',
        'models/game_spread_model_1.pkl',
        'models/game_spread_model_2.pkl',
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file)
            print(f"✅ {model_file} ({size:,} bytes)")
        else:
            print(f"❌ {model_file} MISSING")
    
    # 5. Check actual 2025 season records
    print("\n5️⃣ ANALYZING 2025 SEASON DATA...")
    try:
        import nfl_data_py as nfl
        
        # Get 2025 schedule
        schedule_2025 = nfl.import_schedules([2025])
        schedule_2025 = schedule_2025[schedule_2025['week'] <= 5]  # First 5 weeks
        
        # Calculate actual records
        records = {}
        for _, game in schedule_2025.iterrows():
            if game['home_score'] is not None:  # Game has been played
                # Home team
                if game['home_team'] not in records:
                    records[game['home_team']] = {'wins': 0, 'losses': 0, 'points_for': 0, 'points_against': 0}
                
                if game['home_score'] > game['away_score']:
                    records[game['home_team']]['wins'] += 1
                else:
                    records[game['home_team']]['losses'] += 1
                
                records[game['home_team']]['points_for'] += game['home_score']
                records[game['home_team']]['points_against'] += game['away_score']
                
                # Away team
                if game['away_team'] not in records:
                    records[game['away_team']] = {'wins': 0, 'losses': 0, 'points_for': 0, 'points_against': 0}
                
                if game['away_score'] > game['home_score']:
                    records[game['away_team']]['wins'] += 1
                else:
                    records[game['away_team']]['losses'] += 1
                
                records[game['away_team']]['points_for'] += game['away_score']
                records[game['away_team']]['points_against'] += game['home_score']
        
        if 'ARI' in records and 'IND' in records:
            print("\n📊 ACTUAL 2025 RECORDS:")
            for team in ['ARI', 'IND']:
                rec = records[team]
                games = rec['wins'] + rec['losses']
                if games > 0:
                    ppg = rec['points_for'] / games
                    papg = rec['points_against'] / games
                    print(f"\n{team}: {rec['wins']}-{rec['losses']}")
                    print(f"  Points/Game: {ppg:.1f}")
                    print(f"  Allowed/Game: {papg:.1f}")
                    print(f"  Point Diff: {ppg - papg:+.1f}")
            
            # Compare to model's view
            print("\n📊 MODEL'S VIEW (from team_data.json):")
            for team in ['ARI', 'IND']:
                if team in team_data:
                    data = team_data[team]
                    print(f"\n{team}:")
                    print(f"  Points/Game (L4): {data.get('points_L4', 0):.1f}")
                    print(f"  Allowed/Game (L4): {data.get('opp_points_L4', 0):.1f}")
                    
                    # Check if they match
                    actual_ppg = records[team]['points_for'] / (records[team]['wins'] + records[team]['losses'])
                    model_ppg = data.get('points_L4', 0)
                    
                    if abs(actual_ppg - model_ppg) > 3:
                        print(f"  ⚠️ MISMATCH: Actual={actual_ppg:.1f}, Model={model_ppg:.1f}")
        
    except ImportError:
        print("⚠️ nfl_data_py not available, skipping live data check")
    except Exception as e:
        print(f"⚠️ Could not check 2025 data: {e}")
    
    # 6. Recommendations
    print("\n" + "=" * 70)
    print("💡 RECOMMENDATIONS:")
    print("=" * 70)
    
    print("\n1. If data is stale (>7 days):")
    print("   → Run: python weekly_nfl_update.py")
    
    print("\n2. If model predictions way off from Vegas:")
    print("   → Model may be using old season data")
    print("   → Consider adding 'current season weight' feature")
    print("   → Or use Vegas lines as a calibration baseline")
    
    print("\n3. If specific teams seem wrong:")
    print("   → Check for major injuries or roster changes")
    print("   → Model doesn't know about injuries/trades/firings")
    
    print("\n4. Quick fix for betting:")
    print("   → Use model for PLAYER PROPS (more accurate)")
    print("   → Use model to find discrepancies vs Vegas")
    print("   → Don't blindly follow model on game outcomes yet")
    
    print("\n5. To improve game predictions:")
    print("   → Add current season win% as heavy-weighted feature")
    print("   → Add injury adjustment framework")
    print("   → Consider using Vegas lines as a feature")
    print("   → Add more recent data emphasis (last 2 weeks > last 8)")

if __name__ == "__main__":
    diagnose_model()