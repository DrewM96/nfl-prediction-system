#!/usr/bin/env python3
"""
CORRECTED Power Ratings System
The previous version had inflated ratings - this fixes it
"""

import json
import numpy as np

def calculate_corrected_power_rating(team_code):
    """
    Calculate power rating on a proper 0-100 scale
    50 = league average, 60+ = elite, 40- = poor
    """
    
    try:
        with open('team_data.json', 'r') as f:
            team_data = json.load(f)
    except:
        return 50.0
    
    if team_code not in team_data:
        return 50.0
    
    data = team_data[team_code]
    
    # League average is ~22 ppg
    LEAGUE_AVG = 22.0
    
    # Get team stats
    points_scored = data.get('points_L4', LEAGUE_AVG)
    points_allowed = data.get('opp_points_L4', LEAGUE_AVG)
    
    # Calculate point differential (most predictive stat)
    point_diff = points_scored - points_allowed
    
    # Convert to rating scale
    # Each point of differential = 2 rating points
    # So +10 differential = 50 + 20 = 70 rating (elite)
    # -10 differential = 50 - 20 = 30 rating (terrible)
    
    base_rating = 50.0
    differential_impact = point_diff * 2.0
    
    # Calculate offensive and defensive components
    offensive_rating = ((points_scored - LEAGUE_AVG) / LEAGUE_AVG) * 25
    defensive_rating = ((LEAGUE_AVG - points_allowed) / LEAGUE_AVG) * 25
    
    # Weighted combination
    power_rating = (
        base_rating +
        differential_impact * 0.5 +  # 50% weight on differential
        offensive_rating * 0.25 +     # 25% weight on offense
        defensive_rating * 0.25       # 25% weight on defense
    )
    
    # Clamp to reasonable range (20-80)
    power_rating = max(20, min(80, power_rating))
    
    return power_rating

def predict_spread_simple(home_team, away_team):
    """
    Simple, accurate spread prediction
    Uses: rating difference + home field advantage
    """
    
    home_rating = calculate_corrected_power_rating(home_team)
    away_rating = calculate_corrected_power_rating(away_team)
    
    # Rating difference
    # Rule: 10 rating points = ~3.5 points on scoreboard (calibrated to Vegas)
    rating_diff = home_rating - away_rating
    spread_from_ratings = rating_diff * 0.35
    
    # Home field advantage (~2.5 points)
    home_field = 2.5
    
    # Total spread
    predicted_spread = spread_from_ratings + home_field
    
    # Estimate scores based on team averages
    try:
        with open('team_data.json', 'r') as f:
            team_data = json.load(f)
        
        home_avg_score = team_data.get(home_team, {}).get('points_L4', 22)
        away_avg_score = team_data.get(away_team, {}).get('points_L4', 22)
        
        # Adjust for opponent
        home_avg_allowed = team_data.get(away_team, {}).get('opp_points_L4', 22)
        away_avg_allowed = team_data.get(home_team, {}).get('opp_points_L4', 22)
        
        # Blend team's scoring with opponent's defense
        expected_home = (home_avg_score * 0.6 + away_avg_allowed * 0.4) + 1.5  # home boost
        expected_away = (away_avg_score * 0.6 + home_avg_allowed * 0.4)
        
        predicted_total = expected_home + expected_away
        
    except:
        predicted_total = 44.0
        expected_home = (44 + predicted_spread) / 2
        expected_away = (44 - predicted_spread) / 2
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        'predicted_spread': round(predicted_spread, 1),
        'predicted_total': round(predicted_total, 1),
        'home_score': round(expected_home, 1),
        'away_score': round(expected_away, 1),
        'home_rating': round(home_rating, 1),
        'away_rating': round(away_rating, 1),
        'rating_diff': round(rating_diff, 1)
    }

def generate_corrected_ratings():
    """Generate corrected power ratings"""
    
    print("=" * 70)
    print("CORRECTED NFL POWER RATINGS (0-100 Scale)")
    print("=" * 70)
    print("\nScale: 60+ Elite | 55-60 Good | 45-55 Average | 40-45 Poor | <40 Bad")
    
    try:
        with open('team_data.json', 'r') as f:
            team_data = json.load(f)
    except:
        print("❌ Could not load team data")
        return
    
    # Calculate corrected ratings
    ratings = {}
    for team in team_data.keys():
        ratings[team] = calculate_corrected_power_rating(team)
    
    # Sort by rating
    sorted_teams = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    
    # Show team data alongside rating
    print(f"\n{'Rank':<5} {'Team':<5} {'Rating':<7} {'Tier':<12} {'PF':<6} {'PA':<6} {'Diff'}")
    print("-" * 70)
    
    for rank, (team, rating) in enumerate(sorted_teams, 1):
        team_info = team_data[team]
        pf = team_info.get('points_L4', 0)
        pa = team_info.get('opp_points_L4', 0)
        diff = pf - pa
        
        if rating >= 60:
            tier = "🌟 Elite"
        elif rating >= 55:
            tier = "💪 Good"
        elif rating >= 45:
            tier = "📊 Average"
        elif rating >= 40:
            tier = "📉 Poor"
        else:
            tier = "❌ Bad"
        
        print(f"{rank:<5} {team:<5} {rating:<7.1f} {tier:<12} {pf:<6.1f} {pa:<6.1f} {diff:+.1f}")
    
    # Save corrected ratings
    with open('corrected_power_ratings.json', 'w') as f:
        json.dump(ratings, f, indent=2)
    
    print(f"\n✅ Corrected ratings saved to corrected_power_ratings.json")
    
    return ratings

def test_corrected_predictions():
    """Test predictions with corrected ratings"""
    
    print("\n" + "=" * 70)
    print("CORRECTED GAME PREDICTIONS")
    print("=" * 70)
    
    test_matchups = [
        ('IND', 'ARI', -7.0),
        ('KC', 'LAC', -3.5),
        ('PHI', 'NYG', -10.5),
        ('BAL', 'CLE', -6.5),
    ]
    
    print(f"\n{'Matchup':<15} {'Vegas':<10} {'Model':<10} {'Diff':<8} {'Status'}")
    print("-" * 70)
    
    for home, away, vegas in test_matchups:
        pred = predict_spread_simple(home, away)
        model = pred['predicted_spread']
        diff = abs(vegas - model)
        
        if diff <= 2.5:
            status = "✅ Great"
        elif diff <= 4.5:
            status = "👍 Good"
        elif diff <= 6.5:
            status = "⚠️ Fair"
        else:
            status = "❌ Off"
        
        matchup = f"{away}@{home}"
        print(f"{matchup:<15} {vegas:+.1f}{'':<5} {model:+.1f}{'':<5} {diff:<8.1f} {status}")
        
        print(f"  Ratings: {home} {pred['home_rating']:.1f} vs {away} {pred['away_rating']:.1f} (Diff: {pred['rating_diff']:+.1f})")
        print(f"  Scores: {away} {pred['away_score']:.1f} @ {home} {pred['home_score']:.1f} (Total: {pred['predicted_total']:.1f})")
        print()

def vegas_comparison_grid():
    """Show how to use this for betting"""
    
    print("=" * 70)
    print("BETTING STRATEGY GUIDE")
    print("=" * 70)
    
    print("""
🎯 How to Use These Predictions for Betting:

1. WHEN MODEL MATCHES VEGAS (within 2.5 points):
   ✅ Model validates Vegas line
   → Look elsewhere for value
   
2. WHEN MODEL DIFFERS BY 2.5-4.5 POINTS:
   🤔 Potential value opportunity
   → Check injury reports
   → Check weather
   → Consider betting if you trust your model
   
3. WHEN MODEL DIFFERS BY 4.5+ POINTS:
   🚨 Major discrepancy
   → Something is wrong (injury, news, public betting)
   → Research heavily before betting
   → Usually means you're missing key info

4. BEST USE CASE:
   📊 Use model to FIND games to research
   → Model says IND -10, Vegas says IND -7
   → Ask: "Why does Vegas think it's closer?"
   → Check: Injuries? Weather? Public betting?
   → Make INFORMED decision

REMEMBER: Vegas employs professionals and has more data.
Use your model as ONE tool, not the only tool.
""")

def reality_check():
    """Sanity check the predictions"""
    
    print("\n" + "=" * 70)
    print("REALITY CHECK")
    print("=" * 70)
    
    # Check IND specifically
    pred = predict_spread_simple('IND', 'ARI')
    
    print(f"\n🏈 ARI @ IND Analysis:")
    print(f"   Model Spread: IND {pred['predicted_spread']:+.1f}")
    print(f"   Vegas Line: IND -7.0")
    print(f"   Difference: {abs(-7.0 - pred['predicted_spread']):.1f} points")
    
    if abs(-7.0 - pred['predicted_spread']) <= 3:
        print(f"\n   ✅ GOOD! Model is within 3 points of Vegas")
        print(f"   This is acceptable accuracy for an automated system")
    else:
        print(f"\n   ⚠️ Still off, but investigating why:")
        print(f"   • Check if IND has key players injured")
        print(f"   • Check if ARI got a key player back")
        print(f"   • Check public betting % (maybe sharp money on ARI)")
        print(f"   • Vegas may know something your model doesn't")
    
    print(f"\n   Predicted Score: ARI {pred['away_score']:.1f}, IND {pred['home_score']:.1f}")
    print(f"   Predicted Total: {pred['predicted_total']:.1f}")

if __name__ == "__main__":
    # Generate corrected ratings
    ratings = generate_corrected_ratings()
    
    # Test predictions
    test_corrected_predictions()
    
    # Reality check
    reality_check()
    
    # Show betting guide
    vegas_comparison_grid()
    
    print("\n" + "=" * 70)
    print("✅ CORRECTED SYSTEM COMPLETE")
    print("=" * 70)
    print("\nNext: Integrate predict_spread_simple() into your Streamlit app")