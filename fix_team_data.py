import json
import pandas as pd
import numpy as np

# Load schedule
import nfl_data_py as nfl
schedule = nfl.import_schedules([2023, 2024, 2025])
schedule = schedule[schedule['week'] <= 18]

# Calculate current stats for each team
team_stats = {}

for team in schedule['home_team'].unique():
    # Get all games for this team in 2025
    team_games_2025 = schedule[
        (schedule['season'] == 2025) &
        ((schedule['home_team'] == team) | (schedule['away_team'] == team)) &
        (schedule['home_score'].notna())  # Only completed games
    ]
    
    if len(team_games_2025) == 0:
        # No 2025 games yet, use 2024
        team_games_2025 = schedule[
            (schedule['season'] == 2024) &
            ((schedule['home_team'] == team) | (schedule['away_team'] == team))
        ]
    
    points_for = []
    points_against = []
    
    for _, game in team_games_2025.iterrows():
        if game['home_team'] == team:
            points_for.append(game['home_score'])
            points_against.append(game['away_score'])
        else:
            points_for.append(game['away_score'])
            points_against.append(game['home_score'])
    
    # Calculate stats
    team_stats[team] = {
        'points_L4': float(np.mean(points_for[-4:])) if len(points_for) >= 4 else float(np.mean(points_for)) if points_for else 22.0,
        'points_L8': float(np.mean(points_for[-8:])) if len(points_for) >= 8 else float(np.mean(points_for)) if points_for else 22.0,
        'opp_points_L4': float(np.mean(points_against[-4:])) if len(points_against) >= 4 else float(np.mean(points_against)) if points_against else 22.0,
        'opp_points_L8': float(np.mean(points_against[-8:])) if len(points_against) >= 8 else float(np.mean(points_against)) if points_against else 22.0,
        'yards_L4': 350.0,
        'win_pct_L8': float(sum([1 if pf > pa else 0 for pf, pa in zip(points_for[-8:], points_against[-8:])])) / len(points_for[-8:]) if len(points_for) >= 1 else 0.5,
        'turnovers_L4': 1.0
    }

# Save
with open('team_data.json', 'w') as f:
    json.dump(team_stats, f, indent=2)

print("✅ Fixed team_data.json!")
print(f"IND: {team_stats['IND']}")