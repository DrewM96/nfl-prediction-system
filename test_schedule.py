# test_schedule.py
import nfl_data_py as nfl
import pandas as pd

print("Loading 2025 schedule...")
schedule = nfl.import_schedules([2025])
schedule = schedule[schedule['week'] <= 18]

print(f"\nTotal games in schedule: {len(schedule)}")

# Check which weeks have completed games
completed = schedule[schedule['home_score'].notna()]
print(f"\nCompleted games: {len(completed)}")

# Group by week to see what's completed
week_summary = completed.groupby('week').agg({
    'game_id': 'count',
    'home_score': 'count'
}).reset_index()
week_summary.columns = ['Week', 'Games', 'Completed']

print("\n" + "="*50)
print("COMPLETED WEEKS IN 2025 SEASON:")
print("="*50)
print(week_summary.to_string(index=False))

# Show the most recent completed week
if len(completed) > 0:
    latest_week = completed['week'].max()
    print(f"\n✅ Most recent completed week: {latest_week}")
    
    # Show games from that week
    latest_games = completed[completed['week'] == latest_week]
    print(f"\nGames in Week {latest_week}:")
    for _, game in latest_games.iterrows():
        print(f"  {game['away_team']} @ {game['home_team']}: {game['away_score']}-{game['home_score']}")
else:
    print("\n⚠️ No completed games found!")