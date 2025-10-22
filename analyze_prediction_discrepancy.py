#!/usr/bin/env python3
"""
Analysis script to understand discrepancies between power rankings and model predictions.

This script helps identify why the ML model's predictions might differ significantly from
what you'd expect based on simple power ranking differences.
"""

import json
import sys


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_expected_spread(home_rating, away_rating, home_field_advantage=2.5):
    """
    Calculate expected spread from power ratings.

    Traditional approach: (home_rating - away_rating) * conversion_factor + home_field_advantage
    The conversion factor is typically around 0.35 to convert power rating points to spread points.
    """
    rating_diff = home_rating - away_rating
    # Use 0.35 as a typical conversion factor (can be adjusted)
    spread_from_ratings = rating_diff * 0.35 + home_field_advantage
    return spread_from_ratings


def analyze_matchup(home_team, away_team, season='2025'):
    """Analyze a specific matchup and show all contributing factors."""

    # Load all relevant data
    print("Loading data files...")
    power_ratings = load_json('power_ratings.json')
    team_data = load_json('team_data.json')
    epa_metrics = load_json('epa_metrics.json')
    ol_dl_rankings = load_json('ol_dl_rankings.json')
    weekly_schedule = load_json('weekly_schedule.json')

    try:
        calibration_params = load_json('calibration_params.json')
        spread_factor = calibration_params.get('spread_factor', 0.7)
    except FileNotFoundError:
        spread_factor = 0.7

    # Find the game in the schedule
    game = None
    for g in weekly_schedule:
        if g['home_team'] == home_team and g['away_team'] == away_team:
            game = g
            break

    if not game:
        print(f"ERROR: Could not find game {away_team} @ {home_team} in weekly schedule")
        return

    print("\n" + "="*80)
    print(f"PREDICTION DISCREPANCY ANALYSIS: {away_team} @ {home_team}")
    print("="*80)

    # 1. Model Prediction
    model_spread = game['spread']
    model_home_score = game['home_score']
    model_away_score = game['away_score']

    print(f"\n1. MODEL PREDICTION:")
    print(f"   Home ({home_team}): {model_home_score:.1f}")
    print(f"   Away ({away_team}): {model_away_score:.1f}")
    print(f"   Spread: {model_spread:.1f} (positive = home favored)")

    # 2. Power Rankings
    home_power = power_ratings[home_team]
    away_power = power_ratings[away_team]
    power_diff = home_power - away_power

    # Convert to "vs average" format (assuming 100 is average)
    home_vs_avg = home_power - 100
    away_vs_avg = away_power - 100

    print(f"\n2. POWER RANKINGS:")
    print(f"   {home_team}: {home_power:.1f} ({home_vs_avg:+.1f} vs average)")
    print(f"   {away_team}: {away_power:.1f} ({away_vs_avg:+.1f} vs average)")
    print(f"   Raw Difference: {power_diff:.1f} points")

    # 3. Expected Spread from Power Rankings
    expected_spread = calculate_expected_spread(home_power, away_power)
    expected_spread_neutral = power_diff * 0.35  # Neutral field

    print(f"\n3. EXPECTED SPREAD FROM POWER RANKINGS:")
    print(f"   On neutral field: {expected_spread_neutral:.1f}")
    print(f"   With home field (+2.5): {expected_spread:.1f}")
    print(f"   **DISCREPANCY: {abs(expected_spread - model_spread):.1f} points**")

    # 4. Recent Form (L4 games)
    home_data = team_data[home_team]
    away_data = team_data[away_team]

    print(f"\n4. RECENT FORM (Last 4 Games):")
    print(f"   {home_team}: {home_data['points_L4']:.1f} pts scored, {home_data['opp_points_L4']:.1f} allowed")
    print(f"   {away_team}: {away_data['points_L4']:.1f} pts scored, {away_data['opp_points_L4']:.1f} allowed")
    print(f"   {home_team} Win % (L8): {home_data['win_pct_L8']:.1%}")
    print(f"   {away_team} Win % (L8): {away_data['win_pct_L8']:.1%}")

    # 5. EPA Metrics
    home_epa = epa_metrics[home_team].get(season, {})
    away_epa = epa_metrics[away_team].get(season, {})

    home_epa_play = home_epa.get('epa_per_play', 0)
    away_epa_play = away_epa.get('epa_per_play', 0)
    home_def_epa = home_epa.get('def_epa', 0)
    away_def_epa = away_epa.get('def_epa', 0)

    print(f"\n5. EPA METRICS ({season}):")
    print(f"   {home_team} Offense: {home_epa_play:+.3f} EPA/play")
    print(f"   {away_team} Offense: {away_epa_play:+.3f} EPA/play")
    print(f"   Offensive EPA Difference: {home_epa_play - away_epa_play:+.3f}")
    print(f"   ")
    print(f"   {home_team} Defense: {home_def_epa:+.3f} EPA/play allowed (lower is better)")
    print(f"   {away_team} Defense: {away_def_epa:+.3f} EPA/play allowed")

    # EPA contribution to power ratings (from the codebase analysis)
    home_epa_contrib = home_epa_play * 8
    away_epa_contrib = away_epa_play * 8
    print(f"   EPA Contribution to Power Rating:")
    print(f"     {home_team}: {home_epa_contrib:+.1f}")
    print(f"     {away_team}: {away_epa_contrib:+.1f}")

    # 6. OL/DL Matchups
    home_ol = ol_dl_rankings[home_team][season]['ol']
    away_ol = ol_dl_rankings[away_team][season]['ol']
    home_dl = ol_dl_rankings[home_team][season]['dl']
    away_dl = ol_dl_rankings[away_team][season]['dl']

    # Matchup scores
    home_ol_vs_away_dl = home_ol['score'] - away_dl['score']
    away_ol_vs_home_dl = away_ol['score'] - home_dl['score']

    print(f"\n6. OFFENSIVE LINE / DEFENSIVE LINE MATCHUPS ({season}):")
    print(f"   {home_team} Offense vs {away_team} Defense:")
    print(f"     {home_team} OL: Rank #{int(home_ol['rank'])}, Score {home_ol['score']:.1f}")
    print(f"     {away_team} DL: Rank #{int(away_dl['rank'])}, Score {away_dl['score']:.1f}")
    print(f"     **Matchup: {home_ol_vs_away_dl:+.1f}** ({'Advantage' if home_ol_vs_away_dl > 5 else 'Even' if home_ol_vs_away_dl > -5 else 'Disadvantage'})")
    print(f"   ")
    print(f"   {away_team} Offense vs {home_team} Defense:")
    print(f"     {away_team} OL: Rank #{int(away_ol['rank'])}, Score {away_ol['score']:.1f}")
    print(f"     {home_team} DL: Rank #{int(home_dl['rank'])}, Score {home_dl['score']:.1f}")
    print(f"     **Matchup: {away_ol_vs_home_dl:+.1f}** ({'Advantage' if away_ol_vs_home_dl > 5 else 'Even' if away_ol_vs_home_dl > -5 else 'Disadvantage'})")

    # 7. Calibration Factor
    print(f"\n7. MODEL CALIBRATION:")
    print(f"   Spread Calibration Factor: {spread_factor:.2f}")
    print(f"   This factor is applied to raw model predictions to improve accuracy")
    print(f"   Raw spread (before calibration) would be ~{model_spread / spread_factor:.1f}")

    # 8. Summary and Possible Explanations
    print(f"\n8. POSSIBLE EXPLANATIONS FOR DISCREPANCY:")
    explanations = []

    # Check calibration factor impact
    uncalibrated_spread = model_spread / spread_factor
    if abs(uncalibrated_spread - expected_spread) < abs(model_spread - expected_spread):
        explanations.append(
            f"   • Calibration factor ({spread_factor}) dampens the spread significantly"
        )

    # Check if EPA strongly contradicts power rankings
    epa_diff = home_epa_play - away_epa_play
    if abs(epa_diff) > 0.2:
        if (epa_diff > 0 and model_spread < expected_spread) or (epa_diff < 0 and model_spread > expected_spread):
            explanations.append(
                f"   • EPA metrics show different efficiency than power rankings suggest"
            )
        else:
            explanations.append(
                f"   • EPA metrics support the power ranking difference ({epa_diff:+.3f} EPA/play)"
            )

    # Check OL/DL matchups
    if abs(home_ol_vs_away_dl) > 15 or abs(away_ol_vs_home_dl) > 15:
        explanations.append(
            f"   • Significant OL/DL matchup advantages detected"
        )

    # Check recent form vs power rankings
    home_recent_diff = home_data['points_L4'] - home_data['opp_points_L4']
    away_recent_diff = away_data['points_L4'] - away_data['opp_points_L4']
    recent_form_diff = home_recent_diff - away_recent_diff

    if abs(recent_form_diff) > 15:
        explanations.append(
            f"   • Recent form (L4) shows {recent_form_diff:+.1f} point differential"
        )

    # Machine learning non-linearity
    explanations.append(
        f"   • ML ensemble learns non-linear relationships from historical data"
    )
    explanations.append(
        f"   • The model uses 20 features and is trained on actual game outcomes"
    )

    for exp in explanations:
        print(exp)

    print(f"\n" + "="*80)
    print(f"CONCLUSION:")
    print(f"The {abs(expected_spread - model_spread):.1f}-point discrepancy is likely due to a combination of:")
    print(f"  1. Calibration factor applied to raw predictions")
    print(f"  2. EPA efficiency metrics that differ from power ranking expectations")
    print(f"  3. Machine learning models identifying patterns in historical data")
    print(f"  4. Specific matchup factors (OL/DL, recent form)")
    print("="*80)


def analyze_all_week():
    """Analyze all games in the current week and show discrepancies."""

    print("Loading data files...")
    power_ratings = load_json('power_ratings.json')
    weekly_schedule = load_json('weekly_schedule.json')

    print("\n" + "="*80)
    print("ALL GAMES - POWER RANKING vs MODEL PREDICTION ANALYSIS")
    print("="*80)

    print(f"\n{'Matchup':<20} {'Power Diff':<12} {'Expected':<10} {'Model':<10} {'Discrepancy':<12}")
    print("-" * 80)

    discrepancies = []

    for game in weekly_schedule:
        home = game['home_team']
        away = game['away_team']
        model_spread = game['spread']

        home_power = power_ratings[home]
        away_power = power_ratings[away]
        power_diff = home_power - away_power

        expected_spread = calculate_expected_spread(home_power, away_power)
        discrepancy = abs(expected_spread - model_spread)

        matchup = f"{away} @ {home}"
        print(f"{matchup:<20} {power_diff:>11.1f} {expected_spread:>9.1f} {model_spread:>9.1f} {discrepancy:>11.1f}")

        discrepancies.append({
            'matchup': matchup,
            'home': home,
            'away': away,
            'discrepancy': discrepancy
        })

    # Sort by discrepancy and show top 5
    discrepancies.sort(key=lambda x: x['discrepancy'], reverse=True)

    print(f"\n" + "="*80)
    print("TOP 5 GAMES WITH LARGEST DISCREPANCIES:")
    print("="*80)
    for i, d in enumerate(discrepancies[:5], 1):
        print(f"{i}. {d['matchup']}: {d['discrepancy']:.1f} points")
    print()


def main():
    """Main entry point."""

    if len(sys.argv) > 1:
        if sys.argv[1] == '--all':
            analyze_all_week()
        elif len(sys.argv) == 3:
            away_team = sys.argv[1]
            home_team = sys.argv[2]
            analyze_matchup(home_team, away_team)
        else:
            print("Usage:")
            print("  python analyze_prediction_discrepancy.py TEN IND")
            print("  python analyze_prediction_discrepancy.py --all")
            sys.exit(1)
    else:
        # Default: analyze the TEN @ IND game that prompted this analysis
        print("Running default analysis: TEN @ IND")
        print("(Use --all to see all games, or specify: <away> <home>)")
        print()
        analyze_matchup('IND', 'TEN')


if __name__ == '__main__':
    main()
