"""
Simple Injury Adjustment System
Add this to your Streamlit app for manual injury tracking with auto-adjustments

How it works:
1. You manually note key injuries in injuries.json
2. System automatically adjusts predictions based on position/severity
3. Updates are reflected immediately in predictions
"""

import json
import streamlit as st
from datetime import datetime, timedelta

class InjuryAdjustmentSystem:
    """Simple injury tracking and adjustment system"""
    
    def __init__(self, injury_file='injuries.json'):
        self.injury_file = injury_file
        self.injuries = self._load_injuries()
        
        # Injury impact multipliers by position and status
        self.impact_multipliers = {
            'QB': {
                'OUT': {'team_scoring': 0.75, 'team_spread': -6.0, 'player_passing': 0.0},
                'DOUBTFUL': {'team_scoring': 0.85, 'team_spread': -3.5, 'player_passing': 0.3},
                'QUESTIONABLE': {'team_scoring': 0.92, 'team_spread': -1.5, 'player_passing': 0.7},
                'PROBABLE': {'team_scoring': 0.97, 'team_spread': -0.5, 'player_passing': 0.9}
            },
            'RB1': {
                'OUT': {'team_scoring': 0.90, 'team_spread': -2.5, 'player_rushing': 0.0},
                'DOUBTFUL': {'team_scoring': 0.94, 'team_spread': -1.5, 'player_rushing': 0.3},
                'QUESTIONABLE': {'team_scoring': 0.97, 'team_spread': -0.5, 'player_rushing': 0.7},
                'PROBABLE': {'team_scoring': 0.99, 'team_spread': -0.2, 'player_rushing': 0.9}
            },
            'WR1': {
                'OUT': {'team_scoring': 0.93, 'team_spread': -2.0, 'player_receiving': 0.0},
                'DOUBTFUL': {'team_scoring': 0.96, 'team_spread': -1.0, 'player_receiving': 0.3},
                'QUESTIONABLE': {'team_scoring': 0.98, 'team_spread': -0.5, 'player_receiving': 0.7},
                'PROBABLE': {'team_scoring': 0.99, 'team_spread': -0.2, 'player_receiving': 0.9}
            },
            'TE1': {
                'OUT': {'team_scoring': 0.95, 'team_spread': -1.5, 'player_receiving': 0.0},
                'DOUBTFUL': {'team_scoring': 0.97, 'team_spread': -0.8, 'player_receiving': 0.3},
                'QUESTIONABLE': {'team_scoring': 0.98, 'team_spread': -0.3, 'player_receiving': 0.7},
                'PROBABLE': {'team_scoring': 0.99, 'team_spread': -0.1, 'player_receiving': 0.9}
            },
            'OL_MULTIPLE': {
                'OUT': {'team_scoring': 0.92, 'team_spread': -2.0},
                'DOUBTFUL': {'team_scoring': 0.95, 'team_spread': -1.0},
                'QUESTIONABLE': {'team_scoring': 0.97, 'team_spread': -0.5},
            },
            'DEF_STAR': {
                'OUT': {'opp_scoring': 1.08, 'team_spread': -1.5},  # Opponent scores more
                'DOUBTFUL': {'opp_scoring': 1.04, 'team_spread': -0.8},
                'QUESTIONABLE': {'opp_scoring': 1.02, 'team_spread': -0.3},
            }
        }
    
    def _load_injuries(self):
        """Load current injuries from file"""
        try:
            with open(self.injury_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {'last_updated': datetime.now().isoformat(), 'injuries': []}
    
    def _save_injuries(self):
        """Save injuries to file"""
        self.injuries['last_updated'] = datetime.now().isoformat()
        with open(self.injury_file, 'w') as f:
            json.dump(self.injuries, f, indent=2)
    
    def add_injury(self, team, player_name, position, status, notes=''):
        """Add or update an injury"""
        # Remove old injury for same player if exists
        self.injuries['injuries'] = [
            inj for inj in self.injuries.get('injuries', []) 
            if inj['player_name'] != player_name
        ]
        
        # Add new injury
        self.injuries.setdefault('injuries', []).append({
            'team': team,
            'player_name': player_name,
            'position': position,
            'status': status,
            'notes': notes,
            'added_date': datetime.now().isoformat()
        })
        
        self._save_injuries()
    
    def remove_injury(self, player_name):
        """Remove an injury (player returned)"""
        self.injuries['injuries'] = [
            inj for inj in self.injuries.get('injuries', []) 
            if inj['player_name'] != player_name
        ]
        self._save_injuries()
    
    def get_team_injuries(self, team):
        """Get all injuries for a team"""
        return [
            inj for inj in self.injuries.get('injuries', [])
            if inj['team'] == team
        ]
    
    def calculate_team_adjustment(self, team, is_offense=True):
        """
        Calculate aggregate injury adjustment for a team
        Returns: (scoring_multiplier, spread_adjustment)
        """
        team_injuries = self.get_team_injuries(team)
        
        if not team_injuries:
            return 1.0, 0.0
        
        scoring_multiplier = 1.0
        spread_adjustment = 0.0
        
        for injury in team_injuries:
            position = injury['position']
            status = injury['status']
            
            if position in self.impact_multipliers and status in self.impact_multipliers[position]:
                impact = self.impact_multipliers[position][status]
                
                if is_offense:
                    # Offensive injuries reduce scoring
                    if 'team_scoring' in impact:
                        scoring_multiplier *= impact['team_scoring']
                    if 'team_spread' in impact:
                        spread_adjustment += impact['team_spread']
                else:
                    # Defensive injuries increase opponent scoring
                    if 'opp_scoring' in impact:
                        scoring_multiplier *= impact['opp_scoring']
                    if 'team_spread' in impact:
                        spread_adjustment += impact['team_spread']
        
        return scoring_multiplier, spread_adjustment
    
    def adjust_game_prediction(self, prediction, home_team, away_team):
        """
        Adjust a game prediction based on injuries
        prediction: dict with 'home_score', 'away_score', 'spread', 'total'
        """
        # Get injury adjustments
        home_off_mult, home_off_spread = self.calculate_team_adjustment(home_team, is_offense=True)
        away_off_mult, away_off_spread = self.calculate_team_adjustment(away_team, is_offense=True)
        
        home_def_mult, home_def_spread = self.calculate_team_adjustment(home_team, is_offense=False)
        away_def_mult, away_def_spread = self.calculate_team_adjustment(away_team, is_offense=False)
        
        # Adjust scores
        adjusted_home = prediction['home_score'] * home_off_mult * away_def_mult
        adjusted_away = prediction['away_score'] * away_off_mult * home_def_mult
        
        # Adjust spread (combine offensive and defensive impacts)
        total_spread_adj = (home_off_spread + home_def_spread) - (away_off_spread + away_def_spread)
        adjusted_spread = prediction['spread'] + total_spread_adj
        
        # Calculate if adjustment is significant
        home_injuries = self.get_team_injuries(home_team)
        away_injuries = self.get_team_injuries(away_team)
        
        adjustment_note = ""
        if home_injuries or away_injuries:
            adjustment_note = f"Injury-adjusted: "
            if home_injuries:
                adjustment_note += f"{home_team} ({len(home_injuries)} key injuries) "
            if away_injuries:
                adjustment_note += f"{away_team} ({len(away_injuries)} key injuries)"
        
        return {
            'home_score': round(adjusted_home, 1),
            'away_score': round(adjusted_away, 1),
            'spread': round(adjusted_spread, 1),
            'total': round(adjusted_home + adjusted_away, 1),
            'original_home_score': prediction['home_score'],
            'original_away_score': prediction['away_score'],
            'original_spread': prediction['spread'],
            'injury_adjusted': bool(home_injuries or away_injuries),
            'adjustment_note': adjustment_note
        }
    
    def adjust_player_prediction(self, prediction, player_name, team):
        """
        Adjust a player prediction based on their injury status
        Returns: (adjusted_value, confidence_level)
        """
        team_injuries = self.get_team_injuries(team)
        
        # Check if this specific player is injured
        player_injury = next((inj for inj in team_injuries if inj['player_name'] == player_name), None)
        
        if not player_injury:
            return prediction, "Healthy"
        
        position = player_injury['position']
        status = player_injury['status']
        
        if position not in self.impact_multipliers or status not in self.impact_multipliers[position]:
            return prediction, f"{status} - Use caution"
        
        impact = self.impact_multipliers[position][status]
        
        # Determine which multiplier to use based on position
        multiplier = 1.0
        if 'QB' in position and 'player_passing' in impact:
            multiplier = impact['player_passing']
        elif 'RB' in position and 'player_rushing' in impact:
            multiplier = impact['player_rushing']
        elif 'WR' in position or 'TE' in position:
            if 'player_receiving' in impact:
                multiplier = impact['player_receiving']
        
        adjusted = prediction * multiplier if multiplier > 0 else 0
        
        confidence = {
            'OUT': "❌ Out - Do not bet",
            'DOUBTFUL': "⚠️ Doubtful - High risk",
            'QUESTIONABLE': "🟡 Questionable - Monitor closely", 
            'PROBABLE': "🟢 Probable - Slight concern"
        }.get(status, "Unknown")
        
        return round(adjusted, 1), f"{confidence} ({status})"

def render_injury_manager(injury_system, available_teams):
    """Render the injury management UI in Streamlit sidebar"""
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏥 Injury Tracker")
    
    # Show last update time
    last_update = injury_system.injuries.get('last_updated', 'Never')
    if last_update != 'Never':
        last_update = datetime.fromisoformat(last_update).strftime('%m/%d %H:%M')
    st.sidebar.caption(f"Last updated: {last_update}")
    
    # Current injuries summary
    all_injuries = injury_system.injuries.get('injuries', [])
    if all_injuries:
        st.sidebar.info(f"📋 Tracking {len(all_injuries)} injuries")
        
        # Group by team
        teams_with_injuries = {}
        for inj in all_injuries:
            team = inj['team']
            teams_with_injuries.setdefault(team, []).append(inj)
        
        # Show summary
        with st.sidebar.expander(f"View All ({len(all_injuries)} injuries)"):
            for team, injuries in teams_with_injuries.items():
                st.write(f"**{team}:**")
                for inj in injuries:
                    status_emoji = {
                        'OUT': '❌',
                        'DOUBTFUL': '⚠️',
                        'QUESTIONABLE': '🟡',
                        'PROBABLE': '🟢'
                    }.get(inj['status'], '❓')
                    st.write(f"{status_emoji} {inj['player_name']} ({inj['position']}) - {inj['status']}")
                    if inj.get('notes'):
                        st.caption(f"   {inj['notes']}")
    else:
        st.sidebar.success("✅ No tracked injuries")
    
    # Add injury form
    with st.sidebar.expander("➕ Add/Update Injury"):
        team = st.selectbox("Team", available_teams, key="injury_team")
        player_name = st.text_input("Player Name", key="injury_player")
        position = st.selectbox("Position", 
            ['QB', 'RB1', 'RB2', 'WR1', 'WR2', 'TE1', 'OL_MULTIPLE', 'DEF_STAR'],
            key="injury_position"
        )
        status = st.selectbox("Status", 
            ['OUT', 'DOUBTFUL', 'QUESTIONABLE', 'PROBABLE'],
            key="injury_status"
        )
        notes = st.text_input("Notes (optional)", 
            placeholder="e.g., Ankle injury, game-time decision",
            key="injury_notes"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add/Update", key="add_injury"):
                if player_name:
                    injury_system.add_injury(team, player_name, position, status, notes)
                    st.success(f"Added {player_name}")
                    st.rerun()
        
        with col2:
            if st.button("Clear All", key="clear_injuries"):
                injury_system.injuries['injuries'] = []
                injury_system._save_injuries()
                st.success("Cleared all injuries")
                st.rerun()
    
    # Remove injury
    if all_injuries:
        with st.sidebar.expander("➖ Remove Injury"):
            player_to_remove = st.selectbox(
                "Select player to remove",
                [inj['player_name'] for inj in all_injuries],
                key="remove_player"
            )
            if st.button("Remove", key="remove_injury"):
                injury_system.remove_injury(player_to_remove)
                st.success(f"Removed {player_to_remove}")
                st.rerun()

# Example integration into your existing prediction code:
def integrate_injuries_into_game_prediction(predictor, injury_system, team1, team2, home_team):
    """
    Wrapper function that adds injury adjustments to game predictions
    Use this in place of predictor.predict_game()
    """
    # Get base prediction
    base_prediction = predictor.predict_game(team1, team2, home_team)
    
    # Determine home and away teams
    if home_team == team1:
        home_team_name = team1
        away_team_name = team2
    else:
        home_team_name = team2
        away_team_name = team1
    
    # Create prediction in format expected by injury system
    prediction_for_adjustment = {
        'home_score': base_prediction['team1_score'] if home_team_name == team1 else base_prediction['team2_score'],
        'away_score': base_prediction['team2_score'] if home_team_name == team2 else base_prediction['team1_score'],
        'spread': base_prediction['spread'],
        'total': base_prediction['total']
    }
    
    # Apply injury adjustments
    adjusted = injury_system.adjust_game_prediction(
        prediction_for_adjustment,
        home_team_name,
        away_team_name
    )
    
    # Merge back into original format
    base_prediction.update({
        'team1_score': adjusted['away_score'] if home_team_name != team1 else adjusted['home_score'],
        'team2_score': adjusted['home_score'] if home_team_name == team2 else adjusted['away_score'],
        'spread': adjusted['spread'],
        'total': adjusted['total'],
        'injury_adjusted': adjusted['injury_adjusted'],
        'adjustment_note': adjusted['adjustment_note'],
        'original_spread': adjusted['original_spread'] if adjusted['injury_adjusted'] else None
    })
    
    return base_prediction

def integrate_injuries_into_player_prediction(injury_system, prediction, player_name, team):
    """
    Wrapper for player predictions with injury adjustments
    """
    if prediction is None:
        return None, "Prediction failed"
    
    adjusted_pred, confidence = injury_system.adjust_player_prediction(prediction, player_name, team)
    
    return adjusted_pred, confidence

# Usage example in your Streamlit app:
"""
# In your main app file, initialize the injury system:
injury_system = InjuryAdjustmentSystem()

# Add to sidebar:
render_injury_manager(injury_system, available_teams)

# When making game predictions:
prediction = integrate_injuries_into_game_prediction(
    prediction_system, 
    injury_system,
    away_team, 
    home_team, 
    home_team
)

# Show injury note if adjusted
if prediction.get('injury_adjusted'):
    st.warning(f"⚠️ {prediction['adjustment_note']}")
    st.caption(f"Original spread: {prediction['original_spread']:+.1f} → Adjusted: {prediction['spread']:+.1f}")

# When making player predictions:
base_prediction, status = prediction_system.predict_player_passing(qb_stats, opponent)
adjusted_pred, injury_status = integrate_injuries_into_player_prediction(
    injury_system,
    base_prediction,
    selected_qb,
    qb_stats['team']
)

if "Out" in injury_status or "Doubtful" in injury_status:
    st.error(f"🚨 {injury_status}")
"""