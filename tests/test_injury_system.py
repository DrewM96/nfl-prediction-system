from pathlib import Path

from injury_system import InjuryAdjustmentSystem, integrate_injuries_into_game_prediction


def test_injury_adjustment_keeps_scores_margin_and_total_consistent(tmp_path: Path) -> None:
    system = InjuryAdjustmentSystem(tmp_path / "injuries.json")
    system.add_injury("HOME", "Starting QB", "QB", "OUT")
    base = {
        "team1": "AWAY",
        "team1_score": 20.0,
        "team2": "HOME",
        "team2_score": 24.0,
        "away_team": "AWAY",
        "home_team": "HOME",
        "away_score": 20.0,
        "home_score": 24.0,
        "predicted_home_margin": 4.0,
        "spread": 4.0,
        "total": 44.0,
    }
    adjusted = integrate_injuries_into_game_prediction(base, system, "AWAY", "HOME", "HOME")
    assert adjusted["total"] == round(adjusted["home_score"] + adjusted["away_score"], 1)
    assert adjusted["predicted_home_margin"] == round(
        adjusted["home_score"] - adjusted["away_score"], 1
    )
    assert adjusted["team1_score"] == adjusted["away_score"]
    assert adjusted["team2_score"] == adjusted["home_score"]
