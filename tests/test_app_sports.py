from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_switches_from_nfl_to_college_football() -> None:
    app = AppTest.from_file("app.py").run(timeout=30)
    sport = next(radio for radio in app.radio if radio.label == "Sport")

    assert sport.value == "NFL"
    assert not app.error
    assert not app.exception
    assert any("GRIDLINE" in markdown.value for markdown in app.markdown)
    assert any("Market at forecast" in markdown.value for markdown in app.markdown)
    assert any("Home win" in markdown.value for markdown in app.markdown)
    assert any("Total model / Vegas" in markdown.value for markdown in app.markdown)
    nfl_labels = [m.value for m in app.markdown if 'class="grid-mini-label"' in m.value][:4]
    next(button for button in app.button if button.label == "Details ▼").click().run(timeout=30)
    assert not app.exception
    assert any("Why the model leans this way" in m.value for m in app.markdown)

    sport.set_value("College Football").run(timeout=30)

    assert not app.error
    assert not app.exception
    assert any("College Football" in markdown.value for markdown in app.markdown)
    assert any("Forecasts" in markdown.value for markdown in app.markdown)
    assert any("Featured matchup" in markdown.value for markdown in app.markdown)
    assert any("logo" in markdown.value for markdown in app.markdown)
    assert any("Home win" in markdown.value for markdown in app.markdown)
    assert any("Total model / Vegas" in markdown.value for markdown in app.markdown)
    assert any("GRIDLINE" in markdown.value for markdown in app.markdown)
    assert any("Market at forecast" in markdown.value for markdown in app.markdown)

    cfb_labels = [m.value for m in app.markdown if 'class="grid-mini-label"' in m.value][:4]
    for nfl_label, cfb_label in zip(nfl_labels, cfb_labels, strict=True):
        assert nfl_label.split("</div>")[0] == cfb_label.split("</div>")[0]
    next(button for button in app.button if button.label == "Details ▼").click().run(timeout=30)
    assert not app.exception
    assert any(button.label == "Hide ▲" for button in app.button)

    navigation = next(radio for radio in app.radio if radio.label == "Navigate")
    navigation.set_value("Top 30").run(timeout=30)

    assert not app.error
    assert not app.exception
    assert any("College Football Top 30" in markdown.value for markdown in app.markdown)
    pointer = json.loads(Path("data/cfb/latest_prediction.json").read_text(encoding="utf-8"))
    ranking_path = pointer.get("rankings_path") or "data/cfb/power_rankings.json"
    rankings = json.loads(Path(ranking_path).read_text(encoding="utf-8"))
    if rankings.get("kind") == "blended_common_opponent_results":
        assert any("75% common-opponent model ratings + 25%" in c.value for c in app.caption)
        assert any("completed FBS games" in m.value for m in app.markdown)
        assert any("Model disagreement MAE" in m.value for m in app.markdown)
    else:
        assert any("GRIDLINE scores every scheduled" in c.value for c in app.caption)
        assert any("scheduled FBS games" in m.value for m in app.markdown)
    assert any("logo" in markdown.value for markdown in app.markdown)


def test_power_rankings_default_to_point_calibrated_model_view() -> None:
    app = AppTest.from_file("app.py").run(timeout=45)
    navigation = next(radio for radio in app.radio if radio.label == "Navigate")
    navigation.set_value("Rankings").run(timeout=45)

    source = next(radio for radio in app.radio if radio.label == "Ranking source")
    assert source.value == "GRIDLINE model"
    assert "Recent form index" in source.options
    assert not app.error
    assert not app.exception
    assert any("Model-implied points above or below" in markdown.value for markdown in app.markdown)
    assert any("Rating reconstruction MAE" in markdown.value for markdown in app.markdown)
    assert any("Los Angeles Rams" in markdown.value for markdown in app.markdown)
    assert any("QB returns" in markdown.value for markdown in app.markdown)

    source.set_value("Recent form index").run(timeout=45)
    assert not app.error
    assert not app.exception
    assert any("Recent-form index" in markdown.value for markdown in app.markdown)
    assert any("not point-spread calibrated" in markdown.value for markdown in app.markdown)


def test_results_are_available_for_both_sports() -> None:
    app = AppTest.from_file("app.py").run(timeout=30)
    navigation = next(radio for radio in app.radio if radio.label == "Navigate")
    navigation.set_value("Results").run(timeout=30)
    assert any("Season Results" in heading.value for heading in app.header)
    assert not app.error and not app.exception

    sport = next(radio for radio in app.radio if radio.label == "Sport")
    sport.set_value("College Football").run(timeout=30)
    navigation = next(radio for radio in app.radio if radio.label == "Navigate")
    navigation.set_value("Results").run(timeout=30)
    assert any("Season Results" in heading.value for heading in app.header)
    assert not app.error and not app.exception


def test_props_page_preserves_manual_comparison_and_handles_empty_cache() -> None:
    app = AppTest.from_file("app.py").run(timeout=30)
    next(radio for radio in app.radio if radio.label == "Navigate").set_value("Props").run(
        timeout=30
    )
    assert not app.error and not app.exception
    assert any("Expected starters" in caption.value for caption in app.caption)
    assert any(expander.label == "Manual player comparison" for expander in app.expander)
    assert next(box for box in app.checkbox if box.label == "Include older lines").value is False


def test_nfl_matchup_views_expose_frozen_injury_context() -> None:
    source = Path("app.py").read_text()
    assert "def render_official_injury_snapshot" in source
    assert "Player availability" in source
    assert "Player availability snapshot" in source
    assert "grid-injury-panel" in source
    assert "_format_injury_snapshot_time" in source


def test_featured_nfl_game_prioritizes_intrigue_not_largest_spread() -> None:
    source = Path("app.py").read_text()
    assert "def _featured_game_score" in source
    assert "0.45 * model_competitive" in source
    assert "0.30 * market_competitive" in source
    assert "0.25 * disagreement" in source
    assert "featured = max(schedule, key=_featured_game_score)" in source


def test_cfb_builder_is_available() -> None:
    source = Path("app.py").read_text()
    assert 'CFB_PAGE_LABELS = ["This Week", "Builder", "Top 30", "Results", "Picks"]' in source
    assert "def render_cfb_builder" in source
    assert 'cfb_active_screen == "Builder"' in source
    assert "Schedule-decomposed model scenario" in source


def test_gridline_typography_uses_one_tokenized_font_system() -> None:
    source = Path("app.py").read_text()
    assert "Albert Sans" not in source
    assert "--font-ui: 'Instrument Sans'" in source
    assert "--text-xs: 11px;" in source
    assert "--text-base: 14px;" in source
    assert "--text-title: 26px;" in source
    assert "--radius-pill: 999px;" in source
    assert '[data-testid="stMetricValue"]' in source
    assert '[data-testid="stSelectbox"] label p' in source
