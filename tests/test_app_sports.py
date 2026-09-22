from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_switches_from_nfl_to_college_football() -> None:
    app = AppTest.from_file("app.py").run(timeout=30)
    sport = next(radio for radio in app.radio if radio.label == "Sport")

    assert sport.value == "NFL"
    assert not app.error
    assert not app.exception
    assert any("GRIDLINE" in markdown.value for markdown in app.markdown)
    assert any("Vegas" in markdown.value for markdown in app.markdown)
    assert any("Edge" in markdown.value for markdown in app.markdown)

    sport.set_value("College Football").run(timeout=30)

    assert not app.error
    assert not app.exception
    assert any("College Football" in markdown.value for markdown in app.markdown)
    assert any("Forecasts" in markdown.value for markdown in app.markdown)
    assert any("Featured CFB matchup" in markdown.value for markdown in app.markdown)
    assert any("logo" in markdown.value for markdown in app.markdown)
    assert any("Home win" in markdown.value for markdown in app.markdown)
    assert any("GRIDLINE" in markdown.value for markdown in app.markdown)
    assert any("Vegas" in markdown.value for markdown in app.markdown)

    navigation = next(radio for radio in app.radio if radio.label == "Navigate")
    navigation.set_value("Top 30").run(timeout=30)

    assert not app.error
    assert not app.exception
    assert any("College Football Top 30" in markdown.value for markdown in app.markdown)
    assert any("GRIDLINE scores every scheduled" in caption.value for caption in app.caption)
    assert any("scheduled FBS games" in markdown.value for markdown in app.markdown)
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


def test_nfl_matchup_views_expose_frozen_injury_context() -> None:
    source = Path("app.py").read_text()
    assert "def render_official_injury_snapshot" in source
    assert "context only, not applied to the model" in source
    assert "Official injury report snapshot" in source


def test_cfb_builder_is_available() -> None:
    source = Path("app.py").read_text()
    assert 'CFB_PAGE_LABELS = ["This Week", "Builder", "Top 30", "Results"]' in source
    assert "def render_cfb_builder" in source
    assert 'cfb_active_screen == "Builder"' in source
    assert "Schedule-decomposed model scenario" in source
