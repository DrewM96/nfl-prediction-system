from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_app_switches_from_nfl_to_college_football() -> None:
    app = AppTest.from_file("app.py").run(timeout=30)
    sport = next(radio for radio in app.radio if radio.label == "Sport")

    assert sport.value == "NFL"
    assert not app.error
    assert not app.exception

    sport.set_value("College Football").run(timeout=30)

    assert not app.error
    assert not app.exception
    assert any("College Football" in markdown.value for markdown in app.markdown)
    assert any("2026 Week 1 Forecasts" in markdown.value for markdown in app.markdown)

    navigation = next(radio for radio in app.radio if radio.label == "Navigate")
    navigation.set_value("Top 30").run(timeout=30)

    assert not app.error
    assert not app.exception
    assert any("College Football Top 30" in markdown.value for markdown in app.markdown)
    assert any("This is not the AP poll" in caption.value for caption in app.caption)
    assert any("scheduled FBS games" in markdown.value for markdown in app.markdown)


def test_power_rankings_default_to_latest_market_consensus() -> None:
    app = AppTest.from_file("app.py").run(timeout=30)
    navigation = next(radio for radio in app.radio if radio.label == "Navigate")
    navigation.set_value("Rankings").run(timeout=30)

    source = next(radio for radio in app.radio if radio.label == "Ranking source")
    assert source.value == "Latest market consensus"
    assert not app.error
    assert not app.exception
    assert any("Market-implied points" in markdown.value for markdown in app.markdown)
    assert any("Los Angeles Rams" in markdown.value for markdown in app.markdown)
    assert any("QB returns" in markdown.value for markdown in app.markdown)
    assert any("Weeks 1-4 totals model" in caption.value for caption in app.caption)

    source.set_value("2025 football form").run(timeout=30)
    assert not app.error
    assert not app.exception
    assert any("Completed-game form" in markdown.value for markdown in app.markdown)
