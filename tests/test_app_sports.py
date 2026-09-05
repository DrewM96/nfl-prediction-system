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
    assert any("Forecasts" in markdown.value for markdown in app.markdown)
    assert any("Featured CFB matchup" in markdown.value for markdown in app.markdown)
    assert any("logo" in markdown.value for markdown in app.markdown)
    assert any("Home win" in markdown.value for markdown in app.markdown)

    navigation = next(radio for radio in app.radio if radio.label == "Navigate")
    navigation.set_value("Top 30").run(timeout=30)

    assert not app.error
    assert not app.exception
    assert any("College Football Top 30" in markdown.value for markdown in app.markdown)
    assert any("GRIDLINE scores every scheduled" in caption.value for caption in app.caption)
    assert any("scheduled FBS games" in markdown.value for markdown in app.markdown)
    assert any("logo" in markdown.value for markdown in app.markdown)


def test_power_rankings_offer_a_football_form_view() -> None:
    app = AppTest.from_file("app.py").run(timeout=30)
    navigation = next(radio for radio in app.radio if radio.label == "Navigate")
    navigation.set_value("Rankings").run(timeout=30)

    source = next(radio for radio in app.radio if radio.label == "Ranking source")
    form_view = next(option for option in source.options if option.endswith("football form"))
    source.set_value(form_view).run(timeout=30)
    assert not app.error
    assert not app.exception
    assert any("Completed-game form" in markdown.value for markdown in app.markdown)
    assert any("Los Angeles Rams" in markdown.value for markdown in app.markdown)
    assert any("Los Angeles Rams logo" in markdown.value for markdown in app.markdown)
    assert any("QB returns" in markdown.value for markdown in app.markdown)
    assert any("Weeks 1-4 totals model" in caption.value for caption in app.caption)


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
