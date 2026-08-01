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
