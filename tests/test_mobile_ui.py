from __future__ import annotations

from pathlib import Path

APP_SOURCE = Path("app.py").read_text(encoding="utf-8")


def test_phone_layout_has_touch_navigation_and_compact_game_cards() -> None:
    assert "@media (max-width: 520px)" in APP_SOURCE
    assert (
        '.st-key-active_screen [role="radiogroup"] { grid-template-columns: repeat(6,minmax(0,1fr)); }'
        in APP_SOURCE
    )
    assert ".st-key-active_screen, .st-key-cfb_active_screen" in APP_SOURCE
    assert '[class*="st-key-game_card_"] [data-testid="stHorizontalBlock"]' in APP_SOURCE
    assert ".grid-game-detail { grid-template-columns: 1fr;" in APP_SOURCE
    assert "st.session_state.expanded_game_id = None" in APP_SOURCE
    assert "min-height: 44px" in APP_SOURCE


def test_phone_rankings_and_forecast_table_avoid_page_overflow() -> None:
    assert ".grid-rank-track { grid-column: 2 / 4; }" in APP_SOURCE
    assert '[data-testid="stDataFrame"] { max-width: 100%; overflow: hidden; }' in APP_SOURCE
    assert '[class*="st-key-cfb_game_card_"] [data-testid="stHorizontalBlock"]' in APP_SOURCE
    assert ".grid-cfb-forecast-summary { margin-bottom: 18px; }" in APP_SOURCE
    assert "row-gap: 14px !important;" in APP_SOURCE
    assert "padding: 4px 0 6px;" in APP_SOURCE
    assert "render_cfb_featured_game(featured)" in APP_SOURCE
    assert ".grid-team-logo--hero" in APP_SOURCE
    assert ".grid-team-logo--rank" in APP_SOURCE
    assert ".grid-team-inline {" in APP_SOURCE
    assert "flex-direction: column;" in APP_SOURCE
    assert "grid-team-logo--away" not in APP_SOURCE


def test_application_copy_uses_direct_forecast_language() -> None:
    defensive_phrases = (
        "paper analysis",
        "not betting advice",
        "not financial advice",
        "provisional",
        "preseason limitation",
        "This is not the AP poll",
    )

    assert all(phrase.lower() not in APP_SOURCE.lower() for phrase in defensive_phrases)
    assert "Calibrated forecast · independent comparison" in APP_SOURCE
    assert "Forecasts ready" in APP_SOURCE
