from pathlib import Path

app_path = Path("app.py")
text = app_path.read_text()

old = '''def cfb_spread_label(game: dict[str, Any]) -> str:\n    margin = float(game["predicted_home_margin"])\n    if abs(margin) < 0.05:\n        return "Pick"\n    favorite = game["home_team"] if margin > 0 else game["away_team"]\n    return f"{favorite} -{abs(margin):.1f}"\n'''
new = '''def cfb_margin_label(game: dict[str, Any], margin: float) -> str:\n    if abs(margin) < 0.05:\n        return "Pick"\n    favorite = game["home_team"] if margin > 0 else game["away_team"]\n    return f"{favorite} -{abs(margin):.1f}"\n\n\ndef cfb_spread_label(game: dict[str, Any]) -> str:\n    return cfb_margin_label(game, float(game["predicted_home_margin"]))\n\n\ndef cfb_market_spread_label(game: dict[str, Any]) -> str:\n    market = game.get("market_consensus") or {}\n    spread = market.get("spread") or {}\n    margin = spread.get("market_home_margin")\n    if margin is None:\n        return "—"\n    return cfb_margin_label(game, float(margin))\n'''
assert old in text
text = text.replace(old, new)

old = '''            <div class="grid-tiles">\n              <div class="grid-tile"><div class="grid-tile-label">Spread</div><div class="grid-tile-value" style="font-size:16px">{html_text(cfb_spread_label(game))}</div></div>\n              <div class="grid-tile"><div class="grid-tile-label">Home win</div><div class="grid-tile-value">{format_probability(game["home_win_probability"])}</div></div>\n              <div class="grid-tile"><div class="grid-tile-label">Total O/U</div><div class="grid-tile-value">{float(game["predicted_total"]):.1f}</div></div>\n              <div class="grid-tile"><div class="grid-tile-label">80% margin range</div><div class="grid-tile-value" style="font-size:14px">{margin_range}</div></div>\n            </div>'''
new = '''            <div class="grid-tiles" style="grid-template-columns:repeat(5,minmax(88px,1fr))">\n              <div class="grid-tile"><div class="grid-tile-label">GRIDLINE</div><div class="grid-tile-value" style="font-size:16px">{html_text(cfb_spread_label(game))}</div></div>\n              <div class="grid-tile"><div class="grid-tile-label">Vegas</div><div class="grid-tile-value" style="font-size:16px">{html_text(cfb_market_spread_label(game))}</div></div>\n              <div class="grid-tile"><div class="grid-tile-label">Home win</div><div class="grid-tile-value">{format_probability(game["home_win_probability"])}</div></div>\n              <div class="grid-tile"><div class="grid-tile-label">Total O/U</div><div class="grid-tile-value">{float(game["predicted_total"]):.1f}</div></div>\n              <div class="grid-tile"><div class="grid-tile-label">80% margin range</div><div class="grid-tile-value" style="font-size:14px">{margin_range}</div></div>\n            </div>'''
assert old in text
text = text.replace(old, new)

old = '''        columns = st.columns([1.5, 4.8, 1.0, 1.0, 1.0], vertical_alignment="center")'''
new = '''        columns = st.columns([1.5, 4.2, 1.0, 1.0, 1.0, 1.0], vertical_alignment="center")'''
assert old in text
text = text.replace(old, new)

old = '''        columns[2].markdown(\n            f'<div class="grid-row-value"><div class="grid-mini-label">Spread</div>{html_text(cfb_spread_label(game))}</div>',\n            unsafe_allow_html=True,\n        )\n        columns[3].markdown(\n            f'<div class="grid-row-value"><div class="grid-mini-label">Home win</div>{format_probability(game["home_win_probability"])}</div>',\n            unsafe_allow_html=True,\n        )\n        columns[4].markdown(\n            f'<div class="grid-row-value"><div class="grid-mini-label">Total</div>{float(game["predicted_total"]):.1f}</div>',\n            unsafe_allow_html=True,\n        )'''
new = '''        columns[2].markdown(\n            f'<div class="grid-row-value"><div class="grid-mini-label">GRIDLINE</div>{html_text(cfb_spread_label(game))}</div>',\n            unsafe_allow_html=True,\n        )\n        columns[3].markdown(\n            f'<div class="grid-row-value"><div class="grid-mini-label">Vegas</div>{html_text(cfb_market_spread_label(game))}</div>',\n            unsafe_allow_html=True,\n        )\n        columns[4].markdown(\n            f'<div class="grid-row-value"><div class="grid-mini-label">Home win</div>{format_probability(game["home_win_probability"])}</div>',\n            unsafe_allow_html=True,\n        )\n        columns[5].markdown(\n            f'<div class="grid-row-value"><div class="grid-mini-label">Total</div>{float(game["predicted_total"]):.1f}</div>',\n            unsafe_allow_html=True,\n        )'''
assert old in text
text = text.replace(old, new)

app_path.write_text(text)

test_path = Path("tests/test_app_sports.py")
test = test_path.read_text()
old = '''    assert any("Home win" in markdown.value for markdown in app.markdown)\n'''
new = '''    assert any("Home win" in markdown.value for markdown in app.markdown)\n    assert any("GRIDLINE" in markdown.value for markdown in app.markdown)\n    assert any("Vegas" in markdown.value for markdown in app.markdown)\n'''
assert old in test
test_path.write_text(test.replace(old, new, 1))
