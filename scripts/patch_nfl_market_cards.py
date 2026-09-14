from pathlib import Path

app_path = Path("app.py")
text = app_path.read_text()

old = '''def market_tile(game: dict[str, Any]) -> str:\n    label, value = market_line_label(game)\n    pending = not bool(game.get("market_consensus"))\n    classes = "grid-tile dashed" if pending else "grid-tile"\n    value_class = "grid-tile-value pending" if pending else "grid-tile-value"\n    return f'<div class="{classes}"><div class="grid-tile-label">{html_text(label)}</div><div class="{value_class}">{html_text(value)}</div></div>'\n'''
new = '''def market_tile(game: dict[str, Any]) -> str:\n    label, value = market_line_label(game)\n    pending = not bool(game.get("market_consensus"))\n    classes = "grid-tile dashed" if pending else "grid-tile"\n    value_class = "grid-tile-value pending" if pending else "grid-tile-value"\n    return f'<div class="{classes}"><div class="grid-tile-label">{html_text(label)}</div><div class="{value_class}">{html_text(value)}</div></div>'\n\n\ndef nfl_market_spread_label(game: dict[str, Any]) -> str:\n    spread = (game.get("market_consensus") or {}).get("spread") or {}\n    home_spread = spread.get("home_spread")\n    if home_spread is None:\n        return "—"\n    home_spread = float(home_spread)\n    if abs(home_spread) < 0.05:\n        return "Pick"\n    favorite = game["home_team"] if home_spread < 0 else game["away_team"]\n    return f"{favorite} -{abs(home_spread):.1f}"\n\n\ndef nfl_market_edge_label(game: dict[str, Any]) -> str:\n    spread = (game.get("market_consensus") or {}).get("spread") or {}\n    home_spread = spread.get("home_spread")\n    if home_spread is None:\n        return "—"\n    market_home_margin = -float(home_spread)\n    model_home_margin = float(game.get("predicted_home_margin", game.get("spread", 0.0)))\n    edge = model_home_margin - market_home_margin\n    if abs(edge) < 0.05:\n        return "Even"\n    side = game["home_team"] if edge > 0 else game["away_team"]\n    return f"{side} +{abs(edge):.1f}"\n\n\ndef nfl_total_label(game: dict[str, Any]) -> str:\n    model_total = float(game["total"])\n    market_total = ((game.get("market_consensus") or {}).get("total") or {}).get("total")\n    if market_total is None:\n        return f"{model_total:.1f} · V —"\n    return f"{model_total:.1f} · V {float(market_total):.1f}"\n'''
assert old in text
text = text.replace(old, new)

old = '''              <div class="grid-tile"><div class="grid-tile-label">Spread</div><div class="grid-tile-value">{html_text(spread_label(game))}</div></div>\n              <div class="grid-tile"><div class="grid-tile-label">Model fair ML · {html_text(game["home_team"])}</div><div class="grid-tile-value">{format_american(american_moneyline(game["home_win_probability"]))}</div></div>\n              <div class="grid-tile"><div class="grid-tile-label">Total O/U</div><div class="grid-tile-value">{float(game["total"]):.1f}</div></div>\n              {market_tile(game)}'''
new = '''              <div class="grid-tile"><div class="grid-tile-label">GRIDLINE</div><div class="grid-tile-value">{html_text(spread_label(game))}</div></div>\n              <div class="grid-tile"><div class="grid-tile-label">Vegas</div><div class="grid-tile-value">{html_text(nfl_market_spread_label(game))}</div></div>\n              <div class="grid-tile"><div class="grid-tile-label">Edge</div><div class="grid-tile-value">{html_text(nfl_market_edge_label(game))}</div></div>\n              <div class="grid-tile"><div class="grid-tile-label">Total · model / Vegas</div><div class="grid-tile-value" style="font-size:16px">{html_text(nfl_total_label(game))}</div></div>'''
assert old in text
text = text.replace(old, new)

old = '''        columns = st.columns([1.5, 4.8, 0.9, 0.9, 1.0, 0.9], vertical_alignment="center")'''
new = '''        columns = st.columns([1.5, 4.2, 0.9, 0.9, 0.9, 1.1, 0.8], vertical_alignment="center")'''
assert old in text
text = text.replace(old, new)

old = '''        columns[2].markdown(\n            f'<div class="grid-row-value"><div class="grid-mini-label">Spread</div>{html_text(spread_label(game))}</div>',\n            unsafe_allow_html=True,\n        )\n        columns[3].markdown(\n            f'<div class="grid-row-value"><div class="grid-mini-label">Fair ML · {html_text(game["home_team"])}</div>{format_american(american_moneyline(game["home_win_probability"]))}</div>',\n            unsafe_allow_html=True,\n        )\n        columns[4].markdown(\n            f'<div class="grid-row-value"><div class="grid-mini-label">Total</div>{float(game["total"]):.1f}</div>',\n            unsafe_allow_html=True,\n        )\n        if columns[5].button('''
new = '''        columns[2].markdown(\n            f'<div class="grid-row-value"><div class="grid-mini-label">GRIDLINE</div>{html_text(spread_label(game))}</div>',\n            unsafe_allow_html=True,\n        )\n        columns[3].markdown(\n            f'<div class="grid-row-value"><div class="grid-mini-label">Vegas</div>{html_text(nfl_market_spread_label(game))}</div>',\n            unsafe_allow_html=True,\n        )\n        columns[4].markdown(\n            f'<div class="grid-row-value"><div class="grid-mini-label">Edge</div>{html_text(nfl_market_edge_label(game))}</div>',\n            unsafe_allow_html=True,\n        )\n        columns[5].markdown(\n            f'<div class="grid-row-value"><div class="grid-mini-label">Total · M / V</div>{html_text(nfl_total_label(game))}</div>',\n            unsafe_allow_html=True,\n        )\n        if columns[6].button('''
assert old in text
text = text.replace(old, new)

app_path.write_text(text)

test_path = Path("tests/test_app_sports.py")
test = test_path.read_text()
old = '''    assert sport.value == "NFL"\n    assert not app.error\n    assert not app.exception\n'''
new = '''    assert sport.value == "NFL"\n    assert not app.error\n    assert not app.exception\n    assert any("GRIDLINE" in markdown.value for markdown in app.markdown)\n    assert any("Vegas" in markdown.value for markdown in app.markdown)\n    assert any("Edge" in markdown.value for markdown in app.markdown)\n'''
assert old in test
test_path.write_text(test.replace(old, new, 1))
