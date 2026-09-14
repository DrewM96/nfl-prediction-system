from pathlib import Path

path = Path("app.py")
text = path.read_text()

text = text.replace("import logging\n", "import logging\nimport math\n", 1)
text = text.replace(
    'CFB_PAGE_LABELS = ["This Week", "Top 30", "Results"]',
    'CFB_PAGE_LABELS = ["This Week", "Builder", "Top 30", "Results"]',
    1,
)

anchor = "def render_cfb_rankings(state: dict[str, Any]) -> None:\n"
assert anchor in text
builder = """def render_cfb_builder(state: dict[str, Any]) -> None:
    rankings = state.get("power_rankings") or {}
    manifest = state.get("model_manifest") or {}
    rows = rankings.get("ratings") or []
    if not rows or not manifest:
        page_header("CFB Matchup Builder")
        st.warning("The active CFB model and power-rating artifacts are required for custom matchups.")
        return

    ratings = {str(row["team"]): float(row["rating"]) for row in rows}
    teams = sorted(ratings)
    page_header("CFB Matchup Builder", "Schedule-decomposed model scenario")
    st.caption(
        "Choose any two rated FBS teams. This uses the active CFB model's schedule-decomposed "
        "team strengths plus the published home-field estimate; it is session-only and is not "
        "recorded as a production forecast."
    )

    away_default = teams.index("Tennessee") if "Tennessee" in teams else 0
    home_default = teams.index("Georgia") if "Georgia" in teams else (1 if len(teams) > 1 else 0)
    with st.container(border=True, key="cfb_builder_shell"):
        with st.form("cfb_builder_form", border=False):
            columns = st.columns(2)
            away_team = columns[0].selectbox("Away team", teams, index=away_default)
            home_team = columns[1].selectbox("Home team", teams, index=home_default)
            neutral_site = st.checkbox("Neutral site", value=False)
            submitted = st.form_submit_button("Run matchup", width="stretch")

        if submitted or "cfb_builder_prediction" not in st.session_state:
            if away_team == home_team:
                st.error("Choose two different teams.")
                st.session_state.cfb_builder_prediction = None
            else:
                home_field = 0.0 if neutral_site else float(rankings.get("home_field_points", 0.0))
                margin = ratings[home_team] - ratings[away_team] + home_field
                residual_std = max(
                    float(manifest["models"]["margin"].get("residual_std", 1.0)), 1e-6
                )
                home_win = 0.5 * (1.0 + math.erf(margin / residual_std / math.sqrt(2.0)))
                st.session_state.cfb_builder_prediction = {
                    "away_team": away_team,
                    "home_team": home_team,
                    "neutral_site": neutral_site,
                    "predicted_home_margin": margin,
                    "home_win_probability": home_win,
                    "margin_p10": margin - 1.2816 * residual_std,
                    "margin_p90": margin + 1.2816 * residual_std,
                    "away_rating": ratings[away_team],
                    "home_rating": ratings[home_team],
                    "home_field": home_field,
                }

        prediction = st.session_state.get("cfb_builder_prediction")
        if prediction:
            spread = cfb_margin_label(prediction, float(prediction["predicted_home_margin"]))
            range_text = (
                f'{float(prediction["margin_p10"]):+.1f} to '
                f'{float(prediction["margin_p90"]):+.1f}'
            )
            separator = "vs" if prediction["neutral_site"] else "@"
            st.markdown(
                f'''\n                <div class="grid-hero grid-cfb-hero" style="margin-top:16px">\n                  <div class="grid-kicker">Hypothetical matchup</div>\n                  <div class="grid-matchup" style="margin-top:12px">\n                    <div class="grid-cfb-team">{team_logo_html(str(prediction["away_team"]), "cfb", "hero")}<div class="grid-cfb-team-name-large">{html_text(prediction["away_team"])}</div></div>\n                    <div class="grid-at">{separator}</div>\n                    <div class="grid-cfb-team">{team_logo_html(str(prediction["home_team"]), "cfb", "hero")}<div class="grid-cfb-team-name-large">{html_text(prediction["home_team"])}</div></div>\n                  </div>\n                  <div class="grid-tiles" style="grid-template-columns:repeat(4,minmax(110px,1fr));margin-top:16px">\n                    <div class="grid-tile"><div class="grid-tile-label">GRIDLINE spread</div><div class="grid-tile-value" style="font-size:16px">{html_text(spread)}</div></div>\n                    <div class="grid-tile"><div class="grid-tile-label">Home win</div><div class="grid-tile-value">{format_probability(prediction["home_win_probability"])}</div></div>\n                    <div class="grid-tile"><div class="grid-tile-label">80% margin range</div><div class="grid-tile-value" style="font-size:14px">{range_text}</div></div>\n                    <div class="grid-tile"><div class="grid-tile-label">Home field</div><div class="grid-tile-value">{float(prediction["home_field"]):+.1f}</div></div>\n                  </div>\n                </div>\n                ''',
                unsafe_allow_html=True,
            )
            fit_mae = float(rankings.get("line_fit_mae", 0.0))
            st.caption(
                f'Session-only scenario · active model decomposition · scheduled-margin reconstruction MAE {fit_mae:.2f} points. '
                "Custom builder outputs are excluded from season results."
            )


"""
text = text.replace(anchor, builder + anchor, 1)

old_dispatch = '''    if st.session_state.cfb_active_screen == "Top 30":
        render_cfb_rankings(cfb_state)
    elif st.session_state.cfb_active_screen == "Results":
        render_results(CFB_PREDICTIONS_DIR, league="CFB")
    else:
        render_cfb_foundation(cfb_state)
'''
new_dispatch = '''    if st.session_state.cfb_active_screen == "Builder":
        render_cfb_builder(cfb_state)
    elif st.session_state.cfb_active_screen == "Top 30":
        render_cfb_rankings(cfb_state)
    elif st.session_state.cfb_active_screen == "Results":
        render_results(CFB_PREDICTIONS_DIR, league="CFB")
    else:
        render_cfb_foundation(cfb_state)
'''
assert old_dispatch in text
text = text.replace(old_dispatch, new_dispatch, 1)
path.write_text(text)

# Add focused source-level assertions alongside the existing UI tests.
test_path = Path("tests/test_app_sports.py")
test = test_path.read_text()
if "test_cfb_builder_is_available" not in test:
    test += '''

def test_cfb_builder_is_available() -> None:
    source = Path("app.py").read_text()
    assert 'CFB_PAGE_LABELS = ["This Week", "Builder", "Top 30", "Results"]' in source
    assert "def render_cfb_builder" in source
    assert 'cfb_active_screen == "Builder"' in source
    assert "Schedule-decomposed model scenario" in source
'''
    if "from pathlib import Path" not in test.splitlines()[:10]:
        test = "from pathlib import Path\n" + test
    test_path.write_text(test)
