"""Shared, read-only NFL and college performance dashboard."""

from __future__ import annotations

import math
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from .ledger import PredictionLedger
from .player_props import PROP_TITLES
from .prop_results import player_breakdown, player_forecast_rows
from .results import forecast_rows, select_forecasts, summarize
from .results_tracker import (
    numeric,
    pick_record,
    pick_side,
    score_games,
    team_breakdown,
    weekly_records,
)
from .ui import html_text


def _number(value, digits=1):
    return "—" if numeric(value) is None else f"{float(value):.{digits}f}"


def _percent(value):
    return "—" if numeric(value) is None else f"{value:.1%}"


def _record(record, *, pushes=False):
    if not record["decisions"] and not record["pushes"] and not record["ties"]:
        return "—"
    text = f"{record['wins']}–{record['losses']}"
    return f"{text}–{record['pushes']}" if pushes else text


def _html(markup):
    st.markdown(markup, unsafe_allow_html=True)


def _section(title, subtitle=""):
    _html(
        f'<div class="grid-results-section"><div><h3>{html_text(title)}</h3><p>{html_text(subtitle)}</p></div></div>'
    )


def _empty(title, description):
    _html(
        f'<div class="grid-results-empty"><b>{html_text(title)}</b><p>{html_text(description)}</p></div>'
    )


def _badge(value):
    style = {"Win": "win", "Loss": "loss", "Push": "push", "Tie": "tie"}.get(value, "neutral")
    return f'<span class="grid-results-tag is-{style}">{html_text(value)}</span>'


def _stat_html(label, value, detail, *, rate=None, footer=""):
    meter = (
        f'<div class="grid-results-meter"><span style="width:{max(0, min(rate, 1)) * 100:.2f}%"></span></div>'
        if rate is not None
        else ""
    )
    return (
        f'<div class="grid-results-stat"><div class="grid-results-stat-label">{html_text(label)}</div>'
        f'<div class="grid-results-stat-number">{html_text(value)}</div>'
        f'<div class="grid-results-stat-meta">{html_text(detail)}</div>{meter}'
        f'<div class="grid-results-stat-meta">{html_text(footer)}</div></div>'
    )


def _record_html(label, record, *, pushes=False):
    excluded = f"{record['pushes']} pushes" if pushes else f"{record['ties']} ties"
    missing = f" · {record['no_line']} without lines" if pushes and record["no_line"] else ""
    return _stat_html(
        label,
        _record(record, pushes=pushes),
        f"{_percent(record['rate'])} hit rate · {record['decisions']} decisions",
        rate=record["rate"],
        footer=f"{excluded} · {record['no_pick']} no picks{missing}",
    )


def _hero_html(rows, *, league, season, through, source):
    counts = rows.status.value_counts()
    final = int(counts.get("final", 0))
    pending = int(counts.get("awaiting result", 0)) + int(counts.get("scheduled", 0))
    coverage = "".join(
        f'<span class="grid-results-pill"><b>{count}</b>{label}</span>'
        for count, label in ((final, "final"), (pending, "pending"), (len(rows), "tracked games"))
    )
    winner = pick_record(rows.winner_result)
    return (
        '<div class="grid-results-hero"><div>'
        f'<div class="grid-results-eyebrow">{html_text(league)} · {html_text(season)} · Through week {html_text(through)}</div>'
        '<div class="grid-results-title">Every pick. On the record.</div>'
        f'<div class="grid-results-copy">Follow the winners, the misses, and the teams we read best. '
        f"{html_text(source.title())} forecasts, saved before kickoff.</div>"
        f'<div class="grid-results-coverage">{coverage}</div></div>'
        '<div class="grid-results-hero-stat"><div class="grid-results-stat-label">Winner hit rate</div>'
        f'<div class="grid-results-big">{_percent(winner["rate"])}</div>'
        f'<div class="grid-results-stat-meta">{winner["decisions"]} decided picks · ties excluded</div></div></div>'
    )


def render_results(root: str | Path, *, league: str) -> None:
    st.header("Season Results")
    versions = forecast_rows(root)
    if versions.empty:
        _empty(
            "The record starts with the first forecast",
            "Published pregame forecasts will appear here with their outcomes, picks, and market comparisons.",
        )
        return
    prefix = f"results_{league}"
    controls = st.columns([1, 1, 2, 1.5])
    season = controls[0].selectbox(
        "Season", sorted(versions.season.unique(), reverse=True), key=f"{prefix}_season"
    )
    versions = versions[versions.season.eq(season)]
    through = controls[1].selectbox(
        "Through week", sorted(versions.week.unique(), reverse=True), key=f"{prefix}_week"
    )
    selection = controls[2].selectbox(
        "Forecast selection",
        ["Latest at least 60m before kickoff", "First published"],
        key=f"{prefix}_policy",
    )
    source = (
        controls[3]
        .selectbox("Forecast source", ["Published", "Independent"], key=f"{prefix}_source")
        .lower()
    )
    policy = "first" if selection == "First published" else "horizon"
    versions = versions[versions.week.le(through)]
    rows = score_games(select_forecasts(versions, policy=policy), source=source)
    if rows.empty:
        _empty(
            "No forecasts qualify for this selection",
            "Try First published to see earlier releases. Forecasts issued after kickoff never enter the record.",
        )
        return
    _html(_hero_html(rows, league=league, season=season, through=through, source=source))
    cards = "".join(
        _record_html(label, pick_record(rows[f"{kind}_result"]), pushes=kind != "winner")
        for kind, label in (
            ("winner", "Winner picks"),
            ("ats", "Against the spread"),
            ("total", "Over / under"),
        )
    )
    error = rows.margin_error.dropna()
    cards += _stat_html(
        "Spread accuracy",
        f"{_number(error.mean())} pts",
        "Average absolute margin error",
        footer=f"{len(error)} settled projections · lower is better",
    )
    _html(f'<div class="grid-results-summary">{cards}</div>')
    st.caption(
        "ATS and totals use the market line saved with each forecast, not the closing line. Hit rates exclude pushes, ties, and no-picks; records are W–L or W–L–P. A model within 0.05 of the line makes no pick."
    )
    names = ["Overview", "Wagon tracker", "Game log"] + (
        ["Player props"] if league == "NFL" else []
    )
    tabs = st.tabs(names)
    with tabs[0]:
        _overview(rows, source=source, prefix=prefix)
    with tabs[1]:
        _wagons(rows, source=source, prefix=prefix)
    with tabs[2]:
        _game_log(rows, versions, root, source=source, prefix=prefix, league=league, season=season)
    if league == "NFL":
        with tabs[3]:
            _player_results(root, season=season, through=through, policy=policy, prefix=prefix)
    with st.expander("How this record is scored"):
        st.markdown(
            "One pregame forecast per game. **Latest** uses the newest release at least 60 minutes before kickoff; **First published** can include preseason forecasts. "
            "Winner picks follow the selected model's projected margin. ATS picks follow its difference from the saved market margin. "
            "Missing or invalid market snapshots never become picks. Cancelled and postponed games are excluded. "
            "Coverage describes recorded forecasts, not the entire league schedule. Records track directional calls, not returns after sportsbook prices or fees."
        )
        if rows.scored_at.notna().any():
            st.caption(f"Latest recorded settlement: {rows.scored_at.dropna().max()}")
        _diagnostics(rows, source=source)


def _overview(rows, *, source, prefix):
    _section("The season in motion", "Track the record as each week settles.")
    mode = st.radio(
        "Trend",
        ["Season to date", "By week"],
        horizontal=True,
        key=f"{prefix}_trend",
        label_visibility="collapsed",
    )
    trend = weekly_records(rows, cumulative=mode == "Season to date")
    if trend.empty:
        _empty(
            "Waiting for the first final",
            "Records and weekly trends will fill in as official results arrive.",
        )
    else:
        chart = (
            alt.Chart(trend)
            .mark_line(point=alt.OverlayMarkDef(size=65), strokeWidth=2.5)
            .encode(
                x=alt.X(
                    "week:O", title=None, axis=alt.Axis(labelExpr="'W' + datum.label", labelAngle=0)
                ),
                y=alt.Y(
                    "rate:Q",
                    title="Hit rate",
                    axis=alt.Axis(format=".0%"),
                    scale=alt.Scale(domain=[0, 1]),
                ),
                color=alt.Color(
                    "series:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["Winners", "Against the spread", "Totals"],
                        range=["#ff6b35", "#334155", "#7888b5"],
                    ),
                    legend=alt.Legend(orient="bottom"),
                ),
                tooltip=[
                    alt.Tooltip("series:N", title="Record"),
                    alt.Tooltip("week:O", title="Through week"),
                    alt.Tooltip("rate:Q", title="Hit rate", format=".1%"),
                    "wins:Q",
                    "losses:Q",
                    "pushes:Q",
                    "decisions:Q",
                ],
            )
            .properties(height=260)
            .configure_view(stroke=None)
            .configure_axis(
                gridColor="#f1f3f5",
                domain=False,
                tickSize=0,
                labelColor="#64748b",
                titleColor="#64748b",
            )
        )
        st.altair_chart(chart, use_container_width=True)
    _section("Week by week", "Every slate, with its own sample size.")
    weeks = []
    for week in sorted(rows.week.unique(), reverse=True):
        group = rows[rows.week.eq(week)]
        final = int(group.status.eq("final").sum())
        lines = "".join(
            f'<div class="grid-results-week-line"><span>{label}</span><b>{_record(pick_record(group[f"{kind}_result"]), pushes=kind != "winner")}</b></div>'
            for kind, label in (("winner", "Winners"), ("ats", "ATS"), ("total", "Totals"))
        )
        weeks.append(
            f'<div class="grid-results-week"><div class="grid-results-week-head"><span>Week {int(week)}</span><span class="grid-results-game-meta">{final}/{len(group)} final</span></div>{lines}</div>'
        )
    _html('<div class="grid-results-week-grid">' + "".join(weeks) + "</div>")
    _section("Model vs market", "Error in points on the same games. Lower is better.")
    target = st.radio(
        "Compare accuracy", ["Margin", "Total"], horizontal=True, key=f"{prefix}_comparison"
    ).lower()
    stats = summarize(rows, source=source, target=target)
    cards = _stat_html(
        "Model error",
        _number(stats["matched_model_mae"]),
        f"{stats['matched_games']} matched games",
    )
    cards += _stat_html(
        "Market error", _number(stats["market_mae"]), "Market snapshot at forecast time"
    )
    difference = stats["difference"]
    direction = (
        "Model lower"
        if difference is not None and difference < 0
        else "Market lower"
        if difference is not None and difference > 0
        else "Even"
        if difference == 0
        else "No matched results"
    )
    cards += _stat_html(
        "Accuracy gap",
        f"{_number(abs(difference) if difference is not None else None)} pts",
        direction,
    )
    _html(f'<div class="grid-results-summary">{cards}</div>')
    if stats["difference_low"] is not None:
        st.caption(
            f"Model minus market error: 95% week-block bootstrap range {stats['difference_low']:+.2f} to {stats['difference_high']:+.2f} points. Negative favors the model."
        )
    elif stats["matched_games"]:
        st.caption(
            "Fewer than four matched weeks: too early to estimate a reliable error-difference range."
        )


def _wagons(rows, *, source, prefix):
    _section("Wagon tracker", "Which teams does the model read best?")
    st.caption(
        "Ranked by error in each team's projected score. Winner records cover games involving the team; ATS records count only picks backing that team. A game appears in both teams' histories, but only once in the season record."
    )
    controls = st.columns([1, 2])
    minimum = controls[0].number_input(
        "Minimum settled games", min_value=1, max_value=25, value=3, step=1, key=f"{prefix}_minimum"
    )
    order = controls[1].selectbox(
        "Rank teams by",
        ["Score accuracy", "ATS when backed", "Winner accuracy"],
        key=f"{prefix}_ranking",
    )
    teams = team_breakdown(rows, source=source, minimum=minimum)
    if teams.empty:
        _empty(
            "The wagons are still forming",
            f"No teams have {minimum} settled score projections in this selection. Lower the minimum to explore the early results.",
        )
        return
    if order != "Score accuracy":
        key = "ats_rate" if order == "ATS when backed" else "winner_rate"
        teams = teams.sort_values(
            [key, "games", "score_mae"], ascending=[False, False, True], na_position="last"
        ).reset_index(drop=True)
    best = teams.sort_values("score_mae").iloc[0]
    hardest = teams.sort_values("score_mae").iloc[-1]
    backed = teams.sort_values(["ats_picks", "score_mae"], ascending=[False, True]).iloc[0]
    spotlights = []
    for label, team, detail in (
        ("Best read", best, f"{best.score_mae:.1f} pts average score error"),
        ("Room to improve", hardest, f"{hardest.score_mae:.1f} pts average score error"),
        ("Most backed ATS", backed, f"{int(backed.ats_picks)} settled picks backing this team"),
    ):
        name = team.team if label != "Most backed ATS" or backed.ats_picks else "No picks yet"
        spotlights.append(
            f'<div class="grid-results-wagon"><div class="grid-results-eyebrow">{html_text(label)}</div><div class="grid-results-wagon-team">{html_text(name)}</div><div class="grid-results-stat-meta">{html_text(detail)}<br>{int(team.games)} games in sample</div></div>'
        )
    _html('<div class="grid-results-wagons">' + "".join(spotlights) + "</div>")
    page = _page_control(len(teams), 16, key=f"{prefix}_teams_page")
    markup = []
    for index, team in teams.iloc[page * 16 : (page + 1) * 16].iterrows():
        form = "".join(
            f'<span class="grid-results-tag is-{html_text(value.lower())}" title="{html_text(value)}">{html_text(value[0])}</span>'
            for value in team.form
        )
        ats = (
            f"{int(team.ats_wins)}–{int(team.ats_losses)}–{int(team.ats_pushes)}"
            if team.ats_picks
            else "—"
        )
        markup.append(
            '<div class="grid-results-team"><div class="grid-results-team-name">'
            f'<span class="grid-results-team-rank">{index + 1:02d}</span><div><b>{html_text(team.team)}</b>'
            f'<div class="grid-results-game-meta">{int(team.games)} games · recent winner picks</div><div class="grid-results-form">{form}</div></div></div>'
            f'<div class="grid-results-team-stats"><div><small>Score error</small><b>{team.score_mae:.1f} pts</b></div>'
            f"<div><small>Winners</small>{int(team.winner_wins)}–{int(team.winner_losses)}</div>"
            f"<div><small>ATS backed</small>{ats}</div><div><small>Score bias</small>{team.score_bias:+.1f}</div></div></div>"
        )
    _html('<div class="grid-results-team-list">' + "".join(markup) + "</div>")
    st.caption(
        "Recent form runs oldest to newest. Positive score bias means we projected too many points for that team. Small samples can move sharply after one game."
    )
    team = st.selectbox("Explore a team", sorted(teams.team), key=f"{prefix}_team_detail")
    selected = rows[rows.home_team.eq(team) | rows.away_team.eq(team)].sort_values(
        "kickoff", ascending=False
    )
    _html(
        '<div class="grid-results-games">'
        + "".join(_game_html(r, source=source) for r in selected.to_dict("records"))
        + "</div>"
    )


def _page_control(size, per_page, *, key):
    pages = max(1, math.ceil(size / per_page))
    if pages == 1:
        return 0
    return st.selectbox(
        "Page", list(range(pages)), format_func=lambda n: f"{n + 1} of {pages}", key=key
    )


def _pick_html(title, value, outcome, detail=""):
    return f'<div class="grid-results-pick"><div class="grid-results-pick-label">{html_text(title)}</div><div class="grid-results-pick-value">{html_text(value)} {_badge(outcome)}</div><div class="grid-results-pick-detail">{html_text(detail)}</div></div>'


def _game_html(row, *, source):
    home, away = row["home_team"], row["away_team"]
    margin, total = numeric(row.get("actual_margin")), numeric(row.get("actual_total"))
    score = (
        f"{away} {(total - margin) / 2:g} · {home} {(total + margin) / 2:g}"
        if margin is not None and total is not None and row["status"] == "final"
        else row["status"].title()
    )
    winner = home if row["winner_side"] > 0 else away if row["winner_side"] < 0 else "Pick"
    market = numeric(row.get("market_margin"))
    ats = "No saved line" if market is None else "No lean"
    if market is not None and row["ats_side"]:
        team = home if row["ats_side"] > 0 else away
        spread = -market if row["ats_side"] > 0 else market
        ats = f"{team} {spread:+.1f}" if abs(spread) >= 0.05 else f"{team} Pick"
    total_line = numeric(row.get("market_total"))
    total_pick = (
        "No saved line"
        if total_line is None
        else f"{'Over' if row['total_side'] > 0 else 'Under' if row['total_side'] < 0 else 'No lean'} {_number(total_line)}"
    )
    picks = _pick_html(
        "Winner",
        winner,
        row["winner_result"],
        f"Model home margin {_number(row.get(f'{source}_margin'))}",
    )
    picks += _pick_html("Against the spread", ats, row["ats_result"], "Line at forecast")
    picks += _pick_html(
        "Total points",
        total_pick,
        row["total_result"],
        f"Model {_number(row.get(f'{source}_total'))} · Actual {_number(total)}",
    )
    return (
        '<article class="grid-results-game"><div class="grid-results-game-head"><div>'
        f'<div class="grid-results-game-meta">Week {int(row["week"])} · {html_text(row["status"].title())}</div>'
        f'<div class="grid-results-matchup">{html_text(away)} <span class="grid-muted">at</span> {html_text(home)}</div></div>'
        f'<div class="grid-results-pick-value">{html_text(score)}</div></div><div class="grid-results-game-picks">{picks}</div></article>'
    )


def _game_log(rows, versions, root, *, source, prefix, league, season):
    _section("The game log", "The pick, the line we used, and what actually happened.")
    controls = st.columns(3)
    week = controls[0].selectbox(
        "Game week",
        ["All weeks", *sorted(rows.week.unique(), reverse=True)],
        key=f"{prefix}_log_week",
    )
    team = controls[1].selectbox(
        "Team",
        ["All teams", *sorted(set(rows.home_team) | set(rows.away_team))],
        key=f"{prefix}_log_team",
    )
    outcome = controls[2].selectbox(
        "Show",
        ["Final", "All games", "Pending", "ATS wins", "ATS losses"],
        key=f"{prefix}_log_outcome",
    )
    display = rows.copy()
    if week != "All weeks":
        display = display[display.week.eq(week)]
    if team != "All teams":
        display = display[display.home_team.eq(team) | display.away_team.eq(team)]
    if outcome == "Final":
        display = display[display.status.eq("final")]
    elif outcome == "Pending":
        display = display[display.status.isin(["scheduled", "awaiting result"])]
    elif outcome.startswith("ATS"):
        display = display[display.ats_result.eq("Win" if outcome == "ATS wins" else "Loss")]
    display = display.sort_values(["week", "kickoff", "game_id"], ascending=[False, False, True])
    st.caption(f"{len(display)} games in this view")
    if display.empty:
        _empty(
            "No games match these filters",
            "Choose another week, team, or result to explore the record.",
        )
        return
    page = _page_control(len(display), 12, key=f"{prefix}_games_page")
    _html(
        '<div class="grid-results-games">'
        + "".join(
            _game_html(r, source=source)
            for r in display.iloc[page * 12 : (page + 1) * 12].to_dict("records")
        )
        + "</div>"
    )
    columns = [
        "game_id",
        "week",
        "away_team",
        "home_team",
        "status",
        "published_at",
        f"{source}_margin",
        "actual_margin",
        "market_margin",
        f"{source}_total",
        "actual_total",
        "market_total",
        "winner_result",
        "ats_result",
        "total_result",
        "margin_error",
        "total_error",
        "run_id",
        "revision",
    ]
    st.download_button(
        "Download this game log",
        display[columns].to_csv(index=False),
        file_name=f"gridline-{league}-{season}-results.csv",
        mime="text/csv",
        key=f"{prefix}_download",
    )
    with st.expander("Inspect a frozen forecast"):
        indexed = display.set_index("game_id")
        game_id = st.selectbox(
            "Game",
            display.game_id.tolist(),
            format_func=lambda g: f"W{indexed.loc[g, 'week']} · {indexed.loc[g, 'away_team']} at {indexed.loc[g, 'home_team']}",
            key=f"{prefix}_game",
        )
        history = versions[versions.game_id.eq(game_id)].sort_values("published_at")
        st.dataframe(
            history[
                [
                    "run_id",
                    "published_at",
                    "kickoff",
                    "eligible",
                    "forecast_method",
                    "published_margin",
                    "independent_margin",
                    "published_total",
                    "model_hash",
                    "data_cutoff",
                    "market_at",
                ]
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Ineligible versions were published after kickoff or lack a reliable kickoff timestamp. They never enter the headline record."
        )
        selected = display[display.game_id.eq(game_id)].iloc[0]
        events = PredictionLedger(root).result_events(selected.run_id)
        st.json(
            [
                {**event, "results": [r for r in event["results"] if str(r["game_id"]) == game_id]}
                for event in events
                if any(str(r["game_id"]) == game_id for r in event["results"])
            ],
            expanded=False,
        )


def _player_results(root, *, season, through, policy, prefix):
    _section(
        "Player projection tracker", "Frozen projections. Recorded lines. Published player stats."
    )
    rows = player_forecast_rows(root, policy=policy)
    if rows.empty:
        _empty(
            "A fresh start for the player record",
            "Earlier releases did not archive player projections. Tracking starts with the next weekly NFL release: projections and available consensus lines are saved before kickoff, then graded when player stats arrive. Historical records will not be reconstructed from today's model.",
        )
        return
    rows = rows[rows.season.eq(season) & rows.week.le(through)]
    if rows.empty:
        _empty(
            "No archived player projections in this selection",
            "Choose a later week or another season to see the player record.",
        )
        return
    controls = st.columns(2)
    category = controls[0].selectbox(
        "Prop category",
        list(PROP_TITLES),
        format_func=PROP_TITLES.get,
        key=f"{prefix}_prop_category",
    )
    team = controls[1].selectbox(
        "Player team", ["All teams", *sorted(rows.team.unique())], key=f"{prefix}_prop_team"
    )
    rows = rows[rows.category.eq(category)]
    if team != "All teams":
        rows = rows[rows.team.eq(team)]
    if rows.empty:
        _empty("No projections match these filters", "Try another category or team.")
        return
    record = pick_record(rows.result)
    errors = rows.error.dropna()
    cards = _record_html("Over / under record", record, pushes=True)
    cards += _stat_html(
        "Projection error",
        _number(errors.mean()),
        f"{'Receptions' if category == 'receptions' else 'Yards'} · mean absolute error",
        footer=f"{len(errors)} settled projections",
    )
    cards += _stat_html(
        "Line coverage",
        f"{int(rows.market_line.notna().sum())}/{len(rows)}",
        "Projections with a valid frozen line",
        footer=f"{int(rows.status.ne('final').sum())} awaiting kickoff or player stats",
    )
    _html(f'<div class="grid-results-summary">{cards}</div>')
    st.caption(
        "One projection per player, game, and category under the selected forecast policy. Only fresh lines captured with that release count. Missing player stats stay unresolved; no appearance is assumed to be zero. These are statistical over/under records, not sportsbook-specific settlements or returns."
    )
    players = player_breakdown(rows)
    if not players.empty:
        _section(
            "Player report cards",
            "Accuracy and over / under records within this category. Sample sizes are shown for every player.",
        )
        highlights = []
        for player in players.head(6).to_dict("records"):
            record_label = (
                f"{player['wins']}–{player['losses']}–{player['pushes']}"
                if player["decisions"] or player["pushes"]
                else "No graded lines"
            )
            highlights.append(
                '<div class="grid-results-wagon">'
                f'<div class="grid-results-eyebrow">{html_text(player["team"])} · {player["games"]} games</div>'
                f'<div class="grid-results-wagon-team">{html_text(player["player"])}</div>'
                f'<div class="grid-results-stat-meta">{_number(player["mae"])} average error · {html_text(PROP_TITLES[category])}<br>'
                f"{html_text(record_label)} · {_percent(player['rate'])} hit rate</div></div>"
            )
        _html('<div class="grid-results-wagons">' + "".join(highlights) + "</div>")
    player_names = rows.drop_duplicates("player_id").set_index("player_id").player_name.to_dict()
    player_id = st.selectbox(
        "Player",
        ["All players", *sorted(player_names, key=player_names.get)],
        format_func=lambda value: player_names.get(value, value),
        key=f"{prefix}_prop_player",
    )
    if player_id != "All players":
        rows = rows[rows.player_id.eq(player_id)]
    _section("Projection log", "What was projected before kickoff, alongside the recorded outcome.")
    display = rows.sort_values(["week", "player_name"], ascending=[False, True]).copy()
    display["Lean"] = [
        "Over"
        if pick_side(projection, line) > 0
        else "Under"
        if pick_side(projection, line) < 0
        else "—"
        for projection, line in zip(display.projection, display.market_line, strict=True)
    ]
    table = display[
        [
            "week",
            "player_name",
            "team",
            "opponent",
            "projection",
            "market_line",
            "Lean",
            "actual",
            "result",
            "error",
            "status",
        ]
    ].rename(
        columns={
            "week": "Week",
            "player_name": "Player",
            "team": "Team",
            "opponent": "Opponent",
            "projection": "GRIDLINE",
            "market_line": "Frozen line",
            "actual": "Actual",
            "result": "Result",
            "error": "Error",
            "status": "Status",
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            name: st.column_config.NumberColumn(format="%.1f")
            for name in ("GRIDLINE", "Frozen line", "Actual", "Error")
        },
    )
    st.download_button(
        "Download player results",
        display.to_csv(index=False),
        file_name=f"gridline-nfl-{season}-{category}-results.csv",
        mime="text/csv",
        key=f"{prefix}_props_download",
    )


def _diagnostics(rows, *, source):
    stats = summarize(rows, source=source)
    columns = st.columns(3)
    columns[0].metric("Winner Brier", _number(stats["brier"], 3))
    columns[1].metric("Margin RMSE", _number(stats["rmse"]))
    columns[2].metric("80% interval coverage", _percent(stats["coverage"]))
    st.caption(
        f"Probability sample: {stats['winner_games']} games. Interval sample: {stats['interval_games']} games; average width {_number(stats['interval_width'])} points. Published market-calibrated margins do not inherit independent-model intervals."
    )
    _weekly_chart(rows, target="margin")


def _trend_records(rows: pd.DataFrame, *, target: str, cumulative: bool) -> list[dict]:
    records = []
    for week in sorted(rows.week.unique()):
        group = rows[rows.week.le(week)] if cumulative else rows[rows.week.eq(week)]
        group = group[group.status.eq("final")].dropna(
            subset=[
                f"market_{target}",
                f"actual_{target}",
                f"published_{target}",
                f"independent_{target}",
            ]
        )
        for source in ("published", "independent", "market"):
            if not group.empty:
                records.append(
                    dict(
                        week=int(week),
                        source=source.title(),
                        mae=float(
                            (group[f"{source}_{target}"] - group[f"actual_{target}"]).abs().mean()
                        ),
                        games=len(group),
                    )
                )
    return records


def _weekly_chart(rows: pd.DataFrame, *, target: str) -> None:
    records = _trend_records(rows, target=target, cumulative=False)
    if records:
        st.caption(
            "Weekly error: published, independent, and market on the identical matched sample."
        )
        chart = (
            alt.Chart(pd.DataFrame(records))
            .mark_line(point=True)
            .encode(
                x=alt.X("week:O", title="Week"),
                y=alt.Y("mae:Q", title="MAE (points)", scale=alt.Scale(zero=True)),
                color=alt.Color(
                    "source:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["Published", "Independent", "Market"],
                        range=["#ff6b35", "#7888b5", "#334155"],
                    ),
                ),
                tooltip=["source:N", "week:O", alt.Tooltip("mae:Q", format=".2f"), "games:Q"],
            )
            .properties(height=220)
            .configure_view(stroke=None)
        )
        st.altair_chart(chart, use_container_width=True)
