"""Shared Results page. This module never recomputes a published forecast."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from .ledger import PredictionLedger
from .results import forecast_rows, select_forecasts, summarize


def _number(value, digits=2):
    return "—" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def render_results(root: str | Path, *, league: str) -> None:
    st.header("Season Results")
    st.caption(f"{league} · Frozen forecasts, official outcomes, and matched comparisons.")
    versions = forecast_rows(root)
    if versions.empty:
        st.info("No published forecasts are available yet.")
        return
    prefix = f"results_{league}"
    context = st.columns(3)
    season = context[0].selectbox(
        "Season", sorted(versions.season.unique(), reverse=True), key=f"{prefix}_season"
    )
    versions = versions[versions.season.eq(season)]
    through = context[1].selectbox(
        "Through week", sorted(versions.week.unique(), reverse=True), key=f"{prefix}_week"
    )
    policy = context[2].selectbox(
        "Forecast selection",
        ["First published", "Latest at least 60m before kickoff"],
        key=f"{prefix}_policy",
    )
    versions = versions[versions.week.le(through)]
    rows = select_forecasts(versions, policy="first" if policy == "First published" else "horizon")
    controls = st.columns(3)
    source = (
        controls[0]
        .selectbox("Forecast source", ["Published", "Independent"], key=f"{prefix}_source")
        .lower()
    )
    target = controls[1].selectbox("Target", ["Margin", "Total"], key=f"{prefix}_target").lower()
    matched_only = controls[2].checkbox("Market-matched games only", key=f"{prefix}_matched")
    if rows.empty:
        st.info(
            "No pregame forecasts qualify for this selection. Choose First published to inspect earlier batches."
        )
        return
    counts = rows.status.value_counts()
    st.caption(
        f"{len(rows)} unique forecasted games · {int(counts.get('final', 0))} final · "
        f"{int(counts.get('scheduled', 0))} scheduled · {int(counts.get('awaiting result', 0))} awaiting results · "
        f"{int(counts.get('cancelled', 0))} cancelled · {int(counts.get('postponed', 0))} postponed"
    )
    st.caption(
        "Coverage is among recorded forecasts, not the entire league schedule. One version per game; ties excluded from winner metrics."
    )
    if rows.scored_at.notna().any():
        st.caption(f"Latest settlement: {rows.scored_at.dropna().max()}")
    display = rows.dropna(subset=[f"market_{target}"]) if matched_only else rows
    stats = summarize(display, source=source, target=target)
    cards = st.columns(4)
    cards[0].metric(f"{target.title()} MAE · {stats['games']} games", _number(stats["mae"]))
    cards[1].metric(
        f"Error vs market · {stats['matched_games']} matched", _number(stats["difference"])
    )
    cards[2].metric("Winner Brier · margin", _number(stats["brier"], 3))
    cards[3].metric(
        "80% interval coverage", "—" if stats["coverage"] is None else f"{stats['coverage']:.1%}"
    )
    st.caption(
        f"Errors in points; lower is better. Matched model MAE {_number(stats['matched_model_mae'])} / "
        f"market {_number(stats['market_mae'])}. Winner sample: {stats['winner_games']}. "
        f"Interval sample: {stats['interval_games']} · mean width {_number(stats['interval_width'])} points."
    )
    if stats["difference_low"] is not None:
        st.caption(
            f"95% week-block bootstrap range for error difference: {stats['difference_low']:+.2f} to {stats['difference_high']:+.2f} points."
        )
    elif stats["matched_games"]:
        st.caption(
            "Fewer than four matched weeks: an error-difference interval is not estimated yet."
        )
    if not stats["games"]:
        st.info(
            "No settled forecasts for this selection. Accuracy will appear after official results are recorded."
        )
    else:
        _weekly_chart(display, target=target)
        with st.expander("Probability and interval diagnostics"):
            st.write(
                {
                    "RMSE": stats["rmse"],
                    "bias": stats["bias"],
                    "winner_accuracy": stats["winner_accuracy"],
                    "winner_log_loss": stats["log_loss"],
                }
            )
            decisive = display[display.status.eq("final") & display.actual_margin.ne(0)].dropna(
                subset=["actual_margin", f"{source}_probability"]
            )
            if not decisive.empty:
                calibration = decisive.assign(
                    probability=decisive[f"{source}_probability"],
                    outcome=decisive.actual_margin.gt(0).astype(int),
                )
                calibration["bin"] = pd.cut(
                    calibration.probability, [0, 0.2, 0.4, 0.6, 0.8, 1], include_lowest=True
                )
                table = (
                    calibration.groupby("bin", observed=True)
                    .agg(
                        games=("outcome", "size"),
                        forecast_probability=("probability", "mean"),
                        observed_home_win_rate=("outcome", "mean"),
                    )
                    .reset_index()
                )
                table["bin"] = table["bin"].astype(str)
                st.dataframe(table, hide_index=True, width="stretch")
            st.caption(
                "Published market-calibrated margins do not inherit independent-model interval coverage. Historical Gaussian intervals remain estimates, not a guarantee."
            )
    st.subheader("Game ledger")
    ledger = display.copy()
    ledger["matchup"] = ledger.away_team + " @ " + ledger.home_team
    ledger["error"] = (ledger[f"{source}_{target}"] - ledger[f"actual_{target}"]).abs()
    columns = [
        "week",
        "matchup",
        "status",
        "published_at",
        f"{source}_{target}",
        f"actual_{target}",
        "error",
        f"market_{target}",
        "revision",
    ]
    st.caption(
        "Margin = home points minus away points. Missing values mean unavailable, never zero."
    )
    st.dataframe(ledger[columns], hide_index=True, width="stretch")
    st.download_button(
        "Download selected results",
        ledger[columns].to_csv(index=False),
        file_name=f"gridline-{league}-{season}-{target}.csv",
        mime="text/csv",
        key=f"{prefix}_download",
    )
    if not ledger.empty:
        game_id = st.selectbox(
            "Inspect frozen game",
            ledger.game_id.tolist(),
            format_func=lambda value: ledger.set_index("game_id").loc[value, "matchup"],
            key=f"{prefix}_game",
        )
        with st.expander("Forecast versions and settlement history", expanded=True):
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
                "Versions marked ineligible were published after kickoff or have no reliable kickoff timestamp and are excluded from headline metrics."
            )
            selected = ledger[ledger.game_id.eq(game_id)].iloc[0]
            events = PredictionLedger(root).result_events(selected.run_id)
            st.json(
                [
                    {
                        **event,
                        "results": [r for r in event["results"] if str(r["game_id"]) == game_id],
                    }
                    for event in events
                    if any(str(r["game_id"]) == game_id for r in event["results"])
                ],
                expanded=False,
            )


def _trend_records(rows: pd.DataFrame, *, target: str, cumulative: bool) -> list[dict]:
    records = []
    for week in sorted(rows.week.unique()):
        group = rows[rows.week.le(week)] if cumulative else rows[rows.week.eq(week)]
        for source in ("published", "independent"):
            result = summarize(group, source=source, target=target)
            if result["games"]:
                records.append(
                    dict(
                        week=int(week),
                        source=source.title(),
                        mae=result["mae"],
                        games=result["games"],
                    )
                )
        matched = group[group.status.eq("final")].dropna(
            subset=[f"market_{target}", f"actual_{target}"]
        )
        if not matched.empty:
            records.append(
                dict(
                    week=int(week),
                    source="Market",
                    mae=float(
                        (matched[f"market_{target}"] - matched[f"actual_{target}"]).abs().mean()
                    ),
                    games=len(matched),
                )
            )
    return records


def _weekly_chart(rows: pd.DataFrame, *, target: str) -> None:
    st.subheader("Forecast error over the season")
    st.caption(
        "Weekly shows each slate; Season to date shows the running result through that week. Use Market-matched games only for a like-for-like trend."
    )
    weekly_tab, cumulative_tab = st.tabs(["Weekly", "Season to date"])
    for tab, cumulative in ((weekly_tab, False), (cumulative_tab, True)):
        records = _trend_records(rows, target=target, cumulative=cumulative)
        if not records:
            continue
        chart = (
            alt.Chart(pd.DataFrame(records))
            .mark_line(point=True)
            .encode(
                x=alt.X("week:O", title="Week"),
                y=alt.Y("mae:Q", title="MAE (points)", scale=alt.Scale(zero=True)),
                color=alt.Color(
                    "source:N",
                    title="Forecast",
                    scale=alt.Scale(
                        domain=["Published", "Independent", "Market"],
                        range=["#c94b19", "#5268ad", "#64748b"],
                    ),
                ),
                strokeDash="source:N",
                tooltip=["source:N", "week:O", alt.Tooltip("mae:Q", format=".2f"), "games:Q"],
            )
        )
        tab.altair_chart(chart, use_container_width=True)
