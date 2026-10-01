"""Read-only current signals and the prospective first-observation record."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from .current_market import MarketStore
from .money_signals import first_signal_records, qualifying_signals, signal_label
from .results import forecast_rows
from .results_tracker import pick_record

PICKS_TITLE = "Blood-in-the-water Picks of The Week"
PICKS_RULE = (
    "Tickets favor one side (>50%), while at least 65% of handle favors the opposite side. "
    "GRIDLINE's published forecast must favor that handle side against the fresh spread or total. "
    "Percentages are equal-weight averages across at least two fresh sportsbooks reporting both metrics. "
    "Odds must be no older than 15 minutes; splits no older than one hour. "
    "Handle measures wager volume, not the identity of bettors."
)


def pick_explanation(signal: dict) -> str:
    public = signal.get(f"{signal['public_side']}_team", signal["public_side"].title())
    return (
        f"Why this pick: {signal['ticket_pct']:.2f}% of tickets favor {public}, "
        f"but {signal['handle_pct']:.2f}% of handle backs {signal_label(signal)}. "
        f"GRIDLINE projects a {'home margin' if signal['market'] == 'spread' else 'total'} "
        f"of {signal['projection']:.2f}, a {signal['model_edge']:.2f}-point lean toward the handle side. "
        f"Split sources: {', '.join(sorted(signal['split_books']))}. {PICKS_RULE}"
    )


def render_pick_badge(signal: dict) -> None:
    st.markdown(
        f'<div class="grid-muted" title="{escape(pick_explanation(signal), quote=True)}">'
        f"<b>Blood-in-the-water pick: {escape(signal_label(signal))}</b> ⓘ</div>",
        unsafe_allow_html=True,
    )


@st.fragment(run_every="60s")
def render_weekly_picks(forecasts: list[dict], *, sport: str) -> None:
    st.subheader(PICKS_TITLE, help=PICKS_RULE)
    st.caption("Qualifying now. Open Picks to review recorded observations and results.")
    board = MarketStore().read(sport)
    signals = [signal for game in forecasts for signal in qualifying_signals(game, board)]
    if not signals:
        st.caption("No qualifying picks in the available fresh data.")
        return
    columns = st.columns(2)
    for index, signal in enumerate(signals):
        with columns[index % 2]:
            st.metric(
                f"{signal['away_team']} @ {signal['home_team']} · {signal['market'].title()}",
                signal_label(signal),
                help=pick_explanation(signal),
            )
            st.caption(
                f"Public tickets {signal['ticket_pct']:.1f}% · Opposite handle {signal['handle_pct']:.1f}% "
                f"· {len(signal['split_books'])} books"
            )


def _table(signals: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Observed (UTC)": row["observed_at"],
                "Game": f"{row['away_team']} @ {row['home_team']}",
                "Market": row["market"].title(),
                "Handle + model side": signal_label(row),
                "Public side": row.get(f"{row['public_side']}_team", row["public_side"].title()),
                "Public tickets %": round(row["ticket_pct"], 2),
                "Opposite handle %": round(row["handle_pct"], 2),
                "GRIDLINE margin / total": row["projection"],
                "Model edge (pts)": round(row["model_edge"], 2),
                "Split books": len(row["split_books"]),
                "Result": row.get("result", "Live"),
            }
            for row in signals
        ]
    )


@st.fragment(run_every="60s")
def render_signal_tracker(root: str | Path, *, sport: str, forecasts: list[dict]) -> None:
    st.header(PICKS_TITLE, help=PICKS_RULE)
    st.caption(
        "Spread and totals: tickets >50% on one side, handle ≥65% on the other, "
        "and the published GRIDLINE model favors the handle side against the live line. "
        "Equal weight across at least two fresh books reporting both metrics. "
        "Signals use fresh inputs only; the general public-action panels may include labeled stale data."
    )
    store = MarketStore()
    board = store.read(sport)
    active = [signal for game in forecasts for signal in qualifying_signals(game, board)]
    st.subheader("Qualifying now")
    if active:
        st.dataframe(_table(active), hide_index=True, use_container_width=True)
    else:
        st.info("No qualifying signals in the available fresh data.")
    if board.get("odds_error") or board.get("splits_error"):
        st.caption(board.get("odds_error") or board.get("splits_error"))
    history = store.read_signals(sport)
    st.subheader("First-signal record")
    st.caption(
        "One entry per game and market, fixed at the first qualifying pregame observation. "
        "Later observations and side changes do not add picks to this record. "
        "Final results use the recorded line; pushes and voids are excluded from hit rate. "
        "Logging runs in the market worker even when this page is closed. No historical backfill."
    )
    if history["error"]:
        st.warning(history["error"])
        return
    observations = history["observations"]
    if not observations:
        st.info("The record starts with the first qualifying observation after deployment.")
        return
    rows = first_signal_records(observations, forecast_rows(root))
    season = st.selectbox(
        "Season", sorted(rows.season.unique(), reverse=True), key=f"signals_{sport}_season"
    )
    rows = rows[rows.season.eq(season)]
    market = st.selectbox("Market", ["Both", "Spread", "Total"], key=f"signals_{sport}_market")
    if market != "Both":
        rows = rows[rows.market.eq(market.lower())]
    record = pick_record(rows.result)
    columns = st.columns(3)
    columns[0].metric("Record (W–L–P)", f"{record['wins']}–{record['losses']}–{record['pushes']}")
    columns[1].metric("Hit rate", f"{record['rate']:.1%}" if record["rate"] is not None else "—")
    columns[2].metric("Pending", record["pending"])
    if rows.empty:
        st.info("No recorded signals for these filters.")
        return
    st.dataframe(
        _table(rows.sort_values("observed_at", ascending=False).to_dict("records")),
        hide_index=True,
        use_container_width=True,
    )
    selected_keys = set(zip(rows.game_id, rows.market, strict=True))
    audit = [
        r
        for r in observations
        if (r["game_id"], r["market"]) in selected_keys and r["season"] == season
    ]
    with st.expander(f"Recorded observations ({len(audit)})"):
        st.caption(
            "Per-book inputs and forecast provenance are preserved for each qualifying poll."
        )
        observation = st.selectbox(
            "Observation",
            range(len(audit)),
            format_func=lambda i: f"{audit[i]['observed_at']} · {audit[i]['away_team']} @ {audit[i]['home_team']} · {signal_label(audit[i])}",
            key=f"signals_{sport}_observation",
        )
        st.json(audit[observation])
