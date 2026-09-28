"""Escaped market card presentation, independent of Streamlit and model code."""

from __future__ import annotations

from html import escape
from typing import Any


def public_action_html(game: dict[str, Any], context: dict[str, Any]) -> str:
    """Equal-weight book averages, with complete side pairs per metric."""
    panels = []
    titles = {"draftkings": "DraftKings", "circa": "Circa"}
    books = context.get("splits", {})
    for kind, title in (
        ("spread", "Spread"),
        ("moneyline", "Moneyline"),
        ("total", "Over / Under"),
    ):
        sides = ("over", "under") if kind == "total" else ("away", "home")
        contributors = {
            field: {
                key: book
                for key, book in books.items()
                if all(
                    book.get("markets", {}).get(kind, {}).get(side, {}).get(field) is not None
                    for side in sides
                )
            }
            for field in ("ticket_pct", "handle_pct")
        }
        used = set().union(*contributors.values())
        rows = []
        for side in sides:
            cells = []
            for field, sources in contributors.items():
                values = [book["markets"][kind][side][field] for book in sources.values()]
                label = f"{sum(values) / len(values):.1f}%" if values else "—"
                attribution = ", ".join(titles.get(key, key) for key in sorted(sources))
                cells.append(f'<td title="{escape(attribution or "Unavailable")}">{label}</td>')
            team = escape(str(game.get(f"{side}_team", side.title())))
            rows.append(f'<tr><th scope="row">{team}</th>{"".join(cells)}</tr>')
        stale = (
            '<span class="grid-split-stale">Includes stale data</span>'
            if any(books[key].get("stale") for key in used)
            else ""
        )
        disagreement = any(
            any(
                book["markets"][kind][sides[0]][field] > 50
                for book in sources.values()
                if not book.get("stale")
            )
            and any(
                book["markets"][kind][sides[0]][field] < 50
                for book in sources.values()
                if not book.get("stale")
            )
            for field, sources in contributors.items()
        )
        if disagreement:
            stale += '<span class="grid-split-stale">Books disagree</span>'
        count = len(used)
        coverage = (
            f"Average across {count} books" if count > 1 else "1 book" if count else "Unavailable"
        )
        headings = "".join(
            f'<th scope="col">{label} <span class="grid-split-count">({len(contributors[field])})</span></th>'
            for field, label in (("ticket_pct", "Tickets"), ("handle_pct", "Handle"))
        )
        panels.append(
            '<section class="grid-split-book">'
            f'<div class="grid-split-source">{title}{stale}</div>'
            f'<div class="grid-splits-empty">{coverage}</div>'
            '<table class="grid-split-table"><thead><tr>'
            f'<th scope="col">{"Side" if kind == "total" else "Team"}</th>{headings}</tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></section>"
        )
    return '<div class="grid-public-splits">' + "".join(panels) + "</div>"


def market_context_html(game: dict[str, Any], context: dict[str, Any]) -> str:
    def text(value: Any) -> str:
        return escape(str(value))

    def spread(value: float | None) -> str:
        return "Unavailable" if value is None else f"{text(game['home_team'])} {value:+.2f}"

    def points(value: float | None) -> str:
        return "Unavailable" if value is None else f"{value:+.2f} pts"

    def pct(value: float | None) -> str:
        return "—" if value is None else f"{value:g}%"

    frozen = game.get("market_consensus") or {}
    line = (frozen.get("spread") or {}).get("home_spread")
    if line is None:
        margin = (frozen.get("spread") or {}).get("market_home_margin")
        line = -float(margin) if margin is not None else None
    projection = game.get("predicted_home_margin")
    model_line = -float(projection) if projection is not None else None
    status = context.get("status", "unavailable")
    current_label = (
        "Current market"
        if status == "fresh"
        else "Last cached market — STALE"
        if status == "stale"
        else "Current market unavailable"
    )
    edge_label = "Current model edge (home)" if status == "fresh" else "Edge vs cached line (home)"
    missing_books = context.get("missing_books", [])
    missing_label = (
        "Books missing from latest poll: " + ", ".join(missing_books) if missing_books else ""
    )
    sections = [
        f"<div><b>GRIDLINE · Frozen projection</b><br>{spread(model_line)}<br><small>Forecast: {text(game.get('forecast_at') or 'timestamp unavailable')}</small></div>",
        f"<div><b>MARKET</b><br>Market at forecast: {spread(line)}<br><small>{text(frozen.get('provider') or 'No snapshot')} · {text(frozen.get('snapshot_at') or 'unavailable')}</small><br>{current_label}: {spread(context.get('home_spread'))}<br>Movement: {points(context.get('movement'))}<br>{edge_label}: {points(context.get('current_home_edge'))}<br><small>Updated: {text(context.get('source_timestamp') or 'unknown')}<br>Fetched: {text(context.get('captured_at') or 'unavailable')} · {text(context.get('provider') or 'Owls Insight')}<br>{text(context.get('reason') or '')}<br>{text(missing_label)}</small></div>",
    ]
    public = []
    for key, book in sorted(context.get("splits", {}).items()):
        rows = []
        for kind, market in book["markets"].items():
            sides = ("over", "under") if kind == "total" else ("home", "away")
            for side in sides:
                row = market[side]

                label = game.get(f"{side}_team", side.title())
                delta = row["handle_minus_ticket"]
                delta_label = "—" if delta is None else f"{delta:+.1f} pp"
                rows.append(
                    f"<tr><td>{text(kind)} · {text(label)}</td><td>{pct(row['ticket_pct'])}</td><td>{pct(row['handle_pct'])}</td><td>{delta_label}</td></tr>"
                )
        public.append(
            f"<div><b>{text(key)}</b> · {'Stale' if book.get('stale') else 'Available'}<br><small>{text(book.get('source_timestamp') or 'Timestamp unknown')}</small><table><thead><tr><th>Market / side</th><th>Tickets</th><th>Handle</th><th>Difference (pp)</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        )
    sections.append(
        f"<div><b>PUBLIC ACTION</b><p>Equal weight per book, not volume-weighted. Counts beside Tickets and Handle show contributing books; missing pairs are excluded. Cached stale figures are included only with a visible warning. Books may report different lines and timestamps.</p>{''.join(public) or 'Splits unavailable'}<small>{text(context.get('splits_error') or '')}</small></div>"
    )
    return (
        public_action_html(game, context)
        + '<details class="grid-market-details"><summary>Market details</summary>'
        + '<div class="grid-market-context">'
        + "".join(sections)
        + "</div></details>"
    )
