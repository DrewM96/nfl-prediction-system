"""Publish aggregate Owls NFL consensus without exposing private book data."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import MARKET_CONSENSUS_PATH, get_season_context
from .current_market import MarketStore, poll_market, public_consensus
from .io import atomic_write_json
from .odds import _game_kickoff
from .owls import OwlsClient, OwlsError


def refresh_owls_consensus(
    slate: list[dict[str, Any]] | None = None,
    *,
    store: MarketStore | None = None,
    client: OwlsClient | None = None,
    as_of: datetime | None = None,
    output: Path = MARKET_CONSENSUS_PATH,
) -> dict[str, Any]:
    now = as_of or datetime.now(UTC)
    if slate is None:
        import nflreadpy as nfl

        # Match the live upcoming schedule, even when last week's release is all final.
        schedule = nfl.load_schedules([get_season_context(now).prediction_season]).to_dicts()
        slate = [g for g in schedule if _game_kickoff(g) and _game_kickoff(g) > now]
    board = poll_market("nfl", slate, store=store, client=client, now=now)
    if board.get("odds_error"):
        raise OwlsError(board["odds_error"])
    consensus = public_consensus(board, slate, as_of=now)
    if not consensus["games"]:
        raise OwlsError("No fresh, timestamp-eligible Owls NFL consensus available")
    atomic_write_json(output, consensus)
    return consensus
