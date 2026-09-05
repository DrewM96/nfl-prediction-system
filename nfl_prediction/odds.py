from __future__ import annotations

import json
import os
import statistics
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from .config import MARKET_CONSENSUS_PATH, MARKET_PRIVATE_DIR
from .io import atomic_write_json, read_json
from .market import sportsbook_home_spread_to_market_margin

API_BASE_URL = "https://api.the-odds-api.com/v4"
NFL_SPORT_KEY = "americanfootball_nfl"
DEFAULT_MARKETS = ("spreads", "totals")

TEAM_NAME_TO_CODE = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}


class OddsApiError(RuntimeError):
    """A redacted provider error that is safe to print in CI logs."""


@dataclass(frozen=True)
class CreditUsage:
    remaining: int | None = None
    used: int | None = None
    last_request: int | None = None


@dataclass(frozen=True)
class OddsFetchResult:
    payload: Any
    captured_at: str
    credits: CreditUsage
    request_kind: str


def parse_timestamp(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must include a UTC offset")
    return parsed.astimezone(UTC)


def estimate_historical_credits(
    *, regions: str = "us", markets: tuple[str, ...] = DEFAULT_MARKETS, snapshots: int = 1
) -> int:
    region_count = len([item for item in regions.split(",") if item.strip()])
    return 10 * max(region_count, 1) * len(markets) * snapshots


def _header_integer(headers: Any, name: str) -> int | None:
    value = headers.get(name)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class OddsApiClient:
    def __init__(self, api_key: str | None = None, *, timeout_seconds: int = 30):
        self.api_key = api_key or os.environ.get("ODDS_API_KEY", "")
        self.timeout_seconds = timeout_seconds
        if not self.api_key:
            raise OddsApiError("ODDS_API_KEY is not configured")

    def _get(self, path: str, parameters: dict[str, str], *, request_kind: str) -> OddsFetchResult:
        query = urlencode({**parameters, "apiKey": self.api_key})
        request = Request(
            f"{API_BASE_URL}{path}?{query}",
            headers={"Accept": "application/json", "User-Agent": "nfl-prediction-system/4"},
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
                credits = CreditUsage(
                    remaining=_header_integer(response.headers, "x-requests-remaining"),
                    used=_header_integer(response.headers, "x-requests-used"),
                    last_request=_header_integer(response.headers, "x-requests-last"),
                )
        except HTTPError as exc:
            raise OddsApiError(
                f"The Odds API returned HTTP {exc.code} for {request_kind}"
            ) from None
        except (URLError, TimeoutError):
            raise OddsApiError(f"The Odds API request failed for {request_kind}") from None
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise OddsApiError(f"The Odds API returned invalid JSON for {request_kind}") from None
        return OddsFetchResult(
            payload=payload,
            captured_at=datetime.now(UTC).isoformat(),
            credits=credits,
            request_kind=request_kind,
        )

    def current_odds(
        self, *, regions: str = "us", markets: tuple[str, ...] = DEFAULT_MARKETS
    ) -> OddsFetchResult:
        return self._get(
            f"/sports/{NFL_SPORT_KEY}/odds",
            {
                "regions": regions,
                "markets": ",".join(markets),
                "oddsFormat": "american",
                "dateFormat": "iso",
            },
            request_kind="current",
        )

    def historical_odds(
        self,
        snapshot_at: str | datetime,
        *,
        regions: str = "us",
        markets: tuple[str, ...] = DEFAULT_MARKETS,
    ) -> OddsFetchResult:
        timestamp = parse_timestamp(snapshot_at).isoformat().replace("+00:00", "Z")
        return self._get(
            f"/historical/sports/{NFL_SPORT_KEY}/odds",
            {
                "regions": regions,
                "markets": ",".join(markets),
                "oddsFormat": "american",
                "dateFormat": "iso",
                "date": timestamp,
            },
            request_kind="historical",
        )


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _rounded_median(values: list[float]) -> float | None:
    return round(float(statistics.median(values)), 3) if values else None


def _market_summary(values: list[dict[str, Any]], line_key: str) -> dict[str, Any] | None:
    if not values:
        return None
    lines = [row[line_key] for row in values]
    q1 = _percentile(lines, 0.25)
    q3 = _percentile(lines, 0.75)
    result: dict[str, Any] = {
        "line": _rounded_median(lines),
        "book_count": len(values),
        "line_min": round(min(lines), 3),
        "line_max": round(max(lines), 3),
        "line_iqr": round(float(q3 - q1), 3) if q1 is not None and q3 is not None else None,
    }
    for key in ("home_price", "away_price", "over_price", "under_price"):
        prices = [row[key] for row in values if key in row]
        if prices:
            result[key] = _rounded_median(prices)
    updates = []
    for row in values:
        if not row.get("last_update"):
            continue
        try:
            updates.append(parse_timestamp(row["last_update"]).isoformat())
        except (TypeError, ValueError):
            continue
    updates.sort()
    if updates:
        result["oldest_book_update"] = updates[0]
        result["latest_book_update"] = updates[-1]
    return result


def build_consensus(
    fetch: OddsFetchResult, *, regions: str, markets: tuple[str, ...]
) -> dict[str, Any]:
    wrapper = fetch.payload if isinstance(fetch.payload, dict) else {}
    events = wrapper.get("data", []) if wrapper else fetch.payload
    snapshot_at = wrapper.get("timestamp") or fetch.captured_at
    games: list[dict[str, Any]] = []
    for event in events or []:
        home_name = str(event.get("home_team", ""))
        away_name = str(event.get("away_team", ""))
        home_code = TEAM_NAME_TO_CODE.get(home_name)
        away_code = TEAM_NAME_TO_CODE.get(away_name)
        if not home_code or not away_code:
            continue
        spread_rows: list[dict[str, Any]] = []
        total_rows: list[dict[str, Any]] = []
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                outcomes = market.get("outcomes", [])
                if market.get("key") == "spreads":
                    home = next((row for row in outcomes if row.get("name") == home_name), None)
                    away = next((row for row in outcomes if row.get("name") == away_name), None)
                    if home is not None and home.get("point") is not None:
                        row = {
                            "home_spread": float(home["point"]),
                            "last_update": market.get("last_update")
                            or bookmaker.get("last_update"),
                        }
                        if home.get("price") is not None:
                            row["home_price"] = float(home["price"])
                        if away is not None and away.get("price") is not None:
                            row["away_price"] = float(away["price"])
                        spread_rows.append(row)
                elif market.get("key") == "totals":
                    over = next((row for row in outcomes if row.get("name") == "Over"), None)
                    under = next((row for row in outcomes if row.get("name") == "Under"), None)
                    if over is not None and over.get("point") is not None:
                        row = {
                            "total": float(over["point"]),
                            "last_update": market.get("last_update")
                            or bookmaker.get("last_update"),
                        }
                        if over.get("price") is not None:
                            row["over_price"] = float(over["price"])
                        if under is not None and under.get("price") is not None:
                            row["under_price"] = float(under["price"])
                        total_rows.append(row)
        spread = _market_summary(spread_rows, "home_spread")
        if spread:
            spread["home_spread"] = spread.pop("line")
            spread["market_home_margin"] = round(
                sportsbook_home_spread_to_market_margin(spread["home_spread"]), 3
            )
        total = _market_summary(total_rows, "total")
        if total:
            total["total"] = total.pop("line")
        if spread or total:
            games.append(
                {
                    "commence_time": event.get("commence_time"),
                    "home_team": home_code,
                    "away_team": away_code,
                    "home_team_name": home_name,
                    "away_team_name": away_name,
                    "spread": spread,
                    "total": total,
                }
            )
    return {
        "schema_version": 1,
        "provider": "The Odds API",
        "sport": NFL_SPORT_KEY,
        "snapshot_at": parse_timestamp(snapshot_at).isoformat(),
        "request_kind": fetch.request_kind,
        "regions": regions,
        "markets": list(markets),
        "credits": asdict(fetch.credits),
        "games": games,
    }


class MarketSnapshotStore:
    def __init__(
        self,
        private_root: str | Path = MARKET_PRIVATE_DIR,
        consensus_path: str | Path = MARKET_CONSENSUS_PATH,
    ):
        self.private_root = Path(private_root)
        self.consensus_path = Path(consensus_path)

    def save(self, fetch: OddsFetchResult, consensus: dict[str, Any]) -> tuple[Path, Path]:
        timestamp = parse_timestamp(consensus["snapshot_at"]).strftime("%Y%m%dT%H%M%SZ")
        raw_path = self.private_root / "raw" / f"{fetch.request_kind}-{timestamp}.json"
        if raw_path.exists():
            raise FileExistsError(f"Snapshot already exists: {raw_path.name}")
        atomic_write_json(
            raw_path,
            {
                "captured_at": fetch.captured_at,
                "request_kind": fetch.request_kind,
                "credits": asdict(fetch.credits),
                "payload": fetch.payload,
            },
        )
        atomic_write_json(self.consensus_path, consensus)
        return raw_path, self.consensus_path


def _game_kickoff(game: dict[str, Any]) -> datetime | None:
    commence = game.get("commence_time") or game.get("start_date")
    if commence:
        return parse_timestamp(commence)
    gameday = game.get("gameday")
    gametime = game.get("gametime")
    if not gameday or not gametime or gametime == "TBD":
        return None
    try:
        local = datetime.fromisoformat(f"{gameday}T{gametime}").replace(
            tzinfo=ZoneInfo("America/New_York")
        )
    except ValueError:
        return None
    return local.astimezone(UTC)


def eligible_market_snapshot(
    consensus: dict[str, Any] | None,
    *,
    as_of: datetime | None = None,
    max_age: timedelta = timedelta(days=8),
) -> dict[str, Any] | None:
    """One time gate for comparisons and market-derived forecasts."""
    if not consensus or not consensus.get("snapshot_at"):
        return None
    try:
        captured = parse_timestamp(consensus["snapshot_at"])
    except (ValueError, TypeError):
        return None
    now = (as_of or datetime.now(UTC)).astimezone(UTC)
    if captured > now or now - captured > max_age:
        return None
    games = []
    for game in consensus.get("games", []):
        try:
            kickoff = _game_kickoff(game)
        except (ValueError, TypeError):
            continue
        if kickoff is not None and captured < kickoff and now < kickoff:
            games.append(game)
    return {**consensus, "games": games} if games else None


def attach_market_consensus(
    predictions: list[dict[str, Any]],
    consensus: dict[str, Any] | None,
    *,
    max_age: timedelta = timedelta(days=8),
    as_of: datetime | None = None,
) -> list[dict[str, Any]]:
    knowledge_time = (as_of or datetime.now(UTC)).astimezone(UTC)
    consensus = eligible_market_snapshot(consensus, as_of=knowledge_time, max_age=max_age)
    lookup = {
        (game.get("away_team"), game.get("home_team")): game
        for game in (consensus or {}).get("games", [])
    }
    enriched: list[dict[str, Any]] = []
    for source in predictions:
        prediction = dict(source)
        market = lookup.get((prediction.get("away_team"), prediction.get("home_team")))
        kickoff = _game_kickoff(prediction)
        market_kickoff = _game_kickoff(market) if market else None
        if market is None or (kickoff is not None and kickoff != market_kickoff):
            prediction["market_consensus"] = None
            prediction["market_informed"] = None
            prediction["market_line"] = None
            enriched.append(prediction)
            continue
        spread = market.get("spread")
        total = market.get("total")
        prediction["market_consensus"] = {
            "provider": consensus.get("provider"),
            "snapshot_at": consensus["snapshot_at"],
            "commence_time": market.get("commence_time"),
            "spread": spread,
            "total": total,
        }
        prediction["market_line"] = spread.get("home_spread") if spread else None
        if spread and total:
            margin = float(spread["market_home_margin"])
            points = float(total["total"])
            prediction["market_informed"] = {
                "home_margin": round(margin, 2),
                "total": round(points, 1),
                "home_score": round(max((points + margin) / 2.0, 0.0), 1),
                "away_score": round(max((points - margin) / 2.0, 0.0), 1),
                "method": "market consensus benchmark (not a claimed betting edge)",
            }
        else:
            prediction["market_informed"] = None
        enriched.append(prediction)
    return enriched


def load_market_consensus(path: str | Path = MARKET_CONSENSUS_PATH) -> dict[str, Any] | None:
    return read_json(path)
