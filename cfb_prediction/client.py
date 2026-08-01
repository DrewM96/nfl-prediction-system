from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import CFB_CACHE_DIR, CFBD_API_BASE_URL, load_cfbd_api_key


class CFBDApiError(RuntimeError):
    """Raised for a safe, sanitized CollegeFootballData request failure."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


class CFBDClient:
    """Small authenticated CFBD REST client with a quota-preserving disk cache."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = CFBD_API_BASE_URL,
        cache_root: Path = CFB_CACHE_DIR,
        opener: Callable[..., Any] = urlopen,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        if not api_key.strip():
            raise ValueError("A non-empty CFBD API key is required.")
        self._api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.cache_root = cache_root
        self._opener = opener
        self._now = now

    @classmethod
    def from_environment(cls, **kwargs: Any) -> CFBDClient:
        return cls(load_cfbd_api_key(), **kwargs)

    def _cache_path(self, endpoint: str, params: dict[str, Any]) -> Path:
        identity = json.dumps(
            {"endpoint": endpoint, "params": sorted(params.items())},
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        stem = endpoint.strip("/").replace("/", "_") or "root"
        return self.cache_root / f"{stem}_{digest}.json"

    def _read_cache(self, path: Path, max_age: timedelta) -> Any | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            fetched_at = datetime.fromisoformat(payload["fetched_at"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=UTC)
        if self._now() - fetched_at > max_age:
            return None
        return payload.get("data")

    def _write_cache(
        self,
        path: Path,
        endpoint: str,
        params: dict[str, Any],
        data: Any,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fetched_at": self._now().isoformat(),
            "endpoint": endpoint,
            "params": params,
            "data": data,
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        temporary.replace(path)

    def get(
        self,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        max_age: timedelta = timedelta(hours=6),
        refresh: bool = False,
    ) -> Any:
        normalized_endpoint = "/" + endpoint.strip("/")
        query = {key: value for key, value in (params or {}).items() if value is not None}
        cache_path = self._cache_path(normalized_endpoint, query)
        if not refresh:
            cached = self._read_cache(cache_path, max_age)
            if cached is not None:
                return cached

        url = f"{self.base_url}{normalized_endpoint}"
        if query:
            url = f"{url}?{urlencode(query)}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": "GRIDLINE-CFB/1.0",
            },
        )
        try:
            with self._opener(request, timeout=30) as response:
                data = json.load(response)
        except HTTPError as exc:
            raise CFBDApiError(
                f"CollegeFootballData request {normalized_endpoint} failed with HTTP {exc.code}."
            ) from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise CFBDApiError(
                f"CollegeFootballData request {normalized_endpoint} failed."
            ) from exc

        self._write_cache(cache_path, normalized_endpoint, query, data)
        return data
