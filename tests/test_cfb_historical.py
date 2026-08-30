from __future__ import annotations

from datetime import timedelta
from typing import Any

from cfb_prediction.historical import load_historical_data


class CacheSpyClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def get(
        self,
        endpoint: str,
        *,
        params: dict[str, Any],
        refresh: bool,
        max_age: timedelta,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "endpoint": endpoint,
                "params": params,
                "refresh": refresh,
                "max_age": max_age,
            }
        )
        return []


def test_historical_loader_forwards_long_lived_cache_age() -> None:
    client = CacheSpyClient()
    cache_age = timedelta(days=3650)

    load_historical_data(client, [2025], max_age=cache_age)

    assert len(client.calls) == 7
    assert {call["params"]["year"] for call in client.calls} == {2025}
    assert all(call["max_age"] == cache_age for call in client.calls)
    assert all(call["refresh"] is False for call in client.calls)
