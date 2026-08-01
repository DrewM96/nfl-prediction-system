from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request

import pytest

from cfb_prediction.client import CFBDClient
from cfb_prediction.config import CFBDConfigurationError, load_cfbd_api_key


class FakeResponse(io.BytesIO):
    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def test_client_authenticates_and_reuses_disk_cache(tmp_path: Path) -> None:
    requests: list[Request] = []

    def opener(request: Request, *, timeout: int) -> FakeResponse:
        assert timeout == 30
        requests.append(request)
        return FakeResponse(json.dumps([{"id": 1, "school": "Example"}]).encode())

    client = CFBDClient(
        "private-key",
        cache_root=tmp_path,
        opener=opener,
        now=lambda: datetime(2026, 8, 1, tzinfo=UTC),
    )
    first = client.get("/teams/fbs", params={"year": 2026})
    second = client.get("/teams/fbs", params={"year": 2026})

    assert first == second
    assert len(requests) == 1
    assert requests[0].get_header("Authorization") == "Bearer private-key"
    cache_text = next(tmp_path.glob("*.json")).read_text(encoding="utf-8")
    assert "private-key" not in cache_text


def test_local_env_key_is_loaded_without_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("CFBD_API_KEY=local-secret\n", encoding="utf-8")
    assert load_cfbd_api_key(env_path) == "local-secret"


def test_missing_key_has_actionable_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    with pytest.raises(CFBDConfigurationError, match="CFBD_API_KEY is unavailable"):
        load_cfbd_api_key(tmp_path / ".env")
