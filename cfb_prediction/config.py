from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CFB_DATA_DIR = PROJECT_ROOT / "data" / "cfb"
CFB_CACHE_DIR = CFB_DATA_DIR / "cache"
CFB_FOUNDATION_PATH = CFB_DATA_DIR / "foundation.json"
CFB_HISTORICAL_BENCHMARK_PATH = CFB_DATA_DIR / "historical_benchmark.json"
CFBD_API_BASE_URL = "https://api.collegefootballdata.com"


class CFBDConfigurationError(RuntimeError):
    """Raised when local CollegeFootballData configuration is incomplete."""


def load_cfbd_api_key(env_path: Path | None = None) -> str:
    """Load the CFBD key from the environment or an ignored local .env file."""
    configured = os.environ.get("CFBD_API_KEY", "").strip()
    if configured:
        return configured

    path = env_path or PROJECT_ROOT / ".env"
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "CFBD_API_KEY" and value.strip():
                return value.strip().strip('"').strip("'")

    raise CFBDConfigurationError(
        "CFBD_API_KEY is unavailable. Add it to the environment or the ignored local .env file."
    )
