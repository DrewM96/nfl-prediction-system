#!/usr/bin/env python3
"""Run Streamlit and the backend market poller in one supervised web dyno."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from threading import Event

from nfl_prediction.config import PROJECT_ROOT


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run(stop: Event) -> int:
    web = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            f"--server.port={os.environ.get('PORT', '8501')}",
            "--server.address=0.0.0.0",
        ],
        cwd=PROJECT_ROOT,
    )
    worker = None
    next_start = 0.0
    enabled = os.environ.get("GRIDLINE_MARKET_AUTOSTART", "1") != "0"
    try:
        while not stop.is_set():
            code = web.poll()
            if code is not None:
                return code
            if worker is not None and worker.poll() is not None:
                print("Market worker stopped; restarting in 60 seconds", flush=True)
                worker = None
                next_start = time.monotonic() + 60
            if enabled and worker is None and time.monotonic() >= next_start:
                try:
                    worker = subprocess.Popen(
                        [sys.executable, "-u", "current_market_update.py", "--watch"],
                        cwd=PROJECT_ROOT,
                    )
                except OSError:
                    print("Market worker could not start; retrying in 60 seconds", flush=True)
                    next_start = time.monotonic() + 60
            stop.wait(1)
        return 0
    finally:
        stop_process(worker)
        stop_process(web)


def main() -> int:
    stop = Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    return run(stop)


if __name__ == "__main__":
    raise SystemExit(main())
