from __future__ import annotations

from datetime import date, datetime


def current_cfb_season(as_of: date | datetime | None = None) -> int:
    """Return the season that owns the current CFB operational calendar.

    Bowl games played from January through June belong to the prior fall season.
    July starts the next preseason window so scheduled automation can collect and
    publish new-season inputs before Week 0.
    """
    observed = as_of or datetime.now().astimezone()
    return observed.year if observed.month >= 7 else observed.year - 1
