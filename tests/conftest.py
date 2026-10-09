from __future__ import annotations

import pytest
from streamlit.testing.v1.element_tree import ButtonGroup


@pytest.fixture(autouse=True)
def single_selection_pills_apptest(monkeypatch):
    """Normalize scalar pill values when AppTest serializes widget state.

    Streamlit 1.49 assumes every button group stores a list, but single-selection
    pills store a scalar (or None). Apply the adapter to all app interaction tests.
    """

    def pill_indices(widget):
        value = widget.value
        values = [value] if isinstance(value, str) else (value or [])
        return [widget.options.index(widget.format_func(v)) for v in values]

    monkeypatch.setattr(ButtonGroup, "indices", property(pill_indices))
