"""Shared NFL/CFB forecast input panels; all interactions stay in the browser."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import streamlit as st

from .ui import html_text

STYLES = """<style>
.grid-inputs {border-top:1px solid var(--grid-border);margin-top:12px;color:var(--grid-ink);font-family:var(--font-ui)}
.grid-inputs summary {list-style:none;cursor:pointer;display:flex;align-items:center;justify-content:space-between;gap:12px;padding:16px 0;font-size:14px;font-weight:600;min-height:44px}
.grid-inputs summary::-webkit-details-marker {display:none}
.grid-inputs summary::after {content:'⌄';color:var(--grid-muted);font-size:18px}
.grid-inputs[open]>summary::after,.grid-inputs details[open]>summary::after {transform:rotate(180deg)}
.grid-input-tabs {position:relative}
.grid-input-choice {position:absolute;opacity:0;width:1px;height:1px}
.grid-input-choice+label {display:inline-block;cursor:pointer;padding:12px 0;margin-right:22px;border-bottom:2px solid transparent;color:var(--grid-muted);font-size:13px;min-height:44px}
.grid-input-choice:checked+label {color:var(--grid-orange);border-bottom-color:var(--grid-orange)}
.grid-input-choice:focus-visible+label {outline:2px solid var(--grid-orange);outline-offset:3px}
.grid-input-choice[value=margin]:checked~.grid-input-total,.grid-input-choice[value=total]:checked~.grid-input-margin {display:none}
.grid-input-target {border-top:1px solid var(--grid-border);padding:20px 0 4px}
.grid-input-title {font-size:17px;font-weight:600;letter-spacing:-.02em;margin-bottom:4px}
.grid-input-muted {font-size:12px;color:var(--grid-muted);line-height:1.5}
.grid-input-legend {display:flex;justify-content:space-between;gap:12px;font-size:11px;color:var(--grid-muted);margin:18px 0 4px}
.grid-input-driver {display:grid;grid-template-columns:minmax(0,1.2fr) minmax(80px,1fr) minmax(76px,.45fr);align-items:center;gap:14px;padding:13px 0;border-bottom:1px solid var(--grid-border)}
.grid-input-name {font-size:13px;font-weight:500;overflow-wrap:anywhere}
.grid-input-sub {font-size:11px;color:var(--grid-muted);margin-top:3px;line-height:1.5}
.grid-input-track {position:relative;height:20px}
.grid-input-track::before {content:'';position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--grid-border)}
.grid-input-bar {position:absolute;top:6px;height:8px;border-radius:2px;background:var(--grid-orange)}
.grid-input-bar-left {right:50%;background:var(--grid-muted)}
.grid-input-bar-right {left:50%}
.grid-input-effect {text-align:right;font-size:13px;font-weight:500;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.grid-input-effect small {display:block;font-size:11px;font-weight:400;color:var(--grid-muted)}
.grid-input-reconcile {display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px 18px;margin-top:20px;font-size:12px;color:var(--grid-muted);font-variant-numeric:tabular-nums}
.grid-input-final {padding-top:9px;border-top:1px solid var(--grid-border);font-size:14px;font-weight:600;color:var(--grid-ink)}
.grid-input-all {border-top:1px solid var(--grid-border);margin-top:20px}
.grid-input-all>summary {font-size:12px;font-weight:500}
.grid-input-table {border-collapse:collapse;width:100%;table-layout:fixed;font-size:12px;margin:10px 0}
.grid-input-table th,.grid-input-table td {padding:9px 5px;border-bottom:1px solid var(--grid-border);text-align:right;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
.grid-input-table th {font-weight:500;color:var(--grid-muted)}
.grid-input-table th:first-child,.grid-input-table td:first-child {text-align:left;width:56%;padding-left:0}
.grid-input-footer {border-top:1px solid var(--grid-border);padding:14px 0 4px;margin-top:12px;font-size:11px;color:var(--grid-muted);line-height:1.6}
.grid-input-empty {padding:12px 0;color:var(--grid-muted);font-size:13px}
@media(max-width:540px) {
 .grid-input-driver {grid-template-columns:minmax(0,1fr) minmax(68px,.4fr);gap:5px 10px}
 .grid-input-name {grid-column:1}
 .grid-input-effect {grid-column:2;grid-row:span 2;align-self:center}
 .grid-input-track {grid-column:1;order:3;height:10px}
 .grid-input-bar {top:1px}
 .grid-input-title {font-size:16px}
}
</style>"""

LABELS = {
    "points_for": "Points scored / game",
    "points_against": "Points allowed / game",
    "yards": "Yards / game",
    "off_epa": "Offensive EPA / play",
    "def_epa": "Defensive EPA allowed / play",
    "turnovers": "Turnovers / game",
    "win_pct": "Win rate",
    "pressure_allowed": "Pressure allowed",
    "pressure_generated": "Pressure generated",
    "rest_days": "Rest days",
    "rest_advantage": "Home rest advantage",
    "division_game": "Division matchup",
    "conference_game": "Conference matchup",
    "week": "Week",
    "home_field": "Home field enabled",
    "elo": "Elo rating",
    "elo_diff": "Home − away Elo",
    "elo_expected_margin": "Elo expected home margin",
    "margin_vs_elo": "Margin relative to Elo expectation",
    "opponent_elo": "Opponent Elo rating",
    "plays": "Plays / game",
    "off_ppa": "Offensive PPA / play",
    "def_ppa": "Defensive PPA allowed / play",
    "off_success_rate": "Offensive success rate",
    "def_success_rate": "Defensive success rate allowed",
    "off_explosiveness": "Offensive explosiveness",
    "def_explosiveness": "Defensive explosiveness allowed",
    "roster_qb_returning_delta": "Returning QB continuity delta",
    "roster_ol_continuity_delta": "Offensive line continuity delta",
    "roster_skill_continuity_delta": "Skill position continuity delta",
    "returning_ppa_z": "Returning production (standardized)",
    "returning_passing_ppa_z": "Returning passing production (standardized)",
    "returning_rushing_ppa_z": "Returning rushing production (standardized)",
    "returning_receiving_ppa_z": "Returning receiving production (standardized)",
    "talent_z": "Roster talent (standardized)",
    "recruiting_points_z": "Recruiting points (standardized)",
    "portal_net_rating_z": "Transfer portal net rating (standardized)",
}


def _parts(feature: str) -> tuple[str | None, str, str]:
    side, core = None, feature
    if feature.startswith(("away_", "home_")) and feature != "home_field":
        side, core = feature.split("_", 1)
    period = ""
    if core.endswith(("_l4", "_l6", "_l8")):
        core, period = core.rsplit("_", 1)
    return side, core, period.upper()


def _label(feature: str) -> str:
    _, core, period = _parts(feature)
    return LABELS.get(core, core.replace("_", " ").capitalize()) + (
        f" · {period}" if period else ""
    )


def _group(feature: str) -> str:
    _, core, _ = _parts(feature)
    if core.startswith(("off_epa", "off_ppa", "off_success", "off_explosiveness", "yards")):
        return "Offensive efficiency"
    if core.startswith(("def_epa", "def_ppa", "def_success", "def_explosiveness")):
        return "Defensive efficiency"
    if "pressure" in core or "sack" in core:
        return "Pressure matchup"
    if core.startswith(("points_", "win_pct", "margin_vs_elo")):
        return "Recent scoring form"
    if "turnover" in core:
        return "Turnovers"
    if "roster_" in core or core.endswith("_z"):
        return "Roster continuity & talent"
    if "elo" in core:
        return "Team & opponent strength"
    if "rest" in core:
        return "Rest advantage"
    if core == "plays":
        return "Pace"
    return "Game context"


def _value(feature: str, value: Any) -> str:
    if value is None:
        return "Unavailable"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Unavailable"
    if not math.isfinite(number):
        return "Unavailable"
    if abs(number) < 0.0005:
        number = 0.0
    _, core, _ = _parts(feature)
    if core in {"division_game", "conference_game", "home_field"}:
        return "Yes" if number else "No"
    if any(word in core for word in ("pressure", "win_pct", "success_rate")):
        return f"{number:.1%}"
    if core in {"week", "rest_days", "rest_advantage"}:
        return f"{number:g}"
    if any(word in core for word in ("epa", "ppa", "delta", "_z")):
        return f"{number:+.3f}"
    return f"{number:.1f}"


def _effect(game: dict[str, Any], target: str, value: float) -> str:
    if abs(value) < 0.0005:
        return "Even" if target == "margin" else "0.00 pts"
    if target == "total":
        return f"{value:+.2f} pts"
    team = game["home_team"] if value > 0 else game["away_team"]
    return f"{team} +{abs(value):.2f}"


def _stamp(value: Any) -> str:
    if not value:
        return "Unavailable"
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            return str(value)
        return stamp.astimezone(ZoneInfo("America/New_York")).strftime("%b %d, %I:%M %p ET")
    except ValueError:
        return str(value)


def _input_table(game: dict[str, Any], names: list[str]) -> str:
    features = game.get("features") or {}
    pairs: dict[str, dict[str, str]] = {}
    context = []
    for name in names:
        side, core, period = _parts(name)
        if side:
            pairs.setdefault(f"{core}_{period}", {})[side] = name
        else:
            context.append(name)
    rows = []
    for pair in pairs.values():
        name = next(iter(pair.values()))
        cells = "".join(
            f"<td>{html_text(_value(pair[side], features.get(pair[side]))) if side in pair else 'Not used'}</td>"
            for side in ("away", "home")
        )
        rows.append(f"<tr><td>{html_text(_label(name))}</td>{cells}</tr>")
    rows.extend(
        f'<tr><td>{html_text(_label(name))}</td><td colspan="2">{html_text(_value(name, features.get(name)))}</td></tr>'
        for name in context
    )
    roster_note = (
        '<div class="grid-input-sub">Roster continuity inputs are neutral after Week 4.</div>'
        if any("roster_" in name for name in names) and float(game.get("week", 0)) > 4
        else ""
    )
    return (
        f'<details class="grid-input-all"><summary>Show all {len(names)} model inputs</summary>'
        '<div class="grid-input-muted">L4 / L6 / L8: last 4 / 6 / 8 games. Rolling values include model shrinkage. '
        "EPA / PPA: expected points added per play. Continuity deltas are league-centered model values.</div>"
        '<table class="grid-input-table"><thead><tr><th scope="col">Input used</th>'
        f'<th scope="col">{html_text(game["away_team"])}</th><th scope="col">{html_text(game["home_team"])}</th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>{roster_note}</details>"
    )


def _drivers(game: dict[str, Any], target: str, explanation: dict[str, Any]) -> str:
    features = game.get("features") or {}
    grouped: dict[str, float] = defaultdict(float)
    members: dict[str, list[str]] = defaultdict(list)
    for name, value in explanation["contributions"].items():
        group = _group(name)
        grouped[group] += float(value)
        members[group].append(name)
    ranked = sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)
    shown = ranked[:4]
    if len(ranked) > 4:
        shown.append(("Remaining inputs", sum(value for _, value in ranked[4:])))
    maximum = max((abs(value) for _, value in shown), default=0) or 1
    rows = []
    for label, value in shown:
        if label == "Remaining inputs":
            subtitle = "Combined effect of the other model inputs"
        else:
            # Show values for the strongest feature and its paired team input.
            name = max(members[label], key=lambda name: abs(explanation["contributions"][name]))
            side, core, period = _parts(name)
            if side:
                suffix = f"{core}_{period.lower()}" if period else core
                values = " / ".join(
                    f"{game[f'{side}_team']} {_value(f'{side}_{suffix}', features[f'{side}_{suffix}'])}"
                    for side in ("away", "home")
                    if f"{side}_{suffix}" in explanation["feature_names"]
                    and f"{side}_{suffix}" in features
                )
                subtitle = f"{_label(name)} · {values}"
            else:
                subtitle = f"{_label(name)} · {_value(name, features.get(name))}"
        direction = "right" if value >= 0 else "left"
        action = (
            "points" if target == "margin" else "raises total" if value >= 0 else "lowers total"
        )
        rows.append(
            f'<div class="grid-input-driver"><div class="grid-input-name">{html_text(label)}'
            f'<div class="grid-input-sub">{html_text(subtitle)}</div></div>'
            f'<div class="grid-input-track" aria-hidden="true"><span class="grid-input-bar grid-input-bar-{direction}" '
            f'style="width:{45 * abs(value) / maximum:.2f}%"></span></div>'
            f'<div class="grid-input-effect">{html_text(_effect(game, target, value))}<small>{action}</small></div></div>'
        )
    return "".join(rows)


def _target_panel(game: dict[str, Any], target: str) -> str:
    explanation = ((game.get("model_explanations") or {}).get("targets") or {}).get(target)
    names = (game.get("model_input_schema") or {}).get(target) or []
    # Persisted explanations must still describe the visible forecast (e.g. a
    # manual injury adjustment can change the score after the base prediction).
    if explanation:
        published = (
            game.get("predicted_home_margin")
            if target == "margin"
            else game.get("total", game.get("predicted_total"))
        )
        reconstructed = (
            float(explanation["reference"])
            + sum(explanation["contributions"].values())
            + sum(float(row["value"]) for row in explanation.get("adjustments", []))
        )
        if published is None or not math.isclose(
            reconstructed,
            float(published),
            rel_tol=0,
            abs_tol=0.006 if target == "margin" else 0.051,
        ):
            explanation = None
    if not explanation:
        body = '<div class="grid-input-empty">Contribution breakdown unavailable for this forecast.</div>'
    else:
        names = explanation["feature_names"]
        prediction = float(explanation["prediction"])
        if target == "margin":
            title = f"What drives the lean toward {game['home_team'] if prediction >= 0 else game['away_team']}?"
            subtitle = "Largest effects on the predicted margin. Home-win probability uses this same margin forecast."
            left, right = f"← Toward {game['away_team']}", f"Toward {game['home_team']} →"
        else:
            title = f"What moves the total to {prediction:.1f}?"
            subtitle = "Largest effects on combined points"
            left, right = "← Lower total", "Higher total →"
        adjustments = "".join(
            f"<span>{html_text(row['label'])} ({float(row['weight']):.0%} prior weight)</span>"
            f"<span>{html_text(_effect(game, target, float(row['value'])))}</span>"
            for row in explanation.get("adjustments", [])
        )
        reference = (
            _effect(game, target, float(explanation["reference"]))
            if target == "margin"
            else f"{float(explanation['reference']):.2f} pts"
        )
        final = (
            _effect(game, target, prediction)
            if target == "margin"
            else f"{float(published):.1f} pts"
        )
        body = (
            f'<div class="grid-input-title">{html_text(title)}</div><div class="grid-input-muted">{html_text(subtitle)}</div>'
            f'<div class="grid-input-legend"><span>{html_text(left)}</span><span>{html_text(right)}</span></div>'
            f'{_drivers(game, target, explanation)}<div class="grid-input-reconcile">'
            f"<span>Reference prediction</span><span>{html_text(reference)}</span>"
            f"<span>Net contribution of this game’s inputs</span><span>{html_text(_effect(game, target, sum(explanation['contributions'].values())))}</span>"
            f'{adjustments}<span class="grid-input-final">Final predicted {"margin" if target == "margin" else "total"}</span>'
            f'<span class="grid-input-final">{html_text(final)}</span></div>'
            '<details class="grid-input-all"><summary>How contributions are measured</summary>'
            '<div class="grid-input-muted">The reference combines the fitted ridge intercept and each tree’s starting prediction, '
            "using the model’s ensemble weights. Ridge terms use the training means; tree terms track changes along this game’s decision paths. "
            "Terms are grouped and summed in points. These describe the model, not causal effects. Displayed values are rounded.</div></details>"
        )
    if names and game.get("features"):
        body += _input_table(game, names)
    else:
        body += '<div class="grid-input-muted">Input snapshot unavailable for this forecast. Current team statistics are not substituted.</div>'
    return f'<section class="grid-input-target grid-input-{target}" aria-label="{"Spread and winner" if target == "margin" else "Total points"}">{body}</section>'


def model_inputs_html(game: dict[str, Any], sport: str, key: str) -> str:
    """Render isolated native controls so odds refreshes do not reset expansion."""
    instance = hashlib.sha256(f"{sport}:{key}".encode()).hexdigest()[:16]
    warnings = (game.get("input_quality") or {}).get("warnings") or []
    warning_html = "".join(f"<div>{html_text(warning)}</div>" for warning in warnings)
    calibration = game.get("preseason_calibration") or {}
    used_prior = bool(float(calibration.get("weight", 0.0)))
    context = (
        "Injuries and live market lines are context, not core model inputs."
        + (
            " A market-derived preseason strength prior is applied separately above."
            if used_prior
            else ""
        )
        if sport == "nfl"
        else "Live market lines are context, not model inputs."
    )
    return (
        '<details class="grid-inputs"><summary>Model inputs &amp; contributions</summary><div class="grid-input-tabs" role="group" aria-label="Prediction explained">'
        f'<input class="grid-input-choice" type="radio" name="model-{instance}" id="margin-{instance}" value="margin" checked>'
        f'<label for="margin-{instance}">Spread / winner</label>'
        f'<input class="grid-input-choice" type="radio" name="model-{instance}" id="total-{instance}" value="total">'
        f'<label for="total-{instance}">Total points</label>'
        f"{_target_panel(game, 'margin')}{_target_panel(game, 'total')}</div>"
        '<div class="grid-input-footer">'
        f"Forecast saved: {html_text(_stamp(game.get('forecast_at')))} · Data through: {html_text(game.get('data_cutoff') or 'Unavailable')}"
        f"<div>{html_text(context)}</div>{warning_html}</div></details>"
    )


def render_model_inputs(game: dict[str, Any], sport: str, key: str) -> None:
    st.markdown(STYLES + model_inputs_html(game, sport, key), unsafe_allow_html=True)
