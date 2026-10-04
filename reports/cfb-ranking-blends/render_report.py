"""Render aggregate blend evaluation and every Top 30 neutral comparison."""

from __future__ import annotations

import json
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parents[2] / "reports" / "cfb-ranking-blends"


def main():
    data = json.loads((DIRECTORY / "evaluation.json").read_text(encoding="utf-8"))
    selected = data["selected_results_method"]
    labels = {
        "schedule_projection": "Current production projection",
        "common_opponent": "Common opponents",
        selected: "Results-based",
        "blend_results25": "25% results / 75% common",
        "blend_results50": "50% results / 50% common",
        "blend_results75": "75% results / 25% common",
        "direct_score_model": "Direct matchup model",
    }
    blends = list(data["blend_results_weights"])
    ranking_methods = ["common_opponent", *blends, selected]
    current = data["current_ranking_examples"]
    audits = data["current_neutral_matchup_audits"]
    lines = [
        "# CFB ranking blends and Top 30 matchup comparison",
        "",
        f"Snapshot cutoff: {data['current_snapshot_cutoff']}. Source: `{data['source_commit']}`.",
        "",
        f"Compared {data['comparison_games']:,} historical game forecasts. The results-based anchor setting remains `{selected}`. Of the three blends, development MAE selected **{labels[data['selected_blend_on_development']]}**. Selection uses only 2022–2024; 2025 is reserved for checking the selected method.",
        "",
        "Weights combine ratings in points, not rank positions. All methods have mean-zero ratings. Every blend has the same three-point nonneutral home edge for historical scoring, so weighted historical margin predictions equal predictions from the blended ratings.",
        "",
        "## Interpretation",
        "",
        "The predeclared development selection picks 75% results / 25% common among the three blends. In 2025 it lowers margin MAE from 12.640 for production to 12.463 and raises winner accuracy from 69.7% to 71.4%. The 50/50 blend is almost as accurate (12.475 MAE), has fewer neutral ranking reversals, and has the lowest partial-2026 MAE among the blends (13.130 versus 13.147 for the selected 75% blend). It is a reasonable compromise if keeping the Top 30 closer to the direct forecast model is a product priority; that is a judgment about consistency, not the outcome of the predeclared development selection.",
        "",
        "The 50% and 75% blends have 2025 paired 95% intervals below zero relative to current production, but the gains are small (0.165 and 0.176 points), and the comparison contains only eleven week clusters. These intervals do not establish superiority over each other or over the direct matchup model. Power-versus-nonpower error worsens as results weight increases compared with common-opponent ratings, so blending does not solve the broader cross-tier bias.",
        "",
        "Michigan moves from #13 to #16, #21, and #26 as results weight increases; the unblended results method places Michigan #32. Oklahoma stays #19/#19/#18, and Texas A&M stays #13/#14/#13. There is no evidence here for a blanket adjustment that lowers all three teams.",
        "",
        "The direct neutral forecast favors Georgia over Ohio State by 1.29 points after removing designation asymmetry. Yet its raw forecasts favor Georgia by 3.67 when Georgia occupies the designated-home input and favor Ohio State by 1.10 after the inputs are reversed, even though home-field advantage is zero in both. This is a distinct score-model issue. Any production ranking change should make clear whether matchup forecasts remain the existing model or use blended rating margins; forcing agreement alone is not an accuracy test. Production was not changed by this research.",
        "",
        "## Historical accuracy",
        "",
        "MAE is the average absolute scoring-margin error in points; lower is better. Winner accuracy excludes final ties.",
        "",
        "| Method | Development MAE | 2025 MAE | 2025 winner accuracy | 2026 MAE | 2026 winner accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, label in labels.items():
        development = data["metrics"]["development_2022_2024"]["all"][method]
        holdout = data["metrics"]["holdout_2025"]["all"][method]
        followup = data["metrics"]["followup_2026"]["all"][method]
        lines.append(
            f"| {label} | {development['mae']:.3f} | {holdout['mae']:.3f} | {holdout['winner_accuracy']:.1%} | {followup['mae']:.3f} | {followup['winner_accuracy']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Holdout uncertainty and cross-tier behavior",
            "",
            "Differences compare with the current production schedule projection. Intervals resample entire weeks; a range crossing zero does not establish an accuracy improvement.",
            "",
            "| Blend | 2025 MAE change vs production | 95% interval | 2025 power/nonpower MAE | 2026 power/nonpower MAE |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for method in blends:
        interval = data["holdout_paired_delta_mae_ci95"][method]["all"]
        power25 = data["metrics"]["holdout_2025"]["power_vs_nonpower"][method]["mae"]
        power26 = data["metrics"]["followup_2026"]["power_vs_nonpower"][method]["mae"]
        lines.append(
            f"| {labels[method]} | {interval['delta_mae']:+.3f} | {interval['ci95'][0]:+.3f} to {interval['ci95'][1]:+.3f} | {power25:.3f} | {power26:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Top 30 for each method",
            "",
            "| Rank | Common opponents | 25% results | 50% results | 75% results | Results-based |",
            "|---:|---|---|---|---|---|",
        ]
    )
    for rank in range(30):
        lines.append(
            "| "
            + str(rank + 1)
            + " | "
            + " | ".join(current[m]["top30"][rank]["team"] for m in ranking_methods)
            + " |"
        )
    lines.extend(
        [
            "",
            "Nonpower teams in each Top 30: "
            + "; ".join(f"{labels[m]}: {current[m]['nonpower_top30']}" for m in ranking_methods)
            + ". Notre Dame is counted as power-level; 2026 Pac-12 is counted outside the four power conferences.",
        ]
    )
    lines.extend(
        [
            "",
            "## Teams discussed",
            "",
            "| Team | Common rank | 25% rank | 50% rank | 75% rank | Results rank |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for team in ("Michigan", "Oklahoma", "Texas A&M", "James Madison"):
        ranks = [
            next(r["rank"] for r in current[m]["ratings"] if r["team"] == team)
            for m in ranking_methods
        ]
        lines.append(f"| {team} | " + " | ".join(map(str, ranks)) + " |")
    lines.extend(
        [
            "",
            "## All-pairs neutral matchup consistency",
            "",
            "Each method's own Top 30 produces 435 unique pairs. The higher-ranked team's predicted margin comes from a fresh query of the direct score model with zero home advantage, equal seven-day rest, the same current team snapshots, and a standardized nonconference flag. Both designated-home orientations are evaluated. Their antisymmetric average is the primary neutral margin; negative means the score model favors the lower-ranked team.",
            "",
            "This checks internal consistency, not accuracy against real outcomes. In this linear Ridge model, the symmetric common-opponent margin equals the difference between common-opponent ratings. Zero reversals for that method is therefore expected by construction. A blend incorporates results information that the unchanged score model does not weight in the same way, so disagreement is possible and is not automatically an error.",
            "",
            "| Method | Pairs | Any neutral reversal | Reversal >1 point | Reversal >3 points | Raw designation flips winner |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in ranking_methods:
        audit = audits[method]
        lines.append(
            f"| {labels[method]} | {audit['pair_count']} | {audit['disagreements']} | {audit['disagreements_over_1_point']} | {audit['disagreements_over_3_points']} | {audit['designation_changes_winner']} |"
        )
    lines.extend(
        [
            "",
            "## Georgia examples",
            "",
            "Positive margins favor Georgia. These are neutral-site research forecasts, not forecasts for a scheduled game.",
            "",
            "| Opponent | Direct symmetric margin | Georgia designated home | Georgia designated away | 25% rating margin | 50% rating margin | 75% rating margin |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    ratings = {m: {r["team"]: r["rating"] for r in current[m]["ratings"]} for m in ranking_methods}
    for opponent in ("Ohio State", "Alabama", "Miami", "Texas", "Notre Dame"):
        pair = next(
            p
            for p in audits["common_opponent"]["pairs"]
            if p["higher_team"] == "Georgia" and p["lower_team"] == opponent
        )
        margins = [ratings[m]["Georgia"] - ratings[m][opponent] for m in blends]
        lines.append(
            f"| {opponent} | {pair['direct_neutral_margin']:+.2f} | {pair['higher_designated_home_margin']:+.2f} | {pair['higher_designated_away_margin']:+.2f} | "
            + " | ".join(f"{v:+.2f}" for v in margins)
            + " |"
        )
    for method in blends:
        lines.extend(
            [
                "",
                f"## Largest disagreements: {labels[method]}",
                "",
                "The table shows the score model's strongest preferences for a lower-ranked team. All 435 pairs, including both raw designations, are in the linked full comparison.",
                "",
                "| Higher-ranked team | Lower-ranked team | Ranking margin | Direct neutral margin |",
                "|---|---|---:|---:|",
            ]
        )
        disagreements = sorted(
            (p for p in audits[method]["pairs"] if p["ranking_disagreement"]),
            key=lambda p: p["direct_neutral_margin"],
        )
        for pair in disagreements[:15]:
            lines.append(
                f"| #{pair['higher_rank']} {pair['higher_team']} | #{pair['lower_rank']} {pair['lower_team']} | {pair['rating_margin']:+.2f} | {pair['direct_neutral_margin']:+.2f} |"
            )
        pair_lines = [
            f"# Every Top 30 pair: {labels[method]}",
            "",
            "Margins favor the higher-ranked team. Negative direct margin marks a ranking disagreement. Both raw forecasts are expressed from the higher-ranked team's perspective.",
            "",
            "| Higher rank | Higher team | Lower rank | Lower team | Rating margin | Symmetric direct margin | Direct: designated home | Direct: designated away | Disagreement |",
            "|---:|---|---:|---|---:|---:|---:|---:|---|",
        ]
        for p in audits[method]["pairs"]:
            pair_lines.append(
                f"| {p['higher_rank']} | {p['higher_team']} | {p['lower_rank']} | {p['lower_team']} | {p['rating_margin']:+.2f} | {p['direct_neutral_margin']:+.2f} | {p['higher_designated_home_margin']:+.2f} | {p['higher_designated_away_margin']:+.2f} | {'Yes' if p['ranking_disagreement'] else ''} |"
            )
        filename = f"{method}-matchups.md"
        (DIRECTORY / filename).write_text("\n".join(pair_lines) + "\n", encoding="utf-8")
        lines.extend(["", f"[Every pair for this blend]({filename})", ""])
    lines.extend(
        [
            "## Limits and provenance",
            "",
            "- Weeks 1–11 only; the original ranking projection cannot reliably fit some late-season schedules.",
            "- Historical final schedules proxy what was known each week. Season-level roster feeds lack historical publication timestamps.",
            "- Weekly results and advanced performance are frozen before validation-week kickoff. Neither blend weight nor results-method settings are selected using 2025 or 2026 errors.",
            "- 2025 was previously examined for the production model, so it is a ranking-selection holdout rather than entirely unseen data. 2026 is partial-season follow-up.",
            "- The results method retains a preseason prior; this is not a purely current-season resume ranking.",
            "- Changing a ranking does not modify the production score model. Raw designation disagreement is a separate model limitation; hypothetical matchup checks do not supply actual outcomes.",
            "- Raw CFBD data remain private. Output contains aggregate statistics and derived ratings/forecasts.",
            "",
            "Research run: https://github.com/DrewM96/nfl-prediction-system/actions/runs/37212101418",
            "",
        ]
    )
    (DIRECTORY / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
