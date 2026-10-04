# CFB ranking evaluation — October 4, 2026

Chronological research comparison. Production rankings and forecasts were not changed.

Evaluated **2,606 game forecasts**. The results-based setting selected using only 2022–2024 development MAE was `results_l4_capnone`.

Benchmark source commit: `0a94614e802a0d8da8eaa8ebbf07057cc8973ee6`. Generated: 2026-10-04T13:43:25.161055+00:00.

## Findings and recommendation

Use neutral common-opponent ratings for the predictive Top 30. Every team is evaluated against the same national opponent pool under the same conditions, so the ranking no longer depends on which conference games remain or whether the remaining schedule is disconnected. This recommendation rests on comparability and structure, not a demonstrated large accuracy improvement.

The current published snapshot has James Madison fifth and nine nonpower teams in the Top 30. The common-opponent research snapshot puts JMU 24th, and the selected results-based method puts JMU 23rd. Both contain two nonpower teams: Boise State and JMU. These counts are outcomes, not targets or conference quotas.

The results-based method is a credible second option: it fits completed scoring margins while accounting for opponent ratings and anchoring sparse early-season results to a preseason prior. Its selected setting is prior penalty 4 with no margin cap. It achieved the lowest overall development and 2025 holdout MAE among the tested ranking alternatives, but did not improve the 2025 power-versus-nonpower subset and was weaker than common-opponent ratings on 2026 cross-conference games. It is useful as a companion results-driven ranking rather than the first choice for the forecast-based Top 30.

Neither alternative has a conclusive 2025 accuracy advantage: all paired 95% intervals include zero. Common-opponent ratings were 0.028 points worse overall in 2025, then 0.127 points better in the partial 2026 check. Results-based ratings improved overall MAE by 0.144 points in 2025 and 0.093 points in 2026. These are small changes relative to roughly 12–13 points of typical margin error.

A separate underlying forecast issue remains. In power-versus-nonpower games, even the direct matchup model underestimated the power team's advantage by 4.03 points on average in 2025 and 7.36 points in the 2026 sample. The current ranking projection increased those shortfalls to 6.79 and 10.35 points; common-opponent ratings reduced them to 5.66 and 9.98. Changing the ranking method helps comparability, but does not remove the score model's tendency to overrate nonpower teams in these matchups. Opponent-adjusted performance features merit a separate evaluation before changing score forecasts.

The completed workflow passed Ruff lint/format and the ranking, feature, and evaluation tests. No production ranking or score-prediction change was made.

## Development: 2022–2024

| Method | Games | Margin MAE | Winner accuracy | Cross-conference MAE | Power vs nonpower MAE |
|---|---:|---:|---:|---:|---:|
| Current schedule projection | 1768 | 12.960 | 71.0% | 13.326 | 13.304 |
| Neutral common opponents | 1768 | 12.986 | 71.4% | 13.262 | 13.183 |
| Results with preseason prior | 1768 | 12.804 | 70.5% | 13.283 | 13.179 |
| Direct matchup model (reference) | 1768 | 13.049 | 70.1% | 13.396 | 13.045 |

Cross-conference sample: 567 games. Power versus nonpower sample: 283 games. Lower MAE is better. Winner accuracy excludes tied final scores.

## Holdout: 2025

| Method | Games | Margin MAE | Winner accuracy | Cross-conference MAE | Power vs nonpower MAE |
|---|---:|---:|---:|---:|---:|
| Current schedule projection | 567 | 12.640 | 69.7% | 13.802 | 13.717 |
| Neutral common opponents | 567 | 12.668 | 70.4% | 13.781 | 13.530 |
| Results with preseason prior | 567 | 12.496 | 70.9% | 13.525 | 13.763 |
| Direct matchup model (reference) | 567 | 12.544 | 70.0% | 13.406 | 13.015 |

Cross-conference sample: 186 games. Power versus nonpower sample: 97 games. Lower MAE is better. Winner accuracy excludes tied final scores.

## Follow-up: completed 2026 games

| Method | Games | Margin MAE | Winner accuracy | Cross-conference MAE | Power vs nonpower MAE |
|---|---:|---:|---:|---:|---:|
| Current schedule projection | 271 | 13.278 | 77.5% | 13.509 | 15.147 |
| Neutral common opponents | 271 | 13.151 | 77.1% | 13.256 | 14.721 |
| Results with preseason prior | 271 | 13.184 | 78.2% | 13.571 | 15.054 |
| Direct matchup model (reference) | 271 | 13.028 | 78.2% | 13.052 | 13.644 |

Cross-conference sample: 164 games. Power versus nonpower sample: 76 games. Lower MAE is better. Winner accuracy excludes tied final scores.

## Holdout uncertainty

Changes in mean absolute error relative to the current schedule projection. Negative values favor the alternative. Intervals resample entire weeks rather than treating games within a week as independent.

| Alternative | Group | Change in MAE | 95% interval | Week clusters |
|---|---|---:|---|---:|
| Neutral common opponents | all | +0.028 | -0.093 to +0.147 | 11 |
| Neutral common opponents | cross_conference | -0.021 | -0.347 to +0.412 | 11 |
| Neutral common opponents | power_vs_nonpower | -0.187 | -0.861 to +0.787 | 9 |
| Results with preseason prior | all | -0.144 | -0.339 to +0.052 | 11 |
| Results with preseason prior | cross_conference | -0.277 | -1.095 to +0.111 | 11 |
| Results with preseason prior | power_vs_nonpower | +0.046 | -0.705 to +0.661 | 9 |

## Current ranking examples

The following examples are reconstructed from the benchmark's current team-state snapshot. These are research outputs, not deployed rankings.

| Method | JMU rank | Nonpower teams in Top 30 |
|---|---:|---:|
| Neutral common opponents | 24 | 2 |
| Results with preseason prior | 23 | 2 |

## Selection and safeguards

- The margin model uses the existing production feature set and Ridge alpha 50. It is refitted on earlier seasons and earlier weeks.
- All team snapshots are frozen before the first kickoff of a validation week. Same-week scores and advanced stats are masked, including games that might otherwise have preceded a team's individual kickoff.
- Neutral comparisons evaluate every team against the same pool in both designated-home orientations, with seven days of rest. Symmetrizing the margins removes designation bias.
- Results-based ratings start from each season's preseason neutral model ratings. The grid tests prior penalties 1, 4, and 8, and margin caps 21, 28, or none. Selection uses overall development MAE, not JMU's rank or the number of nonpower teams.
- Common-opponent and results ratings use a fixed three-point home edge for outcome scoring. The current projection uses its fitted home and designation terms. The direct matchup model is shown as a reference rather than a national ranking.
- Power conferences are ACC, Big Ten, Big 12, SEC, plus Pac-12 through 2023; Notre Dame is treated as power-level. The 2024–2026 Pac-12 is counted outside that set.

## Limits

- This comparison evaluates Weeks 1–11. Later weeks can have too few remaining games for the current production ranking fit, so this is not a full-season evaluation.
- Historical final schedules are used as schedule-known proxies. They may include rescheduling and conference championship participant information that was not known at an early-season cutoff; this limits the baseline replay's point-in-time fidelity.
- Returning-production, talent, and recruiting feeds are season-level proxies without historical publication timestamps. Portal records retain the existing pre-season date restriction.
- 2025 has previously been inspected while evaluating the production model. It was kept out of ranking-parameter selection here, but this is not a newly hidden dataset.
- The 2026 check is a partial season. Eleven holdout week clusters give limited precision, especially for cross-conference subsets.
- Raw CFBD inputs remain private; the saved output contains aggregate metrics and derived ranking examples only.

## Reproduction

Research branch: `research/cfb-ranking-evaluation`. `cfb_ranking_evaluation.py` runs through the existing CFBD-connected research workflow; `tests/test_cfb_ranking_evaluation.py` checks symmetry, disconnected prior anchoring, snapshot isolation, serialization, and development/holdout separation.

Completed research run: https://github.com/DrewM96/nfl-prediction-system/actions/runs/37205811980
