# CFB ranking blends and Top 30 matchup comparison

Snapshot cutoff: 2026-10-04T15:17:29.367649+00:00. Source: `2cb584a7cc7a2a2897b763b4c3194fe78f3f45d6`.

Compared 2,606 historical game forecasts. The results-based anchor setting remains `results_l4_capnone`. Of the three blends, development MAE selected **75% results / 25% common**. Selection uses only 2022–2024; 2025 is reserved for checking the selected method.

Weights combine ratings in points, not rank positions. All methods have mean-zero ratings. Every blend has the same three-point nonneutral home edge for historical scoring, so weighted historical margin predictions equal predictions from the blended ratings.

## Interpretation

The predeclared development selection picks 75% results / 25% common among the three blends. In 2025 it lowers margin MAE from 12.640 for production to 12.463 and raises winner accuracy from 69.7% to 71.4%. The 50/50 blend is almost as accurate (12.475 MAE), has fewer neutral ranking reversals, and has the lowest partial-2026 MAE among the blends (13.130 versus 13.147 for the selected 75% blend). It is a reasonable compromise if keeping the Top 30 closer to the direct forecast model is a product priority; that is a judgment about consistency, not the outcome of the predeclared development selection.

The 50% and 75% blends have 2025 paired 95% intervals below zero relative to current production, but the gains are small (0.165 and 0.176 points), and the comparison contains only eleven week clusters. These intervals do not establish superiority over each other or over the direct matchup model. Power-versus-nonpower error worsens as results weight increases compared with common-opponent ratings, so blending does not solve the broader cross-tier bias.

Michigan moves from #13 to #16, #21, and #26 as results weight increases; the unblended results method places Michigan #32. Oklahoma stays #19/#19/#18, and Texas A&M stays #13/#14/#13. There is no evidence here for a blanket adjustment that lowers all three teams.

The direct neutral forecast favors Georgia over Ohio State by 1.29 points after removing designation asymmetry. Yet its raw forecasts favor Georgia by 3.67 when Georgia occupies the designated-home input and favor Ohio State by 1.10 after the inputs are reversed, even though home-field advantage is zero in both. This is a distinct score-model issue. Any production ranking change should make clear whether matchup forecasts remain the existing model or use blended rating margins; forcing agreement alone is not an accuracy test. Production was not changed by this research.

## Historical accuracy

MAE is the average absolute scoring-margin error in points; lower is better. Winner accuracy excludes final ties.

| Method | Development MAE | 2025 MAE | 2025 winner accuracy | 2026 MAE | 2026 winner accuracy |
|---|---:|---:|---:|---:|---:|
| Current production projection | 12.960 | 12.640 | 69.7% | 13.278 | 77.5% |
| Common opponents | 12.986 | 12.668 | 70.4% | 13.151 | 77.1% |
| Results-based | 12.804 | 12.496 | 70.9% | 13.184 | 78.2% |
| 25% results / 75% common | 12.900 | 12.545 | 70.2% | 13.132 | 77.5% |
| 50% results / 50% common | 12.842 | 12.475 | 71.1% | 13.130 | 77.5% |
| 75% results / 25% common | 12.809 | 12.463 | 71.4% | 13.147 | 77.9% |
| Direct matchup model | 13.049 | 12.544 | 70.0% | 13.028 | 78.2% |

## Holdout uncertainty and cross-tier behavior

Differences compare with the current production schedule projection. Intervals resample entire weeks; a range crossing zero does not establish an accuracy improvement.

| Blend | 2025 MAE change vs production | 95% interval | 2025 power/nonpower MAE | 2026 power/nonpower MAE |
|---|---:|---|---:|---:|
| 25% results / 75% common | -0.095 | -0.215 to +0.023 | 13.556 | 14.778 |
| 50% results / 50% common | -0.165 | -0.296 to -0.025 | 13.600 | 14.863 |
| 75% results / 25% common | -0.176 | -0.334 to -0.012 | 13.662 | 14.959 |

## Top 30 for each method

| Rank | Common opponents | 25% results | 50% results | 75% results | Results-based |
|---:|---|---|---|---|---|
| 1 | Georgia | Georgia | Georgia | Georgia | Georgia |
| 2 | Ohio State | Ohio State | Ohio State | Ohio State | Ohio State |
| 3 | Alabama | Alabama | Alabama | Alabama | Alabama |
| 4 | Miami | Miami | Miami | Notre Dame | Notre Dame |
| 5 | Texas | Notre Dame | Notre Dame | Miami | Miami |
| 6 | Notre Dame | Texas | Texas | Texas | Texas |
| 7 | LSU | LSU | LSU | Indiana | Indiana |
| 8 | Indiana | Indiana | Indiana | LSU | Oregon |
| 9 | Oregon | Oregon | Oregon | Oregon | LSU |
| 10 | USC | USC | Utah | Utah | Utah |
| 11 | Tennessee | Utah | Tennessee | Tennessee | Texas Tech |
| 12 | Utah | Tennessee | USC | Texas Tech | Texas A&M |
| 13 | Michigan | Texas A&M | Texas Tech | Texas A&M | Tennessee |
| 14 | Texas A&M | Texas Tech | Texas A&M | USC | USC |
| 15 | Texas Tech | BYU | BYU | Boise State | Boise State |
| 16 | BYU | Michigan | Boise State | BYU | Missouri |
| 17 | Ole Miss | Ole Miss | Ole Miss | Missouri | BYU |
| 18 | Boise State | Boise State | Missouri | Oklahoma | Oklahoma |
| 19 | Oklahoma | Oklahoma | Oklahoma | Nebraska | Florida |
| 20 | Nebraska | Nebraska | Nebraska | Florida | Nebraska |
| 21 | Arizona | Missouri | Michigan | Ole Miss | Northwestern |
| 22 | Missouri | James Madison | Florida | Northwestern | Penn State |
| 23 | UCLA | UCLA | James Madison | James Madison | James Madison |
| 24 | James Madison | Arizona | Northwestern | UCLA | Ole Miss |
| 25 | Florida | Florida | UCLA | Iowa | UCLA |
| 26 | Northwestern | Northwestern | Arizona | Michigan | Iowa |
| 27 | Iowa | Iowa | Iowa | Arizona | Wisconsin |
| 28 | Kansas State | Houston | Wisconsin | Penn State | Mississippi State |
| 29 | Houston | Washington | Houston | Wisconsin | Arizona |
| 30 | Washington | Kansas State | Mississippi State | Mississippi State | South Carolina |

Nonpower teams in each Top 30: Common opponents: 2; 25% results / 75% common: 2; 50% results / 50% common: 2; 75% results / 25% common: 2; Results-based: 2. Notre Dame is counted as power-level; 2026 Pac-12 is counted outside the four power conferences.

## Teams discussed

| Team | Common rank | 25% rank | 50% rank | 75% rank | Results rank |
|---|---:|---:|---:|---:|---:|
| Michigan | 13 | 16 | 21 | 26 | 32 |
| Oklahoma | 19 | 19 | 19 | 18 | 18 |
| Texas A&M | 14 | 13 | 14 | 13 | 12 |
| James Madison | 24 | 22 | 23 | 23 | 23 |

## All-pairs neutral matchup consistency

Each method's own Top 30 produces 435 unique pairs. The higher-ranked team's predicted margin comes from a fresh query of the direct score model with zero home advantage, equal seven-day rest, the same current team snapshots, and a standardized nonconference flag. Both designated-home orientations are evaluated. Their antisymmetric average is the primary neutral margin; negative means the score model favors the lower-ranked team.

This checks internal consistency, not accuracy against real outcomes. In this linear Ridge model, the symmetric common-opponent margin equals the difference between common-opponent ratings. Zero reversals for that method is therefore expected by construction. A blend incorporates results information that the unchanged score model does not weight in the same way, so disagreement is possible and is not automatically an error.

| Method | Pairs | Any neutral reversal | Reversal >1 point | Reversal >3 points | Raw designation flips winner |
|---|---:|---:|---:|---:|---:|
| Common opponents | 435 | 0 | 0 | 0 | 93 |
| 25% results / 75% common | 435 | 11 | 0 | 0 | 93 |
| 50% results / 50% common | 435 | 26 | 7 | 0 | 78 |
| 75% results / 25% common | 435 | 43 | 22 | 9 | 76 |
| Results-based | 435 | 45 | 28 | 12 | 72 |

## Georgia examples

Positive margins favor Georgia. These are neutral-site research forecasts, not forecasts for a scheduled game.

| Opponent | Direct symmetric margin | Georgia designated home | Georgia designated away | 25% rating margin | 50% rating margin | 75% rating margin |
|---|---:|---:|---:|---:|---:|---:|
| Ohio State | +1.29 | +3.67 | -1.10 | +1.12 | +0.94 | +0.77 |
| Alabama | +3.84 | +6.93 | +0.75 | +3.20 | +2.56 | +1.92 |
| Miami | +4.37 | +6.78 | +1.96 | +3.83 | +3.30 | +2.77 |
| Texas | +4.82 | +5.06 | +4.58 | +4.69 | +4.57 | +4.44 |
| Notre Dame | +5.57 | +7.43 | +3.71 | +4.56 | +3.54 | +2.53 |

## Largest disagreements: 25% results / 75% common

The table shows the score model's strongest preferences for a lower-ranked team. All 435 pairs, including both raw designations, are in the linked full comparison.

| Higher-ranked team | Lower-ranked team | Ranking margin | Direct neutral margin |
|---|---|---:|---:|
| #15 BYU | #16 Michigan | +0.13 | -0.89 |
| #5 Notre Dame | #6 Texas | +0.14 | -0.75 |
| #14 Texas Tech | #16 Michigan | +1.37 | -0.54 |
| #22 James Madison | #24 Arizona | +0.06 | -0.48 |
| #29 Washington | #30 Kansas State | +0.02 | -0.39 |
| #23 UCLA | #24 Arizona | +0.05 | -0.25 |
| #22 James Madison | #23 UCLA | +0.01 | -0.22 |
| #11 Utah | #12 Tennessee | +0.40 | -0.14 |
| #13 Texas A&M | #16 Michigan | +1.52 | -0.12 |
| #28 Houston | #30 Kansas State | +0.37 | -0.03 |
| #21 Missouri | #24 Arizona | +0.65 | -0.00 |

[Every pair for this blend](blend_results25-matchups.md)


## Largest disagreements: 50% results / 50% common

The table shows the score model's strongest preferences for a lower-ranked team. All 435 pairs, including both raw designations, are in the linked full comparison.

| Higher-ranked team | Lower-ranked team | Ranking margin | Direct neutral margin |
|---|---|---:|---:|
| #18 Missouri | #21 Michigan | +0.37 | -2.79 |
| #20 Nebraska | #21 Michigan | +0.13 | -2.52 |
| #19 Oklahoma | #21 Michigan | +0.36 | -2.24 |
| #16 Boise State | #21 Michigan | +0.98 | -1.69 |
| #10 Utah | #12 USC | +1.32 | -1.68 |
| #11 Tennessee | #12 USC | +0.38 | -1.54 |
| #28 Wisconsin | #29 Houston | +0.39 | -1.34 |
| #17 Ole Miss | #21 Michigan | +0.48 | -0.94 |
| #15 BYU | #21 Michigan | +1.15 | -0.89 |
| #24 Northwestern | #26 Arizona | +0.52 | -0.84 |
| #16 Boise State | #17 Ole Miss | +0.50 | -0.75 |
| #5 Notre Dame | #6 Texas | +1.02 | -0.75 |
| #22 Florida | #26 Arizona | +0.62 | -0.75 |
| #24 Northwestern | #25 UCLA | +0.17 | -0.59 |
| #18 Missouri | #19 Oklahoma | +0.00 | -0.55 |

[Every pair for this blend](blend_results50-matchups.md)


## Largest disagreements: 75% results / 25% common

The table shows the score model's strongest preferences for a lower-ranked team. All 435 pairs, including both raw designations, are in the linked full comparison.

| Higher-ranked team | Lower-ranked team | Ranking margin | Direct neutral margin |
|---|---|---:|---:|
| #28 Penn State | #29 Wisconsin | +0.09 | -4.87 |
| #25 Iowa | #26 Michigan | +0.20 | -4.73 |
| #28 Penn State | #30 Mississippi State | +0.30 | -4.11 |
| #12 Texas Tech | #14 USC | +1.26 | -4.08 |
| #13 Texas A&M | #14 USC | +0.86 | -3.66 |
| #22 Northwestern | #26 Michigan | +1.18 | -3.63 |
| #20 Florida | #26 Michigan | +1.29 | -3.53 |
| #23 James Madison | #26 Michigan | +1.11 | -3.26 |
| #24 UCLA | #26 Michigan | +0.63 | -3.04 |
| #17 Missouri | #26 Michigan | +1.94 | -2.79 |
| #7 Indiana | #8 LSU | +0.48 | -2.78 |
| #20 Florida | #21 Ole Miss | +0.10 | -2.60 |
| #19 Nebraska | #26 Michigan | +1.46 | -2.52 |
| #18 Oklahoma | #26 Michigan | +1.67 | -2.24 |
| #25 Iowa | #27 Arizona | +0.22 | -1.94 |

[Every pair for this blend](blend_results75-matchups.md)

## Limits and provenance

- Weeks 1–11 only; the original ranking projection cannot reliably fit some late-season schedules.
- Historical final schedules proxy what was known each week. Season-level roster feeds lack historical publication timestamps.
- Weekly results and advanced performance are frozen before validation-week kickoff. Neither blend weight nor results-method settings are selected using 2025 or 2026 errors.
- 2025 was previously examined for the production model, so it is a ranking-selection holdout rather than entirely unseen data. 2026 is partial-season follow-up.
- The results method retains a preseason prior; this is not a purely current-season resume ranking.
- Changing a ranking does not modify the production score model. Raw designation disagreement is a separate model limitation; hypothetical matchup checks do not supply actual outcomes.
- Raw CFBD data remain private. Output contains aggregate statistics and derived ratings/forecasts.

Research run: https://github.com/DrewM96/nfl-prediction-system/actions/runs/37212101418
