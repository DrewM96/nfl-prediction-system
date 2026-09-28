# NFL season-transition weighting benchmark

Evaluation seasons: 2022, 2023, 2024, 2025

## Margin MAE

| Variant | Weeks 1-4 | Weeks 5-8 | Weeks 9-18 | Full season | Winner acc. (full) |
|---|---:|---:|---:|---:|---:|
| baseline | 10.194 | 10.578 | 9.700 | 10.004 | 62.2% |
| current_season_2x | 10.270 | 10.558 | 9.742 | 10.041 | 61.9% |
| offseason_decay | 10.423 | 10.567 | 9.748 | 10.082 | 60.9% |

## Total MAE

| Variant | Weeks 1-4 | Weeks 5-8 | Weeks 9-18 | Full season |
|---|---:|---:|---:|---:|
| baseline | 10.380 | 10.042 | 10.837 | 10.559 |
| current_season_2x | 10.485 | 10.000 | 10.799 | 10.554 |
| offseason_decay | 10.552 | 9.879 | 10.804 | 10.547 |

## Season-by-season full-season MAE

| Season | Variant | Margin MAE | Total MAE |
|---:|---|---:|---:|
| 2022 | baseline | 9.085 | 10.953 |
| 2022 | current_season_2x | 9.139 | 10.933 |
| 2022 | offseason_decay | 9.122 | 10.895 |
| 2023 | baseline | 10.542 | 10.606 |
| 2023 | current_season_2x | 10.590 | 10.617 |
| 2023 | offseason_decay | 10.740 | 10.553 |
| 2024 | baseline | 10.405 | 10.109 |
| 2024 | current_season_2x | 10.394 | 10.129 |
| 2024 | offseason_decay | 10.373 | 10.096 |
| 2025 | baseline | 9.982 | 10.570 |
| 2025 | current_season_2x | 10.039 | 10.537 |
| 2025 | offseason_decay | 10.090 | 10.643 |
