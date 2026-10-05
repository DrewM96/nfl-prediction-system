"""Recover aggregate team inputs from the exact release cache; no API fallback."""

from datetime import datetime
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from cfb_prediction.blended_rankings import frozen_frames, common_opponent_ratings, results_ratings
from cfb_prediction.client import CFBDClient
from cfb_prediction.features import build_point_in_time_features, build_preseason_table
from cfb_prediction.historical import load_historical_data
from cfb_prediction.modeling import load_cfb_model_bundle, make_ridge
from cfb_prediction.production import _cut_off_results


class CacheOnlyClient(CFBDClient):
    def get(self, endpoint, *, params=None, **kwargs):
        normalized = '/' + endpoint.strip('/')
        query = {k: v for k, v in (params or {}).items() if v is not None}
        path = self._cache_path(normalized, query)
        if not path.exists():
            raise RuntimeError(f'Exact-cache file missing: {path.name}')
        payload = json.loads(path.read_text())
        assert payload['endpoint'] == normalized
        assert payload['params'] == query
        cache_inventory.append({'file': path.name, 'fetched_at': payload['fetched_at'], 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        return payload['data']


cache_inventory = []
rankings = json.loads(Path('data/cfb/power_rankings.json').read_text())
assert rankings['model_hash'] == '787beaea1b89079bc6ce3f6a70fcdaccc4dd1058b0f6b236434071ae58ea5136'
models, manifest = load_cfb_model_bundle()
assert hashlib.sha256(Path('data/cfb/models/manifest.json').read_bytes()).hexdigest() == rankings['model_hash']
season = rankings['prediction_season']
week = rankings['forecast_week']
cutoff = datetime.fromisoformat(rankings['data_cutoff'])
data = load_historical_data(CacheOnlyClient('cache-only-placeholder'), manifest['training_seasons'])
data = _cut_off_results(data, cutoff)
features = build_point_in_time_features(data, include_scheduled=True)
training = features[((features.season < season) | (features.season.eq(season) & features.week.lt(week))) & features.completed.fillna(False) & features.start_date.lt(pd.Timestamp(cutoff))]
feature_names = models['margin'].feature_names
preseason_training = training[training.season.lt(season)].dropna(subset=[*feature_names, 'home_margin'])
preseason_model = make_ridge(manifest['models']['margin']['metrics']['ridge_alpha']).fit(preseason_training[feature_names], preseason_training.home_margin)
frozen, _, snapshots, metadata = frozen_frames(data, season, week, pd.Timestamp(cutoff), feature_names)
first_kickoff = data.games.loc[data.games.season.eq(season), 'start_date'].min()
_, _, preseason_snapshots, _ = frozen_frames(data, season, 0, first_kickoff, feature_names)
prior = common_opponent_ratings(preseason_model, preseason_snapshots, 1, feature_names)
common = common_opponent_ratings(models['margin'].estimator, snapshots, week, feature_names)
games = frozen.games[frozen.games.season.eq(season)]
results = results_ratings(games, common, prior, 4.0, None)
checks = {}
for field, rebuilt in [('common_opponent_rating', common), ('results_rating', results)]:
    checks[field + '_max_error'] = max(abs(row[field] - rebuilt[row['team']]) for row in rankings['ratings'])
checks['final_rating_max_error'] = max(abs(row['rating'] - 0.75*common[row['team']] - 0.25*results[row['team']]) for row in rankings['ratings'])
print(json.dumps(checks), flush=True)
assert max(checks.values()) < 1e-7, 'Recovered inputs do not reproduce the published release'
team_keys = [name.removeprefix('home_') for name in feature_names if name.startswith('home_') and name not in ('home_field', 'home_rest_days')]
means = {key: float(np.mean([snapshots[t][key] for t in snapshots])) for key in team_keys}
raw_maps = {}
for frame_name in ('returning', 'talent', 'recruiting'):
    frame = getattr(data, frame_name)
    current = frame[frame.season.eq(season)].drop_duplicates('team', keep='last')
    raw_maps[frame_name] = current.set_index('team').to_dict(orient='index')
played = games[games.completed.fillna(False) & games.fbs_vs_fbs.fillna(False)].dropna(subset=['home_points', 'away_points'])
rows = []
for row in rankings['ratings']:
    team = row['team']
    appearances = played[played.home_team.eq(team) | played.away_team.eq(team)]
    assert len(appearances) == row['completed_games']
    score_margins = []
    opponent_ratings = []
    for _, game in appearances.iterrows():
        home = game.home_team == team
        hm = float(game.home_points - game.away_points) - (0 if game.neutral_site else 3)
        score_margins.append(hm if home else -hm)
        opponent_ratings.append(results[game.away_team if home else game.home_team])
    raw_values = {}
    for frame_name in raw_maps:
        for key, value in raw_maps[frame_name].get(team, {}).items():
            if key != 'season':
                raw_values[key] = None if pd.isna(value) else float(value)
    rows.append({**row, 'team_inputs': {key: float(snapshots[team][key]) for key in team_keys}, 'preseason_prior': float(prior[team]), 'preseason_inputs': {key: float(preseason_snapshots[team][key]) for key in team_keys}, 'raw_roster_inputs': raw_values, 'results_adjusted_margin_sum': sum(score_margins), 'results_opponent_rating_sum': sum(opponent_ratings), 'results_prior_numerator': 4*prior[team], 'results_denominator': len(appearances)+4})
output = {'data_cutoff': rankings['data_cutoff'], 'model_hash': rankings['model_hash'], 'cache_key': 'cfb-data-2026-37214197278', 'cache_inventory': cache_inventory, 'verification': checks, 'team_input_means': means, 'team_input_keys': team_keys, 'team_count': len(rows), 'results_game_count': len(played), 'teams': rows}
out = Path('recovered-cfb-inputs')
out.mkdir(exist_ok=True)
(out / 'team-inputs.json').write_text(json.dumps(output, indent=2, allow_nan=False))
print(f'Recovered and verified {len(rows)} teams, {len(team_keys)} model inputs each, and {len(played)} result games.')
