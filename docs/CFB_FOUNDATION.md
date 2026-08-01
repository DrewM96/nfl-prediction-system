# College Football Foundation

## Current status

The existing GRIDLINE Streamlit app includes a top-level NFL / College Football selector.
The College Football path loads only `data/cfb/foundation.json`; it does not load or mutate
NFL models, injuries, schedules, or prediction artifacts.

The foundation currently provides:

- bearer-authenticated CollegeFootballData REST v2 requests;
- a six-hour local disk cache to conserve the monthly request allowance;
- canonical FBS team IDs, game IDs, conference membership, kickoff timestamps, neutral-site
  flags, and opponent classifications;
- explicit FBS-vs-FBS identification rather than treating FCS opponents as ordinary peers;
- an aggregate public status artifact with no raw team or game records;
- a manual GitHub workflow that validates the configured repository secret.

## Privacy and source boundaries

`CFBD_API_KEY`, `.env`, `data/cfb/cache/`, and `data/cfb/private/` must never be staged. Cache
files intentionally exclude authorization headers and API keys. Public artifacts must remain
derived outputs and declare whether raw provider data is published.

## Next model stage

The first CFB model stage will pull historical games, plays, drives, returning production,
portal movement, talent, recruiting, coaches, and historical lines into private cached inputs.
It will then:

1. restrict the initial training benchmark to FBS-vs-FBS games;
2. build features strictly from information available before each kickoff;
3. opponent-adjust efficiency, success, explosiveness, tempo, and field-position measures;
4. add preseason returning-production, transfer, talent, and coaching priors;
5. evaluate margin and total models with expanding weekly folds;
6. compare against transparent rating and historical market baselines;
7. publish predictions only if held-out results and calibration clear documented gates.

Player props, live play-by-play, and GraphQL are out of scope until the game models are stable.
