# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/irfan-hilmi/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/irfan-hilmi/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for each of Grand Theft Auto VI's 55 PlayStation 5 trophies, five
quantiles of its Exophase earn rate (%) at a target time — a week after
launch for the final case. `main.py` (Python 3.13, standard library only)
reads a case file and writes one forecast file per case.

## The model

An earn rate is a ratio, earners over players tracked, and the two sides move
for different reasons. New players dilute every rate when Exophase refreshes
its counts (launch night: 638 tracked, then 3,868 at the first refresh, every
rate but 100% down), while earners grow as the player base works through the
game (early story trophies fast, the platinum and grinds slowly). So the
program fits the two sides separately, in log-log time since release:
log(players) and, per trophy, log(earners) against log(1 + hours), fitted on
the last 24 hours of reads and pivoted at the last read. Each slope is shrunk
lightly toward the cross-trophy median slope and the extrapolation is damped,
because cumulative adoption curves bend down in log-log space. The median
forecast is predicted earners over predicted players.

The spread is an honest one for a multi-day horizon: a random-walk term
growing with the square root of the horizon, plus terms for how jumpy the
denominator has been and how uncertain each trophy's trend is, floored so a
refresh-sized move stays inside the 90% band. The trophy every tracked player
already holds (100%, the opening mission) is forecast to stay at ~99.9% with
a narrow spread. When the history is flat and the horizon is short, the
program expects a refresh: the denominator roughly doubles and earners rise
with the new players. Everything is deterministic: no randomness, no network,
no clock reads beyond the case file itself.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

This reads every `*.in` case in `input/` and writes `<id>.out` to `out/`:
`{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}`, one row per
trophy. Runs in ~0.1 s per case.

---

<sub>Entered on [make.fun](https://make.fun/?ref=zr9ef9xb&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
