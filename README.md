# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/isnoop4/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/isnoop4/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for each of Grand Theft Auto VI's 55 PlayStation 5 trophies, five
quantiles of its Exophase earn rate (%) at a target time — a week after
launch for the final case. `main.py` (Python 3.13, standard library only)
reads a case file and writes one forecast file per case.

## The model (v2)

An earn rate is a ratio, earners over players tracked, and the two sides move
for different reasons. Exophase does not update its counts continuously: the
page sits frozen for hours, then jumps at a refresh (launch night: 638
tracked for hours, then 3,868 at the first refresh, every rate but 100%
down). So v2 treats the history as a refresh process, not a smooth curve.
When distinct reads are hours apart, it estimates the typical per-refresh
jump in log players and in log earners per trophy from the recent refreshes
(medians, so one backlog-clearing jump is not read as the new normal) and
compounds them out to the target — with one shared damping factor for both
sides, and with the per-refresh jumps decaying as they are compounded, the
discrete version of a power law (a player base is finite; growth is concave
in log time). When the history is continuous drift instead, or fewer than two
refreshes are visible, it falls back to fitting each side separately in
log-log time since release: log(players) and, per trophy, log(earners)
against log(1 + hours), fitted on the last 24 hours of reads and pivoted at
the last read, slopes shrunk lightly toward the cross-trophy median. The
median forecast is predicted earners over predicted players.

Two more facts shape the forecast. The answer is the page as of Exophase's
last refresh before the target, not the true state at the target, so the
program aims about four hours early and adds spread for the timing
uncertainty. And the spread is asymmetric: a refresh dilutes rates, so the
downside (rates lower than the median) is wider than the upside, by a factor
of about 1.3 — honest and wide for a multi-day horizon. The trophy every
tracked player already holds (100%, the opening mission) is forecast to stay
at ~99.9% with a narrow spread. When the history is flat and the horizon is
short, the program expects a refresh: the denominator roughly six-times and
earners rise ~1.2x. Everything is deterministic: no randomness, no network,
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
