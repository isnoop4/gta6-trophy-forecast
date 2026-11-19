"""GTA VI trophy-rarity forecast for make.fun (gta6-trophy-rarity-forecast-01).

Forecasts, for each of the game's 55 PS5 trophies, five quantiles of its
Exophase earn rate (%) at the case's target time.

Model, in one paragraph: an earn rate is a ratio, earners over players
tracked, and the two sides of the ratio move for different reasons. New
players dilute every rate when Exophase refreshes its counts (launch night:
638 tracked, then 3,868 at the first refresh, every rate but 100% down), while
earners grow as the player base works through the game (early story trophies
fast, the platinum and grinds slowly). So the program fits the two sides
separately, in log-log time since release: log(players) and, per trophy,
log(earners) against log(1 + hours), fitted on the last 24 hours of reads
(the recent window tracks the current phase of a bending adoption curve) and
pivoted at the last read, so the fitted level there is the base the forecast
grows from. Each slope is shrunk lightly toward the cross-trophy median slope
and the extrapolation is damped, because cumulative adoption curves bend down
in log-log space. The median forecast is predicted earners over predicted
players. The spread is an honest one for a multi-day horizon: a random-walk
term growing with the square root of the horizon, plus terms for how jumpy
the denominator has been and how uncertain each trophy's trend is, floored
so a refresh-sized move stays inside the 90% band. The trophy every tracked
player already holds (100%) is forecast to stay at ~99.9% with a narrow
spread. Everything is deterministic: no randomness, no network, no clock
reads at forecast time beyond the case file itself, standard library only.

Usage (as the grader runs it): python3 main.py, reading every *.in file in
$MAKEFUN_INPUT_DIR (else /input) and writing <id>.out to $MAKEFUN_OUTPUT_DIR
(else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Spread of a rate's move, in logit units per square-root day (random-walk part).
SIGMA_DAILY = 0.45
# Weight of trend uncertainty in the spread: slope error grows with the
# log-time distance extrapolated.
TREND_COEF = 0.25
# How much of the log-time extrapolation to keep: adoption curves bend down.
DAMP_EARNERS = 0.85
DAMP_PLAYERS = 0.90
# Slope shrinkage: w = k / (k + SHRINK_K), k = distinct reads for the trophy.
# With a dozen-plus distinct reads the per-trophy slope is fairly precise, so
# shrinkage is light; it mainly steadies trophies with few distinct reads.
SHRINK_K = 2.0
# Fit window, in hours before the checkpoint (None = whole history). The
# recent window tracks the current phase of a bending adoption curve.
FIT_WINDOW_HOURS = 24.0
# Spread floor / cap, in logit units.
SIGMA_FLOOR = 0.8
SIGMA_FLAT_SHORT = 1.0
SIGMA_CAP = 2.2
# Flat-history short-horizon refresh prior: a refresh roughly doubles the
# tracked count and new players earn early trophies, so earners rise ~40%.
FLAT_REFRESH_N_MULT = 2.0
FLAT_REFRESH_E_MULT = 1.4


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit_of_rate(rate):
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def rate_of_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4_half_up(x):
    """Round to 4 decimals, halves up. Whole numbers come back as ints."""
    scaled = math.floor(x * 10000 + 0.5)
    value = scaled / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def weighted_line(xs, ys, ws):
    """Weighted least squares for y = a + b*x. Returns (a, b, rms_resid)."""
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx < 1e-12:
        return my, 0.0, 0.0
    b = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / sxx
    a = my - b * mx
    resid = math.sqrt(
        sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
    )
    return a, b, resid


def median_of(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def dedupe_reads(history):
    """Drop consecutive reads that change nothing (Exophase holds its page
    between refreshes); they would overweight flat stretches in the fits."""
    kept = []
    prev_key = None
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if key != prev_key:
            kept.append(read)
            prev_key = key
    return kept


def forecast_case(case):
    trophies = case["trophies"]
    history = dedupe_reads(case.get("history") or [])
    horizon_days = case["horizon_minutes"] / 1440.0

    release = datetime(2026, 11, 19, tzinfo=timezone.utc)
    try:
        release = datetime.fromisoformat(
            case["game"]["release_date"] + "T00:00:00+00:00"
        )
    except Exception:
        pass

    def x_of(moment):
        hours = max(0.0, (parse_time(moment) - release).total_seconds() / 3600.0)
        return math.log1p(hours)

    if not history:
        return {"trophies": [
            {"id": t["id"], "q": _quantiles(_tier_default(t), 1.5)}
            for t in trophies
        ]}

    last = history[-1]
    last_rates = last["rates"]
    n_last = float(last["players_tracked"])
    x_last = x_of(last["at"])
    x_target = x_of(case["target"])

    xs_all = [x_of(r["at"]) for r in history]
    n_reads = len(history)
    weights_all = [0.5 + 0.5 * (j / max(1, n_reads - 1)) for j in range(n_reads)]
    if FIT_WINDOW_HOURS is not None:
        cutoff = x_of(last["at"]) - math.log1p(FIT_WINDOW_HOURS)
        # x is log1p(hours); approximate the window in x-space by keeping
        # reads whose clock time is within the window of the last read.
        last_moment = parse_time(last["at"])
        keep = [
            j for j, r in enumerate(history)
            if (last_moment - parse_time(r["at"])).total_seconds()
            <= FIT_WINDOW_HOURS * 3600
        ]
        if len(keep) >= 3:
            xs = [xs_all[j] for j in keep]
            weights = [weights_all[j] for j in keep]
            fit_idx = keep
        else:
            xs, weights, fit_idx = xs_all, weights_all, list(range(n_reads))
    else:
        xs, weights, fit_idx = xs_all, weights_all, list(range(n_reads))
    x_span = (max(xs) - min(xs)) if xs else 0.0
    n_flat = all(r["players_tracked"] == history[0]["players_tracked"]
                 for r in history)

    # --- denominator: players tracked -------------------------------------
    log_n = [math.log(max(1.0, float(history[j]["players_tracked"])))
             for j in fit_idx]
    a_n, b_n_raw, _ = weighted_line(xs, log_n, weights)
    # Level at the last read, from the raw fit: the slope is shrunk below,
    # and a slope change must not move the level (intercept and slope are
    # coupled -- pivoting at x_last keeps the level fixed).
    log_n_level = a_n + b_n_raw * x_last
    b_n = b_n_raw
    if len(fit_idx) < 3 or x_span < 0.3:
        b_n = 0.6  # uninformative history: power-law growth prior
    b_n = min(2.0, max(0.0, b_n))
    log_n_pred = log_n_level + b_n * DAMP_PLAYERS * max(0.0, x_target - x_last)

    # --- per-trophy earners ----------------------------------------------
    raw_slopes = []
    per_trophy = []
    for i, trophy in enumerate(trophies):
        pts = []
        for pos, j in enumerate(fit_idx):
            read = history[j]
            rate = read["rates"][i] if read["rates"] else None
            if rate is None:
                continue
            earners = max(0.25, rate / 100.0 * float(read["players_tracked"]))
            pts.append((xs[pos], math.log(earners), weights[pos]))
        if len(pts) >= 3 and (max(p[0] for p in pts) - min(p[0] for p in pts)) > 0.3:
            a_i, b_i, resid_i = weighted_line(
                [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]
            )
            raw_slopes.append(b_i)
            # Store the fitted level at the last read, not the intercept:
            # shrinking the slope below must not move the level.
            per_trophy.append((a_i + b_i * x_last, b_i, resid_i, len(pts)))
        else:
            per_trophy.append((None, None, 0.0, len(pts)))

    slope_median = median_of(raw_slopes) if raw_slopes else 0.0
    slope_median = min(2.5, max(-0.5, slope_median))

    # Biggest denominator jump seen, in log units (refresh risk gauge),
    # over the whole deduped history.
    log_n_all = [math.log(max(1.0, float(r["players_tracked"]))) for r in history]
    jump = 0.0
    for j in range(1, len(log_n_all)):
        step = abs(log_n_all[j] - log_n_all[j - 1])
        jump = max(jump, step)

    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = last_rates[i] if last_rates else None
        if last_rate is None:
            last_rate = _tier_default(trophy)
        if last_rate >= 99.99:
            # Everybody tracked already holds it (the opening mission): it
            # stays ~universal, new players earn it within minutes.
            rows.append({"id": trophy["id"], "q": _quantiles(99.9, 0.25)})
            continue

        a_i, b_i, resid_i, k = per_trophy[i]
        if b_i is None:
            b_i = slope_median
        else:
            w = k / (k + SHRINK_K)  # more distinct reads -> trust the trophy's own slope
            b_i = w * b_i + (1.0 - w) * slope_median
        b_i = min(3.0, max(-0.5, b_i))

        if a_i is None:
            earners_last = max(0.25, last_rate / 100.0 * n_last)
            log_e_base = math.log(earners_last)
        else:
            log_e_base = a_i  # already the fitted level at the last read
        log_e_pred = log_e_base + b_i * DAMP_EARNERS * max(0.0, x_target - x_last)

        if n_flat and case["horizon_minutes"] < 360:
            # No refresh has shown up yet in a flat history, but the next
            # one can land inside a short horizon: expect the denominator to
            # roughly double and earners to rise with the new players.
            log_n_pred_i = math.log(n_last * FLAT_REFRESH_N_MULT)
            log_e_pred_i = math.log(
                max(0.25, last_rate / 100.0 * n_last) * FLAT_REFRESH_E_MULT
            )
        else:
            log_n_pred_i = log_n_pred
            log_e_pred_i = log_e_pred

        median = 100.0 * math.exp(log_e_pred_i - log_n_pred_i)
        median = min(95.0, clamp_rate(median))

        base = SIGMA_DAILY * math.sqrt(horizon_days)
        jump_part = 0.6 * min(jump, 2.0)
        trend_part = TREND_COEF * abs(b_i - b_n) * max(0.0, x_target - x_last) \
            + min(resid_i, 0.5)
        sigma = math.sqrt(base ** 2 + jump_part ** 2 + trend_part ** 2)
        floor = SIGMA_FLAT_SHORT if (n_flat and case["horizon_minutes"] < 360) \
            else SIGMA_FLOOR
        sigma = min(SIGMA_CAP, max(floor, sigma))
        rows.append({"id": trophy["id"], "q": _quantiles(median, sigma)})

    return {"trophies": rows}


def _tier_default(trophy):
    return {"platinum": 0.5, "gold": 1.0, "silver": 0.8}.get(
        trophy.get("tier"), 1.0)


def _quantiles(median, sigma):
    centre = logit_of_rate(median)
    out = []
    prev = None
    for z in Z_SCORES:
        value = round4_half_up(rate_of_logit(centre + z * sigma))
        value = min(100.0, max(0.0, value))
        if prev is not None and value < prev:
            value = prev
        out.append(value)
        prev = value
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
