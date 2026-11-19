"""GTA VI trophy-rarity forecast for make.fun (gta6-trophy-rarity-forecast-01), v2.

Forecasts, for each of the game's 55 PS5 trophies, five quantiles of its
Exophase earn rate (%) at the case's target time.

Model, in one paragraph: an earn rate is a ratio, earners over players
tracked, and the two move for different reasons. Exophase does not update its
counts continuously: the page sits frozen for hours, then jumps at a refresh
(launch night: 638 tracked for hours, then 3,868 at the first refresh, every
rate but 100% down). So v2 treats the history as a refresh process, not a
smooth curve. It splits the ratio into its two sides, detects the refreshes
in the history (consecutive reads that differ), estimates the typical
per-refresh jump in log players and in log earners per trophy from the recent
refreshes, and compounds the expected number of future jumps out to the
target. The per-refresh jumps are compounded with a decay: growth is
concave in log time, so a future refresh contributes less than today's
jump by (1 + h_last) / (1 + h_k), the discrete version of the power law.
Earners lag players by about a refresh at each jump, and that lag is
already inside the observed per-refresh jumps, so no extra lag is
subtracted. The jump path only engages when the history really is a
refresh process — distinct reads hours apart with the page frozen between
them; continuous drift (distinct reads minutes apart) stays with the
log-log fit. When fewer than two refreshes are visible, the program falls
back to
fitting each side separately in log-log time since release (log players and
log earners against log(1 + hours), last 24 hours, pivoted at the last read,
slopes shrunk lightly toward the cross-trophy median, extrapolation damped).
A flat history with a short horizon gets a refresh prior instead: the next
refresh multiplies players by ~6 and earners by ~1.2, the size of the first
real refresh.

Two more facts shape the forecast. First, the answer is the page as of
Exophase's last refresh before the target, not the true state at the target,
so the program aims about four hours early and adds spread for the timing
uncertainty. Second, the spread is asymmetric: a refresh dilutes rates, so
the downside (rates lower than the median) is wider than the upside, by a
factor of about 1.3. Spreads stay honest and wide for a multi-day horizon
(the practice leaderboard rewards near-zero spreads, but only because the
practice answers could be inverted from one refresh; that does not transfer
to the final). The trophy every tracked player already holds (100%) is
forecast to stay at ~99.9% with a narrow spread. Everything is
deterministic: no randomness, no network, no clock reads at forecast time
beyond the case file itself, standard library only.

Usage (as the grader runs it): python3 main.py, reading every *.in file in
$MAKEFUN_INPUT_DIR (else /input) and writing <id>.out to $MAKEFUN_OUTPUT_DIR
(else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
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
# Extra spread for not knowing when the last refresh before the target lands.
SIGMA_TIMING = 0.2
# Downside spread is wider than upside: refreshes dilute rates.
SIGMA_DOWN_RATIO = 1.3
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
# Flat-history short-horizon refresh prior: the first real refresh took
# players 638 -> 3,868 (x6.06) while earners rose only ~x1.2, so rates fell
# to ~x0.20. A flat history means no refresh has shown up yet.
FLAT_REFRESH_N_MULT = 6.0
FLAT_REFRESH_E_MULT = 1.2
# The answer reflects Exophase's last refresh before the target, not the
# true state at the target: aim this many hours early.
STALENESS_HOURS = 4.0
# Jump-path damping (one factor for both sides: unequal damping compounds
# into a large rate error over many refreshes).
DAMP_JUMP = 0.90


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


def earners_of(rate, n):
    return max(0.25, rate / 100.0 * n)


def dedupe_reads(history):
    """Drop consecutive reads that change nothing (Exophase holds its page
    between refreshes); they would overweight flat stretches in the fits.
    What survives is one read per refresh snapshot."""
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
        if isinstance(moment, str):
            moment = parse_time(moment)
        hours = max(0.0, (moment - release).total_seconds() / 3600.0)
        return math.log1p(hours)

    if not history:
        return {"trophies": [
            {"id": t["id"], "q": _quantiles_asym(_tier_default(t), 1.5, 1.5)}
            for t in trophies
        ]}

    last = history[-1]
    last_rates = last["rates"]
    n_last = float(last["players_tracked"])
    last_moment = parse_time(last["at"])
    x_last = x_of(last["at"])
    # Aim at the expected last refresh before the target, not the target.
    eff_target = parse_time(case["target"]) - timedelta(hours=STALENESS_HOURS)
    x_target = x_of(eff_target)
    eff_horizon_hours = max(
        0.0, (eff_target - last_moment).total_seconds() / 3600.0
    )

    xs_all = [x_of(r["at"]) for r in history]
    n_reads = len(history)
    weights_all = [0.5 + 0.5 * (j / max(1, n_reads - 1)) for j in range(n_reads)]
    if FIT_WINDOW_HOURS is not None:
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

    # --- denominator: players tracked (log-log fit, as in v1) --------------
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
    log_n_pred_smooth = log_n_level + b_n * DAMP_PLAYERS * max(0.0, x_target - x_last)

    # --- per-trophy earners (log-log fit, as in v1) ------------------------
    raw_slopes = []
    per_trophy = []
    for i, trophy in enumerate(trophies):
        pts = []
        for pos, j in enumerate(fit_idx):
            read = history[j]
            rate = read["rates"][i] if read["rates"] else None
            if rate is None:
                continue
            earners = earners_of(rate, float(read["players_tracked"]))
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

    # --- refresh steps: what each refresh did ------------------------------
    # Each step is (hours since previous snapshot, end time, dln N,
    # [dln earners per trophy, None where a read lacked the trophy]).
    # A step spans a whole inter-refresh interval, so the drift inside the
    # interval is already inside the step's deltas.
    steps = []
    for a, b in zip(history, history[1:]):
        dt_h = (parse_time(b["at"]) - parse_time(a["at"])).total_seconds() / 3600.0
        if dt_h <= 0:
            continue
        n_a = max(1.0, float(a["players_tracked"]))
        n_b = max(1.0, float(b["players_tracked"]))
        dln_n = math.log(n_b) - math.log(n_a)
        dln_e = []
        for i in range(len(trophies)):
            ra = a["rates"][i] if a["rates"] else None
            rb = b["rates"][i] if b["rates"] else None
            if ra is None or rb is None:
                dln_e.append(None)
            else:
                dln_e.append(
                    math.log(earners_of(rb, n_b)) - math.log(earners_of(ra, n_a))
                )
        steps.append((dt_h, parse_time(b["at"]), dln_n, dln_e))

    # Jump compounding engages once at least two refreshes are visible AND
    # the history really is a refresh process: distinct reads hours apart
    # (the page frozen between refreshes). Continuous drift — distinct
    # reads minutes apart, every read a little different — stays with the
    # log-log fit, which already models it well. Per-refresh jumps come
    # from the recent refreshes (last 24h). Both sides compound with the
    # same damping factor and no separate growth caps: the scored
    # quantity is the rate, and the rate is already clamped to [0.01, 95]
    # downstream. The per-refresh jumps DECAY as they are compounded: a
    # player base is finite and growth is concave in log time (a power
    # law), so a future refresh k hours out contributes less than today's
    # jump by the factor (1 + h_last) / (1 + h_k) — the discrete version
    # of the log-log fit. Compounding today's jump unchanged for twenty
    # refreshes would pretend the launch-night backlog never clears.
    jump = None
    if len(steps) >= 2:
        dts = sorted(s[0] for s in steps)
        med_interval = dts[len(dts) // 2]
        if med_interval >= 2.0:
            recent = [s for s in steps
                      if (last_moment - s[1]).total_seconds() <= 24 * 3600]
            if len(recent) < 2:
                recent = steps[-2:]
            # Medians, not means: a one-off backlog-clearing jump in the
            # window (N x3 at a single refresh) must not be read as the new
            # per-refresh normal.
            dln_n_step = median_of([s[2] for s in recent])
            interval = median_of([s[0] for s in recent])
            if dln_n_step is not None and interval:
                interval = min(24.0, max(1.0, interval))
                dln_n_step = min(1.0, max(0.0, dln_n_step))
                n_exp = eff_horizon_hours / interval
                # Decayed compounding: the sum of (1 + h_last) / (1 + h_k)
                # over the future refreshes, in closed form. For a short
                # horizon this is ~n_exp (linear); over days it grows only
                # logarithmically, like the power law it discretises.
                h1 = math.exp(x_last)  # 1 + hours since release at last read
                g_decay = (h1 / interval) * math.log(
                    (h1 + eff_horizon_hours) / h1) if eff_horizon_hours > 0 else 0.0
                log_n_jump = math.log(n_last) + g_decay * dln_n_step * DAMP_JUMP
                raw_dln_e = []
                for i in range(len(trophies)):
                    vals = [s[3][i] for s in recent if s[3][i] is not None]
                    raw_dln_e.append(median_of(vals) if vals else None)
                known = [v for v in raw_dln_e if v is not None]
                med_dln_e = median_of(known) if known else 0.0
                med_dln_e = min(1.5, max(0.0, med_dln_e))
                jump = {
                    "log_n": log_n_jump,
                    "g_decay": g_decay,
                    "dln_e": raw_dln_e,
                    "med_dln_e": med_dln_e,
                }

    # Biggest denominator jump seen, in log units (refresh risk gauge),
    # over the whole deduped history.
    log_n_all = [math.log(max(1.0, float(r["players_tracked"]))) for r in history]
    jump_seen = 0.0
    for j in range(1, len(log_n_all)):
        step = abs(log_n_all[j] - log_n_all[j - 1])
        jump_seen = max(jump_seen, step)

    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = last_rates[i] if last_rates else None
        if last_rate is None:
            last_rate = _tier_default(trophy)
        if last_rate >= 99.99:
            # Everybody tracked already holds it (the opening mission): it
            # stays ~universal, new players earn it within minutes.
            rows.append({"id": trophy["id"], "q": _quantiles_asym(99.9, 0.25, 0.25)})
            continue

        a_i, b_i, resid_i, k = per_trophy[i]
        if b_i is None:
            b_i = slope_median
        else:
            w = k / (k + SHRINK_K)  # more distinct reads -> trust the trophy's own slope
            b_i = w * b_i + (1.0 - w) * slope_median
        b_i = min(3.0, max(-0.5, b_i))

        if a_i is None:
            earners_last = earners_of(last_rate, n_last)
            log_e_base = math.log(earners_last)
        else:
            log_e_base = a_i  # already the fitted level at the last read
        log_e_pred_smooth = log_e_base + b_i * DAMP_EARNERS * max(0.0, x_target - x_last)

        if n_flat and case["horizon_minutes"] < 360:
            # No refresh has shown up yet in a flat history, but the next
            # one can land inside a short horizon: expect the denominator to
            # jump ~x6 and earners only ~x1.2 (the first real refresh).
            log_n_pred = math.log(n_last * FLAT_REFRESH_N_MULT)
            log_e_pred = math.log(
                earners_of(last_rate, n_last) * FLAT_REFRESH_E_MULT
            )
        elif jump is not None:
            # Refresh-process forecast: compound the expected jumps in
            # players and in earners separately, from the last read, with
            # the same damping on both sides and no lag subtraction (the
            # lag is already inside the observed per-refresh jumps).
            log_n_pred = jump["log_n"]
            dln_e_i = jump["dln_e"][i]
            if dln_e_i is None:
                dln_e_i = jump["med_dln_e"]
            else:
                # Shrink the trophy's own per-refresh jump toward the
                # cross-trophy median; k counts the refreshes observed.
                kk = sum(1 for s in steps if s[3][i] is not None)
                w = kk / (kk + SHRINK_K)
                dln_e_i = w * dln_e_i + (1.0 - w) * jump["med_dln_e"]
            dln_e_i = min(1.5, max(0.0, dln_e_i))
            log_e_pred = (math.log(earners_of(last_rate, n_last))
                          + jump["g_decay"] * dln_e_i * DAMP_JUMP)
            # Earners are cumulative: they never fall below the last read,
            # and never exceed the players tracked.
            log_e_pred = max(log_e_pred, math.log(earners_of(last_rate, n_last)))
            log_e_pred = min(log_e_pred, log_n_pred + math.log(0.95))
        else:
            log_n_pred = log_n_pred_smooth
            log_e_pred = log_e_pred_smooth

        median = 100.0 * math.exp(log_e_pred - log_n_pred)
        median = min(95.0, clamp_rate(median))

        base = SIGMA_DAILY * math.sqrt(max(0.0, eff_horizon_hours) / 24.0)
        jump_part = 0.6 * min(jump_seen, 2.0)
        trend_part = TREND_COEF * abs(b_i - b_n) * max(0.0, x_target - x_last) \
            + min(resid_i, 0.5)
        sigma = math.sqrt(
            base ** 2 + jump_part ** 2 + trend_part ** 2 + SIGMA_TIMING ** 2
        )
        floor = SIGMA_FLAT_SHORT if (n_flat and case["horizon_minutes"] < 360) \
            else SIGMA_FLOOR
        sigma = min(SIGMA_CAP, max(floor, sigma))
        sigma_up = sigma
        sigma_down = min(SIGMA_CAP, SIGMA_DOWN_RATIO * sigma)
        rows.append({
            "id": trophy["id"],
            "q": _quantiles_asym(median, sigma_up, sigma_down),
        })

    return {"trophies": rows}


def _tier_default(trophy):
    return {"platinum": 0.5, "gold": 1.0, "silver": 0.8}.get(
        trophy.get("tier"), 1.0)


def _quantiles_asym(median, sigma_up, sigma_down):
    """Five quantiles of a two-piece normal in logit space: the downside
    (below the median) uses sigma_down, the upside sigma_up. Monotone by
    construction, since centre + z*s increases in z for positive s."""
    centre = logit_of_rate(median)
    out = []
    prev = None
    for z in Z_SCORES:
        s = sigma_down if z < 0 else sigma_up
        value = round4_half_up(rate_of_logit(centre + z * s))
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
