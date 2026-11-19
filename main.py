"""gta6-trophy-rarity-forecast-01: a dilution-aware, log-time logit forecaster.

Forecasts every GTA VI PS5 trophy's Exophase earn rate at the case target as
five quantiles, from the case history alone. Python 3.13, standard library
only, deterministic, no network.

Why this shape, and what the history can and cannot say
-------------------------------------------------------
An earn rate is earners / players-tracked, and both move. Exophase refreshes
a game's counts on its own schedule (hours apart, not continuously): between
refreshes the page is frozen, and at a refresh the tracked base jumps as new
players sync in, so hard trophies' rates FALL before they rise again. The
practice cases are exactly that: every read before the checkpoint is one
frozen page, and the answer is the refreshed page. One refresh over 90
minutes is little guide to a week, so nothing here is tuned to the practice.

The model, per trophy, in logit space (logit(p) = ln(p / (1 - p))):
  * x = ln(1 + hours since release). Trophy adoption is roughly linear in
    logit against log time: players reach a milestone at lognormally spread
    times, so the share holding it grows like a CDF in log time.
  * Three median candidates, blended by a last-step holdout:
      A. persistence -- the rate stays where the last read put it.
      B. log-time logit trend -- a least-squares line of logit(rate) on x,
         anchored at the last read, slope shrunk toward the tier's slope
         (platinum/gold/silver/bronze grow at very different speeds).
      C. earners-vs-tracked -- fit log(earners) on x and log(players
         tracked) on x separately and divide at the target, so dilution
         (tracked players flooding in) and progress (earners growing) are
         not confused with each other.
    Consecutive identical reads carry no new information (a frozen page),
    so every fit uses only distinct values, at their first-seen time.
  * The blend leans on persistence for short horizons and on the trend for
    long ones: w = horizon_days / (horizon_days + 0.25).
  * The spread is honest about a 4.5-day extrapolation: a random-walk term
    (the house baseline's 0.5 * sqrt(days)), the trend fit's own
    uncertainty, the disagreement between models B and C, and a refresh
    term -- each refresh can jump logit rates, and the answer is the page
    as of its last refresh before the target -- on top of a floor. When no
    refresh has been seen yet at all, the refresh risk is not smeared into
    a wider normal but kept as what it is: a two-branch mixture, the page
    staying frozen or refreshing and diluting every rate down, with the
    refresh chance estimated from the flat span. A trophy at 100% stays
    100%: every tracked player holds it, and the one refresh on record
    left it there.

Quantiles are q_tau = inv_logit(median_logit + z_tau * sigma), rounded to
4 decimals half-up. A forecast whose quantiles are equal scores its mean
absolute logit error, so the median and the spread are scored on one scale.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE_FALLBACK = datetime(2026, 11, 19, tzinfo=timezone.utc)

# A refresh's typical logit jump when none is observed yet (launch night's
# first refresh moved every non-100% rate down by ~1.5 logit units; later
# refreshes are smaller, so 1.0 is the prior, not the fitted size).
PRIOR_JUMP = 1.0
# Typical hours between Exophase refreshes when none is observed yet.
PRIOR_REFRESH_HOURS = 5.0
# Minimum spread, in logit units: never claim false precision.
SIGMA_FLOOR = 0.35
# Widest spread allowed: wider than this, the quantiles say nothing.
SIGMA_CAP = 2.2


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up, as the kit starter rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def phi(z):
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mixture_quantile(tau, center, sigma, p_refresh, jump):
    """tau-quantile of a two-branch logit distribution: with probability
    1 - p the rate sits at center (the page stays frozen), with probability
    p it has refreshed and dropped by `jump` logit units. Solved by
    bisection on the mixture CDF; deterministic."""
    lo = center - jump - 5.0 * sigma
    hi = center + 5.0 * sigma

    def cdf(y):
        return ((1.0 - p_refresh) * phi((y - center) / sigma)
                + p_refresh * phi((y - (center - jump)) / sigma))

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def parse_release(case):
    raw = (case.get("game") or {}).get("release_date") or ""
    try:
        if "T" in raw:
            return parse_time(raw)
        return datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)
    except ValueError:
        return RELEASE_FALLBACK


def ols(xs, ys):
    """Least squares y ~ a + b x. Returns (a, b, resid_std, se_b)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        var = resid / (n - 2)
        se_b = math.sqrt(var / sxx) if sxx > 0 else 0.0
        return a, b, math.sqrt(var), se_b
    return a, b, 0.0, 0.0


def median(values):
    v = sorted(values)
    n = len(v)
    if n == 0:
        return 0.0
    if n % 2:
        return v[n // 2]
    return 0.5 * (v[n // 2 - 1] + v[n // 2])


def distinct_series(points):
    """Collapse consecutive identical values, keeping first-seen times."""
    out = []
    for x, y in points:
        if not out or y != out[-1][1]:
            out.append((x, y))
    return out


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    release = parse_release(case)
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    horizon_hours = case["horizon_minutes"] / 60.0
    x_target = math.log1p(max(0.0, (target - release).total_seconds() / 3600.0))

    trophies = case["trophies"]
    history = case["history"]
    last = history[-1]

    # Per-read clock positions and the tracked-player series.
    read_x = []
    for read in history:
        t = parse_time(read["at"])
        read_x.append(math.log1p(max(0.0, (t - release).total_seconds() / 3600.0)))
    n_series = distinct_series(
        [(x, read["players_tracked"]) for x, read in zip(read_x, history)
         if read.get("players_tracked")]
    )
    if len(n_series) >= 2:
        _, s_n, _, _ = ols([p[0] for p in n_series],
                           [math.log(p[1]) for p in n_series])
        s_n = min(1.5, max(0.0, s_n))
    else:
        s_n = 0.0
    n_last = n_series[-1][1] if n_series else 1

    # Refresh structure: where the rate vector actually changed, how big the
    # typical cross-trophy logit jump was, and how often refreshes came.
    # When NO refresh has been seen yet, the next one is still coming: its
    # chance in the horizon is estimated Bayesian-style from the flat span
    # (prior: one refresh per 5 hours), and the forecast below is then a
    # two-branch mixture -- the page stays frozen, or it refreshes and
    # every rate but the 100% one drops -- instead of one wide normal.
    jumps = []
    gaps = []
    prev_rates = None
    prev_x = None
    for x, read in zip(read_x, history):
        rates = read["rates"]
        if prev_rates is not None and rates != prev_rates:
            diffs = []
            for a, b in zip(prev_rates, rates):
                if a is not None and b is not None and a < 99.99 and b < 99.99:
                    diffs.append(abs(logit(b) - logit(a)))
            if diffs:
                jumps.append(median(diffs))
            if prev_x is not None:
                gaps.append(x - prev_x)
        prev_rates = rates
        prev_x = x
    # Jumps and gaps live in x = ln(1 + hours); convert a gap back to hours
    # at the recent end of the history for the refresh interval.
    if jumps:
        jump = median(jumps[-8:])
    else:
        jump = PRIOR_JUMP
    if gaps:
        # A gap of dx in log time near the last read is about dx * (1 + t) hours.
        t_last_h = max(0.0, (parse_time(last["at"]) - release).total_seconds() / 3600.0)
        interval_h = median(gaps[-8:]) * (1.0 + t_last_h)
        interval_h = min(48.0, max(0.5, interval_h))
    else:
        interval_h = PRIOR_REFRESH_HOURS
    expected_jumps = min(6.0, horizon_hours / interval_h)
    sigma_refresh = min(1.0, jump * math.sqrt(expected_jumps))
    if jumps:
        p_refresh = None  # refreshes are in the history; the trend absorbs them
    else:
        # No refresh observed over the history's span: posterior refresh
        # rate (one pseudo-refresh over the prior interval), and the chance
        # one lands inside the horizon.
        hist_hours = max(0.0, (parse_time(last["at"])
                               - parse_time(history[0]["at"])).total_seconds() / 3600.0)
        lam = 1.0 / (hist_hours + PRIOR_REFRESH_HOURS)
        p_refresh = 1.0 - math.exp(-lam * horizon_hours)

    # First pass: per-trophy trend slopes, for the tier shrinkage targets.
    per_trophy = []
    tier_slopes = {}
    tier_eslopes = {}
    for i, trophy in enumerate(trophies):
        pts = distinct_series([
            (x, read["rates"][i]) for x, read in zip(read_x, history)
            if read["rates"][i] is not None
        ])
        xs = [p[0] for p in pts]
        lg = [logit(p[1]) for p in pts]
        slope = eslope = None
        if len(pts) >= 2:
            _, slope, _, _ = ols(xs, lg)
            e_series = distinct_series([
                (x, (read["rates"][i] / 100.0) * read["players_tracked"])
                for x, read in zip(read_x, history)
                if read["rates"][i] is not None and read.get("players_tracked")
            ])
            if len(e_series) >= 2:
                _, eslope, _, _ = ols([p[0] for p in e_series],
                                      [math.log(max(1.0, p[1])) for p in e_series])
        per_trophy.append({"pts": pts, "slope": slope, "eslope": eslope})
        tier = trophy["tier"]
        if slope is not None:
            tier_slopes.setdefault(tier, []).append(slope)
        if eslope is not None:
            tier_eslopes.setdefault(tier, []).append(eslope)
    tier_slope = {t: median(v) for t, v in tier_slopes.items()}
    tier_eslope = {t: median(v) for t, v in tier_eslopes.items()}
    all_slope = median([s for v in tier_slopes.values() for s in v]) if tier_slopes else 0.0
    all_eslope = median([s for v in tier_eslopes.values() for s in v]) if tier_eslopes else 0.0

    w_horizon = horizon_days / (horizon_days + 0.25)
    rows = []
    for i, trophy in enumerate(trophies):
        info = per_trophy[i]
        pts = info["pts"]
        last_rate = last["rates"][i]
        if last_rate is None:
            last_rate = pts[-1][1] if pts else 1.0
        # The one trophy every tracked player holds stays held.
        if last_rate >= 99.99:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        l_last = logit(last_rate)
        x_last = pts[-1][0] if pts else x_target
        dx = max(0.0, x_target - x_last)

        # Model B: log-time logit trend, anchored at the last read, slope
        # shrunk toward the tier's slope.
        l_trend = None
        sigma_fit = 0.6
        if info["slope"] is not None and len(pts) >= 2:
            n = len(pts)
            w = n / (n + 3.0)
            tier_b = tier_slope.get(trophy["tier"], all_slope)
            b = w * info["slope"] + (1 - w) * tier_b
            b = min(3.0, max(-3.0, b))
            l_trend = l_last + b * dx
            _, _, resid, se_b = ols([p[0] for p in pts],
                                    [logit(p[1]) for p in pts])
            sigma_fit = math.sqrt((se_b * dx) ** 2 + min(1.0, resid) ** 2)
        elif info["slope"] is not None:
            l_trend = l_last

        # Model C: earners-vs-tracked. Earners grow on their own curve;
        # tracked players grow on theirs; the rate is the ratio.
        l_earn = None
        if info["eslope"] is not None and pts:
            n = len(pts)
            w = n / (n + 3.0)
            tier_e = tier_eslope.get(trophy["tier"], all_eslope)
            s_e = w * info["eslope"] + (1 - w) * tier_e
            s_e = min(3.0, max(-1.0, s_e))
            e_last = (last_rate / 100.0) * n_last
            e_tgt = e_last * math.exp(min(3.0 * dx, s_e * dx))
            n_tgt = n_last * math.exp(min(1.5 * dx, s_n * dx))
            r_earn = 100.0 * e_tgt / max(1.0, n_tgt)
            l_earn = logit(r_earn)

        # Model selection by last-step holdout: fit B and C on all but the
        # last two distinct points and see which predicted them best.
        # Persistence predicts the previous point's logit.
        weights = {}
        if len(pts) >= 5 and l_trend is not None and l_earn is not None:
            hold = pts[-2:]
            fit = pts[:-2]
            errs = {}
            # B on the fit set, anchored at the fit set's last point.
            _, b_fit, _, _ = ols([p[0] for p in fit], [logit(p[1]) for p in fit])
            l_fit_last = logit(fit[-1][1])
            errs["trend"] = sum(
                abs((l_fit_last + b_fit * (hx - fit[-1][0])) - logit(hr))
                for hx, hr in hold
            ) / len(hold)
            # C on the fit set: earners slope from the fit set's reads only
            # (reads at or before the fit set's last distinct point), and
            # the tracked count at that same point.
            fit_x_max = fit[-1][0]
            fit_reads = [(x, read) for x, read in zip(read_x, history)
                         if x <= fit_x_max + 1e-12]
            e_fit = distinct_series([
                (x, (read["rates"][i] / 100.0) * read["players_tracked"])
                for x, read in fit_reads
                if read["rates"][i] is not None and read.get("players_tracked")
            ])
            if len(e_fit) >= 2:
                _, s_fit, _, _ = ols([p[0] for p in e_fit],
                                     [math.log(max(1.0, p[1])) for p in e_fit])
                e_fit_last = e_fit[-1][1]
                n_fit_last = fit_reads[-1][1].get("players_tracked") or n_last
                errs["earn"] = sum(
                    abs(logit(100.0 * (e_fit_last * math.exp(s_fit * (hx - e_fit[-1][0])))
                              / max(1.0, n_fit_last * math.exp(s_n * (hx - e_fit[-1][0]))))
                        - logit(hr))
                    for hx, hr in hold
                ) / len(hold)
            else:
                errs["earn"] = errs["trend"]
            errs["persist"] = sum(
                abs(l_fit_last - logit(hr)) for _, hr in hold
            ) / len(hold)
            for k, e in errs.items():
                weights[k] = 1.0 / (e + 0.15)
        if l_trend is None and l_earn is None:
            l_model = l_last
            sigma_dis = 0.0
        elif l_trend is None:
            l_model = l_earn
            sigma_dis = 0.0
        elif l_earn is None:
            l_model = l_trend
            sigma_dis = 0.0
        else:
            wt = weights.get("trend", 1.0)
            we = weights.get("earn", 1.0)
            l_model = (wt * l_trend + we * l_earn) / (wt + we)
            sigma_dis = 0.5 * abs(l_trend - l_earn)

        l_med = (1 - w_horizon) * l_last + w_horizon * l_model
        l_med = min(l_last + 5.0, max(l_last - 5.0, l_med))

        sigma_rw = 0.5 * math.sqrt(horizon_days)
        sigma_small = math.sqrt(sigma_rw ** 2 + sigma_fit ** 2 + sigma_dis ** 2
                                + SIGMA_FLOOR ** 2)
        if p_refresh is not None:
            # No refresh seen yet: the refresh risk is a two-branch mixture,
            # not a wider normal -- the low quantiles live on the refreshed
            # branch, the high ones on the frozen page.
            qs = [round4(clamp_pct(inv_logit(mixture_quantile(
                tau, l_med, sigma_small, p_refresh, PRIOR_JUMP))))
                for tau in LEVELS]
        else:
            sigma = math.sqrt(sigma_small ** 2 + sigma_refresh ** 2)
            sigma = min(SIGMA_CAP, max(0.30, sigma))
            qs = [round4(clamp_pct(inv_logit(l_med + z * sigma))) for z in Z]
        qs = sorted(qs)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def fallback_case(case, name):
    """Persistence with the house baseline's spread: valid output even if
    the model above fails on an odd case. Never let a case go invalid."""
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    last = case["history"][-1]
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        lg = logit(rate if rate is not None else 0.0)
        rows.append({"id": trophy["id"], "q": [
            round4(clamp_pct(inv_logit(lg + z * spread))) for z in Z
        ]})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception as exc:  # keep the case valid, whatever happens
            print(f"{path.name}: model failed ({exc!r}); using fallback",
                  file=sys.stderr)
            forecast = fallback_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
