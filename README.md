# GTA VI trophy rarity forecast

[![make.fun bounty](https://img.shields.io/badge/make.fun-bounty-7c3aed)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![make.fun prize](https://img.shields.io/badge/prize-%24250%20USDC-16a34a)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![make.fun places](https://img.shields.io/badge/places-top%2020-2563eb)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI PlayStation 5 trophy's Exophase earn rate
at 2026-11-26T00:00:00Z, as five quantiles a trophy, for the
[make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
(`gta6-trophy-rarity-forecast-01`, $250 USDC, top 20). One program,
`main.py` (Python 3.13, standard library only, deterministic, no network):
it reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
`/output/<id>.out` (or `$MAKEFUN_OUTPUT_DIR`).

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Why this shape

An earn rate is earners / players-tracked, and both move. Exophase refreshes
a game's counts on its own schedule (hours apart, not continuously): between
refreshes the page is frozen, and at a refresh the tracked base jumps as new
players sync in, so hard trophies' rates **fall** before they rise again.
The practice cases are exactly that: every read before the checkpoint is one
frozen launch-night page, and the answer is the refreshed page. One refresh
over 90 minutes is little guide to a week, so nothing here is tuned to the
practice cases.

The forecaster, per trophy, in logit space (`logit(p) = ln(p / (1 - p))`):

- `x = ln(1 + hours since release)`. Trophy adoption is roughly linear in
  logit against log time: players reach a milestone at lognormally spread
  times, so the share holding it grows like a CDF in log time.
- Three median candidates, blended by a last-step holdout on the last two
  distinct reads:
  - **persistence** — the rate stays where the last read put it;
  - **log-time logit trend** — a least-squares line of `logit(rate)` on `x`,
    anchored at the last read, slope shrunk toward the tier's slope
    (platinum/gold/silver/bronze grow at very different speeds);
  - **earners-vs-tracked** — `log(earners)` on `x` and `log(players
    tracked)` on `x` fitted separately and divided at the target, so
    dilution (tracked players flooding in) and progress (earners growing)
    are not confused with each other.
- Consecutive identical reads carry no new information (a frozen page), so
  every fit uses only distinct values, at their first-seen time.
- The blend leans on persistence for short horizons and on the trend for
  long ones: `w = horizon_days / (horizon_days + 0.25)`.
- The spread is honest about a 4.5-day extrapolation: a random-walk term
  (the house baseline's `0.5 * sqrt(days)`), the trend fit's own
  uncertainty, the disagreement between the trend and earners models, and a
  refresh term — each refresh can jump logit rates, and the answer is the
  page as of its last refresh before the target — on top of a floor. When
  no refresh has been seen yet at all, the refresh risk is kept as what it
  is: a two-branch mixture, the page staying frozen or refreshing and
  diluting every rate down, with the refresh chance estimated from the
  flat span. A trophy at 100% stays 100%.

Quantiles are `q_tau = inv_logit(median_logit + z_tau * sigma)`, rounded to
4 decimals half-up. If the model fails on an odd case, a persistence
fallback still writes a valid forecast — an invalid case forfeits prizes.

## Validation

The scorer is my own implementation of the RULES.txt pinball loss on the
logit scale; it reproduces the rules' worked example exactly
(loss 0.403404, typical miss 7, coverage 0.5), and the kit starter scored
through it gives 1.467 on the practice cases against answers reconstructed
from the two published aggregates (04:30Z read: 3,868 players tracked,
4,276 trophies earned; every rate but the 100% one lower) — close to the
published "about 1.51" (typical miss 0.69 vs ~0.68, coverage 0 vs 0). The
practice answers themselves are not in the kit, so those practice numbers
are indicative, not exact.

- Practice (reconstructed answers): this entry **0.80 mean loss**
  (p00 0.799, p01 0.812) vs house baseline **1.47** — the gain is the honest
  spread coping with a refresh it has not seen, not a shifted median.
- Synthetic week-long test (a simulated launch week with 14 refreshes, cut
  at a deadline-like checkpoint, target 4.5 days later): this entry beats
  persistence on the long horizon under three different ground-truth
  shapes — logit-linear in log time **0.30 vs 0.54**, logit-linear in hours
  (accelerating) **1.22 vs 1.57**, saturating **0.31 vs 0.45**.
- Determinism: two runs on the practice cases write byte-identical
  outputs. Runtime: ~0.06 s per case locally (limit 8 s).

## License

MIT — see [LICENSE](LICENSE).
