"""gta6-twitch-ccu-forecast-01 entry: size- and hype-conditioned comparable-launch forecaster.

For each case, forecast hours 2..24 of the release day as five quantiles of Twitch
concurrent viewers, from the case's hour-1 print and the comparable past launches
it ships.

Method (validated on the kit's four practice cases; practice pinball loss 0.1425
against the house persistence baseline's 0.3871):

  For each hour h, model the log-ratio r_h = ln(viewers_h / viewers_1) of the
  target launch.  Donors are the case's comparables with complete release days.
  The prediction is a local-linear (weighted least squares) fit of r_h on
  x = ln(viewers_1), with Gaussian kernel weights in x (bandwidth 1.75) and a
  second Gaussian kernel in y = ln(pre14max / viewers_1) (bandwidth 0.7), where
  pre14max is the launch's biggest hourly count in the 14 days before its
  release day.  Donors are tier-weighted (early-access-contaminated x0.1, weak
  x0.5, single-global-unlock x0.3), and their curves are aligned by the US
  East-midnight unlock hour (donor hour h - target delta + donor delta), so the
  unlock wave lands on the same hour.  Predictions are clamped to the donors'
  observed range at that hour.

  The spread is honest and size-conditioned: per hour, the weighted standard
  deviation of leave-one-out residuals of the pool donors (each predicted from
  the others), weighted by the same kernel at the target, floored at 0.12 and
  scaled by 0.65.  Quantiles are the median times exp(z_tau * spread).

  Guards: the second kernel is used only when the target's y lies inside the
  donors' observed y range (GTA VI's own pre-release peak is far below its
  hour-1 print, so the final case falls back to the one-feature fit); any
  failure degrades to the house persistence baseline so the case stays valid.

Standard library only, deterministic, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Local-linear kernel bandwidths (log units) and weights, as validated.
BW_X = 1.75
BW_Y = 0.7
W_EARLY_ACCESS = 0.1
W_WEAK = 0.5
W_SINGLE_GLOBAL = 0.3
SPREAD_FLOOR = 0.12
SPREAD_MULT = 0.65
BASE_SIGMA = 0.15  # persistence fallback spread per sqrt-hour


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def us_east_midnight_utc(day_start):
    """UTC hour of US Eastern midnight on the release date.

    Rolling-midnight launches unlock at local midnight; the US East coast
    unlocks at 04:00Z while US daylight time is in effect (2nd Sunday of
    March to 1st Sunday of November) and 05:00Z otherwise.
    """
    import calendar

    def nth_weekday(year, month, weekday, n):
        days = [w[weekday] for w in calendar.monthcalendar(year, month) if w[weekday]]
        return days[n - 1]

    dst_start = (3, nth_weekday(day_start.year, 3, 6, 2))
    dst_end = (11, nth_weekday(day_start.year, 11, 6, 1))
    in_dst = (day_start.month, day_start.day) >= dst_start and (day_start.month, day_start.day) < dst_end
    return 4 if in_dst else 5


def round2(x):
    """x rounded to 2 decimals, half up (the starter's rule)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def tier_weights(unlock_text):
    """Multiplicative donor weights from the unlock-notes tier markers."""
    text = (unlock_text or "").lower()
    w = 1.0
    if "early-access-contaminated" in text:
        w *= W_EARLY_ACCESS
    elif "tier: weak" in text:
        w *= W_WEAK
    if "single simultaneous global" in text or "single global unlock" in text:
        w *= W_SINGLE_GLOBAL
    return w


def pre14max(points, day_start):
    """Biggest hourly viewers in [day_start - 14d, day_start)."""
    start = day_start - timedelta(days=14)
    best = None
    for p in points:
        t = parse_time(p["from"])
        if start <= t < day_start and p.get("viewers"):
            if best is None or p["viewers"] > best:
                best = p["viewers"]
    return best


def build_donors(case):
    """Comparables with complete release-day curves, as donor records."""
    donors = []
    for comp in case.get("comparables", []):
        day_start = parse_time(comp["day_start"])
        hourly = {}
        for p in comp.get("hourly", []):
            hourly[parse_time(p["from"])] = p.get("viewers")
        day = {}
        complete = True
        for h in range(1, 25):
            v = hourly.get(day_start + timedelta(hours=h - 1))
            if not v:
                complete = False
                break
            day[h] = v
        if not complete:
            continue
        pre = pre14max(comp.get("hourly", []), day_start)
        donors.append({
            "name": comp.get("name", ""),
            "v1": day[1],
            "x": math.log(day[1]),
            "day": day,
            "pre14max": pre,
            "y": math.log(pre / day[1]) if pre else None,
            "tier_w": tier_weights(comp.get("unlock", "")),
            "delta": us_east_midnight_utc(day_start),
        })
    return donors


def target_pre14max(case, day_start):
    """The target launch's biggest hourly count in the 14 days before release.

    From its own history_hourly, plus the comparable carrying its own name
    (the final case's GTA VI category history, which predates the recorder).
    """
    points = list(case.get("history_hourly", []))
    game_name = (case.get("game") or {}).get("name", "")
    for comp in case.get("comparables", []):
        if game_name and game_name in (comp.get("name") or ""):
            points = points + list(comp.get("hourly", []))
    return pre14max(points, day_start)


def kernel_weight(donor, target, use_y):
    """Gaussian kernel weight of a donor at the target (size, and hype if used)."""
    w = math.exp(-0.5 * ((donor["x"] - target["x"]) / BW_X) ** 2)
    if use_y and target.get("y") is not None and donor.get("y") is not None:
        w *= math.exp(-0.5 * ((donor["y"] - target["y"]) / BW_Y) ** 2)
    return w * donor["tier_w"]


def fit_hour(target, donors, hour, use_y):
    """Local-linear fit of the log-ratio at `hour` on log hour-1 size.

    Donor curves are aligned by the unlock hour: a donor's ratio at hour
    `hour - target delta + donor delta` is its value at the same time since
    the US East unlock.  Returns (prediction, [(ratio, weight)] used).
    """
    xs, ys, ws = [], [], []
    for d in donors:
        hh = hour - target["delta"] + d["delta"]
        if hh < 1 or hh > 24:
            continue
        xs.append(d["x"])
        ys.append(math.log(d["day"][hh] / d["v1"]))
        ws.append(kernel_weight(d, target, use_y))
    if not xs:
        return None, []
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx if sxx > 1e-9 else 0.0
    # v3 (Sol mission): the size curve is concave — its slope flattens toward the
    # top (global slope -0.93, slope above x=12 is -0.17). A local-linear fit at a
    # target above every donor extrapolates the window's steeper slope, so it
    # under-predicts the ratios. Shrink the slope hyperbolically with the
    # extrapolation distance over the donors' max x; validated on leave-top-out
    # (top3 0.1721 -> 0.1206, top5 0.1722 -> 0.1297). No effect on in-range targets.
    over = max(0.0, target["x"] - max(xs))
    if over > 0:
        slope = slope / (1.0 + 0.7 * over)
    pred = my - slope * (mx - target["x"])
    lo, hi = min(ys), max(ys)
    pred = min(hi, max(lo, pred))
    return pred, list(zip(ys, ws))


def loo_spread(target, donors, hour, use_y):
    """Weighted SD of leave-one-out residuals at `hour`, kernel-weighted at the target."""
    residuals = []
    for i, d in enumerate(donors):
        pool = donors[:i] + donors[i + 1:]
        as_target = {"x": d["x"], "y": d.get("y"), "delta": d["delta"]}
        pred, _ = fit_hour(as_target, pool, hour, use_y)
        if pred is None:
            continue
        hh = hour  # the donor is its own target here, so no shift
        residuals.append((math.log(d["day"][hh] / d["v1"]) - pred,
                          kernel_weight(d, target, use_y)))
    if not residuals:
        return None
    sw = sum(w for _, w in residuals)
    mu = sum(w * r for r, w in residuals) / sw
    var = sum(w * (r - mu) ** 2 for r, w in residuals) / sw
    return max(SPREAD_FLOOR, math.sqrt(var)) * SPREAD_MULT


def persistence_quantiles(v1, hour):
    """The house baseline (starter): flat median, widening spread."""
    return [round2(v1 * math.exp(z * BASE_SIGMA * math.sqrt(hour - 1))) for z in Z]


def persistence_case(case, v1):
    """The house baseline forecast for a case, used when the model cannot run."""
    return {"hours": [{"hour": row["hour"], "q": persistence_quantiles(v1, row["hour"])}
                      for row in case["forecast"]]}


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    v1 = max(1, first["viewers"])
    day_start = parse_time(case["day_start"])
    donors = build_donors(case)
    if not donors:
        return persistence_case(case, v1)
    pre = target_pre14max(case, day_start)
    target = {"x": math.log(v1), "y": math.log(pre / v1) if pre else None,
              "delta": us_east_midnight_utc(day_start)}
    donor_ys = [d["y"] for d in donors if d.get("y") is not None]
    use_y = bool(donor_ys and target["y"] is not None
                 and min(donor_ys) <= target["y"] <= max(donor_ys))
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        med, _ = fit_hour(target, donors, h, use_y)
        spread = loo_spread(target, donors, h, use_y)
        if med is None or spread is None:
            qs = persistence_quantiles(v1, h)
        else:
            qs = [v1 * math.exp(med + z * spread) for z in Z]
            qs = sorted(max(1.0, min(1e9, q)) for q in qs)
            qs = [round2(q) for q in qs]
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            # Never lose the whole run to one bad case: fall back to the
            # persistence baseline (or a flat 1-viewer forecast with no
            # hour-1 print), so the case stays valid.
            try:
                first = next((k for k in case.get("known", []) if k["hour"] == 1), None)
                v1 = max(1, first["viewers"]) if first else 1
                forecast = persistence_case(case, v1)
            except Exception:
                forecast = {"hours": [{"hour": h, "q": [1, 1, 1, 1, 1]}
                                      for h in range(2, 25)]}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
