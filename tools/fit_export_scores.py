#!/usr/bin/env python3
"""Reconstruct Oura's Readiness and Activity scores from ring-derivable data,
calibrated on the full Oura account export ("App Data/*.csv").

Two layers, like fit_scores_all.py, but driven by the export's raw signals instead
of the trends CSV, so every input is one the app can compute from the ring:

  contributors: each Oura contributor (0-100) = linear fit over spline / lookup
                features of nightly sleep, HR, HRV, skin temperature and daily
                activity, with causal personal baselines (EWMA / trailing windows).
  combiner:     Activity = weighted sum of its 6 contributors (incl. recovery_time,
                which the trends CSV lacks). Readiness = weighted sum of its 8
                contributors + a penalty below a body-temperature contributor of 75
                (sleep_regularity is shown by Oura but carries ~0 weight).

Nothing here reads Oura's own scores or contributors as inputs; they are only the
fit targets. Readiness' "previous night" is fitted from raw sleep metrics, not from
Oura's sleep score.

Reports accuracy on 5-fold CV (shuffled days) and blocked CV (5 contiguous periods
held out in turn, so a test day has no near neighbours in training), and writes the fitted parameters to
local/export_score_params.json (personal calibration, gitignored).

Usage: python tools/fit_export_scores.py "<export>/App Data" [--out PATH]
"""
import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
csv.field_size_limit(sys.maxsize)  # rows embed per-minute JSON series

READINESS = ["activity_balance", "body_temperature", "hrv_balance", "previous_day_activity",
             "previous_night", "recovery_index", "resting_heart_rate", "sleep_balance"]
ACTIVITY = ["meet_daily_targets", "move_every_hour", "recovery_time", "stay_active",
            "training_frequency", "training_volume"]
TEMP_KNEE = 75.0  # readiness drops faster once the temperature contributor is below this


# ---------------------------------------------------------------- loading

def read(app_data, name):
    with open(app_data / f"{name}.csv", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def series_items(raw):
    items = (json.loads(raw or "{}") or {}).get("items") or []
    return np.array([np.nan if v is None else v for v in items], float)


class Export:
    """Day-indexed arrays from the export. Index i is DAYS[i] (contiguous calendar)."""

    def __init__(self, app_data):
        self.act = {r["day"]: r for r in read(app_data, "dailyactivity")}
        self.rdy = {r["day"]: r for r in read(app_data, "dailyreadiness")}
        main = {}
        for r in read(app_data, "sleepmodel"):
            key = (r["type"] == "long_sleep", num(r["total_sleep_duration"]))
            if r["day"] not in main or key > main[r["day"]][0]:
                main[r["day"]] = (key, r)
        self.sleep = {d: r for d, (_, r) in main.items()}
        days = sorted(set(self.act) | set(self.rdy) | set(self.sleep))
        start, end = dt.date.fromisoformat(days[0]), dt.date.fromisoformat(days[-1])
        self.days = [(start + dt.timedelta(i)).isoformat() for i in range((end - start).days + 1)]
        self.night_temp = self._night_skin_temp(app_data)

    def col(self, src, key):
        table = {"act": self.act, "rdy": self.rdy, "sleep": self.sleep}[src]
        return np.array([num(table[d][key]) if d in table else np.nan for d in self.days])

    def per_row(self, src, fn):
        table = {"act": self.act, "sleep": self.sleep}[src]
        return np.array([fn(table[d]) if d in table else np.nan for d in self.days], float)

    def contributor(self, src, key):
        table = {"act": self.act, "rdy": self.rdy}[src]
        out = []
        for d in self.days:
            v = json.loads(table[d]["contributors"] or "{}").get(key) if d in table else None
            out.append(np.nan if v is None else float(v))
        return np.array(out)

    def _night_skin_temp(self, app_data):
        """90th-percentile skin temperature within each main sleep's bedtime window."""
        ts, tv = [], []
        with open(app_data / "temperature.csv") as f:
            next(f)
            for line in f:
                t, v = line.rstrip().split(";")
                try:
                    tv.append(float(v))
                    ts.append(dt.datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp())
                except ValueError:
                    pass
        ts, tv = np.array(ts), np.array(tv)
        order = np.argsort(ts)
        ts, tv = ts[order], tv[order]
        out = np.full(len(self.days), np.nan)
        for i, d in enumerate(self.days):
            r = self.sleep.get(d)
            if not r:
                continue
            a = dt.datetime.fromisoformat(r["bedtime_start"]).timestamp()
            b = dt.datetime.fromisoformat(r["bedtime_end"]).timestamp()
            lo, hi = np.searchsorted(ts, [a, b])
            seg = tv[lo:hi]
            seg = seg[(seg > 30) & (seg < 40)]
            if len(seg) >= 30:
                out[i] = np.percentile(seg, 90)
        return out


# ---------------------------------------------------------------- causal helpers

def prev(a):
    return np.concatenate([[np.nan], a[:-1]])


def ewma(a, half_life):
    alpha = 1 - 0.5 ** (1 / half_life)
    out, s = np.full(len(a), np.nan), np.nan
    for i, x in enumerate(a):
        if np.isfinite(x):
            s = x if np.isnan(s) else s + alpha * (x - s)
        out[i] = s
    return out


def trail(a, n, fn=np.nanmean, lag=0):
    """fn over the n days ending `lag` days before each day (lag=0 includes today)."""
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        w = a[max(0, i + 1 - lag - n):i + 1 - lag]
        if len(w) and np.isfinite(w).any():
            out[i] = fn(w)
    return out


def count_days(flag, n):
    return trail(flag, n, np.nansum)


def as_flag(x, cond):
    return np.where(np.isfinite(x), cond.astype(float), np.nan)


def bout_minutes(met_row, threshold):
    """Minutes at or above `threshold` MET in a day's per-minute MET series."""
    return float(np.nansum(series_items(met_row["met"]) >= threshold))


def metmin_above(met_row, threshold):
    m = series_items(met_row["met"])
    return float(np.nansum(np.where(m >= threshold, m, 0)))


def hr_settle(sleep_row, tol=3.0):
    """(minutes in bed after HR first settles within `tol` bpm of the night minimum,
    that time as a fraction of time in bed). Drives Oura's recovery index."""
    hr = series_items(sleep_row["heart_rate"])
    if not np.isfinite(hr).any():
        return np.nan, np.nan
    first = np.where(hr <= np.nanmin(hr) + tol)[0][0] * 5.0  # 5-min samples
    in_bed = num(sleep_row["time_in_bed"]) / 60
    return in_bed - first, first / in_bed


# ---------------------------------------------------------------- features

def build_inputs(x):
    """Named ring-derivable base features, all causal."""
    a = lambda k: x.col("act", k)
    s = lambda k: x.col("sleep", k)
    lo, hrv, tst = s("lowest_heart_rate"), s("average_hrv"), s("total_sleep_duration")
    allmm = a("high_activity_met_minutes") + a("medium_activity_met_minutes") + a("low_activity_met_minutes")
    hm = a("high_activity_met_minutes") + a("medium_activity_met_minutes")
    steps, cal, tgt = a("steps"), a("active_calories"), a("target_calories")
    target_ratio = np.minimum(1.0, cal / tgt)
    settle = np.array([hr_settle(x.sleep[d]) if d in x.sleep else (np.nan, np.nan) for d in x.days])
    p90 = x.night_temp
    min3 = x.per_row("act", lambda r: bout_minutes(r, 3))
    mm5 = x.per_row("act", lambda r: metmin_above(r, 5))
    return {
        # sleep (previous night)
        "total_sleep": tst, "efficiency": s("efficiency"), "rem": s("rem_sleep_duration"),
        "deep": s("deep_sleep_duration"), "latency": s("latency"), "awake": s("awake_time"),
        # resting HR / HRV vs personal baseline
        "lowest_hr": lo, "lowest_hr_vs_30d": lo / prev(ewma(lo, 30)),
        "hrv_3d_minus_30d": ewma(hrv, 3) - prev(ewma(hrv, 30)),
        "hrv_7d_over_60d": ewma(hrv, 7) / prev(ewma(hrv, 60)),
        # skin temperature deviation (ours, not Oura's)
        "temp_deviation": p90 - prev(ewma(p90, 45)),
        # sleep balance
        "sleep_7d_minus_8h": ewma(tst, 7) - 8 * 3600,
        "sleep_14d_over_90d": ewma(tst, 14) / prev(ewma(tst, 90)),
        # recovery index
        "hr_settled_remaining_min": settle[:, 0], "hr_settle_frac": settle[:, 1],
        # previous-day activity
        "prev_met_min": prev(allmm), "prev_high_met_min": prev(a("high_activity_met_minutes")),
        "prev_medium_met_min": prev(a("medium_activity_met_minutes")),
        "prev_sedentary": prev(a("sedentary_time")), "prev_alerts": prev(a("inactivity_alerts")),
        "prev_met_min_vs_30d": prev(allmm) / prev(ewma(allmm, 30)),
        # activity balance: this week vs ~6 months
        "steps_7d_over_180d": trail(steps, 7, lag=1) / trail(steps, 180, lag=1),
        "steps_180d": trail(steps, 180, lag=1),
        "met_min_7d_over_180d": trail(allmm, 7, lag=1) / trail(allmm, 180, lag=1),
        "met_min_180d": trail(allmm, 180, lag=1),
        # activity contributors
        "days_target_met_7d": count_days(as_flag(cal, cal >= tgt), 7),
        "target_ratio_7d": trail(target_ratio, 7),
        "inactivity_alerts": a("inactivity_alerts"),
        "sedentary": a("sedentary_time"),
        "hard_days_6d": count_days(as_flag(min3, min3 >= 183), 6),
        "training_days_6d": count_days(as_flag(mm5, mm5 >= 104.6), 6),
        "vigorous_met_min_6d": trail(mm5, 6),
        "hm_met_min_7d": trail(hm, 7, np.nansum),
        "training_load_7d_vs_16w": trail(hm, 7, np.nansum) / ewma(hm, 112),
    }


# contributor -> [(feature, kind)]; kind "spline" = piecewise-linear at quantile
# knots, "lookup" = one value per integer count, "linear" = as-is.
SPEC = {
    "previous_night": [(f, "spline") for f in ["total_sleep", "efficiency", "rem", "deep", "latency", "awake"]],
    "resting_heart_rate": [("lowest_hr_vs_30d", "spline"), ("lowest_hr", "spline")],
    "hrv_balance": [("hrv_3d_minus_30d", "spline"), ("hrv_7d_over_60d", "spline")],
    "body_temperature": [("temp_deviation", "spline")],
    "sleep_balance": [("sleep_7d_minus_8h", "spline"), ("sleep_14d_over_90d", "spline")],
    "previous_day_activity": [(f, "spline") for f in ["prev_met_min", "prev_high_met_min", "prev_medium_met_min",
                                                       "prev_sedentary", "prev_alerts", "prev_met_min_vs_30d"]],
    "activity_balance": [(f, "spline") for f in ["steps_7d_over_180d", "steps_180d",
                                                  "met_min_7d_over_180d", "met_min_180d"]],
    "recovery_index": [("hr_settled_remaining_min", "spline"), ("hr_settle_frac", "spline")],
    # per-count table + how close the week came to target: the ratio carries the
    # fit through low-count weeks the table has barely seen
    "meet_daily_targets": [("days_target_met_7d", "lookup"), ("target_ratio_7d", "spline")],
    "move_every_hour": [("inactivity_alerts", "linear")],
    "stay_active": [("sedentary", "spline")],
    "recovery_time": [("hard_days_6d", "lookup")],
    "training_frequency": [("training_days_6d", "linear"), ("vigorous_met_min_6d", "spline")],
    "training_volume": [("hm_met_min_7d", "spline"), ("training_load_7d_vs_16w", "spline")],
}
KNOT_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def basis(x, kind, knots):
    if kind == "linear":
        return x[:, None]
    if kind == "lookup":
        return np.column_stack([(x == k).astype(float) for k in knots])
    return np.column_stack([x] + [np.maximum(0, x - k) for k in knots])


def knots_for(x, kind):
    fin = x[np.isfinite(x)]
    if kind == "lookup":
        return sorted({int(v) for v in fin})
    if kind == "spline":
        return [float(k) for k in np.unique(np.quantile(fin, KNOT_QUANTILES))]
    return []


def design(inputs, spec, knots):
    cols = [basis(inputs[f], kind, knots[f]) for f, kind in spec]
    F = np.hstack(cols)
    missing = np.isnan(F).any(1, keepdims=True).astype(float)
    return np.hstack([np.nan_to_num(F, nan=0.0), missing, np.ones((len(F), 1))])


def lstsq(F, y):
    w, *_ = np.linalg.lstsq(F, y, rcond=None)
    return w


def readiness_combiner(C):
    return np.hstack([C, np.maximum(0, TEMP_KNEE - C[:, [READINESS.index("body_temperature")]])])


def activity_combiner(C):
    return C


SCORES = {
    "Readiness": ("rdy", READINESS, readiness_combiner),
    "Activity": ("act", ACTIVITY, activity_combiner),
}


# ---------------------------------------------------------------- fit / evaluate

def fit_score(x, inputs, src, names, combine, train):
    """Fit every contributor and the combiner on day indices `train`."""
    knots = {f: knots_for(inputs[f], kind) for spec in SPEC.values() for f, kind in spec}
    model = {"contributors": {}, "knots": knots}
    for c in names:
        y = x.contributor(src, c)
        F = design(inputs, SPEC[c], knots)
        m = train[np.isfinite(y[train])]
        model["contributors"][c] = lstsq(F[m], y[m])
    truth = np.column_stack([x.contributor(src, c) for c in names])
    score = x.col(src, "score")
    m = train[np.isfinite(score[train]) & np.isfinite(truth[train]).all(1)]
    model["combiner"] = lstsq(combine(truth[m]), score[m])
    return model


def predict(inputs, names, combine, model):
    C = np.column_stack([np.clip(design(inputs, SPEC[c], model["knots"]) @ model["contributors"][c], 1, 100)
                         for c in names])
    return C, combine(C) @ model["combiner"]


def report(label, x, src, names, combine, inputs, splits):
    score = x.col(src, "score")
    truth = np.column_stack([x.contributor(src, c) for c in names])
    complete = np.isfinite(score) & np.isfinite(truth).all(1)
    pred, cpred = np.full(len(score), np.nan), np.full(truth.shape, np.nan)
    for train, test in splits:
        model = fit_score(x, inputs, src, names, combine, train)
        C, p = predict(inputs, names, combine, model)
        pred[test], cpred[test] = p[test], C[test]
    m = complete & np.isfinite(pred)
    e = score[m] - np.round(pred[m])
    r2 = 1 - np.var(score[m] - pred[m]) / np.var(score[m])
    print(f"  {label:34s} n={m.sum():4d}  R²={r2:.3f}  RMSE={np.sqrt((e ** 2).mean()):.2f}  "
          f"±1={np.mean(abs(e) <= 1):.0%}  ±3={np.mean(abs(e) <= 3):.0%}  ±5={np.mean(abs(e) <= 5):.0%}  "
          f"max={abs(e).max():.0f}")
    return cpred, truth, m


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("app_data", type=Path, help='the export\'s "App Data" directory')
    ap.add_argument("--out", type=Path, default=REPO / "local" / "export_score_params.json")
    args = ap.parse_args()

    x = Export(args.app_data)
    inputs = build_inputs(x)
    print(f"export: {x.days[0]} .. {x.days[-1]}  ({len(x.days)} days)\n")

    idx = np.arange(len(x.days))
    params = {"source": "oura account export", "days": [x.days[0], x.days[-1]],
              "temp_knee": TEMP_KNEE, "spec": SPEC, "scores": {}}
    for name, (src, names, combine) in SCORES.items():
        print(f"=== {name} Score (contributors computed from raw data only)")
        rng = np.random.default_rng(0)
        folds = np.array_split(rng.permutation(idx), 5)
        cpred, truth, m = report("5-fold CV", x, src, names, combine, inputs,
                                 [(np.setdiff1d(idx, f), f) for f in folds])
        for j, c in enumerate(names):
            ok = m & np.isfinite(truth[:, j])
            err = truth[ok, j] - cpred[ok, j]
            print(f"      {c:24s} R²={1 - err.var() / truth[ok, j].var():+.3f}  RMSE={np.sqrt((err ** 2).mean()):5.1f}")
        blocks = np.array_split(idx, 5)
        report("blocked CV (5 contiguous periods)", x, src, names, combine, inputs,
               [(np.setdiff1d(idx, b), b) for b in blocks])

        model = fit_score(x, inputs, src, names, combine, idx)
        params["scores"][name] = {
            "contributors": {c: model["contributors"][c].tolist() for c in names},
            "combiner": model["combiner"].tolist(),
            "knots": {f: model["knots"][f] for c in names for f, _ in SPEC[c]},
        }
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(params, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
