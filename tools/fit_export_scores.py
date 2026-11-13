#!/usr/bin/env python3
"""Reconstruct Oura's Readiness and Activity scores from ring-derivable data,
calibrated on the full Oura account export ("App Data/*.csv").

Two layers, like fit_scores_all.py, but driven by raw signals instead of the trends
CSV, so every input is one the app computes from the ring itself:

  inputs:       per day, the main sleep that ended that morning (lowest HR, mean HRV,
                stage durations, 5-min HR curve, p90 skin temperature) and the day's
                per-minute MET (04:00 to 04:00 local, as Oura's activity day), plus
                body weight and a daily active-kcal target.
  contributors: each Oura contributor (0-100) = linear fit over spline / lookup
                features of those inputs, with causal personal baselines.
  combiner:     Activity = weighted sum of its 6 contributors (incl. recovery_time,
                which the trends CSV lacks). Readiness = weighted sum of its 8
                contributors + a penalty below a body-temperature contributor of 75
                (sleep_regularity is shown by Oura but carries ~0 weight).

Activity features never use Oura's own bucket columns (low/medium/high time,
sedentary time, inactivity alerts, calories, steps, targets): those come from Oura's
cloud, not the ring. They are rebuilt from per-minute MET with 5-minute classes that
reproduce Oura's own class_5_min on ~92% of blocks.

The feature definitions are mirrored exactly by the Rust crate `oura-scores`, which
loads the parameters this writes; `--fixture` also dumps per-day inputs and these
predictions so the Rust side can be checked number for number.

Oura's scores and contributors are only fit targets, never inputs. Reports accuracy
on 5-fold CV (shuffled days) and blocked CV (5 contiguous periods held out in turn).

Usage: python tools/fit_export_scores.py "<export>/App Data" [--out PATH] [--fixture PATH]
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

PARAMS_VERSION = 1
READINESS = ["activity_balance", "body_temperature", "hrv_balance", "previous_day_activity",
             "previous_night", "recovery_index", "resting_heart_rate", "sleep_balance"]
ACTIVITY = ["meet_daily_targets", "move_every_hour", "recovery_time", "stay_active",
            "training_frequency", "training_volume"]
TEMP_KNEE = 75.0  # readiness drops faster once the temperature contributor is below this
# 5-minute mean-MET class edges: rest | sedentary | low | medium | high (< 0.5 = non-wear)
CLASS_EDGES = (1.06, 1.65, 3.8, 7.75)
NIGHT_FIELDS = ["lowest_hr", "avg_hrv", "total_sleep_s", "efficiency", "rem_s", "deep_s",
                "latency_s", "awake_s", "time_in_bed_s", "skin_temp_p90"]


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
    return [None if v is None else float(v) for v in items]


class Export:
    """Ring-like per-day inputs plus Oura's scores (targets), from the export.

    Index i is DAYS[i] (contiguous calendar). `nights[i]` / `met[i]` are None on days
    without data; they hold exactly what the Rust side receives."""

    def __init__(self, app_data):
        self.act = {r["day"]: r for r in read(app_data, "dailyactivity")}
        self.rdy = {r["day"]: r for r in read(app_data, "dailyreadiness")}
        main = {}
        for r in read(app_data, "sleepmodel"):
            key = (r["type"] == "long_sleep", num(r["total_sleep_duration"]))
            if r["day"] not in main or key > main[r["day"]][0]:
                main[r["day"]] = (key, r)
        sleep = {d: r for d, (_, r) in main.items()}
        days = sorted(set(self.act) | set(self.rdy) | set(sleep))
        start, end = dt.date.fromisoformat(days[0]), dt.date.fromisoformat(days[-1])
        self.days = [(start + dt.timedelta(i)).isoformat() for i in range((end - start).days + 1)]

        p90 = self._night_skin_temp(app_data, sleep)
        self.nights = []
        for d in self.days:
            r = sleep.get(d)
            self.nights.append(None if r is None else {
                "lowest_hr": num(r["lowest_heart_rate"]), "avg_hrv": num(r["average_hrv"]),
                "total_sleep_s": num(r["total_sleep_duration"]), "efficiency": num(r["efficiency"]),
                "rem_s": num(r["rem_sleep_duration"]), "deep_s": num(r["deep_sleep_duration"]),
                "latency_s": num(r["latency"]), "awake_s": num(r["awake_time"]),
                "time_in_bed_s": num(r["time_in_bed"]), "skin_temp_p90": p90.get(d, np.nan),
                "hr_5min": series_items(r["heart_rate"]),
            })
        self.met = [series_items(self.act[d]["met"]) if d in self.act else None for d in self.days]
        self.weight_kg = next((num(r["weight"]) for r in read(app_data, "usersettings")), np.nan)

    def score(self, src):
        table = {"act": self.act, "rdy": self.rdy}[src]
        return np.array([num(table[d]["score"]) if d in table else np.nan for d in self.days])

    def contributor(self, src, key):
        table = {"act": self.act, "rdy": self.rdy}[src]
        out = []
        for d in self.days:
            v = json.loads(table[d]["contributors"] or "{}").get(key) if d in table else None
            out.append(np.nan if v is None else float(v))
        return np.array(out)

    def target_kcal(self):
        """Oura's daily active-kcal target, rescaled to our MET-derived kcal units."""
        ours = np.array([active_kcal(m, self.weight_kg) if m else np.nan for m in self.met])
        oura = np.array([num(self.act[d]["active_calories"]) if d in self.act else np.nan for d in self.days])
        goal = np.array([num(self.act[d]["target_calories"]) if d in self.act else np.nan for d in self.days])
        ok = np.isfinite(ours) & np.isfinite(oura) & np.isfinite(goal) & (oura > 0)
        return float(np.median(goal[ok] * ours[ok] / oura[ok]))

    @staticmethod
    def _night_skin_temp(app_data, sleep):
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
        out = {}
        for d, r in sleep.items():
            a = dt.datetime.fromisoformat(r["bedtime_start"]).timestamp()
            b = dt.datetime.fromisoformat(r["bedtime_end"]).timestamp()
            lo, hi = np.searchsorted(ts, [a, b])
            seg = tv[lo:hi]
            seg = seg[(seg > 30) & (seg < 40)]
            if len(seg) >= 30:
                out[d] = float(np.percentile(seg, 90))
        return out


# ---------------------------------------------------------------- per-day aggregates
# Mirrored by oura-scores (Rust): keep the two in step.

def met_array(met):
    return np.array([np.nan if v is None else v for v in met], float)


def met_classes(met):
    """Oura-like activity class per 5-minute block from its mean MET:
    0 non-wear, 1 rest, 2 sedentary, 3 low, 4 medium, 5 high."""
    m = met_array(met)
    m = np.concatenate([m, np.full((-len(m)) % 5, np.nan)]).reshape(-1, 5)
    with np.errstate(invalid="ignore"):
        fin = np.isfinite(m).sum(1)
        b = np.where(fin > 0, np.nansum(m, 1) / np.maximum(fin, 1), np.nan)
    e1, e2, e3, e4 = CLASS_EDGES
    return np.select([~np.isfinite(b) | (b < 0.5), b < e1, b < e2, b < e3, b < e4],
                     [0, 1, 2, 3, 4], default=5)


def active_kcal(met, weight_kg):
    return float(np.nansum(np.maximum(0.0, met_array(met) - 1.0))) * weight_kg / 60.0


def activity_day(met, weight_kg):
    """Daily activity aggregates from one day's per-minute MET."""
    m = met_array(met)
    cls = met_classes(met)
    per_min = np.repeat(cls, 5)[:len(m)]
    excess = np.where(np.isfinite(m), np.maximum(0.0, m - 1.0), 0.0)
    runs, cur = 0, 0  # sedentary runs reaching 60 min between 08:00 and 22:00 local
    for c in cls[48:216]:  # blocks from 04:00 day start: 08:00 = 48, 22:00 = 216
        cur = cur + 1 if c == 2 else 0
        runs += cur == 12
    # MET sums are rounded to 6 decimals so threshold tests don't depend on summation
    # order (numpy sums pairwise, Rust sequentially)
    with np.errstate(invalid="ignore"):
        return {
            "sedentary_s": float((cls == 2).sum() * 300),
            "low_mm": round(float(excess[per_min == 3].sum()), 6),
            "medium_mm": round(float(excess[per_min == 4].sum()), 6),
            "high_mm": round(float(excess[per_min == 5].sum()), 6),
            "sedentary_runs": float(runs),
            "active_kcal": round(float(excess.sum()), 6) * weight_kg / 60.0,
            "minutes_ge3": float((m >= 3).sum()),
            "met_min_ge5": round(float(np.where(m >= 5, m, 0.0).sum()), 6),
        }


def hr_settle(night, tol=3.0):
    """(minutes in bed after HR first settles within `tol` bpm of the night minimum,
    that time as a fraction of time in bed). Drives Oura's recovery index."""
    hr = met_array(night["hr_5min"])
    in_bed = night["time_in_bed_s"] / 60
    if not np.isfinite(hr).any() or not in_bed > 0:
        return np.nan, np.nan
    first = np.where(hr <= np.nanmin(hr) + tol)[0][0] * 5.0  # 5-min samples
    return in_bed - first, first / in_bed


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


def flag(x, cond):
    return np.where(np.isfinite(x), cond.astype(float), np.nan)


# ---------------------------------------------------------------- features

def build_inputs(nights, mets, weight_kg, target_kcal):
    """Named, causal base features from per-day ring inputs (None = no data)."""
    def night(key):
        return np.array([n[key] if n else np.nan for n in nights], float)

    acts = [activity_day(m, weight_kg) if m else None for m in mets]

    def act(key):
        return np.array([a[key] if a else np.nan for a in acts], float)

    lo, hrv, tst, p90 = night("lowest_hr"), night("avg_hrv"), night("total_sleep_s"), night("skin_temp_p90")
    settle = np.array([hr_settle(n) if n else (np.nan, np.nan) for n in nights], float).reshape(-1, 2)
    hm = act("medium_mm") + act("high_mm")
    allmm = act("low_mm") + hm
    kcal, min3, mm5 = act("active_kcal"), act("minutes_ge3"), act("met_min_ge5")
    with np.errstate(divide="ignore", invalid="ignore"):
        feats = {
            # sleep (previous night)
            "total_sleep": tst, "efficiency": night("efficiency"), "rem": night("rem_s"),
            "deep": night("deep_s"), "latency": night("latency_s"), "awake": night("awake_s"),
            # resting HR / HRV vs personal baseline
            "lowest_hr": lo, "lowest_hr_vs_30d": lo / prev(ewma(lo, 30)),
            "hrv_3d_minus_30d": ewma(hrv, 3) - prev(ewma(hrv, 30)),
            "hrv_7d_over_60d": ewma(hrv, 7) / prev(ewma(hrv, 60)),
            # skin temperature deviation
            "temp_deviation": p90 - prev(ewma(p90, 45)),
            # sleep balance
            "sleep_7d_minus_8h": ewma(tst, 7) - 8 * 3600,
            "sleep_14d_over_90d": ewma(tst, 14) / prev(ewma(tst, 90)),
            # recovery index
            "hr_settled_remaining_min": settle[:, 0], "hr_settle_frac": settle[:, 1],
            # previous-day activity
            "prev_met_min": prev(allmm), "prev_high_met_min": prev(act("high_mm")),
            "prev_medium_met_min": prev(act("medium_mm")), "prev_sedentary": prev(act("sedentary_s")),
            "prev_sedentary_runs": prev(act("sedentary_runs")),
            "prev_met_min_vs_30d": prev(allmm) / prev(ewma(allmm, 30)),
            # activity balance: this week vs ~6 months
            "kcal_7d_over_180d": trail(kcal, 7, lag=1) / trail(kcal, 180, lag=1),
            "kcal_180d": trail(kcal, 180, lag=1),
            "met_min_7d_over_180d": trail(allmm, 7, lag=1) / trail(allmm, 180, lag=1),
            "met_min_180d": trail(allmm, 180, lag=1),
            # activity contributors
            "days_target_met_7d": trail(flag(kcal, kcal >= target_kcal), 7, np.nansum),
            "target_ratio_7d": trail(np.minimum(1.0, kcal / target_kcal), 7),
            "sedentary_runs": act("sedentary_runs"),
            "sedentary": act("sedentary_s"),
            "hard_days_6d": trail(flag(min3, min3 >= 183), 6, np.nansum),
            "training_days_6d": trail(flag(mm5, mm5 >= 104.6), 6, np.nansum),
            "vigorous_met_min_6d": trail(mm5, 6),
            "hm_met_min_7d": trail(hm, 7, np.nansum),
            "training_load_7d_vs_16w": trail(hm, 7, np.nansum) / ewma(hm, 112),
        }
    # a ratio against an empty/zero baseline is missing, not infinite
    return {k: np.where(np.isfinite(v), v, np.nan) for k, v in feats.items()}


# contributor -> [(feature, kind)]; kind "spline" = piecewise-linear at quantile
# knots, "lookup" = one indicator per integer count, "linear" = as-is.
SPEC = {
    "previous_night": [(f, "spline") for f in ["total_sleep", "efficiency", "rem", "deep", "latency", "awake"]],
    "resting_heart_rate": [("lowest_hr_vs_30d", "spline"), ("lowest_hr", "spline")],
    "hrv_balance": [("hrv_3d_minus_30d", "spline"), ("hrv_7d_over_60d", "spline")],
    "body_temperature": [("temp_deviation", "spline")],
    "sleep_balance": [("sleep_7d_minus_8h", "spline"), ("sleep_14d_over_90d", "spline")],
    "previous_day_activity": [(f, "spline") for f in ["prev_met_min", "prev_high_met_min", "prev_medium_met_min",
                                                       "prev_sedentary", "prev_sedentary_runs",
                                                       "prev_met_min_vs_30d"]],
    "activity_balance": [(f, "spline") for f in ["kcal_7d_over_180d", "kcal_180d",
                                                  "met_min_7d_over_180d", "met_min_180d"]],
    "recovery_index": [("hr_settled_remaining_min", "spline"), ("hr_settle_frac", "spline")],
    # per-count table + how close the week came to target: the ratio carries the
    # fit through low-count weeks the table has barely seen
    "meet_daily_targets": [("days_target_met_7d", "lookup"), ("target_ratio_7d", "spline")],
    "move_every_hour": [("sedentary_runs", "lookup")],
    "stay_active": [("sedentary", "spline")],
    "recovery_time": [("hard_days_6d", "lookup")],
    "training_frequency": [("training_days_6d", "linear"), ("vigorous_met_min_6d", "spline")],
    "training_volume": [("hm_met_min_7d", "spline"), ("training_load_7d_vs_16w", "spline")],
}
KNOT_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def basis(x, kind, knots):
    if kind == "linear":
        return x[:, None]
    if kind == "lookup":  # a missing count matches no key: all-zero, not missing
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
    """Basis columns, then a 'some input missing' flag, then an intercept; missing
    values contribute 0."""
    F = np.hstack([basis(inputs[f], kind, knots[f]) for f, kind in spec])
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
    score = x.score(src)
    m = train[np.isfinite(score[train]) & np.isfinite(truth[train]).all(1)]
    model["combiner"] = lstsq(combine(truth[m]), score[m])
    return model


def predict(inputs, names, combine, model):
    C = np.column_stack([np.clip(design(inputs, SPEC[c], model["knots"]) @ model["contributors"][c], 1, 100)
                         for c in names])
    return C, combine(C) @ model["combiner"]


def report(label, x, src, names, combine, inputs, splits):
    score = x.score(src)
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


def clean(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("app_data", type=Path, help='the export\'s "App Data" directory')
    ap.add_argument("--out", type=Path, default=REPO / "local" / "export_score_params.json")
    ap.add_argument("--fixture", type=Path,
                    help="also write per-day inputs + predictions for the Rust cross-check")
    args = ap.parse_args()

    x = Export(args.app_data)
    target = x.target_kcal()
    inputs = build_inputs(x.nights, x.met, x.weight_kg, target)
    print(f"export: {x.days[0]} .. {x.days[-1]}  ({len(x.days)} days)  "
          f"weight {x.weight_kg:g} kg, target {target:.0f} active kcal\n")

    idx = np.arange(len(x.days))
    params = {"version": PARAMS_VERSION, "source": "oura account export",
              "days": [x.days[0], x.days[-1]], "weight_kg": x.weight_kg, "target_kcal": target,
              "temp_knee": TEMP_KNEE, "class_edges": list(CLASS_EDGES), "scores": {}}
    predictions = {}
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
        report("blocked CV (5 contiguous periods)", x, src, names, combine, inputs,
               [(np.setdiff1d(idx, b), b) for b in np.array_split(idx, 5)])

        model = fit_score(x, inputs, src, names, combine, idx)
        params["scores"][name.lower()] = {
            "order": names,  # combiner weight order; JSON object order is not relied on
            "contributors": {c: {"inputs": [[f, kind, model["knots"][f]] for f, kind in SPEC[c]],
                                 "weights": model["contributors"][c].tolist()} for c in names},
            "combiner": model["combiner"].tolist(),
        }
        C, p = predict(inputs, names, combine, model)
        predictions[name.lower()] = {"score": p, "contributors": dict(zip(names, C.T))}
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(params, indent=1))
    print(f"wrote {args.out}")

    if args.fixture:
        days = []
        for i, d in enumerate(x.days):
            n = x.nights[i]
            days.append({
                "day": d,
                "night": None if n is None else {**{k: clean(n[k]) for k in NIGHT_FIELDS},
                                                 "hr_5min": n["hr_5min"]},
                "met": x.met[i],
                "expected": {s: {"score": clean(float(v["score"][i])),
                                 "contributors": {c: clean(float(a[i])) for c, a in v["contributors"].items()}}
                             for s, v in predictions.items()},
            })
        args.fixture.write_text(json.dumps({"days": days}))
        print(f"wrote {args.fixture}")


if __name__ == "__main__":
    main()
