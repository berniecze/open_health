#!/usr/bin/env python3
"""Build a trends-style CSV from Oura's full account data export ("App Data/*.csv").

The score tools (fit_scores.py, fit_scores_all.py, calibrate_scores.py) read the web
"Trends" CSV. The full export carries the same information split across files:
dailysleep / dailyreadiness / dailyactivity (scores + contributor JSON) and
sleepmodel (per-period raw sleep metrics). This joins them per day under the trends
header names, durations in seconds.

Usage: python tools/oura_export_to_trends.py "<export>/App Data" [--out local/trends.csv]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

csv.field_size_limit(sys.maxsize)  # sleepmodel rows embed long per-epoch JSON

# trends header -> contributor key in the export's `contributors` JSON
SLEEP_CONTRIB = {
    "Total Sleep Score": "total_sleep", "Sleep Efficiency Score": "efficiency",
    "Restfulness Score": "restfulness", "REM Sleep Score": "rem_sleep",
    "Deep Sleep Score": "deep_sleep", "Sleep Latency Score": "latency",
    "Sleep Timin Score": "timing",
}
READINESS_CONTRIB = {
    "Previous Night Score": "previous_night", "Sleep Balance Score": "sleep_balance",
    "Previous Day Activity Score": "previous_day_activity",
    "Activity Balance Score": "activity_balance", "Temperature Score": "body_temperature",
    "Resting Heart Rate Score": "resting_heart_rate", "HRV Balance Score": "hrv_balance",
    "Recovery Index Score": "recovery_index",
}
ACTIVITY_CONTRIB = {
    "Stay Active Score": "stay_active", "Move Every Hour Score": "move_every_hour",
    "Meet Daily Targets Score": "meet_daily_targets",
    "Training Frequency Score": "training_frequency",
    "Training Volume Score": "training_volume",
}
# trends header -> sleepmodel column (main sleep period of the day)
SLEEP_RAW = {
    "Total Sleep Duration": "total_sleep_duration", "Deep Sleep Duration": "deep_sleep_duration",
    "REM Sleep Duration": "rem_sleep_duration", "Light Sleep Duration": "light_sleep_duration",
    "Sleep Efficiency": "efficiency", "Sleep Latency": "latency", "Awake Time": "awake_time",
    "Total Bedtime ": "time_in_bed",  # trailing space matches Oura's header
    # count of restless periods, not a duration: only feeds the analysis-only
    # restless_frac, where a linear fit absorbs the scale
    "Restless Sleep": "restless_periods",
    "Average Resting Heart Rate": "average_heart_rate",
    "Lowest Resting Heart Rate": "lowest_heart_rate", "Average HRV": "average_hrv",
    "Respiratory Rate": "average_breath",
    "Bedtime Start": "bedtime_start", "Bedtime End": "bedtime_end",
}
# trends header -> dailyactivity column
ACTIVITY_RAW = {
    "Average MET": "average_met_minutes", "Steps": "steps", "Activity Burn": "active_calories",
    "Total Burn": "total_calories", "Inactive Time": "sedentary_time",
    "Long Periods of Inactivity": "inactivity_alerts", "High Activity Time": "high_activity_time",
    "Medium Activity Time": "medium_activity_time", "Low Activity Time": "low_activity_time",
}


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def clean(v):
    return "" if v in (None, "None", "null") else str(v)


def iso(ts):
    # "2022-11-22T06:42:15.000+01:00" -> drop millis so 3.9's fromisoformat accepts it
    return ts.replace(".000", "") if ts else ""


def scores(rows, contrib_map, score_col):
    out = {}
    for r in rows:
        c = json.loads(r["contributors"] or "{}")
        day = {score_col: clean(r["score"])}
        day.update({h: clean(c.get(k)) for h, k in contrib_map.items()})
        out[r["day"]] = (r, day)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("app_data", type=Path, help='the export\'s "App Data" directory')
    ap.add_argument("--out", type=Path, default=REPO / "local" / "trends.csv")
    args = ap.parse_args()
    d = args.app_data

    sleep = scores(read(d / "dailysleep.csv"), SLEEP_CONTRIB, "Sleep Score")
    ready = scores(read(d / "dailyreadiness.csv"), READINESS_CONTRIB, "Readiness Score")
    act = scores(read(d / "dailyactivity.csv"), ACTIVITY_CONTRIB, "Activity Score")

    # main sleep per day: prefer long_sleep, then the longest period
    main_sleep = {}
    for r in read(d / "sleepmodel.csv"):
        key = (r["type"] == "long_sleep", float(r["total_sleep_duration"] or 0))
        cur = main_sleep.get(r["day"])
        if cur is None or key > cur[0]:
            main_sleep[r["day"]] = (key, r)

    header = (["date", "Sleep Score", *SLEEP_CONTRIB, "Readiness Score", *READINESS_CONTRIB,
               "Temperature Deviation (°C)", "Activity Score", *ACTIVITY_CONTRIB,
               *SLEEP_RAW, *ACTIVITY_RAW])
    days = sorted(set(sleep) | set(ready) | set(act))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for day in days:
            row = {"date": day}
            for src in (sleep, ready, act):
                if day in src:
                    row.update(src[day][1])
            if day in ready:
                row["Temperature Deviation (°C)"] = clean(ready[day][0]["temperature_deviation"])
            if day in main_sleep:
                s = main_sleep[day][1]
                row.update({h: clean(s[k]) for h, k in SLEEP_RAW.items()})
                row["Bedtime Start"], row["Bedtime End"] = iso(s["bedtime_start"]), iso(s["bedtime_end"])
            if day in act:
                a = act[day][0]
                row.update({h: clean(a[k]) for h, k in ACTIVITY_RAW.items()})
            w.writerow(row)
    print(f"wrote {args.out}  ({len(days)} days, {days[0]} .. {days[-1]})")


if __name__ == "__main__":
    main()
