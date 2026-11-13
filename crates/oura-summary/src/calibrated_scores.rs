//! Oura-calibrated Readiness and Activity (`oura-scores`), fed from ring data.
//!
//! The model is personal: `tools/fit_export_scores.py` fits it against the wearer's
//! own Oura account export and writes `export_score_params.json`, which lives next to
//! the DB like `profile.json`. Without that file the summary simply has no
//! `calibrated_scores`.
//!
//! [`Collector`] gathers the model's per-day inputs while `build_summary` walks the
//! events: the main sleep that ended each morning and each 04:00–04:00 day's
//! per-minute MET (Oura's activity day).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use oura_scores::{Day, Night, Params};
use serde_json::{json, Value};

pub const PARAMS_FILE: &str = "export_score_params.json";

pub fn params_path(db: &Path) -> PathBuf {
    db.parent().unwrap_or(Path::new(".")).join(PARAMS_FILE)
}

/// Oura's activity day starts at 04:00 local.
const DAY_START_S: i64 = 4 * 3600;

#[derive(Default)]
pub(crate) struct Collector {
    /// wake day (days since epoch, local) → (time in bed s, night). Longest wins.
    nights: BTreeMap<i64, (f64, Night)>,
    /// activity day → per-minute MET from 04:00.
    met: BTreeMap<i64, Vec<Option<f64>>>,
}

/// What `build_summary` knows about one sleep, in its own units.
pub(crate) struct SleepRecord<'a> {
    pub wake_day: i64,
    pub start_ds: i64,
    pub end_ds: i64,
    /// Ring stage codes over the bedtime window: 1 deep, 2 light, 3 REM, 4 awake.
    pub stages: &'a [i64],
    pub onset_latency_min: Option<f64>,
    pub mean_hrv: Option<f64>,
    /// (ring deciseconds, bpm)
    pub hr_t: &'a [(i64, f64)],
    pub temps_c: &'a [f64],
}

impl Collector {
    pub fn add_sleep(&mut self, s: SleepRecord) {
        let in_bed_s = (s.end_ds - s.start_ds) as f64 / 10.0;
        if in_bed_s <= 0.0 {
            return;
        }
        if self
            .nights
            .get(&s.wake_day)
            .is_some_and(|(prev, _)| *prev >= in_bed_s)
        {
            return;
        }
        // stage durations from complete staging only, like the sleep metrics
        let staged = !s.stages.is_empty() && s.stages.iter().all(|c| (1..=4).contains(c));
        let epoch_s = in_bed_s / s.stages.len().max(1) as f64;
        let secs = |code: i64| {
            staged.then(|| s.stages.iter().filter(|&&c| c == code).count() as f64 * epoch_s)
        };
        let asleep = staged
            .then(|| s.stages.iter().filter(|&&c| (1..=3).contains(&c)).count() as f64 * epoch_s);
        let hr_5min = five_minute_means(s.hr_t, s.start_ds, s.end_ds);
        let night = Night {
            lowest_hr: hr_5min.iter().flatten().copied().reduce(f64::min),
            avg_hrv: s.mean_hrv,
            total_sleep_s: asleep,
            efficiency: asleep.map(|a| (a / in_bed_s * 100.0).round()),
            rem_s: secs(3),
            deep_s: secs(1),
            latency_s: s.onset_latency_min.map(|m| m * 60.0),
            awake_s: secs(4),
            time_in_bed_s: Some(in_bed_s),
            skin_temp_p90: p90_skin_temp(s.temps_c),
            hr_5min,
        };
        self.nights.insert(s.wake_day, (in_bed_s, night));
    }

    /// One minute of MET at local wall-clock `local_s` (unix seconds + UTC offset).
    pub fn add_met(&mut self, local_s: f64, met: f64) {
        let shifted = local_s as i64 - DAY_START_S;
        let day = shifted.div_euclid(86_400);
        let minute = (shifted.rem_euclid(86_400) / 60) as usize;
        self.met.entry(day).or_insert_with(|| vec![None; 1440])[minute] = Some(met);
    }

    fn days(&self) -> Vec<(i64, Day)> {
        let first = self.nights.keys().chain(self.met.keys()).min().copied();
        let last = self.nights.keys().chain(self.met.keys()).max().copied();
        let (Some(first), Some(last)) = (first, last) else {
            return Vec::new();
        };
        (first..=last)
            .map(|d| {
                let (y, m, dd) = crate::civil(d);
                (
                    d,
                    Day {
                        day: format!("{y:04}-{m:02}-{dd:02}"),
                        night: self.nights.get(&d).map(|(_, n)| n.clone()),
                        met: self.met.get(&d).cloned(),
                    },
                )
            })
            .collect()
    }

    /// The `calibrated_scores` summary block, or `None` without a params file.
    pub fn to_json(&self, db: &Path) -> Option<Value> {
        let path = params_path(db);
        if !path.exists() {
            return None;
        }
        let days = self.days();
        let scored = Params::load(&path).and_then(|p| {
            oura_scores::score_days(&days.iter().map(|(_, d)| d.clone()).collect::<Vec<_>>(), &p)
        });
        Some(match scored {
            Err(e) => json!({ "error": format!("{e:#}") }),
            Ok(scored) => {
                let render = |s: &oura_scores::Score| {
                    json!({
                        "score": s.score.round().clamp(1.0, 100.0),
                        "contributors": s.contributors.iter()
                            .map(|(c, v)| (c.clone(), json!(v.round())))
                            .collect::<serde_json::Map<_, _>>(),
                    })
                };
                json!({
                    "source": PARAMS_FILE,
                    "days": scored.iter()
                        .filter(|d| d.readiness.is_some() || d.activity.is_some())
                        .map(|d| json!({
                            "date": d.day,
                            "readiness": d.readiness.as_ref().map(render),
                            "activity": d.activity.as_ref().map(render),
                        }))
                        .collect::<Vec<_>>(),
                })
            }
        })
    }
}

/// Mean HR per 5-minute slot from bedtime start; `None` where a slot has no sample.
fn five_minute_means(hr_t: &[(i64, f64)], start_ds: i64, end_ds: i64) -> Vec<Option<f64>> {
    const SLOT_DS: i64 = 3000;
    let n = ((end_ds - start_ds) + SLOT_DS - 1)
        .div_euclid(SLOT_DS)
        .max(0) as usize;
    let mut sum = vec![(0.0, 0u32); n];
    for &(ds, bpm) in hr_t {
        if ds < start_ds || ds >= end_ds || !bpm.is_finite() || bpm <= 0.0 {
            continue;
        }
        let slot = &mut sum[((ds - start_ds) / SLOT_DS) as usize];
        slot.0 += bpm;
        slot.1 += 1;
    }
    sum.into_iter()
        .map(|(s, c)| (c > 0).then(|| s / c as f64))
        .collect()
}

/// 90th percentile (linear interpolation, as numpy) of plausible skin temperatures.
fn p90_skin_temp(temps_c: &[f64]) -> Option<f64> {
    let mut t: Vec<f64> = temps_c
        .iter()
        .copied()
        .filter(|c| *c > 30.0 && *c < 40.0)
        .collect();
    if t.len() < 10 {
        return None;
    }
    t.sort_by(f64::total_cmp);
    let pos = 0.9 * (t.len() - 1) as f64;
    let (lo, frac) = (pos.floor() as usize, pos.fract());
    Some(t[lo] + (t[(lo + 1).min(t.len() - 1)] - t[lo]) * frac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn met_minutes_land_on_the_four_am_day() {
        let mut c = Collector::default();
        let day0 = 20_000 * 86_400; // a local midnight
        c.add_met((day0 + 3 * 3600) as f64, 1.0); // 03:00 → previous activity day
        c.add_met((day0 + 4 * 3600 + 60) as f64, 2.0); // 04:01 → minute 1 of day 20_000
        assert_eq!(c.met[&19_999][23 * 60], Some(1.0));
        assert_eq!(c.met[&20_000][1], Some(2.0));
    }

    #[test]
    fn longest_sleep_per_wake_day_wins() {
        let mut c = Collector::default();
        let rec = |end_ds| SleepRecord {
            wake_day: 1,
            start_ds: 0,
            end_ds,
            stages: &[2, 2, 1, 3],
            onset_latency_min: Some(0.0),
            mean_hrv: Some(40.0),
            hr_t: &[(0, 60.0), (3000, 55.0)],
            temps_c: &[],
        };
        c.add_sleep(rec(36_000));
        c.add_sleep(rec(12_000)); // a nap the same day must not replace the night
        let (in_bed, n) = &c.nights[&1];
        assert_eq!(*in_bed, 3600.0);
        assert_eq!(n.total_sleep_s, Some(3600.0));
        assert_eq!(n.deep_s, Some(900.0));
        assert_eq!(n.lowest_hr, Some(55.0));
        assert_eq!(n.hr_5min.len(), 12);
    }

    #[test]
    fn p90_matches_numpy_linear_percentile() {
        let t: Vec<f64> = (0..11).map(|i| 33.0 + i as f64 * 0.1).collect();
        assert!((p90_skin_temp(&t).unwrap() - 33.9).abs() < 1e-9);
        let odd = [
            33.0, 33.5, 34.2, 35.0, 36.1, 33.3, 34.4, 35.5, 36.0, 33.9, 34.8, 35.2,
        ];
        assert!((p90_skin_temp(&odd).unwrap() - 35.95).abs() < 1e-9);
        assert_eq!(p90_skin_temp(&[34.0; 3]), None);
    }
}
