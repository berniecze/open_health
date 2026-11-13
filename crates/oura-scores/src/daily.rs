//! Per-day inputs and the aggregates derived from them.

use serde::{Deserialize, Serialize};

/// The main sleep that ended on a day's morning. `None` fields are unknown.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Night {
    pub lowest_hr: Option<f64>,
    pub avg_hrv: Option<f64>,
    pub total_sleep_s: Option<f64>,
    /// Sleep efficiency, percent.
    pub efficiency: Option<f64>,
    pub rem_s: Option<f64>,
    pub deep_s: Option<f64>,
    pub latency_s: Option<f64>,
    pub awake_s: Option<f64>,
    pub time_in_bed_s: Option<f64>,
    /// 90th percentile of skin temperature (°C) within the bedtime window.
    pub skin_temp_p90: Option<f64>,
    /// Heart rate in 5-minute samples from bedtime start; `None` = no sample.
    #[serde(default)]
    pub hr_5min: Vec<Option<f64>>,
}

/// One calendar day. Days passed to [`crate::score_days`] must be consecutive.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Day {
    /// `YYYY-MM-DD` (the wake-up day for the night, the activity day for `met`).
    pub day: String,
    pub night: Option<Night>,
    /// Per-minute MET from 04:00 local to 04:00 the next day (Oura's activity day).
    /// `None` minutes are non-wear; Oura's own exports mark non-wear as 0.1.
    pub met: Option<Vec<Option<f64>>>,
}

/// 5-minute mean-MET class edges: rest | sedentary | low | medium | high. A block
/// below 0.5 MET (or with no samples) is non-wear. They reproduce Oura's own
/// `class_5_min` on ~92% of blocks.
pub const DEFAULT_CLASS_EDGES: [f64; 4] = [1.06, 1.65, 3.8, 7.75];

/// Activity class of a 5-minute block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    NonWear,
    Rest,
    Sedentary,
    Low,
    Medium,
    High,
}

/// Classify each 5-minute block of `met` by its mean over present samples.
pub fn met_classes(met: &[Option<f64>], edges: &[f64; 4]) -> Vec<Class> {
    met.chunks(5)
        .map(|block| {
            let present: Vec<f64> = block
                .iter()
                .flatten()
                .copied()
                .filter(|v| v.is_finite())
                .collect();
            if present.is_empty() {
                return Class::NonWear;
            }
            let mean = present.iter().sum::<f64>() / present.len() as f64;
            if mean < 0.5 {
                Class::NonWear
            } else if mean < edges[0] {
                Class::Rest
            } else if mean < edges[1] {
                Class::Sedentary
            } else if mean < edges[2] {
                Class::Low
            } else if mean < edges[3] {
                Class::Medium
            } else {
                Class::High
            }
        })
        .collect()
}

/// Daily activity aggregates from one day's per-minute MET.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActivityDay {
    pub sedentary_s: f64,
    /// Σ max(0, MET − 1) over minutes in low / medium / high blocks.
    pub low_mm: f64,
    pub medium_mm: f64,
    pub high_mm: f64,
    /// Sedentary runs reaching 60 minutes between 08:00 and 22:00.
    pub sedentary_runs: f64,
    pub active_kcal: f64,
    pub minutes_ge3: f64,
    /// Σ MET over minutes at or above 5 MET.
    pub met_min_ge5: f64,
}

impl ActivityDay {
    pub fn from_met(met: &[Option<f64>], weight_kg: f64, edges: &[f64; 4]) -> Self {
        let classes = met_classes(met, edges);
        let (mut low, mut medium, mut high, mut total) = (0.0, 0.0, 0.0, 0.0);
        let (mut ge3, mut ge5) = (0.0, 0.0);
        for (i, v) in met.iter().enumerate() {
            let Some(m) = v.filter(|m| m.is_finite()) else {
                continue;
            };
            let excess = (m - 1.0).max(0.0);
            total += excess;
            match classes[i / 5] {
                Class::Low => low += excess,
                Class::Medium => medium += excess,
                Class::High => high += excess,
                _ => {}
            }
            if m >= 3.0 {
                ge3 += 1.0;
            }
            if m >= 5.0 {
                ge5 += m;
            }
        }
        // blocks counted from the 04:00 day start: 08:00 = 48, 22:00 = 216
        let (mut runs, mut cur) = (0.0, 0);
        for c in classes.iter().take(216).skip(48) {
            cur = if *c == Class::Sedentary { cur + 1 } else { 0 };
            if cur == 12 {
                runs += 1.0;
            }
        }
        let sedentary = classes.iter().filter(|c| **c == Class::Sedentary).count();
        // MET sums are rounded to 6 decimals so threshold tests don't depend on
        // summation order (the Python fit sums pairwise, this sequentially)
        let r6 = |x: f64| (x * 1e6).round() / 1e6;
        ActivityDay {
            sedentary_s: sedentary as f64 * 300.0,
            low_mm: r6(low),
            medium_mm: r6(medium),
            high_mm: r6(high),
            sedentary_runs: runs,
            active_kcal: r6(total) * weight_kg / 60.0,
            minutes_ge3: ge3,
            met_min_ge5: r6(ge5),
        }
    }
}

/// (minutes in bed after HR first settles within `tol` bpm of the night minimum,
/// that time as a fraction of time in bed). Drives Oura's recovery index.
pub fn hr_settle(night: &Night, tol: f64) -> (f64, f64) {
    let in_bed = night.time_in_bed_s.unwrap_or(f64::NAN) / 60.0;
    let min = night
        .hr_5min
        .iter()
        .flatten()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f64::INFINITY, f64::min);
    if !min.is_finite() || !(in_bed > 0.0) {
        return (f64::NAN, f64::NAN);
    }
    let first = night
        .hr_5min
        .iter()
        .position(|v| v.is_some_and(|h| h <= min + tol))
        .expect("the minimum itself qualifies") as f64
        * 5.0;
    (in_bed - first, first / in_bed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minutes(values: &[(f64, usize)]) -> Vec<Option<f64>> {
        values
            .iter()
            .flat_map(|&(v, n)| std::iter::repeat(Some(v)).take(n))
            .collect()
    }

    #[test]
    fn classes_follow_block_means() {
        let met = minutes(&[(0.1, 5), (0.9, 5), (1.3, 5), (2.0, 5), (5.0, 5), (9.0, 5)]);
        let got = met_classes(&met, &DEFAULT_CLASS_EDGES);
        use Class::*;
        assert_eq!(got, vec![NonWear, Rest, Sedentary, Low, Medium, High]);
        assert_eq!(met_classes(&[None; 5], &DEFAULT_CLASS_EDGES), vec![NonWear]);
    }

    #[test]
    fn activity_day_sums_excess_met_by_class() {
        // 04:00-08:00 rest, then 70 sedentary minutes, then 10 medium minutes
        let met = minutes(&[(0.9, 240), (1.2, 70), (4.0, 10)]);
        let a = ActivityDay::from_met(&met, 60.0, &DEFAULT_CLASS_EDGES);
        assert_eq!(a.sedentary_s, 70.0 * 60.0);
        assert_eq!(a.sedentary_runs, 1.0);
        assert!((a.medium_mm - 30.0).abs() < 1e-9);
        assert!((a.active_kcal - (70.0 * 0.2 + 30.0) * 60.0 / 60.0).abs() < 1e-9);
        assert_eq!(a.minutes_ge3, 10.0);
        assert_eq!(a.met_min_ge5, 0.0);
    }

    #[test]
    fn hr_settle_measures_time_after_settling() {
        let night = Night {
            time_in_bed_s: Some(8.0 * 3600.0),
            hr_5min: vec![
                None,
                Some(70.0),
                Some(60.0),
                Some(52.0),
                Some(50.0),
                Some(55.0),
            ],
            ..Default::default()
        };
        let (remaining, frac) = hr_settle(&night, 3.0);
        assert_eq!(remaining, 480.0 - 15.0);
        assert!((frac - 15.0 / 480.0).abs() < 1e-12);
        assert!(hr_settle(&Night::default(), 3.0).0.is_nan());
    }
}
