//! Named, causal per-day features. Mirrors `build_inputs` in
//! `tools/fit_export_scores.py`: NaN means missing and propagates exactly as numpy
//! does there (window means skip NaN; arithmetic on NaN stays NaN; a ratio against a
//! zero baseline is missing, not infinite).

use std::collections::HashMap;

use crate::daily::{hr_settle, ActivityDay, Day};

/// Series indexed like the input days; NaN = missing.
pub type Series = Vec<f64>;

pub fn prev(a: &[f64]) -> Series {
    if a.is_empty() {
        return Vec::new();
    }
    std::iter::once(f64::NAN)
        .chain(a[..a.len() - 1].iter().copied())
        .collect()
}

/// Exponentially weighted mean with the given half-life in days. Missing days hold the
/// last value; the first present value seeds it.
pub fn ewma(a: &[f64], half_life: f64) -> Series {
    let alpha = 1.0 - 0.5f64.powf(1.0 / half_life);
    let mut s = f64::NAN;
    a.iter()
        .map(|&x| {
            if x.is_finite() {
                s = if s.is_nan() { x } else { s + alpha * (x - s) };
            }
            s
        })
        .collect()
}

#[derive(Clone, Copy)]
pub enum Agg {
    Mean,
    Sum,
}

/// `agg` over the `n` days ending `lag` days before each day (lag 0 includes it),
/// skipping missing days; missing when the window has no present day.
pub fn trail(a: &[f64], n: usize, agg: Agg, lag: usize) -> Series {
    (0..a.len())
        .map(|i| {
            let end = (i + 1).saturating_sub(lag);
            let start = (i + 1).saturating_sub(lag + n);
            let present: Vec<f64> = a[start..end]
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .collect();
            match (present.is_empty(), agg) {
                (true, _) => f64::NAN,
                (false, Agg::Sum) => present.iter().sum(),
                (false, Agg::Mean) => present.iter().sum::<f64>() / present.len() as f64,
            }
        })
        .collect()
}

/// 1/0 where `x` is present, missing where it is not.
fn flag(x: &[f64], cond: impl Fn(f64) -> bool) -> Series {
    x.iter()
        .map(|&v| {
            if v.is_nan() {
                f64::NAN
            } else if cond(v) {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

fn zip(a: &[f64], b: &[f64], f: impl Fn(f64, f64) -> f64) -> Series {
    a.iter().zip(b).map(|(&x, &y)| f(x, y)).collect()
}

fn div(a: &[f64], b: &[f64]) -> Series {
    zip(a, b, |x, y| x / y)
}

fn sub(a: &[f64], b: &[f64]) -> Series {
    zip(a, b, |x, y| x - y)
}

fn add(a: &[f64], b: &[f64]) -> Series {
    zip(a, b, |x, y| x + y)
}

fn map(a: &[f64], f: impl Fn(f64) -> f64) -> Series {
    a.iter().map(|&x| f(x)).collect()
}

/// Build every named feature for consecutive `days`.
pub fn build(
    days: &[Day],
    weight_kg: f64,
    target_kcal: f64,
    class_edges: &[f64; 4],
) -> HashMap<&'static str, Series> {
    let night = |f: fn(&crate::Night) -> Option<f64>| -> Series {
        days.iter()
            .map(|d| d.night.as_ref().and_then(f).unwrap_or(f64::NAN))
            .collect()
    };
    let acts: Vec<Option<ActivityDay>> = days
        .iter()
        .map(|d| {
            d.met
                .as_ref()
                .filter(|m| !m.is_empty())
                .map(|m| ActivityDay::from_met(m, weight_kg, class_edges))
        })
        .collect();
    let act = |f: fn(&ActivityDay) -> f64| -> Series {
        acts.iter()
            .map(|a| a.as_ref().map_or(f64::NAN, f))
            .collect()
    };

    let lo = night(|n| n.lowest_hr);
    let hrv = night(|n| n.avg_hrv);
    let tst = night(|n| n.total_sleep_s);
    let p90 = night(|n| n.skin_temp_p90);
    let settle: Vec<(f64, f64)> = days
        .iter()
        .map(|d| {
            d.night
                .as_ref()
                .map_or((f64::NAN, f64::NAN), |n| hr_settle(n, 3.0))
        })
        .collect();
    let high = act(|a| a.high_mm);
    let medium = act(|a| a.medium_mm);
    let hm = add(&medium, &high);
    let allmm = add(&act(|a| a.low_mm), &hm);
    let kcal = act(|a| a.active_kcal);
    let min3 = act(|a| a.minutes_ge3);
    let mm5 = act(|a| a.met_min_ge5);
    let hm_7d = trail(&hm, 7, Agg::Sum, 0);

    let mut f: HashMap<&'static str, Series> = HashMap::new();
    f.insert("total_sleep", tst.clone());
    f.insert("efficiency", night(|n| n.efficiency));
    f.insert("rem", night(|n| n.rem_s));
    f.insert("deep", night(|n| n.deep_s));
    f.insert("latency", night(|n| n.latency_s));
    f.insert("awake", night(|n| n.awake_s));
    f.insert("lowest_hr", lo.clone());
    f.insert("lowest_hr_vs_30d", div(&lo, &prev(&ewma(&lo, 30.0))));
    f.insert(
        "hrv_3d_minus_30d",
        sub(&ewma(&hrv, 3.0), &prev(&ewma(&hrv, 30.0))),
    );
    f.insert(
        "hrv_7d_over_60d",
        div(&ewma(&hrv, 7.0), &prev(&ewma(&hrv, 60.0))),
    );
    f.insert("temp_deviation", sub(&p90, &prev(&ewma(&p90, 45.0))));
    f.insert(
        "sleep_7d_minus_8h",
        map(&ewma(&tst, 7.0), |x| x - 8.0 * 3600.0),
    );
    f.insert(
        "sleep_14d_over_90d",
        div(&ewma(&tst, 14.0), &prev(&ewma(&tst, 90.0))),
    );
    f.insert(
        "hr_settled_remaining_min",
        settle.iter().map(|s| s.0).collect(),
    );
    f.insert("hr_settle_frac", settle.iter().map(|s| s.1).collect());
    f.insert("prev_met_min", prev(&allmm));
    f.insert("prev_high_met_min", prev(&high));
    f.insert("prev_medium_met_min", prev(&medium));
    f.insert("prev_sedentary", prev(&act(|a| a.sedentary_s)));
    f.insert("prev_sedentary_runs", prev(&act(|a| a.sedentary_runs)));
    f.insert(
        "prev_met_min_vs_30d",
        div(&prev(&allmm), &prev(&ewma(&allmm, 30.0))),
    );
    f.insert(
        "kcal_7d_over_180d",
        div(
            &trail(&kcal, 7, Agg::Mean, 1),
            &trail(&kcal, 180, Agg::Mean, 1),
        ),
    );
    f.insert("kcal_180d", trail(&kcal, 180, Agg::Mean, 1));
    f.insert(
        "met_min_7d_over_180d",
        div(
            &trail(&allmm, 7, Agg::Mean, 1),
            &trail(&allmm, 180, Agg::Mean, 1),
        ),
    );
    f.insert("met_min_180d", trail(&allmm, 180, Agg::Mean, 1));
    f.insert(
        "days_target_met_7d",
        trail(&flag(&kcal, |k| k >= target_kcal), 7, Agg::Sum, 0),
    );
    f.insert(
        "target_ratio_7d",
        trail(
            &map(&kcal, |k| {
                if k.is_nan() {
                    k
                } else {
                    (k / target_kcal).min(1.0)
                }
            }),
            7,
            Agg::Mean,
            0,
        ),
    );
    f.insert("sedentary_runs", act(|a| a.sedentary_runs));
    f.insert("sedentary", act(|a| a.sedentary_s));
    f.insert(
        "hard_days_6d",
        trail(&flag(&min3, |m| m >= 183.0), 6, Agg::Sum, 0),
    );
    f.insert(
        "training_days_6d",
        trail(&flag(&mm5, |m| m >= 104.6), 6, Agg::Sum, 0),
    );
    f.insert("vigorous_met_min_6d", trail(&mm5, 6, Agg::Mean, 0));
    f.insert("hm_met_min_7d", hm_7d.clone());
    f.insert("training_load_7d_vs_16w", div(&hm_7d, &ewma(&hm, 112.0)));
    for series in f.values_mut() {
        for v in series.iter_mut() {
            if !v.is_finite() {
                *v = f64::NAN;
            }
        }
    }
    f
}

#[cfg(test)]
mod tests {
    use super::*;

    const N: f64 = f64::NAN;

    fn same(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| (x.is_nan() && y.is_nan()) || (x - y).abs() < 1e-12)
    }

    #[test]
    fn ewma_holds_through_gaps() {
        let got = ewma(&[N, 2.0, N, 4.0], 1.0);
        assert!(same(&got, &[N, 2.0, 2.0, 3.0]));
    }

    #[test]
    fn trail_skips_missing_and_honours_lag() {
        let a = [1.0, N, 3.0, 5.0];
        assert!(same(&trail(&a, 2, Agg::Mean, 0), &[1.0, 1.0, 3.0, 4.0]));
        assert!(same(&trail(&a, 2, Agg::Sum, 1), &[N, 1.0, 1.0, 3.0]));
        assert!(same(&trail(&[N, N], 3, Agg::Sum, 0), &[N, N]));
    }

    #[test]
    fn prev_shifts_by_one_day() {
        assert!(same(&prev(&[1.0, 2.0, 3.0]), &[N, 1.0, 2.0]));
        assert!(prev(&[]).is_empty());
    }
}
