//! Parameter file and evaluation. The JSON is written by
//! `tools/fit_export_scores.py`; see that tool for how it is fitted.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::daily::{Day, DEFAULT_CLASS_EDGES};
use crate::features::{self, Series};

pub const PARAMS_VERSION: u32 = 1;

#[derive(Debug, Clone, Deserialize)]
pub struct Params {
    pub version: u32,
    pub weight_kg: f64,
    /// Daily active-kcal target, in this crate's MET-derived kcal units.
    pub target_kcal: f64,
    pub temp_knee: f64,
    #[serde(default = "default_edges")]
    pub class_edges: [f64; 4],
    pub scores: HashMap<String, ScoreParams>,
}

fn default_edges() -> [f64; 4] {
    DEFAULT_CLASS_EDGES
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScoreParams {
    /// Contributor order of the combiner weights.
    pub order: Vec<String>,
    pub contributors: HashMap<String, ContributorParams>,
    pub combiner: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContributorParams {
    /// `[feature, kind, knots]` with kind `spline` | `lookup` | `linear`.
    pub inputs: Vec<(String, String, Vec<f64>)>,
    /// One weight per basis column, then the missing flag, then the intercept.
    pub weights: Vec<f64>,
}

impl Params {
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let params: Params =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        params.validate()?;
        Ok(params)
    }

    fn validate(&self) -> Result<()> {
        if self.version != PARAMS_VERSION {
            bail!("score params version {} (expected {PARAMS_VERSION}): re-run tools/fit_export_scores.py", self.version);
        }
        for (name, score) in &self.scores {
            let extra = usize::from(name == "readiness"); // temperature hinge
            if score.combiner.len() != score.order.len() + extra {
                bail!(
                    "{name}: {} combiner weights for {} contributors",
                    score.combiner.len(),
                    score.order.len()
                );
            }
            for c in &score.order {
                let p = score
                    .contributors
                    .get(c)
                    .ok_or_else(|| anyhow!("{name}: no params for {c}"))?;
                let cols: usize = p
                    .inputs
                    .iter()
                    .map(|(_, kind, knots)| match kind.as_str() {
                        "linear" => Ok(1),
                        "lookup" => Ok(knots.len()),
                        "spline" => Ok(1 + knots.len()),
                        k => Err(anyhow!("{name}/{c}: unknown basis kind {k}")),
                    })
                    .sum::<Result<_>>()?;
                if p.weights.len() != cols + 2 {
                    bail!(
                        "{name}/{c}: {} weights for {cols} basis columns",
                        p.weights.len()
                    );
                }
            }
        }
        Ok(())
    }
}

/// A score and its contributors for one day.
#[derive(Debug, Clone, Serialize)]
pub struct Score {
    pub score: f64,
    pub contributors: Vec<(String, f64)>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DayScores {
    pub day: String,
    /// Present when the day has a night to score.
    pub readiness: Option<Score>,
    /// Present when the day has activity data.
    pub activity: Option<Score>,
}

/// Evaluate one contributor for every day (unrounded, clipped to 1–100).
fn contributor(
    p: &ContributorParams,
    feats: &HashMap<&'static str, Series>,
    n_days: usize,
) -> Result<Series> {
    let mut out = Vec::with_capacity(n_days);
    for i in 0..n_days {
        let mut cols: Vec<f64> = Vec::with_capacity(p.weights.len());
        for (name, kind, knots) in &p.inputs {
            let x = feats
                .get(name.as_str())
                .ok_or_else(|| anyhow!("unknown feature {name}"))?[i];
            match kind.as_str() {
                "linear" => cols.push(x),
                // a missing count matches no key: all-zero, not missing
                "lookup" => cols.extend(knots.iter().map(|k| if x == *k { 1.0 } else { 0.0 })),
                _ => {
                    cols.push(x);
                    // f64::max would swallow a missing x; numpy's maximum keeps it
                    cols.extend(
                        knots
                            .iter()
                            .map(|k| if x.is_nan() { x } else { (x - k).max(0.0) }),
                    );
                }
            }
        }
        let missing = cols.iter().any(|v| v.is_nan());
        cols.iter_mut()
            .filter(|v| v.is_nan())
            .for_each(|v| *v = 0.0);
        cols.push(if missing { 1.0 } else { 0.0 });
        cols.push(1.0);
        let y: f64 = cols.iter().zip(&p.weights).map(|(x, w)| x * w).sum();
        out.push(y.clamp(1.0, 100.0));
    }
    Ok(out)
}

fn score(
    name: &str,
    sp: &ScoreParams,
    temp_knee: f64,
    feats: &HashMap<&'static str, Series>,
    n: usize,
) -> Result<Vec<Score>> {
    let values: Vec<Series> = sp
        .order
        .iter()
        .map(|c| contributor(&sp.contributors[c], feats, n))
        .collect::<Result<_>>()?;
    let temp = sp.order.iter().position(|c| c == "body_temperature");
    Ok((0..n)
        .map(|i| {
            let mut cols: Vec<f64> = values.iter().map(|v| v[i]).collect();
            if name == "readiness" {
                let t = temp.map_or(f64::NAN, |j| cols[j]);
                cols.push((temp_knee - t).max(0.0));
            }
            Score {
                score: cols.iter().zip(&sp.combiner).map(|(x, w)| x * w).sum(),
                contributors: sp
                    .order
                    .iter()
                    .cloned()
                    .zip(values.iter().map(|v| v[i]))
                    .collect(),
            }
        })
        .collect())
}

/// Readiness and Activity for every day, including days without data (whose values
/// rest on the intercepts alone). Prefer [`score_days`].
pub fn score_all_days(days: &[Day], params: &Params) -> Result<(Vec<Score>, Vec<Score>)> {
    let feats = features::build(
        days,
        params.weight_kg,
        params.target_kcal,
        &params.class_edges,
    );
    let get = |name: &str| -> Result<Vec<Score>> {
        let sp = params
            .scores
            .get(name)
            .ok_or_else(|| anyhow!("params have no {name} score"))?;
        score(name, sp, params.temp_knee, &feats, days.len())
    };
    Ok((get("readiness")?, get("activity")?))
}

/// Score consecutive `days`. Earlier days only provide history for later ones, so
/// pass as much history as is available.
pub fn score_days(days: &[Day], params: &Params) -> Result<Vec<DayScores>> {
    let (readiness, activity) = score_all_days(days, params)?;
    Ok(days
        .iter()
        .zip(readiness.into_iter().zip(activity))
        .map(|(d, (r, a))| DayScores {
            day: d.day.clone(),
            readiness: d.night.is_some().then_some(r),
            activity: d.met.as_ref().is_some_and(|m| !m.is_empty()).then_some(a),
        })
        .collect())
}
