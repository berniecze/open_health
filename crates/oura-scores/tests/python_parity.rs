//! Rust and `tools/fit_export_scores.py` must agree number for number. The fixture and
//! params are personal data (gitignored), so this runs only when pointed at them:
//!
//!   python tools/fit_export_scores.py "<export>/App Data" --fixture local/export_score_fixture.json
//!   OURA_SCORES_DIR=local cargo test -p oura-scores --test python_parity

use std::collections::HashMap;
use std::path::PathBuf;

use oura_scores::{model::score_all_days, Day, Params};
use serde::Deserialize;

#[derive(Deserialize)]
struct Expected {
    score: Option<f64>,
    contributors: HashMap<String, Option<f64>>,
}

#[derive(Deserialize)]
struct FixtureDay {
    #[serde(flatten)]
    day: Day,
    expected: HashMap<String, Expected>,
}

#[derive(Deserialize)]
struct Fixture {
    days: Vec<FixtureDay>,
}

#[test]
fn matches_python_fixture() {
    let Some(dir) = std::env::var_os("OURA_SCORES_DIR").map(PathBuf::from) else {
        eprintln!("skipped: set OURA_SCORES_DIR to the folder with export_score_*.json");
        return;
    };
    let params = Params::load(&dir.join("export_score_params.json")).unwrap();
    let fixture: Fixture = serde_json::from_str(
        &std::fs::read_to_string(dir.join("export_score_fixture.json")).unwrap(),
    )
    .unwrap();
    let days: Vec<Day> = fixture.days.iter().map(|d| d.day.clone()).collect();
    let (readiness, activity) = score_all_days(&days, &params).unwrap();

    let mut worst = (0.0f64, String::new());
    let mut checked = 0;
    for (i, fd) in fixture.days.iter().enumerate() {
        for (name, got) in [("readiness", &readiness[i]), ("activity", &activity[i])] {
            let want = &fd.expected[name];
            let mut pairs = vec![("score".to_string(), want.score, got.score)];
            pairs.extend(
                got.contributors
                    .iter()
                    .map(|(c, v)| (c.clone(), want.contributors[c], *v)),
            );
            for (what, want, got) in pairs {
                let want = want.expect("python predicts every day");
                let diff = (want - got).abs();
                checked += 1;
                if diff > 1e-6 && std::env::var_os("OURA_SCORES_VERBOSE").is_some() {
                    eprintln!("  {} {name}/{what}: python {want} rust {got}", fd.day.day);
                }
                if diff > worst.0 {
                    worst = (
                        diff,
                        format!("{} {name}/{what}: python {want} rust {got}", fd.day.day),
                    );
                }
            }
        }
    }
    eprintln!(
        "checked {checked} values; worst difference {:.2e} ({})",
        worst.0, worst.1
    );
    assert!(worst.0 < 1e-6, "rust diverges from python: {}", worst.1);
}
