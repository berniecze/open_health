//! Oura-calibrated Readiness and Activity scores from ring-derivable daily inputs.
//!
//! The model is fitted offline by `tools/fit_export_scores.py` against a user's own
//! Oura account export and saved as a JSON parameter file; this crate only evaluates
//! it. Every input is something the app computes from the ring itself: the main sleep
//! that ended each morning and that day's per-minute MET. Oura's own scores were the
//! fit targets, never inputs.
//!
//! Two layers, mirroring the fit:
//!
//! * **contributors**: each Oura contributor (0–100) is a linear model over spline,
//!   lookup and linear bases of named, causal features ([`features`]).
//! * **combiner**: Activity is a weighted sum of its six contributors; Readiness is a
//!   weighted sum of its eight plus a penalty below a temperature contributor of
//!   `temp_knee`.
//!
//! [`features`] mirrors the Python feature code line for line (including how missing
//! values propagate), so the same parameters give the same numbers on both sides;
//! `tests/python_parity.rs` checks that against a fixture the Python tool writes.

pub mod daily;
pub mod features;
pub mod model;

pub use daily::{ActivityDay, Day, Night};
pub use model::{score_days, DayScores, Params, Score};
