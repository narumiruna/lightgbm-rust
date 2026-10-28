//! Margin-based prediction stopping from pinned LightGBM
//! `prediction_early_stop.cpp`, `gbdt_prediction.cpp`, and `predictor.hpp`.
//! Copyright (c) Microsoft Corporation and The LightGBM developers. MIT License.

use super::{IterationRange, Model};
use crate::{Error, Result};

/// Check raw classification or ranking margins after each complete period.
/// Other objectives require accurate prediction and ignore this option, as in
/// the native predictor. This is not training early stopping.
#[derive(Clone, Copy, Debug)]
pub struct PredictionEarlyStop {
    period: usize,
    margin: f64,
}

impl PredictionEarlyStop {
    pub fn new(period: usize, margin: f64) -> Result<Self> {
        if period == 0 || period > i32::MAX as usize || !margin.is_finite() || margin < 0.0 {
            return Err(Error::InvalidInput("prediction stopping requires a positive int32 period and finite nonnegative margin".into()));
        }
        Ok(Self { period, margin })
    }
    pub fn period(self) -> usize {
        self.period
    }
    pub fn margin(self) -> f64 {
        self.margin
    }

    fn reached(self, scores: &[f64]) -> bool {
        if scores.len() == 1 {
            return 2.0 * scores[0].abs() > self.margin;
        }
        let (mut first, mut second) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for &score in scores {
            if score > first {
                second = first;
                first = score;
            } else if score > second {
                second = score;
            }
        }
        first - second > self.margin
    }
}

pub(super) fn predict(
    model: &Model,
    get: &impl Fn(usize) -> f64,
    iterations: IterationRange,
    stopping: PredictionEarlyStop,
) -> Result<Vec<f64>> {
    let range = model.tree_range(iterations)?;
    let enabled = matches!(
        model.objective.split_whitespace().next(),
        Some("binary" | "multiclass" | "multiclassova" | "lambdarank" | "rank_xendcg")
    );
    let mut scores = vec![0.0; model.num_tree_per_iteration];
    let mut counter = 0;
    for trees in model.trees[range].chunks_exact(model.num_tree_per_iteration) {
        for (score, tree) in scores.iter_mut().zip(trees) {
            *score += tree.score_with(get);
        }
        counter += 1;
        if enabled && counter == stopping.period {
            if stopping.reached(&scores) {
                break;
            }
            counter = 0;
        }
    }
    Ok(scores)
}
