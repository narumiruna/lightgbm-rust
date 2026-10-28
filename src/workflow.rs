//! Typed, read-only training hooks and Python-compatible stopping.
//!
//! Ported from LightGBM python-package/lightgbm/{engine,callback}.py at
//! 5d84f3f222b41409a6983d2ac90513ad9702fa9e (MIT).
//! Copyright (c) 2016-2026 Microsoft Corporation and The LightGBM developers.

use crate::config::Boosting;
use crate::metric::MetricValue;
use crate::trainer::{Evaluation, IterationEvaluation};
use crate::{Config, Dataset, Error, Model, Result};

/// Callback input. Iteration indexes are zero-based. All buffers are read-only.
/// Equal callback orders retain insertion order. History is recorded at order 20.
pub struct CallbackEnv<'a> {
    pub config: &'a Config,
    pub iteration: usize,
    pub begin_iteration: usize,
    pub end_iteration: usize,
    pub model_current_iteration: usize,
    pub model_num_trees: usize,
    pub best_iteration: usize,
    pub evaluation: Option<&'a [Evaluation]>,
    pub evaluation_history: &'a [IterationEvaluation],
    pub training_scores: &'a [f64],
    pub training_name: &'a str,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CallbackAction {
    Continue,
    SetLearningRate(f64),
    /// Select a one-based iteration and the complete metric snapshot for it.
    Stop {
        best_iteration: usize,
        best_scores: Vec<Evaluation>,
    },
}

pub trait TrainingCallback {
    fn order(&self) -> i32 {
        0
    }
    fn before_iteration(&self) -> bool {
        false
    }
    fn call(&mut self, env: &CallbackEnv<'_>) -> Result<CallbackAction>;
}

/// Derivatives and margins use class-major layout: class * rows + row.
/// Hessians must be finite and nonnegative. Weights are the caller's responsibility.
pub trait CustomObjective {
    fn gradients(&mut self, raw_scores: &[f64], data: &Dataset) -> Result<(Vec<f32>, Vec<f32>)>;
}
impl<F> CustomObjective for F
where
    F: FnMut(&[f64], &Dataset) -> Result<(Vec<f32>, Vec<f32>)>,
{
    fn gradients(&mut self, scores: &[f64], data: &Dataset) -> Result<(Vec<f32>, Vec<f32>)> {
        self(scores, data)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredictionKind {
    Raw,
    Transformed,
}

/// Built-in objectives supply transformed predictions; custom objectives supply
/// raw margins, as in Python feval. Multiclass buffers remain class-major.
pub trait CustomMetric {
    fn evaluate(
        &mut self,
        predictions: &[f64],
        data: &Dataset,
        kind: PredictionKind,
    ) -> Result<Vec<MetricValue>>;
}
impl<F> CustomMetric for F
where
    F: FnMut(&[f64], &Dataset, PredictionKind) -> Result<Vec<MetricValue>>,
{
    fn evaluate(
        &mut self,
        scores: &[f64],
        data: &Dataset,
        kind: PredictionKind,
    ) -> Result<Vec<MetricValue>> {
        self(scores, data, kind)
    }
}

/// Workflow fits attempt every requested round, even if the learner cannot split.
/// The model retains all completed trees. `best_iteration` selects a prediction
/// prefix; stopping does not silently remove the patience rounds from the model.
pub struct FitOptions<'a> {
    /// GBDT/GOSS/DART stored-model warm start, not an RNG checkpoint. RF
    /// continuation is rejected. Supply raw class-major predictions from this
    /// model through each dataset's initial scores.
    /// Validation margins are required too; binned data cannot recover raw routing.
    pub initial_model: Option<&'a Model>,
    pub callbacks: Vec<Box<dyn TrainingCallback + 'a>>,
    pub custom_objective: Option<Box<dyn CustomObjective + 'a>>,
    pub custom_metrics: Vec<Box<dyn CustomMetric + 'a>>,
    /// Evaluate the training data before the validation sets. Passing the training
    /// dataset itself in the validation list also enables this option and uses
    /// that entry's name. Training metrics do not select an early-stopping round.
    pub evaluate_training: bool,
    pub training_name: String,
}
impl Default for FitOptions<'_> {
    fn default() -> Self {
        Self {
            initial_model: None,
            callbacks: Vec::new(),
            custom_objective: None,
            custom_metrics: Vec::new(),
            evaluate_training: false,
            training_name: "training".into(),
        }
    }
}

/// A before-iteration schedule at order 10. Changed DART rates reset dropout
/// RNG and the weight sum, but retain per-tree weights, as in native ResetConfig.
/// Changed rates with `feature_fraction < 1` require a learner reset hook and
/// currently return `Error::Unsupported`; unchanged rates do not reset state.
pub struct LearningRateSchedule<'a> {
    schedule: Schedule<'a>,
}
enum Schedule<'a> {
    Values(Vec<f64>),
    Function(Box<dyn FnMut(usize) -> f64 + 'a>),
}
impl<'a> LearningRateSchedule<'a> {
    pub fn values(values: Vec<f64>) -> Self {
        Self {
            schedule: Schedule::Values(values),
        }
    }
    pub fn function(f: impl FnMut(usize) -> f64 + 'a) -> Self {
        Self {
            schedule: Schedule::Function(Box::new(f)),
        }
    }
}
impl TrainingCallback for LearningRateSchedule<'_> {
    fn order(&self) -> i32 {
        10
    }
    fn before_iteration(&self) -> bool {
        true
    }
    fn call(&mut self, env: &CallbackEnv<'_>) -> Result<CallbackAction> {
        let index = env.iteration - env.begin_iteration;
        let value = match &mut self.schedule {
            Schedule::Values(values) => {
                if values.len() != env.end_iteration - env.begin_iteration {
                    return invalid(
                        "learning-rate schedule length must equal the requested rounds",
                    );
                }
                values[index]
            }
            Schedule::Function(f) => f(index),
        };
        if !value.is_finite() || value <= 0.0 {
            return invalid("learning rate must be finite and positive");
        }
        Ok(CallbackAction::SetLearningRate(value))
    }
}

/// Python callback semantics: each eligible dataset/metric has its own best
/// snapshot. The first metric that exhausts patience selects the result.
pub struct EarlyStopping {
    stopping_rounds: usize,
    first_metric_only: bool,
    min_delta: Vec<f64>,
    enabled: bool,
    first_metric: String,
    deltas: Vec<f64>,
    best: Vec<f64>,
    best_iterations: Vec<usize>,
    snapshots: Vec<Vec<Evaluation>>,
}
impl EarlyStopping {
    pub fn new(stopping_rounds: usize) -> Self {
        Self {
            stopping_rounds,
            first_metric_only: false,
            min_delta: vec![0.0],
            enabled: true,
            first_metric: String::new(),
            deltas: Vec::new(),
            best: Vec::new(),
            best_iterations: Vec::new(),
            snapshots: Vec::new(),
        }
    }
    pub fn first_metric_only(mut self, value: bool) -> Self {
        self.first_metric_only = value;
        self
    }
    /// Empty means zero; one value applies to all metrics; otherwise one value
    /// per distinct metric, repeated in dataset order. Improvements are strict.
    pub fn min_delta(mut self, values: Vec<f64>) -> Self {
        self.min_delta = values;
        self
    }
    fn init(&mut self, env: &CallbackEnv<'_>, values: &[Evaluation]) -> Result<()> {
        if values.is_empty() {
            return invalid("early stopping requires at least one dataset and metric");
        }
        self.enabled = self.stopping_rounds > 0 && env.config.boosting != Boosting::Dart;
        if values.len() == 1 && values[0].dataset == env.training_name {
            self.enabled = false;
        }
        if !self.enabled {
            return Ok(());
        }
        if self.min_delta.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return invalid("early stopping min_delta must be finite and nonnegative");
        }
        let metrics = values
            .iter()
            .map(|v| v.metric.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        if self.min_delta.len() > 1 && self.min_delta.len() != metrics {
            return invalid("min_delta must contain one value or one value per metric");
        }
        self.first_metric = values[0].metric.clone();
        self.deltas = (0..values.len())
            .map(|i| {
                if self.min_delta.is_empty() {
                    0.0
                } else {
                    self.min_delta[i % self.min_delta.len()]
                }
            })
            .collect();
        self.best = values
            .iter()
            .map(|v| {
                if v.higher_is_better {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                }
            })
            .collect();
        self.best_iterations = vec![0; values.len()];
        self.snapshots = Vec::new();
        Ok(())
    }
}
impl TrainingCallback for EarlyStopping {
    fn order(&self) -> i32 {
        30
    }
    fn call(&mut self, env: &CallbackEnv<'_>) -> Result<CallbackAction> {
        let values = env.evaluation.ok_or_else(|| {
            Error::InvalidInput("early stopping requires after-iteration evaluation".into())
        })?;
        if env.iteration == env.begin_iteration {
            self.init(env, values)?;
        }
        if !self.enabled {
            return Ok(CallbackAction::Continue);
        }
        if values.len() != self.best.len() {
            return invalid("evaluation layout changed during early stopping");
        }
        let first = self.snapshots.is_empty();
        for (i, value) in values.iter().enumerate() {
            let improved = if value.higher_is_better {
                value.value > self.best[i] + self.deltas[i]
            } else {
                value.value < self.best[i] - self.deltas[i]
            };
            if first || improved {
                self.best[i] = value.value;
                self.best_iterations[i] = env.iteration;
                if first {
                    self.snapshots.push(values.to_vec());
                } else {
                    self.snapshots[i] = values.to_vec();
                }
            }
            if self.first_metric_only && value.metric != self.first_metric {
                continue;
            }
            if value.dataset == env.training_name {
                continue;
            }
            if env.iteration - self.best_iterations[i] >= self.stopping_rounds
                || env.iteration + 1 == env.end_iteration
            {
                return Ok(CallbackAction::Stop {
                    best_iteration: self.best_iterations[i] + 1,
                    best_scores: self.snapshots[i].clone(),
                });
            }
        }
        Ok(CallbackAction::Continue)
    }
}

fn invalid<T>(message: &str) -> Result<T> {
    Err(Error::InvalidInput(message.into()))
}
