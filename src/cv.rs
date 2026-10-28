//! Serial cross-validation with immutable shared bins and atomic results.
//!
//! Fold generation follows pinned Python engine.py, NumPy RandomState MT19937,
//! and scikit-learn 1.6 StratifiedKFold. Group allocation uses descending group
//! size, descending group index for ties, and the first least-loaded fold.
//! Equal-size group ties match the pinned fixture; NumPy's unstable argsort
//! tie ordering is not a portable contract for other group layouts.
//! # Resource limits
//!
//! Before fold indexes are built or explicit indexes are cloned, CV checks
//! 16 Mi aggregate index entries, 16 Mi aggregate row-feature pairs (including
//! implicit sparse zeros), and a 256 MiB conservative fold-storage estimate.
//! The estimate charges 64 bytes per row-feature pair (bins, sparse capacity,
//! raw values and lazy state), 128 bytes per selected row (indexes, metadata
//! and row state), 1024 bytes per feature-fold (column and learner headers),
//! 32 KiB per fold (fixed state), and actual bin-count lengths for both sides.
//! Training also checks 16 Mi aggregate class-row score entries and charges
//! 64 bytes per entry for scores, derivatives and temporary score state.
//! Initial-margin metadata is charged separately at eight bytes per entry.
//! These are intrinsic admission limits, not measured peak RSS or a bound on
//! models, history, histogram workspaces, caller callbacks or configuration
//! payloads. Sparse shapes use the stated dense upper bound, not stored nnz.
//! Retained fold pools are limited to 256 workers in total; zero or one thread
//! uses serial execution without a pool. All count arithmetic is checked.
//!
//! Ported from LightGBM at 5d84f3f222b41409a6983d2ac90513ad9702fa9e (MIT).
//! Copyright (c) 2016-2026 Microsoft Corporation and The LightGBM developers.

mod random;
use crate::trainer::{Evaluation, IterationEvaluation};
use crate::workflow::{
    CallbackAction, CallbackEnv, CustomMetric, CustomObjective, EarlyStopping, FitOptions,
    LearningRateSchedule, TrainingCallback,
};
use crate::{Config, Dataset, Error, Model, Result, Trainer};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fold {
    pub train_indices: Vec<usize>,
    pub valid_indices: Vec<usize>,
}
#[derive(Clone, Debug)]
pub enum FoldStrategy {
    Explicit(Vec<Fold>),
    Standard {
        nfold: usize,
        shuffle: bool,
        seed: u32,
    },
    Stratified {
        nfold: usize,
        shuffle: bool,
        seed: u32,
    },
    Group {
        nfold: usize,
    },
}

/// One independent objective instance is made for each fold, in fold order.
pub type CvObjectiveFactory<'a> =
    Box<dyn FnMut(usize) -> Result<Box<dyn CustomObjective + 'a>> + 'a>;
/// Each factory makes one independent metric instance per fold.
pub type CvMetricFactory<'a> = Box<dyn FnMut(usize) -> Result<Box<dyn CustomMetric + 'a>> + 'a>;

/// Read-only aggregate callback state. Iteration indexes are zero-based.
/// History is recorded at order 20; stopping runs at order 30.
pub struct CvCallbackEnv<'a> {
    pub config: &'a Config,
    pub iteration: usize,
    pub begin_iteration: usize,
    pub end_iteration: usize,
    pub fold_current_iterations: &'a [usize],
    pub fold_num_trees: &'a [usize],
    pub best_iteration: usize,
    pub evaluation: Option<&'a [CvMetric]>,
    pub evaluation_history: &'a [CvIteration],
}
pub trait CvCallback {
    fn order(&self) -> i32 {
        0
    }
    fn before_iteration(&self) -> bool {
        false
    }
    fn call(&mut self, env: &CvCallbackEnv<'_>) -> Result<CallbackAction>;
}
fn call_training_callback(
    callback: &mut dyn TrainingCallback,
    env: &CvCallbackEnv<'_>,
) -> Result<CallbackAction> {
    let values = env.evaluation.map(|values| {
        values
            .iter()
            .map(|m| m.evaluation.clone())
            .collect::<Vec<_>>()
    });
    let history = env
        .evaluation_history
        .iter()
        .map(|e| IterationEvaluation {
            iteration: e.iteration,
            results: e.results.iter().map(|m| m.evaluation.clone()).collect(),
        })
        .collect::<Vec<_>>();
    callback.call(&CallbackEnv {
        config: env.config,
        iteration: env.iteration,
        begin_iteration: env.begin_iteration,
        end_iteration: env.end_iteration,
        model_current_iteration: env.fold_current_iterations.first().copied().unwrap_or(0),
        model_num_trees: env.fold_num_trees.iter().sum(),
        best_iteration: env.best_iteration,
        evaluation: values.as_deref(),
        evaluation_history: &history,
        training_scores: &[],
        training_name: "train",
    })
}
impl CvCallback for EarlyStopping {
    fn order(&self) -> i32 {
        30
    }
    fn call(&mut self, env: &CvCallbackEnv<'_>) -> Result<CallbackAction> {
        call_training_callback(self, env)
    }
}
impl CvCallback for LearningRateSchedule<'_> {
    fn order(&self) -> i32 {
        10
    }
    fn before_iteration(&self) -> bool {
        true
    }
    fn call(&mut self, env: &CvCallbackEnv<'_>) -> Result<CallbackAction> {
        call_training_callback(self, env)
    }
}

pub struct CvOptions<'a> {
    pub folds: FoldStrategy,
    pub evaluate_training: bool,
    pub early_stopping: Option<EarlyStopping>,
    pub learning_rates: Option<Vec<f64>>,
    pub callbacks: Vec<Box<dyn CvCallback + 'a>>,
    pub custom_objective_factory: Option<CvObjectiveFactory<'a>>,
    pub custom_metric_factories: Vec<CvMetricFactory<'a>>,
}
impl Default for CvOptions<'_> {
    fn default() -> Self {
        Self {
            folds: FoldStrategy::Standard {
                nfold: 5,
                shuffle: true,
                seed: 0,
            },
            evaluate_training: false,
            early_stopping: None,
            learning_rates: None,
            callbacks: Vec::new(),
            custom_objective_factory: None,
            custom_metric_factories: Vec::new(),
        }
    }
}
#[derive(Clone, Debug)]
pub struct CvMetric {
    pub evaluation: Evaluation,
    pub std_dev: f64,
}
#[derive(Clone, Debug)]
pub struct CvIteration {
    pub iteration: usize,
    pub results: Vec<CvMetric>,
}
#[derive(Clone, Debug)]
pub struct CvResult {
    pub models: Vec<Model>,
    pub folds: Vec<Fold>,
    /// Truncated to the common best iteration when early stopping is enabled.
    pub history: Vec<CvIteration>,
    pub best_iteration: Option<usize>,
    /// Includes patience rounds; model prefixes end at this attempted round.
    pub completed_rounds: usize,
}

/// Folds advance sequentially by one round. Callbacks and aggregate stopping
/// run before any fold can start the next round. No partial result is returned
/// if a fold, factory, custom function, or callback fails.
pub fn cross_validate(
    mut config: Config,
    data: &Dataset,
    mut options: CvOptions<'_>,
) -> Result<CvResult> {
    config.validate()?;
    data.check_binning_config(&config)?;
    check_resources(data, &options.folds, Some(&config))?;
    let folds = make_folds(data, &options.folds)?;
    let datasets = folds
        .iter()
        .map(|fold| {
            Ok((
                data.subset(&fold.train_indices)?,
                data.subset(&fold.valid_indices)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut sessions = Vec::with_capacity(folds.len());
    for (index, (train, valid)) in datasets.iter().enumerate() {
        let mut fit = FitOptions {
            evaluate_training: options.evaluate_training,
            training_name: "train".into(),
            ..FitOptions::default()
        };
        if let Some(factory) = &mut options.custom_objective_factory {
            fit.custom_objective = Some(factory(index)?);
        }
        for factory in &mut options.custom_metric_factories {
            fit.custom_metrics.push(factory(index)?);
        }
        sessions.push(Trainer::new(config.clone())?.start_session(
            train,
            &[("valid", valid)],
            fit,
            true,
        )?);
    }
    if let Some(rates) = options.learning_rates {
        options
            .callbacks
            .push(Box::new(LearningRateSchedule::values(rates)));
    }
    if let Some(stopping) = options.early_stopping {
        options.callbacks.push(Box::new(stopping));
    }
    options
        .callbacks
        .sort_by_key(|c| (!c.before_iteration(), c.order()));
    let rounds = config.num_iterations;
    let mut history = Vec::new();
    let mut best_iteration = None;
    let mut completed_rounds = 0;
    for iteration in 0..rounds {
        for callback in options
            .callbacks
            .iter_mut()
            .filter(|c| c.before_iteration())
        {
            let current = sessions
                .iter()
                .map(|s| s.num_trees() / config.num_class)
                .collect::<Vec<_>>();
            let trees = sessions.iter().map(|s| s.num_trees()).collect::<Vec<_>>();
            let env = CvCallbackEnv {
                config: &config,
                iteration,
                begin_iteration: 0,
                end_iteration: rounds,
                fold_current_iterations: &current,
                fold_num_trees: &trees,
                best_iteration: best_iteration.unwrap_or(0),
                evaluation: None,
                evaluation_history: &history,
            };
            apply_action(
                callback.call(&env)?,
                &mut config,
                &mut sessions,
                iteration,
                true,
                &mut best_iteration,
            )?;
        }
        for session in &mut sessions {
            session.step()?;
        }
        completed_rounds = iteration + 1;
        let fold_results = sessions
            .iter()
            .map(|s| {
                s.history()
                    .last()
                    .expect("workflow round records evaluation")
                    .results
                    .as_slice()
            })
            .collect::<Vec<_>>();
        let results = aggregate(&fold_results)?;
        let entry = CvIteration {
            iteration: iteration + 1,
            results: results.clone(),
        };
        let current = sessions
            .iter()
            .map(|s| s.num_trees() / config.num_class)
            .collect::<Vec<_>>();
        let trees = sessions.iter().map(|s| s.num_trees()).collect::<Vec<_>>();
        let mut recorded = false;
        let mut stopped = false;
        for callback in options
            .callbacks
            .iter_mut()
            .filter(|c| !c.before_iteration())
        {
            if !recorded && callback.order() >= 20 {
                history.push(entry.clone());
                recorded = true;
            }
            let env = CvCallbackEnv {
                config: &config,
                iteration,
                begin_iteration: 0,
                end_iteration: rounds,
                fold_current_iterations: &current,
                fold_num_trees: &trees,
                best_iteration: best_iteration.unwrap_or(0),
                evaluation: Some(&results),
                evaluation_history: &history,
            };
            if apply_action(
                callback.call(&env)?,
                &mut config,
                &mut sessions,
                iteration,
                false,
                &mut best_iteration,
            )? {
                stopped = true;
                break;
            }
        }
        if !recorded {
            history.push(entry);
        }
        if stopped {
            break;
        }
    }
    if let Some(best) = best_iteration {
        history.truncate(best);
    }
    let models = sessions
        .into_iter()
        .map(|s| Ok(s.finish()?.model))
        .collect::<Result<_>>()?;
    Ok(CvResult {
        models,
        folds,
        history,
        best_iteration,
        completed_rounds,
    })
}

fn apply_action(
    action: CallbackAction,
    config: &mut Config,
    sessions: &mut [crate::trainer::TrainingSession<'_>],
    iteration: usize,
    before: bool,
    best_iteration: &mut Option<usize>,
) -> Result<bool> {
    match action {
        CallbackAction::Continue => {}
        CallbackAction::SetLearningRate(rate) => {
            if !rate.is_finite() || rate <= 0.0 {
                return invalid("learning rate must be finite and positive");
            }
            for session in sessions {
                session.reset_learning_rate(rate)?;
            }
            config.learning_rate = rate;
        }
        CallbackAction::Stop {
            best_iteration: selected,
            ..
        } => {
            if before {
                return invalid("stop actions require an after-iteration callback");
            }
            if selected == 0 || selected > iteration + 1 {
                return invalid("invalid CV callback best iteration");
            }
            *best_iteration = Some(selected);
            return Ok(true);
        }
    }
    Ok(false)
}

fn aggregate(folds: &[&[Evaluation]]) -> Result<Vec<CvMetric>> {
    let first = folds[0];
    if folds.iter().any(|f| f.len() != first.len()) {
        return invalid("CV fold metric layout differs");
    }
    let mut out = Vec::with_capacity(first.len());
    for (index, metric) in first.iter().enumerate() {
        let mut values = Vec::with_capacity(folds.len());
        for fold in folds {
            let m = &fold[index];
            if m.dataset != metric.dataset
                || m.metric != metric.metric
                || m.higher_is_better != metric.higher_is_better
            {
                return invalid("CV fold metric layout differs");
            }
            values.push(m.value);
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let std_dev =
            (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt();
        let mut evaluation = metric.clone();
        evaluation.value = mean;
        out.push(CvMetric {
            evaluation,
            std_dev,
        });
    }
    Ok(out)
}

/// Return sorted row indexes, as engine.py sorts every split before subsetting.
/// Explicit folds need not cover every row or be disjoint across folds; each
/// individual fold must have nonempty, disjoint, duplicate-free sides. Dataset
/// subsets require complete contiguous query groups for all split strategies.
/// Validation here reads row indexes and query lengths; it does not copy data.
/// The module's aggregate resource limits apply before construction or cloning.
pub fn make_folds(data: &Dataset, strategy: &FoldStrategy) -> Result<Vec<Fold>> {
    check_resources(data, strategy, None)?;
    let n = data.nrows();
    let mut folds = match strategy {
        FoldStrategy::Explicit(folds) => folds.clone(),
        FoldStrategy::Standard {
            nfold,
            shuffle,
            seed,
        } => {
            check_nfold(*nfold, n)?;
            let mut rows = (0..n).collect::<Vec<_>>();
            if *shuffle {
                random::Mt19937::new(*seed).shuffle(&mut rows);
            }
            let mut start = 0;
            (0..*nfold)
                .map(|i| {
                    let len = n / nfold + usize::from(i < n % nfold);
                    let valid = rows[start..start + len].to_vec();
                    start += len;
                    complement(n, valid)
                })
                .collect()
        }
        FoldStrategy::Stratified {
            nfold,
            shuffle,
            seed,
        } => {
            check_nfold(*nfold, n)?;
            let mut labels = Vec::new();
            let mut encoded = Vec::with_capacity(n);
            for &y in data.labels() {
                if y.fract() != 0.0 {
                    return invalid("stratified folds require discrete class labels");
                }
                let c = if let Some(c) = labels.iter().position(|&v| v == y) {
                    c
                } else {
                    labels.push(y);
                    labels.len() - 1
                };
                encoded.push(c);
            }
            let mut counts = vec![0; labels.len()];
            for &c in &encoded {
                counts[c] += 1;
            }
            if counts.iter().all(|&c| c < *nfold) {
                return invalid("every class has fewer rows than folds");
            }
            let mut sorted = encoded.clone();
            sorted.sort_unstable();
            let mut allocation = vec![vec![0; labels.len()]; *nfold];
            for (i, &c) in sorted.iter().enumerate() {
                allocation[i % nfold][c] += 1;
            }
            let mut assignment = vec![0; n];
            let mut rng = random::Mt19937::new(*seed);
            for c in 0..labels.len() {
                let mut class_folds = Vec::with_capacity(counts[c]);
                for (fold, a) in allocation.iter().enumerate() {
                    class_folds.extend(std::iter::repeat_n(fold, a[c]));
                }
                if *shuffle {
                    rng.shuffle(&mut class_folds);
                }
                for (row, fold) in encoded
                    .iter()
                    .enumerate()
                    .filter(|&(_, v)| *v == c)
                    .map(|(r, _)| r)
                    .zip(class_folds)
                {
                    assignment[row] = fold;
                }
            }
            (0..*nfold)
                .map(|i| complement(n, (0..n).filter(|&r| assignment[r] == i).collect()))
                .collect()
        }
        FoldStrategy::Group { nfold } => {
            let groups = data
                .groups()
                .ok_or_else(|| Error::InvalidInput("group folds require query groups".into()))?;
            check_nfold(*nfold, groups.len())?;
            let mut order = (0..groups.len()).collect::<Vec<_>>();
            order.sort_by_key(|&g| std::cmp::Reverse((groups[g], g)));
            let mut sizes = vec![0; *nfold];
            let mut assigned = vec![0; groups.len()];
            for g in order {
                let fold = (0..*nfold).min_by_key(|&i| sizes[i]).unwrap();
                assigned[g] = fold;
                sizes[fold] += groups[g];
            }
            let mut assignment = Vec::with_capacity(n);
            for (g, &size) in groups.iter().enumerate() {
                assignment.extend(std::iter::repeat_n(assigned[g], size));
            }
            (0..*nfold)
                .map(|i| complement(n, (0..n).filter(|&r| assignment[r] == i).collect()))
                .collect()
        }
    };
    if folds.is_empty() {
        return invalid("CV requires at least one fold");
    }
    for fold in &mut folds {
        fold.train_indices.sort_unstable();
        fold.valid_indices.sort_unstable();
        for rows in [&fold.train_indices, &fold.valid_indices] {
            if rows.is_empty()
                || rows.last().is_some_and(|&r| r >= n)
                || rows.windows(2).any(|p| p[0] == p[1])
            {
                return invalid("fold indexes must be nonempty, unique, and in bounds");
            }
        }
        if fold
            .train_indices
            .iter()
            .any(|r| fold.valid_indices.binary_search(r).is_ok())
        {
            return invalid("training and validation rows must not overlap");
        }
        check_groups(data, &fold.train_indices)?;
        check_groups(data, &fold.valid_indices)?;
    }
    Ok(folds)
}
const MAX_ENTRIES: usize = 16 * 1024 * 1024;
const MAX_BYTES: usize = 256 * 1024 * 1024;
const ROW_FEATURE_BYTES: usize = 64;
const ROW_BYTES: usize = 128;
const FEATURE_FOLD_BYTES: usize = 1024;
const FOLD_BYTES: usize = 32 * 1024;
const SCORE_BYTES: usize = 64;
const MAX_WORKERS: usize = 256;

fn checked(count: Option<usize>) -> Result<usize> {
    count.ok_or_else(|| Error::InvalidInput("CV resource count overflow".into()))
}
fn product(a: usize, b: usize) -> Result<usize> {
    checked(a.checked_mul(b))
}
fn check_entries(count: usize, message: &str) -> Result<usize> {
    if count > MAX_ENTRIES {
        return invalid(message);
    }
    Ok(count)
}

// Read lengths only. In particular, do not clone or sort untrusted explicit
// indexes until every aggregate check has succeeded.
fn check_resources(data: &Dataset, strategy: &FoldStrategy, config: Option<&Config>) -> Result<()> {
    let (nfold, rows) = match strategy {
        FoldStrategy::Explicit(folds) => {
            if folds.is_empty() {
                return invalid("CV requires at least one fold");
            }
            let rows = folds.iter().try_fold(0_usize, |sum, fold| {
                checked(
                    sum.checked_add(checked(
                        fold.train_indices
                            .len()
                            .checked_add(fold.valid_indices.len()),
                    )?),
                )
            })?;
            (folds.len(), rows)
        }
        FoldStrategy::Standard { nfold, .. } | FoldStrategy::Stratified { nfold, .. } => {
            check_nfold(*nfold, data.nrows())?;
            (*nfold, product(*nfold, data.nrows())?)
        }
        FoldStrategy::Group { nfold } => {
            let groups = data
                .groups()
                .ok_or_else(|| Error::InvalidInput("group folds require query groups".into()))?;
            check_nfold(*nfold, groups.len())?;
            (*nfold, product(*nfold, data.nrows())?)
        }
    };
    check_entries(rows, "CV aggregate fold indexes exceed 16 Mi entries")?;
    let scores = if let Some(config) = config {
        // Each retained session owns its learner pool, even though folds step
        // sequentially. Serial sessions have no worker pool.
        if config.num_threads > 1 && product(nfold, config.num_threads)? > MAX_WORKERS {
            return invalid("CV aggregate fold pools exceed 256 workers");
        }
        check_entries(
            product(rows, config.num_class)?,
            "CV aggregate scores exceed 16 Mi entries",
        )?
    } else {
        0
    };
    let row_features = check_entries(
        product(rows, data.ncols())?,
        "CV aggregate row-feature pairs exceed 16 Mi entries",
    )?;
    let initial_scores = check_entries(
        product(
            rows,
            data.initial_scores().map_or(0, |s| s.len() / data.nrows()),
        )?,
        "CV aggregate initial margins exceed 16 Mi entries",
    )?;
    let bins = (0..data.ncols()).try_fold(0_usize, |sum, feature| {
        checked(sum.checked_add(data.mapper(feature).num_bins() as usize))
    })?;
    let terms = [
        product(row_features, ROW_FEATURE_BYTES)?,
        product(rows, ROW_BYTES)?,
        product(product(nfold, data.ncols())?, FEATURE_FOLD_BYTES)?,
        product(nfold, FOLD_BYTES)?,
        product(
            product(product(bins, nfold)?, 2)?,
            std::mem::size_of::<usize>(),
        )?,
        product(scores, SCORE_BYTES)?,
        product(initial_scores, std::mem::size_of::<f64>())?,
    ];
    let bytes = terms
        .into_iter()
        .try_fold(0_usize, |sum, term| checked(sum.checked_add(term)))?;
    if bytes > MAX_BYTES {
        return invalid("CV aggregate fold storage estimate exceeds 256 MiB");
    }
    Ok(())
}

// Indexes are already sorted, unique and in bounds. A selected query must
// start at its boundary and include all its consecutive rows. No subset or
// dataset-sized scratch buffer is needed to validate this condition.
fn check_groups(data: &Dataset, rows: &[usize]) -> Result<()> {
    if let Some(groups) = data.groups() {
        let mut start = 0;
        let mut selected = 0;
        for &len in groups {
            let end = start + len; // Dataset validates the checked group sum.
            if selected < rows.len() && rows[selected] < end {
                if rows[selected] != start
                    || rows.len() - selected < len
                    || rows[selected..selected + len]
                        .iter()
                        .enumerate()
                        .any(|(i, &row)| row != start + i)
                {
                    return invalid("ranking subsets must contain complete contiguous groups");
                }
                selected += len;
            }
            start = end;
        }
    }
    Ok(())
}

fn complement(n: usize, valid_indices: Vec<usize>) -> Fold {
    let set = valid_indices
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    Fold {
        train_indices: (0..n).filter(|r| !set.contains(r)).collect(),
        valid_indices,
    }
}
fn check_nfold(nfold: usize, n: usize) -> Result<()> {
    if nfold < 2 || nfold > n {
        return invalid("fold count must be in 2..=number of rows or groups");
    }
    Ok(())
}
fn invalid<T>(message: &str) -> Result<T> {
    Err(Error::InvalidInput(message.into()))
}

#[cfg(test)]
mod resource_tests {
    use super::*;

    #[test]
    fn resource_arithmetic_is_checked_without_allocations() {
        assert!(matches!(
            product(usize::MAX, 2),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            checked(usize::MAX.checked_add(1)),
            Err(Error::InvalidInput(_))
        ));
        assert_eq!(check_entries(MAX_ENTRIES, "limit").unwrap(), MAX_ENTRIES);
        assert!(matches!(
            check_entries(MAX_ENTRIES + 1, "limit"),
            Err(Error::InvalidInput(_))
        ));
    }
}
