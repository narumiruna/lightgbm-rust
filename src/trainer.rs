//! Serial GBDT, DART, and RF training with typed validation results.
//!
//! Tree layout and score updates follow LightGBM commit
//! 5d84f3f222b41409a6983d2ac90513ad9702fa9e,
//! src/boosting/gbdt.cpp and src/treelearner/serial_tree_learner.cpp (MIT).
//! Copyright (c) 2016-2026 Microsoft Corporation and The LightGBM developers.

use crate::boosting::DartState;
use crate::config::{Boosting, DataSampleStrategy};
use crate::learner::{FittedTree, TreeLearner, binned_leaf};
use crate::metric;
use crate::objective::{EPSILON, ObjectiveInput, ObjectiveState};
use crate::sampling::RowSampler;
use crate::workflow::{CallbackAction, CallbackEnv, FitOptions, PredictionKind};
use crate::{Config, Dataset, Error, Model, Objective, Result, Tree};
use std::collections::BTreeSet;

/// One named dataset metric after a completed iteration.
#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation {
    pub dataset: String,
    pub metric: String,
    pub value: f64,
    pub higher_is_better: bool,
}

/// Evaluation results use an iteration count starting at one, not a tree index.
#[derive(Clone, Debug, PartialEq)]
pub struct IterationEvaluation {
    pub iteration: usize,
    pub results: Vec<Evaluation>,
}

/// The immutable prediction model and completed training evidence.
#[derive(Clone, Debug)]
pub struct TrainResult {
    pub model: Model,
    pub evaluation_history: Vec<IterationEvaluation>,
    /// One-based selected iteration; zero means no stopping callback selected one.
    pub best_iteration: usize,
    pub best_scores: Vec<Evaluation>,
    /// Class-major training margins, including external dataset initial margins.
    /// RF uses training averages, while model raw predictions are tree sums.
    /// Query bagging follows native training-buffer score updates.
    pub training_scores: Vec<f64>,
}

/// A validated serial trainer. Datasets own their binning; the trainer owns no
/// borrowed mutable state. Failed fits cannot modify the input datasets.
#[derive(Clone, Debug)]
pub struct Trainer {
    config: Config,
}

/// Internal serial round driver shared by train and CV. No fold can run ahead.
pub(crate) struct TrainingSession<'a> {
    owned_config: Config,
    data: &'a Dataset,
    validation: Vec<(String, &'a Dataset)>,
    options: FitOptions<'a>,
    workflow: bool,
    objective: ObjectiveState<'a>,
    classes: usize,
    begin_iteration: usize,
    end_iteration: usize,
    scores: Vec<f64>,
    validation_scores: Vec<Vec<f64>>,
    history: Vec<IterationEvaluation>,
    best_iteration: usize,
    best_scores: Vec<Evaluation>,
    trees: Vec<Tree>,
    any_feature: bool,
    biases: Vec<f64>,
    rf: bool,
    learner: TreeLearner,
    sampler: RowSampler,
    // Native GOSS warm-up returns without changing the learner partition, even
    // after a reset reseeds sampling and sets its reported bag count to all rows.
    goss_learner_rows: Option<Vec<usize>>,
    dart: DartState,
    thresholds: Vec<Vec<u32>>,
    expected: usize,
    gradients: Vec<f32>,
    hessians: Vec<f32>,
    rf_scores: Vec<f64>,
    engine_iteration: usize,
    attempt: usize,
    rounds: usize,
}

impl TrainingSession<'_> {
    pub(crate) fn num_trees(&self) -> usize {
        self.begin_iteration * self.classes + self.trees.len()
    }
    pub(crate) fn history(&self) -> &[IterationEvaluation] {
        &self.history
    }
    pub(crate) fn reset_learning_rate(&mut self, rate: f64) -> Result<()> {
        set_learning_rate(
            &mut self.owned_config,
            &mut self.objective,
            &mut self.dart,
            &mut self.sampler,
            &mut self.learner,
            self.data,
            rate,
        )
    }
    pub(crate) fn step(&mut self) -> Result<bool> {
        for callback in self
            .options
            .callbacks
            .iter_mut()
            .filter(|c| c.before_iteration())
        {
            let env = CallbackEnv {
                config: &self.owned_config,
                iteration: self.begin_iteration + self.attempt,
                begin_iteration: self.begin_iteration,
                end_iteration: self.end_iteration,
                model_current_iteration: self.begin_iteration + self.trees.len() / self.classes,
                model_num_trees: (self.begin_iteration * self.classes) + self.trees.len(),
                best_iteration: self.best_iteration,
                evaluation: None,
                evaluation_history: &self.history,
                training_scores: &self.scores,
                training_name: &self.options.training_name,
            };
            match callback.call(&env)? {
                CallbackAction::Continue => {}
                CallbackAction::SetLearningRate(rate) => {
                    set_learning_rate(
                        &mut self.owned_config,
                        &mut self.objective,
                        &mut self.dart,
                        &mut self.sampler,
                        &mut self.learner,
                        self.data,
                        rate,
                    )?;
                }
                CallbackAction::Stop { .. } => {
                    return Err(Error::InvalidInput(
                        "stop actions require an after-iteration callback".into(),
                    ));
                }
            }
        }
        let config = &self.owned_config;
        let iteration = self.engine_iteration;
        let (dropped, shrinkage) = if config.boosting == Boosting::Dart {
            self.dart.select(iteration, config)
        } else {
            (Vec::new(), if self.rf { 1.0 } else { config.learning_rate })
        };
        for &round_index in &dropped {
            for class in 0..self.classes {
                let index = round_index * self.classes + class;
                let tree: &mut Tree = &mut self.trees[index];
                tree.scale_outputs(-1.0);
                update_scores(
                    tree,
                    &self.thresholds[index],
                    self.data,
                    &mut self.scores,
                    class,
                )?;
            }
        }
        if let Some(custom) = &mut self.options.custom_objective {
            (self.gradients, self.hessians) = custom.gradients(&self.scores, self.data)?;
            validate_derivatives(&self.gradients, &self.hessians, self.expected)?;
        } else if !self.rf {
            if config.bagging_by_query {
                self.sampler.sample(
                    iteration,
                    config,
                    self.data,
                    &mut self.gradients,
                    &mut self.hessians,
                )?;
                self.objective.gradients_for_queries(
                    &self.scores,
                    self.sampler.queries(),
                    &mut self.gradients,
                    &mut self.hessians,
                )?;
            } else {
                (self.gradients, self.hessians) = self.objective.gradients(&self.scores)?;
            }
        }
        if !config.bagging_by_query || self.rf || self.options.custom_objective.is_some() {
            self.sampler.sample(
                iteration,
                config,
                self.data,
                &mut self.gradients,
                &mut self.hessians,
            )?;
        }
        let goss_warmup = config.data_sample_strategy == DataSampleStrategy::Goss
            && iteration < (1.0 / config.learning_rate) as usize;
        if config.data_sample_strategy == DataSampleStrategy::Goss && !goss_warmup {
            self.goss_learner_rows = Some(self.sampler.rows().to_vec());
        }
        let learner_rows = self
            .goss_learner_rows
            .as_deref()
            .unwrap_or(self.sampler.rows());
        if self.gradients.len() != self.expected || self.hessians.len() != self.expected {
            return Err(Error::InvalidInput(
                "objective derivative shapes differ from scores".into(),
            ));
        }
        let mut round = Vec::new();
        let mut has_split = false;
        for class in 0..self.classes {
            let offset = class * self.data.nrows();
            let mut fitted = if self.objective.needs_training(class) && self.any_feature {
                let constant_hessian = config.data_sample_strategy != DataSampleStrategy::Goss
                    && (config.objective == Objective::Mape
                        || (self.data.weights().is_none()
                            && matches!(
                                config.objective,
                                Objective::Regression
                                    | Objective::RegressionL1
                                    | Objective::Quantile
                                    | Objective::Huber
                            )));
                // Native subset histograms use the first selected Hessian
                // as their constant, not the first original dataset row.
                let first_hessian = self.hessians[offset];
                if constant_hessian
                    && self.sampler.uses_subset()
                    && let Some(&first) = self.sampler.rows().first()
                {
                    self.hessians[offset] = self.hessians[offset + first];
                }
                let fitted = self.learner.fit(
                    self.data,
                    &self.gradients[offset..offset + self.data.nrows()],
                    &self.hessians[offset..offset + self.data.nrows()],
                    config,
                    learner_rows,
                    constant_hessian,
                );
                self.hessians[offset] = first_hessian;
                fitted?
            } else {
                FittedTree::constant(self.data.nrows())
            };
            if config.linear_tree && self.objective.needs_training(class) && self.any_feature {
                crate::linear::fit_linear_leaves(
                    &mut fitted.tree,
                    self.data,
                    &self.gradients[offset..offset + self.data.nrows()],
                    &self.hessians[offset..offset + self.data.nrows()],
                    &fitted.leaf_rows,
                    config,
                    !self.rf && iteration == 0 && self.begin_iteration == 0,
                )?;
            }
            has_split |= fitted.tree.num_leaves() > 1;
            round.push(fitted);
        }
        // Commit whole class rounds. RF always retains each round.
        if !self.rf && !has_split && (!self.trees.is_empty() || self.begin_iteration > 0) {
            // Python ignores the learner's finished flag and continues hooks.
            // GBDT returns before DART normalization when no class can split.
            if !self.workflow {
                return Ok(true);
            }
            round.clear();
        }
        for (class, mut fitted) in round.into_iter().enumerate() {
            let tree = &mut fitted.tree;
            if tree.num_leaves() > 1 {
                if matches!(
                    config.objective,
                    Objective::RegressionL1 | Objective::Quantile | Objective::Mape
                ) {
                    for (leaf, rows) in fitted.leaf_rows.iter().enumerate() {
                        tree.leaf_values[leaf] = self.objective.renew_leaf_output(
                            class,
                            rows,
                            if self.rf {
                                &self.rf_scores
                            } else {
                                &self.scores
                            },
                            tree.leaf_values[leaf],
                        )?;
                    }
                }
                tree.scale_outputs(shrinkage);
                if self.rf && self.biases[class].abs() > EPSILON {
                    tree.add_bias(self.biases[class]);
                }
                if self.rf {
                    multiply_class(&mut self.scores, self.data.nrows(), class, iteration as f64);
                    for ((_, valid), s) in self.validation.iter().zip(&mut self.validation_scores) {
                        multiply_class(s, valid.nrows(), class, iteration as f64);
                    }
                }
                let all_rows;
                let score_rows = if goss_warmup && self.goss_learner_rows.is_some() {
                    learner_rows
                } else if let Some(rows) = self.sampler.query_score_rows() {
                    rows
                } else {
                    all_rows = (0..self.data.nrows()).collect::<Vec<_>>();
                    &all_rows
                };
                for &row in score_rows {
                    let leaf = fitted.leaf_indices[row];
                    let leaf = if leaf == usize::MAX {
                        binned_leaf(tree, &fitted.threshold_bins, self.data, row)
                    } else {
                        leaf
                    };
                    self.scores[class * self.data.nrows() + row] +=
                        crate::linear::predict_from_leaf(tree, self.data, row, leaf)?;
                }
                for ((_, valid), valid_scores) in
                    self.validation.iter().zip(&mut self.validation_scores)
                {
                    update_scores(tree, &fitted.threshold_bins, valid, valid_scores, class)?;
                }
                if self.rf {
                    multiply_class(
                        &mut self.scores,
                        self.data.nrows(),
                        class,
                        1.0 / (iteration + 1) as f64,
                    );
                    for ((_, valid), s) in self.validation.iter().zip(&mut self.validation_scores) {
                        multiply_class(s, valid.nrows(), class, 1.0 / (iteration + 1) as f64);
                    }
                } else if iteration == 0 && self.biases[class].abs() > EPSILON {
                    tree.add_bias(self.biases[class]);
                }
            } else {
                let mut bias = if iteration == 0 {
                    self.biases[class]
                } else {
                    0.0
                };
                if self.rf {
                    bias = if iteration == 0 && !self.objective.needs_training(class) {
                        self.objective.initial_score(class)?
                    } else {
                        0.0
                    };
                    if iteration == 0 {
                        add_class_bias(&mut self.scores, self.data.nrows(), class, bias);
                        for ((_, valid), s) in
                            self.validation.iter().zip(&mut self.validation_scores)
                        {
                            add_class_bias(s, valid.nrows(), class, bias);
                        }
                    }
                } else if iteration == 0
                    && !config.boost_from_average
                    && self.data.initial_scores().is_none()
                    && config.objective != Objective::Custom
                {
                    bias = self.objective.initial_score(class)?;
                    add_class_bias(&mut self.scores, self.data.nrows(), class, bias);
                    for ((_, valid), s) in self.validation.iter().zip(&mut self.validation_scores) {
                        add_class_bias(s, valid.nrows(), class, bias);
                    }
                }
                if self.rf && iteration > 0 {
                    // RF retains an unsplit learner root without updating
                    // scores; an absent class keeps its empty default tree.
                    tree.leaf_counts[0] = 0;
                } else {
                    tree.as_constant(bias, self.data.nrows());
                }
            }
            if tree
                .leaf_values
                .iter()
                .chain(&tree.internal_values)
                .any(|v| !v.is_finite())
            {
                return Err(Error::InvalidInput(
                    "training produced a nonfinite tree".into(),
                ));
            }
            self.thresholds.push(fitted.threshold_bins);
            self.trees.push(fitted.tree);
        }
        if !self.rf {
            self.sampler.retain_query_subset_cache(
                self.data.nrows(),
                &mut self.gradients,
                &mut self.hessians,
            );
        }
        if config.boosting == Boosting::Dart && has_split {
            let k = dropped.len() as f64;
            for &round_index in &dropped {
                for class in 0..self.classes {
                    let index = round_index * self.classes + class;
                    let tree = &mut self.trees[index];
                    tree.scale_outputs(if config.xgboost_dart_mode {
                        shrinkage
                    } else {
                        1.0 / (k + 1.0)
                    });
                    for ((_, valid), s) in self.validation.iter().zip(&mut self.validation_scores) {
                        update_scores(tree, &self.thresholds[index], valid, s, class)?;
                    }
                    tree.scale_outputs(if config.xgboost_dart_mode {
                        -k / config.learning_rate
                    } else {
                        -k
                    });
                    update_scores(
                        tree,
                        &self.thresholds[index],
                        self.data,
                        &mut self.scores,
                        class,
                    )?;
                }
            }
            self.dart.finish(&dropped, shrinkage, config);
        }
        if has_split || self.rf {
            self.engine_iteration += 1;
        }
        let mut results = Vec::new();
        if self.options.evaluate_training {
            append_metrics(
                &mut results,
                &self.options.training_name,
                self.data,
                &self.scores,
                config,
            )?;
            append_custom_metrics(
                &mut results,
                &self.options.training_name,
                self.data,
                &self.scores,
                config,
                &mut self.options.custom_metrics,
            )?;
        }
        for ((name, valid), valid_scores) in self.validation.iter().zip(&self.validation_scores) {
            append_metrics(&mut results, name, valid, valid_scores, config)?;
            append_custom_metrics(
                &mut results,
                name,
                valid,
                valid_scores,
                config,
                &mut self.options.custom_metrics,
            )?;
        }
        let entry = IterationEvaluation {
            iteration: self.begin_iteration + self.attempt + 1,
            results: results.clone(),
        };
        let mut recorded = false;
        let mut stopped = false;
        self.best_scores = results.clone();
        for callback in self
            .options
            .callbacks
            .iter_mut()
            .filter(|c| !c.before_iteration())
        {
            if !recorded && callback.order() >= 20 {
                self.history.push(entry.clone());
                recorded = true;
            }
            let env = CallbackEnv {
                config: &self.owned_config,
                iteration: self.begin_iteration + self.attempt,
                begin_iteration: self.begin_iteration,
                end_iteration: self.end_iteration,
                model_current_iteration: self.begin_iteration + self.trees.len() / self.classes,
                model_num_trees: (self.begin_iteration * self.classes) + self.trees.len(),
                best_iteration: self.best_iteration,
                evaluation: Some(&results),
                evaluation_history: &self.history,
                training_scores: &self.scores,
                training_name: &self.options.training_name,
            };
            match callback.call(&env)? {
                CallbackAction::Continue => {}
                CallbackAction::SetLearningRate(rate) => set_learning_rate(
                    &mut self.owned_config,
                    &mut self.objective,
                    &mut self.dart,
                    &mut self.sampler,
                    &mut self.learner,
                    self.data,
                    rate,
                )?,
                CallbackAction::Stop {
                    best_iteration: selected,
                    best_scores: selected_scores,
                } => {
                    if selected <= self.begin_iteration
                        || selected > self.begin_iteration + self.attempt + 1
                    {
                        return Err(Error::InvalidInput(
                            "invalid callback best iteration".into(),
                        ));
                    }
                    self.best_iteration = selected;
                    self.best_scores = selected_scores;
                    stopped = true;
                    break;
                }
            }
        }
        if !recorded {
            self.history.push(entry);
        }
        self.attempt += 1;
        Ok(stopped || (!self.workflow && !self.rf && !has_split) || self.attempt >= self.rounds)
    }

    pub(crate) fn finish(self) -> Result<TrainResult> {
        let trees = if let Some(initial) = self.options.initial_model {
            let mut combined = initial.trees().to_vec();
            combined.extend(self.trees);
            combined
        } else {
            self.trees
        };
        let model = Model::from_parts(
            trees,
            self.classes,
            self.classes,
            self.data.feature_names().to_vec(),
            (0..self.data.ncols())
                .map(|f| {
                    if self.data.mapper(f).is_trivial() {
                        "none".into()
                    } else {
                        self.data.mapper(f).feature_info()
                    }
                })
                .collect(),
            self.objective.model_objective(),
            self.rf,
        )?;
        Ok(TrainResult {
            model,
            evaluation_history: self.history,
            training_scores: self.scores,
            best_iteration: self.best_iteration,
            best_scores: self.best_scores,
        })
    }
}

impl Trainer {
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        if config.num_iterations == 0 {
            return Err(Error::InvalidInput(
                "training requires at least one iteration".into(),
            ));
        }
        Ok(Self { config })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn fit(&self, data: &Dataset) -> Result<TrainResult> {
        self.fit_with_validation(data, &[])
    }

    /// Validation data must share the training bin maps. Use
    /// `Dataset::validation_from_dense` or a subset of the same dataset.
    pub fn fit_with_validation(
        &self,
        data: &Dataset,
        validation: &[(&str, &Dataset)],
    ) -> Result<TrainResult> {
        self.fit_internal(
            data,
            validation,
            FitOptions {
                evaluate_training: true,
                ..FitOptions::default()
            },
            false,
        )
    }

    /// Train with Python engine round-attempt and callback semantics.
    pub fn fit_with_options(
        &self,
        data: &Dataset,
        validation: &[(&str, &Dataset)],
        mut options: FitOptions<'_>,
    ) -> Result<TrainResult> {
        let mut reduced = Vec::with_capacity(validation.len());
        for &(name, valid) in validation {
            if std::ptr::eq(data, valid) {
                options.evaluate_training = true;
                options.training_name = name.to_owned();
            } else {
                reduced.push((name, valid));
            }
        }
        self.fit_internal(data, &reduced, options, true)
    }

    fn fit_internal<'a>(
        &self,
        data: &'a Dataset,
        validation: &[(&str, &'a Dataset)],
        options: FitOptions<'a>,
        workflow: bool,
    ) -> Result<TrainResult> {
        let mut session = self.start_session(data, validation, options, workflow)?;
        while !session.step()? {}
        session.finish()
    }

    pub(crate) fn start_session<'a>(
        &self,
        data: &'a Dataset,
        validation: &[(&str, &'a Dataset)],
        mut options: FitOptions<'a>,
        workflow: bool,
    ) -> Result<TrainingSession<'a>> {
        let owned_config = self.config.clone();
        let config = &owned_config;
        if (config.objective == Objective::Custom) != options.custom_objective.is_some() {
            return Err(Error::Unsupported(
                "custom objective training requires a gradient callback".into(),
            ));
        }
        data.check_binning_config(config)?;
        if data
            .prefilter_min_data_in_leaf()
            .is_some_and(|minimum| config.min_data_in_leaf < minimum)
        {
            return Err(Error::InvalidInput(
                "rebuild the dataset before reducing min_data_in_leaf with feature_pre_filter"
                    .into(),
            ));
        }
        let mut names = BTreeSet::new();
        if options.training_name.is_empty() {
            return Err(Error::InvalidInput(
                "training name must not be empty".into(),
            ));
        }
        names.insert(options.training_name.as_str());
        for &(name, valid) in validation {
            if name.is_empty() || !names.insert(name) {
                return Err(Error::InvalidInput(
                    "validation names must be unique, nonempty, and not training".into(),
                ));
            }
            if data.ncols() != valid.ncols()
                || (0..data.ncols()).any(|f| !std::ptr::eq(data.mapper(f), valid.mapper(f)))
            {
                return Err(Error::InvalidInput(
                    "validation datasets must share training bin maps".into(),
                ));
            }
        }
        // Check every layout before allocating row-by-class score buffers.
        checked_score_len(data.nrows(), config.num_class)?;
        for &(_, valid) in validation {
            checked_score_len(valid.nrows(), config.num_class)?;
        }
        let mut objective = ObjectiveState::new(
            config,
            ObjectiveInput {
                labels: data.labels(),
                weights: data.weights(),
                groups: data.groups(),
                positions: data.positions(),
            },
        )?;
        let classes = objective.num_classes();
        let begin_iteration = options.initial_model.map_or(0, Model::num_iterations);
        if let Some(initial) = options.initial_model {
            if config.boosting == Boosting::Rf {
                return Err(Error::InvalidInput(
                    "RF does not support model continuation initial margins".into(),
                ));
            }
            if initial.num_features() != data.ncols()
                || initial.feature_names() != data.feature_names()
                || initial.num_classes() != classes
                || initial.num_tree_per_iteration() != classes
                || initial.objective() != objective.model_objective()
                || initial.average_output()
            {
                return Err(Error::InvalidInput(
                    "initial model layout, features, or objective differ from training".into(),
                ));
            }
            if data.initial_scores().is_none()
                || validation.iter().any(|(_, d)| d.initial_scores().is_none())
            {
                return Err(Error::InvalidInput(
                    "continuation requires raw initial model margins for every dataset".into(),
                ));
            }
        }
        let end_iteration = begin_iteration
            .checked_add(config.num_iterations)
            .ok_or_else(|| Error::InvalidInput("continuation iteration count overflow".into()))?;
        let mut scores = initial_scores(data, classes)?;
        let mut validation_scores: Vec<Vec<f64>> = validation
            .iter()
            .map(|(_, d)| initial_scores(d, classes))
            .collect::<Result<_>>()?;
        let history = Vec::new();
        let best_iteration = 0;
        let best_scores = Vec::new();
        options
            .callbacks
            .sort_by_key(|c| (!c.before_iteration(), c.order()));
        let trees = Vec::new();
        let any_feature = (0..data.ncols()).any(|f| !data.mapper(f).is_trivial());
        let mut biases = vec![0.0; classes];
        let rf = config.boosting == Boosting::Rf;
        if rf && data.initial_scores().is_some() {
            return Err(Error::InvalidInput(
                "RF does not support dataset initial margins".into(),
            ));
        }
        if config.objective != Objective::Custom
            && data.initial_scores().is_none()
            && (config.boost_from_average || !any_feature)
        {
            for (class, bias) in biases.iter_mut().enumerate() {
                let automatic = objective.initial_score(class)?;
                if automatic.abs() > EPSILON {
                    *bias = automatic;
                    if !rf {
                        add_class_bias(&mut scores, data.nrows(), class, automatic);
                        for ((_, valid), s) in validation.iter().zip(&mut validation_scores) {
                            add_class_bias(s, valid.nrows(), class, automatic);
                        }
                    }
                }
            }
        }
        let learner = TreeLearner::new(config, data)?;
        let sampler = RowSampler::new(config, data)?;
        let dart = DartState::new(config);
        let thresholds: Vec<Vec<u32>> = Vec::new();
        let expected = checked_score_len(data.nrows(), classes)?;
        let mut gradients = vec![0.0_f32; expected];
        let mut hessians = vec![0.0_f32; expected];
        let mut rf_scores = vec![0.0; expected];
        if rf {
            for (class, &bias) in biases.iter().enumerate() {
                add_class_bias(&mut rf_scores, data.nrows(), class, bias);
            }
            (gradients, hessians) = objective.gradients(&rf_scores)?;
        }
        let engine_iteration = 0;
        Ok(TrainingSession {
            owned_config,
            data,
            validation: validation
                .iter()
                .map(|(name, data)| (name.to_string(), *data))
                .collect(),
            options,
            workflow,
            objective,
            classes,
            begin_iteration,
            end_iteration,
            scores,
            validation_scores,
            history,
            best_iteration,
            best_scores,
            trees,
            any_feature,
            biases,
            rf,
            learner,
            sampler,
            goss_learner_rows: None,
            dart,
            thresholds,
            expected,
            gradients,
            hessians,
            rf_scores,
            engine_iteration,
            attempt: 0,
            rounds: self.config.num_iterations,
        })
    }
}

fn set_learning_rate(
    config: &mut Config,
    objective: &mut ObjectiveState<'_>,
    dart: &mut DartState,
    sampler: &mut RowSampler,
    learner: &mut TreeLearner,
    data: &Dataset,
    rate: f64,
) -> Result<()> {
    if !rate.is_finite() || rate <= 0.0 {
        return Err(Error::InvalidInput(
            "learning rate must be finite and positive".into(),
        ));
    }
    if config.learning_rate == rate {
        return Ok(());
    }
    config.learning_rate = rate;
    objective.reset_learning_rate(rate);
    learner.reset_learning_rate_config(config)?;
    if config.boosting == Boosting::Dart {
        dart.reset_config(config);
    }
    // GOSS ResetSampleConfig recreates its random streams on every reset;
    // ordinary bagging returns early when its sampling parameters did not change.
    if config.data_sample_strategy == DataSampleStrategy::Goss {
        *sampler = RowSampler::new(config, data)?;
    }
    Ok(())
}

fn validate_derivatives(g: &[f32], h: &[f32], expected: usize) -> Result<()> {
    if g.len() != expected || h.len() != expected {
        return Err(Error::InvalidInput(
            "custom derivative shapes differ from class-major scores".into(),
        ));
    }
    if g.iter().any(|v| !v.is_finite()) || h.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(Error::InvalidInput(
            "custom derivatives must be finite with nonnegative Hessians".into(),
        ));
    }
    Ok(())
}

fn append_custom_metrics(
    results: &mut Vec<Evaluation>,
    name: &str,
    data: &Dataset,
    scores: &[f64],
    config: &Config,
    metrics: &mut [Box<dyn crate::workflow::CustomMetric + '_>],
) -> Result<()> {
    if metrics.is_empty() {
        return Ok(());
    }
    let kind = if config.objective == Objective::Custom {
        PredictionKind::Raw
    } else {
        PredictionKind::Transformed
    };
    let mut predictions = scores.to_vec();
    if kind == PredictionKind::Transformed {
        if config.objective == Objective::Multiclass {
            let mut row = vec![0.0; config.num_class];
            for i in 0..data.nrows() {
                for c in 0..config.num_class {
                    row[c] = scores[c * data.nrows() + i];
                }
                crate::objective::softmax(&mut row);
                for c in 0..config.num_class {
                    predictions[c * data.nrows() + i] = row[c];
                }
            }
        } else {
            for p in &mut predictions {
                *p = match config.objective {
                    Objective::Binary | Objective::MulticlassOva => {
                        1.0 / (1.0 + (-config.sigmoid * *p).exp())
                    }
                    Objective::CrossEntropy => 1.0 / (1.0 + (-*p).exp()),
                    Objective::CrossEntropyLambda => p.exp().ln_1p(),
                    Objective::Poisson | Objective::Gamma | Objective::Tweedie => p.exp(),
                    _ if config.reg_sqrt => p.signum() * *p * *p,
                    _ => *p,
                };
            }
        }
    }
    for metric in metrics {
        for value in metric.evaluate(&predictions, data, kind)? {
            if value.name.is_empty() || !value.value.is_finite() {
                return Err(Error::InvalidInput(
                    "custom metric name must be nonempty and value finite".into(),
                ));
            }
            results.push(Evaluation {
                dataset: name.into(),
                metric: value.name,
                value: value.value,
                higher_is_better: value.higher_is_better,
            });
        }
    }
    Ok(())
}

fn append_metrics(
    results: &mut Vec<Evaluation>,
    name: &str,
    data: &Dataset,
    scores: &[f64],
    config: &Config,
) -> Result<()> {
    for m in metric::evaluate_dataset(config, data, scores)? {
        results.push(Evaluation {
            dataset: name.to_owned(),
            metric: m.name,
            value: m.value,
            higher_is_better: m.higher_is_better,
        });
    }
    Ok(())
}

fn update_scores(
    tree: &Tree,
    thresholds: &[u32],
    data: &Dataset,
    scores: &mut [f64],
    class: usize,
) -> Result<()> {
    for (row, score) in scores[class * data.nrows()..(class + 1) * data.nrows()]
        .iter_mut()
        .enumerate()
    {
        let leaf = binned_leaf(tree, thresholds, data, row);
        *score += crate::linear::predict_from_leaf(tree, data, row, leaf)?;
    }
    Ok(())
}

fn multiply_class(scores: &mut [f64], rows: usize, class: usize, factor: f64) {
    for score in &mut scores[class * rows..(class + 1) * rows] {
        *score *= factor;
    }
}

fn checked_score_len(rows: usize, classes: usize) -> Result<usize> {
    rows.checked_mul(classes)
        .filter(|&len| classes > 0 && len <= 16 * 1024 * 1024)
        .ok_or_else(|| {
            Error::InvalidInput(
                "training scores exceed 16 Mi elements or have invalid dimensions".into(),
            )
        })
}

fn initial_scores(data: &Dataset, classes: usize) -> Result<Vec<f64>> {
    let len = checked_score_len(data.nrows(), classes)?;
    if data
        .initial_scores()
        .is_some_and(|s| s.len() != len || s.iter().any(|v| !v.is_finite()))
    {
        return Err(Error::InvalidInput(
            "initial score class layout differs from objective".into(),
        ));
    }
    let mut scores = Vec::new();
    scores.try_reserve_exact(len).map_err(allocation_error)?;
    if let Some(initial) = data.initial_scores() {
        scores.extend_from_slice(initial);
    } else {
        scores.resize(len, 0.0);
    }
    Ok(scores)
}

fn add_class_bias(scores: &mut [f64], rows: usize, class: usize, bias: f64) {
    for score in &mut scores[class * rows..(class + 1) * rows] {
        *score += bias;
    }
}

fn allocation_error(error: std::collections::TryReserveError) -> Error {
    Error::InvalidInput(format!("training allocation failed: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checked_score_dimensions() {
        assert!(checked_score_len(usize::MAX, 2).is_err());
        assert!(checked_score_len(1, 0).is_err());
        assert!(checked_score_len(257, 65_536).is_err());
        assert_eq!(checked_score_len(256, 65_536).unwrap(), 16 * 1024 * 1024);
    }
}
