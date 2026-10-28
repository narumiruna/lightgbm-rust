// Defaults, aliases, precedence, and seed derivation follow LightGBM (MIT).
// Copyright (c) 2016-2026 Microsoft Corporation. All rights reserved.
// Copyright (c) 2016-2026 The LightGBM developers. All rights reserved.
// See THIRD_PARTY_NOTICES.md for the upstream license.

//! Typed configuration for CPU objectives, metrics, datasets, and ensembles.
//!
//! Defaults and aliases follow LightGBM commit
//! `5d84f3f222b41409a6983d2ac90513ad9702fa9e`, except `verbosity = -1`.
//! Unsupported options produce errors, even when supplied with neutral values.
//! Unlike upstream, duplicate keys and two zero leaf minimums are errors rather
//! than warnings or automatic corrections. Class weights require binary or OVA
//! objectives. Nondefault sigmoid requires binary, OVA, or LambdaRank.
//! All floating-point options must be finite.

use crate::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Objective {
    Regression,
    RegressionL1,
    Quantile,
    Huber,
    Fair,
    Poisson,
    Mape,
    Gamma,
    Tweedie,
    Binary,
    Multiclass,
    MulticlassOva,
    CrossEntropy,
    CrossEntropyLambda,
    LambdaRank,
    RankXendcg,
    /// Requires a custom gradient callback; no built-in kernel is substituted.
    Custom,
}

impl Objective {
    pub(crate) fn is_multiclass(self) -> bool {
        matches!(self, Self::Multiclass | Self::MulticlassOva)
    }

    pub(crate) fn is_ranking(self) -> bool {
        matches!(self, Self::LambdaRank | Self::RankXendcg)
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Regression => "regression",
            Self::RegressionL1 => "regression_l1",
            Self::Quantile => "quantile",
            Self::Huber => "huber",
            Self::Fair => "fair",
            Self::Poisson => "poisson",
            Self::Mape => "mape",
            Self::Gamma => "gamma",
            Self::Tweedie => "tweedie",
            Self::Binary => "binary",
            Self::Multiclass => "multiclass",
            Self::MulticlassOva => "multiclassova",
            Self::CrossEntropy => "cross_entropy",
            Self::CrossEntropyLambda => "cross_entropy_lambda",
            Self::LambdaRank => "lambdarank",
            Self::RankXendcg => "rank_xendcg",
            Self::Custom => "custom",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boosting {
    Gbdt,
    Dart,
    Rf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataSampleStrategy {
    Bagging,
    Goss,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonotoneMethod {
    Basic,
    Intermediate,
    Advanced,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub boosting: Boosting,
    pub data_sample_strategy: DataSampleStrategy,
    pub bagging_fraction: f64,
    pub pos_bagging_fraction: f64,
    pub neg_bagging_fraction: f64,
    pub bagging_freq: i32,
    pub bagging_seed: u32,
    pub bagging_by_query: bool,
    pub feature_fraction: f64,
    pub feature_fraction_bynode: f64,
    pub feature_fraction_seed: u32,
    pub extra_trees: bool,
    pub extra_seed: u32,
    pub top_rate: f64,
    pub other_rate: f64,
    pub drop_rate: f64,
    pub max_drop: i32,
    pub skip_drop: f64,
    pub xgboost_dart_mode: bool,
    pub uniform_drop: bool,
    pub drop_seed: u32,
    pub monotone_constraints: Vec<i8>,
    pub monotone_constraints_method: MonotoneMethod,
    pub monotone_penalty: f64,
    pub interaction_constraints: Vec<Vec<usize>>,
    pub forced_splits: Option<serde_json::Value>,
    pub path_smooth: f64,
    pub feature_contri: Vec<f64>,
    pub cegb_tradeoff: f64,
    pub cegb_penalty_split: f64,
    pub cegb_penalty_feature_coupled: Vec<f64>,
    pub cegb_penalty_feature_lazy: Vec<f64>,
    pub use_quantized_grad: bool,
    pub num_grad_quant_bins: usize,
    pub quant_train_renew_leaf: bool,
    pub stochastic_rounding: bool,
    pub objective: Objective,
    pub num_class: usize,
    pub reg_sqrt: bool,
    pub alpha: f64,
    pub fair_c: f64,
    pub poisson_max_delta_step: f64,
    pub tweedie_variance_power: f64,
    pub objective_seed: u32,
    pub lambdarank_truncation_level: usize,
    pub lambdarank_norm: bool,
    /// Empty uses gains `2^label - 1` for labels 0 through 30.
    pub label_gain: Vec<f64>,
    pub lambdarank_position_bias_regularization: f64,
    /// Empty uses cutoffs 1 through 5. Cutoffs are evaluated in ascending order.
    pub eval_at: Vec<usize>,
    pub multi_error_top_k: usize,
    /// Row-major class-by-class matrix. Empty uses unit off-diagonal weights.
    pub auc_mu_weights: Vec<f64>,
    pub num_iterations: usize,
    pub learning_rate: f64,
    pub num_leaves: usize,
    /// Nonpositive values mean unlimited depth. Growth remains leaf-wise.
    pub max_depth: i32,
    pub min_data_in_leaf: usize,
    pub min_sum_hessian_in_leaf: f64,
    pub lambda_l1: f64,
    pub lambda_l2: f64,
    pub min_gain_to_split: f64,
    /// Nonpositive values disable the leaf-output limit.
    pub max_delta_step: f64,
    pub max_bin: usize,
    /// Empty uses `max_bin` for every feature.
    pub max_bin_by_feature: Vec<usize>,
    /// Typed API only. When set, one list per column; empty lists are unforced.
    pub forced_bins: Vec<Vec<f64>>,
    /// Retain raw numeric values. Training requires a separate linear solver.
    pub linear_tree: bool,
    /// L2 penalty for linear-leaf coefficients; not a dataset binning option.
    pub linear_lambda: f64,
    pub max_cat_threshold: usize,
    pub max_cat_to_onehot: usize,
    pub cat_smooth: f64,
    pub cat_l2: f64,
    pub min_data_per_group: usize,
    pub min_data_in_bin: usize,
    pub bin_construct_sample_cnt: usize,
    pub data_random_seed: u32,
    /// `from_pairs` derives data, bagging, dropout, feature, objective, and extra
    /// seeds when supplied explicitly. Explicit individual seeds take priority.
    /// Direct field assignment does not derive other seeds.
    pub seed: u32,
    /// Maximum worker count. Zero selects one worker (no global Rayon pool).
    /// Explicit counts above 256 are rejected before pool allocation.
    pub num_threads: usize,
    /// Fixed per-feature sums and ordered split reduction, including across threads.
    pub deterministic: bool,
    pub force_col_wise: bool,
    pub force_row_wise: bool,
    /// Retained histogram budget in MiB. Nonpositive uses a 128 MiB safety ceiling.
    /// Working histograms are separate and checked before allocation.
    pub histogram_pool_size: f64,
    /// Enable exclusive feature bundling during dataset construction.
    pub enable_bundle: bool,
    pub zero_as_missing: bool,
    pub use_missing: bool,
    pub feature_pre_filter: bool,
    pub boost_from_average: bool,
    pub sigmoid: f64,
    pub is_unbalance: bool,
    pub scale_pos_weight: f64,
    /// Distinct zero-based column indexes for categorical binning and splits.
    pub categorical_features: Vec<usize>,
    /// Empty selects the objective default. `custom` disables built-in metrics.
    /// Direct assignments require canonical CPU metric names.
    /// `from_pairs` normalizes supported aliases and removes repeated metrics.
    pub metrics: Vec<String>,
    pub verbosity: i32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            boosting: Boosting::Gbdt,
            data_sample_strategy: DataSampleStrategy::Bagging,
            bagging_fraction: 1.0,
            pos_bagging_fraction: 1.0,
            neg_bagging_fraction: 1.0,
            bagging_freq: 0,
            bagging_seed: 3,
            bagging_by_query: false,
            feature_fraction: 1.0,
            feature_fraction_bynode: 1.0,
            feature_fraction_seed: 2,
            extra_trees: false,
            extra_seed: 6,
            top_rate: 0.2,
            other_rate: 0.1,
            drop_rate: 0.1,
            max_drop: 50,
            skip_drop: 0.5,
            xgboost_dart_mode: false,
            uniform_drop: false,
            drop_seed: 4,
            monotone_constraints: Vec::new(),
            monotone_constraints_method: MonotoneMethod::Basic,
            monotone_penalty: 0.0,
            interaction_constraints: Vec::new(),
            forced_splits: None,
            path_smooth: 0.0,
            feature_contri: Vec::new(),
            cegb_tradeoff: 1.0,
            cegb_penalty_split: 0.0,
            cegb_penalty_feature_coupled: Vec::new(),
            cegb_penalty_feature_lazy: Vec::new(),
            use_quantized_grad: false,
            num_grad_quant_bins: 4,
            quant_train_renew_leaf: false,
            stochastic_rounding: true,
            objective: Objective::Regression,
            num_class: 1,
            reg_sqrt: false,
            alpha: 0.9,
            fair_c: 1.0,
            poisson_max_delta_step: 0.7,
            tweedie_variance_power: 1.5,
            objective_seed: 5,
            lambdarank_truncation_level: 30,
            lambdarank_norm: true,
            label_gain: Vec::new(),
            lambdarank_position_bias_regularization: 0.0,
            eval_at: Vec::new(),
            multi_error_top_k: 1,
            auc_mu_weights: Vec::new(),
            num_iterations: 100,
            learning_rate: 0.1,
            num_leaves: 31,
            max_depth: -1,
            min_data_in_leaf: 20,
            min_sum_hessian_in_leaf: 1e-3,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
            max_delta_step: 0.0,
            max_bin: 255,
            max_bin_by_feature: Vec::new(),
            forced_bins: Vec::new(),
            linear_tree: false,
            linear_lambda: 0.0,
            max_cat_threshold: 32,
            max_cat_to_onehot: 4,
            cat_smooth: 10.0,
            cat_l2: 10.0,
            min_data_per_group: 100,
            min_data_in_bin: 3,
            bin_construct_sample_cnt: 200_000,
            data_random_seed: 1,
            seed: 0,
            num_threads: 0,
            deterministic: false,
            force_col_wise: false,
            force_row_wise: false,
            histogram_pool_size: -1.0,
            enable_bundle: true,
            zero_as_missing: false,
            use_missing: true,
            feature_pre_filter: true,
            boost_from_average: true,
            sigmoid: 1.0,
            is_unbalance: false,
            scale_pos_weight: 1.0,
            categorical_features: Vec::new(),
            metrics: Vec::new(),
            verbosity: -1,
        }
    }
}

impl Config {
    /// Check domains and the supported CPU subset without changing any fields.
    /// Integer counts use the upstream signed 32-bit domain; seeds use `u32`.
    /// Consumers must also check data-dependent sizes before allocating memory.
    pub fn validate(&self) -> Result<()> {
        for (name, value, positive) in [
            ("bagging_fraction", self.bagging_fraction, true),
            ("pos_bagging_fraction", self.pos_bagging_fraction, true),
            ("neg_bagging_fraction", self.neg_bagging_fraction, true),
            ("feature_fraction", self.feature_fraction, true),
            (
                "feature_fraction_bynode",
                self.feature_fraction_bynode,
                true,
            ),
            ("top_rate", self.top_rate, false),
            ("other_rate", self.other_rate, false),
            ("drop_rate", self.drop_rate, false),
            ("skip_drop", self.skip_drop, false),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) || (positive && value == 0.0) {
                return Err(invalid(
                    name,
                    "must be finite and in the source fraction domain",
                ));
            }
        }
        for (name, value) in [
            ("monotone_penalty", self.monotone_penalty),
            ("path_smooth", self.path_smooth),
            ("cegb_tradeoff", self.cegb_tradeoff),
            ("cegb_penalty_split", self.cegb_penalty_split),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(invalid(name, "must be finite and nonnegative"));
            }
        }
        for (name, values) in [
            ("feature_contri", &self.feature_contri),
            ("cegb_penalty_feature_lazy", &self.cegb_penalty_feature_lazy),
            (
                "cegb_penalty_feature_coupled",
                &self.cegb_penalty_feature_coupled,
            ),
        ] {
            if values.iter().any(|v| !v.is_finite()) {
                return Err(invalid(name, "entries must be finite"));
            }
        }
        if self
            .monotone_constraints
            .iter()
            .any(|&v| !(-1..=1).contains(&v))
        {
            return Err(invalid(
                "monotone_constraints",
                "entries must be -1, 0, or 1",
            ));
        }
        if self.path_smooth > 1e-15_f32 as f64 && self.min_data_in_leaf < 2 {
            return Err(invalid(
                "min_data_in_leaf",
                "path smoothing requires at least two rows",
            ));
        }
        if !self.monotone_constraints.is_empty()
            && matches!(
                self.objective,
                Objective::RegressionL1 | Objective::Quantile | Objective::Mape
            )
        {
            return Err(invalid(
                "monotone_constraints",
                "not supported with objective-specific leaf renewal",
            ));
        }
        if self
            .interaction_constraints
            .iter()
            .flatten()
            .any(|&f| f > i32::MAX as usize)
        {
            return Err(invalid(
                "interaction_constraints",
                "feature indexes must fit signed 32-bit integers",
            ));
        }
        if self.use_quantized_grad && !(2..=127).contains(&self.num_grad_quant_bins) {
            return Err(invalid(
                "num_grad_quant_bins",
                "quantization requires 2..=127 bins",
            ));
        }
        if self.data_sample_strategy == DataSampleStrategy::Goss {
            if self.top_rate <= 0.0
                || self.other_rate <= 0.0
                || self.top_rate + self.other_rate > 1.0
            {
                return Err(invalid(
                    "top_rate",
                    "GOSS requires positive rates with sum at most 1",
                ));
            }
            if self.bagging_freq > 0 && self.bagging_fraction != 1.0 {
                return Err(invalid("bagging_fraction", "cannot use bagging in GOSS"));
            }
            if self.bagging_by_query {
                return Err(invalid(
                    "bagging_by_query",
                    "requires bagging sample strategy",
                ));
            }
        }
        if self.boosting == Boosting::Rf {
            if self.objective == Objective::Custom {
                return Err(invalid("boosting", "RF requires a built-in objective"));
            }
            if self.data_sample_strategy == DataSampleStrategy::Bagging
                && !((self.bagging_freq > 0 && self.bagging_fraction < 1.0)
                    || self.feature_fraction < 1.0)
            {
                return Err(invalid(
                    "boosting",
                    "RF requires active row or tree-feature sampling",
                ));
            }
        }
        if self.linear_tree && (self.zero_as_missing || self.objective == Objective::RegressionL1) {
            return Err(invalid(
                "linear_tree",
                "incompatible with zero_as_missing or regression_l1",
            ));
        }
        for (name, value) in [
            ("num_grad_quant_bins", self.num_grad_quant_bins),
            ("num_iterations", self.num_iterations),
            ("num_leaves", self.num_leaves),
            ("min_data_in_leaf", self.min_data_in_leaf),
            ("max_bin", self.max_bin),
            ("min_data_in_bin", self.min_data_in_bin),
            ("bin_construct_sample_cnt", self.bin_construct_sample_cnt),
            ("num_threads", self.num_threads),
            (
                "lambdarank_truncation_level",
                self.lambdarank_truncation_level,
            ),
            ("multi_error_top_k", self.multi_error_top_k),
            ("max_cat_threshold", self.max_cat_threshold),
            ("max_cat_to_onehot", self.max_cat_to_onehot),
            ("min_data_per_group", self.min_data_per_group),
        ] {
            if value > i32::MAX as usize {
                return Err(invalid(name, "must be at most 2147483647"));
            }
        }
        if !(2..=131_072).contains(&self.num_leaves) {
            return Err(invalid("num_leaves", "must be in 2..=131072"));
        }
        if self.max_bin < 2 {
            return Err(invalid("max_bin", "must be at least 2"));
        }
        for (name, value) in [
            ("min_data_in_bin", self.min_data_in_bin),
            ("bin_construct_sample_cnt", self.bin_construct_sample_cnt),
            (
                "lambdarank_truncation_level",
                self.lambdarank_truncation_level,
            ),
            ("multi_error_top_k", self.multi_error_top_k),
            ("max_cat_threshold", self.max_cat_threshold),
            ("max_cat_to_onehot", self.max_cat_to_onehot),
            ("min_data_per_group", self.min_data_per_group),
        ] {
            if value == 0 {
                return Err(invalid(name, "must be positive"));
            }
        }
        for (name, value, positive) in [
            ("learning_rate", self.learning_rate, true),
            (
                "min_sum_hessian_in_leaf",
                self.min_sum_hessian_in_leaf,
                false,
            ),
            ("lambda_l1", self.lambda_l1, false),
            ("lambda_l2", self.lambda_l2, false),
            ("min_gain_to_split", self.min_gain_to_split, false),
            ("sigmoid", self.sigmoid, true),
            ("scale_pos_weight", self.scale_pos_weight, true),
            ("alpha", self.alpha, true),
            ("fair_c", self.fair_c, true),
            ("poisson_max_delta_step", self.poisson_max_delta_step, true),
            ("cat_smooth", self.cat_smooth, false),
            ("cat_l2", self.cat_l2, false),
            ("linear_lambda", self.linear_lambda, false),
            (
                "lambdarank_position_bias_regularization",
                self.lambdarank_position_bias_regularization,
                false,
            ),
        ] {
            if !value.is_finite() || value < 0.0 || (positive && value == 0.0) {
                return Err(invalid(
                    name,
                    if positive {
                        "must be finite and positive"
                    } else {
                        "must be finite and nonnegative"
                    },
                ));
            }
        }
        if !self.max_delta_step.is_finite() {
            return Err(invalid("max_delta_step", "must be finite"));
        }
        if self.min_data_in_leaf == 0 && self.min_sum_hessian_in_leaf <= 1e-15_f32 as f64 {
            return Err(invalid(
                "min_data_in_leaf",
                "must be positive when min_sum_hessian_in_leaf is at most the upstream epsilon",
            ));
        }
        if self.is_unbalance && self.scale_pos_weight != 1.0 {
            return Err(invalid(
                "is_unbalance",
                "conflicts with scale_pos_weight != 1",
            ));
        }
        if !matches!(self.objective, Objective::Binary | Objective::MulticlassOva)
            && (self.is_unbalance || self.scale_pos_weight != 1.0)
        {
            return Err(invalid(
                "objective",
                "class weights require binary or multiclassova loss",
            ));
        }
        if !matches!(
            self.objective,
            Objective::Binary | Objective::MulticlassOva | Objective::LambdaRank
        ) && self.sigmoid != 1.0
        {
            return Err(invalid(
                "sigmoid",
                "requires binary, multiclassova, or lambdarank",
            ));
        }
        if !(1..=65_536).contains(&self.num_class) {
            return Err(invalid("num_class", "must be in 1..=65536"));
        }
        if (self.objective.is_multiclass() && self.num_class < 2)
            || (!self.objective.is_multiclass()
                && self.objective != Objective::Custom
                && self.num_class != 1)
        {
            return Err(invalid(
                "num_class",
                "must exceed 1 for multiclass and equal 1 for other built-in objectives",
            ));
        }
        if !self.tweedie_variance_power.is_finite()
            || !(1.0..2.0).contains(&self.tweedie_variance_power)
        {
            return Err(invalid(
                "tweedie_variance_power",
                "must be finite and in [1, 2)",
            ));
        }
        if self.objective == Objective::Quantile
            && !((self.alpha as f32) > 0.0 && (self.alpha as f32) < 1.0)
        {
            return Err(invalid("alpha", "quantile requires an f32 value in (0, 1)"));
        }
        if self.reg_sqrt
            && !matches!(
                self.objective,
                Objective::Regression
                    | Objective::RegressionL1
                    | Objective::Quantile
                    | Objective::Fair
                    | Objective::Mape
            )
        {
            return Err(invalid("reg_sqrt", "not supported by this objective"));
        }
        for &bins in &self.max_bin_by_feature {
            if !(2..=i32::MAX as usize).contains(&bins) {
                return Err(invalid(
                    "max_bin_by_feature",
                    "entries must be in 2..=2147483647",
                ));
            }
        }
        for bins in &self.forced_bins {
            if bins.iter().any(|v| !v.is_finite()) || bins.windows(2).any(|v| v[0] >= v[1]) {
                return Err(invalid(
                    "forced_bins",
                    "thresholds must be finite and strictly increasing",
                ));
            }
        }
        if self.label_gain.iter().any(|v| !v.is_finite()) {
            return Err(invalid("label_gain", "entries must be finite"));
        }
        if self
            .eval_at
            .iter()
            .any(|&v| v == 0 || v > i32::MAX as usize)
        {
            return Err(invalid(
                "eval_at",
                "entries must be positive signed 32-bit integers",
            ));
        }
        if !self.auc_mu_weights.is_empty() {
            let size = self
                .num_class
                .checked_mul(self.num_class)
                .filter(|&n| n <= 16 * 1024 * 1024)
                .ok_or_else(|| invalid("auc_mu_weights", "class matrix exceeds 16 Mi entries"))?;
            if self.auc_mu_weights.len() != size {
                return Err(invalid(
                    "auc_mu_weights",
                    "length must equal num_class squared",
                ));
            }
            for (i, &weight) in self.auc_mu_weights.iter().enumerate() {
                if !weight.is_finite()
                    || (i / self.num_class != i % self.num_class && weight.abs() < 1e-35_f32 as f64)
                {
                    return Err(invalid(
                        "auc_mu_weights",
                        "weights must be finite and off-diagonal entries nonzero",
                    ));
                }
            }
        }
        if self.num_threads > 256 {
            return Err(invalid("num_threads", "must be at most 256 workers"));
        }
        if self.force_col_wise && self.force_row_wise {
            return Err(invalid("force_row_wise", "conflicts with force_col_wise"));
        }
        if !self.histogram_pool_size.is_finite() {
            return Err(invalid("histogram_pool_size", "must be finite"));
        }
        let mut categorical = BTreeSet::new();
        for &index in &self.categorical_features {
            if index > i32::MAX as usize {
                return Err(invalid(
                    "categorical_feature",
                    "column index exceeds the signed 32-bit domain",
                ));
            }
            if !categorical.insert(index) {
                return Err(invalid("categorical_feature", "duplicate column index"));
            }
        }
        for metric in &self.metrics {
            if metric_name(metric)? != metric {
                return Err(invalid(
                    "metrics",
                    "direct field assignment requires canonical metric names",
                ));
            }
        }
        Ok(())
    }

    /// Parse upstream names. Keys are case-sensitive; surrounding spaces are removed.
    /// Canonical names take priority. Otherwise the shortest alias wins, with
    /// lexicographic order as the tie-breaker. Only the winning value is parsed.
    /// Repeated input names are errors, unlike upstream's warning-and-first policy.
    /// Empty supported values retain their defaults, as in upstream.
    /// An explicit seed derives each individual RNG seed unless it is also supplied.
    /// Positive max_depth reduces the default num_leaves to 2^depth when smaller;
    /// an explicit num_leaves is unchanged. Direct field assignments are unchanged.
    ///
    /// `device_type=cpu` and `tree_learner=serial` select the CPU learner.
    /// `boosting=goss` maps to GBDT with GOSS sampling. Zero threads selects one
    /// worker. Row-wise and column-wise modes are mutually exclusive. With both
    /// false, the learner selects a kernel without changing per-feature sum order.
    /// Fixed iteration order is not a guarantee of identical results across
    /// compilers, hardware, or library versions.
    pub fn from_pairs<I, K, V>(pairs: I) -> Result<Self>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let mut seen = BTreeSet::new();
        let mut selected: BTreeMap<&str, (String, String)> = BTreeMap::new();
        for (key, value) in pairs {
            let key = key.as_ref().trim();
            if key.is_empty() {
                return Err(invalid("parameter name", "must not be empty"));
            }
            if !seen.insert(key.to_owned()) {
                return Err(invalid(key, "duplicate parameter name"));
            }
            let canonical = canonical_name(key).ok_or_else(|| {
                unsupported(key, "unknown or unsupported parameter for CPU training")
            })?;
            let replace = match selected.get(canonical) {
                None => true,
                Some((previous, _)) => {
                    key == canonical
                        || (previous != canonical
                            && (key.len(), key) < (previous.len(), previous.as_str()))
                }
            };
            if replace {
                selected.insert(
                    canonical,
                    (key.to_owned(), value.as_ref().trim().to_owned()),
                );
            }
        }
        let mut config = Self::default();
        for (&key, (_, value)) in &selected {
            if value.is_empty() {
                continue;
            }
            match key {
                "objective" => config.objective = objective_name(value)?,
                "num_class" => config.num_class = number(key, value)?,
                "reg_sqrt" => config.reg_sqrt = boolean(key, value)?,
                "alpha" => config.alpha = number(key, value)?,
                "fair_c" => config.fair_c = number(key, value)?,
                "poisson_max_delta_step" => config.poisson_max_delta_step = number(key, value)?,
                "tweedie_variance_power" => config.tweedie_variance_power = number(key, value)?,
                "objective_seed" => config.objective_seed = number(key, value)?,
                "lambdarank_truncation_level" => {
                    config.lambdarank_truncation_level = number(key, value)?
                }
                "lambdarank_norm" => config.lambdarank_norm = boolean(key, value)?,
                "label_gain" => config.label_gain = list(key, value)?,
                "lambdarank_position_bias_regularization" => {
                    config.lambdarank_position_bias_regularization = number(key, value)?
                }
                "eval_at" => config.eval_at = list(key, value)?,
                "multi_error_top_k" => config.multi_error_top_k = number(key, value)?,
                "auc_mu_weights" => config.auc_mu_weights = list(key, value)?,
                "max_bin_by_feature" => config.max_bin_by_feature = list(key, value)?,
                "linear_tree" => config.linear_tree = boolean(key, value)?,
                "linear_lambda" => config.linear_lambda = number(key, value)?,
                "max_cat_threshold" => config.max_cat_threshold = number(key, value)?,
                "max_cat_to_onehot" => config.max_cat_to_onehot = number(key, value)?,
                "cat_smooth" => config.cat_smooth = number(key, value)?,
                "cat_l2" => config.cat_l2 = number(key, value)?,
                "min_data_per_group" => config.min_data_per_group = number(key, value)?,
                "num_iterations" => config.num_iterations = number(key, value)?,
                "learning_rate" => config.learning_rate = number(key, value)?,
                "num_leaves" => config.num_leaves = number(key, value)?,
                "max_depth" => config.max_depth = number(key, value)?,
                "min_data_in_leaf" => config.min_data_in_leaf = number(key, value)?,
                "min_sum_hessian_in_leaf" => config.min_sum_hessian_in_leaf = number(key, value)?,
                "lambda_l1" => config.lambda_l1 = number(key, value)?,
                "lambda_l2" => config.lambda_l2 = number(key, value)?,
                "min_gain_to_split" => config.min_gain_to_split = number(key, value)?,
                "max_delta_step" => config.max_delta_step = number(key, value)?,
                "max_bin" => config.max_bin = number(key, value)?,
                "min_data_in_bin" => config.min_data_in_bin = number(key, value)?,
                "bin_construct_sample_cnt" => config.bin_construct_sample_cnt = number(key, value)?,
                "data_random_seed" => config.data_random_seed = number(key, value)?,
                "seed" => config.seed = number(key, value)?,
                "num_threads" => config.num_threads = number(key, value)?,
                "deterministic" => config.deterministic = boolean(key, value)?,
                "force_col_wise" => config.force_col_wise = boolean(key, value)?,
                "force_row_wise" => config.force_row_wise = boolean(key, value)?,
                "histogram_pool_size" => config.histogram_pool_size = number(key, value)?,
                "enable_bundle" => config.enable_bundle = boolean(key, value)?,
                "zero_as_missing" => config.zero_as_missing = boolean(key, value)?,
                "use_missing" => config.use_missing = boolean(key, value)?,
                "feature_pre_filter" => config.feature_pre_filter = boolean(key, value)?,
                "boost_from_average" => config.boost_from_average = boolean(key, value)?,
                "sigmoid" => config.sigmoid = number(key, value)?,
                "is_unbalance" => config.is_unbalance = boolean(key, value)?,
                "scale_pos_weight" => config.scale_pos_weight = number(key, value)?,
                "verbosity" => config.verbosity = number(key, value)?,
                "categorical_feature" => {
                    if value.eq_ignore_ascii_case("auto") || value.starts_with("name:") {
                        return Err(unsupported(
                            key,
                            "only explicit unsigned column indexes are supported; named columns and automatic adapters are not implemented",
                        ));
                    }
                    for part in value.split(',') {
                        config.categorical_features.push(number(key, part.trim())?);
                    }
                }
                "metric" => {
                    for part in value.split(',') {
                        if part.trim().is_empty() {
                            return Err(invalid(key, "metric list contains an empty entry"));
                        }
                        let metric = metric_name(part)?;
                        if !config.metrics.iter().any(|existing| existing == metric) {
                            config.metrics.push(metric.to_owned());
                        }
                    }
                }
                "device_type" => require_value(key, value, &["cpu"])?,
                "boosting" => {
                    config.boosting = match value.to_ascii_lowercase().as_str() {
                        "gbdt" | "gbrt" | "goss" => Boosting::Gbdt,
                        "dart" => Boosting::Dart,
                        "rf" | "random_forest" => Boosting::Rf,
                        _ => return Err(invalid(key, "expected gbdt, dart, rf, or goss")),
                    }
                }
                "data_sample_strategy" => {
                    config.data_sample_strategy = match value.to_ascii_lowercase().as_str() {
                        "bagging" => DataSampleStrategy::Bagging,
                        "goss" => DataSampleStrategy::Goss,
                        _ => return Err(invalid(key, "expected bagging or goss")),
                    }
                }
                "monotone_constraints_method" => {
                    config.monotone_constraints_method = match value.to_ascii_lowercase().as_str() {
                        "basic" => MonotoneMethod::Basic,
                        "intermediate" => MonotoneMethod::Intermediate,
                        "advanced" => MonotoneMethod::Advanced,
                        _ => return Err(invalid(key, "expected basic, intermediate, or advanced")),
                    }
                }
                "interaction_constraints" => {
                    config.interaction_constraints =
                        serde_json::from_str(value).map_err(|e| invalid(key, &e.to_string()))?
                }
                "forcedsplits_filename" => {
                    let json =
                        std::fs::read_to_string(value).map_err(|e| invalid(key, &e.to_string()))?;
                    config.forced_splits = Some(
                        serde_json::from_str(&json).map_err(|e| invalid(key, &e.to_string()))?,
                    );
                }
                "bagging_fraction" => config.bagging_fraction = number(key, value)?,
                "pos_bagging_fraction" => config.pos_bagging_fraction = number(key, value)?,
                "neg_bagging_fraction" => config.neg_bagging_fraction = number(key, value)?,
                "bagging_freq" => config.bagging_freq = number(key, value)?,
                "bagging_seed" => config.bagging_seed = number(key, value)?,
                "bagging_by_query" => config.bagging_by_query = boolean(key, value)?,
                "feature_fraction" => config.feature_fraction = number(key, value)?,
                "feature_fraction_bynode" => config.feature_fraction_bynode = number(key, value)?,
                "feature_fraction_seed" => config.feature_fraction_seed = number(key, value)?,
                "extra_trees" => config.extra_trees = boolean(key, value)?,
                "extra_seed" => config.extra_seed = number(key, value)?,
                "top_rate" => config.top_rate = number(key, value)?,
                "other_rate" => config.other_rate = number(key, value)?,
                "drop_rate" => config.drop_rate = number(key, value)?,
                "max_drop" => config.max_drop = number(key, value)?,
                "skip_drop" => config.skip_drop = number(key, value)?,
                "xgboost_dart_mode" => config.xgboost_dart_mode = boolean(key, value)?,
                "uniform_drop" => config.uniform_drop = boolean(key, value)?,
                "drop_seed" => config.drop_seed = number(key, value)?,
                "monotone_constraints" => config.monotone_constraints = list(key, value)?,
                "monotone_penalty" => config.monotone_penalty = number(key, value)?,
                "path_smooth" => config.path_smooth = number(key, value)?,
                "feature_contri" => config.feature_contri = list(key, value)?,
                "cegb_tradeoff" => config.cegb_tradeoff = number(key, value)?,
                "cegb_penalty_split" => config.cegb_penalty_split = number(key, value)?,
                "cegb_penalty_feature_coupled" => {
                    config.cegb_penalty_feature_coupled = list(key, value)?
                }
                "cegb_penalty_feature_lazy" => config.cegb_penalty_feature_lazy = list(key, value)?,
                "use_quantized_grad" => config.use_quantized_grad = boolean(key, value)?,
                "num_grad_quant_bins" => config.num_grad_quant_bins = number(key, value)?,
                "quant_train_renew_leaf" => config.quant_train_renew_leaf = boolean(key, value)?,
                "stochastic_rounding" => config.stochastic_rounding = boolean(key, value)?,
                "tree_learner" => require_value(key, value, &["serial"])?,
                _ => return Err(unsupported(key, "parameter is not implemented")),
            }
        }
        let explicit = |key| {
            selected
                .get(key)
                .is_some_and(|(_, value)| !value.is_empty())
        };
        if explicit("seed") {
            let mut random = crate::random::Random::new(config.seed);
            let derived = random.next_short(0, 32767)? as u32;
            if !explicit("data_random_seed") {
                config.data_random_seed = derived;
            }
            let derived = random.next_short(0, 32767)? as u32;
            if !explicit("bagging_seed") {
                config.bagging_seed = derived;
            }
            let derived = random.next_short(0, 32767)? as u32;
            if !explicit("drop_seed") {
                config.drop_seed = derived;
            }
            let derived = random.next_short(0, 32767)? as u32;
            if !explicit("feature_fraction_seed") {
                config.feature_fraction_seed = derived;
            }
            let derived = random.next_short(0, 32767)? as u32;
            if !explicit("objective_seed") {
                config.objective_seed = derived;
            }
            let derived = random.next_short(0, 32767)? as u32;
            if !explicit("extra_seed") {
                config.extra_seed = derived;
            }
        }
        if selected
            .get("boosting")
            .is_some_and(|(_, v)| v.eq_ignore_ascii_case("goss"))
        {
            config.data_sample_strategy = DataSampleStrategy::Goss;
        }
        if config.path_smooth > 1e-15_f32 as f64 && config.min_data_in_leaf < 2 {
            config.min_data_in_leaf = 2;
        }
        if config.feature_fraction_bynode != 1.0 {
            config.monotone_constraints_method = MonotoneMethod::Basic;
        }
        config.eval_at.sort_unstable();
        if !explicit("num_leaves") && (1..5).contains(&config.max_depth) {
            config.num_leaves = 1usize << config.max_depth;
        }
        // Config::CheckParamConflict normalizes this source-only combination.
        // Typed Config values are canonical and still reject the conflict.
        if config.data_sample_strategy != DataSampleStrategy::Bagging {
            config.bagging_by_query = false;
        }
        config.validate()?;
        Ok(config)
    }
}

fn invalid(name: &str, reason: &str) -> Error {
    Error::InvalidInput(format!("{name}: {reason}"))
}

fn unsupported(name: &str, reason: &str) -> Error {
    Error::Unsupported(format!("{name}: {reason}"))
}

fn number<T: std::str::FromStr>(name: &str, value: &str) -> Result<T> {
    value
        .parse()
        .map_err(|_| invalid(name, "invalid number or numeric overflow"))
}

fn list<T: std::str::FromStr>(name: &str, value: &str) -> Result<Vec<T>> {
    let count = value.split(',').count();
    if count > 16 * 1024 * 1024 {
        return Err(invalid(name, "numeric list exceeds 16 Mi entries"));
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(count)
        .map_err(|e| invalid(name, &format!("numeric list allocation failed: {e}")))?;
    for part in value.split(',') {
        output.push(number(name, part.trim())?);
    }
    Ok(output)
}

fn boolean(name: &str, value: &str) -> Result<bool> {
    match value.to_ascii_lowercase().as_str() {
        "true" | "+" => Ok(true),
        "false" | "-" => Ok(false),
        _ => Err(invalid(name, "expected true/+ or false/-")),
    }
}

fn require_value(name: &str, value: &str, supported: &[&str]) -> Result<()> {
    if supported
        .iter()
        .any(|item| value.eq_ignore_ascii_case(item))
    {
        Ok(())
    } else {
        Err(unsupported(
            name,
            &format!(
                "unsupported value {value:?}; supported: {}",
                supported.join(", ")
            ),
        ))
    }
}

fn objective_name(value: &str) -> Result<Objective> {
    match value.to_ascii_lowercase().as_str() {
        "regression"
        | "regression_l2"
        | "l2"
        | "mean_squared_error"
        | "mse"
        | "l2_root"
        | "root_mean_squared_error"
        | "rmse" => Ok(Objective::Regression),
        "regression_l1" | "l1" | "mean_absolute_error" | "mae" => Ok(Objective::RegressionL1),
        "quantile" => Ok(Objective::Quantile),
        "huber" => Ok(Objective::Huber),
        "fair" => Ok(Objective::Fair),
        "poisson" => Ok(Objective::Poisson),
        "mape" | "mean_absolute_percentage_error" => Ok(Objective::Mape),
        "gamma" => Ok(Objective::Gamma),
        "tweedie" => Ok(Objective::Tweedie),
        "binary" => Ok(Objective::Binary),
        "multiclass" | "softmax" => Ok(Objective::Multiclass),
        "multiclassova" | "multiclass_ova" | "ova" | "ovr" => Ok(Objective::MulticlassOva),
        "cross_entropy" | "xentropy" => Ok(Objective::CrossEntropy),
        "cross_entropy_lambda" | "xentlambda" => Ok(Objective::CrossEntropyLambda),
        "lambdarank" => Ok(Objective::LambdaRank),
        "rank_xendcg" | "xendcg" | "xe_ndcg" | "xe_ndcg_mart" | "xendcg_mart" => {
            Ok(Objective::RankXendcg)
        }
        "none" | "null" | "na" | "custom" => Ok(Objective::Custom),
        _ => Err(unsupported(
            "objective",
            &format!("unknown objective {value:?}"),
        )),
    }
}

fn metric_name(value: &str) -> Result<&'static str> {
    match value.trim().to_ascii_lowercase().as_str() {
        "regression" | "regression_l2" | "l2" | "mean_squared_error" | "mse" => Ok("l2"),
        "l2_root" | "root_mean_squared_error" | "rmse" => Ok("rmse"),
        "regression_l1" | "l1" | "mean_absolute_error" | "mae" => Ok("l1"),
        "quantile" => Ok("quantile"),
        "huber" => Ok("huber"),
        "fair" => Ok("fair"),
        "poisson" => Ok("poisson"),
        "mape" | "mean_absolute_percentage_error" => Ok("mape"),
        "gamma" => Ok("gamma"),
        "gamma_deviance" => Ok("gamma_deviance"),
        "tweedie" => Ok("tweedie"),
        "r2" => Ok("r2"),
        "binary" | "binary_logloss" => Ok("binary_logloss"),
        "binary_error" => Ok("binary_error"),
        "auc" => Ok("auc"),
        "average_precision" => Ok("average_precision"),
        "auc_mu" => Ok("auc_mu"),
        "ndcg" | "lambdarank" | "rank_xendcg" | "xendcg" | "xe_ndcg" | "xe_ndcg_mart"
        | "xendcg_mart" => Ok("ndcg"),
        "map" | "mean_average_precision" => Ok("map"),
        "multi_logloss" | "multiclass" | "softmax" | "multiclassova" | "multiclass_ova" | "ova"
        | "ovr" => Ok("multi_logloss"),
        "multi_error" => Ok("multi_error"),
        "cross_entropy" | "xentropy" => Ok("cross_entropy"),
        "cross_entropy_lambda" | "xentlambda" => Ok("cross_entropy_lambda"),
        "kullback_leibler" | "kldiv" => Ok("kullback_leibler"),
        "none" | "null" | "na" | "custom" => Ok("custom"),
        _ => Err(unsupported("metric", &format!("unknown metric {value:?}"))),
    }
}

fn canonical_name(name: &str) -> Option<&'static str> {
    Some(match name {
        "objective" | "objective_type" | "app" | "application" | "loss" => "objective",
        "num_iterations" | "num_iteration" | "n_iter" | "num_tree" | "num_trees" | "num_round"
        | "num_rounds" | "nrounds" | "num_boost_round" | "n_estimators" | "max_iter" => {
            "num_iterations"
        }
        "learning_rate" | "shrinkage_rate" | "eta" => "learning_rate",
        "num_leaves" | "num_leaf" | "max_leaves" | "max_leaf" | "max_leaf_nodes" => "num_leaves",
        "num_class" | "num_classes" => "num_class",
        "reg_sqrt" => "reg_sqrt",
        "alpha" => "alpha",
        "fair_c" => "fair_c",
        "poisson_max_delta_step" => "poisson_max_delta_step",
        "tweedie_variance_power" => "tweedie_variance_power",
        "objective_seed" => "objective_seed",
        "lambdarank_truncation_level" => "lambdarank_truncation_level",
        "lambdarank_norm" => "lambdarank_norm",
        "label_gain" => "label_gain",
        "lambdarank_position_bias_regularization" => "lambdarank_position_bias_regularization",
        "eval_at" | "ndcg_eval_at" | "ndcg_at" | "map_eval_at" | "map_at" => "eval_at",
        "multi_error_top_k" => "multi_error_top_k",
        "auc_mu_weights" => "auc_mu_weights",
        "max_bin_by_feature" => "max_bin_by_feature",
        "linear_tree" | "linear_trees" => "linear_tree",
        "linear_lambda" => "linear_lambda",
        "max_cat_threshold" => "max_cat_threshold",
        "max_cat_to_onehot" => "max_cat_to_onehot",
        "cat_smooth" => "cat_smooth",
        "cat_l2" => "cat_l2",
        "min_data_per_group" => "min_data_per_group",
        "max_depth" => "max_depth",
        "min_data_in_leaf" | "min_data_per_leaf" | "min_data" | "min_child_samples"
        | "min_samples_leaf" => "min_data_in_leaf",
        "min_sum_hessian_in_leaf"
        | "min_sum_hessian_per_leaf"
        | "min_sum_hessian"
        | "min_hessian"
        | "min_child_weight" => "min_sum_hessian_in_leaf",
        "lambda_l1" | "reg_alpha" | "l1_regularization" => "lambda_l1",
        "lambda_l2" | "reg_lambda" | "lambda" | "l2_regularization" => "lambda_l2",
        "min_gain_to_split" | "min_split_gain" => "min_gain_to_split",
        "max_delta_step" | "max_tree_output" | "max_leaf_output" => "max_delta_step",
        "max_bin" | "max_bins" => "max_bin",
        "min_data_in_bin" => "min_data_in_bin",
        "bin_construct_sample_cnt" | "subsample_for_bin" => "bin_construct_sample_cnt",
        "data_random_seed" | "data_seed" => "data_random_seed",
        "seed" | "random_seed" | "random_state" => "seed",
        "num_threads" | "num_thread" | "nthread" | "nthreads" | "n_jobs" => "num_threads",
        "zero_as_missing" => "zero_as_missing",
        "use_missing" => "use_missing",
        "feature_pre_filter" => "feature_pre_filter",
        "boost_from_average" => "boost_from_average",
        "sigmoid" => "sigmoid",
        "is_unbalance" | "unbalance" | "unbalanced_sets" => "is_unbalance",
        "scale_pos_weight" => "scale_pos_weight",
        "categorical_feature"
        | "cat_feature"
        | "categorical_column"
        | "cat_column"
        | "categorical_features" => "categorical_feature",
        "metric" | "metrics" | "metric_types" => "metric",
        "verbosity" | "verbose" => "verbosity",
        "device_type" | "device" => "device_type",
        "boosting" | "boosting_type" | "boost" => "boosting",
        "tree_learner" | "tree" | "tree_type" | "tree_learner_type" => "tree_learner",
        "deterministic" => "deterministic",
        "force_col_wise" => "force_col_wise",
        "force_row_wise" => "force_row_wise",
        "histogram_pool_size" | "hist_pool_size" => "histogram_pool_size",
        "enable_bundle" | "is_enable_bundle" | "bundle" => "enable_bundle",
        "bagging_fraction" | "sub_row" | "subsample" | "bagging" => "bagging_fraction",
        "pos_bagging_fraction" | "pos_sub_row" | "pos_subsample" | "pos_bagging" => {
            "pos_bagging_fraction"
        }
        "neg_bagging_fraction" | "neg_sub_row" | "neg_subsample" | "neg_bagging" => {
            "neg_bagging_fraction"
        }
        "bagging_freq" | "subsample_freq" => "bagging_freq",
        "bagging_seed" | "bagging_fraction_seed" => "bagging_seed",
        "bagging_by_query" => "bagging_by_query",
        "feature_fraction" | "sub_feature" | "colsample_bytree" => "feature_fraction",
        "feature_fraction_bynode" | "sub_feature_bynode" | "colsample_bynode" => {
            "feature_fraction_bynode"
        }
        "feature_fraction_seed" => "feature_fraction_seed",
        "extra_trees" | "extra_tree" => "extra_trees",
        "extra_seed" => "extra_seed",
        "top_rate" => "top_rate",
        "other_rate" => "other_rate",
        "drop_rate" | "rate_drop" => "drop_rate",
        "max_drop" => "max_drop",
        "skip_drop" => "skip_drop",
        "xgboost_dart_mode" => "xgboost_dart_mode",
        "uniform_drop" => "uniform_drop",
        "drop_seed" => "drop_seed",
        "monotone_constraints" | "mc" | "monotone_constraint" | "monotonic_cst" => {
            "monotone_constraints"
        }
        "monotone_penalty" | "monotone_splits_penalty" | "ms_penalty" | "mc_penalty" => {
            "monotone_penalty"
        }
        "path_smooth" => "path_smooth",
        "feature_contri" | "feature_contrib" | "fc" | "fp" | "feature_penalty" => "feature_contri",
        "cegb_tradeoff" => "cegb_tradeoff",
        "cegb_penalty_split" => "cegb_penalty_split",
        "cegb_penalty_feature_coupled" => "cegb_penalty_feature_coupled",
        "cegb_penalty_feature_lazy" => "cegb_penalty_feature_lazy",
        "use_quantized_grad" => "use_quantized_grad",
        "num_grad_quant_bins" => "num_grad_quant_bins",
        "quant_train_renew_leaf" => "quant_train_renew_leaf",
        "stochastic_rounding" => "stochastic_rounding",
        "data_sample_strategy" => "data_sample_strategy",
        "monotone_constraints_method" | "monotone_constraining_method" | "mc_method" => {
            "monotone_constraints_method"
        }
        "interaction_constraints" => "interaction_constraints",
        "forcedsplits_filename"
        | "fs"
        | "forced_splits_filename"
        | "forced_splits_file"
        | "forced_splits" => "forcedsplits_filename",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Source evidence: LightGBM 5d84f3f222b41409a6983d2ac90513ad9702fa9e,
    // include/LightGBM/config.h and src/io/{config.cpp,config_auto.cpp}.
    #[test]
    fn pinned_base_defaults() {
        let c = Config::default();
        assert_eq!(c.objective, Objective::Regression);
        assert_eq!(c.num_iterations, 100);
        assert_eq!(c.learning_rate, 0.1);
        assert_eq!(c.num_leaves, 31);
        assert_eq!(c.max_depth, -1);
        assert_eq!(c.min_data_in_leaf, 20);
        assert_eq!(c.min_sum_hessian_in_leaf, 1e-3);
        assert_eq!(c.lambda_l1, 0.0);
        assert_eq!(c.lambda_l2, 0.0);
        assert_eq!(c.min_gain_to_split, 0.0);
        assert_eq!(c.max_delta_step, 0.0);
        assert_eq!(c.max_bin, 255);
        assert_eq!(c.min_data_in_bin, 3);
        assert_eq!(c.bin_construct_sample_cnt, 200_000);
        assert_eq!(c.data_random_seed, 1);
        assert_eq!(c.seed, 0);
        assert_eq!(c.num_threads, 0);
        assert!(!c.zero_as_missing);
        assert!(c.use_missing);
        assert!(c.feature_pre_filter);
        assert!(c.boost_from_average);
        assert_eq!(c.sigmoid, 1.0);
        assert!(!c.is_unbalance);
        assert_eq!(c.scale_pos_weight, 1.0);
        assert!(c.categorical_features.is_empty());
        assert!(c.metrics.is_empty());
        // Rust API default, deliberately quieter than the upstream C++ default 1.
        assert_eq!(c.verbosity, -1);
        c.validate().unwrap();
        assert_eq!(format!("{c:?}"), format!("{:?}", c.clone()));
        assert_eq!(
            format!("{c:?}"),
            format!("{:?}", Config::from_pairs([] as [(&str, &str); 0]).unwrap())
        );
    }

    #[test]
    fn pinned_parameter_aliases() {
        let cases: &[(&str, &str, &[&str])] = &[
            (
                "objective",
                "binary",
                &["objective_type", "app", "application", "loss"],
            ),
            (
                "num_iterations",
                "7",
                &[
                    "num_iteration",
                    "n_iter",
                    "num_tree",
                    "num_trees",
                    "num_round",
                    "num_rounds",
                    "nrounds",
                    "num_boost_round",
                    "n_estimators",
                    "max_iter",
                ],
            ),
            ("learning_rate", "0.3", &["shrinkage_rate", "eta"]),
            (
                "num_leaves",
                "17",
                &["num_leaf", "max_leaves", "max_leaf", "max_leaf_nodes"],
            ),
            (
                "num_threads",
                "1",
                &["num_thread", "nthread", "nthreads", "n_jobs"],
            ),
            ("seed", "42", &["random_seed", "random_state"]),
            (
                "min_data_in_leaf",
                "4",
                &[
                    "min_data_per_leaf",
                    "min_data",
                    "min_child_samples",
                    "min_samples_leaf",
                ],
            ),
            (
                "min_sum_hessian_in_leaf",
                "0.2",
                &[
                    "min_sum_hessian_per_leaf",
                    "min_sum_hessian",
                    "min_hessian",
                    "min_child_weight",
                ],
            ),
            (
                "max_delta_step",
                "2",
                &["max_tree_output", "max_leaf_output"],
            ),
            ("lambda_l1", "2", &["reg_alpha", "l1_regularization"]),
            (
                "lambda_l2",
                "2",
                &["reg_lambda", "lambda", "l2_regularization"],
            ),
            ("min_gain_to_split", "2", &["min_split_gain"]),
            ("verbosity", "0", &["verbose"]),
            ("max_bin", "31", &["max_bins"]),
            ("bin_construct_sample_cnt", "100", &["subsample_for_bin"]),
            ("data_random_seed", "4", &["data_seed"]),
            ("is_unbalance", "false", &["unbalance", "unbalanced_sets"]),
            ("metric", "l2", &["metrics", "metric_types"]),
            ("device_type", "cpu", &["device"]),
            ("boosting", "gbdt", &["boosting_type", "boost"]),
            (
                "tree_learner",
                "serial",
                &["tree", "tree_type", "tree_learner_type"],
            ),
        ];
        for &(canonical, value, aliases) in cases {
            let expected = Config::from_pairs([(canonical, value)]).unwrap();
            for alias in aliases {
                let actual = Config::from_pairs([(*alias, value)]).unwrap();
                assert_eq!(format!("{actual:?}"), format!("{expected:?}"), "{alias}");
            }
        }
    }

    #[test]
    fn objective_and_metric_value_aliases() {
        for name in [
            "regression",
            "regression_l2",
            "l2",
            "mean_squared_error",
            "mse",
            "l2_root",
            "root_mean_squared_error",
            "rmse",
            "REGRESSION",
        ] {
            assert_eq!(
                Config::from_pairs([("objective", name)]).unwrap().objective,
                Objective::Regression
            );
        }
        assert_eq!(
            Config::from_pairs([("objective", "BINARY")])
                .unwrap()
                .objective,
            Objective::Binary
        );
        for (input, expected) in [
            ("regression,regression_l2,l2,mean_squared_error,mse", "l2"),
            ("l2_root,root_mean_squared_error,rmse", "rmse"),
            ("binary,binary_logloss", "binary_logloss"),
            ("none,null,na,custom", "custom"),
        ] {
            let c = Config::from_pairs([("objective", "binary"), ("metric", input)]).unwrap();
            assert_eq!(c.metrics, [expected]);
        }
        let c = Config::from_pairs([("metric", " RMSE, l2,rmse ")]).unwrap();
        assert_eq!(c.metrics, ["rmse", "l2"]);
        assert!(
            Config::from_pairs([("metric", "")])
                .unwrap()
                .metrics
                .is_empty()
        );
    }

    #[test]
    fn upstream_alias_precedence_is_independent_of_input_order() {
        for pairs in [
            vec![
                ("eta", "0.3"),
                ("shrinkage_rate", "0.4"),
                ("learning_rate", "0.2"),
            ],
            vec![
                ("learning_rate", "0.2"),
                ("shrinkage_rate", "0.4"),
                ("eta", "0.3"),
            ],
        ] {
            assert_eq!(Config::from_pairs(pairs).unwrap().learning_rate, 0.2);
        }
        assert_eq!(
            Config::from_pairs([("shrinkage_rate", "0.4"), ("eta", "0.3")])
                .unwrap()
                .learning_rate,
            0.3
        );
        // Equal length: max_leaf sorts before num_leaf.
        assert_eq!(
            Config::from_pairs([("num_leaf", "8"), ("max_leaf", "9")])
                .unwrap()
                .num_leaves,
            9
        );
        assert_eq!(
            Config::from_pairs([("max_leaf", "9"), ("num_leaf", "8")])
                .unwrap()
                .num_leaves,
            9
        );
        // Upstream parses only the winning value, including when the loser is invalid.
        assert_eq!(
            Config::from_pairs([("learning_rate", "0.2"), ("eta", "invalid")])
                .unwrap()
                .learning_rate,
            0.2
        );
        assert!(Config::from_pairs([("learning_rate", "invalid"), ("eta", "0.2")]).is_err());
        // Empty canonical value still overrides the alias and uses the default.
        assert_eq!(
            Config::from_pairs([("learning_rate", ""), ("eta", "0.3")])
                .unwrap()
                .learning_rate,
            0.1
        );
    }

    #[test]
    fn explicit_seed_and_depth_adjustments_follow_source() {
        // First RNG update is (214013 * seed + 2531011), then >>16 & 0x7fff.
        assert_eq!(
            Config::from_pairs([("seed", "0")])
                .unwrap()
                .data_random_seed,
            38
        );
        assert_eq!(
            Config::from_pairs([("random_state", "42")])
                .unwrap()
                .data_random_seed,
            175
        );
        assert_eq!(
            Config::from_pairs([("seed", "42"), ("data_seed", "7")])
                .unwrap()
                .data_random_seed,
            7
        );
        let c = Config::from_pairs([("seed", "4294967295")]).unwrap();
        assert_eq!(c.seed, u32::MAX);
        assert_eq!(
            c.data_random_seed,
            ((u32::MAX.wrapping_mul(214_013).wrapping_add(2_531_011) >> 16) & 0x7fff) % 32767
        );
        let c = Config::from_pairs([("data_seed", "4294967295")]).unwrap();
        assert_eq!(c.data_random_seed, u32::MAX);
        let c = Config {
            seed: 42,
            ..Config::default()
        };
        c.validate().unwrap();
        assert_eq!(c.data_random_seed, 1);
        assert_eq!(
            Config::from_pairs([("max_depth", "2")]).unwrap().num_leaves,
            4
        );
        assert_eq!(
            Config::from_pairs([("max_depth", "2"), ("num_leaf", "8")])
                .unwrap()
                .num_leaves,
            8
        );
        assert_eq!(
            Config::from_pairs([("max_depth", "2147483647")])
                .unwrap()
                .num_leaves,
            31
        );
        assert_eq!(
            Config::from_pairs([("max_depth", "0")]).unwrap().max_depth,
            0
        );
        assert_eq!(
            Config::from_pairs([("max_delta_step", "-1")])
                .unwrap()
                .max_delta_step,
            -1.0
        );
        assert_eq!(
            Config::from_pairs([("num_iterations", "0")])
                .unwrap()
                .num_iterations,
            0
        );
    }

    #[test]
    fn whitespace_and_bool_syntax_are_checked() {
        let c = Config::from_pairs([
            (" learning_rate\t", " 0.25\n"),
            ("zero_as_missing", "+"),
            ("use_missing", "TRUE"),
            ("boost_from_average", "-"),
        ])
        .unwrap();
        assert_eq!(c.learning_rate, 0.25);
        assert!(c.zero_as_missing);
        assert!(!c.boost_from_average);
        for value in ["1", "0", "yes", "true false"] {
            assert!(matches!(
                Config::from_pairs([("use_missing", value)]),
                Err(Error::InvalidInput(_))
            ));
        }
        for key in ["", " ", "learning rate", "Learning_rate", "learn_rate"] {
            assert!(Config::from_pairs([(key, "1")]).is_err(), "{key:?}");
        }
        for value in ["0.1 0.2", "1=2", "1.0junk"] {
            assert!(Config::from_pairs([("learning_rate", value)]).is_err());
        }
        for pairs in [
            [("eta", "0.2"), ("eta", "0.2")],
            [("learning_rate", "0.2"), (" learning_rate ", "0.3")],
        ] {
            assert!(matches!(
                Config::from_pairs(pairs),
                Err(Error::InvalidInput(_))
            ));
        }
    }

    #[test]
    fn rejects_numeric_domains_and_overflow_without_panics() {
        for key in [
            "learning_rate",
            "min_sum_hessian_in_leaf",
            "lambda_l1",
            "lambda_l2",
            "min_gain_to_split",
            "max_delta_step",
            "sigmoid",
            "scale_pos_weight",
        ] {
            for value in ["NaN", "inf", "-inf", "Infinity", "-Infinity", "1e9999"] {
                assert!(
                    matches!(
                        Config::from_pairs([(key, value)]),
                        Err(Error::InvalidInput(_))
                    ),
                    "{key}={value}"
                );
            }
        }
        for (key, values) in [
            ("learning_rate", vec!["0", "-1"]),
            ("num_leaves", vec!["0", "1", "131073"]),
            ("max_bin", vec!["0", "1"]),
            ("min_data_in_bin", vec!["0"]),
            ("bin_construct_sample_cnt", vec!["0"]),
            ("sigmoid", vec!["0", "-1"]),
            ("scale_pos_weight", vec!["0", "-1"]),
            ("lambda_l1", vec!["-1"]),
            ("lambda_l2", vec!["-1"]),
            ("min_gain_to_split", vec!["-1"]),
            ("min_sum_hessian_in_leaf", vec!["-1"]),
        ] {
            for value in values {
                assert!(Config::from_pairs([(key, value)]).is_err(), "{key}={value}");
            }
        }
        for key in [
            "num_iterations",
            "num_leaves",
            "min_data_in_leaf",
            "max_bin",
            "min_data_in_bin",
            "bin_construct_sample_cnt",
            "num_threads",
        ] {
            for value in [
                "-1",
                "2147483648",
                "18446744073709551615",
                "18446744073709551616",
                "1.5",
            ] {
                assert!(Config::from_pairs([(key, value)]).is_err(), "{key}={value}");
            }
        }
        for key in ["max_depth", "verbosity"] {
            for value in ["2147483648", "-2147483649"] {
                assert!(Config::from_pairs([(key, value)]).is_err());
            }
        }
        for key in ["seed", "data_random_seed"] {
            for value in ["-1", "4294967296", "18446744073709551616"] {
                assert!(Config::from_pairs([(key, value)]).is_err());
            }
        }
    }

    #[test]
    fn parsed_values_and_cross_parameter_conflicts() {
        let c = Config::from_pairs([
            ("objective", "binary"),
            ("is_unbalance", "+"),
            ("sigmoid", "2.5"),
            ("max_depth", "-20"),
            ("feature_pre_filter", "false"),
            ("min_data_in_leaf", "0"),
            ("min_sum_hessian_in_leaf", "0.2"),
            ("max_bin", "2147483647"),
            ("min_data_in_bin", "2147483647"),
            ("bin_construct_sample_cnt", "2147483647"),
            ("verbosity", "-2147483648"),
        ])
        .unwrap();
        assert_eq!(c.objective, Objective::Binary);
        assert!(c.is_unbalance);
        assert_eq!(c.sigmoid, 2.5);
        assert_eq!(c.max_depth, -20);
        assert!(!c.feature_pre_filter);
        assert_eq!(c.min_data_in_leaf, 0);
        assert_eq!(c.max_bin, i32::MAX as usize);
        assert_eq!(c.verbosity, i32::MIN);
        assert!(
            Config::from_pairs([
                ("objective", "binary"),
                ("unbalance", "true"),
                ("scale_pos_weight", "2")
            ])
            .is_err()
        );
        assert!(Config::from_pairs([("is_unbalance", "true")]).is_err());
        assert!(Config::from_pairs([("scale_pos_weight", "2")]).is_err());
        assert!(Config::from_pairs([("sigmoid", "2")]).is_err());
        assert!(
            Config::from_pairs([("min_data_in_leaf", "0"), ("min_sum_hessian_in_leaf", "0")])
                .is_err()
        );
        Config::from_pairs([("objective", "binary"), ("scale_pos_weight", "2")]).unwrap();
        Config::from_pairs([("num_leaves", "131072"), ("min_sum_hessian_in_leaf", "0")]).unwrap();
        // Parsed numeric limits do not cause allocations proportional to their values.
        Config::from_pairs([("num_iterations", "2147483647")]).unwrap();
    }

    #[test]
    fn typed_validation_cannot_bypass_domains_or_conflicts() {
        let float_fields: &[fn(&mut Config, f64)] = &[
            |c, v| c.learning_rate = v,
            |c, v| c.min_sum_hessian_in_leaf = v,
            |c, v| c.lambda_l1 = v,
            |c, v| c.lambda_l2 = v,
            |c, v| c.min_gain_to_split = v,
            |c, v| c.max_delta_step = v,
            |c, v| c.sigmoid = v,
            |c, v| c.scale_pos_weight = v,
        ];
        for set in float_fields {
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut c = Config::default();
                set(&mut c, value);
                assert!(matches!(c.validate(), Err(Error::InvalidInput(_))));
            }
        }
        let usize_fields: &[fn(&mut Config, usize)] = &[
            |c, v| c.num_iterations = v,
            |c, v| c.num_leaves = v,
            |c, v| c.min_data_in_leaf = v,
            |c, v| c.max_bin = v,
            |c, v| c.min_data_in_bin = v,
            |c, v| c.bin_construct_sample_cnt = v,
            |c, v| c.num_threads = v,
        ];
        for set in usize_fields {
            let mut c = Config::default();
            set(&mut c, usize::MAX);
            assert!(c.validate().is_err());
        }
        let mut c = Config {
            objective: Objective::Binary,
            is_unbalance: true,
            scale_pos_weight: 2.0,
            ..Config::default()
        };
        assert!(matches!(c.validate(), Err(Error::InvalidInput(_))));
        c.scale_pos_weight = 1.0;
        c.validate().unwrap();
        c.objective = Objective::Regression;
        assert!(c.validate().is_err());
        c.is_unbalance = false;
        c.sigmoid = 2.0;
        assert!(c.validate().is_err());
        let c = Config {
            min_data_in_leaf: 0,
            min_sum_hessian_in_leaf: 0.0,
            ..Config::default()
        };
        assert!(c.validate().is_err());
        let c = Config {
            categorical_features: vec![0, 2],
            ..Config::default()
        };
        c.validate().unwrap();
        for categorical_features in [vec![0, 0], vec![usize::MAX]] {
            let c = Config {
                categorical_features,
                ..Config::default()
            };
            assert!(matches!(c.validate(), Err(Error::InvalidInput(_))));
        }
        let c = Config {
            metrics: vec!["unknown_metric".into()],
            ..Config::default()
        };
        assert!(matches!(c.validate(), Err(Error::Unsupported(_))));
        let c = Config {
            metrics: vec!["RMSE".into()],
            ..Config::default()
        };
        assert!(matches!(c.validate(), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn extended_source_defaults_and_aliases() {
        let c = Config::default();
        assert_eq!(c.num_class, 1);
        assert!(!c.reg_sqrt);
        assert_eq!(c.alpha, 0.9);
        assert_eq!(c.fair_c, 1.0);
        assert_eq!(c.poisson_max_delta_step, 0.7);
        assert_eq!(c.tweedie_variance_power, 1.5);
        assert_eq!(c.objective_seed, 5);
        assert_eq!(c.lambdarank_truncation_level, 30);
        assert!(c.lambdarank_norm);
        assert!(c.label_gain.is_empty());
        assert_eq!(c.lambdarank_position_bias_regularization, 0.0);
        assert!(c.eval_at.is_empty());
        assert_eq!(c.multi_error_top_k, 1);
        assert!(c.auc_mu_weights.is_empty());
        assert!(c.max_bin_by_feature.is_empty());
        assert!(c.forced_bins.is_empty());
        assert!(!c.linear_tree);
        assert_eq!(c.linear_lambda, 0.0);
        assert_eq!(c.max_cat_threshold, 32);
        assert_eq!(c.max_cat_to_onehot, 4);
        assert_eq!(c.cat_l2, 10.0);
        assert_eq!(c.cat_smooth, 10.0);
        assert_eq!(c.min_data_per_group, 100);
        for (name, objective) in [
            ("mae", Objective::RegressionL1),
            ("quantile", Objective::Quantile),
            ("huber", Objective::Huber),
            ("fair", Objective::Fair),
            ("poisson", Objective::Poisson),
            ("mean_absolute_percentage_error", Objective::Mape),
            ("gamma", Objective::Gamma),
            ("tweedie", Objective::Tweedie),
            ("softmax", Objective::Multiclass),
            ("ova", Objective::MulticlassOva),
            ("xentropy", Objective::CrossEntropy),
            ("xentlambda", Objective::CrossEntropyLambda),
            ("lambdarank", Objective::LambdaRank),
            ("xe_ndcg_mart", Objective::RankXendcg),
            ("none", Objective::Custom),
        ] {
            let classes = if objective.is_multiclass() { "3" } else { "1" };
            let actual =
                Config::from_pairs([("objective", name), ("num_classes", classes)]).unwrap();
            assert_eq!(actual.objective, objective);
        }
        for alias in ["ndcg_eval_at", "ndcg_at", "map_eval_at", "map_at"] {
            assert_eq!(
                Config::from_pairs([(alias, "5,1,3")]).unwrap().eval_at,
                [1, 3, 5]
            );
        }
        let c =
            Config::from_pairs([("linear_trees", "true"), ("max_bin_by_feature", "2,31")]).unwrap();
        assert!(c.linear_tree);
        assert_eq!(c.max_bin_by_feature, [2, 31]);
    }

    #[test]
    fn extended_source_domains_and_capability_checks() {
        for (key, value) in [
            ("num_class", "0"),
            ("num_class", "65537"),
            ("objective", "multiclass"),
            ("num_class", "2"),
            ("alpha", "0"),
            ("fair_c", "0"),
            ("poisson_max_delta_step", "0"),
            ("tweedie_variance_power", "0.999"),
            ("tweedie_variance_power", "2"),
            ("lambdarank_truncation_level", "0"),
            ("multi_error_top_k", "0"),
            ("lambdarank_position_bias_regularization", "-1"),
            ("eval_at", "0"),
            ("eval_at", "1,"),
            ("label_gain", "0,NaN"),
            ("max_bin_by_feature", "1"),
            ("max_cat_threshold", "0"),
            ("max_cat_to_onehot", "0"),
            ("min_data_per_group", "0"),
            ("cat_smooth", "-1"),
            ("cat_l2", "NaN"),
            ("linear_lambda", "NaN"),
            ("linear_lambda", "-1"),
        ] {
            assert!(Config::from_pairs([(key, value)]).is_err(), "{key}={value}");
        }
        Config::from_pairs([("objective", "huber"), ("alpha", "2")]).unwrap();
        assert!(Config::from_pairs([("objective", "quantile"), ("alpha", "1")]).is_err());
        for objective in [
            "multiclass",
            "cross_entropy",
            "cross_entropy_lambda",
            "lambdarank",
            "rank_xendcg",
            "fair",
        ] {
            let classes = if objective == "multiclass" { "3" } else { "1" };
            assert!(
                Config::from_pairs([
                    ("objective", objective),
                    ("num_class", classes),
                    ("scale_pos_weight", "2")
                ])
                .is_err()
            );
        }
        Config::from_pairs([("objective", "lambdarank"), ("sigmoid", "2")]).unwrap();
        Config::from_pairs([
            ("objective", "ova"),
            ("num_class", "3"),
            ("sigmoid", "2"),
            ("scale_pos_weight", "2"),
        ])
        .unwrap();
        assert!(
            Config::from_pairs([
                ("objective", "multiclass"),
                ("num_class", "65536"),
                ("auc_mu_weights", "1")
            ])
            .is_err()
        );
        let mut c = Config {
            forced_bins: vec![vec![1.0, 1.0]],
            ..Config::default()
        };
        assert!(c.validate().is_err());
        c.forced_bins = vec![vec![], vec![1.0, 2.0]];
        c.validate().unwrap();
        let c = Config::from_pairs([("seed", "42")]).unwrap();
        let mut random = crate::random::Random::new(42);
        for _ in 0..4 {
            random.next_short(0, 32767).unwrap();
        }
        assert_eq!(
            c.objective_seed,
            random.next_short(0, 32767).unwrap() as u32
        );
        assert_eq!(
            Config::from_pairs([("seed", "42"), ("objective_seed", "7")])
                .unwrap()
                .objective_seed,
            7
        );
    }

    #[test]
    fn categorical_column_indexes_for_dataset_binning() {
        for key in [
            "categorical_feature",
            "cat_feature",
            "categorical_column",
            "cat_column",
            "categorical_features",
        ] {
            let c = Config::from_pairs([(key, " 4, 0,2 ")]).unwrap();
            assert_eq!(c.categorical_features, [4, 0, 2]);
            c.validate().unwrap();
            assert!(
                Config::from_pairs([(key, "")])
                    .unwrap()
                    .categorical_features
                    .is_empty()
            );
        }
        for value in [
            "0,0",
            "0,",
            ",0",
            "0,,1",
            "-1",
            "1.5",
            "2147483648",
            "18446744073709551616",
            "garbage",
        ] {
            assert!(
                matches!(
                    Config::from_pairs([("categorical_feature", value)]),
                    Err(Error::InvalidInput(_))
                ),
                "{value}"
            );
        }
    }

    #[test]
    fn unsupported_features_are_never_silently_accepted() {
        for (key, value) in [
            ("objective", "not_an_objective"),
            ("device", "gpu"),
            ("tree", "data"),
            ("is_enable_sparse", "false"),
            ("forcedbins_filename", "bins.json"),
            ("early_stopping_round", "0"),
            ("categorical_feature", "name:category"),
            ("cat_feature", "auto"),
            ("categorical_features", "name:a,b"),
            ("metric", "unknown_metric"),
            ("unknown_option", "0"),
        ] {
            match Config::from_pairs([(key, value)]) {
                Err(Error::Unsupported(message)) => assert!(!message.is_empty()),
                other => panic!("{key}={value}: {other:?}"),
            }
        }
        Config::from_pairs([
            ("deterministic", "true"),
            ("force_col_wise", "true"),
            ("num_threads", "1"),
        ])
        .unwrap();
        for key in [
            "deterministic",
            "force_col_wise",
            "force_row_wise",
            "enable_bundle",
        ] {
            Config::from_pairs([(key, "false")]).unwrap();
        }
        for metric in [",", "l2,", ",l2", "l2,,rmse"] {
            assert!(matches!(
                Config::from_pairs([("metric", metric)]),
                Err(Error::InvalidInput(_))
            ));
        }
    }
}
