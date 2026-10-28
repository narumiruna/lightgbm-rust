//! Checked CPU objective state and default-reference precision boundaries.
//!
//! Ported from LightGBM src/objective/ at commit
//! 5d84f3f222b41409a6983d2ac90513ad9702fa9e (MIT).
//! Copyright (c) 2016-2026 Microsoft Corporation and The LightGBM developers.
//! See THIRD_PARTY_NOTICES.md.

mod rank;
pub(crate) use rank::{discount, ideal_dcg, label_gains, validate_rank_labels};
#[cfg(test)]
mod conformance;

use crate::{Config, Error, Objective, Result};

/// The reference declares this constant as a float, then promotes it.
pub(crate) const EPSILON: f64 = 1e-15_f32 as f64;
pub(crate) const MAX_ENTRIES: usize = 16 * 1024 * 1024;

/// Group entries are nonnegative sizes, not prefix boundaries. Empty queries
/// retain their indexes for metric denominators and ranking random seeds.
pub(crate) struct ObjectiveInput<'a> {
    pub labels: &'a [f32],
    pub weights: Option<&'a [f32]>,
    pub groups: Option<&'a [usize]>,
    pub positions: Option<&'a [i32]>,
}

/// Mutable ranking random and position state is separate from model scores.
pub(crate) struct ObjectiveState<'a> {
    config: Config,
    input: ObjectiveInput<'a>,
    labels: Vec<f32>,
    mape_weights: Vec<f32>,
    ranking: Option<rank::RankingState>,
}

impl<'a> ObjectiveState<'a> {
    pub(crate) fn new(config: &Config, input: ObjectiveInput<'a>) -> Result<Self> {
        config.validate()?;
        if let Some(groups) = input.groups {
            validate_groups(groups, input.labels.len())?;
            if config.bagging_by_query
                && config.bagging_freq > 0
                && (config.bagging_fraction < 1.0
                    || (config.objective == Objective::Binary
                        && (config.pos_bagging_fraction < 1.0
                            || config.neg_bagging_fraction < 1.0)
                        && input.labels.iter().any(|&label| label > 0.0)))
                && groups.len() > input.labels.len()
            {
                // Pinned bagging.hpp:176-181 allocates query-index and RNG buffers
                // by row count. Empty queries can make query indexes exceed them.
                return Err(Error::Unsupported(
                    "query bagging with more queries than rows exceeds pinned native query buffers"
                        .into(),
                ));
            }
            if config.objective.is_ranking() && input.weights.is_some() && groups.contains(&0) {
                return Err(Error::Unsupported(
                    "weights with empty queries have undefined native query weights (sum / 0)"
                        .into(),
                ));
            }
        }
        validate_metadata(input.labels, input.weights)?;
        checked_len(input.labels.len(), config.num_class)?;
        if input
            .positions
            .is_some_and(|p| p.len() != input.labels.len())
        {
            return invalid("positions must match label count");
        }
        let objective = config.objective;
        if objective.is_multiclass() {
            validate_classes(input.labels, config.num_class)?;
        }
        if matches!(
            objective,
            Objective::CrossEntropy | Objective::CrossEntropyLambda
        ) {
            validate_probabilities(input.labels)?;
        }
        if objective == Objective::CrossEntropyLambda
            && input.weights.is_some_and(|w| w.contains(&0.0))
        {
            return invalid("cross_entropy_lambda requires strictly positive weights");
        }
        if matches!(
            objective,
            Objective::Poisson | Objective::Gamma | Objective::Tweedie
        ) && (input.labels.iter().any(|&y| y < 0.0) || !input.labels.iter().any(|&y| y > 0.0))
        {
            return invalid(
                "log-link regression requires nonnegative labels and a positive label sum",
            );
        }
        let mut labels = reserved(input.labels.len())?;
        labels.extend(input.labels.iter().map(|&y| {
            if config.reg_sqrt {
                sign(f64::from(y)) as f32 * y.abs().sqrt()
            } else {
                y
            }
        }));
        let mut mape_weights = Vec::new();
        if objective == Objective::Mape {
            mape_weights = reserved(labels.len())?;
            mape_weights.extend(
                labels.iter().enumerate().map(|(i, &y)| {
                    1.0_f32 / y.abs().max(1.0) * input.weights.map_or(1.0, |w| w[i])
                }),
            );
        }
        let ranking = if objective.is_ranking() {
            Some(rank::RankingState::new(config, &input)?)
        } else {
            None
        };
        Ok(Self {
            config: config.clone(),
            input,
            labels,
            mape_weights,
            ranking,
        })
    }

    pub(crate) fn reset_learning_rate(&mut self, rate: f64) {
        self.config.learning_rate = rate;
    }

    pub(crate) fn num_classes(&self) -> usize {
        self.config.num_class
    }

    pub(crate) fn initial_score(&self, class: usize) -> Result<f64> {
        self.check_class(class)?;
        let c = &self.config;
        let sum_weights = sum_weights(self.input.weights, self.labels.len());
        let weighted_mean = |binary: bool| -> f64 {
            self.labels
                .iter()
                .enumerate()
                .map(|(i, &y)| {
                    let y = if binary {
                        f64::from(if c.objective == Objective::MulticlassOva {
                            y as usize == class
                        } else {
                            y > 0.0
                        })
                    } else {
                        f64::from(y)
                    };
                    y * self.input.weights.map_or(1.0, |w| f64::from(w[i]))
                })
                .sum::<f64>()
                / sum_weights
        };
        let output = match c.objective {
            Objective::Binary | Objective::MulticlassOva => logit(weighted_mean(true)) / c.sigmoid,
            Objective::Multiclass => self.class_probability(class).max(EPSILON).ln(),
            Objective::CrossEntropy => logit(weighted_mean(false)),
            Objective::CrossEntropyLambda => weighted_mean(false).exp_m1().ln(),
            Objective::LambdaRank | Objective::RankXendcg => 0.0,
            Objective::Poisson | Objective::Gamma | Objective::Tweedie => weighted_mean(false).ln(),
            Objective::RegressionL1 | Objective::Quantile | Objective::Mape => {
                let alpha = if c.objective == Objective::Quantile {
                    f64::from(c.alpha as f32)
                } else {
                    0.5
                };
                let weights = if c.objective == Objective::Mape {
                    Some(self.mape_weights.as_slice())
                } else {
                    self.input.weights
                };
                let mut values = reserved(self.labels.len())?;
                values.extend(self.labels.iter().map(|&y| f64::from(y)));
                percentile(&values, weights, alpha, true)?
            }
            Objective::Regression | Objective::Huber | Objective::Fair => weighted_mean(false),
            Objective::Custom => 0.0,
        };
        finite(output, "initial score")
    }

    pub(crate) fn needs_training(&self, class: usize) -> bool {
        if class >= self.num_classes() {
            return false;
        }
        match self.config.objective {
            Objective::Binary | Objective::MulticlassOva => {
                let positives = self.positive_count(class);
                positives > 0 && positives < self.labels.len()
            }
            Objective::Multiclass => {
                let p = self.class_probability(class);
                p.abs() > EPSILON && p.abs() < 1.0 - EPSILON
            }
            _ => true,
        }
    }

    pub(crate) fn gradients(&mut self, scores: &[f64]) -> Result<(Vec<f32>, Vec<f32>)> {
        if self.config.objective == Objective::Custom {
            return Err(Error::Unsupported(
                "custom objective requires a gradient callback".into(),
            ));
        }
        validate_scores(scores, self.labels.len(), self.num_classes())?;
        let mut g: Vec<f32> = zeros(scores.len())?;
        let mut h: Vec<f32> = zeros(scores.len())?;
        if let Some(ranking) = &mut self.ranking {
            ranking.gradients(&self.config, &self.input, scores, &mut g, &mut h, None)?;
        } else if self.config.objective == Objective::Multiclass {
            let n = self.labels.len();
            let k = self.num_classes();
            let factor = k as f64 / (k as f64 - 1.0);
            let mut probabilities = zeros(k)?;
            for row in 0..n {
                for class in 0..k {
                    probabilities[class] = scores[class * n + row];
                }
                softmax(&mut probabilities);
                let weight = self.input.weights.map_or(1.0, |w| f64::from(w[row]));
                for class in 0..k {
                    let p = probabilities[class];
                    g[class * n + row] =
                        ((p - f64::from(self.labels[row] as usize == class)) * weight) as f32;
                    h[class * n + row] = (factor * p * (1.0 - p) * weight) as f32;
                }
            }
        } else {
            for class in 0..self.num_classes() {
                if !self.needs_training(class) {
                    continue;
                }
                let class_weights = self.class_weights(class);
                for (row, &label) in self.labels.iter().enumerate() {
                    let index = class * self.labels.len() + row;
                    let (gradient, hessian) =
                        self.point_derivatives(label, scores[index], row, class, class_weights);
                    g[index] = gradient;
                    h[index] = hessian;
                }
            }
        }
        if g.iter().any(|v| !v.is_finite()) || h.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return invalid("objective produced nonfinite gradients or invalid Hessians");
        }
        Ok((g, h))
    }

    /// Update selected query derivatives only. Other entries retain their previous
    /// values, as required by the native position-bias update over all rows.
    pub(crate) fn gradients_for_queries(
        &mut self,
        scores: &[f64],
        queries: &[usize],
        g: &mut [f32],
        h: &mut [f32],
    ) -> Result<()> {
        validate_scores(scores, self.labels.len(), self.num_classes())?;
        if g.len() != scores.len() || h.len() != scores.len() {
            return invalid("query derivative buffers differ from score layout");
        }
        let groups = self
            .input
            .groups
            .ok_or_else(|| Error::InvalidInput("query sampling requires groups".into()))?;
        if queries.iter().any(|&q| q >= groups.len()) || queries.windows(2).any(|q| q[0] >= q[1]) {
            return invalid("sampled queries must be distinct ordered query indexes");
        }
        if let Some(ranking) = &mut self.ranking {
            ranking.gradients(&self.config, &self.input, scores, g, h, Some(queries))
        } else {
            let (next_g, next_h) = self.gradients(scores)?;
            g.copy_from_slice(&next_g);
            h.copy_from_slice(&next_h);
            Ok(())
        }
    }

    fn point_derivatives(
        &self,
        label: f32,
        score: f64,
        row: usize,
        class: usize,
        class_weights: [f64; 2],
    ) -> (f32, f32) {
        let c = &self.config;
        let weight32 = self.input.weights.map_or(1.0, |w| w[row]);
        let weight = f64::from(weight32);
        let y = f64::from(label);
        let diff = score - y;
        let (g, h) = match c.objective {
            Objective::Regression => return ((diff as f32) * weight32, weight32),
            Objective::RegressionL1 => (sign(diff) * weight, weight),
            Objective::Quantile => {
                let alpha = c.alpha as f32;
                return (
                    if (diff as f32) >= 0.0 {
                        (1.0_f32 - alpha) * weight32
                    } else {
                        -alpha * weight32
                    },
                    weight32,
                );
            }
            Objective::Huber => (
                if diff.abs() <= c.alpha {
                    diff * weight
                } else {
                    sign(diff) * weight * c.alpha
                },
                weight,
            ),
            Objective::Fair => {
                let denominator = diff.abs() + c.fair_c;
                (
                    c.fair_c * diff / denominator * weight,
                    c.fair_c * c.fair_c / (denominator * denominator) * weight,
                )
            }
            Objective::Poisson => (
                (score.exp() - y) * weight,
                score.exp() * c.poisson_max_delta_step.exp() * weight,
            ),
            Objective::Mape => (sign(diff) * f64::from(self.mape_weights[row]), weight),
            Objective::Gamma => (
                (1.0 - y * (-score).exp()) * weight,
                y * (-score).exp() * weight,
            ),
            Objective::Tweedie => {
                let rho = c.tweedie_variance_power;
                let a = ((1.0 - rho) * score).exp();
                let b = ((2.0 - rho) * score).exp();
                (
                    (-y * a + b) * weight,
                    (-y * (1.0 - rho) * a + (2.0 - rho) * b) * weight,
                )
            }
            Objective::Binary | Objective::MulticlassOva => {
                let positive = if c.objective == Objective::Binary {
                    label > 0.0
                } else {
                    label as usize == class
                };
                let sign = if positive { 1.0 } else { -1.0 };
                let response = -sign * c.sigmoid / (1.0 + (sign * c.sigmoid * score).exp());
                let absolute = response.abs();
                let cw = class_weights[usize::from(positive)];
                (
                    response * cw * weight,
                    absolute * (c.sigmoid - absolute) * cw * weight,
                )
            }
            Objective::CrossEntropy => {
                if score > -37.0 {
                    let exp = (-score).exp();
                    (
                        (f64::from(1.0_f32 - label) - y * exp) / (1.0 + exp) * weight,
                        exp / ((1.0 + exp) * (1.0 + exp)) * weight,
                    )
                } else {
                    let exp = score.exp();
                    ((exp - y) * weight, exp * weight)
                }
            }
            Objective::CrossEntropyLambda => {
                if self.input.weights.is_none() {
                    let z = 1.0 / (1.0 + (-score).exp());
                    (z - y, z * (1.0 - z))
                } else {
                    let epf = score.exp();
                    let hhat = epf.ln_1p();
                    let z = 1.0 - (-weight * hhat).exp();
                    let enf = 1.0 / epf;
                    let c = 1.0 / (1.0 - z);
                    let d = 1.0 + epf;
                    let a = weight * epf / (d * d);
                    let d = c - 1.0;
                    let b = c / (d * d) * (1.0 + weight * epf - c);
                    ((1.0 - y / z) * weight / (1.0 + enf), a * (1.0 + y * b))
                }
            }
            // These variants are handled before the point-wise loop.
            Objective::Multiclass
            | Objective::LambdaRank
            | Objective::RankXendcg
            | Objective::Custom => unreachable!(),
        };
        (g as f32, h as f32)
    }

    /// Check the full score layout once per tree, then renew each leaf in order.
    /// Output entries contain Newton values on entry and renewed values on return.
    pub(crate) fn renew_leaf_outputs(
        &self,
        class: usize,
        leaf_rows: &[impl AsRef<[usize]>],
        scores: &[f64],
        outputs: &mut [f64],
    ) -> Result<()> {
        self.check_class(class)?;
        validate_scores(scores, self.labels.len(), self.num_classes())?;
        if leaf_rows.len() != outputs.len() {
            return invalid("leaf row and output counts must match");
        }
        for (rows, output) in leaf_rows.iter().zip(outputs) {
            *output = self.renew_leaf_output_validated(class, rows.as_ref(), scores, *output)?;
        }
        Ok(())
    }

    // Only the checked batch entry above calls this method.
    fn renew_leaf_output_validated(
        &self,
        class: usize,
        rows: &[usize],
        scores: &[f64],
        newton: f64,
    ) -> Result<f64> {
        if rows.is_empty() || rows.iter().any(|&row| row >= self.labels.len()) {
            return invalid("leaf rows must be nonempty valid row indexes");
        }
        finite(newton, "Newton leaf output")?;
        if !matches!(
            self.config.objective,
            Objective::RegressionL1 | Objective::Quantile | Objective::Mape
        ) {
            return Ok(newton);
        }
        let mut residuals = reserved(rows.len())?;
        residuals.extend(
            rows.iter()
                .map(|&row| f64::from(self.labels[row]) - scores[class * self.labels.len() + row]),
        );
        let source_weights = if self.config.objective == Objective::Mape {
            Some(self.mape_weights.as_slice())
        } else {
            self.input.weights
        };
        let weights = source_weights
            .map(|w| -> Result<Vec<f32>> {
                let mut selected = reserved(rows.len())?;
                selected.extend(rows.iter().map(|&row| w[row]));
                Ok(selected)
            })
            .transpose()?;
        let alpha = if self.config.objective == Objective::Quantile {
            f64::from(self.config.alpha as f32)
        } else {
            0.5
        };
        finite(
            percentile(&residuals, weights.as_deref(), alpha, false)?,
            "renewed leaf output",
        )
    }

    pub(crate) fn model_objective(&self) -> String {
        let name = self.config.objective.name();
        match self.config.objective {
            Objective::Binary => format!("{name} sigmoid:{}", cpp_number(self.config.sigmoid)),
            Objective::Multiclass => format!("{name} num_class:{}", self.num_classes()),
            Objective::MulticlassOva => format!(
                "{name} num_class:{} sigmoid:{}",
                self.num_classes(),
                cpp_number(self.config.sigmoid)
            ),
            _ if self.config.reg_sqrt => format!("{name} sqrt"),
            _ => name.into(),
        }
    }

    fn check_class(&self, class: usize) -> Result<()> {
        if class >= self.num_classes() {
            invalid("class index is out of range")
        } else {
            Ok(())
        }
    }

    fn positive_count(&self, class: usize) -> usize {
        self.labels
            .iter()
            .filter(|&&y| {
                if self.config.objective == Objective::Binary {
                    y > 0.0
                } else {
                    y as usize == class
                }
            })
            .count()
    }

    fn class_probability(&self, class: usize) -> f64 {
        self.labels
            .iter()
            .enumerate()
            .filter(|&(_, &y)| y as usize == class)
            .map(|(i, _)| self.input.weights.map_or(1.0, |w| f64::from(w[i])))
            .sum::<f64>()
            / sum_weights(self.input.weights, self.labels.len())
    }

    fn class_weights(&self, class: usize) -> [f64; 2] {
        let positives = self.positive_count(class);
        let negatives = self.labels.len() - positives;
        let mut weights = [1.0, 1.0];
        if self.config.is_unbalance && positives > 0 && negatives > 0 {
            if positives > negatives {
                weights[0] = positives as f64 / negatives as f64;
            } else {
                weights[1] = negatives as f64 / positives as f64;
            }
        }
        weights[1] *= self.config.scale_pos_weight;
        weights
    }
}

/// Compute class-zero initialization. Ranking requires `ObjectiveInput` groups.
pub fn initial_score(config: &Config, labels: &[f32], weights: Option<&[f32]>) -> Result<f64> {
    ObjectiveState::new(
        config,
        ObjectiveInput {
            labels,
            weights,
            groups: None,
            positions: None,
        },
    )?
    .initial_score(0)
}

/// Compute class-major default-reference `f32` gradients and Hessians.
/// Ranking metadata is not fabricated by this wrapper.
pub fn gradients(
    config: &Config,
    labels: &[f32],
    weights: Option<&[f32]>,
    scores: &[f64],
) -> Result<(Vec<f32>, Vec<f32>)> {
    ObjectiveState::new(
        config,
        ObjectiveInput {
            labels,
            weights,
            groups: None,
            positions: None,
        },
    )?
    .gradients(scores)
}

pub(crate) fn validate_metadata(labels: &[f32], weights: Option<&[f32]>) -> Result<()> {
    if labels.is_empty() || labels.len() > MAX_ENTRIES || labels.iter().any(|v| !v.is_finite()) {
        return invalid("labels must be nonempty, finite, and contain at most 16 Mi entries");
    }
    if let Some(weights) = weights
        && (weights.len() != labels.len()
            || weights.iter().any(|v| !v.is_finite() || *v < 0.0)
            || !weights.iter().any(|v| *v > 0.0))
    {
        return invalid(
            "weights must be finite, nonnegative, match labels, and have positive total weight",
        );
    }
    Ok(())
}

pub(crate) fn validate_groups(groups: &[usize], rows: usize) -> Result<()> {
    if groups.is_empty() || groups.len() > MAX_ENTRIES {
        return invalid("query groups must be nonempty and contain at most 16 Mi entries");
    }
    let total = groups.iter().try_fold(0usize, |sum, &n| sum.checked_add(n));
    if total != Some(rows) {
        return invalid("nonnegative group sizes must sum to label count");
    }
    Ok(())
}

pub(crate) fn validate_classes(labels: &[f32], classes: usize) -> Result<()> {
    if labels
        .iter()
        .any(|&y| y < 0.0 || y.fract() != 0.0 || f64::from(y) >= classes as f64)
    {
        return invalid("multiclass labels must be integers in [0, num_class)");
    }
    Ok(())
}

pub(crate) fn validate_probabilities(labels: &[f32]) -> Result<()> {
    if labels.iter().any(|y| !(0.0..=1.0).contains(y)) {
        invalid("probability labels must be in [0, 1]")
    } else {
        Ok(())
    }
}

pub(crate) fn checked_len(rows: usize, classes: usize) -> Result<usize> {
    rows.checked_mul(classes)
        .filter(|&n| n <= MAX_ENTRIES)
        .ok_or_else(|| Error::InvalidInput("score array exceeds 16 Mi entries".into()))
}

pub(crate) fn validate_scores(scores: &[f64], rows: usize, classes: usize) -> Result<()> {
    if scores.len() != checked_len(rows, classes)? || scores.iter().any(|v| !v.is_finite()) {
        return invalid("raw scores must be finite and match the class-major shape");
    }
    Ok(())
}

pub(crate) fn reserved<T>(len: usize) -> Result<Vec<T>> {
    if len > MAX_ENTRIES {
        return invalid("numeric allocation exceeds 16 Mi entries");
    }
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|e| Error::InvalidInput(format!("numeric allocation failed: {e}")))?;
    Ok(v)
}

pub(crate) fn zeros<T: Default + Clone>(len: usize) -> Result<Vec<T>> {
    let mut v = reserved(len)?;
    v.resize(len, T::default());
    Ok(v)
}

pub(crate) fn sum_weights(weights: Option<&[f32]>, rows: usize) -> f64 {
    weights.map_or(rows as f64, |w| w.iter().map(|&v| f64::from(v)).sum())
}

pub(crate) fn softmax(values: &mut [f64]) {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum;
    }
}

pub(crate) fn finite(value: f64, name: &str) -> Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        invalid(&format!("{name} is not finite"))
    }
}

fn sign(value: f64) -> f64 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}
fn logit(p: f64) -> f64 {
    let p = p.clamp(EPSILON, 1.0 - EPSILON);
    (p / (1.0 - p)).ln()
}

fn percentile(
    values: &[f64],
    weights: Option<&[f32]>,
    alpha: f64,
    f32_boundary: bool,
) -> Result<f64> {
    let round = |v: f64| if f32_boundary { f64::from(v as f32) } else { v };
    if values.len() == 1 {
        return Ok(values[0]);
    }
    let mut indexes = reserved(values.len())?;
    indexes.extend(0..values.len());
    indexes.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    if let Some(weights) = weights {
        let mut cdf = reserved(values.len())?;
        let mut total = 0.0;
        for &index in &indexes {
            total += f64::from(weights[index]);
            cdf.push(total);
        }
        let threshold = total * alpha;
        let pos = cdf
            .partition_point(|&v| v <= threshold)
            .min(values.len() - 1);
        if pos == 0 || pos == values.len() - 1 {
            return Ok(values[indexes[pos]]);
        }
        let v1 = values[indexes[pos - 1]];
        let v2 = values[indexes[pos]];
        let width = cdf[pos] - cdf[pos - 1];
        if width >= 1.0 {
            Ok(round(
                (threshold - cdf[pos - 1]) / width * round(v2 - v1) + v1,
            ))
        } else {
            Ok(v1)
        }
    } else {
        let float_pos = (values.len() - 1) as f64 * (1.0 - alpha);
        let pos = float_pos as usize + 1;
        if pos >= values.len() {
            return Ok(values[indexes[0]]);
        }
        let v1 = values[indexes[values.len() - pos]];
        let v2 = values[indexes[values.len() - pos - 1]];
        Ok(round(v1 - round(v1 - v2) * (float_pos - (pos - 1) as f64)))
    }
}

// C++ defaultfloat writes six significant digits in objective metadata.
fn cpp_number(value: f64) -> String {
    let scientific = format!("{value:.5e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    if !(-4..6).contains(&exponent) {
        let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
        format!("{mantissa}e{exponent:+03}")
    } else {
        let decimals = (5 - exponent).max(0) as usize;
        let result = format!("{value:.decimals$}");
        if decimals > 0 {
            result.trim_end_matches('0').trim_end_matches('.').into()
        } else {
            result
        }
    }
}

fn invalid<T>(message: &str) -> Result<T> {
    Err(Error::InvalidInput(message.into()))
}
