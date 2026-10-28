//! Checked CPU metric registry, objective output conversion, and aggregation.
//!
//! Ported from LightGBM src/metric/ at commit
//! 5d84f3f222b41409a6983d2ac90513ad9702fa9e (MIT).
//! Copyright (c) 2016-2026 Microsoft Corporation and The LightGBM developers.
//! See THIRD_PARTY_NOTICES.md.

mod discrimination;
mod rank;

use crate::objective::{
    EPSILON, checked_len, finite, reserved, softmax, sum_weights, validate_classes,
    validate_groups, validate_metadata, validate_probabilities, validate_scores, zeros,
};
use crate::{Config, Dataset, Error, Objective, Result};

/// A named metric with its optimization direction.
#[derive(Clone, Debug, PartialEq)]
pub struct MetricValue {
    pub name: String,
    pub value: f64,
    pub higher_is_better: bool,
}

/// Evaluate raw class-major scores. Empty metrics select the objective default;
/// `custom` disables built-in metrics. Ranking metrics require dataset groups.
pub fn evaluate(
    config: &Config,
    labels: &[f32],
    weights: Option<&[f32]>,
    scores: &[f64],
) -> Result<Vec<MetricValue>> {
    evaluate_input(config, labels, weights, None, scores)
}

/// Evaluate using dataset labels, weights, and nonnegative query-group lengths.
/// Scores already contain any dataset initial scores; they are not added again.
pub fn evaluate_dataset(
    config: &Config,
    dataset: &Dataset,
    scores: &[f64],
) -> Result<Vec<MetricValue>> {
    evaluate_input(
        config,
        dataset.labels(),
        dataset.weights(),
        dataset.groups(),
        scores,
    )
}

fn evaluate_input(
    config: &Config,
    labels: &[f32],
    weights: Option<&[f32]>,
    groups: Option<&[usize]>,
    scores: &[f64],
) -> Result<Vec<MetricValue>> {
    config.validate()?;
    validate_metadata(labels, weights)?;
    validate_scores(scores, labels.len(), config.num_class)?;
    if let Some(groups) = groups {
        validate_groups(groups, labels.len())?;
    }
    let default = match config.objective {
        Objective::Regression => "l2",
        Objective::RegressionL1 => "l1",
        Objective::Binary => "binary_logloss",
        Objective::Multiclass | Objective::MulticlassOva => "multi_logloss",
        Objective::LambdaRank | Objective::RankXendcg => "ndcg",
        objective => objective.name(),
    };
    let defaults = [default.to_owned()];
    let names = if config.metrics.is_empty() {
        &defaults[..]
    } else {
        &config.metrics
    };
    let result_count = names.iter().try_fold(0usize, |total, name| {
        let count = match name.as_str() {
            "custom" => 0,
            "ndcg" | "map" => {
                if config.eval_at.is_empty() {
                    5
                } else {
                    config.eval_at.len()
                }
            }
            _ => 1,
        };
        total
            .checked_add(count)
            .ok_or_else(|| Error::InvalidInput("metric output count overflow".into()))
    })?;
    let mut output = reserved(result_count)?;
    for name in names {
        if name == "custom" {
            continue;
        }
        if matches!(name.as_str(), "auc_mu" | "multi_error" | "multi_logloss") {
            if config.num_class < 2 {
                return invalid("multiclass metrics require at least two classes");
            }
            validate_classes(labels, config.num_class)?;
            if name == "auc_mu" {
                output.push(metric_value(
                    name,
                    discrimination::auc_mu(config, labels, weights, scores)?,
                    true,
                )?);
            } else {
                let value = multi_loss(config, labels, weights, scores, name)?;
                let output_name = if name == "multi_error" && config.multi_error_top_k != 1 {
                    format!("multi_error@{}", config.multi_error_top_k)
                } else {
                    name.clone()
                };
                output.push(metric_value(&output_name, value, false)?);
            }
            continue;
        }
        if config.num_class != 1 {
            return invalid("scalar metrics cannot evaluate multiclass score matrices");
        }
        if matches!(name.as_str(), "ndcg" | "map") {
            output.extend(rank::evaluate(
                config, labels, weights, groups, scores, name,
            )?);
            continue;
        }
        if matches!(name.as_str(), "auc" | "average_precision") {
            output.push(metric_value(
                name,
                discrimination::binary_curve(labels, weights, scores, name == "average_precision")?,
                true,
            )?);
            continue;
        }
        if matches!(
            name.as_str(),
            "cross_entropy" | "cross_entropy_lambda" | "kullback_leibler"
        ) {
            validate_probabilities(labels)?;
        }
        if matches!(name.as_str(), "gamma" | "gamma_deviance") && labels.iter().any(|&y| y <= 0.0) {
            return invalid("Gamma metrics require strictly positive labels");
        }
        if name == "cross_entropy_lambda" && weights.is_some_and(|w| w.contains(&0.0)) {
            return invalid("cross_entropy_lambda metric requires strictly positive weights");
        }
        let mut sum = 0.0;
        let mut entropy = 0.0;
        for (i, (&label, &raw)) in labels.iter().zip(scores).enumerate() {
            let score = convert_scalar(config, raw);
            let weight = weights.map_or(1.0, |w| f64::from(w[i]));
            let loss = point_loss(config, label, score, name, weight)?;
            sum += if name == "cross_entropy_lambda" {
                loss
            } else {
                loss * weight
            };
            if name == "kullback_leibler" {
                let p = f64::from(label);
                let q = 1.0 - p;
                entropy += (if p > 0.0 { p * p.ln() } else { 0.0 }
                    + if q > 0.0 { q * q.ln() } else { 0.0 })
                    * weight;
            }
        }
        let total = sum_weights(weights, labels.len());
        let value = match name.as_str() {
            "rmse" => (sum / total).sqrt(),
            "gamma_deviance" => sum * 2.0,
            "cross_entropy_lambda" => sum / labels.len() as f64,
            "kullback_leibler" => entropy / total + sum / total,
            "r2" => {
                // The reference multiplies label and weight in label_t before promotion.
                let mean = labels
                    .iter()
                    .enumerate()
                    .map(|(i, &y)| weights.map_or(f64::from(y), |w| f64::from(y * w[i])))
                    .sum::<f64>()
                    / total;
                let variance = labels
                    .iter()
                    .enumerate()
                    .map(|(i, &y)| {
                        let diff = f64::from(y) - mean;
                        diff * diff * weights.map_or(1.0, |w| f64::from(w[i]))
                    })
                    .sum::<f64>();
                if variance.abs() < 1e-35_f32 as f64 {
                    if sum.abs() < 1e-35_f32 as f64 {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    1.0 - sum / variance
                }
            }
            _ => sum / total,
        };
        output.push(metric_value(name, value, name == "r2")?);
    }
    Ok(output)
}

fn point_loss(c: &Config, label: f32, score: f64, name: &str, weight: f64) -> Result<f64> {
    let y = f64::from(label);
    let diff = score - y;
    Ok(match name {
        "l2" | "rmse" | "r2" => diff * diff,
        "l1" => diff.abs(),
        "mape" => diff.abs() / f64::from(label.abs().max(1.0)),
        "quantile" => {
            let delta = y - score;
            if delta < 0.0 {
                (c.alpha - 1.0) * delta
            } else {
                c.alpha * delta
            }
        }
        "huber" => {
            if diff.abs() <= c.alpha {
                0.5 * diff * diff
            } else {
                c.alpha * (diff.abs() - 0.5 * c.alpha)
            }
        }
        "fair" => c.fair_c * diff.abs() - c.fair_c * c.fair_c * (diff.abs() / c.fair_c).ln_1p(),
        "poisson" => {
            let score = score.max(1e-10_f32 as f64);
            score - y * score.ln()
        }
        "gamma" => {
            let theta = -1.0 / score;
            let b = -(-theta).ln();
            // SafeLog(label) returns label_t; SafeLog(label / psi) returns double.
            let offset = y.ln() - f64::from(label.ln());
            -((y * theta - b) + offset)
        }
        "gamma_deviance" => {
            let ratio = y / (score + 1e-9);
            ratio - ratio.ln() - 1.0
        }
        "tweedie" => {
            let score = score.max(1e-10_f32 as f64);
            let rho = c.tweedie_variance_power;
            let a = y * ((1.0 - rho) * score.ln()).exp() / (1.0 - rho);
            let b = ((2.0 - rho) * score.ln()).exp() / (2.0 - rho);
            -a + b
        }
        "binary_logloss" => {
            let p = if label <= 0.0 { 1.0 - score } else { score };
            if p > EPSILON {
                -p.ln()
            } else {
                // The pinned factory selects std::log(float), then promotes.
                -f64::from((EPSILON as f32).ln())
            }
        }
        "binary_error" => f64::from(if score <= 0.5 {
            label > 0.0
        } else {
            label <= 0.0
        }),
        "cross_entropy" | "kullback_leibler" => xent(label, score),
        "cross_entropy_lambda" => xent(label, 1.0 - (-weight * score).exp()),
        _ => return Err(Error::Unsupported(format!("metric {name}"))),
    })
}

fn xent(label: f32, probability: f64) -> f64 {
    -(f64::from(label) * probability.max(1e-12).ln()
        + f64::from(1.0_f32 - label) * (1.0 - probability).max(1e-12).ln())
}

pub(super) fn metric_value(name: &str, value: f64, higher_is_better: bool) -> Result<MetricValue> {
    Ok(MetricValue {
        name: name.into(),
        value: finite(value, &format!("metric {name}"))?,
        higher_is_better,
    })
}

fn multi_loss(
    c: &Config,
    labels: &[f32],
    weights: Option<&[f32]>,
    scores: &[f64],
    name: &str,
) -> Result<f64> {
    let mut rec = zeros(c.num_class)?;
    let mut sum = 0.0;
    for (row, &label) in labels.iter().enumerate() {
        for class in 0..c.num_class {
            rec[class] = scores[class * labels.len() + row];
        }
        match c.objective {
            Objective::Multiclass => softmax(&mut rec),
            Objective::MulticlassOva => {
                for score in &mut rec {
                    *score = 1.0 / (1.0 + (-c.sigmoid * *score).exp());
                }
            }
            Objective::Custom => {}
            _ => return invalid("multiclass metric requires multiclass or custom objective"),
        }
        let target = rec[label as usize];
        let loss = if name == "multi_logloss" {
            if target > EPSILON {
                -target.ln()
            } else {
                // std::log(kEpsilon) selects the float overload in this factory.
                -f64::from((EPSILON as f32).ln())
            }
        } else {
            f64::from(rec.iter().filter(|&&v| v >= target).count() > c.multi_error_top_k)
        };
        sum += loss * weights.map_or(1.0, |w| f64::from(w[row]));
    }
    Ok(sum / sum_weights(weights, labels.len()))
}

fn convert_scalar(c: &Config, score: f64) -> f64 {
    match c.objective {
        Objective::Binary => 1.0 / (1.0 + (-c.sigmoid * score).exp()),
        Objective::CrossEntropy => 1.0 / (1.0 + (-score).exp()),
        Objective::CrossEntropyLambda => score.exp().ln_1p(),
        Objective::Poisson | Objective::Gamma | Objective::Tweedie => score.exp(),
        _ if c.reg_sqrt => {
            if score > 0.0 {
                score * score
            } else if score < 0.0 {
                -score * score
            } else {
                0.0
            }
        }
        _ => score,
    }
}

fn invalid<T>(message: &str) -> Result<T> {
    Err(Error::InvalidInput(message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_losses_and_disabled_metric() {
        let config = Config {
            metrics: vec!["l2".into(), "rmse".into()],
            ..Config::default()
        };
        let metrics = evaluate(&config, &[0.0, 2.0], Some(&[1.0, 3.0]), &[2.0, 4.0]).unwrap();
        assert_eq!(metrics[0].value, 4.0);
        assert_eq!(metrics[1].value, 2.0);
        let disabled = Config {
            metrics: vec!["custom".into()],
            ..config
        };
        assert!(
            evaluate(&disabled, &[0.0], None, &[0.0])
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn multiclass_logloss_clamp_uses_source_float_logarithm() {
        let config = Config {
            objective: Objective::Multiclass,
            num_class: 2,
            ..Config::default()
        };
        let metrics = evaluate(&config, &[0.0], None, &[-1000.0, 1000.0]).unwrap();
        assert_eq!(metrics[0].value, -f64::from((EPSILON as f32).ln()));
    }

    #[test]
    fn binary_logloss_is_finite_at_extreme_scores() {
        let config = Config {
            objective: Objective::Binary,
            ..Config::default()
        };
        let score = evaluate(&config, &[0.0, 1.0], None, &[1000.0, -1000.0]).unwrap();
        assert_eq!(score[0].value, -f64::from((EPSILON as f32).ln()));
        assert!(evaluate(&config, &[1.0], None, &[f64::NAN]).is_err());
        assert!(evaluate(&config, &[1.0], Some(&[0.0]), &[0.0]).is_err());
    }
}
