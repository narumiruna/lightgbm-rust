//! CPU query objectives and their persistent random and position-bias state.
//!
//! Ported from LightGBM src/objective/rank_objective.hpp at
//! 5d84f3f222b41409a6983d2ac90513ad9702fa9e (MIT).
//! Copyright (c) 2020-2026 Microsoft Corporation and The LightGBM developers.
//! See THIRD_PARTY_NOTICES.md.

use super::*;
use crate::random::Random;

pub(super) struct RankingState {
    gains: Vec<f64>,
    inverse_dcg: Vec<f64>,
    random: Vec<Random>,
    positions: Vec<usize>,
    biases: Vec<f32>,
}

impl RankingState {
    pub(super) fn new(config: &Config, input: &ObjectiveInput<'_>) -> Result<Self> {
        let groups = input
            .groups
            .ok_or_else(|| Error::InvalidInput("ranking requires query groups".into()))?;
        if groups.contains(&0) {
            if input.weights.is_some() {
                return Err(Error::Unsupported(
                    "weights with empty queries have undefined native query weights (sum / 0)"
                        .into(),
                ));
            }
            if config.objective == Objective::LambdaRank {
                // Pinned rank_objective.hpp:201 reads sorted_idx[0] for cnt=0.
                // Do not reproduce or execute this undefined native access.
                return Err(Error::Unsupported(
                    "LambdaRank empty queries are undefined in the pinned native objective (sorted_idx[0])".into(),
                ));
            }
        }
        let gains = label_gains(config)?;
        if config.objective == Objective::LambdaRank {
            validate_rank_labels(input.labels, &gains)?;
            if groups.iter().any(|&n| n > 10_000) {
                return invalid("LambdaRank queries cannot exceed 10000 rows");
            }
        } else if input
            .labels
            .iter()
            .any(|&y| y < 0.0 || y.fract() != 0.0 || y > 30.0)
        {
            return invalid("XE-NDCG labels must be integers in 0..=30");
        }
        let mut inverse_dcg = reserved(groups.len())?;
        let mut random = reserved(groups.len())?;
        let mut offset = 0;
        for (query, &size) in groups.iter().enumerate() {
            if config.objective == Objective::LambdaRank {
                let max = ideal_dcg(
                    &input.labels[offset..offset + size],
                    &gains,
                    config.lambdarank_truncation_level,
                )?;
                inverse_dcg.push(if max > 0.0 { 1.0 / max } else { max });
            }
            random.push(Random::new(
                config.objective_seed.wrapping_add(query as u32),
            ));
            offset += size;
        }
        let mut positions = Vec::new();
        let mut biases = Vec::new();
        if let Some(ids) = input.positions {
            let mut unique = reserved(ids.len())?;
            unique.extend_from_slice(ids);
            unique.sort_unstable();
            unique.dedup();
            positions = reserved(ids.len())?;
            positions.extend(ids.iter().map(|id| unique.binary_search(id).unwrap()));
            biases = zeros(unique.len())?;
        }
        Ok(Self {
            gains,
            inverse_dcg,
            random,
            positions,
            biases,
        })
    }

    pub(super) fn gradients(
        &mut self,
        config: &Config,
        input: &ObjectiveInput<'_>,
        scores: &[f64],
        g: &mut [f32],
        h: &mut [f32],
        queries: Option<&[usize]>,
    ) -> Result<()> {
        let mut start = 0;
        for (query, &size) in input.groups.unwrap().iter().enumerate() {
            let end = start + size;
            if queries.is_some_and(|q| q.binary_search(&query).is_err()) {
                start = end;
                continue;
            }
            g[start..end].fill(0.0);
            h[start..end].fill(0.0);
            let mut adjusted = reserved(size)?;
            adjusted.extend((start..end).map(|i| {
                scores[i]
                    + if self.positions.is_empty() {
                        0.0
                    } else {
                        f64::from(self.biases[self.positions[i]])
                    }
            }));
            if adjusted.iter().any(|v| !v.is_finite()) {
                return invalid("position-adjusted scores are not finite");
            }
            if config.objective == Objective::LambdaRank {
                self.lambda_query(
                    config,
                    query,
                    &input.labels[start..end],
                    &adjusted,
                    &mut g[start..end],
                    &mut h[start..end],
                )?;
            } else {
                xe_query(
                    &mut self.random[query],
                    &input.labels[start..end],
                    &adjusted,
                    &mut g[start..end],
                    &mut h[start..end],
                )?;
            }
            if let Some(weights) = input.weights {
                for i in start..end {
                    g[i] *= weights[i];
                    h[i] *= weights[i];
                }
            }
            start = end;
        }
        if g.iter().any(|v| !v.is_finite()) || h.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return invalid("ranking produced nonfinite derivatives or negative Hessians");
        }
        if config.objective == Objective::LambdaRank && !self.positions.is_empty() {
            let mut first: Vec<f64> = zeros(self.biases.len())?;
            let mut second: Vec<f64> = zeros(self.biases.len())?;
            let mut counts: Vec<usize> = zeros(self.biases.len())?;
            for row in 0..scores.len() {
                let p = self.positions[row];
                first[p] -= f64::from(g[row]);
                second[p] -= f64::from(h[row]);
                counts[p] += 1;
            }
            let mut next_biases = reserved(self.biases.len())?;
            for p in 0..self.biases.len() {
                first[p] -= f64::from(self.biases[p])
                    * config.lambdarank_position_bias_regularization
                    * counts[p] as f64;
                second[p] -= config.lambdarank_position_bias_regularization * counts[p] as f64;
                let bias = (f64::from(self.biases[p])
                    + config.learning_rate * first[p] / (second[p].abs() + 0.001))
                    as f32;
                if !bias.is_finite() {
                    return invalid("position bias update is not finite");
                }
                next_biases.push(bias);
            }
            self.biases = next_biases;
        }
        Ok(())
    }

    fn lambda_query(
        &self,
        config: &Config,
        query: usize,
        labels: &[f32],
        scores: &[f64],
        g: &mut [f32],
        h: &mut [f32],
    ) -> Result<()> {
        let mut order = reserved(labels.len())?;
        order.extend(0..labels.len());
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        let best = scores[order[0]];
        let worst = scores[order[order.len() - 1]];
        let mut sum_lambdas = 0.0;
        for i in 0..(labels.len() - 1).min(config.lambdarank_truncation_level) {
            for j in i + 1..labels.len() {
                if labels[order[i]] == labels[order[j]] {
                    continue;
                }
                let (high_rank, low_rank) = if labels[order[i]] > labels[order[j]] {
                    (i, j)
                } else {
                    (j, i)
                };
                let high = order[high_rank];
                let low = order[low_rank];
                let delta_score = scores[high] - scores[low];
                let gap = self.gains[labels[high] as usize] - self.gains[labels[low] as usize];
                let discount = (discount(high_rank) - discount(low_rank)).abs();
                let mut delta_ndcg = gap * discount * self.inverse_dcg[query];
                if config.lambdarank_norm && best != worst {
                    delta_ndcg /= f64::from(0.01_f32) + delta_score.abs();
                }
                let mut lambda = table_sigmoid(delta_score, config.sigmoid);
                let mut hessian = lambda * (1.0 - lambda);
                lambda *= -config.sigmoid * delta_ndcg;
                hessian *= config.sigmoid * config.sigmoid * delta_ndcg;
                g[low] -= lambda as f32;
                h[low] += hessian as f32;
                g[high] += lambda as f32;
                h[high] += hessian as f32;
                sum_lambdas -= 2.0 * lambda;
            }
        }
        if config.lambdarank_norm && sum_lambdas > 0.0 {
            let norm = (1.0 + sum_lambdas).log2() / sum_lambdas;
            for i in 0..labels.len() {
                g[i] = (f64::from(g[i]) * norm) as f32;
                h[i] = (f64::from(h[i]) * norm) as f32;
            }
        }
        Ok(())
    }
}

fn table_sigmoid(score: f64, sigmoid: f64) -> f64 {
    // Compute exactly the selected entry of the reference's 2^20-entry table.
    let min = -50.0 / sigmoid / 2.0;
    let max = -min;
    let factor = (1024 * 1024) as f64 / (max - min);
    let index = if score <= min {
        0
    } else if score >= max {
        1024 * 1024 - 1
    } else {
        ((score - min) * factor) as usize
    };
    let entry_score = index as f64 / factor + min;
    1.0 / (1.0 + (entry_score * sigmoid).exp())
}

fn xe_query(
    random: &mut Random,
    labels: &[f32],
    scores: &[f64],
    g: &mut [f32],
    h: &mut [f32],
) -> Result<()> {
    if labels.len() <= 1 {
        return Ok(());
    }
    let mut rho = reserved(scores.len())?;
    rho.extend_from_slice(scores);
    softmax(&mut rho);
    let mut params = reserved(labels.len())?;
    let mut denominator = 0.0;
    for &label in labels {
        let value = (2.0_f64).powi(label as i32) - f64::from(random.next_float());
        params.push(value);
        denominator += value;
    }
    let inverse = 1.0 / denominator.max(EPSILON);
    let mut sum_l1 = 0.0;
    for i in 0..labels.len() {
        let term = -params[i] * inverse + rho[i];
        g[i] = term as f32;
        params[i] = term / (1.0 - rho[i]);
        sum_l1 += params[i];
    }
    let mut sum_l2 = 0.0;
    for i in 0..labels.len() {
        let term = rho[i] * (sum_l1 - params[i]);
        g[i] += term as f32;
        params[i] = term / (1.0 - rho[i]);
        sum_l2 += params[i];
    }
    for i in 0..labels.len() {
        g[i] += (rho[i] * (sum_l2 - params[i])) as f32;
        h[i] = (rho[i] * (1.0 - rho[i])) as f32;
    }
    Ok(())
}

pub(crate) fn label_gains(config: &Config) -> Result<Vec<f64>> {
    let mut gains = reserved(if config.label_gain.is_empty() {
        31
    } else {
        config.label_gain.len()
    })?;
    if config.label_gain.is_empty() {
        gains.extend((0..31).map(|i| ((1_u32 << i) - 1) as f64));
    } else {
        gains.extend_from_slice(&config.label_gain);
    }
    Ok(gains)
}

pub(crate) fn validate_rank_labels(labels: &[f32], gains: &[f64]) -> Result<()> {
    if labels
        .iter()
        .any(|&y| y < 0.0 || y.fract() != 0.0 || f64::from(y) >= gains.len() as f64)
    {
        return invalid(
            "ranking labels must be nonnegative integers smaller than label_gain length",
        );
    }
    Ok(())
}

pub(crate) fn discount(rank: usize) -> f64 {
    1.0 / (2.0 + rank as f64).log2()
}

pub(crate) fn ideal_dcg(labels: &[f32], gains: &[f64], cutoff: usize) -> Result<f64> {
    let mut sorted = reserved(labels.len())?;
    sorted.extend_from_slice(labels);
    sorted.sort_by(|a, b| b.total_cmp(a));
    finite(
        sorted
            .iter()
            .take(cutoff)
            .enumerate()
            .map(|(rank, &label)| discount(rank) * gains[label as usize])
            .sum(),
        "ideal DCG",
    )
}

#[cfg(test)]
mod empty_query_tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn native_empty_query_rng_and_sampled_query_indexes_match_each_step() {
        let fixture: Value =
            serde_json::from_str(include_str!("../../tests/fixtures/empty_queries.json")).unwrap();
        let c = Config {
            objective: Objective::RankXendcg,
            objective_seed: 7,
            ..Config::default()
        };
        for case in fixture["cases"].as_array().unwrap() {
            let labels: Vec<f32> = serde_json::from_value(case["labels"].clone()).unwrap();
            let groups: Vec<usize> = serde_json::from_value(case["groups"].clone()).unwrap();
            let positions: Option<Vec<i32>> = case
                .get("positions")
                .map(|p| serde_json::from_value(p.clone()).unwrap());
            let mut state = ObjectiveState::new(
                &c,
                ObjectiveInput {
                    labels: &labels,
                    weights: None,
                    groups: Some(&groups),
                    positions: positions.as_deref(),
                },
            )
            .unwrap();
            let mut g = vec![0.; labels.len()];
            let mut h = vec![0.; labels.len()];
            for step in case["steps"].as_array().unwrap() {
                let scores: Vec<f64> = serde_json::from_value(step["scores"].clone()).unwrap();
                if let Some(queries) = step.get("queries") {
                    let queries: Vec<usize> = serde_json::from_value(queries.clone()).unwrap();
                    state
                        .gradients_for_queries(&scores, &queries, &mut g, &mut h)
                        .unwrap();
                } else {
                    (g, h) = state.gradients(&scores).unwrap();
                }
                let eg: Vec<f32> = serde_json::from_value(step["gradients"].clone()).unwrap();
                let eh: Vec<f32> = serde_json::from_value(step["hessians"].clone()).unwrap();
                assert_eq!(g, eg, "{} gradients", case["name"]);
                assert_eq!(h, eh, "{} Hessians", case["name"]);
            }
        }
    }

    #[test]
    fn empty_query_validation_checks_total_and_rejects_undefined_combinations() {
        for groups in [vec![], vec![0, 0], vec![0, 3], vec![usize::MAX, 1, 4]] {
            assert!(validate_groups(&groups, 4).is_err());
        }
        assert!(validate_groups(&[0, 2, 0, 2, 0], 4).is_ok());
        let labels = [0., 1., 1., 0.];
        for objective in [Objective::RankXendcg, Objective::LambdaRank] {
            let c = Config {
                objective,
                ..Config::default()
            };
            for weights in [&[][..], &[1.; 4][..]] {
                let result = ObjectiveState::new(
                    &c,
                    ObjectiveInput {
                        labels: &labels,
                        weights: Some(weights),
                        groups: Some(&[0, 2, 2]),
                        positions: None,
                    },
                );
                assert!(matches!(result, Err(Error::Unsupported(_))));
            }
        }
    }
}
