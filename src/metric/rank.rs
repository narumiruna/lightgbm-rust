//! Query-weighted NDCG and mean average precision.
//!
//! Ported from LightGBM src/metric/{rank_metric,map_metric,dcg_calculator} at
//! 5d84f3f222b41409a6983d2ac90513ad9702fa9e (MIT).
//! Copyright (c) 2016-2026 Microsoft Corporation and The LightGBM developers.
//! See THIRD_PARTY_NOTICES.md.

use super::*;
use crate::objective::{discount, ideal_dcg, label_gains, validate_rank_labels};

pub(super) fn evaluate(
    config: &Config,
    labels: &[f32],
    weights: Option<&[f32]>,
    groups: Option<&[usize]>,
    scores: &[f64],
    name: &str,
) -> Result<Vec<MetricValue>> {
    let groups =
        groups.ok_or_else(|| Error::InvalidInput("ranking metrics require query groups".into()))?;
    if weights.is_some() && groups.contains(&0) {
        // Metadata::CalculateQueryWeights divides a label_t sum by query size.
        // For empty queries this is 0/0; no finite weighted metric is defined.
        return Err(Error::Unsupported(
            "weighted ranking metrics with empty queries are undefined in the pinned native metadata (sum / 0)".into(),
        ));
    }
    let mut cutoffs = reserved(if config.eval_at.is_empty() {
        5
    } else {
        config.eval_at.len()
    })?;
    if config.eval_at.is_empty() {
        cutoffs.extend(1..=5);
    } else {
        cutoffs.extend_from_slice(&config.eval_at);
        cutoffs.sort_unstable();
    }
    let gains = label_gains(config)?;
    if name == "ndcg" {
        validate_rank_labels(labels, &gains)?;
        if groups.iter().any(|&n| n > 10_000) {
            return invalid("NDCG queries cannot exceed 10000 rows");
        }
    }
    let mut totals: Vec<f64> = zeros(cutoffs.len())?;
    let mut total_weight = 0.0;
    let mut start = 0;
    for &size in groups {
        let end = start + size;
        // Metadata::CalculateQueryWeights accumulates and divides in label_t.
        let query_weight = weights.map_or(1.0, |w| {
            let mut sum = 0.0_f32;
            for &weight in &w[start..end] {
                sum += weight;
            }
            f64::from(sum / size as f32)
        });
        total_weight += query_weight;
        let mut order = reserved(size)?;
        order.extend(start..end);
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
        if name == "ndcg" {
            let first_max = ideal_dcg(&labels[start..end], &gains, cutoffs[0])?;
            if first_max <= 0.0 {
                // The pinned metric deliberately adds 1, not query_weight, for
                // a query without positive ideal DCG (also in the weighted path).
                for total in &mut totals {
                    *total += 1.0;
                }
            } else {
                for (index, &cutoff) in cutoffs.iter().enumerate() {
                    let max = ideal_dcg(&labels[start..end], &gains, cutoff)?;
                    let dcg = order
                        .iter()
                        .take(cutoff)
                        .enumerate()
                        .map(|(rank, &row)| gains[labels[row] as usize] * discount(rank))
                        .sum::<f64>();
                    totals[index] +=
                        dcg * (if max > 0.0 { 1.0 / max } else { -1.0 }) * query_weight;
                }
            }
        } else {
            let positives = labels[start..end].iter().filter(|&&y| y > 0.5).count();
            for (index, &cutoff) in cutoffs.iter().enumerate() {
                let mut hits = 0;
                let mut precision = 0.0;
                for (rank, &row) in order.iter().take(cutoff).enumerate() {
                    if labels[row] > 0.5 {
                        hits += 1;
                        precision += hits as f64 / (rank + 1) as f64;
                    }
                }
                totals[index] += if positives > 0 {
                    precision / positives.min(cutoff.min(size)) as f64 * query_weight
                } else {
                    query_weight
                };
            }
        }
        start = end;
    }
    let mut output = reserved(cutoffs.len())?;
    for (cutoff, total) in cutoffs.into_iter().zip(totals) {
        output.push(metric_value(
            &format!("{name}@{cutoff}"),
            total / total_weight,
            true,
        )?);
    }
    Ok(output)
}

#[cfg(test)]
mod empty_query_tests {
    use super::*;

    #[test]
    fn weighted_empty_queries_fail_before_access_to_weight_or_score_buffers() {
        for name in ["ndcg", "map"] {
            let result = evaluate(
                &Config::default(),
                &[0., 1., 1., 0.],
                Some(&[]),
                Some(&[0, 2, 2]),
                &[],
                name,
            );
            assert!(matches!(result, Err(Error::Unsupported(_))));
        }
    }
}
