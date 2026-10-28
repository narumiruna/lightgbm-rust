//! Trainer-owned serial row sampling, translated from the pinned LightGBM
//! bagging.hpp and goss.hpp (MIT). Copyright (c) Microsoft Corporation and
//! The LightGBM developers. See THIRD_PARTY_NOTICES.md.

use crate::config::DataSampleStrategy;
use crate::random::Random;
use crate::{Config, Dataset, Error, Objective, Result};

#[cfg(test)]
mod conformance;

pub(crate) struct RowSampler {
    random: Vec<Random>,
    rows: Vec<usize>,
    queries: Vec<usize>,
    active: bool,
    balanced: bool,
    sampled: bool,
    query_score_rows: Option<Vec<usize>>,
    query_subset: bool,
    use_subset: bool,
}

impl RowSampler {
    pub(crate) fn new(config: &Config, data: &Dataset) -> Result<Self> {
        if config.bagging_by_query && data.groups().is_none() {
            return Err(Error::InvalidInput(
                "query bagging requires query groups".into(),
            ));
        }
        let balanced = config.objective == Objective::Binary
            && data.labels().iter().any(|&y| y > 0.0)
            && (config.pos_bagging_fraction < 1.0 || config.neg_bagging_fraction < 1.0);
        let active = config.data_sample_strategy == DataSampleStrategy::Goss
            || (config.bagging_freq > 0 && (config.bagging_fraction < 1.0 || balanced));
        // Native query bagging overwrites only the selected prefix of its row
        // buffer. The non-subset score updater also visits the retained suffix.
        let initial_count = if balanced {
            let positive = data.labels().iter().filter(|&&y| y > 0.0).count();
            (positive as f64 * config.pos_bagging_fraction) as usize
                + ((data.nrows() - positive) as f64 * config.neg_bagging_fraction) as usize
        } else {
            (config.bagging_fraction * data.nrows() as f64) as usize
        };
        let subset = active
            && config.bagging_freq > 0
            && initial_count as f64 / data.nrows() as f64 / config.bagging_freq as f64 <= 0.5
            && (0..data.ncols())
                .filter(|&f| !data.mapper(f).is_trivial())
                .count()
                < 100;
        Ok(Self {
            random: (0..data.nrows().div_ceil(1024))
                .map(|i| Random::new(config.bagging_seed.wrapping_add(i as u32)))
                .collect(),
            rows: (0..data.nrows()).collect(),
            queries: (0..data.groups().map_or(0, <[usize]>::len)).collect(),
            active,
            balanced,
            sampled: false,
            query_score_rows: (active && config.bagging_by_query && !subset)
                .then(|| vec![0; data.nrows()]),
            query_subset: config.bagging_by_query && subset,
            use_subset: subset,
        })
    }

    pub(crate) fn uses_subset(&self) -> bool {
        self.use_subset
    }

    pub(crate) fn rows(&self) -> &[usize] {
        &self.rows
    }
    pub(crate) fn queries(&self) -> &[usize] {
        &self.queries
    }
    pub(crate) fn query_score_rows(&self) -> Option<&[usize]> {
        self.query_score_rows.as_deref()
    }

    pub(crate) fn retain_query_subset_cache(
        &self,
        n: usize,
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) {
        if !self.query_subset || self.rows.len() == n {
            return;
        }
        // GBDT copies selected derivatives into the same buffers before fitting
        // a dataset subset. Ranking position updates read that cache next round.
        for (g, h) in gradients
            .chunks_exact_mut(n)
            .zip(hessians.chunks_exact_mut(n))
        {
            for (index, &row) in self.rows.iter().enumerate() {
                g[index] = g[row];
                h[index] = h[row];
            }
        }
    }

    pub(crate) fn sample(
        &mut self,
        iteration: usize,
        config: &Config,
        data: &Dataset,
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) -> Result<()> {
        if !self.active {
            return Ok(());
        }
        if config.data_sample_strategy == DataSampleStrategy::Goss {
            return self.goss(iteration, config, data.nrows(), gradients, hessians);
        }
        // Native Bagging resamples only while bag_data_cnt_ < num_data_.
        // Once every row is selected, the same sample is retained even when
        // query sampling omitted an empty logical query.
        if self.sampled
            && (self.rows.len() == data.nrows()
                || !iteration.is_multiple_of(config.bagging_freq as usize))
        {
            return Ok(());
        }
        self.sampled = true;
        self.rows.clear();
        if config.bagging_by_query {
            self.queries.clear();
            let mut start = 0;
            for (query, &size) in data.groups().unwrap().iter().enumerate() {
                if f64::from(self.random[query / 1024].next_float()) < config.bagging_fraction {
                    self.queries.push(query);
                    self.rows.extend(start..start + size);
                }
                start += size;
            }
        } else {
            for row in 0..data.nrows() {
                let fraction = if self.balanced {
                    if data.labels()[row] > 0.0 {
                        config.pos_bagging_fraction
                    } else {
                        config.neg_bagging_fraction
                    }
                } else {
                    config.bagging_fraction
                };
                if f64::from(self.random[row / 1024].next_float()) < fraction {
                    self.rows.push(row);
                }
            }
        }
        if let Some(buffer) = &mut self.query_score_rows {
            buffer[..self.rows.len()].copy_from_slice(&self.rows);
        }
        Ok(())
    }

    fn goss(
        &mut self,
        iteration: usize,
        config: &Config,
        n: usize,
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) -> Result<()> {
        if iteration < (1.0 / config.learning_rate) as usize {
            return Ok(());
        }
        let top = ((n as f64 * config.top_rate) as usize).max(1);
        let other = (n as f64 * config.other_rate) as usize;
        let mut statistics = vec![0.0_f32; n];
        for class in 0..config.num_class {
            for row in 0..n {
                statistics[row] += (gradients[class * n + row] * hessians[class * n + row]).abs();
            }
        }
        if statistics.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidInput(
                "GOSS selection statistic is not finite".into(),
            ));
        }
        let mut sorted = statistics.clone();
        sorted.select_nth_unstable_by(top - 1, |a, b| b.total_cmp(a));
        let threshold = sorted[top - 1];
        let multiply = (n - top) as f32 / other as f32;
        self.rows.clear();
        let mut large = 0_i64;
        for (row, &statistic) in statistics.iter().enumerate() {
            if statistic >= threshold {
                self.rows.push(row);
                large += 1;
            } else {
                let sampled = self.rows.len() as i64 - large;
                let need = other as i64 - sampled;
                let remaining = (n - row) as i64 - (top as i64 - large);
                let probability = need as f64 / remaining as f64;
                if f64::from(self.random[row / 1024].next_float()) < probability {
                    self.rows.push(row);
                    for class in 0..config.num_class {
                        gradients[class * n + row] *= multiply;
                        hessians[class * n + row] *= multiply;
                    }
                }
            }
        }
        Ok(())
    }
}
