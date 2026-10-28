//! Persistent DART dropout state, translated from pinned LightGBM dart.hpp
//! (MIT). Copyright (c) Microsoft Corporation and The LightGBM developers.
//! See THIRD_PARTY_NOTICES.md.

use crate::Config;
use crate::random::Random;

#[cfg(test)]
mod conformance;

pub(crate) struct DartState {
    random: Random,
    weights: Vec<f64>,
    sum: f64,
}

impl DartState {
    pub(crate) fn new(config: &Config) -> Self {
        Self {
            random: Random::new(config.drop_seed),
            weights: Vec::new(),
            sum: 0.0,
        }
    }

    /// Native ResetConfig reseeds dropout and clears only the sum. The existing
    /// weight vector is deliberately retained, including its nonzero entries.
    pub(crate) fn reset_config(&mut self, config: &Config) {
        self.random = Random::new(config.drop_seed);
        self.sum = 0.0;
    }

    pub(crate) fn select(&mut self, iteration: usize, config: &Config) -> (Vec<usize>, f64) {
        let mut dropped = Vec::new();
        let skip = f64::from(self.random.next_float()) < config.skip_drop;
        if !skip {
            let mut rate = config.drop_rate;
            let inverse_average = iteration as f64 / self.sum;
            if config.max_drop > 0 {
                rate = rate.min(if config.uniform_drop {
                    config.max_drop as f64 / iteration as f64
                } else {
                    config.max_drop as f64 * inverse_average / self.sum
                });
            }
            for i in 0..iteration {
                let probability = if config.uniform_drop {
                    rate
                } else {
                    rate * self.weights[i] * inverse_average
                };
                if f64::from(self.random.next_float()) < probability {
                    dropped.push(i);
                    if config.max_drop >= 0 && dropped.len() >= config.max_drop as usize {
                        break;
                    }
                }
            }
        }
        let k = dropped.len() as f64;
        let shrinkage = if config.xgboost_dart_mode && k > 0.0 {
            config.learning_rate / (config.learning_rate + k)
        } else {
            config.learning_rate / (1.0 + k)
        };
        (dropped, shrinkage)
    }

    pub(crate) fn finish(&mut self, dropped: &[usize], shrinkage: f64, config: &Config) {
        if config.uniform_drop {
            return;
        }
        let k = dropped.len() as f64;
        let denominator = k + if config.xgboost_dart_mode {
            config.learning_rate
        } else {
            1.0
        };
        for &i in dropped {
            self.sum -= self.weights[i] * (1.0 / denominator);
            self.weights[i] *= k / denominator;
        }
        self.weights.push(shrinkage);
        self.sum += shrinkage;
    }
}
