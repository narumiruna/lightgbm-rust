//! Retained histogram storage budget. Working matrices are checked separately.
//! The default unlimited source setting maps to a 128 MiB safety ceiling.
use crate::{Config, Error, Result};

pub(super) struct HistogramBudget {
    pub slots: usize,
}

impl HistogramBudget {
    pub fn new(config: &Config, bytes_per_histogram: usize) -> Result<Self> {
        if bytes_per_histogram == 0 || !config.histogram_pool_size.is_finite() {
            return Err(Error::InvalidInput("invalid histogram cache size".into()));
        }
        let ceiling = 128 * 1024 * 1024;
        let bytes = if config.histogram_pool_size > 0.0 {
            (config.histogram_pool_size * (1024 * 1024) as f64).min(ceiling as f64) as usize
        } else {
            ceiling
        };
        Ok(Self {
            slots: bytes / bytes_per_histogram,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_is_checked_before_retention_and_rounds_down() {
        let mut config = Config::default();
        assert_eq!(
            HistogramBudget::new(&config, 1024).unwrap().slots,
            128 * 1024
        );
        config.histogram_pool_size = 1.0 / 1024.0;
        assert_eq!(HistogramBudget::new(&config, 1024).unwrap().slots, 1);
        config.histogram_pool_size /= 2.0;
        assert_eq!(HistogramBudget::new(&config, 1024).unwrap().slots, 0);
        assert!(HistogramBudget::new(&config, 0).is_err());
    }
}
