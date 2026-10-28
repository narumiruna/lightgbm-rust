//! Bounded CPU execution without the global Rayon pool.
//!
//! Zero threads selects one worker. Counts above 256 are rejected. Each owner
//! reuses its pool. Calls made from an existing Rayon worker run serially, so
//! nested workflows do not create additional workers. Per-feature accumulation
//! and per-row tree traversal remain serial; indexed results keep input order.
use crate::{Error, Model, Result};
use rayon::prelude::*;

pub(crate) struct Executor {
    pool: Option<rayon::ThreadPool>,
}

impl Executor {
    pub(crate) fn new(threads: usize) -> Result<Self> {
        if threads > 256 {
            return Err(Error::InvalidInput(
                "num_threads must be at most 256".into(),
            ));
        }
        let pool = if threads <= 1 || rayon::current_thread_index().is_some() {
            None
        } else {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(|i| format!("lightgbm-{i}"))
                    .build()
                    .map_err(|e| Error::InvalidInput(format!("cannot create CPU pool: {e}")))?,
            )
        };
        Ok(Self { pool })
    }

    pub(crate) fn threads(&self) -> usize {
        self.pool
            .as_ref()
            .map_or(1, rayon::ThreadPool::current_num_threads)
    }

    pub(crate) fn map<T: Send>(
        &self,
        count: usize,
        minimum: usize,
        f: impl Fn(usize) -> T + Sync + Send,
    ) -> Vec<T> {
        if count >= minimum
            && rayon::current_thread_index().is_none()
            && let Some(pool) = &self.pool
        {
            pool.install(|| (0..count).into_par_iter().map(&f).collect())
        } else {
            (0..count).map(f).collect()
        }
    }
}

/// Reusable, bounded batch predictor. Model prediction semantics are unchanged.
/// Construct once outside timed prediction spans, then reuse for successive batches.
pub struct ParallelPredictor {
    executor: Executor,
}

impl ParallelPredictor {
    pub fn new(num_threads: usize) -> Result<Self> {
        Ok(Self {
            executor: Executor::new(num_threads)?,
        })
    }

    /// Actual pool size; one means serial execution.
    pub fn num_threads(&self) -> usize {
        self.executor.threads()
    }

    pub fn predict_dense(
        &self,
        model: &Model,
        values: &[f64],
        nrows: usize,
        ncols: usize,
    ) -> Result<Vec<Vec<f64>>> {
        if ncols != model.num_features() || nrows.checked_mul(ncols) != Some(values.len()) {
            return Err(Error::InvalidInput(
                "dense prediction shape does not match model".into(),
            ));
        }
        check_output_size(nrows, model.num_classes())?;
        self.executor
            .map(nrows, 128, |row| {
                model.predict_row(&values[row * ncols..(row + 1) * ncols])
            })
            .into_iter()
            .collect()
    }

    /// Canonical CSR batches. Individual rows use the checked model CSR API.
    pub fn predict_csr(
        &self,
        model: &Model,
        indptr: &[usize],
        indices: &[usize],
        values: &[f64],
    ) -> Result<Vec<Vec<f64>>> {
        if indptr.is_empty()
            || indptr[0] != 0
            || indptr.last() != Some(&indices.len())
            || indices.len() != values.len()
            || indptr.windows(2).any(|w| w[0] > w[1])
        {
            return Err(Error::InvalidInput("invalid CSR prediction shape".into()));
        }
        check_output_size(indptr.len() - 1, model.num_classes())?;
        self.executor
            .map(indptr.len() - 1, 128, |row| {
                let range = indptr[row]..indptr[row + 1];
                model.predict_csr_row(&indices[range.clone()], &values[range])
            })
            .into_iter()
            .collect()
    }
}

fn check_output_size(rows: usize, classes: usize) -> Result<()> {
    if rows
        .checked_mul(classes)
        .is_none_or(|n| n > 16 * 1024 * 1024)
    {
        return Err(Error::InvalidInput(
            "batch exceeds 16777216 output element limit".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_limits_order_and_nested_fallback() {
        assert!(Executor::new(257).is_err());
        assert_eq!(Executor::new(0).unwrap().threads(), 1);
        let executor = Executor::new(2).unwrap();
        assert_eq!(executor.threads(), 2);
        let values = executor.map(100, 1, |i| (i, Executor::new(4).unwrap().threads()));
        assert_eq!(values, (0..100).map(|i| (i, 1)).collect::<Vec<_>>());
    }
}
