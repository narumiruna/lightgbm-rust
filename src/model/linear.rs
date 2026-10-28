//! Validated linear-leaf coefficients for the native v4 text format.

use super::{MAX_ELEMENTS, append_array, bounded, invalid, required, required_array, same_len};
use crate::Result;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

/// Read-only linear coefficients. A selected NaN feature uses the tree's
/// constant fallback leaf value, not a partial dot product.
#[derive(Clone, Debug, Serialize)]
pub struct LinearLeaves {
    pub(crate) constants: Vec<f64>,
    pub(crate) features: Vec<Vec<usize>>,
    pub(crate) coefficients: Vec<Vec<f64>>,
}

impl LinearLeaves {
    pub fn constants(&self) -> &[f64] {
        &self.constants
    }
    pub fn features(&self) -> &[Vec<usize>] {
        &self.features
    }
    pub fn coefficients(&self) -> &[Vec<f64>] {
        &self.coefficients
    }

    pub(super) fn from_fields(fields: &BTreeMap<&str, &str>, leaves: usize) -> Result<Self> {
        let constants = required_array(fields, "leaf_const", leaves)?;
        let counts: Vec<usize> = required_array(fields, "num_features", leaves)?;
        let total = counts
            .iter()
            .try_fold(0usize, |total, &count| total.checked_add(count))
            .ok_or_else(|| {
                crate::Error::InvalidModel("linear coefficient count overflow".into())
            })?;
        bounded(total, MAX_ELEMENTS, "linear coefficient count", true)?;
        // Require all fields even when the first tree has zero coefficients.
        required(fields, "leaf_features")?;
        required(fields, "leaf_coeff")?;
        let flat_features: Vec<usize> = required_array(fields, "leaf_features", total)?;
        let flat_coefficients: Vec<f64> = required_array(fields, "leaf_coeff", total)?;
        let mut features = Vec::with_capacity(leaves);
        let mut coefficients = Vec::with_capacity(leaves);
        let mut start = 0;
        for count in counts {
            let end = start + count;
            features.push(flat_features[start..end].to_vec());
            coefficients.push(flat_coefficients[start..end].to_vec());
            start = end;
        }
        Ok(Self {
            constants,
            features,
            coefficients,
        })
    }

    pub(super) fn validate(&self, leaves: usize, num_features: usize) -> Result<()> {
        same_len(self.constants.len(), leaves, "leaf_const")?;
        same_len(self.features.len(), leaves, "linear leaf features")?;
        same_len(self.coefficients.len(), leaves, "linear leaf coefficients")?;
        if self.constants.iter().any(|v| !v.is_finite()) {
            return invalid("nonfinite linear leaf constant");
        }
        let mut total = 0usize;
        for (features, coefficients) in self.features.iter().zip(&self.coefficients) {
            same_len(features.len(), coefficients.len(), "linear coefficients")?;
            total = total.checked_add(features.len()).ok_or_else(|| {
                crate::Error::InvalidModel("linear coefficient count overflow".into())
            })?;
            let mut unique = BTreeSet::new();
            if features
                .iter()
                .any(|&f| f >= num_features || !unique.insert(f))
            {
                return invalid("invalid or duplicate linear feature index");
            }
            if coefficients.iter().any(|v| !v.is_finite()) {
                return invalid("nonfinite linear coefficient");
            }
        }
        bounded(total, MAX_ELEMENTS, "linear coefficient count", true)
    }

    pub(super) fn prediction(&self, leaf: usize, get: &impl Fn(usize) -> f64) -> Option<f64> {
        let mut output = self.constants[leaf];
        for (&feature, &coefficient) in self.features[leaf].iter().zip(&self.coefficients[leaf]) {
            let value = get(feature);
            if value.is_nan() {
                return None;
            }
            output += coefficient * value;
        }
        Some(output)
    }

    pub(super) fn append_fields(&self, output: &mut String) {
        append_array(output, "leaf_const", &self.constants);
        let counts: Vec<_> = self.features.iter().map(Vec::len).collect();
        append_array(output, "num_features", &counts);
        let features: Vec<_> = self.features.iter().flatten().copied().collect();
        let coefficients: Vec<_> = self.coefficients.iter().flatten().copied().collect();
        append_array(output, "leaf_features", &features);
        append_array(output, "leaf_coeff", &coefficients);
    }
}
