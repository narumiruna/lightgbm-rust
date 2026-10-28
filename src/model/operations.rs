//! Checked immutable model operations.
//!
//! Refit follows LightGBM commit 5d84f3f222b41409a6983d2ac90513ad9702fa9e,
//! gbdt.cpp::RefitTree and serial_tree_learner.cpp::FitByExistingTree.
//! Linear refit follows linear_tree_learner.cpp::CalculateLinear(is_refit=true).
//! Copyright (c) 2016-2026 Microsoft Corporation and The LightGBM developers.
//! Licensed under the MIT License. See THIRD_PARTY_NOTICES.md.

use super::{Model, Tree, dense_value};
use crate::objective::{EPSILON, ObjectiveInput, ObjectiveState, checked_len, finite, zeros};
use crate::{Boosting, Config, Dataset, Error, Result};

fn invalid(message: &str) -> Error {
    Error::InvalidInput(format!("model operation: {message}"))
}

impl Model {
    /// Get a zero-based tree/leaf output. For a linear leaf this is the NaN
    /// fallback, not its intercept or a prediction at a particular point.
    pub fn leaf_output(&self, tree: usize, leaf: usize) -> Result<f64> {
        self.trees
            .get(tree)
            .and_then(|tree| tree.leaf_values.get(leaf))
            .copied()
            .ok_or_else(|| invalid("tree or leaf index exceeds model"))
    }

    /// Return a validated copy with one fallback output changed. Values in the
    /// inclusive native near-zero interval become positive zero. Linear-leaf
    /// intercepts and coefficients, tree structure, and counts do not change.
    pub fn with_leaf_output(&self, tree: usize, leaf: usize, value: f64) -> Result<Self> {
        self.leaf_output(tree, leaf)?;
        if !value.is_finite() {
            return Err(invalid("leaf output must be finite"));
        }
        let mut model = self.clone();
        model.trees[tree].leaf_values[leaf] = dense_value(value);
        model.validate()?;
        Ok(model)
    }

    /// Keep the first `iterations` complete iterations, including all classes.
    /// Zero returns an empty ensemble with the same objective and feature metadata.
    /// This is a structural prefix, not a training-state or random-state restore.
    /// In particular, retained DART trees keep their current normalized outputs.
    pub fn truncate(&self, iterations: usize) -> Result<Self> {
        if iterations > self.num_iterations() {
            return Err(invalid("prefix exceeds model iteration count"));
        }
        let mut model = self.clone();
        model
            .trees
            .truncate(iterations * self.num_tree_per_iteration);
        model.validate()?;
        Ok(model)
    }

    /// Remove the final complete iteration. An empty ensemble stays empty.
    /// This immutable operation does not restore trainer scores, RNG state, or
    /// historical DART normalization. Use a training checkpoint for that purpose.
    pub fn rollback_one_iteration(&self) -> Result<Self> {
        self.truncate(self.num_iterations().saturating_sub(1))
    }

    /// Refit existing leaves on new labels without changing tree structure.
    ///
    /// `leaf_indices[row][tree]` must be the ORIGINAL model's leaf prediction
    /// on that row's raw features, in full tree/class order. Repeated leaf IDs
    /// across rows are valid. Binned data cannot recover these predictions:
    /// routing old thresholds through new bin maps can change leaf membership.
    /// The caller is responsible for supplying the correct original routing.
    ///
    /// Scores start at the dataset's class-major initial margins, or zero;
    /// neither old predictions nor a new label mean are added. Derivatives are
    /// recalculated once per complete iteration. Outputs use the old tree's
    /// shrinkage and `decay * old + (1 - decay) * new`. Counts, weights, gains,
    /// and internal outputs remain unchanged, as in native refit. Native refit
    /// uses Newton outputs even for objectives with training-time leaf renewal.
    ///
    /// GBDT, GOSS, DART, RF, and query-bagging models use the native refit
    /// rules. RF recomputes derivatives at zero margins, not its accumulated
    /// score. Path smoothing is unsupported: imported native trees omit the
    /// leaf-parent state that its refit path reads.
    /// Linear models require retained raw values (`linear_tree=true`). All
    /// shapes and leaf IDs are checked before fitting; the input is not changed.
    /// Rows times trees and rows times classes are each limited to 16 Mi entries.
    /// Columns use model feature order; callers must retain category codebooks.
    pub fn refit(
        &self,
        data: &Dataset,
        leaf_indices: &[Vec<usize>],
        config: &Config,
        decay: f64,
    ) -> Result<Self> {
        self.validate_refit(data, leaf_indices, config, decay)?;
        let mut objective = ObjectiveState::new(
            config,
            ObjectiveInput {
                labels: data.labels(),
                weights: data.weights(),
                groups: data.groups(),
                positions: data.positions(),
            },
        )?;
        let size = checked_len(data.nrows(), self.num_class)?;
        let mut scores = zeros::<f64>(size)?;
        if let Some(initial) = data.initial_scores() {
            scores.copy_from_slice(initial);
        }
        let mut model = self.clone();
        let rf_zero = if config.boosting == Boosting::Rf {
            Some(zeros::<f64>(size)?)
        } else {
            None
        };
        for iteration in 0..self.num_iterations() {
            // RF::Boosting asks BoostFromAverage after model merge. The model
            // is nonempty, so that initializer returns zero on every refit round.
            let (gradients, hessians) =
                objective.gradients(rf_zero.as_deref().unwrap_or(&scores))?;
            for class in 0..self.num_tree_per_iteration {
                let index = iteration * self.num_tree_per_iteration + class;
                let range = class * data.nrows()..(class + 1) * data.nrows();
                let tree = &mut model.trees[index];
                refit_constant(
                    tree,
                    leaf_indices,
                    index,
                    &gradients[range.clone()],
                    &hessians[range.clone()],
                    config,
                    decay,
                )?;
                if tree.linear.is_some() {
                    refit_linear(
                        tree,
                        data,
                        leaf_indices,
                        index,
                        &gradients[range.clone()],
                        &hessians[range.clone()],
                        config,
                        decay,
                    )?;
                }
                for (row, leaves) in leaf_indices.iter().enumerate() {
                    let output = if tree.linear.is_some() {
                        crate::linear::predict_from_leaf(tree, data, row, leaves[index])?
                    } else {
                        tree.leaf_values[leaves[index]]
                    };
                    let target = &mut scores[range.start + row];
                    *target = finite(*target + output, "refit score")?;
                }
            }
        }
        model.validate()?;
        Ok(model)
    }

    fn validate_refit(
        &self,
        data: &Dataset,
        leaves: &[Vec<usize>],
        config: &Config,
        decay: f64,
    ) -> Result<()> {
        config.validate()?;
        if !decay.is_finite() || !(0.0..=1.0).contains(&decay) {
            return Err(invalid("decay must be finite and in [0, 1]"));
        }
        if self.trees.is_empty() || data.nrows() == 0 {
            return Err(invalid("refit requires nonempty data and model"));
        }
        checked_len(data.nrows(), self.trees.len())?;
        let size = checked_len(data.nrows(), self.num_class)?;
        if data.ncols() != self.num_features() || config.num_class != self.num_class {
            return Err(invalid(
                "dataset features or configured classes differ from model",
            ));
        }
        if leaves.len() != data.nrows() || leaves.iter().any(|row| row.len() != self.trees.len()) {
            return Err(invalid(
                "leaf indices must have shape [dataset rows][model trees]",
            ));
        }
        // Check all IDs before any leaf access or temporary allocation.
        for row in leaves {
            for (&leaf, tree) in row.iter().zip(&self.trees) {
                if leaf >= tree.num_leaves() {
                    return Err(invalid("leaf index exceeds tree leaf count"));
                }
            }
        }
        if data
            .initial_scores()
            .is_some_and(|s| s.len() != size || s.iter().any(|v| !v.is_finite()))
        {
            return Err(invalid(
                "initial margins must be finite and match rows times classes",
            ));
        }
        if self.average_output != (config.boosting == Boosting::Rf) {
            return Err(invalid(
                "ensemble averaging configuration differs from model",
            ));
        }
        if config.boosting == Boosting::Rf && data.initial_scores().is_some() {
            return Err(invalid("RF does not support dataset initial margins"));
        }
        if config.path_smooth > EPSILON {
            return Err(Error::Unsupported(
                "imported-tree refit does not support path smoothing".into(),
            ));
        }
        data.check_binning_config(config)?;
        let mut tokens = self.objective.split_whitespace();
        if tokens.next() != Some(config.objective.name())
            || self.objective.split_whitespace().any(|t| t == "sqrt") != config.reg_sqrt
            || self
                .objective
                .split_whitespace()
                .filter_map(|t| t.strip_prefix("sigmoid:"))
                .any(|v| v.parse::<f64>().ok() != Some(config.sigmoid))
        {
            return Err(invalid(
                "configured objective or transform differs from model",
            ));
        }
        if self
            .trees
            .iter()
            .any(|tree| tree.linear.is_some() != config.linear_tree)
        {
            return Err(invalid("linear-tree configuration differs from model"));
        }
        // Feature type must agree even though new bins need not match old bins.
        for tree in &self.trees {
            for (&feature, &decision) in tree.split_features.iter().zip(&tree.decision_types) {
                if data.mapper(feature).is_categorical() != (decision & 1 != 0) {
                    return Err(invalid("dataset feature type differs from model split"));
                }
            }
        }
        if config.linear_tree {
            if !data.has_raw_values() {
                return Err(invalid("linear refit requires retained raw values"));
            }
            for tree in &self.trees {
                for &feature in tree
                    .linear
                    .as_ref()
                    .expect("checked linear layout")
                    .features
                    .iter()
                    .flatten()
                {
                    if data.mapper(feature).is_categorical() {
                        return Err(invalid("linear leaf requires numerical features"));
                    }
                }
            }
            for feature in 0..data.ncols() {
                if !data.mapper(feature).is_categorical() {
                    for row in 0..data.nrows() {
                        let value = data.raw_value(row, feature)?;
                        if !value.is_nan() && !value.is_finite() {
                            return Err(invalid("linear raw features must be finite or NaN"));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

fn refit_constant(
    tree: &mut Tree,
    leaves: &[Vec<usize>],
    index: usize,
    gradients: &[f32],
    hessians: &[f32],
    config: &Config,
    decay: f64,
) -> Result<()> {
    let mut sums = zeros::<(f64, f64)>(tree.num_leaves())?;
    for sum in &mut sums {
        sum.1 = EPSILON;
    }
    // ResetByLeafPred preserves increasing row order within each leaf.
    for (row, ids) in leaves.iter().enumerate() {
        let sum = &mut sums[ids[index]];
        sum.0 += f64::from(gradients[row]);
        sum.1 += f64::from(hessians[row]);
    }
    for (old, (g, h)) in tree.leaf_values.iter_mut().zip(sums) {
        let regularized = g.signum() * (g.abs() - config.lambda_l1).max(0.0);
        let mut output = -regularized / (h + config.lambda_l2);
        if config.max_delta_step > 0.0 && output.abs() > config.max_delta_step {
            output = output.signum() * config.max_delta_step;
        }
        *old = dense_value(finite(
            decay * *old + (1.0 - decay) * (output * tree.shrinkage),
            "refit leaf output",
        )?);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn refit_linear(
    tree: &mut Tree,
    data: &Dataset,
    leaves: &[Vec<usize>],
    index: usize,
    gradients: &[f32],
    hessians: &[f32],
    config: &Config,
    decay: f64,
) -> Result<()> {
    let mut counts = zeros::<usize>(tree.num_leaves())?;
    for row in leaves {
        counts[row[index]] += 1;
    }
    let mut ordered = zeros::<Vec<usize>>(tree.num_leaves())?;
    for (rows, &count) in ordered.iter_mut().zip(&counts) {
        rows.try_reserve_exact(count)
            .map_err(|e| invalid(&format!("refit row allocation failed: {e}")))?;
    }
    for (row, ids) in leaves.iter().enumerate() {
        ordered[ids[index]].push(row);
    }
    // HAS_NAN is global to numeric split features, not the current leaf.
    let mut has_nan = false;
    for &feature in &tree.split_features {
        if !data.mapper(feature).is_categorical() {
            for row in 0..data.nrows() {
                let value = data.raw_value(row, feature)?;
                if value.is_nan() {
                    has_nan = true;
                } else {
                    finite(value, "linear refit raw feature")?;
                }
            }
        }
    }
    let linear = tree.linear.as_mut().expect("checked linear tree");
    let mut work = 0usize;
    for (leaf, rows) in ordered.iter().enumerate() {
        // Refit uses previously selected features, NOT all branch features.
        // Sorting leaves old_coeffs in their source order, as in native refit.
        let mut features = linear.features[leaf].clone();
        features.sort_unstable();
        features.dedup();
        let n = features.len() + 1;
        let mut count = 0usize;
        if has_nan {
            for &row in rows {
                for &feature in &features {
                    let value = data.raw_value(row, feature)?;
                    if value.is_nan() {
                        break;
                    }
                    finite(value, "linear refit raw feature")?;
                    // Source counts each finite feature before a NaN, even
                    // when it subsequently rejects this entire row.
                    count = count
                        .checked_add(1)
                        .ok_or_else(|| invalid("linear refit NaN count overflow"))?;
                }
            }
        } else {
            count = rows.len();
        }
        if count < n {
            linear.constants[leaf] = dense_value(finite(
                decay * linear.constants[leaf]
                    + (1.0 - decay) * tree.leaf_values[leaf] * tree.shrinkage,
                "linear refit fallback intercept",
            )?);
            linear.coefficients[leaf].fill(0.0);
            continue;
        }
        let square = crate::linear::checked_solver_size(n)?;
        work = rows
            .len()
            .checked_mul(square)
            .and_then(|a| n.checked_mul(square).and_then(|b| a.checked_add(b)))
            .and_then(|v| work.checked_add(v))
            .filter(|&v| v <= 64 * 1024 * 1024)
            .ok_or_else(|| invalid("linear refit exceeds 64 Mi matrix operations"))?;
        let mut matrix = zeros::<f64>(square)?;
        let mut rhs = zeros::<f64>(n)?;
        let mut values = zeros::<f64>(n)?;
        values[n - 1] = 1.0;
        for &row in rows {
            let mut missing = false;
            for (i, &feature) in features.iter().enumerate() {
                values[i] = data.raw_value(row, feature)?;
                if has_nan && values[i].is_nan() {
                    missing = true;
                    break;
                }
                finite(values[i], "linear refit raw feature")?;
            }
            if missing {
                continue;
            }
            let g = f64::from(gradients[row]);
            let h = f64::from(hessians[row]);
            for i in 0..n {
                rhs[i] = finite(rhs[i] + values[i] * g, "linear refit RHS")?;
                let weighted = values[i] * h;
                for j in i..n {
                    matrix[i * n + j] = finite(
                        matrix[i * n + j] + weighted * values[j],
                        "linear refit Gram matrix",
                    )?;
                }
            }
        }
        for i in 0..n {
            for j in i..n {
                matrix[j * n + i] = matrix[i * n + j];
            }
            if i + 1 < n {
                matrix[i * n + i] = finite(
                    matrix[i * n + i] + config.linear_lambda,
                    "linear refit regularization",
                )?;
            }
        }
        let coefficients = crate::linear::negative_inverse_product(matrix, &rhs)?;
        for (old, &new) in linear.coefficients[leaf].iter_mut().zip(&coefficients) {
            *old = dense_value(finite(
                decay * *old + (1.0 - decay) * new * tree.shrinkage,
                "linear refit coefficient",
            )?);
        }
        linear.constants[leaf] = dense_value(finite(
            decay * linear.constants[leaf] + (1.0 - decay) * coefficients[n - 1] * tree.shrinkage,
            "linear refit intercept",
        )?);
        // Do not prune zero coefficients and do not use a first-tree shortcut.
        linear.features[leaf] = features;
    }
    Ok(())
}
