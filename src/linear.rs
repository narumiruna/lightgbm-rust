//! Serial linear-leaf fitting from native-f32 features and derivatives.
//!
//! Training equations and NaN counting follow LightGBM at
//! 5d84f3f222b41409a6983d2ac90513ad9702fa9e, linear_tree_learner.cpp.
//! Copyright (c) 2020-2026 Microsoft Corporation. All rights reserved.
//! Copyright (c) 2020-2026 The LightGBM developers. All rights reserved.
//! Licensed under the MIT License. See LICENSE and THIRD_PARTY_NOTICES.md.
//!
//! The independent complete-pivot solver implements the observable behavior of
//! Eigen FullPivLU: column-major pivot ties, epsilon*n rank test, zero free
//! variables, and an explicit inverse followed by multiplication. A singular
//! system does not use a ridge or a least-norm solution.

use crate::{Config, Dataset, Error, LinearLeaves, Result, Tree};

const MAX_COEFFICIENTS: usize = 1_000_000;
const MAX_ELEMENTS: usize = 16 * 1024 * 1024;
const MAX_WORK: usize = 64 * 1024 * 1024;
const ZERO_THRESHOLD: f64 = 1e-35_f32 as f64;

fn invalid(message: &str) -> Error {
    Error::InvalidInput(format!("linear leaves: {message}"))
}

fn buffer<T: Clone>(len: usize, value: T) -> Result<Vec<T>> {
    if len > MAX_ELEMENTS {
        return Err(invalid("temporary allocation exceeds 16 Mi elements"));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|e| invalid(&format!("allocation failed: {e}")))?;
    values.resize(len, value);
    Ok(values)
}

fn finite(value: f64) -> Result<f64> {
    if !value.is_finite() {
        return Err(invalid("nonfinite value or numerical overflow"));
    }
    Ok(value)
}

/// Call after the Newton tree fit, BEFORE objective-specific leaf renewal and
/// learning-rate shrinkage. GBDT uses true for every class in its first round;
/// RF always uses false. The caller retains the Newton outputs as NaN fallbacks.
pub(crate) fn fit_linear_leaves(
    tree: &mut Tree,
    data: &Dataset,
    gradients: &[f32],
    hessians: &[f32],
    leaf_rows: &[Vec<usize>],
    config: &Config,
    is_first_tree: bool,
) -> Result<()> {
    let leaves = tree.num_leaves();
    if leaves == 0 || leaves > MAX_COEFFICIENTS || leaf_rows.len() != leaves {
        return Err(invalid("leaf row shape differs from tree"));
    }
    if data.nrows() > MAX_ELEMENTS
        || gradients.len() != data.nrows()
        || hessians.len() != data.nrows()
    {
        return Err(invalid(
            "derivative shape differs from dataset or exceeds limit",
        ));
    }
    if !config.linear_lambda.is_finite() || config.linear_lambda < 0.0 {
        return Err(invalid("linear_lambda must be finite and nonnegative"));
    }
    if tree.leaf_values.iter().any(|v| !v.is_finite())
        || gradients.iter().chain(hessians).any(|v| !v.is_finite())
    {
        return Err(invalid("nonfinite fallback or derivative"));
    }
    // Validate all untrusted row indexes before any derivative or raw-data access.
    // Missing rows are permitted for bagging; repeated rows are not.
    let mut row_leaf = buffer(data.nrows(), usize::MAX)?;
    for (leaf, rows) in leaf_rows.iter().enumerate() {
        for &row in rows {
            let target = row_leaf
                .get_mut(row)
                .ok_or_else(|| invalid("leaf row index exceeds dataset"))?;
            if *target != usize::MAX {
                return Err(invalid("leaf rows contain duplicate indexes"));
            }
            *target = leaf;
        }
    }
    let paths = numeric_paths(tree, data)?;
    let mut linear = LinearLeaves {
        constants: tree.leaf_values.clone(),
        features: buffer(leaves, Vec::new())?,
        coefficients: buffer(leaves, Vec::new())?,
    };
    if is_first_tree {
        tree.linear = Some(linear);
        return Ok(());
    }
    // Source HAS_NAN is global to numeric split features, not to a leaf or its
    // selected rows. An unselected row can activate the counting branch.
    let mut used = tree.split_features.clone();
    used.sort_unstable();
    used.dedup();
    let mut has_nan = false;
    for feature in used {
        if !data.get_mapper(feature)?.is_categorical() {
            for row in 0..data.nrows() {
                let value = data.raw_value(row, feature)?;
                if value.is_nan() {
                    has_nan = true;
                } else {
                    finite(value)?;
                }
            }
        }
    }
    // Source accumulation visits original row indexes in increasing order,
    // independent of the learner's partition order.
    let mut ordered = buffer(leaves, Vec::<usize>::new())?;
    for (leaf, rows) in ordered.iter_mut().zip(leaf_rows) {
        leaf.try_reserve_exact(rows.len())
            .map_err(|e| invalid(&format!("allocation failed: {e}")))?;
    }
    for (row, &leaf) in row_leaf.iter().enumerate() {
        if leaf != usize::MAX {
            ordered[leaf].push(row);
        }
    }
    let mut total_coefficients = 0usize;
    let mut total_work = 0usize;
    for (leaf, features) in paths.iter().enumerate() {
        let n = features.len() + 1;
        let mut count = 0usize;
        if has_nan {
            for &row in &ordered[leaf] {
                for &feature in features {
                    let value = data.raw_value(row, feature)?;
                    if value.is_nan() {
                        break;
                    }
                    finite(value)?;
                    // Intentional source behavior: count each non-NaN feature,
                    // including features before a NaN in a rejected row.
                    count = count
                        .checked_add(1)
                        .ok_or_else(|| invalid("NaN row count overflow"))?;
                }
            }
        } else {
            count = ordered[leaf].len();
        }
        // Decide this before calculating or allocating a potentially large Gram.
        if count < n {
            continue;
        }
        let square = checked_solver_size(n)?;
        let accumulation = ordered[leaf]
            .len()
            .checked_mul(square)
            .ok_or_else(|| invalid("matrix work overflow"))?;
        total_work = total_work
            .checked_add(accumulation)
            .and_then(|v| n.checked_mul(square).and_then(|solve| v.checked_add(solve)))
            .filter(|&v| v <= MAX_WORK)
            .ok_or_else(|| invalid("matrix work exceeds 64 Mi operations"))?;
        let mut matrix = buffer(square, 0.0)?;
        let mut rhs = buffer(n, 0.0)?;
        let mut values = buffer(n, 1.0)?;
        for &row in &ordered[leaf] {
            let mut missing = false;
            for (i, &feature) in features.iter().enumerate() {
                values[i] = data.raw_value(row, feature)?;
                if values[i].is_nan() {
                    missing = true;
                    break;
                }
                finite(values[i])?;
            }
            if missing {
                continue;
            }
            let g = f64::from(gradients[row]);
            let h = f64::from(hessians[row]);
            for i in 0..n {
                rhs[i] = finite(rhs[i] + values[i] * g)?;
                let weighted = values[i] * h;
                for j in i..n {
                    matrix[i * n + j] = finite(matrix[i * n + j] + weighted * values[j])?;
                }
            }
        }
        for i in 0..n {
            for j in i..n {
                matrix[j * n + i] = matrix[i * n + j];
            }
            // Do not regularize the intercept.
            if i + 1 < n {
                matrix[i * n + i] = finite(matrix[i * n + i] + config.linear_lambda)?;
            }
        }
        let coefficients = negative_inverse_product(matrix, &rhs)?;
        linear.constants[leaf] = coefficients[n - 1];
        for (&feature, &coefficient) in features.iter().zip(&coefficients) {
            if coefficient.abs() > ZERO_THRESHOLD {
                total_coefficients += 1;
                if total_coefficients > MAX_COEFFICIENTS {
                    return Err(invalid("coefficient count exceeds 1m"));
                }
                linear.features[leaf].push(feature);
                linear.coefficients[leaf].push(coefficient);
            }
        }
    }
    // Do not expose a partially fitted model when any check or solve fails.
    tree.linear = Some(linear);
    Ok(())
}

/// Evaluate a known leaf with native-f32 retained raw features. Do not use the
/// C API's near-zero normalization. A selected NaN discards the whole dot product.
pub(crate) fn predict_from_leaf(
    tree: &Tree,
    data: &Dataset,
    row: usize,
    leaf: usize,
) -> Result<f64> {
    if row >= data.nrows() || leaf >= tree.num_leaves() {
        return Err(invalid("prediction row or leaf exceeds shape"));
    }
    let fallback = finite(tree.leaf_values[leaf])?;
    let Some(linear) = &tree.linear else {
        return Ok(fallback);
    };
    let mut value = linear.constants[leaf];
    for (&feature, &coefficient) in linear.features[leaf].iter().zip(&linear.coefficients[leaf]) {
        let raw = data.raw_value(row, feature)?;
        if raw.is_nan() {
            return Ok(fallback);
        }
        value = finite(value + coefficient * finite(raw)?)?;
    }
    finite(value)
}

/// Recover original feature indexes from parent links, not from a previous
/// coefficient list. A repeated path feature appears only once in the solve.
fn numeric_paths(tree: &Tree, data: &Dataset) -> Result<Vec<Vec<usize>>> {
    let leaves = tree.num_leaves();
    let nodes = leaves - 1;
    if tree.split_features.len() != nodes
        || tree.left_children.len() != nodes
        || tree.right_children.len() != nodes
        || tree.decision_types.len() != nodes
    {
        return Err(invalid("tree split arrays have inconsistent lengths"));
    }
    let mut node_parent = buffer(nodes, usize::MAX)?;
    let mut leaf_parent = buffer(leaves, usize::MAX)?;
    for node in 0..nodes {
        data.get_mapper(tree.split_features[node])?;
        for child in [tree.left_children[node], tree.right_children[node]] {
            let target = if child < 0 {
                leaf_parent.get_mut((!child) as usize)
            } else {
                node_parent.get_mut(child as usize)
            }
            .ok_or_else(|| invalid("tree child index exceeds shape"))?;
            if *target != usize::MAX {
                return Err(invalid("tree child has multiple parents"));
            }
            *target = node;
        }
    }
    if nodes > 0
        && (node_parent[0] != usize::MAX
            || node_parent[1..].contains(&usize::MAX)
            || leaf_parent.contains(&usize::MAX))
    {
        return Err(invalid("tree has an invalid root or disconnected child"));
    }
    let mut paths = buffer(leaves, Vec::new())?;
    let mut visits = 0usize;
    let mut total = 0usize;
    for (leaf, path) in paths.iter_mut().enumerate() {
        let mut node = leaf_parent[leaf];
        let mut depth = 0;
        while node != usize::MAX {
            depth += 1;
            visits += 1;
            if depth > nodes || visits > MAX_WORK {
                return Err(invalid("tree path cycle or work limit exceeded"));
            }
            let feature = tree.split_features[node];
            if tree.decision_types[node] & 1 == 0 && !data.get_mapper(feature)?.is_categorical() {
                if path.len() >= MAX_COEFFICIENTS {
                    return Err(invalid("leaf feature count exceeds 1m"));
                }
                path.push(feature);
            }
            node = node_parent[node];
        }
        path.sort_unstable();
        path.dedup();
        total = total
            .checked_add(path.len())
            .filter(|&v| v <= MAX_COEFFICIENTS)
            .ok_or_else(|| invalid("path coefficient count exceeds 1m"))?;
    }
    Ok(paths)
}

pub(crate) fn checked_solver_size(n: usize) -> Result<usize> {
    n.checked_mul(n)
        .filter(|&square| {
            n > 0
                && square.checked_mul(3).is_some_and(|v| v <= MAX_ELEMENTS)
                && square.checked_mul(n).is_some_and(|v| v <= MAX_WORK)
        })
        .ok_or_else(|| invalid("solver matrix or work limit exceeded"))
}

/// Compute -(A.fullPivLu().inverse())*rhs. Solving directly against rhs has
/// different rounding and does not reproduce the native inverse expression.
pub(crate) fn negative_inverse_product(mut lu: Vec<f64>, rhs: &[f64]) -> Result<Vec<f64>> {
    let n = rhs.len();
    let square = checked_solver_size(n)?;
    if lu.len() != square || lu.iter().chain(rhs).any(|v| !v.is_finite()) {
        return Err(invalid("invalid solver matrix shape or nonfinite input"));
    }
    let mut rows: Vec<usize> = (0..n).collect();
    let mut columns = rows.clone();
    let mut max_pivot = 0.0_f64;
    for k in 0..n {
        let (mut pr, mut pc, mut largest) = (k, k, 0.0);
        // MatrixXd is column-major. Strict comparison selects its first tie.
        for col in k..n {
            for row in k..n {
                let magnitude = lu[row * n + col].abs();
                if magnitude > largest {
                    largest = magnitude;
                    pr = row;
                    pc = col;
                }
            }
        }
        if largest == 0.0 {
            break;
        }
        max_pivot = max_pivot.max(largest);
        for col in 0..n {
            lu.swap(k * n + col, pr * n + col);
        }
        rows.swap(k, pr);
        for row in 0..n {
            lu.swap(row * n + k, row * n + pc);
        }
        columns.swap(k, pc);
        let pivot = lu[k * n + k];
        for row in k + 1..n {
            lu[row * n + k] = finite(lu[row * n + k] / pivot)?;
        }
        for col in k + 1..n {
            for row in k + 1..n {
                lu[row * n + col] = finite(lu[row * n + col] - lu[row * n + k] * lu[k * n + col])?;
            }
        }
    }
    let threshold = max_pivot * (f64::EPSILON * n as f64);
    let rank = (0..n).filter(|&i| lu[i * n + i].abs() > threshold).count();
    let mut inverse = buffer(square, 0.0)?;
    let mut work = buffer(n, 0.0)?;
    for col in 0..n {
        // Apply the row permutation to an identity column, then unit-lower solve.
        for row in 0..n {
            work[row] = if rows[row] == col { 1.0 } else { 0.0 };
        }
        for k in 0..n {
            for row in k + 1..n {
                work[row] = finite(work[row] - lu[row * n + k] * work[k])?;
            }
        }
        // Non-pivot variables are zero; do not back-substitute against them.
        for k in (0..rank).rev() {
            work[k] = finite(work[k] / lu[k * n + k])?;
            for row in 0..k {
                work[row] = finite(work[row] - lu[row * n + k] * work[k])?;
            }
        }
        for row in 0..rank {
            inverse[columns[row] * n + col] = work[row];
        }
    }
    let mut result = buffer(n, 0.0)?;
    for row in 0..n {
        for col in 0..n {
            result[row] = finite(result[row] + -inverse[row * n + col] * rhs[col])?;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_pivot_rank_deficiency_and_ties() {
        // Two identical columns: column-major pivot order selects column 0.
        assert_eq!(
            negative_inverse_product(vec![1.0, 1.0, 1.0, 1.0], &[2.0, 2.0]).unwrap(),
            vec![-2.0, 0.0]
        );
        // A larger second column selects that feature, not a least-norm solution.
        assert_eq!(
            negative_inverse_product(vec![1.0, 2.0, 2.0, 4.0], &[3.0, 6.0]).unwrap(),
            vec![0.0, -1.5]
        );
        assert_eq!(
            negative_inverse_product(vec![0.0; 4], &[1.0, 2.0]).unwrap(),
            vec![0.0, 0.0]
        );
        assert_eq!(
            negative_inverse_product(vec![0.0, 1.0, 1.0, 0.0], &[2.0, 3.0]).unwrap(),
            vec![-3.0, -2.0]
        );
    }

    #[test]
    fn recorded_native_trees_isolate_fitting_from_split_search() {
        use serde_json::Value;
        for source in [
            include_str!("../tests/fixtures/prediction_linear.json"),
            include_str!("../tests/fixtures/prediction_linear_training.json"),
        ] {
            let suite: Value = serde_json::from_str(source).unwrap();
            for case in suite["cases"].as_array().unwrap() {
                let name = case["name"].as_str().unwrap();
                let pairs = case["params"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .filter(|(key, _)| key.as_str() != "forcedsplits_filename")
                    .map(|(key, value)| {
                        (
                            key.clone(),
                            value
                                .as_str()
                                .map(str::to_owned)
                                .unwrap_or_else(|| value.to_string()),
                        )
                    });
                let config = Config::from_pairs(pairs).unwrap();
                let x: Vec<f64> = case["training"]["rows"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|row| {
                        row.as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_f64().unwrap_or(f64::NAN))
                    })
                    .collect();
                let labels: Vec<f32> =
                    serde_json::from_value(case["training"]["labels"].clone()).unwrap();
                let n = labels.len();
                let data = Dataset::from_dense(&x, n, 2, labels.clone(), &config).unwrap();
                let weights: Vec<f32> = case["training"]
                    .get("weights")
                    .map(|v| serde_json::from_value(v.clone()).unwrap())
                    .unwrap_or_else(|| vec![1.0; n]);
                let bias = labels
                    .iter()
                    .zip(&weights)
                    .map(|(&y, &w)| f64::from(y) * f64::from(w))
                    .sum::<f64>()
                    / weights.iter().map(|&w| f64::from(w)).sum::<f64>();
                let mut scores = vec![bias; n];
                let native = crate::Model::from_text(
                    &std::fs::read_to_string(format!(
                        "{}/tests/fixtures/{}",
                        env!("CARGO_MANIFEST_DIR"),
                        case["model_file"].as_str().unwrap()
                    ))
                    .unwrap(),
                )
                .unwrap();
                for (index, expected) in native.trees().iter().enumerate() {
                    let gradients: Vec<f32> = (0..n)
                        .map(|r| (scores[r] - f64::from(labels[r])) as f32 * weights[r])
                        .collect();
                    let mut rows = vec![Vec::new(); expected.num_leaves()];
                    for row in 0..n {
                        rows[case["leaves"][row][index].as_u64().unwrap() as usize].push(row);
                    }
                    let mut tree = expected.clone();
                    tree.linear = None;
                    tree.shrinkage = 1.0;
                    for v in &mut tree.leaf_values {
                        *v = (*v - if index == 0 { bias } else { 0.0 }) / config.learning_rate;
                    }
                    fit_linear_leaves(
                        &mut tree,
                        &data,
                        &gradients,
                        &weights,
                        &rows,
                        &config,
                        index == 0,
                    )
                    .unwrap();
                    tree.scale_outputs(config.learning_rate);
                    if index == 0 {
                        tree.add_bias(bias);
                    }
                    let got = tree.linear_leaves().unwrap();
                    let want = expected.linear_leaves().unwrap();
                    assert_eq!(got.features(), want.features(), "{name} tree {index}");
                    for (&a, &e) in got.constants().iter().zip(want.constants()) {
                        assert!(
                            (a - e).abs() <= 1e-12 + 1e-10 * e.abs(),
                            "{name} tree {index} constant {a} != {e}"
                        );
                    }
                    for (a, e) in got.coefficients().iter().zip(want.coefficients()) {
                        for (&a, &e) in a.iter().zip(e) {
                            assert!(
                                (a - e).abs() <= 1e-12 + 1e-10 * e.abs(),
                                "{name} tree {index} coefficient {a} != {e}"
                            );
                        }
                    }
                    for row in 0..n {
                        let leaf = case["leaves"][row][index].as_u64().unwrap() as usize;
                        let a = predict_from_leaf(&tree, &data, row, leaf).unwrap();
                        let e = predict_from_leaf(expected, &data, row, leaf).unwrap();
                        assert!(
                            (a - e).abs() <= 1e-12 + 1e-10 * e.abs(),
                            "{name} tree {index} row {row}: {a} != {e}"
                        );
                        scores[row] += e - if index == 0 { bias } else { 0.0 };
                    }
                }
            }
        }
    }

    fn test_config() -> Config {
        Config {
            linear_tree: true,
            min_data_in_bin: 1,
            min_data_in_leaf: 1,
            feature_pre_filter: false,
            ..Config::default()
        }
    }

    fn two_feature_path() -> Tree {
        let mut tree = Tree::constant(99.0, 3);
        tree.leaf_values = vec![99.0; 3];
        tree.split_features = vec![0, 1];
        tree.decision_types = vec![0, 0];
        tree.left_children = vec![1, !0];
        tree.right_children = vec![!1, !2];
        tree
    }

    #[test]
    fn nan_counter_counts_features_before_rejected_rows() {
        let config = test_config();
        let data = Dataset::from_dense(
            &[-3.0, f64::NAN, -2.0, f64::NAN, -1.0, f64::NAN],
            3,
            2,
            vec![0.0; 3],
            &config,
        )
        .unwrap();
        let mut tree = two_feature_path();
        fit_linear_leaves(
            &mut tree,
            &data,
            &[1.0; 3],
            &[1.0; 3],
            &[vec![2, 0, 1], vec![], vec![]],
            &config,
            false,
        )
        .unwrap();
        // Three finite feature-0 values pass n=3 despite zero complete rows.
        // The singular intercept is set to zero rather than the fallback 99.
        assert_eq!(
            tree.linear_leaves().unwrap().constants(),
            &[0.0, 99.0, 99.0]
        );
        assert!(
            tree.linear_leaves()
                .unwrap()
                .features()
                .iter()
                .all(Vec::is_empty)
        );
        assert_eq!(tree.leaf_values, vec![99.0; 3]);
    }

    #[test]
    fn categorical_paths_are_excluded_and_repeated_features_are_unique() {
        let mut config = test_config();
        config.categorical_features = vec![1];
        let data = Dataset::from_dense(
            &[-3.0, 0.0, -2.0, 1.0, -1.0, 0.0],
            3,
            2,
            vec![0.0; 3],
            &config,
        )
        .unwrap();
        let mut tree = two_feature_path();
        tree.decision_types[1] = 1;
        assert_eq!(
            numeric_paths(&tree, &data).unwrap(),
            vec![vec![0], vec![0], vec![0]]
        );
        tree.split_features[1] = 0;
        tree.decision_types[1] = 0;
        assert_eq!(
            numeric_paths(&tree, &data).unwrap(),
            vec![vec![0], vec![0], vec![0]]
        );
    }

    #[test]
    fn raw_prediction_preserves_native_f32_and_nan_fallback() {
        let config = test_config();
        let data = Dataset::from_dense(&[1e-36, f64::NAN], 2, 1, vec![0.0; 2], &config).unwrap();
        let mut tree = Tree::constant(7.0, 2);
        tree.linear = Some(LinearLeaves {
            constants: vec![2.0],
            features: vec![vec![0]],
            coefficients: vec![vec![1e35]],
        });
        assert_eq!(
            predict_from_leaf(&tree, &data, 0, 0).unwrap(),
            2.0 + f64::from(1e-36_f32) * 1e35
        );
        assert_eq!(predict_from_leaf(&tree, &data, 1, 0).unwrap(), 7.0);
        assert!(predict_from_leaf(&tree, &data, 2, 0).is_err());
        assert!(predict_from_leaf(&tree, &data, 0, 1).is_err());
    }

    #[test]
    fn invalid_rows_and_derivatives_fail_without_mutating_tree() {
        let config = test_config();
        let data = Dataset::from_dense(&[1.0, 2.0], 2, 1, vec![0.0; 2], &config).unwrap();
        let mut tree = Tree::constant(7.0, 2);
        for rows in [vec![vec![2]], vec![vec![0, 0]], vec![]] {
            assert!(
                fit_linear_leaves(&mut tree, &data, &[1.0; 2], &[1.0; 2], &rows, &config, true)
                    .is_err()
            );
            assert!(tree.linear.is_none());
        }
        assert!(
            fit_linear_leaves(
                &mut tree,
                &data,
                &[f32::NAN; 2],
                &[1.0; 2],
                &[vec![0]],
                &config,
                false
            )
            .is_err()
        );
        assert!(tree.linear.is_none());
    }

    #[test]
    fn pruning_uses_the_native_f32_threshold_and_strict_comparison() {
        let config = test_config();
        let data = Dataset::from_dense(&[-1.0, 1.0], 2, 1, vec![0.0; 2], &config).unwrap();
        for (amplitude, retained) in [
            (1e-35_f32, false),
            (f32::from_bits(1e-35_f32.to_bits() + 1), true),
        ] {
            let mut tree = Tree::constant(9.0, 2);
            tree.leaf_values = vec![9.0; 2];
            tree.split_features = vec![0];
            tree.decision_types = vec![0];
            tree.left_children = vec![!0];
            tree.right_children = vec![!1];
            fit_linear_leaves(
                &mut tree,
                &data,
                &[amplitude, -amplitude],
                &[1.0; 2],
                &[vec![0, 1], vec![]],
                &config,
                false,
            )
            .unwrap();
            assert_eq!(
                !tree.linear_leaves().unwrap().features()[0].is_empty(),
                retained
            );
        }
    }

    #[test]
    fn raw_f32_overflow_returns_an_error_not_a_nonfinite_model() {
        let config = test_config();
        let data = Dataset::from_dense(&[f64::MAX, 1.0], 2, 1, vec![0.0; 2], &config).unwrap();
        let mut tree = Tree::constant(9.0, 2);
        tree.leaf_values = vec![9.0; 2];
        tree.split_features = vec![0];
        tree.decision_types = vec![0];
        tree.left_children = vec![!0];
        tree.right_children = vec![!1];
        let error = fit_linear_leaves(
            &mut tree,
            &data,
            &[1.0; 2],
            &[1.0; 2],
            &[vec![0, 1], vec![]],
            &config,
            false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("nonfinite"));
        assert!(tree.linear.is_none());
    }

    #[test]
    fn undersampled_paths_skip_an_oversized_solver_before_allocation() {
        let depth = 512;
        assert!(checked_solver_size(depth + 1).is_err());
        let config = test_config();
        let data = Dataset::from_dense(&vec![1.0; depth], 1, depth, vec![0.0], &config).unwrap();
        let mut tree = Tree::constant(9.0, 1);
        tree.leaf_values = vec![9.0; depth + 1];
        tree.split_features = (0..depth).collect();
        tree.decision_types = vec![0; depth];
        tree.right_children = (0..depth).map(|i| !(i as i32)).collect();
        tree.left_children = (1..depth)
            .map(|i| i as i32)
            .chain([!(depth as i32)])
            .collect();
        let mut rows = vec![Vec::new(); depth + 1];
        rows[depth] = vec![0];
        fit_linear_leaves(&mut tree, &data, &[1.0], &[1.0], &rows, &config, false).unwrap();
        assert_eq!(
            tree.linear_leaves().unwrap().constants(),
            &vec![9.0; depth + 1]
        );
    }

    #[test]
    fn rank_threshold_is_relative_to_largest_pivot() {
        let solve =
            |small| negative_inverse_product(vec![1.0, 0.0, 0.0, small], &[1.0, small]).unwrap();
        assert_eq!(solve(f64::EPSILON), vec![-1.0, 0.0]);
        assert_eq!(solve(4.0 * f64::EPSILON), vec![-1.0, -1.0]);
        assert!(checked_solver_size(usize::MAX).is_err());
        assert!(checked_solver_size(4096).is_err());
        assert!(negative_inverse_product(vec![f64::INFINITY], &[1.0]).is_err());
    }
}
