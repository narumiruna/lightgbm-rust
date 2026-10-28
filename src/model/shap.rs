//! Exact count-weighted TreeSHAP and native split/gain importance.
//!
//! Translated from LightGBM commit 5d84f3f222b41409a6983d2ac90513ad9702fa9e,
//! `src/io/tree.cpp`, `include/LightGBM/tree.h`, and
//! `src/boosting/{gbdt.cpp,gbdt_model_text.cpp}`.
//!
//! Copyright (c) 2016-2026 Microsoft Corporation and The LightGBM developers.
//! Licensed under the MIT License.
//!
//! Tree walks are iterative. Contributions support tree depth at most 256 and
//! a per-call path-work budget of 67,108,864 units, where each visited node or
//! leaf costs `(depth + 1)^2` units. These limits bound the quadratic path
//! workspace and work before numerical traversal. The output limit is
//! 16,777,216 entries, including every class bias. Split trees require positive
//! sample counts with each parent equal to the sum of its children. Hessian
//! weights are not a substitute for absent counts.

use super::{IterationRange, Model, Tree};
use crate::{Error, Result};

const MAX_OUTPUT_ENTRIES: usize = 16 * 1024 * 1024;
const MAX_DEPTH: usize = 256;
const MAX_PATH_WORK: usize = 64 * 1024 * 1024;

/// The native feature importance measure. Only strictly positive gains count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImportanceType {
    /// Number of splits with positive gain, in original feature order.
    Split,
    /// Sum of positive gains, in original feature order.
    Gain,
}

pub(super) fn importance(
    model: &Model,
    kind: ImportanceType,
    iterations: IterationRange,
) -> Result<Vec<f64>> {
    let range = model.tree_range(iterations)?;
    let mut output = vec![0.0; model.num_features()];
    for tree in &model.trees[range] {
        for (&feature, &gain) in tree.split_features.iter().zip(&tree.split_gains) {
            // Text import already converts gains to the native f32 storage.
            if gain > 0.0 {
                output[feature] += match kind {
                    ImportanceType::Split => 1.0,
                    ImportanceType::Gain => gain,
                };
            }
        }
    }
    // Native f32 gain storage can contain positive infinity even when every
    // prediction is finite. Importance preserves that diagnostic value.
    Ok(output)
}

pub(super) fn contributions(
    model: &Model,
    get: &impl Fn(usize) -> f64,
    iterations: IterationRange,
) -> Result<Vec<Vec<f64>>> {
    let range = model.tree_range(iterations)?;
    let width = model.num_features().checked_add(1);
    let entries = width.and_then(|width| width.checked_mul(model.num_class));
    if entries.is_none_or(|entries| entries > MAX_OUTPUT_ENTRIES) {
        return Err(Error::Unsupported(
            "TreeSHAP output exceeds 16,777,216 entries".into(),
        ));
    }
    // Native contribution prediction rejects linear models, including a first
    // tree with empty coefficient arrays. Selection does not remove this flag.
    if model.trees.iter().any(|tree| tree.linear.is_some()) {
        return Err(Error::Unsupported(
            "TreeSHAP contributions are not supported for linear trees".into(),
        ));
    }
    let mut work = 0;
    for tree in &model.trees[range.clone()] {
        check_tree(tree, &mut work)?;
    }
    let mut output = vec![vec![0.0; width.unwrap()]; model.num_class];
    for (offset, tree) in model.trees[range].iter().enumerate() {
        let class = offset % model.num_tree_per_iteration;
        let phi = &mut output[class];
        phi[model.num_features()] += expected_value(tree);
        if tree.num_leaves() > 1 {
            tree_shap(tree, get, phi);
        }
    }
    // RF contribution prediction, like raw prediction, is a sum. Native only
    // divides by the iteration count in its transformed prediction method.
    if output.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Error::Unsupported(
            "TreeSHAP path or contribution exceeds finite numerical range".into(),
        ));
    }
    Ok(output)
}

fn count(tree: &Tree, child: i32) -> usize {
    if child < 0 {
        tree.leaf_counts[(!child) as usize]
    } else {
        tree.internal_counts[child as usize]
    }
}

fn check_tree(tree: &Tree, work: &mut usize) -> Result<()> {
    if tree.num_leaves() == 1 {
        return Ok(());
    }
    let mut pending = vec![(0i32, 0usize)];
    while let Some((child, depth)) = pending.pop() {
        if depth > MAX_DEPTH {
            return Err(Error::Unsupported("TreeSHAP tree depth exceeds 256".into()));
        }
        *work = work
            .checked_add((depth + 1) * (depth + 1))
            .ok_or_else(|| Error::Unsupported("TreeSHAP path-work budget overflow".into()))?;
        if *work > MAX_PATH_WORK {
            return Err(Error::Unsupported(
                "TreeSHAP path-work budget exceeds 67,108,864 units".into(),
            ));
        }
        if child < 0 {
            continue;
        }
        let node = child as usize;
        let left = tree.left_children[node];
        let right = tree.right_children[node];
        let left_count = count(tree, left);
        let right_count = count(tree, right);
        if left_count == 0
            || right_count == 0
            || left_count.checked_add(right_count) != Some(tree.internal_counts[node])
        {
            return Err(Error::InvalidModel(
                "TreeSHAP requires positive sample counts and consistent parent/child counts"
                    .into(),
            ));
        }
        pending.push((right, depth + 1));
        pending.push((left, depth + 1));
    }
    Ok(())
}

fn expected_value(tree: &Tree) -> f64 {
    if tree.num_leaves() == 1 {
        return tree.leaf_values[0];
    }
    let total_count = tree.internal_counts[0] as f64;
    let mut expected = 0.0;
    // Preserve native leaf-array accumulation order, not recursive expectations
    // or Hessian weights. Leaf values already contain shrinkage and baseline.
    for (&count, &value) in tree.leaf_counts.iter().zip(&tree.leaf_values) {
        expected += (count as f64 / total_count) * value;
    }
    expected
}

#[derive(Clone, Copy)]
struct PathElement {
    feature: usize,
    zero: f64,
    one: f64,
    weight: f64,
}

impl PathElement {
    fn new(feature: usize, zero: f64, one: f64) -> Self {
        Self {
            feature,
            zero,
            one,
            weight: 0.0,
        }
    }
}

fn extend_path(path: &mut Vec<PathElement>, mut incoming: PathElement) {
    let depth = path.len();
    incoming.weight = if depth == 0 { 1.0 } else { 0.0 };
    path.push(incoming);
    for i in (0..depth).rev() {
        path[i + 1].weight += incoming.one * path[i].weight * (i + 1) as f64 / (depth + 1) as f64;
        path[i].weight = incoming.zero * path[i].weight * (depth - i) as f64 / (depth + 1) as f64;
    }
}

fn unwind_path(path: &mut Vec<PathElement>, index: usize) {
    let depth = path.len() - 1;
    let element = path[index];
    let mut next_one = path[depth].weight;
    for i in (0..depth).rev() {
        if element.one != 0.0 {
            let previous = path[i].weight;
            path[i].weight = next_one * (depth + 1) as f64 / ((i + 1) as f64 * element.one);
            next_one =
                previous - path[i].weight * element.zero * (depth - i) as f64 / (depth + 1) as f64;
        } else {
            path[i].weight =
                (path[i].weight * (depth + 1) as f64) / (element.zero * (depth - i) as f64);
        }
    }
    for i in index..depth {
        // Weights belong to permutation sizes, not feature identities. Do not
        // shift the weights when removing a repeated feature from the path.
        path[i].feature = path[i + 1].feature;
        path[i].zero = path[i + 1].zero;
        path[i].one = path[i + 1].one;
    }
    path.pop();
}

fn unwound_path_sum(path: &[PathElement], index: usize) -> f64 {
    let depth = path.len() - 1;
    let element = path[index];
    let mut next_one = path[depth].weight;
    let mut total = 0.0;
    for i in (0..depth).rev() {
        if element.one != 0.0 {
            let portion = next_one * (depth + 1) as f64 / ((i + 1) as f64 * element.one);
            total += portion;
            next_one =
                path[i].weight - portion * element.zero * ((depth - i) as f64 / (depth + 1) as f64);
        } else {
            total += (path[i].weight / element.zero) / ((depth - i) as f64 / (depth + 1) as f64);
        }
    }
    total
}

struct Task {
    child: i32,
    path: Vec<PathElement>,
    incoming: PathElement,
}

fn tree_shap(tree: &Tree, get: &impl Fn(usize) -> f64, phi: &mut [f64]) {
    let mut pending = vec![Task {
        child: 0,
        path: Vec::new(),
        incoming: PathElement::new(usize::MAX, 1.0, 1.0),
    }];
    while let Some(Task {
        child,
        mut path,
        incoming,
    }) = pending.pop()
    {
        extend_path(&mut path, incoming);
        if child < 0 {
            let value = tree.leaf_values[(!child) as usize];
            for i in 1..path.len() {
                let weight = unwound_path_sum(&path, i);
                let element = path[i];
                phi[element.feature] += weight * (element.one - element.zero) * value;
            }
            continue;
        }
        let node = child as usize;
        let (hot, cold) = if tree.goes_left_with(node, get) {
            (tree.left_children[node], tree.right_children[node])
        } else {
            (tree.right_children[node], tree.left_children[node])
        };
        let total_count = tree.internal_counts[node] as f64;
        let hot_zero = count(tree, hot) as f64 / total_count;
        let cold_zero = count(tree, cold) as f64 / total_count;
        let feature = tree.split_features[node];
        let mut incoming_zero = 1.0;
        let mut incoming_one = 1.0;
        if let Some(index) = path.iter().position(|element| element.feature == feature) {
            incoming_zero = path[index].zero;
            incoming_one = path[index].one;
            unwind_path(&mut path, index);
        }
        // LIFO preserves the native hot-then-cold accumulation order. At most
        // depth+1 pending paths are live; each has at most depth+1 elements.
        pending.push(Task {
            child: cold,
            path: path.clone(),
            incoming: PathElement::new(feature, cold_zero * incoming_zero, 0.0),
        });
        pending.push(Task {
            child: hot,
            path,
            incoming: PathElement::new(feature, hot_zero * incoming_zero, incoming_one),
        });
    }
}
