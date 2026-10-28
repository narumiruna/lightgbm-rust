//! Monotone cross-branch constraint traversal, from LightGBM (MIT).
//! Copyright (c) 2020-2026 Microsoft Corporation and The LightGBM developers.
use crate::histogram::Split;
use crate::{Config, Tree};

type Path = Vec<(usize, u32, bool)>;
fn keep(tree: &Tree, bins: &[u32], node: usize, path: &Path) -> (bool, bool) {
    let (mut left, mut right) = (true, true);
    if tree.decision_types[node] & 1 == 0 {
        for &(feature, threshold, is_right) in path {
            if feature == tree.split_features[node] {
                if bins[node] >= threshold && !is_right {
                    right = false;
                }
                if bins[node] <= threshold && is_right {
                    left = false;
                }
            }
        }
    }
    (left, right)
}
fn ancestors(tree: &Tree) -> (Vec<Option<usize>>, Vec<Option<usize>>) {
    let mut nodes = vec![None; tree.split_features.len()];
    let mut leaves = vec![None; tree.num_leaves()];
    for node in 0..nodes.len() {
        for child in [tree.left_children[node], tree.right_children[node]] {
            if child < 0 {
                leaves[(!child) as usize] = Some(node);
            } else {
                nodes[child as usize] = Some(node);
            }
        }
    }
    (nodes, leaves)
}

/// Tighten existing leaves after a split. Children inherit their own bounds
/// separately. The path scan and visit order are the serial reference order.
pub(super) fn updates(
    tree: &Tree,
    bins: &[u32],
    new_node: usize,
    split: &Split,
    config: &Config,
) -> Vec<(usize, bool, f64)> {
    let (parents, _) = ancestors(tree);
    let mut node = new_node;
    let mut path = Vec::new();
    let mut result = Vec::new();
    while let Some(parent) = parents[node] {
        let feature = tree.split_features[parent];
        let right = tree.right_children[parent] == node as i32;
        let numerical = tree.decision_types[parent] & 1 == 0;
        if numerical && !path.iter().any(|&(f, _, r)| f == feature && r == right) {
            let monotone = config
                .monotone_constraints
                .get(feature)
                .copied()
                .unwrap_or(0);
            if monotone != 0 {
                let opposite = if right {
                    tree.left_children[parent]
                } else {
                    tree.right_children[parent]
                };
                let max = if monotone < 0 { !right } else { right };
                update_down(
                    tree,
                    bins,
                    opposite,
                    &path,
                    max,
                    split,
                    true,
                    true,
                    &mut result,
                );
            }
            path.push((feature, bins[parent], right));
        }
        node = parent;
    }
    result
}
#[allow(clippy::too_many_arguments)]
fn update_down(
    tree: &Tree,
    bins: &[u32],
    node: i32,
    path: &Path,
    max: bool,
    split: &Split,
    use_left: bool,
    use_right: bool,
    result: &mut Vec<(usize, bool, f64)>,
) {
    if node < 0 {
        let value = if use_left && use_right {
            if max {
                split.left_output.min(split.right_output)
            } else {
                split.left_output.max(split.right_output)
            }
        } else if use_left {
            split.left_output
        } else {
            split.right_output
        };
        result.push(((!node) as usize, max, value));
        return;
    }
    let node = node as usize;
    let (left, right) = keep(tree, bins, node, path);
    let same = tree.decision_types[node] & 1 == 0 && tree.split_features[node] == split.feature;
    if left {
        update_down(
            tree,
            bins,
            tree.left_children[node],
            path,
            max,
            split,
            use_left,
            use_right && (!same || bins[node] > split.threshold),
            result,
        );
    }
    if right {
        update_down(
            tree,
            bins,
            tree.right_children[node],
            path,
            max,
            split,
            use_left && (!same || bins[node] < split.threshold),
            use_right,
            result,
        );
    }
}

/// Rebuild the precise method's per-bin limits from contiguous leaves. Prefix
/// and suffix extrema then give separate limits for each candidate child.
pub(super) fn feature_bounds(
    tree: &Tree,
    bins: &[u32],
    leaf: usize,
    feature: usize,
    num_bins: usize,
    config: &Config,
) -> Vec<[(f64, f64); 2]> {
    let (parents, leaf_parents) = ancestors(tree);
    let mut limits = vec![(f64::NEG_INFINITY, f64::INFINITY); num_bins];
    for minimum in [true, false] {
        let mut current = !(leaf as i32);
        let mut parent = leaf_parents[leaf];
        let mut path = Vec::new();
        let (mut start, mut end) = (0, num_bins as u32);
        while let Some(node) = parent {
            let f = tree.split_features[node];
            let numerical = tree.decision_types[node] & 1 == 0;
            let right = tree.right_children[node] == current;
            if numerical && f == feature {
                if right {
                    start = start.max(bins[node]);
                } else {
                    end = end.min(bins[node] + 1);
                }
            }
            if numerical && !path.iter().any(|&(pf, _, pr)| pf == f && pr == right) {
                let monotone = config.monotone_constraints.get(f).copied().unwrap_or(0);
                let update_min = if monotone < 0 { !right } else { right };
                if monotone != 0 && update_min == minimum {
                    let opposite = if right {
                        tree.left_children[node]
                    } else {
                        tree.right_children[node]
                    };
                    bounds_down(
                        tree,
                        bins,
                        opposite,
                        &path,
                        feature,
                        f,
                        minimum,
                        start,
                        end,
                        config,
                        &mut limits,
                    );
                }
                path.push((f, bins[node], right));
            }
            current = node as i32;
            parent = parents[node];
        }
    }
    let mut prefix = limits.clone();
    let mut suffix = limits;
    for i in 1..num_bins {
        prefix[i].0 = prefix[i].0.max(prefix[i - 1].0);
        prefix[i].1 = prefix[i].1.min(prefix[i - 1].1);
    }
    for i in (0..num_bins.saturating_sub(1)).rev() {
        suffix[i].0 = suffix[i].0.max(suffix[i + 1].0);
        suffix[i].1 = suffix[i].1.min(suffix[i + 1].1);
    }
    (0..num_bins)
        .map(|t| [prefix[t], suffix[(t + 1).min(num_bins - 1)]])
        .collect()
}
#[allow(clippy::too_many_arguments)]
fn bounds_down(
    tree: &Tree,
    bins: &[u32],
    node: i32,
    path: &Path,
    feature: usize,
    root_feature: usize,
    minimum: bool,
    start: u32,
    end: u32,
    config: &Config,
    limits: &mut [(f64, f64)],
) {
    if start >= end {
        return;
    }
    if node < 0 {
        let value = tree.leaf_values[(!node) as usize];
        for bound in &mut limits[start as usize..end as usize] {
            if minimum {
                bound.0 = bound.0.max(value);
            } else {
                bound.1 = bound.1.min(value);
            }
        }
        return;
    }
    let node = node as usize;
    let (left, right) = keep(tree, bins, node, path);
    let f = tree.split_features[node];
    let same = f == feature;
    let monotone = config.monotone_constraints.get(f).copied().unwrap_or(0);
    let (relevant_left, relevant_right) = if (same && root_feature != feature) || monotone == 0 {
        (true, true)
    } else if (monotone < 0 && minimum) || (monotone > 0 && !minimum) {
        (true, false)
    } else {
        (false, true)
    };
    if left && (relevant_left || !right) {
        bounds_down(
            tree,
            bins,
            tree.left_children[node],
            path,
            feature,
            root_feature,
            minimum,
            start,
            if same { end.min(bins[node] + 1) } else { end },
            config,
            limits,
        );
    }
    if right && (relevant_right || !left) {
        bounds_down(
            tree,
            bins,
            tree.right_children[node],
            path,
            feature,
            root_feature,
            minimum,
            if same {
                start.max(bins[node] + 1)
            } else {
                start
            },
            end,
            config,
            limits,
        );
    }
}
