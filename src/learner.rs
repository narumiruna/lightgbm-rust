//! Best-first CPU tree learner with ordered feature-parallel kernels. Ported from LightGBM commit
//! 5d84f3f222b41409a6983d2ac90513ad9702fa9e (MIT).
//! Copyright (c) 2016-2026 Microsoft Corporation and The LightGBM developers.
use crate::data::MissingType;
use crate::histogram::{self, Split, TrainingStats as BinStats, leaf_output};
use crate::model::Tree;
use crate::{Config, Dataset, Error, Result};

#[path = "cache.rs"]
mod cache;
#[path = "constraints.rs"]
mod constraints;
#[path = "quantize.rs"]
mod quantize;
use crate::config::MonotoneMethod;
use crate::histogram::SplitControl;
use crate::random::Random;
use quantize::{Quantized, Quantizer};
pub(crate) struct TreeLearner {
    feature_random: Random,
    extra_random: Vec<Random>,
    eligible: Vec<usize>,
    tree_features: Vec<usize>,
    tree_feature_mask: Vec<bool>,
    coupled: Vec<bool>,
    lazy: Vec<Vec<bool>>,
    quantizer: Option<Quantizer>,
    executor: crate::parallel::Executor,
}
impl TreeLearner {
    pub(crate) fn new(config: &Config, data: &Dataset) -> Result<Self> {
        let eligible: Vec<usize> = (0..data.ncols())
            .filter(|&f| !data.mapper(f).is_trivial())
            .collect();
        if !config.cegb_penalty_feature_lazy.is_empty()
            && data
                .nrows()
                .checked_mul(data.ncols())
                .is_none_or(|n| n > 16 * 1024 * 1024)
        {
            return Err(Error::Unsupported(
                "CEGB lazy state exceeds 16 Mi feature-row pairs".into(),
            ));
        }
        let mut learner = Self {
            executor: crate::parallel::Executor::new(config.num_threads)?,
            quantizer: if config.use_quantized_grad {
                Some(Quantizer::new(config.seed, data.nrows()))
            } else {
                None
            },
            feature_random: Random::new(config.feature_fraction_seed),
            extra_random: eligible
                .iter()
                .map(|&f| {
                    Random::new(
                        config
                            .extra_seed
                            .wrapping_add(data.inner_feature_index(f) as u32),
                    )
                })
                .collect(),
            tree_features: Vec::new(),
            tree_feature_mask: vec![false; data.ncols()],
            eligible,
            coupled: vec![false; data.ncols()],
            lazy: if config.cegb_penalty_feature_lazy.is_empty() {
                Vec::new()
            } else {
                vec![vec![false; data.nrows()]; data.ncols()]
            },
        };
        learner.sample_tree(config)?;
        Ok(learner)
    }
    /// Native ColSampler::SetConfig consumes a new tree sample even when
    /// only the learning rate changes. Preserve the existing random stream.
    pub(crate) fn reset_learning_rate_config(&mut self, config: &Config) -> Result<()> {
        self.sample_tree(config)
    }

    fn sample_tree(&mut self, config: &Config) -> Result<()> {
        self.tree_features = if config.feature_fraction < 1.0 {
            self.feature_random
                .sample(
                    self.eligible.len(),
                    feature_count(self.eligible.len(), config.feature_fraction),
                )?
                .into_iter()
                .map(|i| self.eligible[i])
                .collect()
        } else {
            self.eligible.clone()
        };
        // Keep tree_features ordered for node sampling; reuse this mask for split scans.
        self.tree_feature_mask.fill(false);
        for &feature in &self.tree_features {
            self.tree_feature_mask[feature] = true;
        }
        Ok(())
    }
    fn sample_node(&mut self, leaf: &Leaf, config: &Config) -> Result<Vec<bool>> {
        let allowed = |f: usize| {
            config.interaction_constraints.is_empty()
                || leaf.branch.contains(&f)
                || config
                    .interaction_constraints
                    .iter()
                    .any(|g| g.contains(&f) && leaf.branch.iter().all(|f| g.contains(f)))
        };
        if config.feature_fraction_bynode >= 1.0 {
            let mut mask = vec![false; self.coupled.len()];
            for &f in &self.eligible {
                mask[f] = allowed(f);
            }
            return Ok(mask);
        }
        let features: Vec<usize> = self
            .tree_features
            .iter()
            .copied()
            .filter(|&f| allowed(f))
            .collect();
        let count = feature_count(self.tree_features.len(), config.feature_fraction_bynode)
            .min(features.len());
        let selected = if config.feature_fraction_bynode < 1.0 {
            self.feature_random.sample(features.len(), count)?
        } else {
            (0..features.len()).collect()
        };
        let mut mask = vec![false; self.coupled.len()];
        for i in selected {
            mask[features[i]] = true;
        }
        Ok(mask)
    }
    pub(crate) fn fit(
        &mut self,
        data: &Dataset,
        grad: &[f32],
        hess: &[f32],
        config: &Config,
        rows: &[usize],
        constant_hessian: bool,
    ) -> Result<FittedTree> {
        self.sample_tree(config)?;
        let quantized = if let Some(q) = &mut self.quantizer {
            Some(q.discretize(grad, hess, constant_hessian, config)?)
        } else {
            None
        };
        fit_tree(
            self,
            data,
            grad,
            hess,
            config,
            rows,
            constant_hessian,
            quantized.as_ref(),
        )
    }
}
fn feature_count(n: usize, f: f64) -> usize {
    ((n as f64 * f).round() as usize).max(usize::from(n > 0))
}
pub(crate) struct FittedTree {
    pub tree: Tree,
    pub threshold_bins: Vec<u32>,
    pub leaf_indices: Vec<usize>,
    pub leaf_rows: Vec<Vec<usize>>,
}

impl FittedTree {
    pub(crate) fn constant(rows: usize) -> Self {
        Self {
            tree: Tree::constant(0.0, rows),
            threshold_bins: Vec::new(),
            leaf_indices: Vec::new(),
            leaf_rows: Vec::new(),
        }
    }
}

struct Leaf {
    rows: Vec<usize>,
    depth: usize,
    gradient: f64,
    hessian: f64,
    histograms: Vec<Vec<BinStats>>,
    split: Option<Split>,
    candidates: Vec<Option<Split>>,
    active: Vec<bool>,
    branch: Vec<usize>,
    bounds: (f64, f64),
    parent_output: f64,
    advanced_bounds: Vec<Vec<[(f64, f64); 2]>>,
    quant_scales: Option<(f64, f64)>,
}

fn find_splits(
    learner: &mut TreeLearner,
    data: &Dataset,
    first: &mut Leaf,
    mut second: Option<&mut Leaf>,
    config: &Config,
) -> Result<()> {
    if (config.max_depth > 0 && first.depth >= config.max_depth as usize)
        || (first.rows.len() < 2 * config.min_data_in_leaf
            && second
                .as_ref()
                .is_none_or(|l| l.rows.len() < 2 * config.min_data_in_leaf))
    {
        return Ok(());
    }
    let first_mask = learner.sample_node(first, config)?;
    let second_mask = if let Some(leaf) = second.as_ref() {
        learner.sample_node(leaf, config)?
    } else {
        Vec::new()
    };
    // Draw ExtraTrees thresholds in the original feature/first/second order.
    // Workers read immutable RNG-independent jobs and never mutate learner state.
    let mut jobs = Vec::new();
    for inner in 0..learner.eligible.len() {
        let feature = learner.eligible[inner];
        if !learner.tree_feature_mask[feature] || !first.active[feature] {
            continue;
        }
        let a = extra_threshold(
            &mut learner.extra_random[inner],
            data,
            first,
            feature,
            config,
        )?;
        let b = if let Some(leaf) = second.as_ref() {
            extra_threshold(
                &mut learner.extra_random[inner],
                data,
                leaf,
                feature,
                config,
            )?
        } else {
            None
        };
        jobs.push((feature, a, b));
    }
    // Small leaves do not amortize a pool dispatch. Parallelize only coarse
    // feature scans; this does not change which jobs or thresholds are used.
    let minimum = if first.rows.len() >= 2048 {
        4
    } else {
        usize::MAX
    };
    let results = learner.executor.map(jobs.len(), minimum, |index| {
        let (feature, a, b) = jobs[index];
        let first_result = evaluate_feature(learner, data, first, feature, a, config);
        let second_result = second
            .as_ref()
            .map(|leaf| evaluate_feature(learner, data, leaf, feature, b, config));
        (feature, first_result, second_result)
    });
    // Ordered reduction preserves ties and all CEGB/constraint state updates.
    for (feature, a, b) in results {
        apply_feature(first, feature, first_mask[feature], a);
        if let Some(leaf) = second.as_deref_mut() {
            apply_feature(
                leaf,
                feature,
                second_mask[feature],
                b.expect("second result exists"),
            );
        }
    }
    Ok(())
}

fn extra_threshold(
    random: &mut Random,
    data: &Dataset,
    leaf: &Leaf,
    feature: usize,
    config: &Config,
) -> Result<Option<u32>> {
    if !config.extra_trees || (config.max_depth > 0 && leaf.depth >= config.max_depth as usize) {
        return Ok(None);
    }
    let mapper = data.mapper(feature);
    Ok(Some(if mapper.is_categorical() {
        if mapper.num_bins() as usize <= config.max_cat_to_onehot {
            let offset = i32::from(mapper.most_freq_bin() == 0);
            (random.next_int(1 - offset, mapper.num_bins() as i32 - offset)? + offset) as u32
        } else {
            let factor = leaf.rows.len() as f64 / (leaf.hessian + 2.0 * crate::objective::EPSILON);
            let used = leaf.histograms[feature]
                .iter()
                .skip(1)
                .filter(|b| (b.hessian * factor).round() >= config.cat_smooth)
                .count();
            let max = config
                .max_cat_threshold
                .min(used.div_ceil(2))
                .min(used)
                .saturating_sub(1);
            if max > 0 {
                random.next_int(0, max as i32)? as u32
            } else {
                0
            }
        }
    } else if mapper.num_bins() > 2 {
        random.next_int(0, mapper.num_bins() as i32 - 2)? as u32
    } else {
        0
    }))
}

fn apply_feature(
    leaf: &mut Leaf,
    feature: usize,
    enabled: bool,
    result: (Option<Split>, Option<Split>),
) {
    if let (Some(candidate), Some(split)) = result {
        leaf.candidates[feature] = Some(candidate);
        if enabled
            && leaf.split.as_ref().is_none_or(|b| {
                split.gain > b.gain || (split.gain == b.gain && feature < b.feature)
            })
        {
            leaf.split = Some(split);
        }
    }
}

fn evaluate_feature(
    learner: &TreeLearner,
    data: &Dataset,
    leaf: &Leaf,
    feature: usize,
    threshold: Option<u32>,
    config: &Config,
) -> (Option<Split>, Option<Split>) {
    if config.max_depth > 0 && leaf.depth >= config.max_depth as usize {
        return (None, None);
    }
    let mapper = data.mapper(feature);
    let control = SplitControl {
        threshold,
        parent_output: leaf.parent_output,
        monotone: config
            .monotone_constraints
            .get(feature)
            .copied()
            .unwrap_or(0),
        bounds: if config.monotone_constraints.is_empty() {
            None
        } else {
            Some(leaf.bounds)
        },
        threshold_bounds: leaf
            .advanced_bounds
            .get(feature)
            .cloned()
            .unwrap_or_default(),
    };
    let split = if mapper.is_categorical() {
        histogram::best_categorical_split_controlled(
            &leaf.histograms[feature],
            feature,
            leaf.gradient,
            leaf.hessian,
            leaf.rows.len(),
            config,
            &control,
        )
    } else if let Some((gs, hs)) = leaf.quant_scales {
        histogram::best_quantized_split(
            &leaf.histograms[feature],
            feature,
            mapper.missing_type(),
            mapper.default_bin(),
            mapper.most_freq_bin() == 0,
            leaf.gradient,
            leaf.hessian,
            leaf.rows.len(),
            config,
            &control,
            gs,
            hs,
        )
    } else {
        histogram::best_split_controlled(
            &leaf.histograms[feature],
            feature,
            mapper.missing_type(),
            mapper.default_bin(),
            mapper.most_freq_bin() == 0,
            leaf.gradient,
            leaf.hessian,
            leaf.rows.len(),
            config,
            &control,
        )
    };
    if let Some(mut split) = split {
        split.gain *= config.feature_contri.get(feature).copied().unwrap_or(1.0);
        // Pinned native DeltaGain stores SplitInfo by value before the caller
        // deducts CEGB costs or applies the monotone penalty. Keep that raw
        // candidate: pending-leaf refunds depend on this inherited behavior.
        let candidate = split.clone();
        let mut penalty = config.cegb_penalty_split * leaf.rows.len() as f64;
        if !learner.coupled[feature] {
            penalty += config
                .cegb_penalty_feature_coupled
                .get(feature)
                .copied()
                .unwrap_or(0.0);
        }
        if let Some(&cost) = config.cegb_penalty_feature_lazy.get(feature) {
            for &row in &leaf.rows {
                if !learner.lazy[feature][row] {
                    penalty += cost;
                }
            }
        }
        split.gain -= config.cegb_tradeoff * penalty;
        if control.monotone != 0 && config.monotone_penalty > 0.0 {
            let d = leaf.depth as f64;
            let p = config.monotone_penalty;
            let eps = crate::objective::EPSILON;
            split.gain *= if p >= d + 1.0 {
                eps
            } else if p <= 1.0 {
                1.0 - p / 2.0_f64.powf(d) + eps
            } else {
                1.0 - 2.0_f64.powf(p - 1.0 - d) + eps
            };
        }
        return (Some(candidate), Some(split));
    }
    (None, None)
}
#[allow(clippy::too_many_arguments)]
fn fit_tree(
    learner: &mut TreeLearner,
    data: &Dataset,
    gradients: &[f32],
    hessians: &[f32],
    config: &Config,
    rows: &[usize],
    constant_hessian: bool,
    quantized: Option<&Quantized>,
) -> Result<FittedTree> {
    if gradients.len() != data.nrows()
        || hessians.len() != data.nrows()
        || gradients.iter().any(|g| !g.is_finite())
        || hessians.iter().any(|h| !h.is_finite() || *h < 0.0)
    {
        return Err(Error::InvalidInput(
            "invalid tree derivative layout or values".into(),
        ));
    }
    if rows.iter().any(|&r| r >= data.nrows()) {
        return Err(Error::InvalidInput("tree row index out of range".into()));
    }
    let mut seen = vec![false; data.nrows()];
    for &row in rows {
        if seen[row] {
            return Err(Error::InvalidInput("duplicate tree row index".into()));
        }
        seen[row] = true;
    }
    let histogram_bytes = histogram::histogram_bytes(data)?;
    let budget = cache::HistogramBudget::new(config, histogram_bytes)?;
    // No histogram can be evicted when the cache holds every possible leaf.
    // In that case replay would only duplicate row partitions that stay live.
    let retain_replay = budget.slots < config.num_leaves;
    let mut recipe_bytes = 0;
    let mut replay = Vec::new();
    let rows = rows.to_vec();
    let (gradient, hessian) = if let Some(q) = quantized {
        (
            rows.iter().map(|&r| q.gradient[r]).sum::<f64>() * q.gradient_scale,
            rows.iter().map(|&r| q.hessian[r]).sum::<f64>() * q.hessian_scale,
        )
    } else {
        (
            rows.iter().map(|&r| f64::from(gradients[r])).sum(),
            rows.iter().map(|&r| f64::from(hessians[r])).sum(),
        )
    };
    let value = if hessian > 0.0 {
        leaf_output(gradient, hessian, config)
    } else {
        0.0
    };
    let mut tree = Tree::constant(value, rows.len());
    let mut histograms = build_hist(
        data,
        &rows,
        gradients,
        hessians,
        constant_hessian,
        quantized,
        config,
        &learner.executor,
        None,
    )?;
    if quantized.is_none() {
        histogram::fix(data, &mut histograms, gradient, hessian);
    }
    let root_recipe = direct_recipe(
        &rows,
        gradient,
        hessian,
        &mut recipe_bytes,
        &mut replay,
        retain_replay,
    )?;
    let mut root = Leaf {
        rows,
        depth: 0,
        gradient,
        hessian,
        histograms,
        split: None,
        candidates: vec![None; data.ncols()],
        active: vec![true; data.ncols()],
        branch: Vec::new(),
        bounds: (f64::NEG_INFINITY, f64::INFINITY),
        parent_output: value,
        advanced_bounds: Vec::new(),
        quant_scales: quantized.map(|q| (q.gradient_scale, q.hessian_scale)),
    };
    find_splits(learner, data, &mut root, None, config)?;
    let mut leaves = vec![Some(root)];
    let mut recipes = vec![root_recipe];
    let mut parents: Vec<Option<(usize, bool)>> = vec![None];
    let mut threshold_bins = Vec::new();
    let mut forced = std::collections::VecDeque::new();
    if let Some(json) = config
        .forced_splits
        .as_ref()
        .filter(|j| j.get("feature").is_some())
    {
        forced.push_back((0, json.clone()));
    }
    while tree.num_leaves() < config.num_leaves {
        let mut selected: Option<(usize, Split)> = None;
        for (index, leaf) in leaves.iter().enumerate() {
            if let Some(split) = leaf.as_ref().and_then(|l| l.split.as_ref())
                && selected.as_ref().is_none_or(|(_, s)| {
                    split.gain > s.gain || (split.gain == s.gain && split.feature < s.feature)
                })
            {
                selected = Some((index, split.clone()));
            }
        }
        let mut forced_node = None;
        if let Some((index, json)) = forced.pop_front() {
            let feature = json
                .get("feature")
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| {
                    Error::InvalidInput("forced split feature must be an unsigned index".into())
                })? as usize;
            let threshold = json
                .get("threshold")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| {
                    Error::InvalidInput("forced split threshold must be numeric".into())
                })?;
            if feature >= data.ncols() {
                return Err(Error::InvalidInput(
                    "forced split feature out of range".into(),
                ));
            }
            let mapper = data.mapper(feature);
            let leaf = leaves[index].as_mut().expect("forced leaf exists");
            let was_evicted = leaf.histograms.is_empty();
            restore_hist(
                leaf,
                recipes[index],
                &replay,
                data,
                gradients,
                hessians,
                constant_hessian,
                quantized,
                config,
                &learner.executor,
            )?;
            let parent = if index == 0 && tree.num_leaves() == 1 {
                0.0
            } else {
                leaf.parent_output
            };
            if let Some(split) = histogram::forced_split(
                &leaf.histograms[feature],
                feature,
                mapper.bin(threshold),
                mapper.missing_type(),
                mapper.default_bin(),
                mapper.is_categorical(),
                leaf.gradient,
                leaf.hessian,
                leaf.rows.len(),
                parent,
                config,
            ) {
                selected = Some((index, split));
                forced_node = Some(json);
            } else {
                forced.clear();
            }
            if was_evicted {
                leaf.histograms = Vec::new();
            }
        }
        let Some((left_index, split)) = selected else {
            break;
        };
        if split.gain <= 0.0 {
            break;
        }
        let mut old = leaves[left_index].take().expect("selected leaf exists");
        restore_hist(
            &mut old,
            recipes[left_index],
            &replay,
            data,
            gradients,
            hessians,
            constant_hessian,
            quantized,
            config,
            &learner.executor,
        )?;
        if !learner.coupled[split.feature] {
            learner.coupled[split.feature] = true;
            let refund = config.cegb_tradeoff
                * config
                    .cegb_penalty_feature_coupled
                    .get(split.feature)
                    .copied()
                    .unwrap_or(0.0);
            if refund != 0.0 {
                for leaf in leaves.iter_mut().flatten() {
                    if let Some(candidate) = leaf.candidates[split.feature].as_mut() {
                        // Native UpdateLeafBestSplits adds this to the raw
                        // cached gain, not to the cost-adjusted leaf split.
                        candidate.gain += refund;
                        if leaf.split.as_ref().is_some_and(|s| candidate.gain > s.gain) {
                            leaf.split = Some(candidate.clone());
                        }
                    }
                }
            }
        }
        if !learner.lazy.is_empty() {
            for &row in &old.rows {
                learner.lazy[split.feature][row] = true;
            }
        }
        let mut branch = old.branch.clone();
        branch.push(split.feature);
        let mut left_bounds = old.bounds;
        let mut right_bounds = old.bounds;
        let monotone = config
            .monotone_constraints
            .get(split.feature)
            .copied()
            .unwrap_or(0);
        let (left_limit, right_limit) =
            if config.monotone_constraints_method == MonotoneMethod::Basic {
                let mid = (split.left_output + split.right_output) / 2.0;
                (mid, mid)
            } else {
                (split.right_output, split.left_output)
            };
        if monotone > 0 {
            left_bounds.1 = left_bounds.1.min(left_limit);
            right_bounds.0 = right_bounds.0.max(right_limit);
        }
        if monotone < 0 {
            left_bounds.0 = left_bounds.0.max(left_limit);
            right_bounds.1 = right_bounds.1.min(right_limit);
        }
        let mapper = data.mapper(split.feature);
        let missing_bin = match mapper.missing_type() {
            MissingType::Zero => Some(mapper.default_bin()),
            MissingType::NaN => mapper.missing_bin(),
            MissingType::None => None,
        };
        let (left_rows, right_rows) = if mapper.is_categorical() {
            partition_rows(data, &old.rows, split.feature, |bin| {
                split.categories.contains(&bin)
            })?
        } else if let Some(missing) = missing_bin {
            partition_rows(data, &old.rows, split.feature, |bin| {
                if bin == missing {
                    split.default_left
                } else {
                    bin <= split.threshold
                }
            })?
        } else {
            partition_rows(data, &old.rows, split.feature, |bin| bin <= split.threshold)?
        };
        drop(old.rows);
        if left_rows.is_empty() || right_rows.is_empty() {
            return Err(Error::InvalidInput(
                "split produced an empty partition".into(),
            ));
        }
        let right_index = tree.num_leaves();
        if let Some(json) = forced_node {
            for (key, index) in [("left", left_index), ("right", right_index)] {
                if let Some(child) = json.get(key).filter(|j| j.get("feature").is_some()) {
                    forced.push_back((index, child.clone()));
                }
            }
        }
        let node = tree.split_features.len();
        if let Some((parent, left)) = parents[left_index] {
            if left {
                tree.left_children[parent] = node as i32;
            } else {
                tree.right_children[parent] = node as i32;
            }
        }
        parents[left_index] = Some((node, true));
        parents.push(Some((node, false)));
        tree.split_features.push(split.feature);
        if mapper.is_categorical() {
            let raw_categories: Vec<u32> = split
                .categories
                .iter()
                .map(|&bin| mapper.categories()[bin as usize] as u32)
                .collect();
            let bitset = category_bitset(&raw_categories, tree.cat_threshold.len())?;
            if tree.cat_boundaries.is_empty() {
                tree.cat_boundaries.push(0);
            }
            tree.thresholds.push((tree.cat_boundaries.len() - 1) as f64);
            tree.cat_threshold.extend(bitset);
            tree.cat_boundaries.push(tree.cat_threshold.len());
        } else {
            // Preserve native text thresholds, including infinity. Only the JSON
            // inspection view applies Common::AvoidInf.
            tree.thresholds
                .push(mapper.thresholds()[split.threshold as usize]);
        }
        tree.left_children.push(!(left_index as i32));
        tree.right_children.push(!(right_index as i32));
        let missing = match data.mapper(split.feature).missing_type() {
            MissingType::None => 0,
            MissingType::Zero => 1,
            MissingType::NaN => 2,
        };
        tree.decision_types.push(
            (missing << 2)
                | if mapper.is_categorical() {
                    1
                } else if split.default_left {
                    2
                } else {
                    0
                },
        );
        tree.internal_values.push(tree.leaf_values[left_index]);
        tree.internal_weights
            .push(split.left_hessian + split.right_hessian);
        tree.internal_counts
            .push(left_rows.len() + right_rows.len());
        tree.split_gains
            .push(f64::from((split.gain + config.min_gain_to_split) as f32));
        tree.leaf_values[left_index] = split.left_output;
        tree.leaf_weights[left_index] = split.left_hessian;
        tree.leaf_counts[left_index] = left_rows.len();
        tree.leaf_values.push(split.right_output);
        tree.leaf_weights.push(split.right_hessian);
        tree.leaf_counts.push(right_rows.len());
        threshold_bins.push(split.threshold);
        let smaller_left = left_rows.len() < right_rows.len();
        // SerialTreeLearner::FindBestSplits omits features that cannot split
        // the parent. Forced splits can request these features, so retain them.
        let active: Vec<bool> = old.candidates.iter().map(Option::is_some).collect();
        let histogram_active = config.forced_splits.is_none().then_some(active.as_slice());
        let mut small = build_hist(
            data,
            if smaller_left {
                &left_rows
            } else {
                &right_rows
            },
            gradients,
            hessians,
            constant_hessian,
            quantized,
            config,
            &learner.executor,
            histogram_active,
        )?;
        let (small_gradient, small_hessian) = if smaller_left {
            (split.left_gradient, split.left_hessian)
        } else {
            (split.right_gradient, split.right_hessian)
        };
        if quantized.is_none() {
            histogram::fix(data, &mut small, small_gradient, small_hessian);
        }
        let small_rows = if smaller_left {
            &left_rows
        } else {
            &right_rows
        };
        let small_recipe = direct_recipe(
            small_rows,
            small_gradient,
            small_hessian,
            &mut recipe_bytes,
            &mut replay,
            retain_replay,
        )?;
        let large_recipe = if quantized.is_some() {
            let (r, g, h) = if smaller_left {
                (&right_rows, split.right_gradient, split.right_hessian)
            } else {
                (&left_rows, split.left_gradient, split.left_hessian)
            };
            direct_recipe(r, g, h, &mut recipe_bytes, &mut replay, retain_replay)?
        } else if retain_replay {
            record_recipe(
                &mut replay,
                HistogramRecipe::Difference {
                    parent: recipes[left_index],
                    small: small_recipe,
                },
            )?
        } else {
            0
        };
        let large: Vec<Vec<BinStats>> = if quantized.is_some() {
            build_hist(
                data,
                if smaller_left {
                    &right_rows
                } else {
                    &left_rows
                },
                gradients,
                hessians,
                constant_hessian,
                quantized,
                config,
                &learner.executor,
                histogram_active,
            )?
        } else {
            // Reuse the parent allocation for its larger child.
            subtract_in_place(&mut old.histograms, &small)?;
            std::mem::take(&mut old.histograms)
        };
        drop(old.histograms);
        let (left_hist, right_hist) = if smaller_left {
            (small, large)
        } else {
            (large, small)
        };
        let (left_recipe, right_recipe) = if smaller_left {
            (small_recipe, large_recipe)
        } else {
            (large_recipe, small_recipe)
        };
        let mut left = Leaf {
            rows: left_rows,
            depth: old.depth + 1,
            gradient: split.left_gradient,
            hessian: split.left_hessian,
            histograms: left_hist,
            split: None,
            candidates: vec![None; data.ncols()],
            active: active.clone(),
            branch: branch.clone(),
            bounds: left_bounds,
            parent_output: split.left_output,
            advanced_bounds: Vec::new(),
            quant_scales: old.quant_scales,
        };
        let mut right = Leaf {
            rows: right_rows,
            depth: old.depth + 1,
            gradient: split.right_gradient,
            hessian: split.right_hessian,
            histograms: right_hist,
            split: None,
            candidates: vec![None; data.ncols()],
            active: active.clone(),
            branch,
            bounds: right_bounds,
            parent_output: split.right_output,
            advanced_bounds: Vec::new(),
            quant_scales: old.quant_scales,
        };
        if !config.monotone_constraints.is_empty()
            && config.monotone_constraints_method != MonotoneMethod::Basic
        {
            let advanced = config.monotone_constraints_method == MonotoneMethod::Advanced;
            if advanced {
                left.advanced_bounds = inherit_advanced(
                    &old.advanced_bounds,
                    data,
                    monotone,
                    split.right_output,
                    true,
                );
                right.advanced_bounds = inherit_advanced(
                    &old.advanced_bounds,
                    data,
                    monotone,
                    split.left_output,
                    false,
                );
            }
            let mut pending = vec![(false, false); leaves.len()];
            let mut rescore = Vec::new();
            for (index, maximum, value) in
                constraints::updates(&tree, &threshold_bins, node, &split, config)
            {
                let Some(leaf) = leaves[index].as_mut() else {
                    continue;
                };
                if leaf.split.is_none() {
                    continue;
                }
                let changed = if maximum {
                    let old = leaf.bounds.1;
                    leaf.bounds.1 = old.min(value);
                    leaf.bounds.1 < old
                } else {
                    let old = leaf.bounds.0;
                    leaf.bounds.0 = old.max(value);
                    leaf.bounds.0 > old
                };
                if advanced {
                    tighten_advanced(&mut leaf.advanced_bounds, maximum, value);
                    if maximum {
                        pending[index].1 = true;
                    } else {
                        pending[index].0 = true;
                    }
                }
                if changed || advanced {
                    rescore.push(index);
                }
            }
            for index in rescore {
                let leaf = leaves[index].as_mut().expect("constraint leaf exists");
                let was_evicted = leaf.histograms.is_empty();
                if advanced && (pending[index].0 || pending[index].1) {
                    set_advanced_bounds(
                        leaf,
                        index,
                        &tree,
                        &threshold_bins,
                        data,
                        config,
                        pending[index].0,
                    );
                    pending[index] = (false, false);
                }
                restore_hist(
                    leaf,
                    recipes[index],
                    &replay,
                    data,
                    gradients,
                    hessians,
                    constant_hessian,
                    quantized,
                    config,
                    &learner.executor,
                )?;
                leaf.active = leaf.candidates.iter().map(Option::is_some).collect();
                leaf.split = None;
                leaf.candidates.fill(None);
                leaf.parent_output = if config.path_smooth > crate::objective::EPSILON {
                    leaf_output(leaf.gradient, leaf.hessian, config)
                } else {
                    0.0
                };
                find_splits(learner, data, leaf, None, config)?;
                if was_evicted {
                    leaf.histograms = Vec::new();
                }
            }
        }
        if tree.num_leaves() < config.num_leaves {
            if smaller_left {
                find_splits(learner, data, &mut left, Some(&mut right), config)?;
            } else {
                find_splits(learner, data, &mut right, Some(&mut left), config)?;
            }
        }
        recipes[left_index] = left_recipe;
        recipes.push(right_recipe);
        leaves[left_index] = Some(left);
        leaves.push(Some(right));
        // Only retained matrices count toward the cache. Constraint rescoring
        // uses at most four checked working matrices (two new children and two
        // replay matrices). Replayed inactive leaves are released immediately.
        let mut retained = 0;
        for leaf in leaves.iter_mut().rev().flatten() {
            if !leaf.histograms.is_empty() {
                if retained < budget.slots {
                    retained += 1;
                } else {
                    leaf.histograms = Vec::new();
                }
            }
        }
    }
    let mut leaf_indices = vec![usize::MAX; data.nrows()];
    let mut leaf_rows = Vec::new();
    for (index, leaf) in leaves.into_iter().enumerate() {
        let rows = leaf.expect("all final leaf partitions exist").rows;
        for &row in &rows {
            leaf_indices[row] = index;
        }
        leaf_rows.push(rows);
    }
    if quantized.is_some() && config.quant_train_renew_leaf {
        for (leaf, rows) in leaf_rows.iter().enumerate() {
            let g = rows.iter().map(|&r| f64::from(gradients[r])).sum();
            let h = rows.iter().map(|&r| f64::from(hessians[r])).sum();
            tree.leaf_values[leaf] = leaf_output(g, h, config);
        }
    }
    Ok(FittedTree {
        tree,
        threshold_bins,
        leaf_indices,
        leaf_rows,
    })
}

// Replay histogram arithmetic, not split search. In particular, rebuilding a
// larger leaf directly would lose parent-minus-child roundoff. Recipes retain
// only root/smaller-child row lists, never RNG or mutable constraint state.
enum HistogramRecipe {
    Direct {
        rows: Vec<usize>,
        gradient: f64,
        hessian: f64,
    },
    Difference {
        parent: usize,
        small: usize,
    },
}

fn direct_recipe(
    rows: &[usize],
    gradient: f64,
    hessian: f64,
    bytes: &mut usize,
    replay: &mut Vec<HistogramRecipe>,
    retain: bool,
) -> Result<usize> {
    if !retain {
        return Ok(0);
    }
    *bytes = rows
        .len()
        .checked_mul(std::mem::size_of::<usize>())
        .and_then(|n| bytes.checked_add(n))
        .filter(|&n| n <= 128 * 1024 * 1024)
        .ok_or_else(|| {
            Error::Unsupported("histogram replay rows exceed 128 MiB per tree".into())
        })?;
    let mut copy = Vec::new();
    copy.try_reserve_exact(rows.len())
        .map_err(allocation_error)?;
    copy.extend_from_slice(rows);
    record_recipe(
        replay,
        HistogramRecipe::Direct {
            rows: copy,
            gradient,
            hessian,
        },
    )
}

// Flat indices keep both replay and destruction iterative at arbitrary depth.
fn record_recipe(replay: &mut Vec<HistogramRecipe>, recipe: HistogramRecipe) -> Result<usize> {
    if replay.len() >= 2 * 131_072 - 1 {
        return Err(Error::Unsupported(
            "histogram replay exceeds the leaf-count limit".into(),
        ));
    }
    replay.try_reserve(1).map_err(allocation_error)?;
    let index = replay.len();
    replay.push(recipe);
    Ok(index)
}

fn subtract_in_place(parent: &mut [Vec<BinStats>], child: &[Vec<BinStats>]) -> Result<()> {
    if parent.len() != child.len() {
        return Err(Error::InvalidInput("histogram shapes differ".into()));
    }
    for (p, c) in parent.iter_mut().zip(child) {
        if p.len() != c.len() {
            return Err(Error::InvalidInput("histogram shapes differ".into()));
        }
        for (p, c) in p.iter_mut().zip(c) {
            p.gradient -= c.gradient;
            p.hessian -= c.hessian;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn restore_hist(
    leaf: &mut Leaf,
    recipe: usize,
    replay: &[HistogramRecipe],
    data: &Dataset,
    gradients: &[f32],
    hessians: &[f32],
    constant: bool,
    q: Option<&Quantized>,
    config: &Config,
    executor: &crate::parallel::Executor,
) -> Result<()> {
    if !leaf.histograms.is_empty() {
        return Ok(());
    }
    let mut current = &replay[recipe];
    let mut subtractions = Vec::new();
    while let HistogramRecipe::Difference { parent, small } = current {
        subtractions.push(*small);
        current = &replay[*parent];
    }
    let direct = |recipe: &HistogramRecipe| -> Result<Vec<Vec<BinStats>>> {
        let HistogramRecipe::Direct {
            rows,
            gradient,
            hessian,
        } = recipe
        else {
            return Err(Error::InvalidInput(
                "invalid histogram replay recipe".into(),
            ));
        };
        let mut hist = build_hist(
            data, rows, gradients, hessians, constant, q, config, executor, None,
        )?;
        if q.is_none() {
            histogram::fix(data, &mut hist, *gradient, *hessian);
        }
        Ok(hist)
    };
    let mut hist = direct(current)?;
    for small in subtractions.into_iter().rev() {
        let child = direct(&replay[small])?;
        subtract_in_place(&mut hist, &child)?;
    }
    leaf.histograms = hist;
    Ok(())
}

fn tighten_advanced(bounds: &mut [Vec<[(f64, f64); 2]>], maximum: bool, value: f64) {
    for feature in bounds {
        for threshold in feature {
            for side in threshold {
                if maximum {
                    side.1 = side.1.min(value);
                } else {
                    side.0 = side.0.max(value);
                }
            }
        }
    }
}
fn inherit_advanced(
    old: &[Vec<[(f64, f64); 2]>],
    data: &Dataset,
    monotone: i8,
    value: f64,
    left: bool,
) -> Vec<Vec<[(f64, f64); 2]>> {
    let mut bounds = if old.is_empty() {
        (0..data.ncols())
            .map(|f| {
                vec![[(f64::NEG_INFINITY, f64::INFINITY); 2]; data.mapper(f).num_bins() as usize]
            })
            .collect()
    } else {
        old.to_vec()
    };
    if monotone != 0 {
        tighten_advanced(&mut bounds, (monotone > 0) == left, value);
    }
    bounds
}
#[allow(clippy::too_many_arguments)]
fn set_advanced_bounds(
    leaf: &mut Leaf,
    index: usize,
    tree: &Tree,
    bins: &[u32],
    data: &Dataset,
    config: &Config,
    minimum: bool,
) {
    for feature in 0..data.ncols() {
        if data.mapper(feature).is_categorical() {
            continue;
        }
        let bounds = constraints::feature_bounds(
            tree,
            bins,
            index,
            feature,
            data.mapper(feature).num_bins() as usize,
            config,
        );
        for (old, new) in leaf.advanced_bounds[feature].iter_mut().zip(bounds) {
            for (old, new) in old.iter_mut().zip(new) {
                if minimum {
                    old.0 = new.0;
                } else {
                    old.1 = new.1;
                }
            }
        }
    }
}

fn partition_rows(
    data: &Dataset,
    rows: &[usize],
    feature: usize,
    goes_left: impl Fn(u32) -> bool,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let mut left = Vec::new();
    let mut right = Vec::new();
    left.try_reserve(rows.len()).map_err(allocation_error)?;
    right.try_reserve(rows.len()).map_err(allocation_error)?;
    let mut partition = |row, bin| {
        if goes_left(bin) {
            left.push(row);
        } else {
            right.push(row);
        }
    };
    if rows.windows(2).all(|w| w[0] < w[1]) {
        data.visit_bins(feature, rows, partition);
    } else {
        for &row in rows {
            partition(row, data.bin(row, feature));
        }
    }
    Ok((left, right))
}

fn goes_left(
    data: &Dataset,
    row: usize,
    feature: usize,
    threshold: u32,
    default_left: bool,
) -> bool {
    bin_goes_left(
        data.mapper(feature),
        data.bin(row, feature),
        threshold,
        default_left,
    )
}

fn bin_goes_left(mapper: &crate::BinMapper, bin: u32, threshold: u32, default_left: bool) -> bool {
    match mapper.missing_type() {
        MissingType::Zero if bin == mapper.default_bin() => default_left,
        MissingType::NaN if Some(bin) == mapper.missing_bin() => default_left,
        _ => bin <= threshold,
    }
}

pub(crate) fn binned_leaf(
    tree: &Tree,
    threshold_bins: &[u32],
    data: &Dataset,
    row: usize,
) -> usize {
    if tree.num_leaves() == 1 {
        return 0;
    }
    let mut node = 0_i32;
    while node >= 0 {
        let n = node as usize;
        let left = if tree.decision_types[n] & 1 != 0 {
            let mapper = data.mapper(tree.split_features[n]);
            let code = mapper.categories()[data.bin(row, tree.split_features[n]) as usize];
            let boundary = tree.thresholds[n] as usize;
            let words = &tree.cat_threshold
                [tree.cat_boundaries[boundary]..tree.cat_boundaries[boundary + 1]];
            code >= 0
                && words
                    .get(code as usize / 32)
                    .is_some_and(|w| w & (1 << (code as u32 % 32)) != 0)
        } else {
            goes_left(
                data,
                row,
                tree.split_features[n],
                threshold_bins[n],
                tree.decision_types[n] & 2 != 0,
            )
        };
        node = if left {
            tree.left_children[n]
        } else {
            tree.right_children[n]
        };
    }
    (!node) as usize
}

fn category_bitset(categories: &[u32], existing_words: usize) -> Result<Vec<u32>> {
    const MAX_WORDS: usize = 1_000_000;
    let maximum = categories
        .iter()
        .copied()
        .max()
        .ok_or_else(|| Error::InvalidInput("empty categorical split".into()))?;
    let words = (maximum as usize / 32).checked_add(1)
        .filter(|&n| existing_words.checked_add(n).is_some_and(|sum| sum <= MAX_WORDS))
        .ok_or_else(|| Error::Unsupported("categorical split bitsets exceed the 1,000,000-word model limit; recode sparse category IDs".into()))?;
    let mut bitset = Vec::new();
    bitset.try_reserve_exact(words).map_err(allocation_error)?;
    bitset.resize(words, 0_u32);
    for &category in categories {
        bitset[category as usize / 32] |= 1 << (category % 32);
    }
    Ok(bitset)
}

#[allow(clippy::too_many_arguments)]
fn build_hist(
    data: &Dataset,
    rows: &[usize],
    g: &[f32],
    h: &[f32],
    constant: bool,
    q: Option<&Quantized>,
    config: &Config,
    executor: &crate::parallel::Executor,
    active: Option<&[bool]>,
) -> Result<Vec<Vec<BinStats>>> {
    let Some(q) = q else {
        return histogram::build_training(data, rows, g, h, constant, config, executor, active);
    };
    let mut bins = histogram::build_training(
        data,
        rows,
        &q.histogram_gradient,
        &q.histogram_hessian,
        false,
        config,
        executor,
        active,
    )?;
    histogram::fix(
        data,
        &mut bins,
        rows.iter().map(|&r| q.gradient[r]).sum(),
        rows.iter().map(|&r| q.hessian[r]).sum(),
    );
    for feature in &mut bins {
        for bin in feature {
            bin.gradient *= q.gradient_scale;
            bin.hessian *= q.hessian_scale;
        }
    }
    Ok(bins)
}

#[cfg(test)]
#[path = "../tests/learner/unit.rs"]
mod tests;

fn allocation_error(error: std::collections::TryReserveError) -> Error {
    Error::InvalidInput(format!("learner allocation failed: {error}"))
}

#[cfg(test)]
mod feature_scan_tests {
    use super::*;

    fn check_mask(learner: &TreeLearner) {
        for feature in 0..learner.tree_feature_mask.len() {
            assert_eq!(
                learner.tree_feature_mask[feature],
                learner.tree_features.contains(&feature)
            );
        }
        // Include inactive features as well as holes in eligible column indexes.
        let jobs = |masked: bool| {
            learner
                .eligible
                .iter()
                .copied()
                .filter(|&f| {
                    (if masked {
                        learner.tree_feature_mask[f]
                    } else {
                        learner.tree_features.contains(&f)
                    }) && f % 3 != 1
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(jobs(false), jobs(true));
    }

    #[test]
    fn tree_feature_mask_reuses_storage_across_fits_and_rate_resets() {
        for fraction in [1.0, 0.5] {
            for node_fraction in [1.0, 0.5] {
                let mut config = Config {
                    feature_fraction: fraction,
                    feature_fraction_bynode: node_fraction,
                    extra_trees: true,
                    min_data_in_leaf: 1,
                    min_data_in_bin: 1,
                    feature_pre_filter: false,
                    num_leaves: 4,
                    ..Config::default()
                };
                let x: Vec<f64> = (0..32 * 7)
                    .map(|i| {
                        let f = i % 7;
                        if f % 3 == 0 {
                            0.0
                        } else {
                            ((i / 7 * f) % 19) as f64
                        }
                    })
                    .collect();
                let data = Dataset::from_dense(&x, 32, 7, vec![0.0; 32], &config).unwrap();
                let mut learner = TreeLearner::new(&config, &data).unwrap();
                assert_eq!(learner.eligible, [1, 2, 4, 5]);
                let storage = learner.tree_feature_mask.as_ptr();
                check_mask(&learner);
                for _ in 0..4 {
                    learner
                        .fit(
                            &data,
                            &(0..32).map(|i| i as f32 - 16.0).collect::<Vec<_>>(),
                            &[1.0; 32],
                            &config,
                            &(0..32).collect::<Vec<_>>(),
                            true,
                        )
                        .unwrap();
                    check_mask(&learner);
                    // A reset must draw the next tree sample from the current RNG state.
                    let mut random = learner.feature_random.clone();
                    let expected = if fraction < 1.0 {
                        random
                            .sample(4, 2)
                            .unwrap()
                            .into_iter()
                            .map(|i| learner.eligible[i])
                            .collect::<Vec<_>>()
                    } else {
                        learner.eligible.clone()
                    };
                    config.learning_rate *= 0.9;
                    learner.reset_learning_rate_config(&config).unwrap();
                    assert_eq!(learner.tree_features, expected);
                    assert_eq!(
                        learner.feature_random.clone().next_float(),
                        random.next_float()
                    );
                    assert_eq!(learner.tree_feature_mask.as_ptr(), storage);
                    check_mask(&learner);
                }
            }
        }
    }

    #[test]
    fn tree_feature_mask_covers_the_last_supported_column_and_empty_samples() {
        let config = Config {
            min_data_in_leaf: 1,
            min_data_in_bin: 1,
            feature_pre_filter: false,
            ..Config::default()
        };
        for indexes in [vec![65_535, 65_535], Vec::new()] {
            let present = !indexes.is_empty();
            let data = Dataset::from_csr(
                if present { &[0, 1, 2] } else { &[0, 0, 0] },
                &indexes,
                if present { &[1.0, 2.0] } else { &[] },
                2,
                65_536,
                vec![0.0; 2],
                &config,
            )
            .unwrap();
            let learner = TreeLearner::new(&config, &data).unwrap();
            assert_eq!(learner.tree_feature_mask.len(), 65_536);
            assert_eq!(learner.tree_feature_mask[65_535], present);
            check_mask(&learner);
        }
    }
}
