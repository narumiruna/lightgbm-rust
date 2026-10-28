//! Checked LightGBM v4 tree models.
//!
//! Models are immutable through the public API. Category values are raw integer
//! codes, not Python category labels. Callers must align their own codebooks.
//! Integer negative codes, NaN, and unseen codes route right at categorical
//! splits. A Python adapter can instead map an unseen label to NaN before this
//! API; this library does not perform that mapping. Fractional codes truncate
//! toward zero, as in the reference (so -0.5 is category zero). Codes outside
//! the C++ `int` range are treated as unseen, not as saturating integer aliases.
//!
//! Traversal and text layout follow LightGBM commit
//! 5d84f3f222b41409a6983d2ac90513ad9702fa9e, `tree.h`, `tree.cpp`,
//! `gbdt_model_text.cpp`, `gbdt_prediction.cpp`, and the dense input adapter
//! in `c_api.cpp`. Values in the inclusive near-zero interval are converted to
//! zero before tree traversal, as in the reference public prediction API.
//!
//! Model input is limited to 64 MiB and 1,000,000 lines, features, trees, and
//! total leaves. Class count is limited to 65,536. Dense batch output is limited
//! to 16,777,216 elements per call. Use smaller batches for larger output.
//!
//! Copyright (c) 2016-2026 Microsoft Corporation and The LightGBM developers.
//! Licensed under the MIT License.

mod early_stop;
mod linear;
mod operations;
mod shap;
pub use early_stop::PredictionEarlyStop;
pub use linear::LinearLeaves;
pub use shap::ImportanceType;

use crate::{Error, Result};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

// Fixed resource limits apply before allocation from declared dimensions.
const MAX_TEXT_BYTES: usize = 64 * 1024 * 1024;
const MAX_ELEMENTS: usize = 1_000_000;
const MAX_CLASSES: usize = 65_536;
static TEMP_ID: AtomicU64 = AtomicU64::new(0);

// Dense C API input omits near-zero values. Native CSR keeps explicit tiny values.
fn dense_value(value: f64) -> f64 {
    const ZERO: f64 = 1e-35_f32 as f64;
    if (-ZERO..=ZERO).contains(&value) {
        0.0
    } else {
        value
    }
}

fn serialize_inspection_numbers<S: serde::Serializer>(
    thresholds: &[f64],
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    serializer.collect_seq(thresholds.iter().map(|v| v.clamp(-1e300, 1e300)))
}

fn serialize_gain_numbers<S: serde::Serializer>(
    gains: &[f64],
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    const LIMIT: f64 = 1e38_f32 as f64;
    serializer.collect_seq(gains.iter().map(|value| value.clamp(-LIMIT, LIMIT)))
}

/// A validated tree. Negative child indexes encode the bitwise complement of
/// the leaf index. Use getters or serialization for structured inspection.
#[derive(Clone, Debug, Serialize)]
pub struct Tree {
    pub(crate) split_features: Vec<usize>,
    #[serde(serialize_with = "serialize_inspection_numbers")]
    pub(crate) thresholds: Vec<f64>,
    pub(crate) left_children: Vec<i32>,
    pub(crate) right_children: Vec<i32>,
    pub(crate) decision_types: Vec<u8>,
    pub(crate) leaf_values: Vec<f64>,
    pub(crate) leaf_weights: Vec<f64>,
    pub(crate) leaf_counts: Vec<usize>,
    #[serde(serialize_with = "serialize_gain_numbers")]
    pub(crate) split_gains: Vec<f64>,
    pub(crate) internal_values: Vec<f64>,
    pub(crate) internal_weights: Vec<f64>,
    pub(crate) internal_counts: Vec<usize>,
    pub(crate) cat_boundaries: Vec<usize>,
    pub(crate) cat_threshold: Vec<u32>,
    pub(crate) shrinkage: f64,
    pub(crate) linear: Option<LinearLeaves>,
}

impl Tree {
    pub(crate) fn constant(value: f64, count: usize) -> Self {
        Self {
            split_features: vec![],
            thresholds: vec![],
            left_children: vec![],
            right_children: vec![],
            decision_types: vec![],
            leaf_values: vec![value],
            leaf_weights: vec![0.0],
            leaf_counts: vec![count],
            split_gains: vec![],
            internal_values: vec![],
            internal_weights: vec![],
            internal_counts: vec![],
            cat_boundaries: vec![],
            cat_threshold: vec![],
            shrinkage: 1.0,
            linear: None,
        }
    }

    pub(crate) fn as_constant(&mut self, value: f64, count: usize) {
        let linear = self.linear.is_some();
        *self = Self::constant(value, count);
        if linear {
            self.linear = Some(LinearLeaves {
                constants: vec![value],
                features: vec![Vec::new()],
                coefficients: vec![Vec::new()],
            });
        }
    }

    pub(crate) fn scale_outputs(&mut self, rate: f64) {
        for value in self.leaf_values.iter_mut().chain(&mut self.internal_values) {
            *value = dense_value(*value * rate);
        }
        if let Some(linear) = &mut self.linear {
            for value in linear
                .constants
                .iter_mut()
                .chain(linear.coefficients.iter_mut().flatten())
            {
                *value = dense_value(*value * rate);
            }
        }
        self.shrinkage *= rate;
    }

    pub(crate) fn add_bias(&mut self, bias: f64) {
        for value in self.leaf_values.iter_mut().chain(&mut self.internal_values) {
            *value = dense_value(*value + bias);
        }
        if let Some(linear) = &mut self.linear {
            for value in &mut linear.constants {
                *value = dense_value(*value + bias);
            }
        }
        self.shrinkage = 1.0;
    }

    pub fn linear_leaves(&self) -> Option<&LinearLeaves> {
        self.linear.as_ref()
    }

    pub fn num_leaves(&self) -> usize {
        self.leaf_values.len()
    }
    pub fn split_features(&self) -> &[usize] {
        &self.split_features
    }
    pub fn thresholds(&self) -> &[f64] {
        &self.thresholds
    }
    pub fn left_children(&self) -> &[i32] {
        &self.left_children
    }
    pub fn right_children(&self) -> &[i32] {
        &self.right_children
    }
    pub fn decision_types(&self) -> &[u8] {
        &self.decision_types
    }
    pub fn leaf_values(&self) -> &[f64] {
        &self.leaf_values
    }
    pub fn leaf_counts(&self) -> &[usize] {
        &self.leaf_counts
    }
    pub fn leaf_weights(&self) -> &[f64] {
        &self.leaf_weights
    }
    pub fn split_gains(&self) -> &[f64] {
        &self.split_gains
    }
    pub fn internal_values(&self) -> &[f64] {
        &self.internal_values
    }
    pub fn internal_counts(&self) -> &[usize] {
        &self.internal_counts
    }
    pub fn internal_weights(&self) -> &[f64] {
        &self.internal_weights
    }
    pub fn cat_boundaries(&self) -> &[usize] {
        &self.cat_boundaries
    }
    pub fn cat_threshold(&self) -> &[u32] {
        &self.cat_threshold
    }
    pub fn shrinkage(&self) -> f64 {
        self.shrinkage
    }

    fn validate(&self, features: usize) -> Result<()> {
        let leaves = self.leaf_values.len();
        bounded(leaves, MAX_ELEMENTS, "leaf count", false)?;
        let nodes = leaves - 1;
        for (name, len) in [
            ("split_feature", self.split_features.len()),
            ("threshold", self.thresholds.len()),
            ("left_child", self.left_children.len()),
            ("right_child", self.right_children.len()),
            ("decision_type", self.decision_types.len()),
            ("split_gain", self.split_gains.len()),
            ("internal_value", self.internal_values.len()),
            ("internal_weight", self.internal_weights.len()),
            ("internal_count", self.internal_counts.len()),
        ] {
            same_len(len, nodes, name)?;
        }
        same_len(self.leaf_counts.len(), leaves, "leaf_count")?;
        same_len(self.leaf_weights.len(), leaves, "leaf_weight")?;
        if let Some(linear) = &self.linear {
            linear.validate(leaves, features)?;
        }
        // Native v4 text can contain infinite numeric split endpoints.
        // Only its JSON inspection view clamps these endpoints to 1e300.
        if self.thresholds.iter().any(|v| v.is_nan()) {
            return invalid("NaN tree threshold");
        }
        for values in [
            &self.leaf_values,
            &self.leaf_weights,
            &self.internal_values,
            &self.internal_weights,
        ] {
            if values.iter().any(|v| !v.is_finite()) {
                return invalid("nonfinite tree parameter");
            }
        }
        // Native float gain storage can overflow for finite f32 labels while
        // leaves and predictions remain finite. Preserve this diagnostic value.
        if self.split_gains.iter().any(|gain| gain.is_nan()) {
            return invalid("NaN split gain");
        }
        if !self.shrinkage.is_finite() || self.shrinkage < 0.0 {
            return invalid("invalid shrinkage");
        }
        if self
            .leaf_weights
            .iter()
            .chain(&self.internal_weights)
            .any(|&v| v < 0.0)
        {
            return invalid("negative tree weight");
        }
        if self
            .leaf_counts
            .iter()
            .chain(&self.internal_counts)
            .any(|&v| v > i32::MAX as usize)
        {
            return invalid("tree count exceeds reference integer range");
        }
        let cats = self.cat_boundaries.len().saturating_sub(1);
        bounded(
            self.cat_threshold.len(),
            MAX_ELEMENTS,
            "category bitset",
            true,
        )?;
        if self.cat_boundaries.is_empty() {
            if !self.cat_threshold.is_empty() {
                return invalid("category words without boundaries");
            }
        } else if cats == 0
            || self.cat_boundaries[0] != 0
            || self.cat_boundaries.last().copied() != Some(self.cat_threshold.len())
            || self.cat_boundaries.windows(2).any(|w| w[0] >= w[1])
        {
            return invalid("invalid category boundaries");
        }
        let mut used_cats = vec![false; cats];
        for i in 0..nodes {
            if self.split_features[i] >= features {
                return invalid("split feature out of bounds");
            }
            let decision = self.decision_types[i];
            if decision > 11 || (decision >> 2) == 3 {
                return invalid("invalid decision type");
            }
            if decision & 1 != 0 {
                let threshold = self.thresholds[i];
                if threshold < 0.0 || threshold.fract() != 0.0 || threshold >= cats as f64 {
                    return invalid("invalid category index");
                }
                let cat = threshold as usize;
                if used_cats[cat] {
                    return invalid("duplicate category bitset ownership");
                }
                used_cats[cat] = true;
            }
        }
        if used_cats.iter().any(|&used| !used) {
            return invalid("unused category bitset");
        }
        // Iterative graph walk avoids stack overflow on adversarial deep trees.
        let mut seen_nodes = vec![false; nodes];
        let mut seen_leaves = vec![false; leaves];
        let mut pending = vec![if nodes == 0 { -1 } else { 0 }];
        while let Some(child) = pending.pop() {
            if child < 0 {
                let leaf = (!child) as usize;
                if leaf >= leaves || seen_leaves[leaf] {
                    return invalid("invalid or shared leaf");
                }
                seen_leaves[leaf] = true;
            } else {
                let node = child as usize;
                if node >= nodes || seen_nodes[node] {
                    return invalid("invalid, shared, or cyclic internal node");
                }
                seen_nodes[node] = true;
                pending.push(self.left_children[node]);
                pending.push(self.right_children[node]);
            }
        }
        if seen_nodes.iter().chain(&seen_leaves).any(|&seen| !seen) {
            return invalid("unreachable tree node");
        }
        Ok(())
    }

    fn from_fields(fields: &BTreeMap<&str, &str>) -> Result<Self> {
        const KEYS: &[&str] = &[
            "num_leaves",
            "num_cat",
            "split_feature",
            "threshold",
            "left_child",
            "right_child",
            "decision_type",
            "leaf_value",
            "leaf_weight",
            "leaf_count",
            "split_gain",
            "internal_value",
            "internal_weight",
            "internal_count",
            "cat_boundaries",
            "cat_threshold",
            "is_linear",
            "leaf_const",
            "num_features",
            "leaf_features",
            "leaf_coeff",
            "shrinkage",
        ];
        if fields.keys().any(|key| !KEYS.contains(key)) {
            return Err(Error::Unsupported("unknown tree field".into()));
        }
        let is_linear = match fields.get("is_linear").copied().unwrap_or("0") {
            "0" => false,
            "1" => true,
            _ => return invalid("invalid is_linear flag"),
        };
        if !is_linear
            && ["leaf_const", "num_features", "leaf_features", "leaf_coeff"]
                .iter()
                .any(|key| fields.contains_key(key))
        {
            return invalid("linear fields on a constant-leaf tree");
        }
        let leaves: usize = scalar(required(fields, "num_leaves")?, "num_leaves")?;
        bounded(leaves, MAX_ELEMENTS, "num_leaves", false)?;
        let nodes = leaves - 1;
        let cats: usize = scalar(required(fields, "num_cat")?, "num_cat")?;
        bounded(cats, nodes, "num_cat", true)?;
        let cat_boundaries = if cats == 0 {
            required_array(fields, "cat_boundaries", 0)?
        } else {
            array(
                required(fields, "cat_boundaries")?,
                cats + 1,
                "cat_boundaries",
            )?
        };
        let words = cat_boundaries.last().copied().unwrap_or(0);
        bounded(words, MAX_ELEMENTS, "category words", true)?;
        let cat_threshold = if cats == 0 {
            required_array(fields, "cat_threshold", 0)?
        } else {
            array(required(fields, "cat_threshold")?, words, "cat_threshold")?
        };
        // Dimensions must also fit in the supplied token stream. Optional arrays
        // use defaults only after the required leaf array has passed this check.
        let leaf_values = array(required(fields, "leaf_value")?, leaves, "leaf_value")?;
        Ok(Self {
            split_features: required_array(fields, "split_feature", nodes)?,
            thresholds: required_array(fields, "threshold", nodes)?,
            left_children: required_array(fields, "left_child", nodes)?,
            right_children: required_array(fields, "right_child", nodes)?,
            decision_types: optional_array(fields, "decision_type", nodes)?,
            leaf_values,
            // Upstream constant trees can export an empty leaf_weight array.
            leaf_weights: if leaves == 1
                && fields
                    .get("leaf_weight")
                    .is_some_and(|v| v.trim().is_empty())
            {
                vec![0.0]
            } else {
                optional_array(fields, "leaf_weight", leaves)?
            },
            leaf_counts: optional_array(fields, "leaf_count", leaves)?,
            split_gains: optional_array::<f64>(fields, "split_gain", nodes)?
                .into_iter()
                .map(|gain| f64::from(gain as f32))
                .collect(),
            internal_values: optional_array(fields, "internal_value", nodes)?,
            internal_weights: optional_array(fields, "internal_weight", nodes)?,
            internal_counts: optional_array(fields, "internal_count", nodes)?,
            cat_boundaries,
            cat_threshold,
            shrinkage: scalar(fields.get("shrinkage").copied().unwrap_or("1"), "shrinkage")?,
            linear: if is_linear {
                Some(LinearLeaves::from_fields(fields, leaves)?)
            } else {
                None
            },
        })
    }

    fn leaf(&self, row: &[f64]) -> usize {
        self.leaf_with(&|feature| dense_value(row[feature]))
    }

    fn leaf_with(&self, get: &impl Fn(usize) -> f64) -> usize {
        if self.leaf_values.len() == 1 {
            return 0;
        }
        let mut node = 0usize;
        loop {
            let left = self.goes_left_with(node, get);
            let child = if left {
                self.left_children[node]
            } else {
                self.right_children[node]
            };
            if child < 0 {
                return (!child) as usize;
            }
            node = child as usize;
        }
    }

    fn goes_left_with(&self, node: usize, get: &impl Fn(usize) -> f64) -> bool {
        let mut value = get(self.split_features[node]);
        const ZERO_THRESHOLD: f64 = 1e-35_f32 as f64;
        let decision = self.decision_types[node];
        if decision & 1 != 0 {
            // C++ truncates toward zero before checking the sign. Thus -0.5
            // is category zero, but -1 and NaN go right, not default-left.
            // Values outside C++ int range have no portable native cast;
            // treat them as unseen categories without a saturating alias.
            let category = value.trunc();
            if !category.is_finite() || category < 0.0 || category > i32::MAX as f64 {
                false
            } else {
                let category = category as usize;
                let index = self.thresholds[node] as usize;
                let start = self.cat_boundaries[index];
                let end = self.cat_boundaries[index + 1];
                let word = category / 32;
                word < end - start && (self.cat_threshold[start + word] >> (category % 32)) & 1 != 0
            }
        } else {
            let missing = (decision >> 2) & 3;
            if value.is_nan() && missing != 2 {
                value = 0.0;
            }
            if (missing == 1 && (-ZERO_THRESHOLD..=ZERO_THRESHOLD).contains(&value))
                || (missing == 2 && value.is_nan())
            {
                decision & 2 != 0
            } else {
                value <= self.thresholds[node]
            }
        }
    }

    fn score(&self, row: &[f64]) -> f64 {
        self.score_with(&|feature| dense_value(row[feature]))
    }

    fn score_with(&self, get: &impl Fn(usize) -> f64) -> f64 {
        let leaf = self.leaf_with(get);
        self.linear
            .as_ref()
            .and_then(|linear| linear.prediction(leaf, get))
            .unwrap_or(self.leaf_values[leaf])
    }

    fn text(&self, index: usize) -> String {
        let mut out = format!(
            "Tree={index}\nnum_leaves={}\nnum_cat={}\n",
            self.num_leaves(),
            self.cat_boundaries.len().saturating_sub(1)
        );
        append_array(&mut out, "split_feature", &self.split_features);
        append_array(&mut out, "split_gain", &self.split_gains);
        append_array(&mut out, "threshold", &self.thresholds);
        append_array(&mut out, "decision_type", &self.decision_types);
        append_array(&mut out, "left_child", &self.left_children);
        append_array(&mut out, "right_child", &self.right_children);
        append_array(&mut out, "leaf_value", &self.leaf_values);
        append_array(&mut out, "leaf_weight", &self.leaf_weights);
        append_array(&mut out, "leaf_count", &self.leaf_counts);
        append_array(&mut out, "internal_value", &self.internal_values);
        append_array(&mut out, "internal_weight", &self.internal_weights);
        append_array(&mut out, "internal_count", &self.internal_counts);
        if !self.cat_boundaries.is_empty() {
            append_array(&mut out, "cat_boundaries", &self.cat_boundaries);
            append_array(&mut out, "cat_threshold", &self.cat_threshold);
        }
        writeln!(out, "is_linear={}", usize::from(self.linear.is_some())).unwrap();
        if let Some(linear) = &self.linear {
            linear.append_fields(&mut out);
        }
        writeln!(out, "shrinkage={}\n\n", self.shrinkage).unwrap();
        out
    }
}

#[derive(Clone, Copy, Debug)]
enum Transform {
    Identity,
    SignedSquare,
    Sigmoid(f64),
    Softmax,
    Exp,
    Softplus,
}

/// A nonempty, half-open range of boosting iterations, not tree indexes.
/// Construct this through [`Model::iteration_range`]. A range is checked again
/// when used with a different model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct IterationRange {
    start: usize,
    end: usize,
}

impl IterationRange {
    pub fn start(self) -> usize {
        self.start
    }
    pub fn end(self) -> usize {
        self.end
    }
    pub fn len(self) -> usize {
        self.end - self.start
    }
    pub fn is_empty(self) -> bool {
        self.start == self.end
    }
}

/// An immutable, validated tree ensemble. Serialization exposes a structured
/// inspection view; it does not provide an unchecked deserialization path.
#[derive(Clone, Debug, Serialize)]
pub struct Model {
    pub(crate) trees: Vec<Tree>,
    pub(crate) num_class: usize,
    pub(crate) num_tree_per_iteration: usize,
    pub(crate) feature_names: Vec<String>,
    pub(crate) feature_infos: Vec<String>,
    pub(crate) objective: String,
    pub(crate) average_output: bool,
    // Native live objectives use the original double; ToString writes six digits.
    #[serde(skip)]
    live_sigmoid: Option<f64>,
    metadata: BTreeMap<String, String>,
    footer: String,
}

impl Model {
    pub(crate) fn from_parts(
        trees: Vec<Tree>,
        num_class: usize,
        num_tree_per_iteration: usize,
        feature_names: Vec<String>,
        feature_infos: Vec<String>,
        objective: String,
        average_output: bool,
    ) -> Result<Self> {
        let model = Self {
            trees,
            num_class,
            num_tree_per_iteration,
            feature_names,
            feature_infos,
            objective,
            average_output,
            live_sigmoid: None,
            metadata: BTreeMap::new(),
            footer: String::new(),
        };
        model.validate()?;
        Ok(model)
    }

    pub(crate) fn with_live_sigmoid(mut self, config: &crate::Config) -> Result<Self> {
        self.live_sigmoid = match config.objective {
            crate::Objective::Binary | crate::Objective::MulticlassOva => Some(config.sigmoid),
            _ => None,
        };
        self.validate()?;
        Ok(self)
    }

    pub fn num_features(&self) -> usize {
        self.feature_names.len()
    }
    pub fn num_classes(&self) -> usize {
        self.num_class
    }
    pub fn num_iterations(&self) -> usize {
        self.trees.len() / self.num_tree_per_iteration
    }
    pub fn trees(&self) -> &[Tree] {
        &self.trees
    }
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }
    pub fn feature_infos(&self) -> &[String] {
        &self.feature_infos
    }
    pub fn objective(&self) -> &str {
        &self.objective
    }
    pub fn average_output(&self) -> bool {
        self.average_output
    }
    pub fn num_tree_per_iteration(&self) -> usize {
        self.num_tree_per_iteration
    }

    fn validate(&self) -> Result<()> {
        bounded(self.num_class, MAX_CLASSES, "num_class", false)?;
        if self.num_tree_per_iteration != self.num_class {
            return invalid("class/tree iteration layout mismatch");
        }
        bounded(self.num_features(), MAX_ELEMENTS, "feature count", false)?;
        same_len(
            self.feature_infos.len(),
            self.num_features(),
            "feature_infos",
        )?;
        let mut names = BTreeSet::new();
        for name in &self.feature_names {
            if name.is_empty()
                || name.chars().any(|c| c.is_whitespace() || c.is_control())
                || !names.insert(name)
            {
                return invalid("empty, duplicate, or whitespace-containing feature name");
            }
        }
        if self
            .feature_infos
            .iter()
            .any(|s| s.is_empty() || s.chars().any(|c| c.is_whitespace() || c.is_control()))
        {
            return invalid("invalid feature information");
        }
        bounded(self.trees.len(), MAX_ELEMENTS, "tree count", true)?;
        if !self.trees.len().is_multiple_of(self.num_tree_per_iteration) {
            return invalid("incomplete tree iteration");
        }
        objective_transform(&self.objective, self.num_class)?;
        if let Some(sigmoid) = self.live_sigmoid
            && (!sigmoid.is_finite()
                || sigmoid <= 0.0
                || !matches!(
                    self.objective.split_whitespace().next(),
                    Some("binary" | "multiclassova")
                ))
        {
            return invalid("invalid live objective sigmoid");
        }
        let mut total = 0usize;
        for tree in &self.trees {
            total = total
                .checked_add(tree.num_leaves())
                .ok_or_else(|| Error::InvalidModel("node count overflow".into()))?;
            bounded(total, MAX_ELEMENTS, "ensemble leaf count", false)?;
            tree.validate(self.num_features())?;
        }
        Ok(())
    }

    /// Select a half-open iteration range. Each selected iteration contains
    /// one tree per class. Only an empty ensemble permits the empty 0..0 range.
    pub fn iteration_range(&self, range: std::ops::Range<usize>) -> Result<IterationRange> {
        if self.trees.is_empty() && range.start == 0 && range.end == 0 {
            return Ok(IterationRange { start: 0, end: 0 });
        }
        if range.start >= range.end || range.end > self.num_iterations() {
            return Err(Error::InvalidInput(
                "iteration range must be nonempty and within the model".into(),
            ));
        }
        Ok(IterationRange {
            start: range.start,
            end: range.end,
        })
    }

    pub fn all_iterations(&self) -> IterationRange {
        IterationRange {
            start: 0,
            end: self.num_iterations(),
        }
    }

    fn check_row(&self, row: &[f64]) -> Result<()> {
        if row.len() != self.num_features() {
            return Err(Error::InvalidInput(format!(
                "expected {} features, got {}",
                self.num_features(),
                row.len()
            )));
        }
        Ok(())
    }

    fn tree_range(&self, iterations: IterationRange) -> Result<std::ops::Range<usize>> {
        self.iteration_range(iterations.start..iterations.end)?;
        Ok(iterations.start * self.num_tree_per_iteration
            ..iterations.end * self.num_tree_per_iteration)
    }

    /// Raw scores are sums, including for RF models. Tree values already include
    /// shrinkage and the initial score. Do not apply shrinkage a second time.
    pub fn predict_raw_row(&self, row: &[f64]) -> Result<Vec<f64>> {
        self.predict_raw_row_with_iterations(row, self.all_iterations())
    }

    pub fn predict_raw_row_with_iterations(
        &self,
        row: &[f64],
        iterations: IterationRange,
    ) -> Result<Vec<f64>> {
        self.check_row(row)?;
        let range = self.tree_range(iterations)?;
        let mut output = vec![0.0; self.num_tree_per_iteration];
        for (offset, tree) in self.trees[range].iter().enumerate() {
            output[offset % self.num_tree_per_iteration] += tree.score(row);
        }
        Ok(output)
    }

    pub fn predict_row(&self, row: &[f64]) -> Result<Vec<f64>> {
        self.predict_row_with_iterations(row, self.all_iterations())
    }

    pub fn predict_row_with_iterations(
        &self,
        row: &[f64],
        iterations: IterationRange,
    ) -> Result<Vec<f64>> {
        let output = self.predict_raw_row_with_iterations(row, iterations)?;
        self.transform_output(output, iterations)
    }

    fn transform_output(
        &self,
        mut output: Vec<f64>,
        iterations: IterationRange,
    ) -> Result<Vec<f64>> {
        if self.average_output {
            if iterations.is_empty() {
                return Err(Error::InvalidInput(
                    "RF transformed prediction requires a nonempty iteration range".into(),
                ));
            }
            for value in &mut output {
                *value /= iterations.len() as f64;
            }
        }
        match objective_transform(&self.objective, self.num_class)? {
            Transform::Identity => {}
            Transform::SignedSquare => {
                let sign = if output[0] > 0.0 {
                    1.0
                } else if output[0] < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                output[0] = sign * output[0] * output[0];
            }
            Transform::Sigmoid(sigmoid) => {
                let sigmoid = self.live_sigmoid.unwrap_or(sigmoid);
                for value in &mut output {
                    *value = 1.0 / (1.0 + (-sigmoid * *value).exp());
                }
            }
            Transform::Softmax => {
                let maximum = output.iter().copied().fold(output[0], f64::max);
                let mut sum = 0.0;
                for value in &mut output {
                    *value = (*value - maximum).exp();
                    sum += *value;
                }
                for value in &mut output {
                    *value /= sum;
                }
            }
            Transform::Exp => output[0] = output[0].exp(),
            // Match the pinned reference even when exp overflows. This is not
            // the numerically stabilized softplus used by some other libraries.
            Transform::Softplus => output[0] = output[0].exp().ln_1p(),
        }
        Ok(output)
    }

    pub fn predict_leaf_row(&self, row: &[f64]) -> Result<Vec<usize>> {
        self.predict_leaf_row_with_iterations(row, self.all_iterations())
    }

    pub fn predict_leaf_row_with_iterations(
        &self,
        row: &[f64],
        iterations: IterationRange,
    ) -> Result<Vec<usize>> {
        self.check_row(row)?;
        let range = self.tree_range(iterations)?;
        Ok(self.trees[range]
            .iter()
            .map(|tree| tree.leaf(row))
            .collect())
    }

    /// Row-major dense input with explicit dimensions. Empty batches are valid;
    /// each row must still have exactly `num_features()` columns. Output is
    /// limited to 16,777,216 elements per call; divide larger input into batches.
    pub fn predict_dense(
        &self,
        data: &[f64],
        rows: usize,
        columns: usize,
    ) -> Result<Vec<Vec<f64>>> {
        self.check_dense(data, rows, columns, self.num_class)?;
        data.chunks_exact(columns)
            .map(|row| self.predict_row(row))
            .collect()
    }

    pub fn predict_raw_dense(
        &self,
        data: &[f64],
        rows: usize,
        columns: usize,
    ) -> Result<Vec<Vec<f64>>> {
        self.check_dense(data, rows, columns, self.num_class)?;
        data.chunks_exact(columns)
            .map(|row| self.predict_raw_row(row))
            .collect()
    }

    pub fn predict_leaf_dense(
        &self,
        data: &[f64],
        rows: usize,
        columns: usize,
    ) -> Result<Vec<Vec<usize>>> {
        self.check_dense(data, rows, columns, self.trees.len())?;
        data.chunks_exact(columns)
            .map(|row| self.predict_leaf_row(row))
            .collect()
    }

    fn check_dense(&self, data: &[f64], rows: usize, columns: usize, outputs: usize) -> Result<()> {
        const MAX_BATCH_OUTPUTS: usize = 16 * 1024 * 1024;
        if columns != self.num_features() || rows.checked_mul(columns) != Some(data.len()) {
            return Err(Error::InvalidInput(
                "dense input shape mismatch or overflow".into(),
            ));
        }
        if rows
            .checked_mul(outputs)
            .is_none_or(|n| n > MAX_BATCH_OUTPUTS)
        {
            return Err(Error::InvalidInput(
                "batch exceeds 16777216 output element limit".into(),
            ));
        }
        Ok(())
    }

    /// Predict one sparse row with the same near-zero normalization as
    /// `predict_row`. This is a dense-equivalent row API, not a native CSR
    /// compatibility adapter. Absent features are zero. Indexes must be strictly
    /// increasing; duplicate indexes are rejected rather than summed.
    pub fn predict_sparse_row(&self, indices: &[usize], values: &[f64]) -> Result<Vec<f64>> {
        self.check_sparse_row(indices, values)?;
        let get = |feature| dense_value(indices.binary_search(&feature).map_or(0.0, |i| values[i]));
        let mut output = vec![0.0; self.num_tree_per_iteration];
        for (index, tree) in self.trees.iter().enumerate() {
            output[index % self.num_tree_per_iteration] += tree.score_with(&get);
        }
        self.transform_output(output, self.all_iterations())
    }

    /// Leaf IDs with dense-equivalent near-zero normalization. This sparse
    /// adapter does not allocate a dense row; use the CSR API to keep tiny values.
    pub fn predict_leaf_sparse_row(&self, indices: &[usize], values: &[f64]) -> Result<Vec<usize>> {
        self.check_sparse_row(indices, values)?;
        let get = |feature| dense_value(indices.binary_search(&feature).map_or(0.0, |i| values[i]));
        Ok(self.trees.iter().map(|tree| tree.leaf_with(&get)).collect())
    }

    /// Approximate raw prediction. Checks occur after full class rounds and
    /// start at the selected range, not the beginning of the model.
    pub fn predict_raw_row_early_stop(
        &self,
        row: &[f64],
        iterations: IterationRange,
        stopping: PredictionEarlyStop,
    ) -> Result<Vec<f64>> {
        self.check_row(row)?;
        early_stop::predict(
            self,
            &|feature| dense_value(row[feature]),
            iterations,
            stopping,
        )
    }

    pub fn predict_row_early_stop(
        &self,
        row: &[f64],
        iterations: IterationRange,
        stopping: PredictionEarlyStop,
    ) -> Result<Vec<f64>> {
        let scores = self.predict_raw_row_early_stop(row, iterations, stopping)?;
        // Native RF divides by the full selected range, not rounds visited.
        self.transform_output(scores, iterations)
    }

    pub fn predict_raw_csr_row_early_stop(
        &self,
        indices: &[usize],
        values: &[f64],
        iterations: IterationRange,
        stopping: PredictionEarlyStop,
    ) -> Result<Vec<f64>> {
        self.check_sparse_row(indices, values)?;
        let get = |feature| indices.binary_search(&feature).map_or(0.0, |i| values[i]);
        early_stop::predict(self, &get, iterations, stopping)
    }

    pub fn predict_csr_row_early_stop(
        &self,
        indices: &[usize],
        values: &[f64],
        iterations: IterationRange,
        stopping: PredictionEarlyStop,
    ) -> Result<Vec<f64>> {
        let scores = self.predict_raw_csr_row_early_stop(indices, values, iterations, stopping)?;
        self.transform_output(scores, iterations)
    }

    /// Count-weighted TreeSHAP values in class-major rows. Each row contains
    /// all original features and a final bias. Its sum equals the raw score.
    /// Linear trees and missing cover counts are not supported.
    pub fn predict_contrib_row(&self, row: &[f64]) -> Result<Vec<Vec<f64>>> {
        self.predict_contrib_row_with_iterations(row, self.all_iterations())
    }

    pub fn predict_contrib_row_with_iterations(
        &self,
        row: &[f64],
        iterations: IterationRange,
    ) -> Result<Vec<Vec<f64>>> {
        self.check_row(row)?;
        shap::contributions(self, &|feature| dense_value(row[feature]), iterations)
    }

    /// Native CSR input semantics, with no dense row allocation.
    pub fn predict_contrib_csr_row(
        &self,
        indices: &[usize],
        values: &[f64],
    ) -> Result<Vec<Vec<f64>>> {
        self.predict_contrib_csr_row_with_iterations(indices, values, self.all_iterations())
    }

    pub fn predict_contrib_csr_row_with_iterations(
        &self,
        indices: &[usize],
        values: &[f64],
        iterations: IterationRange,
    ) -> Result<Vec<Vec<f64>>> {
        self.check_sparse_row(indices, values)?;
        let get = |feature| indices.binary_search(&feature).map_or(0.0, |i| values[i]);
        shap::contributions(self, &get, iterations)
    }

    /// Importance in original feature order. Only positive split gains count.
    pub fn feature_importance(&self, kind: ImportanceType) -> Result<Vec<f64>> {
        self.feature_importance_with_iterations(kind, self.all_iterations())
    }

    pub fn feature_importance_with_iterations(
        &self,
        kind: ImportanceType,
        iterations: IterationRange,
    ) -> Result<Vec<f64>> {
        shap::importance(self, kind, iterations)
    }

    fn check_sparse_row(&self, indices: &[usize], values: &[f64]) -> Result<()> {
        if indices.len() != values.len()
            || indices.iter().any(|&i| i >= self.num_features())
            || indices.windows(2).any(|w| w[0] >= w[1])
        {
            return Err(Error::InvalidInput(
                "invalid sparse indexes or value shape".into(),
            ));
        }
        Ok(())
    }

    /// Native CSR semantics: explicit near-zero entries are preserved. Indexes
    /// must be strictly increasing. This method does not allocate a dense row.
    pub fn predict_csr_row(&self, indices: &[usize], values: &[f64]) -> Result<Vec<f64>> {
        self.predict_csr_row_with_iterations(indices, values, self.all_iterations())
    }

    pub fn predict_csr_row_with_iterations(
        &self,
        indices: &[usize],
        values: &[f64],
        iterations: IterationRange,
    ) -> Result<Vec<f64>> {
        let raw = self.predict_raw_csr_row_with_iterations(indices, values, iterations)?;
        self.transform_output(raw, iterations)
    }

    pub fn predict_raw_csr_row(&self, indices: &[usize], values: &[f64]) -> Result<Vec<f64>> {
        self.predict_raw_csr_row_with_iterations(indices, values, self.all_iterations())
    }

    pub fn predict_raw_csr_row_with_iterations(
        &self,
        indices: &[usize],
        values: &[f64],
        iterations: IterationRange,
    ) -> Result<Vec<f64>> {
        self.check_sparse_row(indices, values)?;
        let range = self.tree_range(iterations)?;
        let get = |feature| indices.binary_search(&feature).map_or(0.0, |i| values[i]);
        let mut output = vec![0.0; self.num_tree_per_iteration];
        for (offset, tree) in self.trees[range].iter().enumerate() {
            output[offset % self.num_tree_per_iteration] += tree.score_with(&get);
        }
        Ok(output)
    }

    pub fn predict_leaf_csr_row(&self, indices: &[usize], values: &[f64]) -> Result<Vec<usize>> {
        self.predict_leaf_csr_row_with_iterations(indices, values, self.all_iterations())
    }

    pub fn predict_leaf_csr_row_with_iterations(
        &self,
        indices: &[usize],
        values: &[f64],
        iterations: IterationRange,
    ) -> Result<Vec<usize>> {
        self.check_sparse_row(indices, values)?;
        let range = self.tree_range(iterations)?;
        let get = |feature| indices.binary_search(&feature).map_or(0.0, |i| values[i]);
        Ok(self.trees[range]
            .iter()
            .map(|tree| tree.leaf_with(&get))
            .collect())
    }

    /// Read the pinned v4 text format. Unknown versions and capabilities fail
    /// explicitly. Declared dimensions are bounded before allocation.
    pub fn from_text(text: &str) -> Result<Self> {
        if text.len() > MAX_TEXT_BYTES {
            return invalid("model text exceeds 64 MiB limit");
        }
        if text.contains('\0') {
            return invalid("NUL in model text");
        }
        let mut lines = Vec::new();
        let mut offset = 0;
        for raw in text.split_inclusive('\n') {
            if lines.len() >= MAX_ELEMENTS {
                return invalid("model line count exceeds 1000000 limit");
            }
            lines.push((offset, raw.trim_end_matches(['\r', '\n'])));
            offset += raw.len();
        }
        if lines.first().map(|l| l.1) != Some("tree") {
            return invalid("expected tree model header");
        }
        let end = lines
            .iter()
            .position(|l| l.1 == "end of trees")
            .ok_or_else(|| Error::InvalidModel("missing end of trees".into()))?;
        let first_tree = lines
            .iter()
            .position(|l| l.1.starts_with("Tree="))
            .unwrap_or(end);
        if end < first_tree {
            return invalid("invalid tree section");
        }
        let mut header = BTreeMap::new();
        let mut average_output = false;
        for &(_, line) in &lines[1..first_tree] {
            if line.is_empty() {
                continue;
            }
            if line == "average_output" {
                if average_output {
                    return invalid("duplicate average_output");
                }
                average_output = true;
            } else {
                insert_field(&mut header, line)?;
            }
        }
        const HEADER_KEYS: &[&str] = &[
            "version",
            "num_class",
            "num_tree_per_iteration",
            "label_index",
            "max_feature_idx",
            "objective",
            "feature_names",
            "feature_infos",
            "tree_sizes",
            "monotone_constraints",
        ];
        if let Some(key) = header.keys().find(|key| !HEADER_KEYS.contains(key)) {
            return Err(Error::Unsupported(format!("model header field {key}")));
        }
        if required(&header, "version")? != "v4" {
            return Err(Error::Unsupported(
                "model version (only v4 is supported)".into(),
            ));
        }
        let num_class = scalar(required(&header, "num_class")?, "num_class")?;
        bounded(num_class, MAX_CLASSES, "num_class", false)?;
        let per_iteration = scalar(
            header
                .get("num_tree_per_iteration")
                .copied()
                .unwrap_or(required(&header, "num_class")?),
            "num_tree_per_iteration",
        )?;
        bounded(per_iteration, MAX_CLASSES, "num_tree_per_iteration", false)?;
        let max_feature: usize = scalar(required(&header, "max_feature_idx")?, "max_feature_idx")?;
        bounded(max_feature, MAX_ELEMENTS - 1, "max_feature_idx", true)?;
        let num_features = max_feature + 1;
        let _: i32 = scalar(required(&header, "label_index")?, "label_index")?;
        let feature_names = string_array(
            required(&header, "feature_names")?,
            num_features,
            "feature_names",
        )?;
        let feature_infos = string_array(
            required(&header, "feature_infos")?,
            num_features,
            "feature_infos",
        )?;
        let objective = header
            .get("objective")
            .copied()
            .unwrap_or("custom")
            .to_owned();
        objective_transform(&objective, num_class)?;
        if let Some(constraints) = header.get("monotone_constraints") {
            let constraints: Vec<i8> = array(constraints, num_features, "monotone_constraints")?;
            if constraints.iter().any(|v| !(-1..=1).contains(v)) {
                return invalid("invalid monotone constraint");
            }
        }
        let tree_lines: Vec<usize> = (first_tree..end)
            .filter(|&i| lines[i].1.starts_with("Tree="))
            .collect();
        bounded(tree_lines.len(), MAX_ELEMENTS, "tree count", true)?;
        let sizes: Option<Vec<usize>> = header
            .get("tree_sizes")
            .map(|s| array(s, tree_lines.len(), "tree_sizes"))
            .transpose()?;
        let mut trees = Vec::new();
        let mut total = 0usize;
        for (index, &start) in tree_lines.iter().enumerate() {
            let stop = tree_lines.get(index + 1).copied().unwrap_or(end);
            if lines[start].1 != format!("Tree={index}") {
                return invalid("nonsequential tree index");
            }
            if let Some(sizes) = &sizes
                && sizes[index] != lines[stop].0 - lines[start].0
            {
                return invalid("tree_sizes byte length mismatch");
            }
            let mut fields = BTreeMap::new();
            for &(_, line) in &lines[start + 1..stop] {
                if !line.is_empty() {
                    insert_field(&mut fields, line)?;
                }
            }
            let tree = Tree::from_fields(&fields)?;
            total = total
                .checked_add(tree.num_leaves())
                .ok_or_else(|| Error::InvalidModel("leaf count overflow".into()))?;
            bounded(total, MAX_ELEMENTS, "ensemble leaf count", false)?;
            tree.validate(num_features)?;
            trees.push(tree);
        }
        let reserved = [
            "version",
            "num_class",
            "num_tree_per_iteration",
            "max_feature_idx",
            "objective",
            "feature_names",
            "feature_infos",
            "tree_sizes",
        ];
        let metadata = header
            .into_iter()
            .filter(|(key, _)| !reserved.contains(key))
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect();
        let footer_start = lines.get(end + 1).map_or(text.len(), |l| l.0);
        let mut model = Self::from_parts(
            trees,
            num_class,
            per_iteration,
            feature_names,
            feature_infos,
            objective,
            average_output,
        )?;
        model.metadata = metadata;
        model.footer = text[footer_start..].to_owned();
        Ok(model)
    }

    /// Export complete tree blocks. `tree_sizes` counts UTF-8 bytes, including
    /// tree headings and separator newlines. Imported footer metadata is kept.
    pub fn to_text(&self) -> Result<String> {
        self.validate()?;
        let blocks: Vec<String> = self
            .trees
            .iter()
            .enumerate()
            .map(|(i, t)| t.text(i))
            .collect();
        let mut out = format!(
            "tree\nversion=v4\nnum_class={}\nnum_tree_per_iteration={}\nlabel_index={}\nmax_feature_idx={}\nobjective={}\n",
            self.num_class,
            self.num_tree_per_iteration,
            self.metadata.get("label_index").map_or("0", String::as_str),
            self.num_features() - 1,
            self.objective
        );
        if self.average_output {
            out.push_str("average_output\n");
        }
        append_array(&mut out, "feature_names", &self.feature_names);
        append_array(&mut out, "feature_infos", &self.feature_infos);
        for (key, value) in &self.metadata {
            if key != "label_index" {
                writeln!(out, "{key}={value}").unwrap();
            }
        }
        append_array(
            &mut out,
            "tree_sizes",
            &blocks.iter().map(String::len).collect::<Vec<_>>(),
        );
        out.push('\n');
        for block in blocks {
            out.push_str(&block);
        }
        out.push_str("end of trees\n");
        out.push_str(&self.footer);
        if out.len() > MAX_TEXT_BYTES {
            return invalid("export exceeds 64 MiB model limit");
        }
        Ok(out)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let mut text = String::new();
        file.take((MAX_TEXT_BYTES + 1) as u64)
            .read_to_string(&mut text)?;
        Self::from_text(&text)
    }

    /// Atomically publish a new file without replacing any existing path.
    /// Existing files, source models, directories, and symlinks are never
    /// overwritten. Use a new destination for each saved artifact.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let text = self.to_text()?;
        let path = path.as_ref();
        if path.file_name().is_none() {
            return Err(Error::InvalidInput("destination needs a file name".into()));
        }
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let mut temporary = None;
        for _ in 0..128 {
            let id = TEMP_ID.fetch_add(1, Ordering::Relaxed);
            let temp = parent.join(format!(".lightgbm-rust-{}-{id}.tmp", std::process::id()));
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp)
            {
                Ok(file) => {
                    temporary = Some((temp, file));
                    break;
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e.into()),
            }
        }
        let (temp, mut file) = temporary.ok_or_else(|| {
            Error::Io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "cannot create model temporary file",
            ))
        })?;
        let result = (|| -> Result<()> {
            file.write_all(text.as_bytes())?;
            file.sync_all()?;
            // Same-directory hard link is an atomic, no-replace publication.
            // Unlike rename, it cannot clobber a concurrently created target.
            std::fs::hard_link(&temp, path)?;
            Ok(())
        })();
        drop(file);
        let cleanup = std::fs::remove_file(&temp);
        result?;
        cleanup?;
        Ok(())
    }
}

fn objective_transform(objective: &str, classes: usize) -> Result<Transform> {
    let mut tokens = objective.split_whitespace();
    let name = tokens
        .next()
        .ok_or_else(|| Error::InvalidModel("empty objective".into()))?;
    let mut sqrt = false;
    let mut sigmoid = None;
    let mut objective_classes = None;
    for token in tokens {
        if token == "sqrt" && !sqrt {
            sqrt = true;
        } else if let Some(value) = token.strip_prefix("sigmoid:") {
            if sigmoid.is_some() {
                return invalid("duplicate sigmoid");
            }
            let value: f64 = scalar(value, "sigmoid")?;
            if !value.is_finite() || value <= 0.0 {
                return invalid("invalid objective sigmoid");
            }
            sigmoid = Some(value);
        } else if let Some(value) = token.strip_prefix("num_class:") {
            if objective_classes.is_some() {
                return invalid("duplicate objective num_class");
            }
            let value: usize = scalar(value, "objective num_class")?;
            if value != classes {
                return invalid("objective class count mismatch");
            }
            objective_classes = Some(value);
        } else {
            return Err(Error::Unsupported(format!("objective parameter {token}")));
        }
    }
    let multiclass = matches!(name, "multiclass" | "multiclassova");
    if multiclass {
        if classes < 2 || objective_classes.is_none() {
            return invalid("invalid multiclass layout");
        }
    } else if classes != 1 && name != "custom" {
        return invalid("scalar objective with multiple classes");
    }
    if sqrt
        && !matches!(
            name,
            "regression" | "regression_l1" | "quantile" | "huber" | "fair" | "mape"
        )
    {
        return invalid("sqrt on incompatible objective");
    }
    if sigmoid.is_some() && !matches!(name, "binary" | "multiclassova") {
        return invalid("sigmoid on incompatible objective");
    }
    if objective_classes.is_some() && !multiclass {
        return invalid("num_class on scalar objective");
    }
    match name {
        "regression" | "regression_l1" | "quantile" | "fair" | "mape" => Ok(if sqrt {
            Transform::SignedSquare
        } else {
            Transform::Identity
        }),
        // The pinned Huber reader explicitly disables sqrt.
        "huber" => Ok(Transform::Identity),
        "custom" | "lambdarank" | "rank_xendcg" => Ok(Transform::Identity),
        "binary" | "multiclassova" => {
            Ok(Transform::Sigmoid(sigmoid.ok_or_else(|| {
                Error::InvalidModel("missing objective sigmoid".into())
            })?))
        }
        "multiclass" => Ok(Transform::Softmax),
        "poisson" | "gamma" | "tweedie" => Ok(Transform::Exp),
        "cross_entropy" => Ok(Transform::Sigmoid(1.0)),
        "cross_entropy_lambda" => Ok(Transform::Softplus),
        _ => Err(Error::Unsupported(format!("objective {name}"))),
    }
}

fn invalid<T>(message: &str) -> Result<T> {
    Err(Error::InvalidModel(message.into()))
}
fn bounded(value: usize, maximum: usize, name: &str, zero: bool) -> Result<()> {
    if value > maximum || (!zero && value == 0) {
        return invalid(&format!("{name} exceeds supported bounds"));
    }
    Ok(())
}
fn same_len(actual: usize, expected: usize, name: &str) -> Result<()> {
    if actual != expected {
        return invalid(&format!(
            "{name}: expected {expected} elements, got {actual}"
        ));
    }
    Ok(())
}
fn scalar<T: std::str::FromStr>(text: &str, name: &str) -> Result<T> {
    text.parse()
        .map_err(|_| Error::InvalidModel(format!("invalid {name}")))
}
fn required<'a>(fields: &BTreeMap<&str, &'a str>, key: &str) -> Result<&'a str> {
    fields
        .get(key)
        .copied()
        .ok_or_else(|| Error::InvalidModel(format!("missing {key}")))
}
fn insert_field<'a>(fields: &mut BTreeMap<&'a str, &'a str>, line: &'a str) -> Result<()> {
    let (key, value) = line
        .split_once('=')
        .ok_or_else(|| Error::InvalidModel("expected key=value".into()))?;
    if key.is_empty() || fields.insert(key, value).is_some() {
        return invalid("empty or duplicate field");
    }
    Ok(())
}
fn array<T: std::str::FromStr>(text: &str, count: usize, name: &str) -> Result<Vec<T>> {
    bounded(count, MAX_ELEMENTS + 1, name, true)?;
    let mut result = Vec::new();
    for item in text.split_whitespace() {
        if result.len() == count {
            return invalid(&format!("too many {name} values"));
        }
        result.push(scalar(item, name)?);
    }
    same_len(result.len(), count, name)?;
    Ok(result)
}
fn required_array<T: std::str::FromStr>(
    fields: &BTreeMap<&str, &str>,
    key: &str,
    count: usize,
) -> Result<Vec<T>> {
    match fields.get(key) {
        Some(text) => array(text, count, key),
        None if count == 0 => Ok(vec![]),
        None => invalid(&format!("missing {key}")),
    }
}
fn optional_array<T: std::str::FromStr + Default + Clone>(
    fields: &BTreeMap<&str, &str>,
    key: &str,
    count: usize,
) -> Result<Vec<T>> {
    match fields.get(key) {
        Some(text) => array(text, count, key),
        None => Ok(vec![T::default(); count]),
    }
}
fn string_array(text: &str, count: usize, name: &str) -> Result<Vec<String>> {
    let values: Vec<String> = array(text, count, name)?;
    Ok(values)
}
fn append_array<T: std::fmt::Display>(out: &mut String, key: &str, values: &[T]) {
    write!(out, "{key}=").unwrap();
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            out.push(' ');
        }
        write!(out, "{value}").unwrap();
    }
    out.push('\n');
}

#[cfg(test)]
mod live_sigmoid_tests {
    use super::*;

    #[test]
    fn live_sigmoid_requires_positive_finite_value_and_compatible_objective() {
        let mut model = Model::from_parts(
            vec![Tree::constant(1.0, 1)],
            1,
            1,
            vec!["x".into()],
            vec!["none".into()],
            "binary sigmoid:1".into(),
            false,
        )
        .unwrap();
        assert!(model.live_sigmoid.is_none());
        for value in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            model.live_sigmoid = Some(value);
            assert!(model.validate().is_err());
        }
        model.live_sigmoid = Some(1.23456789);
        model.validate().unwrap();
        for objective in ["regression", "cross_entropy", "custom"] {
            model.objective = objective.into();
            assert!(model.validate().is_err());
        }
    }
}
