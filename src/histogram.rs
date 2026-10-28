//! Order-preserving gradient histograms and numeric/categorical split search.
//!
//! Split scan order, Hessian-based count estimation, and epsilon placement
//! follow LightGBM commit 5d84f3f222b41409a6983d2ac90513ad9702fa9e,
//! src/treelearner/feature_histogram.hpp and feature_histogram.cpp (MIT).
//! Copyright (c) 2016-2026 Microsoft Corporation and The LightGBM developers.

use crate::data::{DenseBins, MissingType};
use crate::objective::EPSILON;
use crate::{Config, Dataset, Error, Result};

// Counted histograms retain actual row counts. The learner stores only GH
// pairs: split search estimates counts from Hessian proportions, not this field.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Stats<C> {
    pub gradient: f64,
    pub hessian: f64,
    pub count: C,
}

pub(crate) type BinStats = Stats<usize>;
pub(crate) type TrainingStats = Stats<()>;

trait Count: Copy + Default + Send + Sync {
    fn from_rows(rows: usize) -> Self;
    fn add(&mut self);
    fn remove(&mut self);
}

impl Count for usize {
    fn from_rows(rows: usize) -> Self {
        rows
    }
    fn add(&mut self) {
        *self += 1;
    }
    fn remove(&mut self) {
        *self -= 1;
    }
}

impl Count for () {
    fn from_rows(_: usize) -> Self {}
    fn add(&mut self) {}
    fn remove(&mut self) {}
}

#[derive(Clone, Debug)]
pub(crate) struct Split {
    pub feature: usize,
    pub threshold: u32,
    /// Bin codes sent left. Empty for a numerical split.
    pub categories: Vec<u32>,
    pub default_left: bool,
    pub gain: f64,
    pub left_output: f64,
    pub right_output: f64,
    pub left_gradient: f64,
    pub right_gradient: f64,
    pub left_hessian: f64,
    pub right_hessian: f64,
}

/// Bytes for one full histogram, including the column-vector storage.
/// Check the declared working set before any histogram allocation.
pub(crate) fn histogram_bytes(dataset: &Dataset) -> Result<usize> {
    let mut bytes = dataset
        .ncols()
        .checked_mul(std::mem::size_of::<Vec<BinStats>>())
        .ok_or_else(|| Error::InvalidInput("histogram shape overflow".into()))?;
    for feature in 0..dataset.ncols() {
        bytes = (dataset.mapper(feature).num_bins() as usize)
            .checked_mul(std::mem::size_of::<BinStats>())
            .and_then(|n| bytes.checked_add(n))
            .ok_or_else(|| Error::InvalidInput("histogram shape overflow".into()))?;
    }
    if bytes > 128 * 1024 * 1024 {
        return Err(Error::Unsupported(
            "one histogram exceeds the 128 MiB working-set ceiling".into(),
        ));
    }
    Ok(bytes)
}

#[cfg(test)]
pub(crate) fn build(
    dataset: &Dataset,
    rows: &[usize],
    gradients: &[f32],
    hessians: &[f32],
    constant_hessian: bool,
) -> Result<Vec<Vec<BinStats>>> {
    build_with_executor(
        dataset,
        rows,
        gradients,
        hessians,
        constant_hessian,
        &Config::default(),
        &crate::parallel::Executor::new(1)?,
        None,
    )
}

fn empty_bins<C: Count>(dataset: &Dataset, feature: usize) -> Result<Vec<Stats<C>>> {
    let count = dataset.mapper(feature).num_bins() as usize;
    let mut bins = Vec::new();
    bins.try_reserve_exact(count).map_err(allocation_error)?;
    bins.resize(count, Stats::default());
    Ok(bins)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_training(
    dataset: &Dataset,
    rows: &[usize],
    gradients: &[f32],
    hessians: &[f32],
    constant_hessian: bool,
    config: &Config,
    executor: &crate::parallel::Executor,
    active: Option<&[bool]>,
) -> Result<Vec<Vec<TrainingStats>>> {
    if constant_hessian {
        build_kernel::<true, ()>(dataset, rows, gradients, hessians, config, executor, active)
    } else {
        build_kernel::<false, ()>(dataset, rows, gradients, hessians, config, executor, active)
    }
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_with_executor(
    dataset: &Dataset,
    rows: &[usize],
    gradients: &[f32],
    hessians: &[f32],
    constant_hessian: bool,
    config: &Config,
    executor: &crate::parallel::Executor,
    active: Option<&[bool]>,
) -> Result<Vec<Vec<BinStats>>> {
    if constant_hessian {
        build_kernel::<true, usize>(dataset, rows, gradients, hessians, config, executor, active)
    } else {
        build_kernel::<false, usize>(dataset, rows, gradients, hessians, config, executor, active)
    }
}

fn build_kernel<const CONSTANT: bool, C: Count>(
    dataset: &Dataset,
    rows: &[usize],
    gradients: &[f32],
    hessians: &[f32],
    config: &Config,
    executor: &crate::parallel::Executor,
    active: Option<&[bool]>,
) -> Result<Vec<Vec<Stats<C>>>> {
    histogram_bytes(dataset)?;
    let sparse_totals = if (0..dataset.ncols()).any(|f| dataset.sparse_bins(f).is_some()) {
        let mut members = Vec::new();
        members
            .try_reserve_exact(dataset.nrows())
            .map_err(allocation_error)?;
        members.resize(dataset.nrows(), false);
        let (mut gradient, mut hessian) = (0.0, 0.0);
        for &row in rows {
            members[row] = true;
            gradient += f64::from(gradients[row]);
            if !CONSTANT {
                hessian += f64::from(hessians[row]);
            }
        }
        Some((members, gradient, hessian))
    } else {
        None
    };
    // Select column-wise for sparse or wide data. Narrow dense data uses the
    // row-major kernel. Explicit force flags override this deterministic choice.
    let row_wise = config.force_row_wise
        || (!config.force_col_wise && dataset.ncols() <= 4 && sparse_totals.is_none());
    let sorted = rows.windows(2).all(|w| w[0] < w[1]);
    let full_root = sorted
        && rows.len() == dataset.nrows()
        && rows.first() == Some(&0)
        && rows.last() == Some(&(dataset.nrows() - 1));
    let column = |feature: usize| -> Result<Vec<Stats<C>>> {
        let mut bins = empty_bins(dataset, feature)?;
        if active.is_some_and(|mask| !mask[feature]) {
            return Ok(bins);
        }
        if let Some(entries) = dataset.sparse_bins(feature) {
            let (members, gradient, hessian) = sparse_totals.as_ref().expect("sparse totals exist");
            let default = dataset.mapper(feature).bin(0.0) as usize;
            bins[default] = Stats {
                gradient: *gradient,
                hessian: if CONSTANT {
                    rows.len() as f64
                } else {
                    *hessian
                },
                count: C::from_rows(rows.len()),
            };
            for (row, value) in entries {
                if !members[row] {
                    continue;
                }
                let gradient = f64::from(gradients[row]);
                let hessian = if CONSTANT {
                    1.0
                } else {
                    f64::from(hessians[row])
                };
                bins[default].gradient -= gradient;
                bins[default].hessian -= hessian;
                bins[default].count.remove();
                bins[value as usize].gradient += gradient;
                bins[value as usize].hessian += hessian;
                bins[value as usize].count.add();
            }
        } else if let Some(values) = dataset.dense_bins(feature) {
            match values {
                DenseBins::U8(values) => dense_column::<CONSTANT, _, _>(
                    &mut bins, values, rows, gradients, hessians, full_root,
                ),
                DenseBins::U16(values) => dense_column::<CONSTANT, _, _>(
                    &mut bins, values, rows, gradients, hessians, full_root,
                ),
                DenseBins::U32(values) => dense_column::<CONSTANT, _, _>(
                    &mut bins, values, rows, gradients, hessians, full_root,
                ),
            }
        } else if !CONSTANT && bins.len() <= 256 {
            // Every stored code is below the mapper's bin count. Narrow only
            // after this check, so the fixed array index is always in bounds.
            let mut small = [Stats::<C>::default(); 256];
            let mut accumulate = |row: usize, value: u32| {
                debug_assert!((value as usize) < bins.len());
                let bin = &mut small[value as u8 as usize];
                bin.gradient += f64::from(gradients[row]);
                bin.hessian += if CONSTANT {
                    1.0
                } else {
                    f64::from(hessians[row])
                };
                bin.count.add();
            };
            if sorted {
                dataset.visit_bins(feature, rows, accumulate);
            } else {
                for &row in rows {
                    accumulate(row, dataset.bin(row, feature));
                }
            }
            let count = bins.len();
            bins.copy_from_slice(&small[..count]);
        } else {
            let mut accumulate = |row: usize, value: u32| {
                let bin = &mut bins[value as usize];
                bin.gradient += f64::from(gradients[row]);
                bin.hessian += if CONSTANT {
                    1.0
                } else {
                    f64::from(hessians[row])
                };
                bin.count.add();
            };
            if sorted {
                dataset.visit_bins(feature, rows, accumulate);
            } else {
                for &row in rows {
                    accumulate(row, dataset.bin(row, feature));
                }
            }
        }
        Ok(bins)
    };
    let mut histograms: Vec<Vec<Stats<C>>> = if row_wise {
        // Each job owns a disjoint block of features. Dense bins are updated
        // row-first, without thread-local partial sums or floating reductions.
        // Sparse columns retain their reference default-bin subtraction order.
        let width = 4;
        executor
            .map(
                dataset.ncols().div_ceil(width),
                if rows.len() >= 1024 { 2 } else { usize::MAX },
                |block| {
                    let start = block * width;
                    let end = (start + width).min(dataset.ncols());
                    let mut columns = (start..end)
                        .map(|feature| {
                            if dataset.sparse_bins(feature).is_some() {
                                column(feature)
                            } else {
                                empty_bins(dataset, feature)
                            }
                        })
                        .collect::<Result<Vec<_>>>()?;
                    for &row in rows {
                        for (offset, bins) in columns.iter_mut().enumerate() {
                            let feature = start + offset;
                            if active.is_some_and(|mask| !mask[feature])
                                || dataset.sparse_bins(feature).is_some()
                            {
                                continue;
                            }
                            let bin = &mut bins[dataset.bin(row, feature) as usize];
                            bin.gradient += f64::from(gradients[row]);
                            bin.hessian += if CONSTANT {
                                1.0
                            } else {
                                f64::from(hessians[row])
                            };
                            bin.count.add();
                        }
                    }
                    Ok(columns)
                },
            )
            .into_iter()
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect()
    } else {
        executor
            .map(
                dataset.ncols(),
                if rows.len() >= 1024 { 4 } else { usize::MAX },
                column,
            )
            .into_iter()
            .collect::<Result<_>>()?
    };
    if CONSTANT {
        // The Hessian slot holds an exact integer count until this point.
        // Dataset row counts fit int32, so all increments/subtractions are exact.
        // Native counts * hessians[0], before FixHistogram (including MAPE).
        for bins in &mut histograms {
            for bin in bins {
                bin.hessian *= f64::from(hessians[0]);
            }
        }
    }
    Ok(histograms)
}

fn dense_column<const CONSTANT: bool, C: Count, B: Copy + Into<u32>>(
    bins: &mut [Stats<C>],
    values: &[B],
    rows: &[usize],
    gradients: &[f32],
    hessians: &[f32],
    full_root: bool,
) {
    if !CONSTANT && bins.len() <= 256 {
        let mut small = [Stats::<C>::default(); 256];
        visit_dense::<CONSTANT, _>(
            values,
            rows,
            gradients,
            hessians,
            full_root,
            |value, g, h| {
                debug_assert!((value as usize) < bins.len());
                let bin = &mut small[value as u8 as usize];
                bin.gradient += g;
                bin.hessian += h;
                bin.count.add();
            },
        );
        bins.copy_from_slice(&small[..bins.len()]);
    } else {
        visit_dense::<CONSTANT, _>(
            values,
            rows,
            gradients,
            hessians,
            full_root,
            |value, g, h| {
                let bin = &mut bins[value as usize];
                bin.gradient += g;
                bin.hessian += h;
                bin.count.add();
            },
        );
    }
}

fn visit_dense<const CONSTANT: bool, B: Copy + Into<u32>>(
    values: &[B],
    rows: &[usize],
    gradients: &[f32],
    hessians: &[f32],
    full_root: bool,
    mut visit: impl FnMut(u32, f64, f64),
) {
    if full_root {
        if CONSTANT {
            for (&value, &gradient) in values.iter().zip(gradients) {
                visit(value.into(), f64::from(gradient), 1.0);
            }
        } else {
            for ((&value, &gradient), &hessian) in values.iter().zip(gradients).zip(hessians) {
                visit(value.into(), f64::from(gradient), f64::from(hessian));
            }
        }
    } else {
        for &row in rows {
            visit(
                values[row].into(),
                f64::from(gradients[row]),
                if CONSTANT {
                    1.0
                } else {
                    f64::from(hessians[row])
                },
            );
        }
    }
}

// Dataset::FixHistogram reconstructs a nonzero most-frequent bin from
// leaf totals. Correct the smaller child before parent-minus-child subtraction;
// even its tiny negative roundoff can determine the sibling's split direction.
pub(crate) fn fix<C>(
    data: &Dataset,
    histograms: &mut [Vec<Stats<C>>],
    gradient: f64,
    hessian: f64,
) {
    for (feature, bins) in histograms.iter_mut().enumerate() {
        let most = data.mapper(feature).most_freq_bin() as usize;
        if most == 0 {
            continue;
        }
        let (mut g, mut h) = (gradient, hessian);
        for (index, bin) in bins.iter().enumerate() {
            if index != most {
                g -= bin.gradient;
                h -= bin.hessian;
            }
        }
        bins[most].gradient = g;
        bins[most].hessian = h;
    }
}

#[cfg(test)]
pub(crate) fn subtract(parent: &[BinStats], child: &[BinStats]) -> Result<Vec<BinStats>> {
    if parent.len() != child.len() {
        return Err(Error::InvalidInput("histogram shapes differ".into()));
    }
    parent
        .iter()
        .zip(child)
        .map(|(p, c)| {
            Ok(BinStats {
                gradient: p.gradient - c.gradient,
                hessian: p.hessian - c.hessian,
                count: p.count.checked_sub(c.count).ok_or_else(|| {
                    Error::InvalidInput("child histogram count exceeds parent".into())
                })?,
            })
        })
        .collect()
}

pub(crate) fn leaf_output(gradient: f64, hessian: f64, config: &Config) -> f64 {
    let regularized = gradient.signum() * (gradient.abs() - config.lambda_l1).max(0.0);
    let mut value = -regularized / (hessian + config.lambda_l2);
    if config.max_delta_step > 0.0 {
        value = value.clamp(-config.max_delta_step, config.max_delta_step);
    }
    value
}

fn leaf_gain(gradient: f64, hessian: f64, config: &Config) -> f64 {
    let regularized = gradient.signum() * (gradient.abs() - config.lambda_l1).max(0.0);
    if config.max_delta_step > 0.0 {
        let output = leaf_output(gradient, hessian, config);
        -(2.0 * regularized * output + (hessian + config.lambda_l2) * output * output)
    } else {
        regularized * regularized / (hessian + config.lambda_l2)
    }
}

#[derive(Default)]
pub(crate) struct SplitControl {
    pub threshold: Option<u32>,
    pub parent_output: f64,
    pub monotone: i8,
    pub bounds: Option<(f64, f64)>,
    pub threshold_bounds: Vec<[(f64, f64); 2]>,
}
fn valid_bounds(control: &SplitControl, threshold: u32) -> bool {
    control
        .threshold_bounds
        .get(threshold as usize)
        .is_none_or(|bounds| bounds.iter().all(|&(min, max)| min <= max))
        && (control.threshold_bounds.get(threshold as usize).is_some()
            || control.bounds.is_none_or(|(min, max)| min <= max))
}

fn controlled_output(
    g: f64,
    h: f64,
    count: usize,
    config: &Config,
    parent: f64,
    control: &SplitControl,
    side: Option<(u32, bool)>,
) -> f64 {
    let mut output = leaf_output(g, h, config);
    if config.path_smooth > 0.0 {
        let ratio = count as f64 / config.path_smooth;
        output = output * (ratio / (ratio + 1.0)) + parent / (ratio + 1.0);
    }
    if let Some((threshold, left)) = side {
        let bounds = control
            .threshold_bounds
            .get(threshold as usize)
            .map(|b| b[usize::from(!left)])
            .or(control.bounds);
        if let Some((min, max)) = bounds {
            output = output.max(min).min(max);
        }
    }
    output
}
fn controlled_gain(
    g: f64,
    h: f64,
    count: usize,
    config: &Config,
    parent: f64,
    control: &SplitControl,
    side: Option<(u32, bool)>,
) -> f64 {
    if config.path_smooth == 0.0
        && (side.is_none() || (control.bounds.is_none() && control.threshold_bounds.is_empty()))
    {
        return leaf_gain(g, h, config);
    }
    let output = controlled_output(g, h, count, config, parent, control, side);
    let regularized = g.signum() * (g.abs() - config.lambda_l1).max(0.0);
    -(2.0 * regularized * output + (h + config.lambda_l2) * output * output)
}
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn best_split(
    histogram: &[Stats<impl Copy>],
    feature: usize,
    missing: MissingType,
    default_bin: u32,
    offset: bool,
    total_gradient: f64,
    total_hessian: f64,
    rows: usize,
    config: &Config,
) -> Option<Split> {
    best_split_controlled(
        histogram,
        feature,
        missing,
        default_bin,
        offset,
        total_gradient,
        total_hessian,
        rows,
        config,
        &SplitControl::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn best_split_controlled(
    histogram: &[Stats<impl Copy>],
    feature: usize,
    missing: MissingType,
    default_bin: u32,
    offset: bool,
    total_gradient: f64,
    total_hessian: f64,
    rows: usize,
    config: &Config,
    control: &SplitControl,
) -> Option<Split> {
    let simple = config.lambda_l1 == 0.0
        && config.max_delta_step == 0.0
        && config.path_smooth == 0.0
        && control.monotone == 0
        && control.bounds.is_none()
        && control.threshold_bounds.is_empty();
    if simple {
        numeric_split::<true>(
            histogram,
            feature,
            missing,
            default_bin,
            offset,
            total_gradient,
            total_hessian,
            rows,
            config,
            control,
        )
    } else {
        numeric_split::<false>(
            histogram,
            feature,
            missing,
            default_bin,
            offset,
            total_gradient,
            total_hessian,
            rows,
            config,
            control,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn numeric_split<const SIMPLE: bool>(
    histogram: &[Stats<impl Copy>],
    feature: usize,
    missing: MissingType,
    default_bin: u32,
    offset: bool,
    total_gradient: f64,
    total_hessian: f64,
    rows: usize,
    config: &Config,
    control: &SplitControl,
) -> Option<Split> {
    if histogram.len() < 2 || total_hessian <= 0.0 {
        return None;
    }
    let hessian = total_hessian + 2.0 * EPSILON;
    let minimum_gain = controlled_gain(
        total_gradient,
        hessian,
        rows,
        config,
        control.parent_output,
        control,
        None,
    ) + config.min_gain_to_split;
    let factor = rows as f64 / hessian;
    let mut best: Option<Split> = None;
    let separate_missing = histogram.len() > 2 && missing != MissingType::None;
    let directions: &[bool] = if separate_missing {
        &[true, false]
    } else {
        &[true]
    };
    for &reverse in directions {
        let mut accumulated_gradient = 0.0;
        let mut accumulated_hessian = EPSILON;
        let mut estimated_count = 0_i64;
        let reconstruct_first = !reverse && offset && missing == MissingType::NaN;
        if reconstruct_first {
            accumulated_gradient = total_gradient;
            accumulated_hessian = hessian - EPSILON;
            estimated_count = rows as i64;
            for bin in &histogram[1..] {
                accumulated_gradient -= bin.gradient;
                accumulated_hessian -= bin.hessian;
                estimated_count -= (bin.hessian * factor).round() as i64;
            }
        }
        let (start, end) = if reverse {
            (
                1,
                histogram.len() - usize::from(separate_missing && missing == MissingType::NaN),
            )
        } else {
            (
                usize::from(offset && !reconstruct_first),
                histogram.len() - 1,
            )
        };
        let mut local_best: Option<(f64, f64, f64, u32, f64, usize)> = None;
        for index in start..end {
            let bin = if reverse {
                end - 1 - (index - start)
            } else {
                index
            };
            if separate_missing && missing == MissingType::Zero && bin == default_bin as usize {
                continue;
            }
            if !(reconstruct_first && bin == 0) {
                accumulated_gradient += histogram[bin].gradient;
                accumulated_hessian += histogram[bin].hessian;
                estimated_count += (histogram[bin].hessian * factor).round() as i64;
            }
            let other_count = rows as i64 - estimated_count;
            if estimated_count < config.min_data_in_leaf as i64
                || accumulated_hessian < config.min_sum_hessian_in_leaf
            {
                continue;
            }
            if other_count < config.min_data_in_leaf as i64 {
                break;
            }
            let other_hessian = hessian - accumulated_hessian;
            if other_hessian < config.min_sum_hessian_in_leaf {
                break;
            }
            let other_gradient = total_gradient - accumulated_gradient;
            let (left_g, left_h, right_g, right_h, threshold) = if reverse {
                (
                    other_gradient,
                    other_hessian,
                    accumulated_gradient,
                    accumulated_hessian,
                    (bin - 1) as u32,
                )
            } else {
                (
                    accumulated_gradient,
                    accumulated_hessian,
                    other_gradient,
                    other_hessian,
                    bin as u32,
                )
            };
            if control.threshold.is_some_and(|t| t != threshold) {
                continue;
            }
            if !SIMPLE && !valid_bounds(control, threshold) {
                continue;
            }
            let left_count = if reverse {
                other_count
            } else {
                estimated_count
            } as usize;
            let right_count = rows - left_count;
            let left_output = if SIMPLE {
                0.0
            } else {
                controlled_output(
                    left_g,
                    left_h,
                    left_count,
                    config,
                    control.parent_output,
                    control,
                    Some((threshold, true)),
                )
            };
            let right_output = if SIMPLE {
                0.0
            } else {
                controlled_output(
                    right_g,
                    right_h,
                    right_count,
                    config,
                    control.parent_output,
                    control,
                    Some((threshold, false)),
                )
            };
            // With neutral controls, outputs are needed only for a new best split.
            // Keep the same per-side gain formula and left-before-right addition.
            let gain = if SIMPLE {
                left_g * left_g / (left_h + config.lambda_l2)
                    + right_g * right_g / (right_h + config.lambda_l2)
            } else if (control.monotone > 0 && left_output > right_output)
                || (control.monotone < 0 && left_output < right_output)
            {
                0.0
            } else {
                controlled_gain(
                    left_g,
                    left_h,
                    left_count,
                    config,
                    control.parent_output,
                    control,
                    Some((threshold, true)),
                ) + controlled_gain(
                    right_g,
                    right_h,
                    right_count,
                    config,
                    control.parent_output,
                    control,
                    Some((threshold, false)),
                )
            };
            if !gain.is_finite()
                || gain <= minimum_gain
                || local_best.as_ref().is_some_and(|(s, ..)| gain <= *s)
            {
                continue;
            }
            local_best = Some((gain, left_g, left_h, threshold, left_output, right_count));
        }
        if let Some((gain, left_g, left_h, threshold, left_output, right_count)) = local_best
            && best.as_ref().is_none_or(|s| gain > s.gain + minimum_gain)
        {
            best = Some(Split {
                feature,
                threshold,
                categories: Vec::new(),
                default_left: if !separate_missing && missing == MissingType::NaN {
                    false
                } else {
                    reverse
                },
                gain: gain - minimum_gain,
                left_output: if SIMPLE {
                    leaf_output(left_g, left_h, config)
                } else {
                    left_output
                },
                // Store the reference's subtraction result, not the
                // algebraically equal reverse-scan accumulator.
                right_output: controlled_output(
                    total_gradient - left_g,
                    hessian - left_h,
                    right_count,
                    config,
                    control.parent_output,
                    control,
                    Some((threshold, false)),
                ),
                left_gradient: left_g,
                right_gradient: total_gradient - left_g,
                left_hessian: left_h - EPSILON,
                right_hessian: hessian - left_h - EPSILON,
            });
        }
    }
    best
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn best_quantized_split(
    histogram: &[Stats<impl Copy>],
    feature: usize,
    missing: MissingType,
    default_bin: u32,
    offset: bool,
    total_gradient: f64,
    total_hessian: f64,
    rows: usize,
    config: &Config,
    control: &SplitControl,
    gradient_scale: f64,
    hessian_scale: f64,
) -> Option<Split> {
    if histogram.len() < 2 || total_hessian <= 0.0 {
        return None;
    }
    let hessian = total_hessian;
    let minimum_gain = controlled_gain(
        total_gradient,
        hessian,
        rows,
        config,
        control.parent_output,
        control,
        None,
    ) + config.min_gain_to_split;
    let factor = rows as f64 / hessian;
    let mut best: Option<Split> = None;
    let separate_missing = histogram.len() > 2 && missing != MissingType::None;
    let directions: &[bool] = if separate_missing {
        &[true, false]
    } else {
        &[true]
    };
    for &reverse in directions {
        let mut accumulated_gradient = 0.0;
        let mut accumulated_hessian = 0.0;
        let reconstruct_first = !reverse && offset && missing == MissingType::NaN;
        if reconstruct_first {
            accumulated_gradient = total_gradient;
            accumulated_hessian = hessian;
            for bin in &histogram[1..] {
                accumulated_gradient -= bin.gradient;
                accumulated_hessian -= bin.hessian;
            }
        }
        let scan: Box<dyn Iterator<Item = usize>> = if reverse {
            let end =
                histogram.len() - usize::from(separate_missing && missing == MissingType::NaN);
            Box::new((1..end).rev())
        } else {
            let start = usize::from(offset && !reconstruct_first);
            Box::new(start..histogram.len() - 1)
        };
        let mut local_best: Option<(f64, Split)> = None;
        for bin in scan {
            if separate_missing && missing == MissingType::Zero && bin == default_bin as usize {
                continue;
            }
            if !(reconstruct_first && bin == 0) {
                accumulated_gradient += histogram[bin].gradient;
                accumulated_hessian += histogram[bin].hessian;
                accumulated_gradient =
                    (accumulated_gradient / gradient_scale).round() * gradient_scale;
                accumulated_hessian = (accumulated_hessian / hessian_scale).round() * hessian_scale;
            }
            let estimated_count = (accumulated_hessian * factor).round() as i64;
            let other_count = rows as i64 - estimated_count;
            if estimated_count < config.min_data_in_leaf as i64
                || accumulated_hessian < config.min_sum_hessian_in_leaf
            {
                continue;
            }
            if other_count < config.min_data_in_leaf as i64 {
                break;
            }
            let other_hessian = ((hessian / hessian_scale).round()
                - (accumulated_hessian / hessian_scale).round())
                * hessian_scale;
            if other_hessian < config.min_sum_hessian_in_leaf {
                break;
            }
            let other_gradient = ((total_gradient / gradient_scale).round()
                - (accumulated_gradient / gradient_scale).round())
                * gradient_scale;
            let (left_g, left_h, right_g, right_h, threshold) = if reverse {
                (
                    other_gradient,
                    other_hessian,
                    accumulated_gradient,
                    accumulated_hessian,
                    (bin - 1) as u32,
                )
            } else {
                (
                    accumulated_gradient,
                    accumulated_hessian,
                    other_gradient,
                    other_hessian,
                    bin as u32,
                )
            };
            let left_h = left_h + EPSILON;
            let right_h = right_h + EPSILON;
            if control.threshold.is_some_and(|t| t != threshold) {
                continue;
            }
            if !valid_bounds(control, threshold) {
                continue;
            }
            let left_count = if reverse {
                other_count
            } else {
                estimated_count
            } as usize;
            let right_count = rows - left_count;
            let left_output = controlled_output(
                left_g,
                left_h,
                left_count,
                config,
                control.parent_output,
                control,
                Some((threshold, true)),
            );
            let right_output = controlled_output(
                right_g,
                right_h,
                right_count,
                config,
                control.parent_output,
                control,
                Some((threshold, false)),
            );
            let gain = if (control.monotone > 0 && left_output > right_output)
                || (control.monotone < 0 && left_output < right_output)
            {
                0.0
            } else {
                controlled_gain(
                    left_g,
                    left_h,
                    left_count,
                    config,
                    control.parent_output,
                    control,
                    Some((threshold, true)),
                ) + controlled_gain(
                    right_g,
                    right_h,
                    right_count,
                    config,
                    control.parent_output,
                    control,
                    Some((threshold, false)),
                )
            };
            if !gain.is_finite()
                || gain <= minimum_gain
                || local_best.as_ref().is_some_and(|(s, _)| gain <= *s)
            {
                continue;
            }
            local_best = Some((
                gain,
                Split {
                    feature,
                    threshold,
                    categories: Vec::new(),
                    default_left: if !separate_missing && missing == MissingType::NaN {
                        false
                    } else {
                        reverse
                    },
                    gain,
                    left_output,
                    // Store the reference's subtraction result, not the
                    // algebraically equal reverse-scan accumulator. The last
                    // bits affect missing-direction ties in the next leaf.
                    right_output: controlled_output(
                        right_g,
                        right_h,
                        right_count,
                        config,
                        control.parent_output,
                        control,
                        Some((threshold, false)),
                    ),
                    left_gradient: left_g,
                    right_gradient: right_g,
                    left_hessian: left_h - EPSILON,
                    right_hessian: right_h - EPSILON,
                },
            ));
        }
        if let Some((gain, mut split)) = local_best
            && best.as_ref().is_none_or(|s| gain > s.gain + minimum_gain)
        {
            split.gain = gain - minimum_gain;
            best = Some(split);
        }
    }
    best
}

/// Pinned FindBestThresholdCategoricalInner with neutral learner controls.
/// Bin zero is the dummy/missing bin, including when it is the most frequent.
/// The histogram stores full bin codes, so the native offset needs no adjustment.
#[cfg(test)]
pub(crate) fn best_categorical_split(
    histogram: &[Stats<impl Copy>],
    feature: usize,
    total_gradient: f64,
    total_hessian: f64,
    rows: usize,
    config: &Config,
) -> Option<Split> {
    best_categorical_split_controlled(
        histogram,
        feature,
        total_gradient,
        total_hessian,
        rows,
        config,
        &SplitControl::default(),
    )
}

pub(crate) fn best_categorical_split_controlled(
    histogram: &[Stats<impl Copy>],
    feature: usize,
    total_gradient: f64,
    total_hessian: f64,
    rows: usize,
    config: &Config,
    control: &SplitControl,
) -> Option<Split> {
    if histogram.len() < 2 || total_hessian <= 0.0 {
        return None;
    }
    let hessian = total_hessian + 2.0 * EPSILON;
    let minimum_gain = if config.path_smooth > 0.0 {
        let g = total_gradient.signum() * (total_gradient.abs() - config.lambda_l1).max(0.0);
        -(2.0 * g * control.parent_output
            + (hessian + config.lambda_l2) * control.parent_output * control.parent_output)
            + config.min_gain_to_split
    } else {
        leaf_gain(total_gradient, hessian, config) + config.min_gain_to_split
    };
    let factor = rows as f64 / hessian;
    let onehot = histogram.len() <= config.max_cat_to_onehot;
    let mut child_config = config.clone();
    if !onehot {
        child_config.lambda_l2 += config.cat_l2;
    }
    let mut best: Option<Split> = None;
    let mut best_gain = f64::NEG_INFINITY;
    let mut consider =
        |gradient: f64, left_hessian: f64, count: usize, gain: f64, categories: Vec<u32>| {
            if !gain.is_finite() || gain <= minimum_gain || gain <= best_gain {
                return;
            }
            best_gain = gain;
            best = Some(Split {
                feature,
                threshold: 0,
                categories,
                default_left: false,
                gain: gain - minimum_gain,
                left_output: controlled_output(
                    gradient,
                    left_hessian,
                    count,
                    &child_config,
                    control.parent_output,
                    control,
                    Some((0, true)),
                ),
                right_output: controlled_output(
                    total_gradient - gradient,
                    hessian - left_hessian,
                    rows - count,
                    &child_config,
                    control.parent_output,
                    control,
                    Some((0, false)),
                ),
                left_gradient: gradient,
                right_gradient: total_gradient - gradient,
                left_hessian: left_hessian - EPSILON,
                right_hessian: hessian - left_hessian - EPSILON,
            });
        };
    if onehot {
        for (bin, stats) in histogram.iter().enumerate().skip(1) {
            if control.threshold.is_some_and(|t| t != bin as u32) {
                continue;
            }
            let count = (stats.hessian * factor).round() as i64;
            let other = rows as i64 - count;
            if count < config.min_data_in_leaf as i64
                || stats.hessian < config.min_sum_hessian_in_leaf
                || other < config.min_data_in_leaf as i64
            {
                continue;
            }
            let other_hessian = hessian - stats.hessian - EPSILON;
            if other_hessian < config.min_sum_hessian_in_leaf {
                continue;
            }
            // Preserve native right-before-left gain evaluation and epsilon order.
            let gain = controlled_gain(
                total_gradient - stats.gradient,
                other_hessian,
                other as usize,
                config,
                control.parent_output,
                control,
                Some((0, false)),
            ) + controlled_gain(
                stats.gradient,
                stats.hessian + EPSILON,
                count as usize,
                config,
                control.parent_output,
                control,
                Some((0, true)),
            );
            consider(
                stats.gradient,
                stats.hessian + EPSILON,
                count as usize,
                gain,
                vec![bin as u32],
            );
        }
    } else {
        let mut sorted: Vec<usize> = (1..histogram.len())
            .filter(|&bin| (histogram[bin].hessian * factor).round() >= config.cat_smooth)
            .collect();
        sorted.sort_by(|&a, &b| {
            let ratio =
                |bin: usize| histogram[bin].gradient / (histogram[bin].hessian + config.cat_smooth);
            // Native stable_sort compares with `<`: equal keys retain bin order.
            ratio(a)
                .partial_cmp(&ratio(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let maximum = config.max_cat_threshold.min(sorted.len().div_ceil(2));
        for reverse in [false, true] {
            let mut gradient = 0.0;
            let mut left_hessian = EPSILON;
            let mut count = 0_i64;
            let mut group_count = 0_i64;
            let mut categories = Vec::new();
            for i in 0..maximum {
                let bin = sorted[if reverse { sorted.len() - 1 - i } else { i }];
                categories.push(bin as u32);
                let stats = histogram[bin];
                let estimated = (stats.hessian * factor).round() as i64;
                gradient += stats.gradient;
                left_hessian += stats.hessian;
                count += estimated;
                group_count += estimated;
                if count < config.min_data_in_leaf as i64
                    || left_hessian < config.min_sum_hessian_in_leaf
                {
                    continue;
                }
                let other = rows as i64 - count;
                if other < config.min_data_in_leaf as i64
                    || other < config.min_data_per_group as i64
                {
                    break;
                }
                let right_hessian = hessian - left_hessian;
                if right_hessian < config.min_sum_hessian_in_leaf {
                    break;
                }
                if group_count < config.min_data_per_group as i64 {
                    continue;
                }
                group_count = 0;
                if control.threshold.is_some_and(|t| t != i as u32) {
                    continue;
                }
                let gain = controlled_gain(
                    gradient,
                    left_hessian,
                    count as usize,
                    &child_config,
                    control.parent_output,
                    control,
                    Some((0, true)),
                ) + controlled_gain(
                    total_gradient - gradient,
                    right_hessian,
                    other as usize,
                    &child_config,
                    control.parent_output,
                    control,
                    Some((0, false)),
                );
                consider(
                    gradient,
                    left_hessian,
                    count as usize,
                    gain,
                    categories.clone(),
                );
            }
        }
    }
    best
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn forced_split(
    hist: &[Stats<impl Copy>],
    feature: usize,
    threshold: u32,
    missing: MissingType,
    default_bin: u32,
    categorical: bool,
    gradient: f64,
    hessian: f64,
    rows: usize,
    parent: f64,
    config: &Config,
) -> Option<Split> {
    if hist.len() < 2 || threshold as usize >= hist.len() || (categorical && threshold == 0) {
        return None;
    }
    let factor = rows as f64 / hessian;
    let (lg, lh, lc) = if categorical {
        let s = hist[threshold as usize];
        (
            s.gradient,
            s.hessian + EPSILON,
            (s.hessian * factor).round() as usize,
        )
    } else {
        let mut rg = 0.0;
        let mut rh = EPSILON;
        let mut rc = 0;
        for bin in (threshold as usize + 1..hist.len()).rev() {
            if (missing == MissingType::NaN && bin == hist.len() - 1)
                || (missing == MissingType::Zero && bin == default_bin as usize)
            {
                continue;
            }
            rg += hist[bin].gradient;
            rh += hist[bin].hessian;
            rc += (hist[bin].hessian * factor).round() as usize;
        }
        (gradient - rg, hessian - rh, rows.checked_sub(rc)?)
    };
    if lc == 0 || lc >= rows {
        return None;
    }
    let control = SplitControl::default();
    let rg = gradient - lg;
    let rh = hessian - lh;
    let left_output = controlled_output(lg, lh, lc, config, parent, &control, None);
    let right_output = controlled_output(rg, rh, rows - lc, config, parent, &control, None);
    let regularized = gradient.signum() * (gradient.abs() - config.lambda_l1).max(0.0);
    let base = -(2.0 * regularized * parent + (hessian + config.lambda_l2) * parent * parent);
    let gain = controlled_gain(lg, lh, lc, config, parent, &control, None)
        + controlled_gain(rg, rh, rows - lc, config, parent, &control, None)
        - base
        - config.min_gain_to_split;
    if !gain.is_finite() || gain <= 0.0 {
        return None;
    }
    Some(Split {
        feature,
        threshold,
        categories: if categorical {
            vec![threshold]
        } else {
            Vec::new()
        },
        default_left: !categorical,
        gain,
        left_output,
        right_output,
        left_gradient: lg,
        right_gradient: rg,
        left_hessian: lh - EPSILON,
        right_hessian: rh - EPSILON,
    })
}

fn allocation_error(error: std::collections::TryReserveError) -> Error {
    Error::InvalidInput(format!("histogram allocation failed: {error}"))
}

#[cfg(test)]
#[path = "../tests/histogram_optimization/unit.rs"]
mod optimization_tests;

#[cfg(test)]
mod modes_tests {
    use super::*;
    use crate::parallel::Executor;

    #[test]
    fn histogram_modes_keep_exact_feature_read_order() {
        for seed in 0..8 {
            let mut config = Config {
                min_data_in_bin: 1,
                min_data_in_leaf: 1,
                max_bin: 17,
                feature_pre_filter: false,
                categorical_features: vec![1],
                zero_as_missing: seed % 2 == 0,
                ..Config::default()
            };
            let rows = if seed == 0 { 2083 } else { 83 };
            let cols = 9;
            let x: Vec<f64> = (0..rows * cols)
                .map(|i| {
                    let row = i / cols;
                    if i % cols == 1 {
                        (row % 6) as f64
                    } else if (i + seed) % 29 == 0 {
                        f64::NAN
                    } else if (i * 17 + seed) % 19 < 14 {
                        0.0
                    } else {
                        ((i * 13 + seed) % 73) as f64 / 11.0
                    }
                })
                .collect();
            let mut indptr = vec![0];
            let mut indices = Vec::new();
            let mut values = Vec::new();
            for row in x.chunks(cols) {
                for (feature, &value) in row.iter().enumerate() {
                    if value != 0.0 {
                        indices.push(feature);
                        values.push(value);
                    }
                }
                indptr.push(indices.len());
            }
            let dense = Dataset::from_dense(&x, rows, cols, vec![0.; rows], &config).unwrap();
            let sparse = Dataset::from_csr(
                &indptr,
                &indices,
                &values,
                rows,
                cols,
                vec![0.; rows],
                &config,
            )
            .unwrap();
            let gradients: Vec<f32> = (0..rows)
                .map(|i| ((i * 19) % 31) as f32 / 7.0 - 2.0)
                .collect();
            let hessians: Vec<f32> = (0..rows).map(|i| 0.01 + (i % 7) as f32 / 13.0).collect();
            for data in [&dense, &sparse] {
                for constant in [false, true] {
                    for unsorted in [false, true] {
                        let mut selected: Vec<usize> =
                            (0..rows).filter(|i| (i + seed) % 3 != 0).collect();
                        if unsorted {
                            selected.reverse();
                        }
                        config.force_row_wise = false;
                        config.force_col_wise = true;
                        let expected = build_with_executor(
                            data,
                            &selected,
                            &gradients,
                            &hessians,
                            constant,
                            &config,
                            &Executor::new(1).unwrap(),
                            None,
                        )
                        .unwrap();
                        for threads in [1, 2, 4] {
                            config.force_row_wise = true;
                            config.force_col_wise = false;
                            let actual = build_with_executor(
                                data,
                                &selected,
                                &gradients,
                                &hessians,
                                constant,
                                &config,
                                &Executor::new(threads).unwrap(),
                                None,
                            )
                            .unwrap();
                            assert_eq!(
                                actual, expected,
                                "seed={seed}, constant={constant}, threads={threads}"
                            );
                            let pairs = build_training(
                                data,
                                &selected,
                                &gradients,
                                &hessians,
                                constant,
                                &config,
                                &Executor::new(threads).unwrap(),
                                None,
                            )
                            .unwrap();
                            for (pairs, counted) in pairs.iter().zip(&actual) {
                                for (pair, counted) in pairs.iter().zip(counted) {
                                    assert_eq!(pair.gradient.to_bits(), counted.gradient.to_bits());
                                    assert_eq!(pair.hessian.to_bits(), counted.hessian.to_bits());
                                }
                            }
                            let g = selected.iter().map(|&r| f64::from(gradients[r])).sum();
                            let h = selected.iter().map(|&r| f64::from(hessians[r])).sum();
                            let mut actual_fixed = actual;
                            let mut expected_fixed = expected.clone();
                            fix(data, &mut actual_fixed, g, h);
                            fix(data, &mut expected_fixed, g, h);
                            assert_eq!(actual_fixed, expected_fixed);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn typed_dense_slices_preserve_full_root_and_subset_order() {
        fn check<B: Copy + Into<u32>>(values: &[B], bin_count: usize) {
            let gradients = [1e20, 1.0, -1e20, -3.0, 0.25, 1.0];
            let hessians = [0.5, 1.0, 2.0, 3.0, 0.25, 1.0];
            for selected in [vec![0, 1, 2, 3, 4, 5], vec![5, 2, 0, 3], vec![1, 3, 4]] {
                for constant in [false, true] {
                    let mut counted = vec![BinStats::default(); bin_count];
                    let mut pairs = vec![TrainingStats::default(); bin_count];
                    let full = selected == [0, 1, 2, 3, 4, 5];
                    if constant {
                        dense_column::<true, _, _>(
                            &mut counted,
                            values,
                            &selected,
                            &gradients,
                            &hessians,
                            full,
                        );
                        dense_column::<true, _, _>(
                            &mut pairs, values, &selected, &gradients, &hessians, full,
                        );
                    } else {
                        dense_column::<false, _, _>(
                            &mut counted,
                            values,
                            &selected,
                            &gradients,
                            &hessians,
                            full,
                        );
                        dense_column::<false, _, _>(
                            &mut pairs, values, &selected, &gradients, &hessians, full,
                        );
                    }
                    let mut expected = vec![BinStats::default(); bin_count];
                    for &row in &selected {
                        let bin = &mut expected[values[row].into() as usize];
                        bin.gradient += f64::from(gradients[row]);
                        bin.hessian += if constant {
                            1.0
                        } else {
                            f64::from(hessians[row])
                        };
                        bin.count += 1;
                    }
                    assert_eq!(counted, expected);
                    for (pair, expected) in pairs.iter().zip(&expected) {
                        assert_eq!(pair.gradient.to_bits(), expected.gradient.to_bits());
                        assert_eq!(pair.hessian.to_bits(), expected.hessian.to_bits());
                    }
                }
            }
        }
        check(&[0u8, 1, 0, 255, 1, 0], 256);
        check(&[0u16, 257, 0, 299, 257, 0], 300);
        check(&[0u32, 65_536, 0, 65_537, 65_536, 0], 65_538);
    }

    #[test]
    fn training_pairs_preserve_bits_and_actual_counts_at_stack_boundary() {
        use crate::parallel::Executor;
        assert_eq!(std::mem::size_of::<TrainingStats>(), 16);
        assert_eq!(std::mem::size_of::<BinStats>(), 24);
        for max_bin in [255, 256, 257, 511] {
            let rows = 1024;
            let config = Config {
                max_bin,
                min_data_in_bin: 1,
                min_data_in_leaf: 1,
                feature_pre_filter: false,
                force_col_wise: true,
                ..Config::default()
            };
            let x: Vec<f64> = (0..rows * 2).map(|i| (i / 2) as f64).collect();
            let data = Dataset::from_dense(&x, rows, 2, vec![0.; rows], &config).unwrap();
            assert_eq!(data.mapper(0).num_bins() as usize, max_bin);
            let g: Vec<f32> = (0..rows)
                .map(|i| match i % 4 {
                    0 => 1e20,
                    1 => -1e20,
                    2 => 1.0,
                    _ => -0.25,
                })
                .collect();
            let h: Vec<f32> = (0..rows).map(|i| 0.25 + (i % 9) as f32).collect();
            for constant in [false, true] {
                for reverse in [false, true] {
                    let mut selected: Vec<usize> = (0..rows).filter(|r| r % 3 != 0).collect();
                    if reverse {
                        selected.reverse();
                    }
                    let active = [true, false];
                    let counted = build_with_executor(
                        &data,
                        &selected,
                        &g,
                        &h,
                        constant,
                        &config,
                        &Executor::new(1).unwrap(),
                        Some(&active),
                    )
                    .unwrap();
                    let pairs = build_training(
                        &data,
                        &selected,
                        &g,
                        &h,
                        constant,
                        &config,
                        &Executor::new(4).unwrap(),
                        Some(&active),
                    )
                    .unwrap();
                    let mut counts = vec![0; max_bin];
                    for &row in &selected {
                        counts[data.bin(row, 0) as usize] += 1;
                    }
                    for feature in 0..2 {
                        for (index, (pair, counted)) in
                            pairs[feature].iter().zip(&counted[feature]).enumerate()
                        {
                            assert_eq!(counted.count, if feature == 0 { counts[index] } else { 0 });
                            assert_eq!(pair.gradient.to_bits(), counted.gradient.to_bits());
                            assert_eq!(pair.hessian.to_bits(), counted.hessian.to_bits());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn histograms_and_split_outputs_match_pinned_cpp_driver() {
        let fixture: Value =
            serde_json::from_str(include_str!("../tests/fixtures/histograms.json")).unwrap();
        assert_eq!(
            fixture["upstream_commit"],
            "5d84f3f222b41409a6983d2ac90513ad9702fa9e"
        );
        assert_eq!(fixture["schema_version"], 1);
        let cases = fixture["cases"].as_array().unwrap();
        assert!(cases.len() >= 2);
        for case in cases {
            let bins: Vec<u32> = serde_json::from_value(case["bins"].clone()).unwrap();
            let g: Vec<f32> = serde_json::from_value(case["gradients"].clone()).unwrap();
            let h: Vec<f32> = serde_json::from_value(case["hessians"].clone()).unwrap();
            let expected: Vec<[f64; 2]> =
                serde_json::from_value(case["histogram"].clone()).unwrap();
            let mut hist = vec![BinStats::default(); expected.len()];
            for ((&b, &g), &h) in bins.iter().zip(&g).zip(&h) {
                hist[b as usize].gradient += f64::from(g);
                hist[b as usize].hessian += f64::from(h);
                hist[b as usize].count += 1;
            }
            for (bin, want) in hist.iter().zip(expected) {
                assert_eq!([bin.gradient, bin.hessian], want);
            }
            let p = &case["params"];
            let config = Config {
                min_data_in_leaf: p["min_data_in_leaf"].as_u64().unwrap() as usize,
                lambda_l1: p["lambda_l1"].as_f64().unwrap(),
                lambda_l2: p["lambda_l2"].as_f64().unwrap(),
                min_gain_to_split: p["min_gain_to_split"].as_f64().unwrap(),
                max_delta_step: p["max_delta_step"].as_f64().unwrap(),
                ..Config::default()
            };
            let sum_g = g.iter().map(|&x| f64::from(x)).sum();
            let sum_h = h.iter().map(|&x| f64::from(x)).sum();
            let split = best_split(
                &hist,
                0,
                MissingType::None,
                0,
                false,
                sum_g,
                sum_h,
                bins.len(),
                &config,
            )
            .unwrap();
            let want = &case["expected_split"];
            assert_eq!(split.threshold, want["threshold"].as_u64().unwrap() as u32);
            assert_eq!(split.default_left, want["default_left"].as_bool().unwrap());
            for (actual, field) in [
                (split.left_output, "left_output"),
                (split.right_output, "right_output"),
                (split.left_gradient, "left_sum_gradient"),
                (split.right_gradient, "right_sum_gradient"),
                (split.left_hessian, "left_sum_hessian"),
                (split.right_hessian, "right_sum_hessian"),
                (split.gain, "gain"),
            ] {
                assert!(
                    (actual - want[field].as_f64().unwrap()).abs() <= 1e-12,
                    "{} {field}: {actual} != {}",
                    case["name"],
                    want[field]
                );
            }
        }
    }

    #[test]
    fn constant_hessian_histograms_use_first_derivative_even_with_weights() {
        // RegressionMAPELOSS::IsConstantHessian always returns true, even for
        // weighted data. Dataset::ConstructHistogramsInner multiplies counts
        // by hessians[0]; the leaf totals still use the actual derivatives.
        let config = Config {
            min_data_in_leaf: 1,
            min_data_in_bin: 1,
            feature_pre_filter: false,
            ..Config::default()
        };
        let data =
            Dataset::from_dense(&[0., 0., 1., 1.], 4, 1, vec![0., 0., 1., 1.], &config).unwrap();
        let histograms = build(
            &data,
            &[0, 1, 2, 3],
            &[1., 2., 3., 4.],
            &[0.5, 2., 3., 4.],
            true,
        )
        .unwrap();
        for bin in &histograms[0] {
            assert_eq!(bin.hessian, bin.count as f64 * 0.5);
        }
    }

    // feature_histogram.cpp: bin zero is never a category candidate; one-hot
    // ties keep the first bin. Sorted partitions use stable gradient/Hessian
    // order, add cat_l2 only to child gains, and estimate counts from Hessians.
    #[test]
    fn categorical_source_scan_partitions_and_limits() {
        let bins = [
            BinStats {
                gradient: 0.0,
                hessian: 2.0,
                count: 2,
            },
            BinStats {
                gradient: -8.0,
                hessian: 4.0,
                count: 4,
            },
            BinStats {
                gradient: -8.0,
                hessian: 4.0,
                count: 4,
            },
            BinStats {
                gradient: 8.0,
                hessian: 4.0,
                count: 4,
            },
            BinStats {
                gradient: 8.0,
                hessian: 4.0,
                count: 4,
            },
        ];
        let config = Config {
            min_data_in_leaf: 1,
            max_cat_to_onehot: 8,
            cat_smooth: 0.0,
            cat_l2: 0.0,
            min_data_per_group: 1,
            ..Config::default()
        };
        let onehot = best_categorical_split(&bins, 0, 0.0, 18.0, 18, &config).unwrap();
        assert_eq!(onehot.categories, [1]);
        assert!(!onehot.default_left);
        let sorted = Config {
            max_cat_to_onehot: 2,
            ..config.clone()
        };
        let split = best_categorical_split(&bins, 0, 0.0, 18.0, 18, &sorted).unwrap();
        assert_eq!(split.categories, [1, 2]);
        assert!(split.gain > onehot.gain);
        let restricted = Config {
            max_cat_threshold: 1,
            ..sorted.clone()
        };
        assert_eq!(
            best_categorical_split(&bins, 0, 0.0, 18.0, 18, &restricted)
                .unwrap()
                .categories,
            [1]
        );
        let smoothed = Config {
            cat_smooth: 5.0,
            ..sorted.clone()
        };
        assert!(best_categorical_split(&bins, 0, 0.0, 18.0, 18, &smoothed).is_none());
        let grouped = Config {
            min_data_per_group: 10,
            ..sorted.clone()
        };
        assert!(best_categorical_split(&bins, 0, 0.0, 18.0, 18, &grouped).is_none());
        let regularized = Config {
            cat_l2: 10.0,
            ..sorted
        };
        assert!(
            best_categorical_split(&bins, 0, 0.0, 18.0, 18, &regularized)
                .unwrap()
                .gain
                < split.gain
        );
        // Actual counts deliberately disagree: reference uses Hessian estimates.
        let weighted = Config {
            min_data_in_leaf: 5,
            ..config
        };
        let unequal_counts = bins.map(|stats| BinStats {
            count: 100,
            ..stats
        });
        assert!(best_categorical_split(&unequal_counts, 0, 0.0, 18.0, 18, &weighted).is_none());
    }

    #[test]
    fn subtraction_and_leaf_regularization_have_checked_invariants() {
        let parent = [BinStats {
            gradient: 3.0,
            hessian: 4.0,
            count: 3,
        }];
        let child = [BinStats {
            gradient: 1.0,
            hessian: 1.5,
            count: 1,
        }];
        let sibling = subtract(&parent, &child).unwrap();
        assert_eq!(
            sibling,
            [BinStats {
                gradient: 2.0,
                hessian: 2.5,
                count: 2
            }]
        );
        assert!(subtract(&child, &parent).is_err());
        assert!(subtract(&parent, &[]).is_err());
        let config = Config {
            lambda_l1: 1.0,
            lambda_l2: 1.0,
            max_delta_step: 0.5,
            ..Config::default()
        };
        assert_eq!(leaf_output(3.0, 2.0, &config), -0.5);
        assert_eq!(leaf_output(0.5, 2.0, &config), 0.0);
    }
}
