//! Immutable numerical and categorical bin maps.
//!
//! Ported from LightGBM src/io/bin.cpp and include/LightGBM/bin.h at
//! 5d84f3f222b41409a6983d2ac90513ad9702fa9e.
//! Copyright (c) 2016-2026 Microsoft Corporation and The LightGBM developers.
//! Licensed under the MIT License; see THIRD_PARTY_NOTICES.md.

use super::{invalid, reserve};
use crate::{Config, Result};
use std::collections::BTreeMap;

// The upstream constant is a float literal converted to double.
pub(super) const ZERO_THRESHOLD: f64 = 1e-35_f32 as f64;

/// Missing-value handling learned from the training sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissingType {
    None,
    Zero,
    NaN,
}

/// Immutable thresholds and category codes. Validation data must reuse this map.
#[derive(Clone, Debug)]
pub struct BinMapper {
    thresholds: Vec<f64>,
    categories: Vec<i32>,
    category_bins: BTreeMap<i32, u32>,
    categorical: bool,
    missing_type: MissingType,
    default_bin: u32,
    most_freq_bin: u32,
    trivial: bool,
    constant: bool,
    sparse_rate: f64,
    min_value: f64,
    max_value: f64,
}

impl BinMapper {
    pub(super) fn fit(
        mut values: Vec<f64>,
        total: usize,
        nrows: usize,
        categorical: bool,
        feature: usize,
        config: &Config,
    ) -> Result<Self> {
        let original_len = values.len();
        values.retain(|v| !v.is_nan());
        let mut na_count = 0;
        let missing_type = if !config.use_missing {
            MissingType::None
        } else if config.zero_as_missing {
            MissingType::Zero
        } else if values.len() != original_len {
            na_count = original_len - values.len();
            MissingType::NaN
        } else {
            MissingType::None
        };
        let zero_count = total - values.len() - na_count;
        values.sort_by(f64::total_cmp);
        let mut distinct = reserve(values.len() + 1)?;
        let mut counts = reserve(values.len() + 1)?;
        if values.is_empty() || (values[0] > 0.0 && zero_count > 0) {
            distinct.push(0.0);
            counts.push(zero_count);
        }
        if let Some(&first) = values.first() {
            distinct.push(first);
            counts.push(1);
        }
        for pair in values.windows(2) {
            if pair[1] > pair[0].next_up() {
                if pair[0] < 0.0 && pair[1] > 0.0 {
                    distinct.push(0.0);
                    counts.push(zero_count);
                }
                distinct.push(pair[1]);
                counts.push(1);
            } else {
                *distinct.last_mut().expect("nonempty distinct values") = pair[1];
                *counts.last_mut().expect("nonempty counts") += 1;
            }
        }
        if values.last().is_some_and(|v| *v < 0.0) && zero_count > 0 {
            distinct.push(0.0);
            counts.push(zero_count);
        }
        let mut mapper = Self {
            thresholds: Vec::new(),
            categories: Vec::new(),
            category_bins: BTreeMap::new(),
            categorical,
            missing_type,
            default_bin: 0,
            most_freq_bin: 0,
            trivial: false,
            constant: false,
            sparse_rate: 1.0,
            min_value: distinct[0],
            max_value: *distinct.last().expect("nonempty distinct values"),
        };
        let max_bin = config
            .max_bin_by_feature
            .get(feature)
            .copied()
            .unwrap_or(config.max_bin);
        let bin_counts = if categorical {
            mapper.fit_categories(&distinct, &counts, total, na_count, max_bin, config)?
        } else {
            let max_bin = max_bin - usize::from(missing_type == MissingType::NaN);
            let forced = config
                .forced_bins
                .get(feature)
                .map_or(&[][..], Vec::as_slice);
            mapper.thresholds = if forced.is_empty() {
                zero_as_one_bin(
                    &distinct,
                    &counts,
                    max_bin,
                    total - na_count,
                    config.min_data_in_bin,
                )?
            } else {
                predefined(
                    &distinct,
                    &counts,
                    max_bin,
                    total - na_count,
                    config.min_data_in_bin,
                    forced,
                )?
            };
            if missing_type == MissingType::NaN {
                // Upstream pushes the unscoped MissingType::NaN enumerator (2),
                // not an IEEE NaN. This reserved endpoint is never searched.
                mapper.thresholds.push(2.0);
            } else if missing_type == MissingType::Zero && mapper.thresholds.len() == 2 {
                mapper.missing_type = MissingType::None;
            }
            let mut bin_counts = filled(mapper.thresholds.len(), 0_usize)?;
            let mut bin = 0;
            for (&value, &count) in distinct.iter().zip(&counts) {
                while bin < mapper.thresholds.len() - 1 && value > mapper.thresholds[bin] {
                    bin += 1;
                }
                bin_counts[bin] += count;
            }
            if missing_type == MissingType::NaN {
                *bin_counts.last_mut().expect("missing bin") = na_count;
            }
            bin_counts
        };
        mapper.constant = bin_counts.iter().filter(|&&n| n > 0).count() <= 1;
        let filter_count =
            ((config.min_data_in_leaf as f64 * total as f64) / nrows as f64) as usize;
        mapper.trivial = mapper.num_bins() <= 1
            || (config.feature_pre_filter
                && need_filter(&bin_counts, total, filter_count, categorical));
        if !mapper.trivial {
            mapper.default_bin = mapper.bin(0.0);
            let mut most = 0;
            for i in 1..bin_counts.len() {
                if bin_counts[i] > bin_counts[most] {
                    most = i;
                }
            }
            if most != mapper.default_bin as usize && (bin_counts[most] as f64 / total as f64) < 0.7
            {
                most = mapper.default_bin as usize;
            }
            mapper.most_freq_bin = most as u32;
            mapper.sparse_rate = bin_counts[most] as f64 / total as f64;
        }
        Ok(mapper)
    }

    fn fit_categories(
        &mut self,
        distinct: &[f64],
        counts: &[usize],
        total: usize,
        mut na_count: usize,
        max_bin: usize,
        config: &Config,
    ) -> Result<Vec<usize>> {
        let mut categories: Vec<(i32, usize)> = reserve(distinct.len())?;
        for (&value, &count) in distinct.iter().zip(counts) {
            validate_category(value)?;
            let code = value as i32;
            if code < 0 {
                na_count += count;
            } else if categories.last().is_some_and(|&(c, _)| c == code) {
                categories.last_mut().expect("category").1 += count;
            } else {
                categories.push((code, count));
            }
        }
        self.categories.push(-1);
        self.category_bins.insert(-1, 0);
        let mut bin_counts = reserve(categories.len() + 1)?;
        bin_counts.push(total);
        if total == na_count {
            self.missing_type = MissingType::NaN;
            return Ok(bin_counts);
        }
        categories.sort_by(|a, b| b.1.cmp(&a.1));
        // Match the float32 product followed by Common::RoundInt(double).
        let cut_count = (f64::from((total - na_count) as f32 * 0.99_f32) + 0.5) as usize;
        let max_bin = max_bin.min(categories.len() + usize::from(na_count > 0));
        let mut used = 0;
        let mut index = 0;
        while index < categories.len() && (used < cut_count || self.categories.len() < max_bin) {
            let (code, count) = categories[index];
            if count < config.min_data_in_bin && index > 1 {
                break;
            }
            let bin = self.categories.len() as u32;
            self.categories.try_reserve(1).map_err(|e| {
                crate::Error::InvalidInput(format!("category allocation failed: {e}"))
            })?;
            self.categories.push(code);
            self.category_bins.insert(code, bin);
            used += count;
            bin_counts.push(count);
            index += 1;
        }
        self.missing_type = if index == categories.len() && na_count == 0 {
            MissingType::None
        } else {
            MissingType::NaN
        };
        bin_counts[0] = total - used;
        Ok(bin_counts)
    }

    /// Map a raw value. Unknown, negative integer, or invalid category codes use bin 0.
    pub fn bin(&self, mut value: f64) -> u32 {
        if value.is_nan() {
            if self.categorical {
                return 0;
            }
            if self.missing_type == MissingType::NaN {
                return self.num_bins() - 1;
            }
            value = 0.0;
        }
        if self.categorical {
            if !value.is_finite() || value.trunc() < 0.0 || value.trunc() > f64::from(i32::MAX) {
                return 0;
            }
            return self
                .category_bins
                .get(&(value as i32))
                .copied()
                .unwrap_or(0);
        }
        let end = self.thresholds.len() - usize::from(self.missing_type == MissingType::NaN);
        let mut left = 0;
        let mut right = end - 1;
        while left < right {
            let mid = (right + left - 1) / 2;
            if value <= self.thresholds[mid] {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        left as u32
    }

    /// Numerical upper endpoints, including infinity and any reserved missing endpoint.
    /// For NaN missingness, exclude the last entry when scanning split candidates.
    pub fn thresholds(&self) -> &[f64] {
        &self.thresholds
    }
    /// Category code for each bin. Bin 0 is the missing/unseen code -1.
    pub fn categories(&self) -> &[i32] {
        &self.categories
    }
    pub fn num_bins(&self) -> u32 {
        if self.categorical {
            self.categories.len() as u32
        } else {
            self.thresholds.len() as u32
        }
    }
    pub fn default_bin(&self) -> u32 {
        self.default_bin
    }
    pub fn most_freq_bin(&self) -> u32 {
        self.most_freq_bin
    }
    pub fn missing_type(&self) -> MissingType {
        self.missing_type
    }
    pub fn missing_bin(&self) -> Option<u32> {
        match self.missing_type {
            MissingType::None => None,
            MissingType::Zero => Some(self.default_bin),
            MissingType::NaN => Some(if self.categorical {
                0
            } else {
                self.num_bins() - 1
            }),
        }
    }
    pub fn is_categorical(&self) -> bool {
        self.categorical
    }
    /// Match the upstream feature pre-filter flag for this training sample.
    pub fn is_trivial(&self) -> bool {
        self.trivial
    }
    /// Whether all sampled observations have the same bin.
    pub fn is_constant(&self) -> bool {
        self.constant
    }
    pub fn sparse_rate(&self) -> f64 {
        self.sparse_rate
    }

    /// Sampled numerical extrema after zero insertion and adjacent-value merging.
    /// These are training-map metadata, not validation or full-matrix extrema.
    pub fn numerical_range(&self) -> Option<(f64, f64)> {
        (!self.categorical).then_some((self.min_value, self.max_value))
    }

    /// Reference bin information: `[min:max]` or bin-ordered category codes.
    pub fn feature_info(&self) -> String {
        if self.categorical {
            // Native leaves its category vector empty when every observation is
            // missing. Keep our bin-0 sentinel for lookup, but not in this text.
            if self.categories.len() == 1 {
                return String::new();
            }
            self.categories
                .iter()
                .map(i32::to_string)
                .collect::<Vec<_>>()
                .join(":")
        } else {
            format!(
                "[{}:{}]",
                reference_number(self.min_value),
                reference_number(self.max_value)
            )
        }
    }
}

// C++ defaultfloat with setprecision(17), including its exponent spelling.
fn reference_number(value: f64) -> String {
    if !value.is_finite() {
        return value.to_string();
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.into();
    }
    let scientific = format!("{value:.16e}");
    let (mantissa, exponent) = scientific.split_once('e').expect("scientific float");
    let exponent: i32 = exponent.parse().expect("float exponent");
    if !(-4..17).contains(&exponent) {
        let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
        format!("{mantissa}e{exponent:+03}")
    } else {
        let fixed = format!("{:.*}", (16 - exponent) as usize, value);
        if fixed.contains('.') {
            fixed.trim_end_matches('0').trim_end_matches('.').into()
        } else {
            fixed
        }
    }
}

pub(super) fn validate_category(value: f64) -> Result<()> {
    if value.is_nan() {
        return Ok(());
    }
    if !value.is_finite()
        || value.trunc() > f64::from(i32::MAX)
        || value.trunc() < f64::from(i32::MIN)
    {
        return invalid("category codes must be NaN or fit int32 after truncation");
    }
    Ok(())
}

fn filled<T: Clone>(len: usize, value: T) -> Result<Vec<T>> {
    let mut out = reserve(len)?;
    out.resize(len, value);
    Ok(out)
}

fn need_filter(counts: &[usize], total: usize, minimum: usize, categorical: bool) -> bool {
    if categorical && counts.len() > 2 {
        return false;
    }
    let mut left = 0;
    for &count in &counts[..counts.len().saturating_sub(1)] {
        if categorical {
            left = count;
        } else {
            left += count;
        }
        if left >= minimum && total - left >= minimum {
            return false;
        }
    }
    true
}

fn greedy(
    values: &[f64],
    counts: &[usize],
    mut max_bin: usize,
    total: usize,
    minimum: usize,
) -> Result<Vec<f64>> {
    let mut bounds = reserve(max_bin.min(values.len()).max(1))?;
    if values.len() <= max_bin {
        let mut current = 0;
        for i in 0..values.len().saturating_sub(1) {
            current += counts[i];
            if current >= minimum {
                let value = ((values[i] + values[i + 1]) / 2.0).next_up();
                if bounds
                    .last()
                    .is_none_or(|last: &f64| value > last.next_up())
                {
                    bounds.push(value);
                    current = 0;
                }
            }
        }
        bounds.push(f64::INFINITY);
        return Ok(bounds);
    }
    if minimum > 0 {
        max_bin = max_bin.min(total / minimum).max(1);
    }
    let mut mean = total as f64 / max_bin as f64;
    let mut rest_bins = max_bin as isize;
    let mut rest_samples = total as isize;
    let mut big = filled(values.len(), false)?;
    for i in 0..values.len() {
        if counts[i] as f64 >= mean {
            big[i] = true;
            rest_bins -= 1;
            rest_samples -= counts[i] as isize;
        }
    }
    mean = rest_samples as f64 / rest_bins as f64;
    let mut upper = filled(max_bin, f64::INFINITY)?;
    let mut lower = filled(max_bin, f64::INFINITY)?;
    let mut bin = 0;
    lower[0] = values[0];
    let mut current = 0;
    for i in 0..values.len() - 1 {
        if !big[i] {
            rest_samples -= counts[i] as isize;
        }
        current += counts[i];
        if big[i]
            || current as f64 >= mean
            || (big[i + 1] && current as f64 >= (mean * 0.5).max(1.0))
        {
            upper[bin] = values[i];
            bin += 1;
            // For max_bin=1 there is no useful boundary to store.
            if bin >= max_bin {
                bin = max_bin - 1;
                break;
            }
            lower[bin] = values[i + 1];
            if bin >= max_bin - 1 {
                break;
            }
            current = 0;
            if !big[i] {
                rest_bins -= 1;
                mean = rest_samples as f64 / rest_bins as f64;
            }
        }
    }
    for i in 0..bin {
        let value = ((upper[i] + lower[i + 1]) / 2.0).next_up();
        if bounds.last().is_none_or(|last| value > last.next_up()) {
            bounds.push(value);
        }
    }
    bounds.push(f64::INFINITY);
    Ok(bounds)
}

// FindBinWithPredefinedBin: reserve zero endpoints first, then take forced
// endpoints in caller order. Divide the remaining budget among those intervals.
fn predefined(
    values: &[f64],
    counts: &[usize],
    max_bin: usize,
    total: usize,
    minimum: usize,
    forced: &[f64],
) -> Result<Vec<f64>> {
    let left = values.partition_point(|&v| v <= -ZERO_THRESHOLD);
    let has_right = values[left..].iter().any(|&v| v > ZERO_THRESHOLD);
    let mut bounds = reserve(max_bin.min(forced.len().saturating_add(3)))?;
    if max_bin == 2 {
        bounds.push(if left == 0 {
            ZERO_THRESHOLD
        } else {
            -ZERO_THRESHOLD
        });
    } else if max_bin >= 3 {
        if left > 0 {
            bounds.push(-ZERO_THRESHOLD);
        }
        if has_right {
            bounds.push(ZERO_THRESHOLD);
        }
    }
    bounds.push(f64::INFINITY);
    let capacity = max_bin - bounds.len();
    bounds.extend(
        forced
            .iter()
            .copied()
            .filter(|v| v.abs() > ZERO_THRESHOLD)
            .take(capacity),
    );
    bounds.sort_by(f64::total_cmp);
    let free_bins = max_bin - bounds.len();
    let mut additions = reserve(free_bins.min(values.len()))?;
    let mut value_index = 0;
    for (i, &bound) in bounds.iter().enumerate() {
        let start = value_index;
        let mut count = 0;
        // Strict comparison here is intentional; bin lookup uses <= instead.
        while value_index < values.len() && values[value_index] < bound {
            count += counts[value_index];
            value_index += 1;
        }
        let remaining = free_bins - additions.len();
        let bins = if i + 1 == bounds.len() {
            remaining + 1
        } else {
            // total can be zero for an all-NaN sample. There is then no data
            // with which to subdivide an interval.
            let share = if total == 0 {
                0
            } else {
                (count as f64 * free_bins as f64 / total as f64).round() as usize
            };
            share.min(remaining) + 1
        };
        let sub = greedy(
            &values[start..value_index],
            &counts[start..value_index],
            bins,
            count,
            minimum,
        )?;
        additions
            .try_reserve(sub.len() - 1)
            .map_err(|e| crate::Error::InvalidInput(format!("bin allocation failed: {e}")))?;
        additions.extend_from_slice(&sub[..sub.len() - 1]);
    }
    bounds
        .try_reserve(additions.len())
        .map_err(|e| crate::Error::InvalidInput(format!("bin allocation failed: {e}")))?;
    bounds.extend(additions);
    bounds.sort_by(f64::total_cmp);
    Ok(bounds)
}

fn zero_as_one_bin(
    values: &[f64],
    counts: &[usize],
    max_bin: usize,
    total: usize,
    minimum: usize,
) -> Result<Vec<f64>> {
    let (mut left_count, mut zero_count, mut right_count) = (0, 0, 0);
    for (&value, &count) in values.iter().zip(counts) {
        if value <= -ZERO_THRESHOLD {
            left_count += count;
        } else if value > ZERO_THRESHOLD {
            right_count += count;
        } else {
            zero_count += count;
        }
    }
    let left = values
        .iter()
        .position(|&v| v > -ZERO_THRESHOLD)
        .unwrap_or(values.len());
    let mut bounds = if left > 0 && max_bin > 1 {
        let left_bins = ((left_count as f64 / (total - zero_count) as f64 * (max_bin - 1) as f64)
            as usize)
            .max(1);
        let mut bounds = greedy(
            &values[..left],
            &counts[..left],
            left_bins,
            left_count,
            minimum,
        )?;
        *bounds.last_mut().expect("left bin") = -ZERO_THRESHOLD;
        bounds
    } else {
        Vec::new()
    };
    let right = values[left..]
        .iter()
        .position(|&v| v > ZERO_THRESHOLD)
        .map(|i| left + i);
    let right_bins = max_bin.saturating_sub(1 + bounds.len());
    if let Some(right) = right.filter(|_| right_bins > 0) {
        let right_bounds = greedy(
            &values[right..],
            &counts[right..],
            right_bins,
            right_count,
            minimum,
        )?;
        bounds
            .try_reserve(right_bounds.len() + 1)
            .map_err(|e| crate::Error::InvalidInput(format!("bin allocation failed: {e}")))?;
        bounds.push(ZERO_THRESHOLD);
        bounds.extend(right_bounds);
    } else {
        bounds.push(f64::INFINITY);
    }
    Ok(bounds)
}
