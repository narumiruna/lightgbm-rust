//! Checked datasets with immutable shared bin maps and compact column storage.

mod bin;
mod bundle;
#[cfg(test)]
mod bundle_tests;
mod metadata;
pub use bin::{BinMapper, MissingType};

use crate::{Config, Error, Result};
use std::sync::Arc;

/// Borrowed dense storage for checked internal histogram kernels.
#[derive(Clone, Copy)]
pub(crate) enum DenseBins<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
    U32(&'a [u32]),
}

#[derive(Clone, Debug)]
enum CompactBins {
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    Bundle {
        packed: Arc<bundle::Packed>,
        offset: u32,
        num_bins: u32,
        default: u32,
    },
}

impl CompactBins {
    fn width(num_bins: u32) -> usize {
        if num_bins <= 256 {
            1
        } else if num_bins <= 65536 {
            2
        } else {
            4
        }
    }

    fn storage_bytes(&self) -> usize {
        match self {
            Self::U8(v) => v.capacity(),
            Self::U16(v) => v.capacity() * 2,
            Self::U32(v) => v.capacity() * 4,
            Self::Bundle { .. } => 0,
        }
    }

    fn from_iter(values: impl Iterator<Item = u32>, len: usize, num_bins: u32) -> Result<Self> {
        if num_bins <= 256 {
            let mut out = reserve(len)?;
            out.extend(values.map(|v| v as u8));
            Ok(Self::U8(out))
        } else if num_bins <= 65536 {
            let mut out = reserve(len)?;
            out.extend(values.map(|v| v as u16));
            Ok(Self::U16(out))
        } else {
            let mut out = reserve(len)?;
            out.extend(values);
            Ok(Self::U32(out))
        }
    }

    #[inline(always)]
    fn get(&self, row: usize) -> u32 {
        match self {
            Self::U8(v) => u32::from(v[row]),
            Self::U16(v) => u32::from(v[row]),
            Self::U32(v) => v[row],
            Self::Bundle {
                packed,
                offset,
                num_bins,
                default,
            } => {
                let code = packed.values.get(row);
                if code >= *offset && code - offset < *num_bins {
                    code - offset
                } else {
                    *default
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
struct Column {
    values: CompactBins,
    // None is dense; sparse values contain only entries different from bin(0).
    rows: Option<Vec<usize>>,
    default: u32,
    counts: Vec<usize>,
}

impl Column {
    fn storage_bytes(&self) -> usize {
        self.values.storage_bytes()
            + self
                .rows
                .as_ref()
                .map_or(0, |r| r.capacity() * std::mem::size_of::<usize>())
    }

    fn get(&self, row: usize) -> u32 {
        match &self.rows {
            None => self.values.get(row),
            Some(rows) => rows
                .binary_search(&row)
                .map_or(self.default, |i| self.values.get(i)),
        }
    }
}

/// A binned dataset. Feature indexes are the original input column indexes.
///
/// CSR and CSC constructors accept unsorted indexes but reject duplicate coordinates.
/// Implicit sparse entries are zero. NaN is the missing-value input representation.
/// Labels and weights use `f32`; feature values and bin thresholds use `f64`.
/// Training datasets are limited to 65,536 features before column allocation.
#[derive(Clone, Debug)]
pub struct Dataset {
    nrows: usize,
    maps: Arc<[BinMapper]>,
    columns: Vec<Column>,
    bundle_layout: Arc<bundle::Layout>,
    labels: Vec<f32>,
    weights: Option<Vec<f32>>,
    metadata: metadata::Metadata,
    raw: Option<Vec<f32>>,
    prefilter_min_data_in_leaf: Option<usize>,
    binning_config: Arc<Config>,
}

impl Dataset {
    /// Construct from a row-major matrix. Retain raw numerical values only with
    /// `linear_tree=true`, as `f32` with a limit of 16 Mi elements.
    pub fn from_dense(
        data: &[f64],
        nrows: usize,
        ncols: usize,
        labels: Vec<f32>,
        config: &Config,
    ) -> Result<Self> {
        config.validate()?;
        validate_shape(nrows, ncols, &labels)?;
        if data.len() != matrix_len(nrows, ncols)? {
            return invalid("dense matrix length does not match its shape");
        }
        validate_feature_config(config, ncols)?;
        let raw = retained_dense(data, nrows, ncols, config)?;
        let sample = crate::random::Random::new(config.data_random_seed)
            .sample(nrows, nrows.min(config.bin_construct_sample_cnt))?;
        let mut maps = reserve(ncols)?;
        let mut columns = reserve(ncols)?;
        let mut bundle_samples = reserve(ncols)?;
        for feature in 0..ncols {
            let categorical = config.categorical_features.contains(&feature);
            if categorical {
                for row in 0..nrows {
                    bin::validate_category(data[row * ncols + feature])?;
                }
            }
            let mut sampled = reserve(sample.len())?;
            sampled.extend(
                sample
                    .iter()
                    .map(|&row| data[row * ncols + feature])
                    .filter(|&v| v.abs() > bin::ZERO_THRESHOLD || v.is_nan()),
            );
            let mapper =
                BinMapper::fit(sampled, sample.len(), nrows, categorical, feature, config)?;
            columns.push(dense_column(
                (0..nrows).map(|row| mapper.bin(data[row * ncols + feature])),
                nrows,
                &mapper,
            )?);
            if config.enable_bundle {
                let mut entries = reserve(sample.len())?;
                entries.extend(sample.iter().enumerate().filter_map(|(i, &row)| {
                    let value = data[row * ncols + feature];
                    (value.abs() > bin::ZERO_THRESHOLD || value.is_nan()).then_some((i, value))
                }));
                bundle_samples.push(bundle::Sample::new(entries, sample.len(), &mapper)?);
            }
            maps.push(mapper);
        }
        let bundle_layout = Arc::new(bundle::Layout::new(
            &maps,
            &bundle_samples,
            sample.len(),
            nrows,
            config.enable_bundle,
        )?);
        bundle_layout.pack(&mut columns, &maps, nrows, config.enable_bundle)?;
        Ok(Self {
            nrows,
            maps: maps.into(),
            columns,
            bundle_layout,
            labels,
            weights: None,
            metadata: metadata::Metadata::new(ncols)?,
            raw,
            prefilter_min_data_in_leaf: config
                .feature_pre_filter
                .then_some(config.min_data_in_leaf),
            binning_config: Arc::new(config.clone()),
        })
    }

    /// Construct CSR input with checked `usize` pointers and column indexes.
    #[allow(clippy::too_many_arguments)]
    pub fn from_csr(
        indptr: &[usize],
        indices: &[usize],
        values: &[f64],
        nrows: usize,
        ncols: usize,
        labels: Vec<f32>,
        config: &Config,
    ) -> Result<Self> {
        config.validate()?;
        validate_shape(nrows, ncols, &labels)?;
        validate_feature_config(config, ncols)?;
        check_raw_shape(nrows, ncols, config.linear_tree)?;
        validate_sparse(indptr, indices, values, nrows, ncols)?;
        let mut columns: Vec<Vec<(usize, f64)>> = reserve(ncols)?;
        columns.resize_with(ncols, Vec::new);
        for row in 0..nrows {
            for i in indptr[row]..indptr[row + 1] {
                let col = &mut columns[indices[i]];
                col.try_reserve(1).map_err(allocation_error)?;
                col.push((row, values[i]));
            }
        }
        Self::from_sparse_columns(columns.into_iter().map(Ok), nrows, ncols, labels, config)
    }

    /// Construct CSC input with checked `usize` pointers and row indexes.
    #[allow(clippy::too_many_arguments)]
    pub fn from_csc(
        indptr: &[usize],
        indices: &[usize],
        values: &[f64],
        nrows: usize,
        ncols: usize,
        labels: Vec<f32>,
        config: &Config,
    ) -> Result<Self> {
        config.validate()?;
        validate_shape(nrows, ncols, &labels)?;
        validate_feature_config(config, ncols)?;
        check_raw_shape(nrows, ncols, config.linear_tree)?;
        validate_sparse(indptr, indices, values, ncols, nrows)?;
        let columns = (0..ncols).map(|feature| {
            let range = indptr[feature]..indptr[feature + 1];
            let mut column = reserve(range.len())?;
            column.extend(range.map(|i| (indices[i], values[i])));
            Ok(column)
        });
        Self::from_sparse_columns(columns, nrows, ncols, labels, config)
    }

    fn from_sparse_columns(
        raw_columns: impl Iterator<Item = Result<Vec<(usize, f64)>>>,
        nrows: usize,
        ncols: usize,
        labels: Vec<f32>,
        config: &Config,
    ) -> Result<Self> {
        validate_feature_config(config, ncols)?;
        let mut raw_values = raw_buffer(nrows, ncols, config.linear_tree)?;
        let sample = crate::random::Random::new(config.data_random_seed)
            .sample(nrows, nrows.min(config.bin_construct_sample_cnt))?;
        let mut maps = reserve(ncols)?;
        let mut columns = reserve(ncols)?;
        let mut bundle_samples = reserve(ncols)?;
        for (feature, raw) in raw_columns.enumerate() {
            let mut raw = raw?;
            raw.sort_unstable_by_key(|&(row, _)| row);
            if raw.windows(2).any(|w| w[0].0 == w[1].0) {
                return invalid("duplicate sparse coordinates are not supported");
            }
            let categorical = config.categorical_features.contains(&feature);
            if categorical {
                for &(_, value) in &raw {
                    bin::validate_category(value)?;
                }
            }
            let mut sampled = reserve(sample.len().min(raw.len()))?;
            sampled.extend(
                raw.iter()
                    .filter(|(row, value)| {
                        sample.binary_search(row).is_ok()
                            && (value.abs() > bin::ZERO_THRESHOLD || value.is_nan())
                    })
                    .map(|&(_, value)| value),
            );
            let mapper =
                BinMapper::fit(sampled, sample.len(), nrows, categorical, feature, config)?;
            if config.enable_bundle {
                let mut entries = reserve(sample.len().min(raw.len()))?;
                entries.extend(raw.iter().filter_map(|&(row, value)| {
                    if value.abs() > bin::ZERO_THRESHOLD || value.is_nan() {
                        sample.binary_search(&row).ok().map(|i| (i, value))
                    } else {
                        None
                    }
                }));
                bundle_samples.push(bundle::Sample::new(entries, sample.len(), &mapper)?);
            }
            let default = mapper.bin(0.0);
            let mut rows = reserve(raw.len())?;
            let mut bins = reserve(raw.len())?;
            let mut counts = zeros(mapper.num_bins() as usize)?;
            counts[default as usize] = nrows;
            for (row, value) in raw {
                if !categorical && let Some(values) = &mut raw_values {
                    values[row * ncols + feature] = value as f32;
                }
                let b = mapper.bin(value);
                if b != default {
                    rows.push(row);
                    bins.push(b);
                    counts[default as usize] -= 1;
                    counts[b as usize] += 1;
                }
            }
            columns.push(Column {
                values: CompactBins::from_iter(
                    bins.iter().copied(),
                    bins.len(),
                    mapper.num_bins(),
                )?,
                rows: Some(rows),
                default,
                counts,
            });
            maps.push(mapper);
        }
        let bundle_layout = Arc::new(bundle::Layout::new(
            &maps,
            &bundle_samples,
            sample.len(),
            nrows,
            config.enable_bundle,
        )?);
        bundle_layout.pack(&mut columns, &maps, nrows, config.enable_bundle)?;
        Ok(Self {
            nrows,
            maps: maps.into(),
            columns,
            bundle_layout,
            labels,
            weights: None,
            metadata: metadata::Metadata::new(ncols)?,
            raw: raw_values,
            prefilter_min_data_in_leaf: config
                .feature_pre_filter
                .then_some(config.min_data_in_leaf),
            binning_config: Arc::new(config.clone()),
        })
    }

    /// Attach finite, nonnegative weights. At least one weight must be positive.
    pub fn with_weights(mut self, weights: Vec<f32>) -> Result<Self> {
        if self.groups().is_some_and(|groups| groups.contains(&0)) {
            return Err(Error::Unsupported(
                "weights with empty queries have undefined native query weights (sum / 0)".into(),
            ));
        }
        if weights.len() != self.nrows
            || weights.iter().any(|w| !w.is_finite() || *w < 0.0)
            || !weights.iter().any(|w| *w > 0.0)
        {
            return invalid(
                "weights must match rows, be finite and nonnegative, and have positive total weight",
            );
        }
        self.weights = Some(weights);
        Ok(self)
    }

    /// Construct validation data with exactly the training bin maps.
    pub fn validation_from_dense(
        &self,
        data: &[f64],
        nrows: usize,
        ncols: usize,
        labels: Vec<f32>,
    ) -> Result<Self> {
        validate_shape(nrows, ncols, &labels)?;
        if ncols != self.ncols() || data.len() != matrix_len(nrows, ncols)? {
            return invalid("validation matrix shape does not match training features");
        }
        let raw = retained_dense(data, nrows, ncols, &self.binning_config)?;
        let mut columns = reserve(ncols)?;
        for (feature, mapper) in self.maps.iter().enumerate() {
            if mapper.is_categorical() {
                for row in 0..nrows {
                    bin::validate_category(data[row * ncols + feature])?;
                }
            }
            columns.push(dense_column(
                (0..nrows).map(|row| mapper.bin(data[row * ncols + feature])),
                nrows,
                mapper,
            )?);
        }
        self.bundle_layout.pack(
            &mut columns,
            &self.maps,
            nrows,
            self.binning_config.enable_bundle,
        )?;
        Ok(Self {
            nrows,
            maps: Arc::clone(&self.maps),
            columns,
            bundle_layout: Arc::clone(&self.bundle_layout),
            labels,
            weights: None,
            metadata: self.metadata.validation(),
            raw,
            prefilter_min_data_in_leaf: self.prefilter_min_data_in_leaf,
            binning_config: Arc::clone(&self.binning_config),
        })
    }

    /// Copy rows in caller order and preserve metadata. Without groups, repeated
    /// rows are permitted. With groups, select complete, nonrepeated groups only.
    /// Empty queries are preserved for identity selections. Other row-only selections
    /// with empty queries are unsupported: rows do not identify empty-query ownership,
    /// and native Metadata::Init indexes before the selected row for zero lengths.
    pub fn subset(&self, rows: &[usize]) -> Result<Self> {
        if rows.is_empty()
            || rows.len() > i32::MAX as usize
            || rows.iter().any(|&r| r >= self.nrows)
        {
            return invalid("subset must contain valid row indexes and fit int32");
        }
        matrix_len(rows.len(), self.ncols())?;
        let metadata = self.metadata.subset(rows, self.nrows)?;
        let raw = self
            .raw
            .as_ref()
            .map(|source| -> Result<Vec<f32>> {
                check_raw_shape(rows.len(), self.ncols(), true)?;
                let mut out = reserve(matrix_len(rows.len(), self.ncols())?)?;
                for &row in rows {
                    out.extend_from_slice(&source[row * self.ncols()..(row + 1) * self.ncols()]);
                }
                Ok(out)
            })
            .transpose()?;
        let mut columns = reserve(self.ncols())?;
        let mut packed_subsets: Vec<Option<Arc<bundle::Packed>>> = reserve(self.ncols())?;
        packed_subsets.resize_with(self.ncols(), || None);
        for group in &self.bundle_layout.groups {
            if let Some(&first) = group.first()
                && let CompactBins::Bundle { packed, .. } = &self.columns[first].values
            {
                let packed = Arc::new(bundle::Packed {
                    values: CompactBins::from_iter(
                        rows.iter().map(|&r| packed.values.get(r)),
                        rows.len(),
                        packed.num_bins,
                    )?,
                    num_bins: packed.num_bins,
                });
                for &feature in group {
                    packed_subsets[feature] = Some(Arc::clone(&packed));
                }
            }
        }
        for (feature, mapper) in self.maps.iter().enumerate() {
            let column = &self.columns[feature];
            if let CompactBins::Bundle {
                offset,
                num_bins,
                default,
                ..
            } = &column.values
            {
                let mut counts = zeros(mapper.num_bins() as usize)?;
                for &r in rows {
                    counts[column.get(r) as usize] += 1;
                }
                columns.push(Column {
                    values: CompactBins::Bundle {
                        packed: Arc::clone(
                            packed_subsets[feature]
                                .as_ref()
                                .expect("packed subset exists"),
                        ),
                        offset: *offset,
                        num_bins: *num_bins,
                        default: *default,
                    },
                    rows: None,
                    default: *default,
                    counts,
                });
            } else if column.rows.is_some() {
                let default = column.default;
                let mut out_rows = Vec::new();
                let mut bins = Vec::new();
                let mut counts = zeros(mapper.num_bins() as usize)?;
                for (i, &row) in rows.iter().enumerate() {
                    let b = column.get(row);
                    counts[b as usize] += 1;
                    if b != default {
                        out_rows.try_reserve(1).map_err(allocation_error)?;
                        bins.try_reserve(1).map_err(allocation_error)?;
                        out_rows.push(i);
                        bins.push(b);
                    }
                }
                columns.push(Column {
                    values: CompactBins::from_iter(
                        bins.iter().copied(),
                        bins.len(),
                        mapper.num_bins(),
                    )?,
                    rows: Some(out_rows),
                    default,
                    counts,
                });
            } else {
                columns.push(dense_column(
                    rows.iter().map(|&r| column.get(r)),
                    rows.len(),
                    mapper,
                )?);
            }
        }
        let mut labels = reserve(rows.len())?;
        labels.extend(rows.iter().map(|&r| self.labels[r]));
        let weights = self
            .weights
            .as_ref()
            .map(|w| -> Result<Vec<f32>> {
                let mut out = reserve(rows.len())?;
                out.extend(rows.iter().map(|&r| w[r]));
                Ok(out)
            })
            .transpose()?;
        Ok(Self {
            nrows: rows.len(),
            maps: Arc::clone(&self.maps),
            columns,
            bundle_layout: Arc::clone(&self.bundle_layout),
            labels,
            weights,
            metadata,
            raw,
            prefilter_min_data_in_leaf: self.prefilter_min_data_in_leaf,
            binning_config: Arc::clone(&self.binning_config),
        })
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }
    pub fn ncols(&self) -> usize {
        self.maps.len()
    }
    pub fn labels(&self) -> &[f32] {
        &self.labels
    }
    pub fn weights(&self) -> Option<&[f32]> {
        self.weights.as_deref()
    }
    pub(crate) fn check_binning_config(&self, config: &Config) -> Result<()> {
        let stored = &self.binning_config;
        if stored.max_bin_by_feature != config.max_bin_by_feature
            || stored.forced_bins != config.forced_bins
            || stored.enable_bundle != config.enable_bundle
            || stored.linear_tree != config.linear_tree
            || stored.max_bin != config.max_bin
            || stored.min_data_in_bin != config.min_data_in_bin
            || stored.bin_construct_sample_cnt != config.bin_construct_sample_cnt
            || stored.data_random_seed != config.data_random_seed
            || stored.use_missing != config.use_missing
            || stored.zero_as_missing != config.zero_as_missing
            || stored.feature_pre_filter != config.feature_pre_filter
            || stored.categorical_features != config.categorical_features
        {
            return invalid(
                "trainer binning, category, or missing-value settings differ from the dataset; rebuild it with matching settings",
            );
        }
        Ok(())
    }

    /// Native inner index; filtered features return `usize::MAX` (native -1).
    pub(crate) fn inner_feature_index(&self, original: usize) -> usize {
        self.bundle_layout.inner[original]
    }

    /// Allocated bin-value and sparse row-index bytes, counting shared packed
    /// buffers once. Excludes bin maps, counts, metadata, raw values and allocator
    /// overhead. Use process peak RSS separately for construction benchmarks.
    pub fn bin_storage_bytes(&self) -> usize {
        let mut bytes = self
            .columns
            .iter()
            .map(Column::storage_bytes)
            .sum::<usize>();
        for group in &self.bundle_layout.groups {
            if let Some(&feature) = group.first()
                && let CompactBins::Bundle { packed, .. } = &self.columns[feature].values
            {
                bytes += packed.values.storage_bytes()
                    + std::mem::size_of::<bundle::Packed>()
                    + 2 * std::mem::size_of::<usize>();
            }
        }
        bytes
    }

    pub(crate) fn prefilter_min_data_in_leaf(&self) -> Option<usize> {
        self.prefilter_min_data_in_leaf
    }
    /// Return a bin, or an error if either index is outside the dataset shape.
    pub fn get_bin(&self, row: usize, feature: usize) -> Result<u32> {
        if row >= self.nrows || feature >= self.ncols() {
            return invalid("bin indexes exceed the dataset shape");
        }
        Ok(self.columns[feature].get(row))
    }

    /// Return a mapper, or an error if the feature index is outside the dataset shape.
    pub fn get_mapper(&self, feature: usize) -> Result<&BinMapper> {
        self.maps
            .get(feature)
            .ok_or_else(|| Error::InvalidInput("feature index exceeds dataset columns".into()))
    }

    /// Internal hot-path access; callers validate both indexes.
    pub(crate) fn bin(&self, row: usize, feature: usize) -> u32 {
        assert!(row < self.nrows, "row index out of bounds");
        self.columns[feature].get(row)
    }
    /// Internal hot-path access; callers validate the feature index.
    pub(crate) fn mapper(&self, feature: usize) -> &BinMapper {
        &self.maps[feature]
    }
    /// Borrow a plain dense column without unpacking sparse or bundled data.
    pub(crate) fn dense_bins(&self, feature: usize) -> Option<DenseBins<'_>> {
        let column = self.columns.get(feature)?;
        if column.rows.is_some() {
            return None;
        }
        match &column.values {
            CompactBins::U8(values) => Some(DenseBins::U8(values)),
            CompactBins::U16(values) => Some(DenseBins::U16(values)),
            CompactBins::U32(values) => Some(DenseBins::U32(values)),
            CompactBins::Bundle { .. } => None,
        }
    }

    pub(crate) fn sparse_bins(
        &self,
        feature: usize,
    ) -> Option<impl Iterator<Item = (usize, u32)> + '_> {
        let column = &self.columns[feature];
        column.rows.as_ref().map(move |rows| {
            rows.iter()
                .enumerate()
                .map(move |(position, &row)| (row, column.values.get(position)))
        })
    }

    // Learner row partitions are sorted. Walk a sparse column once instead of
    // performing one binary search for every row.
    pub(crate) fn visit_bins(
        &self,
        feature: usize,
        rows: &[usize],
        mut visit: impl FnMut(usize, u32),
    ) {
        debug_assert!(rows.windows(2).all(|w| w[0] < w[1]));
        let column = &self.columns[feature];
        if let Some(indexes) = &column.rows {
            let mut position = rows
                .first()
                .map_or(0, |first| indexes.partition_point(|r| r < first));
            for &row in rows {
                while position < indexes.len() && indexes[position] < row {
                    position += 1;
                }
                let bin = if indexes.get(position) == Some(&row) {
                    column.values.get(position)
                } else {
                    column.default
                };
                visit(row, bin);
            }
        } else {
            match &column.values {
                CompactBins::U8(values) => {
                    for &row in rows {
                        visit(row, u32::from(values[row]));
                    }
                }
                CompactBins::U16(values) => {
                    for &row in rows {
                        visit(row, u32::from(values[row]));
                    }
                }
                CompactBins::U32(values) => {
                    for &row in rows {
                        visit(row, values[row]);
                    }
                }
                CompactBins::Bundle { .. } => {
                    for &row in rows {
                        visit(row, column.get(row));
                    }
                }
            }
        }
    }

    /// Full dataset counts, including implicit sparse zeros.
    /// Return an error if the feature index is outside the dataset shape.
    pub fn get_bin_counts(&self, feature: usize) -> Result<&[usize]> {
        if feature >= self.ncols() {
            return invalid("feature index exceeds dataset columns");
        }
        Ok(self.bin_counts(feature))
    }

    /// Internal hot-path access; callers validate the feature index.
    pub(crate) fn bin_counts(&self, feature: usize) -> &[usize] {
        &self.columns[feature].counts
    }
}

fn dense_column(
    values: impl Iterator<Item = u32>,
    nrows: usize,
    mapper: &BinMapper,
) -> Result<Column> {
    let mut counts = zeros(mapper.num_bins() as usize)?;
    let values = CompactBins::from_iter(
        values.inspect(|&b| counts[b as usize] += 1),
        nrows,
        mapper.num_bins(),
    )?;
    Ok(Column {
        values,
        rows: None,
        default: mapper.bin(0.0),
        counts,
    })
}

pub(super) fn reserve<T>(len: usize) -> Result<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(allocation_error)?;
    Ok(out)
}

fn zeros(len: usize) -> Result<Vec<usize>> {
    let mut out = reserve(len)?;
    out.resize(len, 0);
    Ok(out)
}
fn allocation_error(error: std::collections::TryReserveError) -> Error {
    Error::InvalidInput(format!("dataset allocation failed: {error}"))
}
pub(super) fn invalid<T>(message: &str) -> Result<T> {
    Err(Error::InvalidInput(message.into()))
}

fn matrix_len(nrows: usize, ncols: usize) -> Result<usize> {
    nrows
        .checked_mul(ncols)
        .filter(|&n| n <= isize::MAX as usize / std::mem::size_of::<f64>())
        .ok_or_else(|| Error::InvalidInput("matrix shape exceeds allocation limits".into()))
}

fn validate_shape(nrows: usize, ncols: usize, labels: &[f32]) -> Result<()> {
    if nrows == 0 || ncols == 0 || nrows > i32::MAX as usize || ncols > i32::MAX as usize {
        return invalid("dimensions must be positive and fit int32");
    }
    if ncols > 65_536 {
        return invalid("dataset feature count exceeds 65536 allocation limit");
    }
    matrix_len(nrows, ncols)?;
    if labels.len() != nrows || labels.iter().any(|v| !v.is_finite()) {
        return invalid("labels must match rows and be finite");
    }
    Ok(())
}

fn validate_feature_config(config: &Config, ncols: usize) -> Result<()> {
    if !config.max_bin_by_feature.is_empty() && config.max_bin_by_feature.len() != ncols {
        return invalid("max_bin_by_feature must be empty or match the column count");
    }
    if !config.forced_bins.is_empty() && config.forced_bins.len() != ncols {
        return invalid("forced_bins must be empty or match the column count");
    }
    if config.categorical_features.iter().any(|&f| f >= ncols) {
        return invalid("categorical feature index exceeds matrix columns");
    }
    Ok(())
}

const MAX_RAW_ELEMENTS: usize = 16 * 1024 * 1024;

fn check_raw_shape(nrows: usize, ncols: usize, retain: bool) -> Result<()> {
    if retain && matrix_len(nrows, ncols)? > MAX_RAW_ELEMENTS {
        return invalid("linear raw matrix exceeds 16 Mi element allocation limit");
    }
    Ok(())
}

fn raw_buffer(nrows: usize, ncols: usize, retain: bool) -> Result<Option<Vec<f32>>> {
    check_raw_shape(nrows, ncols, retain)?;
    if !retain {
        return Ok(None);
    }
    let len = matrix_len(nrows, ncols)?;
    let mut raw = reserve(len)?;
    raw.resize(len, 0.0);
    Ok(Some(raw))
}

fn retained_dense(
    data: &[f64],
    nrows: usize,
    ncols: usize,
    config: &Config,
) -> Result<Option<Vec<f32>>> {
    let mut raw = raw_buffer(nrows, ncols, config.linear_tree)?;
    if let Some(raw) = &mut raw {
        for (i, value) in raw.iter_mut().enumerate() {
            *value = if config.categorical_features.contains(&(i % ncols)) {
                f32::NAN
            } else {
                data[i] as f32
            };
        }
    }
    Ok(raw)
}

fn validate_sparse(
    indptr: &[usize],
    indices: &[usize],
    values: &[f64],
    major: usize,
    minor: usize,
) -> Result<()> {
    if indptr.len()
        != major
            .checked_add(1)
            .ok_or_else(|| Error::InvalidInput("sparse shape overflow".into()))?
        || indptr.first() != Some(&0)
        || indptr.last() != Some(&values.len())
        || indices.len() != values.len()
        || indptr.windows(2).any(|w| w[0] > w[1])
        || indices.iter().any(|&i| i >= minor)
    {
        return invalid("invalid sparse pointers, indexes, or value lengths");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_borrows_preserve_each_storage_width_and_exclude_sparse_columns() {
        let config = Config {
            min_data_in_leaf: 1,
            min_data_in_bin: 1,
            feature_pre_filter: false,
            enable_bundle: false,
            ..Config::default()
        };
        let mut dense = Dataset::from_dense(&[0., 1., 2., 3.], 4, 1, vec![0.; 4], &config).unwrap();
        let codes: Vec<_> = (0..4).map(|row| dense.get_bin(row, 0).unwrap()).collect();
        for (bins, width) in [(256, 8), (65536, 16), (65537, 32)] {
            dense.columns[0].values =
                CompactBins::from_iter(codes.iter().copied(), 4, bins).unwrap();
            let borrowed: Vec<u32> = match dense.dense_bins(0).unwrap() {
                DenseBins::U8(values) => {
                    assert_eq!(width, 8);
                    values.iter().map(|&v| u32::from(v)).collect()
                }
                DenseBins::U16(values) => {
                    assert_eq!(width, 16);
                    values.iter().map(|&v| u32::from(v)).collect()
                }
                DenseBins::U32(values) => {
                    assert_eq!(width, 32);
                    values.to_vec()
                }
            };
            assert_eq!(borrowed, codes);
        }
        assert!(dense.dense_bins(usize::MAX).is_none());
        let mut offsets = vec![0; 33];
        offsets[32] = 1;
        let sparse =
            Dataset::from_csr(&offsets, &[0], &[1.], 32, 1, vec![0.; 32], &config).unwrap();
        assert!(sparse.sparse_bins(0).is_some());
        assert!(sparse.dense_bins(0).is_none());
    }

    #[test]
    fn raw_allocation_limit_is_checked_without_allocating() {
        check_raw_shape(256, 65_536, true).unwrap();
        assert!(check_raw_shape(257, 65_536, true).is_err());
        assert!(check_raw_shape(usize::MAX, 2, true).is_err());
        check_raw_shape(257, 65_536, false).unwrap();
    }

    #[test]
    fn binning_fingerprint_includes_dataset_options_but_not_tree_options() {
        let config = Config::default();
        let dataset = Dataset::from_dense(&[0.0, 1.0], 2, 1, vec![0.0; 2], &config).unwrap();
        let changes: &[fn(&mut Config)] = &[
            |c| c.max_bin_by_feature = vec![4],
            |c| c.forced_bins = vec![vec![0.5]],
            |c| c.linear_tree = true,
        ];
        for change in changes {
            let mut changed = config.clone();
            change(&mut changed);
            assert!(dataset.check_binning_config(&changed).is_err());
            assert!(
                dataset
                    .subset(&[0])
                    .unwrap()
                    .check_binning_config(&changed)
                    .is_err()
            );
            assert!(
                dataset
                    .validation_from_dense(&[1.0], 1, 1, vec![0.0])
                    .unwrap()
                    .check_binning_config(&changed)
                    .is_err()
            );
        }
        let mut changed = config.clone();
        changed.max_cat_threshold += 1;
        changed.max_cat_to_onehot += 1;
        changed.cat_smooth += 1.0;
        changed.cat_l2 += 1.0;
        changed.min_data_per_group += 1;
        dataset.check_binning_config(&changed).unwrap();
    }

    #[test]
    fn feature_prefilter_metadata_survives_reference_and_subset_operations() {
        for feature_pre_filter in [false, true] {
            let config = Config {
                feature_pre_filter,
                ..Config::default()
            };
            let dataset = Dataset::from_dense(&[0.0, 1.0], 2, 1, vec![0.0; 2], &config).unwrap();
            let expected = feature_pre_filter.then_some(config.min_data_in_leaf);
            assert_eq!(dataset.clone().prefilter_min_data_in_leaf(), expected);
            assert_eq!(
                dataset.subset(&[1]).unwrap().prefilter_min_data_in_leaf(),
                expected
            );
            assert_eq!(
                dataset
                    .validation_from_dense(&[2.0], 1, 1, vec![0.0])
                    .unwrap()
                    .prefilter_min_data_in_leaf(),
                expected
            );
        }
    }

    #[test]
    fn sparse_columns_store_only_nondefault_entries() {
        let config = Config {
            min_data_in_leaf: 1,
            min_data_in_bin: 1,
            ..Config::default()
        };
        let dataset = Dataset::from_csc(
            &[0, 2, 2],
            &[0, 999],
            &[-1.0, 1.0],
            1000,
            2,
            vec![0.0; 1000],
            &config,
        )
        .unwrap();
        for column in &dataset.columns {
            assert!(column.rows.as_ref().unwrap().len() <= 2);
        }
        let subset = dataset.subset(&[0, 1, 999]).unwrap();
        assert!(subset.columns.iter().all(|c| c.rows.is_some()));
        assert_eq!(subset.columns[1].rows.as_ref().unwrap().capacity(), 0);
    }

    #[test]
    fn compact_storage_preserves_each_supported_width() {
        for (num_bins, values, width) in [
            (256, vec![0, 255, 1], 8),
            (65536, vec![0, 65535, 256], 16),
            (65537, vec![0, 65536, 256], 32),
        ] {
            let packed =
                CompactBins::from_iter(values.iter().copied(), values.len(), num_bins).unwrap();
            assert!(matches!(
                (&packed, width),
                (CompactBins::U8(_), 8) | (CompactBins::U16(_), 16) | (CompactBins::U32(_), 32)
            ));
            for (i, &value) in values.iter().enumerate() {
                assert_eq!(packed.get(i), value);
            }
        }
    }
}
