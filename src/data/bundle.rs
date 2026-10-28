//! Native CPU feature order and lossless exclusive storage are separate contracts.
//!
//! Group order follows Dataset::FastFeatureBundling at the pinned source revision.
//! Sorted coordinates replace native per-group row bitmaps. Physical packing is
//! stricter: only groups with no full-data conflicts and smaller storage are packed.
//!
//! Order ported from LightGBM src/io/dataset.cpp at
//! 5d84f3f222b41409a6983d2ac90513ad9702fa9e (MIT).
//! Copyright (c) 2016-2026 Microsoft Corporation and The LightGBM developers.
//! See THIRD_PARTY_NOTICES.md.

use super::{BinMapper, Column, CompactBins, allocation_error, reserve, zeros};
use crate::{Result, random::Random};
use std::sync::Arc;

#[derive(Debug)]
pub(super) struct Sample {
    rows: Vec<usize>,
    raw_count: usize,
}

impl Sample {
    pub(super) fn new(
        mut entries: Vec<(usize, f64)>,
        total: usize,
        mapper: &BinMapper,
    ) -> Result<Self> {
        let raw_count = entries.len();
        let mut rows = reserve(raw_count)?;
        rows.extend(entries.iter().map(|&(row, _)| row));
        if !mapper.is_trivial() && mapper.default_bin() != mapper.most_freq_bin() {
            // Native FindBin compacts NaNs then sorts its input values in place,
            // but leaves sample row indexes unchanged. FixSampleIndices sees this
            // mutated array, including the unmodified trailing entries.
            let mut values = reserve(raw_count)?;
            values.extend(entries.iter().map(|&(_, value)| value));
            let mut non_nan = 0;
            for i in 0..values.len() {
                if !values[i].is_nan() {
                    values[non_nan] = values[i];
                    non_nan += 1;
                }
            }
            values[..non_nan].sort_by(f64::total_cmp);
            for (entry, value) in entries.iter_mut().zip(values) {
                entry.1 = value;
            }
            let mut fixed = reserve(total)?;
            let mut j = 0;
            for row in 0..total {
                if j < entries.len() && entries[j].0 == row {
                    if mapper.bin(entries[j].1) != mapper.most_freq_bin() {
                        fixed.push(row);
                    }
                    j += 1;
                } else {
                    fixed.push(row);
                }
            }
            // Native retains the original row indexes when the result is empty.
            if !fixed.is_empty() {
                rows = fixed;
            }
        }
        Ok(Self { rows, raw_count })
    }
}

#[derive(Clone, Debug)]
pub(super) struct Layout {
    pub(super) inner: Vec<usize>,
    pub(super) groups: Vec<Vec<usize>>,
}

#[derive(Debug)]
struct Group {
    features: Vec<usize>,
    rows: Vec<usize>,
    total: usize,
}

fn conflicts(a: &[usize], b: &[usize]) -> usize {
    b.iter().filter(|r| a.binary_search(r).is_ok()).count()
}

fn union(a: &mut Vec<usize>, b: &[usize]) -> Result<()> {
    a.try_reserve(b.len()).map_err(allocation_error)?;
    a.extend_from_slice(b);
    a.sort_unstable();
    a.dedup();
    Ok(())
}

fn find_groups(
    order: &[usize],
    samples: &[Sample],
    n: usize,
    nrows: usize,
) -> Result<Vec<Vec<usize>>> {
    let max_conflict = n / 10000;
    let mut rng = Random::new(nrows as u32);
    let mut groups: Vec<Group> = reserve(order.len())?;
    for &feature in order {
        let rows = &samples[feature].rows;
        let mut available = reserve(groups.len())?;
        available.extend(
            groups
                .iter()
                .enumerate()
                .filter_map(|(i, g)| (g.total + rows.len() <= n + max_conflict).then_some(i)),
        );
        let mut search = reserve(100.min(available.len()))?;
        if let Some(&last) = available.last() {
            search.push(last);
            let count = available.len() - 1;
            search.extend(
                rng.sample(count, count.min(99))?
                    .into_iter()
                    .map(|i| available[i]),
            );
        }
        let found = search.into_iter().find(|&i| {
            let g = &groups[i];
            let rest = max_conflict + g.rows.len() - g.total;
            let count = conflicts(&g.rows, rows);
            count <= rest && count <= rows.len() / 2
        });
        if let Some(i) = found {
            groups[i]
                .features
                .try_reserve(1)
                .map_err(allocation_error)?;
            groups[i].features.push(feature);
            groups[i].total += rows.len();
            union(&mut groups[i].rows, rows)?;
        } else {
            let mut stored_rows = reserve(rows.len())?;
            stored_rows.extend_from_slice(rows);
            let mut features = reserve(1)?;
            features.push(feature);
            groups.push(Group {
                features,
                rows: stored_rows,
                total: rows.len(),
            });
        }
    }
    // Sparse-enabled CPU loader: all groups below density 0.4 are combined
    // into one multi-value group. This changes feature order even if not packed.
    let mut output = reserve(groups.len())?;
    let mut sparse = Vec::new();
    for g in groups {
        if g.rows.len() as f64 / n as f64 >= 0.4 {
            output.push(g.features);
        } else {
            sparse
                .try_reserve(g.features.len())
                .map_err(allocation_error)?;
            sparse.extend(g.features);
        }
    }
    if !sparse.is_empty() {
        output.push(sparse);
    }
    Ok(output)
}

impl Layout {
    pub(super) fn new(
        maps: &[BinMapper],
        samples: &[Sample],
        n: usize,
        nrows: usize,
        enabled: bool,
    ) -> Result<Self> {
        let mut used = reserve(maps.len())?;
        used.extend(
            maps.iter()
                .enumerate()
                .filter_map(|(i, m)| (!m.is_trivial()).then_some(i)),
        );
        let mut groups = reserve(used.len())?;
        if enabled && !used.is_empty() {
            let mut sorted = reserve(used.len())?;
            sorted.extend_from_slice(&used);
            sorted.sort_by_key(|&f| std::cmp::Reverse(samples[f].raw_count));
            groups = find_groups(&used, samples, n, nrows)?;
            let other = find_groups(&sorted, samples, n, nrows)?;
            if other.len() < groups.len() {
                groups = other;
            }
            let mut rng = Random::new(nrows as u32);
            for i in 0..groups.len().saturating_sub(1) {
                let j = rng.next_short((i + 1) as i32, groups.len() as i32)? as usize;
                groups.swap(i, j);
            }
        } else {
            for f in used {
                let mut group = reserve(1)?;
                group.push(f);
                groups.push(group);
            }
        }
        let mut inner = zeros(maps.len())?;
        inner.fill(usize::MAX);
        for (i, &f) in groups.iter().flatten().enumerate() {
            inner[f] = i;
        }
        Ok(Self { inner, groups })
    }

    pub(super) fn pack(
        &self,
        columns: &mut [Column],
        maps: &[BinMapper],
        nrows: usize,
        enabled: bool,
    ) -> Result<()> {
        if !enabled {
            return Ok(());
        }
        for features in &self.groups {
            if features.len() < 2 {
                continue;
            }
            let Some(num_bins) = features
                .iter()
                .try_fold(1u32, |n, &f| n.checked_add(maps[f].num_bins()))
            else {
                continue;
            };
            let mut baseline_bytes = 0usize;
            let mut nondefaults = 0usize;
            for &f in features {
                baseline_bytes = baseline_bytes.saturating_add(columns[f].storage_bytes());
                nondefaults = nondefaults
                    .saturating_add(nrows - columns[f].counts[maps[f].most_freq_bin() as usize]);
            }
            // Check costs BEFORE allocating any row-sized structure. Sparse shape
            // products must not cause allocations proportional to nrows*ncols.
            let packed_bytes = nrows.saturating_mul(CompactBins::width(num_bins));
            if packed_bytes
                .saturating_add(std::mem::size_of::<Packed>() + 2 * std::mem::size_of::<usize>())
                >= baseline_bytes
                || nondefaults > nrows
            {
                continue;
            }
            let mut entries = reserve(nondefaults)?;
            let mut offset = 1;
            for &f in features {
                let column = &columns[f];
                let default = maps[f].most_freq_bin();
                if let Some(rows) = &column.rows
                    && column.default == default
                {
                    for (i, &row) in rows.iter().enumerate() {
                        entries.push((row, offset + column.values.get(i)));
                    }
                } else {
                    for row in 0..nrows {
                        let bin = column.get(row);
                        if bin != default {
                            entries.push((row, offset + bin));
                        }
                    }
                }
                offset += maps[f].num_bins();
            }
            entries.sort_unstable_by_key(|e| e.0);
            if entries.windows(2).any(|w| w[0].0 == w[1].0) {
                continue;
            }
            let mut pos = 0;
            let packed = Arc::new(Packed {
                num_bins,
                values: CompactBins::from_iter(
                    (0..nrows).map(|row| {
                        if pos < entries.len() && entries[pos].0 == row {
                            let code = entries[pos].1;
                            pos += 1;
                            code
                        } else {
                            0
                        }
                    }),
                    nrows,
                    num_bins,
                )?,
            });
            offset = 1;
            for &f in features {
                columns[f].values = CompactBins::Bundle {
                    packed: Arc::clone(&packed),
                    offset,
                    num_bins: maps[f].num_bins(),
                    default: maps[f].most_freq_bin(),
                };
                columns[f].rows = None;
                columns[f].default = maps[f].most_freq_bin();
                offset += maps[f].num_bins();
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(super) struct Packed {
    pub(super) values: CompactBins,
    pub(super) num_bins: u32,
}
