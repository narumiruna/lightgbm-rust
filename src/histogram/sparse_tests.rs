use super::*;

// The pre-R18 sparse bitmap/entry scan, and the original row-ordered dense
// accumulation. Keep this independent of the adaptive intersection helpers.
fn legacy<const CONSTANT: bool, C: Count>(
    data: &Dataset,
    rows: &[usize],
    gradients: &[f32],
    hessians: &[f32],
    active: Option<&[bool]>,
) -> Vec<Vec<Stats<C>>> {
    let mut members = vec![false; data.nrows()];
    let (mut total_g, mut total_h) = (0.0, 0.0);
    for &row in rows {
        members[row] = true;
        total_g += f64::from(gradients[row]);
        if !CONSTANT {
            total_h += f64::from(hessians[row]);
        }
    }
    let mut histograms = Vec::new();
    for feature in 0..data.ncols() {
        let mut bins = empty_bins::<C>(data, feature).unwrap();
        if active.is_none_or(|mask| mask[feature]) {
            if let Some(entries) = data.sparse_bins(feature) {
                let default = data.mapper(feature).bin(0.0) as usize;
                bins[default] = Stats {
                    gradient: total_g,
                    hessian: if CONSTANT { rows.len() as f64 } else { total_h },
                    count: C::from_rows(rows.len()),
                };
                for (row, value) in entries {
                    if !members[row] {
                        continue;
                    }
                    let g = f64::from(gradients[row]);
                    let h = if CONSTANT {
                        1.0
                    } else {
                        f64::from(hessians[row])
                    };
                    bins[default].gradient -= g;
                    bins[default].hessian -= h;
                    bins[default].count.remove();
                    bins[value as usize].gradient += g;
                    bins[value as usize].hessian += h;
                    bins[value as usize].count.add();
                }
            } else {
                for &row in rows {
                    let bin = &mut bins[data.bin(row, feature) as usize];
                    bin.gradient += f64::from(gradients[row]);
                    bin.hessian += if CONSTANT {
                        1.0
                    } else {
                        f64::from(hessians[row])
                    };
                    bin.count.add();
                }
            }
        }
        if CONSTANT {
            for bin in &mut bins {
                bin.hessian *= f64::from(hessians[0]);
            }
        }
        histograms.push(bins);
    }
    histograms
}

fn assert_bits<C: Count + PartialEq + std::fmt::Debug>(
    actual: &[Vec<Stats<C>>],
    expected: &[Vec<Stats<C>>],
) {
    assert_eq!(actual.len(), expected.len());
    for (feature, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.len(), expected.len());
        for (bin, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                a.gradient.to_bits(),
                e.gradient.to_bits(),
                "g: {feature}/{bin}"
            );
            assert_eq!(
                a.hessian.to_bits(),
                e.hessian.to_bits(),
                "h: {feature}/{bin}"
            );
            assert_eq!(a.count, e.count, "count: {feature}/{bin}");
        }
    }
}

fn derivatives(n: usize) -> (Vec<f32>, Vec<f32>) {
    (
        (0..n)
            .map(|r| [1e20, 1.0, -1e20, 3.25, -0.125, 0.0, -7.0][r % 7])
            .collect(),
        (0..n)
            .map(|r| [0.375, 1e20, 0.125, 1.0, 2.0][r % 5])
            .collect(),
    )
}

fn check<const CONSTANT: bool>(data: &Dataset, selections: &[Vec<usize>], config: &Config) {
    let (g, h) = derivatives(data.nrows());
    let mixed: Vec<bool> = (0..data.ncols()).map(|f| f % 3 != 1).collect();
    let none = vec![false; data.ncols()];
    for active in [None, Some(mixed.as_slice()), Some(none.as_slice())] {
        for rows in selections {
            let counted = legacy::<CONSTANT, usize>(data, rows, &g, &h, active);
            let training = legacy::<CONSTANT, ()>(data, rows, &g, &h, active);
            for threads in [1, 4] {
                let executor = crate::parallel::Executor::new(threads).unwrap();
                for row_wise in [false, true] {
                    let config = Config {
                        force_row_wise: row_wise,
                        force_col_wise: !row_wise,
                        ..config.clone()
                    };
                    assert_bits(
                        &build_with_executor(
                            data, rows, &g, &h, CONSTANT, &config, &executor, active,
                        )
                        .unwrap(),
                        &counted,
                    );
                    assert_bits(
                        &build_training(data, rows, &g, &h, CONSTANT, &config, &executor, active)
                            .unwrap(),
                        &training,
                    );
                }
            }
        }
    }
}

fn layouts(values: &[f64], n: usize, p: usize, config: &Config) -> Vec<Dataset> {
    let mut rp = vec![0];
    let (mut ci, mut rv) = (Vec::new(), Vec::new());
    for row in values.chunks(p) {
        for (f, &v) in row.iter().enumerate() {
            if v != 0.0 {
                ci.push(f);
                rv.push(v);
            }
        }
        rp.push(ci.len());
    }
    let mut cp = vec![0];
    let (mut ri, mut cv) = (Vec::new(), Vec::new());
    for f in 0..p {
        for r in 0..n {
            let v = values[r * p + f];
            if v != 0.0 {
                ri.push(r);
                cv.push(v);
            }
        }
        cp.push(ri.len());
    }
    let datasets = vec![
        Dataset::from_dense(values, n, p, vec![0.0; n], config).unwrap(),
        Dataset::from_csr(&rp, &ci, &rv, n, p, vec![0.0; n], config).unwrap(),
        Dataset::from_csc(&cp, &ri, &cv, n, p, vec![0.0; n], config).unwrap(),
    ];
    for f in 0..p {
        for r in 0..n {
            assert_eq!(datasets[0].bin(r, f), datasets[1].bin(r, f));
            assert_eq!(datasets[0].bin(r, f), datasets[2].bin(r, f));
        }
    }
    datasets
}

#[test]
fn adaptive_sparse_histograms_match_legacy_bits_all_layouts_and_modes() {
    let (n, p) = (4096, 12);
    let values: Vec<f64> = (0..n * p)
        .map(|i| {
            let (r, f) = (i / p, i % p);
            match f {
                0 => 0.0, // No stored entries.
                1 => {
                    if r == 1 {
                        -3.0
                    } else {
                        0.0
                    }
                } // Very sparse column.
                2 => 5.0, // Constant nondefault feature.
                _ if r % 13 == 0 => f64::NAN,
                _ if (r + f) % 4 == 0 => 0.0,
                _ if f == 3 => (r % 9) as f64 - 1.0, // Categorical, including negative/missing.
                _ => (r % 37) as f64 - 21.0,         // Nonzero default bin.
            }
        })
        .collect();
    let mut selections = vec![
        (0..n).collect(),
        (0..n).rev().collect(),
        (0..n).step_by(2).collect(),
        vec![],
        vec![0, 1, 2, 13, 39, 4000],
        vec![4000, 13, 2, 1, 0, 39],
        vec![2, 0, 1, 2, 13, 1, 4000, 0],
        vec![13], // NaN stored entry.
        vec![4],  // Implicit/default entries in some features.
    ];
    let mut duplicate_root: Vec<usize> = (0..n).rev().collect();
    duplicate_root.extend_from_slice(&[0, 1, 1, 13]);
    selections.push(duplicate_root);
    for (use_missing, zero_as_missing) in [(false, false), (true, false), (true, true)] {
        let config = Config {
            enable_bundle: false,
            feature_pre_filter: false,
            min_data_in_bin: 1,
            min_data_in_leaf: 1,
            max_bin: 47,
            use_missing,
            zero_as_missing,
            categorical_features: vec![3],
            ..Config::default()
        };
        for data in layouts(&values, n, p, &config) {
            check::<false>(&data, &selections, &config);
            check::<true>(&data, &selections, &config);
        }
    }
}

#[test]
fn adaptive_sparse_histograms_cover_compact_widths() {
    for (n, expected_width) in [(97, 1), (400, 2), (65_540, 4)] {
        let config = Config {
            enable_bundle: false,
            feature_pre_filter: false,
            min_data_in_bin: 1,
            min_data_in_leaf: 1,
            max_bin: n + 1,
            bin_construct_sample_cnt: n,
            ..Config::default()
        };
        let values: Vec<f64> = (0..n).map(|r| r as f64 - 3.0).collect();
        let datasets = layouts(&values, n, 1, &config);
        for data in &datasets[1..] {
            let entries = data.sparse_bins(0).unwrap();
            assert_eq!(entries.size_hint(), (n - 1, Some(n - 1)));
            let bins = data.mapper(0).num_bins();
            assert_eq!(
                if bins <= 256 {
                    1
                } else if bins <= 65536 {
                    2
                } else {
                    4
                },
                expected_width
            );
            assert_ne!(data.mapper(0).bin(0.0), 0);
            let selections = vec![(0..n).collect(), vec![n - 1, 2, 0, 1, 2, 3], vec![]];
            check::<false>(data, &selections, &config);
            check::<true>(data, &selections, &config);
        }
    }
}

#[test]
fn indexed_sparse_children_match_legacy_under_parallel_execution() {
    let (n, p) = (32_768, 12);
    let config = Config {
        enable_bundle: false,
        feature_pre_filter: false,
        min_data_in_bin: 1,
        min_data_in_leaf: 1,
        ..Config::default()
    };
    let mut pointers = vec![0];
    let (mut indexes, mut values) = (Vec::new(), Vec::new());
    for f in 0..p {
        for r in (f..n).step_by(3) {
            indexes.push(r);
            values.push((r % 31) as f64 - 15.0);
        }
        pointers.push(indexes.len());
    }
    let data =
        Dataset::from_csc(&pointers, &indexes, &values, n, p, vec![0.0; n], &config).unwrap();
    let sorted: Vec<usize> = (0..1024).map(|r| r * 31).collect();
    let unsorted: Vec<usize> = (0..1024).map(|r| (r * 17 % 1024) * 31).collect();
    let mut duplicates = unsorted.clone();
    duplicates.extend_from_slice(&[0, 31]);
    let selections = vec![sorted, unsorted, duplicates];
    for rows in &selections {
        assert!(rows.len() >= 1024); // Existing executor parallel-work gate.
        assert!(sparse_indexed(rows.len(), n));
    }
    check::<false>(&data, &selections, &config);
    check::<true>(&data, &selections, &config);
}

#[test]
fn tiny_selections_scan_short_sparse_columns_without_a_bitmap() {
    let (n, p, k) = (131_072, 256, 512);
    assert!(sparse_indexed(k, n));
    assert!(sparse_short_scan(k, 1));
    assert!(sparse_short_scan(k, k));
    assert!(!sparse_short_scan(k, k + 1));
    let config = Config {
        enable_bundle: false,
        feature_pre_filter: false,
        min_data_in_bin: 1,
        min_data_in_leaf: 1,
        ..Config::default()
    };
    let pointers: Vec<usize> = (0..=p).collect();
    let indexes: Vec<usize> = (0..p).collect();
    let values = vec![1.0; p];
    let data =
        Dataset::from_csc(&pointers, &indexes, &values, n, p, vec![0.0; n], &config).unwrap();
    for feature in 0..p {
        let entries = data.sparse_bins(feature).unwrap();
        assert_eq!(entries.size_hint(), (1, Some(1)));
        assert!(sparse_short_scan(k, entries.size_hint().0));
    }
    let rows: Vec<usize> = (0..p).chain(n - (k - p)..n).collect();
    assert_eq!(rows.len(), k);
    let selections = vec![rows];
    check::<false>(&data, &selections, &config);
    check::<true>(&data, &selections, &config);
    assert_eq!(
        sparse_bitmap(n, !sparse_indexed(k, n)).unwrap().capacity(),
        0
    );
}

#[test]
fn sparse_strategy_bounds_lookup_work_and_avoids_tiny_child_bitmap() {
    for entries in [0, 1, 2, 7, 1024, 1_000_000, usize::MAX] {
        assert!(sparse_indexed(0, entries));
        for selected in [1, 3, 16, 1024, 1_000_000, usize::MAX] {
            let indexed = sparse_indexed(selected, entries);
            if entries != 0 {
                let comparisons = (usize::BITS - entries.leading_zeros()) as u128 + 1;
                assert_eq!(indexed, selected as u128 * comparisons <= entries as u128);
            } else {
                assert!(indexed);
            }
        }
    }
    let n = 1_000_000;
    let scan = !sparse_indexed(3, n);
    assert!(!scan);
    let tiny = sparse_bitmap(n, scan).unwrap();
    assert_eq!(tiny.capacity(), 0);
    assert!(tiny.is_empty());
    assert!(!sparse_indexed(n, n));
    let root = sparse_bitmap(n, true).unwrap();
    assert_eq!(root.len(), n);
    assert!(root.iter().all(|member| !member));
}
