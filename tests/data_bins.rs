use lightgbm_rust::{Config, Dataset, data::MissingType};

fn config() -> Config {
    Config {
        min_data_in_leaf: 1,
        min_data_in_bin: 1,
        feature_pre_filter: false,
        ..Config::default()
    }
}

// Source: bin.cpp always separates a zero bin, even when zero is not missing.
#[test]
fn zero_nan_and_disabled_missing_have_distinct_contracts() {
    let values = [-2.0, -1.0, 0.0, 1.0, 2.0, f64::NAN];
    for (use_missing, zero_as_missing, missing_type) in [
        (true, false, MissingType::NaN),
        (true, true, MissingType::Zero),
        (false, false, MissingType::None),
        (false, true, MissingType::None),
    ] {
        let c = Config {
            use_missing,
            zero_as_missing,
            ..config()
        };
        let d = Dataset::from_dense(&values, values.len(), 1, vec![0.0; values.len()], &c).unwrap();
        let m = d.get_mapper(0).unwrap();
        assert_eq!(m.missing_type(), missing_type);
        if missing_type == MissingType::NaN {
            assert_ne!(m.bin(f64::NAN), m.bin(0.0));
        } else {
            assert_eq!(m.bin(f64::NAN), m.bin(0.0));
        }
        assert_eq!(
            d.get_bin_counts(0).unwrap().iter().sum::<usize>(),
            values.len()
        );
        assert_eq!(
            d.get_bin_counts(0).unwrap()[m.bin(f64::NAN) as usize],
            if missing_type == MissingType::NaN {
                1
            } else {
                2
            }
        );
        assert!(m.thresholds().contains(&f64::INFINITY));
    }
}

#[test]
fn two_bin_zero_missing_is_disabled_like_upstream() {
    let c = Config {
        max_bin: 2,
        zero_as_missing: true,
        ..config()
    };
    let d = Dataset::from_dense(&[0.0, 1.0, 2.0], 3, 1, vec![0.0; 3], &c).unwrap();
    assert_eq!(d.get_mapper(0).unwrap().num_bins(), 2);
    assert_eq!(d.get_mapper(0).unwrap().missing_type(), MissingType::None);
}

#[test]
fn numerical_edge_values_do_not_panic_or_escape_bins() {
    let cases: &[&[f64]] = &[
        &[f64::NEG_INFINITY, f64::INFINITY, f64::NAN, 0.0],
        &[
            f64::MIN,
            -1.0,
            -f64::MIN_POSITIVE,
            -0.0,
            0.0,
            f64::MIN_POSITIVE,
            1.0,
            f64::MAX,
        ],
        &[f64::NAN, f64::NAN],
        &[0.0, 0.0],
    ];
    for &values in cases {
        for max_bin in 2..8 {
            for min_data_in_bin in 1..5 {
                for zero_as_missing in [false, true] {
                    let c = Config {
                        max_bin,
                        min_data_in_bin,
                        zero_as_missing,
                        ..config()
                    };
                    let d =
                        Dataset::from_dense(values, values.len(), 1, vec![0.0; values.len()], &c)
                            .unwrap();
                    let m = d.get_mapper(0).unwrap();
                    for row in 0..values.len() {
                        assert!(d.get_bin(row, 0).unwrap() < m.num_bins());
                    }
                    assert_eq!(
                        d.get_bin_counts(0).unwrap().iter().sum::<usize>(),
                        values.len()
                    );
                }
            }
        }
    }
}

#[test]
fn constants_keep_original_feature_positions_and_unsplittable_flags() {
    let c = Config {
        feature_pre_filter: true,
        ..config()
    };
    let d = Dataset::from_dense(
        &[5.0, 0.0, 5.0, 1.0, 5.0, 2.0, 5.0, 3.0],
        4,
        2,
        vec![0.0; 4],
        &c,
    )
    .unwrap();
    assert_eq!(d.ncols(), 2);
    assert!(d.get_mapper(0).unwrap().is_trivial());
    assert!(!d.get_mapper(1).unwrap().is_trivial());
    for feature in 0..d.ncols() {
        assert_eq!(
            d.get_bin_counts(feature).unwrap().iter().sum::<usize>(),
            d.nrows()
        );
    }
}

#[test]
fn sparse_implicit_entries_use_zero_mapping_not_assumed_bin_zero() {
    let c = config();
    let dense = Dataset::from_dense(&[-3.0, 0.0, 2.0, 0.0], 4, 1, vec![0.0; 4], &c).unwrap();
    let sparse = Dataset::from_csc(&[0, 2], &[2, 0], &[2.0, -3.0], 4, 1, vec![0.0; 4], &c).unwrap();
    assert_ne!(sparse.get_mapper(0).unwrap().default_bin(), 0);
    assert_eq!(
        sparse.get_bin_counts(0).unwrap(),
        dense.get_bin_counts(0).unwrap()
    );
    for row in 0..4 {
        assert_eq!(
            dense.get_bin(row, 0).unwrap(),
            sparse.get_bin(row, 0).unwrap()
        );
    }
    let subset = sparse.subset(&[1, 3, 0]).unwrap();
    assert_eq!(subset.get_bin_counts(0).unwrap().iter().sum::<usize>(), 3);
    assert_eq!(subset.get_bin(0, 0).unwrap(), sparse.get_bin(1, 0).unwrap());
}

#[test]
fn sampled_bins_remain_monotone_and_cover_every_input() {
    let values: Vec<_> = (0..800).map(|i| (i % 199) as f64 - 97.0).collect();
    for max_bin in [2, 3, 7, 255, 513] {
        let c = Config {
            max_bin,
            bin_construct_sample_cnt: 47,
            ..config()
        };
        let d = Dataset::from_dense(&values, values.len(), 1, vec![0.0; values.len()], &c).unwrap();
        let m = d.get_mapper(0).unwrap();
        assert!(m.num_bins() <= max_bin as u32);
        let mut last = 0;
        for raw in -200..200 {
            let bin = m.bin(raw as f64);
            assert!(bin >= last && bin < m.num_bins());
            last = bin;
        }
        assert_eq!(
            d.get_bin_counts(0).unwrap().iter().sum::<usize>(),
            values.len()
        );
    }
}
