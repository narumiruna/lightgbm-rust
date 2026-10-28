use lightgbm_rust::{Config, Dataset, Trainer};

fn config() -> Config {
    Config {
        min_data_in_leaf: 1,
        min_data_in_bin: 1,
        num_iterations: 2,
        ..Config::default()
    }
}

#[test]
fn exclusive_columns_use_less_real_storage_dense_and_csr() {
    let (n, p) = (1024, 16);
    let x: Vec<_> = (0..n)
        .flat_map(|r| (0..p).map(move |f| f64::from(r % p == f)))
        .collect();
    let rows: Vec<_> = (0..=n).collect();
    let indexes: Vec<_> = (0..n).map(|r| r % p).collect();
    let values = vec![1.; n];
    for enabled in [true, false] {
        let cfg = Config {
            enable_bundle: enabled,
            ..config()
        };
        let dense = Dataset::from_dense(&x, n, p, vec![0.; n], &cfg).unwrap();
        let sparse = Dataset::from_csr(&rows, &indexes, &values, n, p, vec![0.; n], &cfg).unwrap();
        if enabled {
            assert!(dense.bin_storage_bytes() < n * p / 4);
            assert!(sparse.bin_storage_bytes() < n * 2);
            let subset = sparse.subset(&(0..n).rev().collect::<Vec<_>>()).unwrap();
            assert_eq!(subset.bin_storage_bytes(), sparse.bin_storage_bytes());
            for r in 0..n {
                for f in 0..p {
                    assert_eq!(
                        subset.get_bin(r, f).unwrap(),
                        sparse.get_bin(n - 1 - r, f).unwrap()
                    );
                }
            }
        } else {
            assert_eq!(dense.bin_storage_bytes(), n * p);
            assert_eq!(
                sparse.bin_storage_bytes(),
                n * (std::mem::size_of::<usize>() + 1)
            );
        }
    }
}

#[test]
fn overlapping_dense_columns_do_not_increase_bin_storage() {
    let (n, p) = (200, 8);
    let x: Vec<_> = (0..n)
        .flat_map(|i| (0..p).map(move |j| ((i + j) % 5) as f64))
        .collect();
    let a = Dataset::from_dense(&x, n, p, vec![0.; n], &config()).unwrap();
    let b = Dataset::from_dense(
        &x,
        n,
        p,
        vec![0.; n],
        &Config {
            enable_bundle: false,
            ..config()
        },
    )
    .unwrap();
    assert_eq!(a.bin_storage_bytes(), b.bin_storage_bytes());
}

#[test]
fn validation_conflicts_are_lossless_and_never_change_training_maps() {
    let (n, p) = (256, 8);
    let x: Vec<_> = (0..n)
        .flat_map(|i| (0..p).map(move |j| f64::from(i % p == j)))
        .collect();
    let train = Dataset::from_dense(&x, n, p, vec![0.; n], &config())
        .unwrap()
        .with_feature_names((0..p).map(|i| format!("source_{i}")).collect())
        .unwrap();
    let mut validation = vec![1.; n * p];
    validation[0] = f64::NAN;
    let valid = train
        .validation_from_dense(&validation, n, p, vec![0.; n])
        .unwrap();
    assert_eq!(train.feature_names(), valid.feature_names());
    for r in 0..n {
        for f in 0..p {
            assert_eq!(
                valid.get_bin(r, f).unwrap(),
                train.get_mapper(f).unwrap().bin(validation[r * p + f])
            );
            assert_eq!(
                train.get_bin(r, f).unwrap(),
                train.get_mapper(f).unwrap().bin(x[r * p + f])
            );
        }
    }
    let subset = valid.subset(&[2, 0, 2, 1]).unwrap();
    for (i, r) in [2, 0, 2, 1].into_iter().enumerate() {
        for f in 0..p {
            assert_eq!(subset.get_bin(i, f).unwrap(), valid.get_bin(r, f).unwrap());
        }
    }
}

#[test]
fn sparse_huge_shape_does_not_allocate_a_feature_row_matrix() {
    // 409.6 million logical cells, two explicit entries. An unconditional dense
    // bin matrix or one row bitmap per feature exceeds this test's input size.
    let (n, p) = (100_000, 4096);
    let mut pointers = vec![2; p + 1];
    pointers[0] = 0;
    pointers[1] = 1;
    let d = Dataset::from_csc(
        &pointers,
        &[1, 99_999],
        &[1., 2.],
        n,
        p,
        vec![0.; n],
        &config(),
    )
    .unwrap();
    assert_eq!(
        d.bin_storage_bytes(),
        2 * (std::mem::size_of::<usize>() + 1)
    );
    assert_eq!(d.get_bin(1, 0).unwrap(), d.get_mapper(0).unwrap().bin(1.));
    assert_eq!(
        d.get_bin(99_999, 1).unwrap(),
        d.get_mapper(1).unwrap().bin(2.)
    );
    assert!(Dataset::from_csc(&[0, usize::MAX], &[], &[], n, 1, vec![0.; n], &config()).is_err());
}

#[test]
fn pack_decode_property_missing_categories_defaults_and_subsets() {
    let mut state = 19u32;
    for seed in 0..24 {
        let (n, p) = (160, 8);
        let mut x = vec![0.; n * p];
        for r in 0..n {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let f = state as usize % p;
            x[r * p + f] = match (state >> 8) % 5 {
                0 => f64::NAN,
                1 => -1.,
                2 => 2.,
                _ => 1.,
            };
        }
        let cfg = Config {
            zero_as_missing: seed % 2 == 0,
            categorical_features: if seed % 3 == 0 { vec![1, 3, 5] } else { vec![] },
            ..config()
        };
        let y: Vec<_> = (0..n).map(|r| (r % 7) as f32).collect();
        let a = Dataset::from_dense(&x, n, p, y.clone(), &cfg).unwrap();
        let cfg_off = Config {
            enable_bundle: false,
            ..cfg.clone()
        };
        let b = Dataset::from_dense(&x, n, p, y, &cfg_off).unwrap();
        for r in 0..n {
            for f in 0..p {
                assert_eq!(a.get_bin(r, f).unwrap(), b.get_bin(r, f).unwrap());
            }
        }
        for f in 0..p {
            assert_eq!(a.get_bin_counts(f).unwrap(), b.get_bin_counts(f).unwrap());
        }
        let rows: Vec<_> = (0..n).step_by(3).rev().collect();
        let subset = a.subset(&rows).unwrap();
        for (i, &r) in rows.iter().enumerate() {
            for f in 0..p {
                assert_eq!(subset.get_bin(i, f).unwrap(), b.get_bin(r, f).unwrap());
            }
        }
        let on = Trainer::new(cfg).unwrap().fit(&a).unwrap().model;
        let off = Trainer::new(cfg_off).unwrap().fit(&b).unwrap().model;
        assert_eq!(
            on.predict_raw_dense(&x, n, p).unwrap(),
            off.predict_raw_dense(&x, n, p).unwrap()
        );
    }
}
