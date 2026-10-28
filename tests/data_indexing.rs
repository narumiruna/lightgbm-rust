use lightgbm_rust::{Config, Dataset, Error};

fn config() -> Config {
    Config {
        min_data_in_leaf: 1,
        min_data_in_bin: 1,
        feature_pre_filter: false,
        ..Config::default()
    }
}

#[test]
fn public_dataset_accessors_check_all_row_and_feature_endpoints() {
    let c = config();
    let dense = Dataset::from_dense(&[0., 1., 2., 3., 4., 5.], 3, 2, vec![0.; 3], &c).unwrap();
    let csr = Dataset::from_csr(
        &[0, 1, 2, 3],
        &[1, 0, 1],
        &[1., 2., 5.],
        3,
        2,
        vec![0.; 3],
        &c,
    )
    .unwrap();
    let csc =
        Dataset::from_csc(&[0, 1, 3], &[1, 0, 2], &[2., 1., 5.], 3, 2, vec![0.; 3], &c).unwrap();
    let linear = Config {
        linear_tree: true,
        ..c.clone()
    };
    let raw = Dataset::from_dense(&[0., 1., 2., 3., 4., 5.], 3, 2, vec![0.; 3], &linear).unwrap();
    let valid = dense
        .validation_from_dense(&[0., 1., 2., 3., 4., 5.], 3, 2, vec![0.; 3])
        .unwrap();
    let subset = dense.subset(&[2, 0, 2]).unwrap();
    for d in [&dense, &csr, &csc, &raw, &valid, &subset] {
        for row in [0, d.nrows() - 1] {
            for feature in [0, d.ncols() - 1] {
                let mapper = d.get_mapper(feature).unwrap();
                assert!(d.get_bin(row, feature).unwrap() < mapper.num_bins());
                let counts = d.get_bin_counts(feature).unwrap();
                assert_eq!(counts.len(), mapper.num_bins() as usize);
                assert_eq!(counts.iter().sum::<usize>(), d.nrows());
            }
        }
        for row in [d.nrows(), d.nrows() + 1, usize::MAX] {
            for feature in [0, d.ncols(), usize::MAX] {
                assert!(matches!(
                    d.get_bin(row, feature),
                    Err(Error::InvalidInput(_))
                ));
                assert!(matches!(
                    d.raw_value(row, feature),
                    Err(Error::InvalidInput(_))
                ));
            }
        }
        for feature in [d.ncols(), d.ncols() + 1, usize::MAX] {
            assert!(matches!(d.get_mapper(feature), Err(Error::InvalidInput(_))));
            assert!(matches!(
                d.get_bin_counts(feature),
                Err(Error::InvalidInput(_))
            ));
            assert!(matches!(d.get_bin(0, feature), Err(Error::InvalidInput(_))));
            assert!(matches!(
                d.raw_value(0, feature),
                Err(Error::InvalidInput(_))
            ));
        }
    }
    assert_eq!(raw.raw_value(0, 0).unwrap(), 0.);
    assert_eq!(raw.raw_value(2, 1).unwrap(), 5.);
    assert!(dense.raw_value(0, 0).is_err());
    let categorical = Config {
        categorical_features: vec![1],
        ..linear
    };
    let d =
        Dataset::from_dense(&[0., 1., 2., 3., 4., 5.], 3, 2, vec![0.; 3], &categorical).unwrap();
    assert!(d.raw_value(0, 1).is_err());
}

#[test]
fn zero_and_overflow_shapes_fail_before_storage_or_index_access() {
    let c = config();
    for (n, p) in [(0, 0), (0, 1), (1, 0), (usize::MAX, 2), (2, usize::MAX)] {
        assert!(Dataset::from_dense(&[], n, p, vec![], &c).is_err());
        assert!(Dataset::from_csr(&[0], &[], &[], n, p, vec![], &c).is_err());
        assert!(Dataset::from_csc(&[0], &[], &[], n, p, vec![], &c).is_err());
    }
}
