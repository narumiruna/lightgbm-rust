use lightgbm_rust::{Config, Dataset, Error};

#[test]
fn empty_queries_retain_logical_entries_and_shared_metadata() {
    let c = Config {
        linear_tree: true,
        min_data_in_bin: 1,
        min_data_in_leaf: 1,
        ..Config::default()
    };
    let d = Dataset::from_dense(&[0., 1., 2., 3.], 4, 1, vec![0., 1., 1., 0.], &c)
        .unwrap()
        .with_groups(vec![0, 0, 2, 0, 2, 0, 0])
        .unwrap()
        .with_positions(vec![9, -7, 9, -7])
        .unwrap()
        .with_initial_scores(vec![0.1, 0.2, 0.3, 0.4], 1)
        .unwrap()
        .with_feature_names(vec!["signal".into()])
        .unwrap();
    let subset = d.subset(&[0, 1, 2, 3]).unwrap();
    assert_eq!(subset.groups(), d.groups());
    assert_eq!(subset.positions(), d.positions());
    assert_eq!(subset.initial_scores(), d.initial_scores());
    assert_eq!(subset.labels(), d.labels());
    assert_eq!(subset.feature_names(), d.feature_names());
    assert!(std::ptr::eq(
        subset.get_mapper(0).unwrap(),
        d.get_mapper(0).unwrap()
    ));
    for row in 0..4 {
        assert_eq!(
            subset.raw_value(row, 0).unwrap(),
            d.raw_value(row, 0).unwrap()
        );
        assert_eq!(subset.get_bin(row, 0).unwrap(), d.get_bin(row, 0).unwrap());
    }
    for rows in [&[0, 1][..], &[2, 3][..], &[2, 3, 0, 1][..]] {
        assert!(matches!(d.subset(rows), Err(Error::Unsupported(_))));
    }
    let valid = d
        .validation_from_dense(&[10., 11.], 2, 1, vec![0., 1.])
        .unwrap()
        .with_groups(vec![0, 1, 0, 1, 0])
        .unwrap();
    assert_eq!(valid.groups(), Some(&[0, 1, 0, 1, 0][..]));
    assert!(valid.positions().is_none());
    assert!(valid.initial_scores().is_none());
    assert!(std::ptr::eq(
        valid.get_mapper(0).unwrap(),
        d.get_mapper(0).unwrap()
    ));
    assert_eq!(d.groups(), Some(&[0, 0, 2, 0, 2, 0, 0][..]));
    for groups in [
        vec![],
        vec![0, 0, 0],
        vec![0, 3],
        vec![0, 5],
        vec![usize::MAX, 5],
    ] {
        assert!(d.clone().with_groups(groups).is_err());
    }
}
