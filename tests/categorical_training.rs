use lightgbm_rust::{Config, Dataset, Error, Model, Trainer};

fn config() -> Config {
    Config {
        categorical_features: vec![0],
        max_cat_to_onehot: 8,
        cat_smooth: 0.,
        cat_l2: 0.,
        min_data_per_group: 1,
        min_data_in_leaf: 1,
        min_data_in_bin: 1,
        feature_pre_filter: false,
        num_iterations: 3,
        num_leaves: 3,
        ..Config::default()
    }
}

#[test]
fn raw_category_bitsets_and_validation_routing_match_serialized_prediction() {
    let config = config();
    let x = [40., 40., 2., 2., 7., 7., -1., f64::NAN];
    let data =
        Dataset::from_dense(&x, 8, 1, vec![8., 8., -2., -2., 1., 1., 0., 0.], &config).unwrap();
    let probes = [40., 2., 7., -1., f64::NAN, 100., 0.];
    let valid = data
        .validation_from_dense(&probes, 7, 1, vec![0.; 7])
        .unwrap();
    let result = Trainer::new(config)
        .unwrap()
        .fit_with_validation(&data, &[("valid", &valid)])
        .unwrap();
    assert!(
        result
            .model
            .trees()
            .iter()
            .any(|tree| !tree.cat_threshold().is_empty())
    );
    let roundtrip = Model::from_text(&result.model.to_text().unwrap()).unwrap();
    let raw = roundtrip.predict_raw_dense(&x, 8, 1).unwrap();
    assert_eq!(
        raw.iter().map(|r| r[0]).collect::<Vec<_>>(),
        result.training_scores
    );
    let predicted = roundtrip.predict_raw_dense(&probes, 7, 1).unwrap();
    assert_eq!(predicted[3], predicted[4]);
    assert_eq!(predicted[4], predicted[5]);
    let l2 = predicted.iter().map(|r| r[0] * r[0]).sum::<f64>() / 7.;
    let valid_l2 = result
        .evaluation_history
        .last()
        .unwrap()
        .results
        .iter()
        .find(|r| r.dataset == "valid" && r.metric == "l2")
        .unwrap()
        .value;
    assert!((l2 - valid_l2).abs() <= 1e-12);
}

#[test]
fn huge_sparse_raw_category_code_returns_unsupported_before_bitset_allocation() {
    let config = config();
    let data = Dataset::from_dense(
        &[
            2_000_000_000.,
            2_000_000_000.,
            1_000_000_000.,
            1_000_000_000.,
        ],
        4,
        1,
        vec![1., 1., -1., -1.],
        &config,
    )
    .unwrap();
    let result = Trainer::new(config).unwrap().fit(&data);
    assert!(
        matches!(result, Err(Error::Unsupported(ref message)) if message.contains("1,000,000-word"))
    );
}

#[test]
fn dense_and_sparse_categorical_histograms_keep_sorted_partition_rows() {
    let config = Config {
        max_cat_to_onehot: 2,
        ..config()
    };
    let x = [0., 1., 2., 3., 4., 5., 0., 1., 2., 3., 4., 5.];
    let labels = vec![0., 0., 1., 1., 3., 3., 0., 0., 1., 1., 3., 3.];
    let dense = Dataset::from_dense(&x, 12, 1, labels.clone(), &config).unwrap();
    let ptr: Vec<usize> = (0..=12).collect();
    let csr = Dataset::from_csr(&ptr, &[0; 12], &x, 12, 1, labels, &config).unwrap();
    let trainer = Trainer::new(config).unwrap();
    let a = trainer.fit(&dense).unwrap();
    let b = trainer.fit(&csr).unwrap();
    assert_eq!(a.model.to_text().unwrap(), b.model.to_text().unwrap());
    assert_eq!(a.training_scores, b.training_scores);
}
