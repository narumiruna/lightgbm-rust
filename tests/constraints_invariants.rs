use lightgbm_rust::{Config, Dataset, MonotoneMethod, Trainer};

#[test]
fn all_methods_preserve_monotonicity_on_unseen_grid_points() {
    for method in [
        MonotoneMethod::Basic,
        MonotoneMethod::Intermediate,
        MonotoneMethod::Advanced,
    ] {
        let config = Config {
            monotone_constraints: vec![1, -1],
            monotone_constraints_method: method,
            min_data_in_leaf: 2,
            min_data_in_bin: 1,
            feature_pre_filter: false,
            num_leaves: 12,
            num_iterations: 6,
            path_smooth: 2.,
            ..Config::default()
        };
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..12 {
            for j in 0..12 {
                x.extend([i as f64, j as f64]);
                y.push((i as f64 * 0.3 - j as f64 * 0.2 + (i as f64 + j as f64).sin()) as f32);
            }
        }
        let data = Dataset::from_dense(&x, 144, 2, y, &config).unwrap();
        let model = Trainer::new(config).unwrap().fit(&data).unwrap().model;
        for i in 0..23 {
            for j in 0..23 {
                let a = i as f64 * 0.5;
                let b = j as f64 * 0.5;
                let p = model.predict_raw_row(&[a, b]).unwrap()[0];
                assert!(p <= model.predict_raw_row(&[a + 0.25, b]).unwrap()[0] + 1e-12);
                assert!(p >= model.predict_raw_row(&[a, b + 0.25]).unwrap()[0] - 1e-12);
            }
        }
    }
}
#[test]
fn infeasible_forced_split_falls_back_and_malformed_feature_is_checked() {
    let config = Config {
        min_data_in_leaf: 1,
        min_data_in_bin: 1,
        feature_pre_filter: false,
        num_iterations: 2,
        ..Config::default()
    };
    let x = [0., 1., 2., 3., 4., 5.];
    let data = Dataset::from_dense(&x, 6, 1, vec![0., 0., 0., 1., 1., 1.], &config).unwrap();
    let expected = Trainer::new(config.clone()).unwrap().fit(&data).unwrap();
    let mut forced = config.clone();
    forced.forced_splits = Some(serde_json::json!({"feature":0,"threshold":1e20}));
    let actual = Trainer::new(forced).unwrap().fit(&data).unwrap();
    assert_eq!(actual.training_scores, expected.training_scores);
    let mut invalid = config;
    invalid.forced_splits = Some(serde_json::json!({"feature":1,"threshold":0.0}));
    assert!(Trainer::new(invalid).unwrap().fit(&data).is_err());
}
#[test]
fn quantization_zero_derivatives_and_lazy_cost_saturation_are_finite() {
    let config = Config {
        min_data_in_leaf: 1,
        min_data_in_bin: 1,
        feature_pre_filter: false,
        use_quantized_grad: true,
        quant_train_renew_leaf: true,
        cegb_penalty_feature_lazy: vec![1.],
        cegb_penalty_feature_coupled: vec![1.],
        ..Config::default()
    };
    let data = Dataset::from_dense(&[0., 1., 2., 3.], 4, 1, vec![2.; 4], &config).unwrap();
    let result = Trainer::new(config).unwrap().fit(&data).unwrap();
    assert!(result.training_scores.iter().all(|&s| s == 2.));
}
