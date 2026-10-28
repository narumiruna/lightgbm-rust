use lightgbm_rust::{Config, Dataset, Error, Model, Objective, Trainer};

fn small_config() -> Config {
    Config {
        min_data_in_leaf: 1,
        min_data_in_bin: 1,
        num_iterations: 3,
        num_leaves: 3,
        feature_pre_filter: false,
        ..Config::default()
    }
}

#[test]
fn initial_margins_are_external_and_validation_owns_its_margins() {
    let config = small_config();
    let data = Dataset::from_dense(&[0., 1., 2., 3.], 4, 1, vec![0., 0., 2., 2.], &config)
        .unwrap()
        .with_initial_scores(vec![0.5; 4], 1)
        .unwrap()
        .with_feature_names(vec!["feature".into()])
        .unwrap();
    let valid = data
        .validation_from_dense(&[0., 3.], 2, 1, vec![0., 2.])
        .unwrap()
        .with_initial_scores(vec![2., -1.], 1)
        .unwrap();
    let result = Trainer::new(config)
        .unwrap()
        .fit_with_validation(&data, &[("valid", &valid)])
        .unwrap();
    let raw = result
        .model
        .predict_raw_dense(&[0., 1., 2., 3.], 4, 1)
        .unwrap();
    for (row, score) in raw.iter().zip(&result.training_scores) {
        assert!((row[0] + 0.5 - score).abs() <= 1e-12);
    }
    let validation_raw = result.model.predict_raw_dense(&[0., 3.], 2, 1).unwrap();
    let expected_l2 =
        ((validation_raw[0][0] + 2.).powi(2) + (validation_raw[1][0] - 1. - 2.).powi(2)) / 2.;
    let metric = result
        .evaluation_history
        .last()
        .unwrap()
        .results
        .iter()
        .find(|r| r.dataset == "valid" && r.metric == "l2")
        .unwrap();
    assert!((metric.value - expected_l2).abs() <= 1e-12);
    let text = result.model.to_text().unwrap();
    assert!(text.contains("feature_names=feature"));
    assert!(text.contains("feature_infos=[0:3]"));
    assert_eq!(
        Model::from_text(&text)
            .unwrap()
            .predict_raw_dense(&[0., 3.], 2, 1)
            .unwrap(),
        validation_raw
    );
}

#[test]
fn class_major_initial_margins_and_shapes_are_checked() {
    let config = Config {
        objective: Objective::Multiclass,
        num_class: 3,
        ..small_config()
    };
    let x = [0., 1., 2., 3., 4., 5.];
    let data = Dataset::from_dense(&x, 6, 1, vec![0., 0., 1., 1., 2., 2.], &config).unwrap();
    let wrong = data.clone().with_initial_scores(vec![0.; 6], 1).unwrap();
    assert!(Trainer::new(config.clone()).unwrap().fit(&wrong).is_err());
    let wrong_valid = data
        .validation_from_dense(&[0., 2.], 2, 1, vec![0., 1.])
        .unwrap()
        .with_initial_scores(vec![0.; 2], 1)
        .unwrap();
    assert!(
        Trainer::new(config.clone())
            .unwrap()
            .fit_with_validation(&data, &[("valid", &wrong_valid)])
            .is_err()
    );
    let initial: Vec<f64> = (0..18).map(|i| i as f64 / 20.).collect();
    let data = data.with_initial_scores(initial.clone(), 3).unwrap();
    let result = Trainer::new(config).unwrap().fit(&data).unwrap();
    assert_eq!(result.training_scores.len(), 18);
    assert_eq!(result.model.trees().len() % 3, 0);
    let raw = result.model.predict_raw_dense(&x, 6, 1).unwrap();
    for (row, values) in raw.iter().enumerate() {
        for (class, &value) in values.iter().enumerate() {
            assert!(
                (result.training_scores[class * 6 + row] - initial[class * 6 + row] - value).abs()
                    <= 1e-12
            );
        }
    }
}

#[test]
fn no_split_round_keeps_exactly_one_complete_set_of_class_constants() {
    let config = Config {
        objective: Objective::Multiclass,
        num_class: 3,
        ..small_config()
    };
    let data = Dataset::from_dense(&[1.; 4], 4, 1, vec![0., 0., 1., 1.], &config).unwrap();
    let result = Trainer::new(config).unwrap().fit(&data).unwrap();
    assert_eq!(result.model.num_iterations(), 1);
    assert_eq!(result.model.trees().len(), 3);
    assert!(result.model.trees().iter().all(|t| t.num_leaves() == 1));
    assert_eq!(result.evaluation_history.len(), 1);
    let prediction = result.model.predict_raw_dense(&[1.; 4], 4, 1).unwrap();
    for (row, values) in prediction.iter().enumerate() {
        for (class, &value) in values.iter().enumerate() {
            assert_eq!(result.training_scores[class * 4 + row], value);
        }
    }
}

#[test]
fn unsplittable_external_margins_do_not_enter_the_model() {
    let config = small_config();
    let initial = vec![1., 2., 3., 4.];
    let data = Dataset::from_dense(&[1.; 4], 4, 1, vec![0., 1., 2., 3.], &config)
        .unwrap()
        .with_initial_scores(initial.clone(), 1)
        .unwrap();
    let result = Trainer::new(config).unwrap().fit(&data).unwrap();
    assert_eq!(result.training_scores, initial);
    assert_eq!(result.model.trees()[0].leaf_values(), [0.]);
}

#[test]
fn custom_objective_requires_a_gradient_callback() {
    let config = Config {
        objective: Objective::Custom,
        ..small_config()
    };
    let data = Dataset::from_dense(&[0., 1.], 2, 1, vec![0., 1.], &config).unwrap();
    assert!(matches!(
        Trainer::new(config).unwrap().fit(&data),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn linear_training_requires_raw_retention_in_the_dataset_fingerprint() {
    let config = small_config();
    let data = Dataset::from_dense(&[0., 1.], 2, 1, vec![0., 1.], &config).unwrap();
    let linear = Config {
        linear_tree: true,
        ..config
    };
    assert!(matches!(
        Trainer::new(linear).unwrap().fit(&data),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn training_rejects_class_score_resource_limit_before_allocation() {
    let config = Config {
        objective: Objective::Multiclass,
        num_class: 65_536,
        ..small_config()
    };
    let data = Dataset::from_dense(&vec![0.; 257], 257, 1, vec![0.; 257], &config).unwrap();
    assert!(matches!(
        Trainer::new(config).unwrap().fit(&data),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn new_dataset_construction_fields_remain_immutable() {
    let config = small_config();
    let data = Dataset::from_dense(&[0., 1., 2., 3.], 4, 1, vec![0., 0., 1., 1.], &config).unwrap();
    for changed in [
        Config {
            max_bin_by_feature: vec![5],
            ..config.clone()
        },
        Config {
            forced_bins: vec![vec![1.5]],
            ..config.clone()
        },
        Config {
            linear_tree: true,
            ..config
        },
    ] {
        assert!(Trainer::new(changed).unwrap().fit(&data).is_err());
    }
}
