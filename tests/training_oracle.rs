use lightgbm_rust::{Config, Dataset, Model, Objective, Trainer};
use serde_json::Value;

fn config(case: &Value) -> Config {
    let mut pairs = Vec::new();
    for (key, value) in case["params"].as_object().unwrap() {
        let value = match value {
            Value::String(s) => s.clone(),
            Value::Array(values) => values
                .iter()
                .map(Value::to_string)
                .collect::<Vec<_>>()
                .join(","),
            _ => value.to_string(),
        };
        pairs.push((key.clone(), value));
    }
    pairs.push((
        "num_iterations".into(),
        case["training"]["num_boost_round"].to_string(),
    ));
    Config::from_pairs(pairs).unwrap_or_else(|e| panic!("{} config: {e}", case["name"]))
}

fn rows(value: &Value) -> (Vec<f64>, usize, usize) {
    let rows = value.as_array().unwrap();
    let cols = rows[0].as_array().unwrap().len();
    let values = rows
        .iter()
        .flat_map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap_or(f64::NAN))
        })
        .collect();
    (values, rows.len(), cols)
}

#[test]
fn serial_gbdt_matches_pinned_trees_and_predictions() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/prediction.json")).unwrap();
    let required = [
        "numeric_l2",
        "numeric_weighted",
        "binary",
        "missing_nan",
        "missing_zero",
        "single_leaf",
        "missing_only",
        "missing_direction_tie",
        "categorical",
        "multiclass",
        "multiclassova",
        "regression_sqrt",
        "binary_sigmoid",
        "poisson",
        "gamma",
        "tweedie",
        "cross_entropy",
        "cross_entropy_lambda",
        "missing_none",
        "regression_extended",
        "regression_l1_extended",
        "quantile_extended",
        "huber_extended",
        "fair_extended",
        "poisson_extended",
        "binary_extended",
        "lambdarank_extended",
        "rank_xendcg_extended",
        "multiclass_extended",
        "multiclass_absent_extended",
        "multiclassova_extended",
        "multiclassova_absent_extended",
        "cross_entropy_extended",
        "cross_entropy_lambda_extended",
        "mape_extended",
        "gamma_extended",
        "tweedie_extended",
        "categorical_onehot",
        "categorical_onehot_weighted",
        "categorical_partition",
        "categorical_partition_weighted",
    ];
    let mut seen = std::collections::BTreeSet::new();
    let regressions: Value =
        serde_json::from_str(include_str!("fixtures/prediction_regressions.json")).unwrap();
    let extended: Value =
        serde_json::from_str(include_str!("fixtures/prediction_extended.json")).unwrap();
    for case in fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .chain(regressions["cases"].as_array().unwrap())
        .chain(extended["cases"].as_array().unwrap())
    {
        if !required.contains(&case["name"].as_str().unwrap()) {
            continue;
        }
        assert!(
            seen.insert(case["name"].as_str().unwrap()),
            "duplicate training fixture"
        );
        let config = config(case);
        let (x, n, p) = rows(&case["training"]["rows"]);
        let labels: Vec<f32> = serde_json::from_value(case["training"]["labels"].clone()).unwrap();
        let mut data = Dataset::from_dense(&x, n, p, labels, &config).unwrap();
        if let Some(w) = case["training"].get("weights") {
            data = data
                .with_weights(serde_json::from_value(w.clone()).unwrap())
                .unwrap();
        }
        if let Some(groups) = case["training"].get("groups") {
            data = data
                .with_groups(serde_json::from_value(groups.clone()).unwrap())
                .unwrap();
        }
        if let Some(positions) = case["training"].get("positions") {
            data = data
                .with_positions(serde_json::from_value(positions.clone()).unwrap())
                .unwrap();
        }
        let result = Trainer::new(config)
            .unwrap()
            .fit(&data)
            .unwrap_or_else(|e| panic!("{} fit: {e}", case["name"]));
        let native = Model::from_text(
            &std::fs::read_to_string(format!(
                "{}/tests/fixtures/{}",
                env!("CARGO_MANIFEST_DIR"),
                case["model_file"].as_str().unwrap()
            ))
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            result.model.num_iterations(),
            native.num_iterations(),
            "{} iterations",
            case["name"]
        );
        for (index, (got, expected)) in result.model.trees().iter().zip(native.trees()).enumerate()
        {
            assert_eq!(
                got.split_features(),
                expected.split_features(),
                "{} tree{index} features: {got:?}",
                case["name"]
            );
            assert_eq!(
                got.thresholds(),
                expected.thresholds(),
                "{} tree{index} thresholds",
                case["name"]
            );
            assert_eq!(
                got.left_children(),
                expected.left_children(),
                "{} tree{index} left",
                case["name"]
            );
            assert_eq!(
                got.right_children(),
                expected.right_children(),
                "{} tree{index} right",
                case["name"]
            );
            assert_eq!(
                got.decision_types(),
                expected.decision_types(),
                "{} tree{index} missing directions",
                case["name"]
            );
            assert_eq!(got.leaf_counts(), expected.leaf_counts());
            assert_eq!(
                got.cat_boundaries(),
                expected.cat_boundaries(),
                "{} tree{index} category boundaries",
                case["name"]
            );
            assert_eq!(
                got.cat_threshold(),
                expected.cat_threshold(),
                "{} tree{index} category bitsets",
                case["name"]
            );
            for (&actual, &expected) in got.leaf_values().iter().zip(expected.leaf_values()) {
                assert!(
                    (actual - expected).abs() <= 1e-12 + 1e-10 * expected.abs(),
                    "{} tree{index} leaf {actual} != {expected}",
                    case["name"]
                );
            }
        }
        let (probes, n, p) = rows(&case["rows"]);
        let predicted = result.model.predict_raw_dense(&probes, n, p).unwrap();
        for (row, expected) in predicted.iter().zip(case["raw"].as_array().unwrap()) {
            assert_eq!(row.len(), expected.as_array().unwrap().len());
            for (&actual, expected) in row.iter().zip(expected.as_array().unwrap()) {
                let expected = expected.as_f64().unwrap();
                assert!(
                    (actual - expected).abs() <= 1e-12 + 1e-10 * expected.abs(),
                    "{} prediction {actual} != {expected}",
                    case["name"]
                );
            }
        }
        let transformed = result.model.predict_dense(&probes, n, p).unwrap();
        for (row, expected) in transformed
            .iter()
            .zip(case["transformed"].as_array().unwrap())
        {
            assert_eq!(row.len(), expected.as_array().unwrap().len());
            for (&actual, expected) in row.iter().zip(expected.as_array().unwrap()) {
                let expected = expected.as_f64().unwrap();
                assert!(
                    (actual - expected).abs() <= 1e-12 + 1e-10 * expected.abs(),
                    "{} transformed {actual} != {expected}",
                    case["name"]
                );
            }
        }
        let expected_leaves: Vec<Vec<usize>> =
            serde_json::from_value(case["leaves"].clone()).unwrap();
        assert_eq!(
            result.model.predict_leaf_dense(&probes, n, p).unwrap(),
            expected_leaves,
            "{} leaf prediction",
            case["name"]
        );
        let reloaded = Model::from_text(&result.model.to_text().unwrap()).unwrap();
        assert_eq!(
            reloaded
                .predict_raw_dense(&x, data.nrows(), data.ncols())
                .unwrap()
                .iter()
                .fold(
                    vec![Vec::new(); reloaded.num_classes()],
                    |mut classes, row| {
                        for (class, &score) in classes.iter_mut().zip(row) {
                            class.push(score);
                        }
                        classes
                    }
                )
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
            result.training_scores
        );
    }
    assert_eq!(
        seen,
        required.into_iter().collect(),
        "required training fixture is missing"
    );
}

#[test]
fn trainer_rejects_changed_dataset_construction_settings() {
    let config = Config {
        min_data_in_leaf: 1,
        min_data_in_bin: 1,
        ..Config::default()
    };
    let dataset = Dataset::from_dense(
        &[0.0, 1.0, 2.0, 3.0],
        4,
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        &config,
    )
    .unwrap();
    let mut mismatches = Vec::new();
    let mut c = config.clone();
    c.categorical_features = vec![0];
    mismatches.push(c);
    let mut c = config.clone();
    c.use_missing = false;
    mismatches.push(c);
    let mut c = config.clone();
    c.zero_as_missing = true;
    mismatches.push(c);
    let mut c = config.clone();
    c.max_bin = 7;
    mismatches.push(c);
    let mut c = config.clone();
    c.min_data_in_bin = 2;
    mismatches.push(c);
    let mut c = config.clone();
    c.data_random_seed = 42;
    mismatches.push(c);
    let mut c = config.clone();
    c.feature_pre_filter = false;
    mismatches.push(c);
    for mismatch in mismatches {
        assert!(Trainer::new(mismatch).unwrap().fit(&dataset).is_err());
    }
    // Objective and learning parameters do not change the dataset's bin maps.
    let binary = Config {
        objective: Objective::Binary,
        learning_rate: 0.2,
        ..config
    };
    Trainer::new(binary).unwrap().fit(&dataset).unwrap();
}

#[test]
fn trainer_checks_validation_bins_and_accepts_categorical_learning() {
    let config = Config {
        min_data_in_leaf: 1,
        min_data_in_bin: 1,
        ..Config::default()
    };
    let data = Dataset::from_dense(
        &[0.0, 1.0, 2.0, 3.0],
        4,
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        &config,
    )
    .unwrap();
    let other = Dataset::from_dense(
        &[0.0, 1.0, 2.0, 3.0],
        4,
        1,
        vec![0.0, 0.0, 1.0, 1.0],
        &config,
    )
    .unwrap();
    let trainer = Trainer::new(config.clone()).unwrap();
    assert!(
        trainer
            .fit_with_validation(&data, &[("other", &other)])
            .is_err()
    );
    let validation = data
        .validation_from_dense(&[1.0, 3.0], 2, 1, vec![0.0, 1.0])
        .unwrap();
    let result = trainer
        .fit_with_validation(&data, &[("valid", &validation)])
        .unwrap();
    assert!(!result.evaluation_history.is_empty());
    assert!(
        result.evaluation_history[0]
            .results
            .iter()
            .any(|r| r.dataset == "valid")
    );
    let cat = Config {
        categorical_features: vec![0],
        ..config
    };
    let categorical =
        Dataset::from_dense(&[0.0, 1.0, 2.0, 3.0], 4, 1, vec![0.0, 0.0, 1.0, 1.0], &cat).unwrap();
    assert!(Trainer::new(cat).unwrap().fit(&categorical).is_ok());
}
