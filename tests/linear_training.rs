//! Native linear-leaf evidence. These tests use saved models, not a native runtime.
use lightgbm_rust::{Config, Dataset, Model, Trainer, Tree};
use serde_json::{Value, json};
use std::collections::BTreeSet;

fn suites() -> Vec<Value> {
    [
        include_str!("fixtures/prediction_linear.json"),
        include_str!("fixtures/prediction_linear_training.json"),
    ]
    .iter()
    .map(|s| serde_json::from_str(s).unwrap())
    .collect()
}

fn config(case: &Value) -> Config {
    let pairs = case["params"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(key, _)| key.as_str() != "forcedsplits_filename")
        .map(|(key, value)| {
            (
                key.clone(),
                value
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| value.to_string()),
            )
        });
    let mut config = Config::from_pairs(pairs).unwrap();
    config.num_iterations = case["training"]["num_boost_round"].as_u64().unwrap() as usize;
    if case["params"].get("forcedsplits_filename").is_some() {
        // The saved generator's absolute development path is not a test dependency.
        config.forced_splits = Some(json!({"feature":0,"threshold":0.0,
            "left":{"feature":1,"threshold":-12.0},"right":{"feature":1,"threshold":12.0}}));
    }
    config
}

fn matrix(value: &Value) -> (Vec<f64>, usize, usize) {
    let rows = value.as_array().unwrap();
    let width = rows[0].as_array().unwrap().len();
    assert!(rows.iter().all(|r| r.as_array().unwrap().len() == width));
    (
        rows.iter()
            .flat_map(|r| {
                r.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(f64::NAN))
            })
            .collect(),
        rows.len(),
        width,
    )
}

fn near(actual: f64, expected: f64, context: &str) {
    assert!(
        actual.is_finite()
            && expected.is_finite()
            && (actual - expected).abs() <= 1e-12 + 1e-10 * expected.abs(),
        "{context}: {actual} != {expected}"
    );
}

fn tree_equal(actual: &Tree, expected: &Tree, context: &str) {
    assert_eq!(actual.num_leaves(), expected.num_leaves(), "{context}");
    assert_eq!(
        actual.split_features(),
        expected.split_features(),
        "{context}"
    );
    assert_eq!(actual.thresholds(), expected.thresholds(), "{context}");
    assert_eq!(
        actual.left_children(),
        expected.left_children(),
        "{context}"
    );
    assert_eq!(
        actual.right_children(),
        expected.right_children(),
        "{context}"
    );
    assert_eq!(
        actual.decision_types(),
        expected.decision_types(),
        "{context}"
    );
    assert_eq!(actual.leaf_counts(), expected.leaf_counts(), "{context}");
    let a = actual.linear_leaves().expect("trained tree must be linear");
    let e = expected.linear_leaves().unwrap();
    assert_eq!(a.features(), e.features(), "{context} feature pruning");
    for (&a, &e) in actual.leaf_values().iter().zip(expected.leaf_values()) {
        near(a, e, context);
    }
    for (&a, &e) in a.constants().iter().zip(e.constants()) {
        near(a, e, context);
    }
    for (a, e) in a.coefficients().iter().zip(e.coefficients()) {
        assert_eq!(a.len(), e.len(), "{context}");
        for (&a, &e) in a.iter().zip(e) {
            near(a, e, context);
        }
    }
}

fn recorded_structure(tree: &Tree, node: &Value) {
    if let Some(leaf) = node["leaf_index"].as_u64() {
        let leaf = leaf as usize;
        let linear = tree.linear_leaves().unwrap();
        near(
            tree.leaf_values()[leaf],
            node["leaf_value"].as_f64().unwrap(),
            "recorded leaf output",
        );
        near(
            linear.constants()[leaf],
            node["leaf_const"].as_f64().unwrap(),
            "recorded leaf constant",
        );
        let features: Vec<usize> = serde_json::from_value(node["leaf_features"].clone()).unwrap();
        assert_eq!(linear.features()[leaf], features);
        let coefficients: Vec<f64> = serde_json::from_value(node["leaf_coeff"].clone()).unwrap();
        assert_eq!(linear.coefficients()[leaf].len(), coefficients.len());
        for (&a, &e) in linear.coefficients()[leaf].iter().zip(&coefficients) {
            near(a, e, "recorded leaf coefficient");
        }
    } else {
        let split = node["split_index"].as_u64().unwrap() as usize;
        assert_eq!(
            tree.split_features()[split],
            node["split_feature"].as_u64().unwrap() as usize
        );
        assert_eq!(
            tree.thresholds()[split],
            node["threshold"].as_f64().unwrap()
        );
        for (child, actual) in [
            (&node["left_child"], tree.left_children()[split]),
            (&node["right_child"], tree.right_children()[split]),
        ] {
            let expected = child["leaf_index"]
                .as_u64()
                .map(|i| !(i as i32))
                .unwrap_or_else(|| child["split_index"].as_u64().unwrap() as i32);
            assert_eq!(actual, expected);
            recorded_structure(tree, child);
        }
    }
}

#[test]
fn saved_native_linear_structures_match_the_recorded_training_evidence() {
    let mut count = 0;
    for suite in suites() {
        for case in suite["cases"].as_array().unwrap() {
            let model = Model::from_text(
                &std::fs::read_to_string(format!(
                    "{}/tests/fixtures/{}",
                    env!("CARGO_MANIFEST_DIR"),
                    case["model_file"].as_str().unwrap()
                ))
                .unwrap(),
            )
            .unwrap();
            assert_eq!(
                model.trees().len(),
                case["training"]["expected_structure"]
                    .as_array()
                    .unwrap()
                    .len()
            );
            for (tree, recorded) in model
                .trees()
                .iter()
                .zip(case["training"]["expected_structure"].as_array().unwrap())
            {
                recorded_structure(tree, recorded);
            }
            let first = &model.trees()[0];
            assert!(
                first
                    .linear_leaves()
                    .unwrap()
                    .features()
                    .iter()
                    .all(Vec::is_empty)
            );
            assert_eq!(
                first.linear_leaves().unwrap().constants(),
                first.leaf_values()
            );
            count += 1;
        }
    }
    assert_eq!(count, 7);
}

#[test]
fn trainer_matches_all_seven_native_linear_cases() {
    let mut names = BTreeSet::new();
    for suite in suites() {
        assert_eq!(suite["schema_version"], 1);
        assert_eq!(
            suite["upstream_commit"],
            "5d84f3f222b41409a6983d2ac90513ad9702fa9e"
        );
        for case in suite["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            assert!(names.insert(name.to_owned()));
            let config = config(case);
            let (x, n, p) = matrix(&case["training"]["rows"]);
            let mut data = Dataset::from_dense(
                &x,
                n,
                p,
                serde_json::from_value(case["training"]["labels"].clone()).unwrap(),
                &config,
            )
            .unwrap();
            if let Some(weights) = case["training"].get("weights") {
                data = data
                    .with_weights(serde_json::from_value(weights.clone()).unwrap())
                    .unwrap();
            }
            let mut valid = data
                .validation_from_dense(&x, n, p, data.labels().to_vec())
                .unwrap();
            if let Some(weights) = data.weights() {
                valid = valid.with_weights(weights.to_vec()).unwrap();
            }
            let fitted = Trainer::new(config)
                .unwrap()
                .fit_with_validation(&data, &[("same", &valid)])
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            for round in &fitted.evaluation_history {
                assert!(!round.results.is_empty());
                for training in round.results.iter().filter(|r| r.dataset == "training") {
                    let validation = round
                        .results
                        .iter()
                        .find(|r| r.dataset == "same" && r.metric == training.metric)
                        .unwrap();
                    near(
                        validation.value,
                        training.value,
                        &format!("{name} validation scores"),
                    );
                }
            }
            let native = Model::from_text(
                &std::fs::read_to_string(format!(
                    "{}/tests/fixtures/{}",
                    env!("CARGO_MANIFEST_DIR"),
                    case["model_file"].as_str().unwrap()
                ))
                .unwrap(),
            )
            .unwrap();
            assert_eq!(fitted.model.trees().len(), native.trees().len(), "{name}");
            assert_eq!(
                fitted.model.trees().len(),
                case["training"]["expected_structure"]
                    .as_array()
                    .unwrap()
                    .len()
            );
            for (index, (actual, expected)) in
                fitted.model.trees().iter().zip(native.trees()).enumerate()
            {
                recorded_structure(expected, &case["training"]["expected_structure"][index]);
                tree_equal(actual, expected, &format!("{name} tree {index}"));
                if index == 0 {
                    assert!(
                        actual
                            .linear_leaves()
                            .unwrap()
                            .features()
                            .iter()
                            .all(Vec::is_empty)
                    );
                    assert_eq!(
                        actual.linear_leaves().unwrap().constants(),
                        actual.leaf_values()
                    );
                }
            }
            let (probes, rows, cols) = matrix(&case["rows"]);
            let predictions = fitted.model.predict_raw_dense(&probes, rows, cols).unwrap();
            assert_eq!(predictions.len(), case["raw"].as_array().unwrap().len());
            for (a, e) in predictions.iter().zip(case["raw"].as_array().unwrap()) {
                assert_eq!(a.len(), e.as_array().unwrap().len());
                for (&a, e) in a.iter().zip(e.as_array().unwrap()) {
                    near(a, e.as_f64().unwrap(), name);
                }
            }
            let transformed = fitted.model.predict_dense(&probes, rows, cols).unwrap();
            assert_eq!(
                transformed.len(),
                case["transformed"].as_array().unwrap().len()
            );
            for (actual, expected) in transformed
                .iter()
                .zip(case["transformed"].as_array().unwrap())
            {
                assert_eq!(actual.len(), expected.as_array().unwrap().len());
                for (&a, e) in actual.iter().zip(expected.as_array().unwrap()) {
                    near(a, e.as_f64().unwrap(), name);
                }
            }
            let leaves = fitted
                .model
                .predict_leaf_dense(&probes, rows, cols)
                .unwrap();
            assert_eq!(
                serde_json::to_value(leaves).unwrap(),
                case["leaves"],
                "{name} leaves"
            );
            for (row, expected) in fitted.training_scores.iter().zip(&predictions[..n]) {
                near(*row, expected[0], name);
            }
            let restored = Model::from_text(&fitted.model.to_text().unwrap()).unwrap();
            for (a, e) in restored
                .predict_raw_dense(&probes, rows, cols)
                .unwrap()
                .iter()
                .zip(&predictions)
            {
                for (&a, &e) in a.iter().zip(e) {
                    near(a, e, name);
                }
            }
        }
    }
    assert_eq!(
        names,
        [
            "linear_independent_lambda0",
            "linear_independent_lambda05",
            "linear_rank_deficient_lambda0",
            "linear_rank_deficient_lambda05",
            "linear_nan_training",
            "linear_weighted_training",
            "linear_nan_weighted_training"
        ]
        .map(str::to_owned)
        .into_iter()
        .collect()
    );
}
