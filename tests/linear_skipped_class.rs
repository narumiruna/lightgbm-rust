//! Native skipped linear-tree metadata and source-checked safe refit evidence.
use lightgbm_rust::{Config, Dataset, Error, Model, Trainer, Tree};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/review_linear.json")).unwrap()
}

fn config(case: &Value) -> Config {
    let mut config =
        Config::from_pairs(case["params"].as_object().unwrap().iter().map(|(k, v)| {
            (
                k.clone(),
                v.as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| v.to_string()),
            )
        }))
        .unwrap();
    config.num_iterations = case["training"]["num_boost_round"].as_u64().unwrap() as usize;
    config
}

fn matrix(rows: &Value) -> (Vec<f64>, usize, usize) {
    let rows = rows.as_array().unwrap();
    let width = rows[0].as_array().unwrap().len();
    assert!(
        rows.iter()
            .all(|row| row.as_array().unwrap().len() == width)
    );
    (
        rows.iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect(),
        rows.len(),
        width,
    )
}

fn dataset(rows: &Value, labels: &Value, config: &Config) -> Dataset {
    let (x, n, p) = matrix(rows);
    Dataset::from_dense(
        &x,
        n,
        p,
        serde_json::from_value(labels.clone()).unwrap(),
        config,
    )
    .unwrap()
}

fn native(case: &Value, key: &str) -> Model {
    let path = case[key].as_str().unwrap();
    Model::from_text(
        &std::fs::read_to_string(format!(
            "{}/tests/fixtures/{path}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap(),
    )
    .unwrap()
}

fn near(actual: f64, expected: f64, context: &str) {
    assert!(
        actual.is_finite()
            && expected.is_finite()
            && (actual - expected).abs() <= 1e-12 + 1e-10 * expected.abs(),
        "{context}: {actual} != {expected}"
    );
}

fn snapshots(model: &Model, rows: &Value, expected: &Value, name: &str) {
    assert_eq!(
        model.trees().len(),
        expected["num_trees"].as_u64().unwrap() as usize,
        "{name}"
    );
    assert_eq!(
        model.num_iterations(),
        expected["current_iteration"].as_u64().unwrap() as usize,
        "{name}"
    );
    let (x, n, p) = matrix(rows);
    for (actual, key) in [
        (model.predict_raw_dense(&x, n, p).unwrap(), "raw"),
        (model.predict_dense(&x, n, p).unwrap(), "transformed"),
    ] {
        let expected: Vec<Vec<f64>> = serde_json::from_value(expected[key].clone()).unwrap();
        assert_eq!(actual.len(), expected.len(), "{name}");
        for (a, e) in actual.iter().zip(expected) {
            assert_eq!(a.len(), e.len(), "{name}");
            for (&a, e) in a.iter().zip(e) {
                near(a, e, name);
            }
        }
    }
    let expected: Vec<Vec<usize>> = serde_json::from_value(expected["leaves"].clone()).unwrap();
    assert_eq!(
        model.predict_leaf_dense(&x, n, p).unwrap(),
        expected,
        "{name}"
    );
}

fn tree_equal(actual: &Tree, expected: &Tree, name: &str) {
    assert_eq!(actual.num_leaves(), expected.num_leaves(), "{name}");
    assert_eq!(actual.split_features(), expected.split_features(), "{name}");
    assert_eq!(actual.thresholds(), expected.thresholds(), "{name}");
    assert_eq!(actual.left_children(), expected.left_children(), "{name}");
    assert_eq!(actual.right_children(), expected.right_children(), "{name}");
    assert_eq!(actual.decision_types(), expected.decision_types(), "{name}");
    assert_eq!(actual.leaf_counts(), expected.leaf_counts(), "{name}");
    for (&a, &e) in actual.leaf_values().iter().zip(expected.leaf_values()) {
        near(a, e, name);
    }
    match (actual.linear_leaves(), expected.linear_leaves()) {
        (Some(a), Some(e)) => {
            assert_eq!(a.features(), e.features(), "{name}");
            for (&a, &e) in a.constants().iter().zip(e.constants()) {
                near(a, e, name);
            }
            for (a, e) in a.coefficients().iter().zip(e.coefficients()) {
                assert_eq!(a.len(), e.len(), "{name}");
                for (&a, &e) in a.iter().zip(e) {
                    near(a, e, name);
                }
            }
        }
        (None, None) => {}
        _ => panic!("{name}: linear metadata differs"),
    }
}

fn models_equal(actual: &Model, expected: &Model, name: &str) {
    assert_eq!(actual.trees().len(), expected.trees().len(), "{name}");
    for (index, (a, e)) in actual.trees().iter().zip(expected.trees()).enumerate() {
        tree_equal(a, e, &format!("{name} tree {index}"));
    }
}

#[test]
fn fresh_and_reloaded_fits_preserve_native_skipped_class_layout_and_scores() {
    let fixture = fixture();
    assert_eq!(fixture["schema_version"], 1);
    assert_eq!(
        fixture["upstream_commit"],
        "5d84f3f222b41409a6983d2ac90513ad9702fa9e"
    );
    assert_eq!(
        fixture["library"]["loaded_sha256"],
        "9ed1d46d2d60bac2966fd03bd81e7d192025804f6a3e3ef24c3aa483d0ec7c1e"
    );
    assert_eq!(
        fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|case| case["name"].as_str().unwrap())
            .collect::<Vec<_>>(),
        [
            "linear_missing_class",
            "linear_all_classes",
            "constant_missing_class",
            "linear_constant_features",
            "linear_no_features",
            "linear_binary_single_class",
            "linear_cancellation_missing_class",
            "linear_cancellation_all_classes"
        ]
    );
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let config = config(case);
        let data = dataset(
            &case["training"]["rows"],
            &case["training"]["labels"],
            &config,
        );
        let fitted = Trainer::new(config).unwrap().fit(&data).unwrap().model;
        let expected = native(case, "model_file");
        let reloaded = Model::from_text(&fitted.to_text().unwrap()).unwrap();
        for model in [&expected, &fitted, &reloaded] {
            let layout: Vec<bool> = model
                .trees()
                .iter()
                .map(|t| t.linear_leaves().is_some())
                .collect();
            assert_eq!(
                layout,
                serde_json::from_value::<Vec<bool>>(case["linear_layout"].clone()).unwrap(),
                "{name}"
            );
            models_equal(model, &expected, name);
            snapshots(model, &case["rows"], &case["fresh"], name);
            snapshots(model, &case["rows"], &case["reloaded"], name);
        }
        if name == "linear_missing_class" {
            assert_eq!(fitted.trees().len(), 9);
            for (round, trees) in fitted.trees().chunks_exact(3).enumerate() {
                assert!(trees[0].linear_leaves().is_some());
                assert!(trees[1].linear_leaves().is_some());
                assert!(trees[2].linear_leaves().is_none());
                assert_eq!(trees[2].num_leaves(), 1);
                assert_eq!(trees[2].leaf_counts(), &[24]);
                if round > 0 {
                    assert_eq!(trees[2].leaf_values(), &[0.0]);
                }
            }
            let (x, n, p) = matrix(&case["rows"]);
            let raw = fitted.predict_raw_dense(&x, n, p).unwrap();
            for row in raw {
                assert_eq!(row[2], fitted.trees()[2].leaf_values()[0]);
            }
        }
    }
}

#[test]
fn safe_native_refits_match_fresh_and_reloaded_rust_models() {
    for case in fixture()["cases"].as_array().unwrap() {
        if case["refit"]["status"] != "executed" {
            continue;
        }
        let name = case["name"].as_str().unwrap();
        let config = config(case);
        let training = dataset(
            &case["training"]["rows"],
            &case["training"]["labels"],
            &config,
        );
        let fitted = Trainer::new(config.clone())
            .unwrap()
            .fit(&training)
            .unwrap()
            .model;
        let reloaded = Model::from_text(&fitted.to_text().unwrap()).unwrap();
        let imported = native(case, "model_file");
        let refit = &case["refit"];
        let data = dataset(&refit["rows"], &refit["labels"], &config);
        let (x, n, p) = matrix(&refit["rows"]);
        for model in [&fitted, &reloaded, &imported] {
            let leaves = model.predict_leaf_dense(&x, n, p).unwrap();
            let result = model
                .refit(
                    &data,
                    &leaves,
                    &config,
                    refit["decay_rate"].as_f64().unwrap(),
                )
                .unwrap();
            models_equal(&result, &native(&refit["model"], "model_file"), name);
            snapshots(&result, &refit["rows"], &refit["fresh"], name);
            snapshots(&result, &refit["rows"], &refit["reloaded"], name);
            let saved = Model::from_text(&result.to_text().unwrap()).unwrap();
            models_equal(&saved, &result, name);
            snapshots(&saved, &refit["rows"], &refit["reloaded"], name);
        }
    }
}

#[test]
fn undefined_native_linear_refits_are_rejected_without_retagging_trees() {
    for case in fixture()["cases"].as_array().unwrap() {
        if case["refit"]["status"] != "not_executed_undefined_native_state" {
            continue;
        }
        assert!(case["refit"]["fresh"].is_null());
        assert!(case["refit"]["reloaded"].is_null());
        assert!(case["refit"]["model"].is_null());
        let config = config(case);
        let training = dataset(
            &case["training"]["rows"],
            &case["training"]["labels"],
            &config,
        );
        let fitted = Trainer::new(config.clone())
            .unwrap()
            .fit(&training)
            .unwrap()
            .model;
        let reloaded = Model::from_text(&fitted.to_text().unwrap()).unwrap();
        let imported = native(case, "model_file");
        let data = dataset(&case["refit"]["rows"], &case["refit"]["labels"], &config);
        let (x, n, p) = matrix(&case["refit"]["rows"]);
        for model in [&fitted, &reloaded, &imported] {
            let before = model.to_text().unwrap();
            let leaves = model.predict_leaf_dense(&x, n, p).unwrap();
            let error = model.refit(&data, &leaves, &config, 0.5).unwrap_err();
            assert!(
                matches!(&error, Error::Unsupported(message) if message.contains("linear") && message.contains("refit") && message.contains("undefined native")),
                "{error}"
            );
            assert_eq!(model.to_text().unwrap(), before);
        }
    }
}

#[test]
fn truly_mismatched_linear_configurations_still_fail() {
    for case in fixture()["cases"].as_array().unwrap() {
        if ![
            "linear_all_classes",
            "constant_missing_class",
            "linear_missing_class",
        ]
        .contains(&case["name"].as_str().unwrap())
        {
            continue;
        }
        let mut config = config(case);
        let model = native(case, "model_file");
        config.linear_tree = !config.linear_tree;
        // Rebuild with the mismatched mode so the model check, not a stale
        // Dataset binning configuration, rejects the request.
        let data = dataset(&case["refit"]["rows"], &case["refit"]["labels"], &config);
        let (x, n, p) = matrix(&case["refit"]["rows"]);
        let leaves = model.predict_leaf_dense(&x, n, p).unwrap();
        let error = model.refit(&data, &leaves, &config, 0.5).unwrap_err();
        if config.linear_tree {
            // This wrong mode would also enter the native empty-vector path
            // for the absent class. It remains a rejected configuration.
            assert!(
                matches!(&error, Error::Unsupported(message) if message.contains("undefined native")),
                "{error}"
            );
        } else {
            assert!(
                matches!(&error, Error::InvalidInput(message) if message.contains("linear-tree configuration differs")),
                "{error}"
            );
        }
    }
}
