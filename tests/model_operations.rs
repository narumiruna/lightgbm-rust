//! Saved native operation evidence. Normal tests do not load a native library.
use lightgbm_rust::{Boosting, Config, Dataset, Error, Model};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn suite() -> Value {
    let suite: Value =
        serde_json::from_str(include_str!("fixtures/prediction_operations.json")).unwrap();
    assert_eq!(
        suite["upstream_commit"],
        "5d84f3f222b41409a6983d2ac90513ad9702fa9e"
    );
    assert_eq!(
        suite["provenance"]["loaded_library_sha256"],
        "9ed1d46d2d60bac2966fd03bd81e7d192025804f6a3e3ef24c3aa483d0ec7c1e"
    );
    suite
}

fn case<'a>(suite: &'a Value, name: &str) -> &'a Value {
    suite["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == name)
        .unwrap()
}

fn model(snapshot: &Value) -> Model {
    Model::load(fixtures().join(snapshot["model_file"].as_str().unwrap())).unwrap()
}

fn config(case: &Value) -> Config {
    Config::from_pairs(
        case["params"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(key, value)| {
                (
                    key.clone(),
                    if let Some(s) = value.as_str() {
                        s.to_owned()
                    } else if let Some(a) = value.as_array() {
                        a.iter().map(Value::to_string).collect::<Vec<_>>().join(",")
                    } else {
                        value.to_string()
                    },
                )
            }),
    )
    .unwrap()
}

fn rows(case: &Value) -> Vec<Vec<f64>> {
    case["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap_or(f64::NAN))
                .collect()
        })
        .collect()
}

fn data(case: &Value, config: &Config) -> Dataset {
    let rows = rows(case);
    let labels = serde_json::from_value(case["labels"].clone()).unwrap();
    let mut data =
        Dataset::from_dense(&rows.concat(), rows.len(), rows[0].len(), labels, config).unwrap();
    if !case["weights"].is_null() {
        data = data
            .with_weights(serde_json::from_value(case["weights"].clone()).unwrap())
            .unwrap();
    }
    if !case["groups"].is_null() {
        data = data
            .with_groups(serde_json::from_value(case["groups"].clone()).unwrap())
            .unwrap();
    }
    if !case["positions"].is_null() {
        data = data
            .with_positions(serde_json::from_value(case["positions"].clone()).unwrap())
            .unwrap();
    }
    if !case["initial_scores"].is_null() {
        data = data
            .with_initial_scores(
                serde_json::from_value(case["initial_scores"].clone()).unwrap(),
                config.num_class,
            )
            .unwrap();
    }
    data
}

fn close(actual: f64, expected: f64) {
    assert!(actual.is_finite());
    assert!(
        (actual - expected).abs() <= 1e-12 + 1e-10 * expected.abs(),
        "{actual} != {expected}"
    );
}

fn compare_predictions(model: &Model, case: &Value, snapshot: &Value) {
    assert_eq!(
        model.num_iterations(),
        snapshot["current_iteration"].as_u64().unwrap() as usize
    );
    assert_eq!(
        model.trees().len(),
        snapshot["num_trees"].as_u64().unwrap() as usize
    );
    for (i, row) in rows(case).iter().enumerate() {
        for (a, e) in model
            .predict_raw_row(row)
            .unwrap()
            .into_iter()
            .zip(snapshot["raw"][i].as_array().unwrap())
        {
            close(a, e.as_f64().unwrap());
        }
        for (a, e) in model
            .predict_row(row)
            .unwrap()
            .into_iter()
            .zip(snapshot["transformed"][i].as_array().unwrap())
        {
            close(a, e.as_f64().unwrap());
        }
        let expected: Vec<usize> = serde_json::from_value(snapshot["leaves"][i].clone()).unwrap();
        assert_eq!(model.predict_leaf_row(row).unwrap(), expected);
    }
}

#[test]
fn native_leaf_set_get_preserves_structure_and_input() {
    let suite = suite();
    let case = case(&suite, "leaf_output");
    let original = model(&case["initial_model"]);
    let before = original.to_text().unwrap();
    let tree = case["tree_id"].as_u64().unwrap() as usize;
    let leaf = case["leaf_id"].as_u64().unwrap() as usize;
    close(
        original.leaf_output(tree, leaf).unwrap(),
        case["old_output"].as_f64().unwrap(),
    );
    let updated = original
        .with_leaf_output(tree, leaf, case["set_output"].as_f64().unwrap())
        .unwrap();
    close(
        updated.leaf_output(tree, leaf).unwrap(),
        case["get_output"].as_f64().unwrap(),
    );
    compare_predictions(&updated, case, &case["final_model"]);
    compare_predictions(
        &Model::from_text(&updated.to_text().unwrap()).unwrap(),
        case,
        &case["final_model"],
    );
    assert_eq!(original.to_text().unwrap(), before);
    let mut old = serde_json::to_value(&original).unwrap();
    old["trees"][tree]["leaf_values"][leaf] = case["get_output"].clone();
    assert_eq!(serde_json::to_value(&updated).unwrap(), old);
}

#[test]
fn checked_leaf_set_rounding_and_linear_fallback_only() {
    let suite = suite();
    let case = case(&suite, "refit_linear_decay05");
    let original = model(&case["initial_model"]);
    for value in [0.0, -0.0, 1e-35_f32 as f64, -(1e-35_f32 as f64)] {
        let updated = original.with_leaf_output(0, 0, value).unwrap();
        assert_eq!(
            updated.leaf_output(0, 0).unwrap().to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            serde_json::to_value(updated.trees()[0].linear_leaves()).unwrap(),
            serde_json::to_value(original.trees()[0].linear_leaves()).unwrap()
        );
    }
    let just_above = f64::from_bits((1e-35_f32 as f64).to_bits() + 1);
    assert_eq!(
        original
            .with_leaf_output(0, 0, just_above)
            .unwrap()
            .leaf_output(0, 0)
            .unwrap(),
        just_above
    );
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            original.with_leaf_output(0, 0, value),
            Err(Error::InvalidInput(_))
        ));
    }
    for (tree, leaf) in [
        (usize::MAX, 0),
        (0, usize::MAX),
        (original.trees().len(), 0),
        (0, original.trees()[0].num_leaves()),
    ] {
        assert!(matches!(
            original.leaf_output(tree, leaf),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            original.with_leaf_output(tree, leaf, 1.0),
            Err(Error::InvalidInput(_))
        ));
    }
}

#[test]
fn native_rollback_reaches_zero_and_stays_zero() {
    let suite = suite();
    let case = case(&suite, "rollback");
    let original = model(&case["initial_model"]);
    let before = original.to_text().unwrap();
    let mut current = original.clone();
    for step in case["steps"].as_array().unwrap() {
        current = current.rollback_one_iteration().unwrap();
        compare_predictions(&current, case, &step["model"]);
        compare_predictions(
            &Model::from_text(&current.to_text().unwrap()).unwrap(),
            case,
            &step["model"],
        );
        let prefix = original.truncate(current.num_iterations()).unwrap();
        assert_eq!(current.to_text().unwrap(), prefix.to_text().unwrap());
    }
    assert_eq!(current.num_iterations(), 0);
    assert_eq!(current.feature_names(), original.feature_names());
    assert_eq!(current.feature_infos(), original.feature_infos());
    assert_eq!(current.objective(), original.objective());
    assert_eq!(original.to_text().unwrap(), before);
    assert!(matches!(
        original.truncate(usize::MAX),
        Err(Error::InvalidInput(_))
    ));
}

#[test]
fn native_multiclass_prefix_contains_whole_class_rounds() {
    let suite = suite();
    let case = case(&suite, "predict_early_multiclass");
    let original = model(&case["initial_model"]);
    for (i, predictions) in case["prefix_raw"].as_array().unwrap().iter().enumerate() {
        let prefix = original.truncate(i + 1).unwrap();
        assert_eq!(prefix.trees().len(), (i + 1) * original.num_classes());
        for (row, expected) in rows(case).iter().zip(predictions.as_array().unwrap()) {
            for (a, e) in prefix
                .predict_raw_row(row)
                .unwrap()
                .into_iter()
                .zip(expected.as_array().unwrap())
            {
                close(a, e.as_f64().unwrap());
            }
        }
    }
    let mut current = original;
    while current.num_iterations() != 0 {
        let n = current.num_iterations();
        current = current.rollback_one_iteration().unwrap();
        assert_eq!(current.trees().len(), (n - 1) * current.num_classes());
    }
    assert_eq!(
        current.predict_raw_row(&rows(case)[0]).unwrap(),
        vec![0.0; 3]
    );
}

#[test]
fn native_refit_nonlinear_predictions_and_metadata() {
    let suite = suite();
    let mut covered = BTreeSet::new();
    for case in suite["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        if case["operation"] != "refit"
            || name.starts_with("refit_linear_")
            || case.get("error").is_some()
        {
            continue;
        }
        let config = config(case);
        let data = data(case, &config);
        let leaves: Vec<Vec<usize>> = serde_json::from_value(case["leaf_pred"].clone()).unwrap();
        let original = model(&case["initial_model"]);
        let before = original.to_text().unwrap();
        let actual = original
            .refit(
                &data,
                &leaves,
                &config,
                case["decay_rate"].as_f64().unwrap(),
            )
            .unwrap();
        compare_predictions(&actual, case, &case["final_model"]);
        compare_predictions(
            &Model::from_text(&actual.to_text().unwrap()).unwrap(),
            case,
            &case["final_model"],
        );
        let expected = model(&case["final_model"]);
        for ((a, old), e) in actual
            .trees()
            .iter()
            .zip(original.trees())
            .zip(expected.trees())
        {
            assert_eq!(a.split_features(), e.split_features(), "{name}");
            assert_eq!(a.thresholds(), e.thresholds(), "{name}");
            assert_eq!(a.left_children(), e.left_children(), "{name}");
            assert_eq!(a.right_children(), e.right_children(), "{name}");
            assert_eq!(a.decision_types(), e.decision_types(), "{name}");
            assert_eq!(a.cat_threshold(), e.cat_threshold(), "{name}");
            assert_eq!(a.leaf_counts(), old.leaf_counts(), "{name}");
            assert_eq!(a.leaf_counts(), e.leaf_counts(), "{name}");
            assert_eq!(a.leaf_weights(), old.leaf_weights(), "{name}");
            assert_eq!(a.internal_counts(), old.internal_counts(), "{name}");
            assert_eq!(a.internal_values(), old.internal_values(), "{name}");
            assert_eq!(a.shrinkage(), e.shrinkage(), "{name}");
            for (&a, &e) in a.leaf_values().iter().zip(e.leaf_values()) {
                close(a, e);
            }
        }
        assert_eq!(original.to_text().unwrap(), before);
        covered.insert(name.to_owned());
    }
    for kind in [
        "regression",
        "binary",
        "multiclass",
        "categorical",
        "ranking",
        "rf",
        "dart",
        "l1",
        "quantile",
        "mape",
        "gbdt_querybag",
        "goss_querybag",
    ] {
        for decay in ["00", "05", "09", "10"] {
            assert!(covered.contains(&format!("refit_{kind}_decay{decay}")));
        }
    }
    assert_eq!(covered.len(), 48);
}

#[test]
fn native_refit_linear_preserves_selected_features_and_matches_coefficients() {
    let suite = suite();
    let mut covered = 0;
    for case in suite["cases"].as_array().unwrap() {
        if !case["name"].as_str().unwrap().starts_with("refit_linear_") {
            continue;
        }
        let config = config(case);
        let data = data(case, &config);
        let original = model(&case["initial_model"]);
        let leaves: Vec<Vec<usize>> = serde_json::from_value(case["leaf_pred"].clone()).unwrap();
        let actual = original
            .refit(
                &data,
                &leaves,
                &config,
                case["decay_rate"].as_f64().unwrap(),
            )
            .unwrap();
        let expected = model(&case["final_model"]);
        compare_predictions(&actual, case, &case["final_model"]);
        compare_predictions(
            &Model::from_text(&actual.to_text().unwrap()).unwrap(),
            case,
            &case["final_model"],
        );
        for ((a, e), old) in actual
            .trees()
            .iter()
            .zip(expected.trees())
            .zip(original.trees())
        {
            let al = a.linear_leaves().unwrap();
            let el = e.linear_leaves().unwrap();
            assert_eq!(al.features(), el.features());
            assert_eq!(al.features(), old.linear_leaves().unwrap().features());
            assert_eq!(a.leaf_counts(), e.leaf_counts());
            for (&a, &e) in a.leaf_values().iter().zip(e.leaf_values()) {
                close(a, e);
            }
            for (&a, &e) in al.constants().iter().zip(el.constants()) {
                close(a, e);
            }
            for (&a, &e) in al
                .coefficients()
                .iter()
                .flatten()
                .zip(el.coefficients().iter().flatten())
            {
                close(a, e);
            }
        }
        covered += 1;
    }
    assert_eq!(covered, 4);
}

#[test]
fn native_refit_class_major_score_buffers_match_each_iteration() {
    let suite = suite();
    let mut covered = 0;
    for case in suite["cases"].as_array().unwrap() {
        if case["operation"] != "refit" || case.get("error").is_some() {
            continue;
        }
        let config = config(case);
        let data = data(case, &config);
        let original = model(&case["initial_model"]);
        let leaves: Vec<Vec<usize>> = serde_json::from_value(case["leaf_pred"].clone()).unwrap();
        for iterations in 1..=original.num_iterations() {
            let prefix = original.truncate(iterations).unwrap();
            let ids: Vec<Vec<usize>> = leaves
                .iter()
                .map(|row| row[..prefix.trees().len()].to_vec())
                .collect();
            let actual = prefix
                .refit(&data, &ids, &config, case["decay_rate"].as_f64().unwrap())
                .unwrap();
            let expected = if iterations < original.num_iterations() {
                &case["refit_steps"][iterations]["training_scores_before"]
            } else {
                &case["training_scores_after"]
            };
            assert_eq!(
                expected.as_array().unwrap().len(),
                data.nrows() * original.num_classes()
            );
            for (row, input) in rows(case).iter().enumerate() {
                for (class, value) in actual
                    .predict_raw_row(input)
                    .unwrap()
                    .into_iter()
                    .enumerate()
                {
                    close(
                        value,
                        expected[class * data.nrows() + row].as_f64().unwrap(),
                    );
                }
            }
        }
        covered += 1;
    }
    assert_eq!(covered, 52);
}

#[test]
fn refit_uses_source_shrinkage_not_new_learning_rate_and_accepts_zero_margins() {
    let suite = suite();
    let case = case(&suite, "refit_regression_decay05");
    let mut config = config(case);
    let data = data(case, &config)
        .with_initial_scores(vec![0.0; case["rows"].as_array().unwrap().len()], 1)
        .unwrap();
    let original = model(&case["initial_model"]);
    let leaves: Vec<Vec<usize>> = serde_json::from_value(case["leaf_pred"].clone()).unwrap();
    config.learning_rate = 0.73;
    let actual = original.refit(&data, &leaves, &config, 0.5).unwrap();
    compare_predictions(&actual, case, &case["final_model"]);
}

#[test]
fn refit_uses_original_leaf_indices_not_new_lossy_bin_thresholds() {
    let suite = suite();
    let case = case(&suite, "refit_regression_decay05");
    let mut config = config(case);
    config.max_bin = 2;
    let data = data(case, &config);
    let original = model(&case["initial_model"]);
    let leaves: Vec<Vec<usize>> = serde_json::from_value(case["leaf_pred"].clone()).unwrap();
    for (row, expected) in rows(case).iter().zip(&leaves) {
        assert_eq!(original.predict_leaf_row(row).unwrap(), *expected);
    }
    assert!(leaves.windows(2).any(|pair| pair[0] == pair[1]));
    let actual = original.refit(&data, &leaves, &config, 0.5).unwrap();
    compare_predictions(&actual, case, &case["final_model"]);
}

#[test]
fn refit_invalid_shapes_indices_and_parameters_return_typed_errors() {
    let suite = suite();
    let case = case(&suite, "refit_regression_decay05");
    let config = config(case);
    let data = data(case, &config);
    let original = model(&case["initial_model"]);
    let before = original.to_text().unwrap();
    let leaves: Vec<Vec<usize>> = serde_json::from_value(case["leaf_pred"].clone()).unwrap();
    for decay in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            original.refit(&data, &leaves, &config, decay),
            Err(Error::InvalidInput(_))
        ));
    }
    for invalid in [
        vec![],
        vec![vec![]; data.nrows()],
        vec![vec![0; original.trees().len() + 1]; data.nrows()],
        vec![vec![usize::MAX; original.trees().len()]; data.nrows()],
    ] {
        assert!(matches!(
            original.refit(&data, &invalid, &config, 0.5),
            Err(Error::InvalidInput(_))
        ));
    }
    let mut invalid = leaves.clone();
    let last = invalid.len() - 1;
    invalid[last][0] = original.trees()[0].num_leaves();
    assert!(matches!(
        original.refit(&data, &invalid, &config, 0.5),
        Err(Error::InvalidInput(_))
    ));
    let bad_margin = data
        .clone()
        .with_initial_scores(vec![0.0; data.nrows() * 2], 2)
        .unwrap();
    assert!(matches!(
        original.refit(&bad_margin, &leaves, &config, 0.5),
        Err(Error::InvalidInput(_))
    ));
    let wrong_features = Dataset::from_dense(
        &vec![0.0; data.nrows()],
        data.nrows(),
        1,
        data.labels().to_vec(),
        &config,
    )
    .unwrap();
    assert!(matches!(
        original.refit(&wrong_features, &leaves, &config, 0.5),
        Err(Error::InvalidInput(_))
    ));
    let overflow = data
        .clone()
        .with_weights(vec![f32::MAX; data.nrows()])
        .unwrap();
    assert!(matches!(
        original.refit(&overflow, &leaves, &config, 0.5),
        Err(Error::InvalidInput(_))
    ));
    let mut changed = config.clone();
    changed.max_bin = 2;
    assert!(matches!(
        original.refit(&data, &leaves, &changed, 0.5),
        Err(Error::InvalidInput(_))
    ));
    let mut changed = config.clone();
    changed.linear_tree = true;
    assert!(matches!(
        original.refit(&data, &leaves, &changed, 0.5),
        Err(Error::InvalidInput(_))
    ));
    let mut changed = config.clone();
    changed.learning_rate = f64::NAN;
    assert!(original.refit(&data, &leaves, &changed, 0.5).is_err());
    let mut changed = config.clone();
    changed.objective = lightgbm_rust::Objective::Binary;
    assert!(matches!(
        original.refit(&data, &leaves, &changed, 0.5),
        Err(Error::InvalidInput(_))
    ));
    let mut changed = config.clone();
    changed.boosting = Boosting::Rf;
    changed.bagging_fraction = 0.8;
    changed.bagging_freq = 1;
    assert!(matches!(
        original.refit(&data, &leaves, &changed, 0.5),
        Err(Error::InvalidInput(_))
    ));
    let mut changed = config.clone();
    changed.boosting = Boosting::Dart;
    assert!(original.refit(&data, &leaves, &changed, 0.5).is_ok());
    let mut changed = config.clone();
    changed.path_smooth = 1.0;
    changed.min_data_in_leaf = 2;
    assert!(matches!(
        original.refit(&data, &leaves, &changed, 0.5),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        original
            .truncate(0)
            .unwrap()
            .refit(&data, &leaves, &config, 0.5),
        Err(Error::InvalidInput(_))
    ));
    assert_eq!(original.to_text().unwrap(), before);
}
