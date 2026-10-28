//! R1: compare serialized transforms without rounding refit derivatives.
use lightgbm_rust::{Config, Dataset, Error, Model, Trainer};
use serde_json::Value;
use std::path::{Component, Path};

fn suite() -> Value {
    let suite: Value = serde_json::from_str(include_str!("fixtures/review_refit.json")).unwrap();
    assert_eq!(
        suite["upstream_commit"],
        "5d84f3f222b41409a6983d2ac90513ad9702fa9e"
    );
    assert_eq!(
        suite["library"]["loaded_sha256"],
        "9ed1d46d2d60bac2966fd03bd81e7d192025804f6a3e3ef24c3aa483d0ec7c1e"
    );
    assert_eq!(suite["cases"].as_array().unwrap().len(), 4);
    suite
}

fn config(case: &Value) -> Config {
    let mut config = Config::from_pairs(case["params"].as_object().unwrap().iter().map(
        |(key, value)| {
            (
                key.clone(),
                value
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| value.to_string()),
            )
        },
    ))
    .unwrap();
    config.num_iterations = case["training"]["num_boost_round"].as_u64().unwrap() as usize;
    config
}

fn rows(value: &Value) -> Vec<Vec<f64>> {
    serde_json::from_value(value["rows"].clone()).unwrap()
}

fn data(value: &Value, config: &Config) -> Dataset {
    let rows = rows(value);
    Dataset::from_dense(
        &rows.concat(),
        rows.len(),
        rows[0].len(),
        serde_json::from_value(value["labels"].clone()).unwrap(),
        config,
    )
    .unwrap()
    .with_weights(serde_json::from_value(value["weights"].clone()).unwrap())
    .unwrap()
}

fn load(snapshot: &Value) -> Model {
    let path = Path::new(snapshot["model_file"].as_str().unwrap());
    assert!(!path.is_absolute());
    assert!(path.components().all(|p| matches!(p, Component::Normal(_))));
    assert!(path.starts_with("models/review_refit"));
    Model::load(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(path),
    )
    .unwrap()
}

fn close(actual: f64, expected: f64) {
    assert!(actual.is_finite());
    assert!(
        (actual - expected).abs() <= 1e-12 + 1e-10 * expected.abs(),
        "{actual} != {expected}"
    );
}

fn compare_raw(model: &Model, case: &Value, expected: &Value) {
    assert_eq!(
        model.num_iterations(),
        expected["current_iteration"].as_u64().unwrap() as usize
    );
    assert_eq!(
        model.trees().len(),
        expected["num_trees"].as_u64().unwrap() as usize
    );
    for (row, input) in rows(case).iter().enumerate() {
        let actual = model.predict_raw_row(input).unwrap();
        let values = expected["raw"][row].as_array().unwrap();
        assert_eq!(actual.len(), values.len());
        for (actual, expected) in actual.into_iter().zip(values) {
            close(actual, expected.as_f64().unwrap());
        }
        let leaves: Vec<usize> = serde_json::from_value(expected["leaves"][row].clone()).unwrap();
        assert_eq!(model.predict_leaf_row(input).unwrap(), leaves);
    }
}

fn compare_refit(model: &Model, case: &Value, config: &Config, expected: &Value) {
    let before = model.to_text().unwrap();
    let leaves: Vec<Vec<usize>> = serde_json::from_value(case["leaf_pred"].clone()).unwrap();
    let result = model
        .refit(
            &data(case, config),
            &leaves,
            config,
            case["decay_rate"].as_f64().unwrap(),
        )
        .unwrap_or_else(|e| {
            panic!(
                "{}: unchanged sigmoid {}: {e}",
                case["name"], config.sigmoid
            )
        });
    compare_raw(&result, case, expected);
    compare_raw(
        &Model::from_text(&result.to_text().unwrap()).unwrap(),
        case,
        expected,
    );
    assert_eq!(model.to_text().unwrap(), before);
    for (actual, old) in result.trees().iter().zip(model.trees()) {
        assert_eq!(actual.split_features(), old.split_features());
        assert_eq!(actual.thresholds(), old.thresholds());
        assert_eq!(actual.leaf_counts(), old.leaf_counts());
        assert_eq!(actual.shrinkage(), old.shrinkage());
    }
    // Live refit uses the caller's full sigmoid. Text reload uses its token.
    for (row, input) in rows(case).iter().enumerate() {
        for (a, e) in result
            .predict_row(input)
            .unwrap()
            .into_iter()
            .zip(expected["transformed"][row].as_array().unwrap())
        {
            close(a, e.as_f64().unwrap());
        }
    }
    if config.sigmoid == case["params"]["sigmoid"].as_f64().unwrap() {
        let native = load(&case["final_model"]);
        for (actual, expected) in result.trees().iter().zip(native.trees()) {
            for (&a, &e) in actual.leaf_values().iter().zip(expected.leaf_values()) {
                close(a, e);
            }
        }
        for (row, input) in rows(case).iter().enumerate() {
            for (a, e) in Model::from_text(&result.to_text().unwrap())
                .unwrap()
                .predict_row(input)
                .unwrap()
                .into_iter()
                .zip(case["final_model"]["transformed"][row].as_array().unwrap())
            {
                close(a, e.as_f64().unwrap());
            }
        }
    }
}

fn check_fresh(objective: &str) {
    for case in suite()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["params"]["objective"] == objective)
    {
        let config = config(case);
        let original = Trainer::new(config.clone())
            .unwrap()
            .fit(&data(&case["training"], &config))
            .unwrap()
            .model;
        assert_eq!(original.objective(), case["objective"].as_str().unwrap());
        assert_ne!(config.sigmoid, case["serialized_sigmoid"].as_f64().unwrap());
        compare_raw(&original, case, &case["initial_model"]);
        compare_refit(&original, case, &config, &case["native_refit"]["fresh"]);
    }
}

fn check_reloaded(objective: &str) {
    for case in suite()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["params"]["objective"] == objective)
    {
        let config = config(case);
        let native = load(&case["initial_model"]);
        let original = Model::from_text(&native.to_text().unwrap()).unwrap();
        compare_refit(&original, case, &config, &case["native_refit"]["reloaded"]);
        let mut rounded = config.clone();
        rounded.sigmoid = case["serialized_sigmoid"].as_f64().unwrap();
        assert_ne!(
            case["rounded_refit"]["raw"],
            case["native_refit"]["reloaded"]["raw"]
        );
        compare_refit(&original, case, &rounded, &case["rounded_refit"]);
    }
}

#[test]
fn fresh_binary_refit_preserves_precise_sigmoid() {
    check_fresh("binary");
}

#[test]
fn fresh_ova_refit_preserves_precise_sigmoid() {
    check_fresh("multiclassova");
}

#[test]
fn reloaded_binary_refit_preserves_precise_sigmoid() {
    check_reloaded("binary");
}

#[test]
fn reloaded_ova_refit_preserves_precise_sigmoid() {
    check_reloaded("multiclassova");
}

#[test]
fn exact_imported_sigmoid_and_equivalent_token_spelling_stay_compatible() {
    for case in suite()["cases"].as_array().unwrap() {
        let config = config(case);
        let original = load(&case["initial_model"]);
        let token = original
            .objective()
            .split_whitespace()
            .find(|t| t.starts_with("sigmoid:"))
            .unwrap();
        let leaves: Vec<Vec<usize>> = serde_json::from_value(case["leaf_pred"].clone()).unwrap();
        for (sigmoid, expected) in [
            (config.sigmoid, &case["native_refit"]["reloaded"]),
            (
                case["serialized_sigmoid"].as_f64().unwrap(),
                &case["final_model"],
            ),
        ] {
            let text =
                original
                    .to_text()
                    .unwrap()
                    .replacen(token, &format!("sigmoid:{sigmoid:.16e}"), 1);
            let imported = Model::from_text(&text).unwrap();
            let result = imported
                .refit(&data(case, &config), &leaves, &config, 0.5)
                .unwrap();
            compare_raw(&result, case, expected);
            for (row, input) in rows(case).iter().enumerate() {
                for (a, e) in result.predict_row(input).unwrap().into_iter().zip(
                    case["native_refit"]["reloaded"]["transformed"][row]
                        .as_array()
                        .unwrap(),
                ) {
                    close(a, e.as_f64().unwrap());
                }
                for (a, e) in Model::from_text(&result.to_text().unwrap())
                    .unwrap()
                    .predict_row(input)
                    .unwrap()
                    .into_iter()
                    .zip(expected["transformed"][row].as_array().unwrap())
                {
                    close(a, e.as_f64().unwrap());
                }
            }
        }
    }
}

#[test]
fn genuinely_distinct_serialized_sigmoid_is_rejected_without_mutation() {
    for case in suite()["cases"].as_array().unwrap() {
        let config = config(case);
        let trained = Trainer::new(config.clone())
            .unwrap()
            .fit(&data(&case["training"], &config))
            .unwrap()
            .model;
        for original in [trained, load(&case["initial_model"])] {
            let before = original.to_text().unwrap();
            let mut changed = config.clone();
            changed.sigmoid = case["distinct_sigmoid"].as_f64().unwrap();
            let leaves: Vec<Vec<usize>> =
                serde_json::from_value(case["leaf_pred"].clone()).unwrap();
            assert!(
                matches!(original.refit(&data(case, &changed), &leaves, &changed, 0.5), Err(Error::InvalidInput(message))
                if message.contains("configured objective or transform differs from model"))
            );
            assert_eq!(original.to_text().unwrap(), before);
        }
    }
}
