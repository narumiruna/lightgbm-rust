use lightgbm_rust::workflow::{EarlyStopping, FitOptions, LearningRateSchedule};
use lightgbm_rust::{Config, Dataset, Model, Trainer};
use serde_json::Value;

fn assert_structure(actual: &Model, expected: &Model) {
    assert_eq!(actual.trees().len(), expected.trees().len());
    for (a, b) in actual.trees().iter().zip(expected.trees()) {
        assert_eq!(a.num_leaves(), b.num_leaves());
        assert_eq!(a.split_features(), b.split_features());
        assert_eq!(a.thresholds(), b.thresholds());
        assert_eq!(a.left_children(), b.left_children());
        assert_eq!(a.right_children(), b.right_children());
        assert_eq!(a.decision_types(), b.decision_types());
        assert_eq!(a.cat_boundaries(), b.cat_boundaries());
        assert_eq!(a.cat_threshold(), b.cat_threshold());
        assert_eq!(a.leaf_counts(), b.leaf_counts());
        for (a, b) in a.leaf_values().iter().zip(b.leaf_values()) {
            close(*a, *b);
        }
    }
}

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/workflow.json")).unwrap()
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-12 + 1e-10 * b.abs(), "{a} != {b}");
}

#[test]
fn native_python_workflows() {
    let fixture = fixture();
    assert_eq!(
        fixture["provenance"]["loaded_library_sha256"],
        "9ed1d46d2d60bac2966fd03bd81e7d192025804f6a3e3ef24c3aa483d0ec7c1e"
    );
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        if name.starts_with("custom_") || name == "invalid_custom_length" {
            continue;
        }
        let pairs: Vec<(String, String)> = case["params"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    if let Some(a) = v.as_array() {
                        a.iter()
                            .map(|x| {
                                x.as_str()
                                    .map(str::to_owned)
                                    .unwrap_or_else(|| x.to_string())
                            })
                            .collect::<Vec<_>>()
                            .join(",")
                    } else {
                        v.as_str()
                            .map(str::to_owned)
                            .unwrap_or_else(|| v.to_string())
                    },
                )
            })
            .collect();
        let mut config = Config::from_pairs(pairs).unwrap();
        config.num_iterations = case["num_boost_round"].as_u64().unwrap() as usize;
        let rows: Vec<Vec<f64>> = serde_json::from_value(case["rows"].clone()).unwrap();
        let flat: Vec<_> = rows.iter().flatten().copied().collect();
        let labels: Vec<f32> = serde_json::from_value(case["labels"].clone()).unwrap();
        let data = Dataset::from_dense(&flat, rows.len(), rows[0].len(), labels, &config).unwrap();
        let mut valid = Vec::new();
        let mut options = FitOptions::default();
        for set in case["valid_sets"].as_array().unwrap() {
            let set_name = set["name"].as_str().unwrap();
            if set["is_training"].as_bool().unwrap() {
                options.evaluate_training = true;
                options.training_name = set_name.to_owned();
            } else {
                valid.push((
                    set_name,
                    Dataset::validation_from_dense(
                        &data,
                        &flat,
                        rows.len(),
                        rows[0].len(),
                        serde_json::from_value(set["labels"].clone()).unwrap(),
                    )
                    .unwrap(),
                ));
            }
        }
        if let Some(schedule) = case["learning_rate_schedule"].as_array() {
            options
                .callbacks
                .push(Box::new(LearningRateSchedule::values(
                    schedule.iter().map(|v| v.as_f64().unwrap()).collect(),
                )));
        }
        if case["stopping"].is_object() {
            let s = &case["stopping"];
            let delta = if let Some(a) = s["min_delta"].as_array() {
                a.iter().map(|v| v.as_f64().unwrap()).collect()
            } else {
                vec![s["min_delta"].as_f64().unwrap()]
            };
            options.callbacks.push(Box::new(
                EarlyStopping::new(s["stopping_rounds"].as_u64().unwrap() as usize)
                    .first_metric_only(s["first_metric_only"].as_bool().unwrap())
                    .min_delta(delta),
            ));
        }
        let refs: Vec<_> = valid.iter().map(|(n, d)| (*n, d)).collect();
        let fitted = Trainer::new(config)
            .unwrap()
            .fit_with_options(&data, &refs, options);
        if case["error_type"].is_string() {
            assert!(fitted.is_err(), "{name}");
            continue;
        }
        let fitted = fitted.unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(
            fitted.best_iteration,
            case["best_iteration"].as_u64().unwrap() as usize,
            "{name}"
        );
        assert_eq!(
            fitted.model.num_iterations(),
            case["live_model"]["current_iteration"].as_u64().unwrap() as usize,
            "{name}"
        );
        let reference = Model::load(format!(
            "{}/tests/fixtures/{}",
            env!("CARGO_MANIFEST_DIR"),
            case["full_model"]["model_file"].as_str().unwrap()
        ))
        .unwrap();
        assert_structure(&fitted.model, &reference);
        for (i, row) in rows.iter().enumerate() {
            for (a, b) in fitted
                .model
                .predict_raw_row(row)
                .unwrap()
                .iter()
                .zip(case["live_model"]["raw"][i].as_array().unwrap())
            {
                close(*a, b.as_f64().unwrap());
            }
        }
        for eval in &fitted.evaluation_history {
            for value in &eval.results {
                close(
                    value.value,
                    case["history"][&value.dataset][&value.metric][eval.iteration - 1]
                        .as_f64()
                        .unwrap(),
                );
            }
        }
        let expected_len = case["history"]
            .as_object()
            .unwrap()
            .values()
            .flat_map(|ds| ds.as_object().unwrap().values())
            .map(|v| v.as_array().unwrap().len())
            .max()
            .unwrap_or(7);
        assert_eq!(fitted.evaluation_history.len(), expected_len, "{name}");
        for value in &fitted.best_scores {
            close(
                value.value,
                case["best_score"][&value.dataset][&value.metric]
                    .as_f64()
                    .unwrap(),
            );
        }
    }
}
