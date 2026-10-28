use lightgbm_rust::random::Random;
use lightgbm_rust::workflow::{CallbackAction, CallbackEnv, FitOptions, TrainingCallback};
use lightgbm_rust::{Config, Dataset, Model, Trainer};
use serde_json::Value;

fn cases() -> Vec<Value> {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/review_dart.json")).unwrap();
    assert_eq!(fixture["schema_version"], 1);
    assert_eq!(
        fixture["upstream_commit"],
        "5d84f3f222b41409a6983d2ac90513ad9702fa9e"
    );
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 13);
    cases.clone()
}
fn config(case: &Value, rounds: usize) -> Config {
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
    config.num_iterations = rounds;
    config
}
fn load(value: &Value) -> Model {
    Model::load(format!(
        "{}/tests/fixtures/{}",
        env!("CARGO_MANIFEST_DIR"),
        value["model_file"].as_str().unwrap()
    ))
    .unwrap()
}
fn close(a: f64, b: f64) {
    assert!(a.is_finite() && b.is_finite());
    assert!((a - b).abs() <= 1e-12 + 1e-10 * b.abs(), "{a} != {b}");
}
fn margins(model: &Model, reverse: bool) -> Vec<f64> {
    let rows = if reverse {
        vec![3., 2., 1., 0.]
    } else {
        vec![0., 1., 2., 3.]
    };
    let predictions: Vec<_> = rows
        .iter()
        .map(|x| model.predict_raw_row(&[*x]).unwrap())
        .collect();
    (0..model.num_classes())
        .flat_map(|class| predictions.iter().map(move |p| p[class]))
        .collect()
}
fn datasets(case: &Value, config: &Config, initial: Option<&Model>) -> (Dataset, Dataset) {
    let labels: Vec<f32> = serde_json::from_value(case["labels"].clone()).unwrap();
    let mut data = Dataset::from_dense(&[0., 1., 2., 3.], 4, 1, labels.clone(), config).unwrap();
    if let Some(model) = initial {
        data = data
            .with_initial_scores(margins(model, false), config.num_class)
            .unwrap();
    }
    let mut valid = Dataset::validation_from_dense(
        &data,
        &[3., 2., 1., 0.],
        4,
        1,
        labels.into_iter().rev().collect(),
    )
    .unwrap();
    if let Some(model) = initial {
        valid = valid
            .with_initial_scores(margins(model, true), config.num_class)
            .unwrap();
    }
    (data, valid)
}
fn same_model(actual: &Model, expected: &Model) {
    assert_eq!(actual.num_iterations(), expected.num_iterations());
    assert_eq!(actual.trees().len(), expected.trees().len());
    for (a, b) in actual.trees().iter().zip(expected.trees()) {
        assert_eq!(a.split_features(), b.split_features());
        assert_eq!(a.thresholds(), b.thresholds());
        close(a.shrinkage(), b.shrinkage());
        assert_eq!(a.leaf_values().len(), b.leaf_values().len());
        for (a, b) in a.leaf_values().iter().zip(b.leaf_values()) {
            close(*a, *b);
        }
    }
    for (a, b) in margins(actual, false)
        .into_iter()
        .zip(margins(expected, false))
    {
        close(a, b);
    }
    let reloaded = Model::from_text(&actual.to_text().unwrap()).unwrap();
    for (a, b) in margins(actual, false)
        .into_iter()
        .zip(margins(&reloaded, false))
    {
        close(a, b);
    }
}
fn expected_at(case: &Value, attempt: usize) -> Model {
    // Only the safe no-drop control completes a second native round. Failed
    // Rust rounds use the native SUCCESSFUL prefix, never negative native text.
    if attempt > 0 && case["name"] == "review_dart_safe_no_drop" {
        load(&case["native_final_model"])
    } else {
        load(&case["last_valid_model"])
    }
}
struct Capture<'a> {
    case: &'a Value,
    calls: &'a mut usize,
}
impl TrainingCallback for Capture<'_> {
    fn order(&self) -> i32 {
        25
    }
    fn call(&mut self, env: &CallbackEnv<'_>) -> lightgbm_rust::Result<CallbackAction> {
        let attempt = env.iteration - env.begin_iteration;
        assert_eq!(env.evaluation_history.len(), attempt + 1);
        if attempt < 2 {
            let expected = expected_at(self.case, attempt);
            assert_eq!(
                env.model_current_iteration,
                expected.num_iterations(),
                "{} attempt {attempt}",
                self.case["name"]
            );
            assert_eq!(env.model_num_trees, expected.trees().len());
            for (a, b) in env.training_scores.iter().zip(margins(&expected, false)) {
                close(*a, b);
            }
            let native_step = if attempt > 0 && self.case["name"] == "review_dart_safe_no_drop" {
                1
            } else {
                0
            };
            let native = self.case["native_steps"][native_step]["evaluation"]
                .as_array()
                .unwrap();
            let actual = env.evaluation.unwrap();
            assert_eq!(actual.len(), native.len());
            for (a, b) in actual.iter().zip(native) {
                assert_eq!(a.dataset, b["dataset_name"].as_str().unwrap());
                assert_eq!(a.metric, b["metric_name"].as_str().unwrap());
                close(a.value, b["metric_value"].as_f64().unwrap());
            }
        }
        assert!(env.training_scores.iter().all(|s| s.is_finite()));
        assert!(env.evaluation.unwrap().iter().all(|m| m.value.is_finite()));
        *self.calls += 1;
        Ok(CallbackAction::Continue)
    }
}

#[test]
fn four_row_seed_four_first_bag_splits_second_bag_is_empty() {
    for case in cases() {
        let mut rng = Random::new(case["params"]["bagging_seed"].as_u64().unwrap() as u32);
        for expected in case["bag_rows"].as_array().unwrap() {
            let actual: Vec<_> = (0..4).filter(|_| rng.next_float() < 0.5).collect();
            assert_eq!(
                actual,
                serde_json::from_value::<Vec<usize>>(expected.clone()).unwrap()
            );
        }
    }
}

#[test]
fn core_fit_returns_last_completed_valid_ensemble() {
    for case in cases().iter().filter(|c| c["initial_model"].is_null()) {
        let config = config(case, 2);
        let (data, valid) = datasets(case, &config, None);
        let result = Trainer::new(config)
            .unwrap()
            .fit_with_validation(&data, &[("valid", &valid)])
            .unwrap();
        let expected = expected_at(case, 1);
        same_model(&result.model, &expected);
        for (a, b) in result.training_scores.iter().zip(margins(&expected, false)) {
            close(*a, b);
        }
        let expected_history = if case["name"] == "review_dart_safe_no_drop" {
            2
        } else {
            1
        };
        assert_eq!(result.evaluation_history.len(), expected_history);
    }
}

#[test]
fn workflow_restores_buffers_prefix_and_continues_attempts() {
    for case in cases() {
        for rounds in [2, case["rust_attempts"].as_u64().unwrap() as usize] {
            let config = config(&case, rounds);
            let initial = (!case["initial_model"].is_null()).then(|| load(&case["initial_model"]));
            let (data, valid) = datasets(&case, &config, initial.as_ref());
            let mut calls = 0;
            let mut metric_buffers = Vec::new();
            let metric =
                |scores: &[f64], _: &Dataset, _: lightgbm_rust::workflow::PredictionKind| {
                    metric_buffers.push(scores.to_vec());
                    Ok(Vec::new())
                };
            let result = Trainer::new(config)
                .unwrap()
                .fit_with_options(
                    &data,
                    &[("valid", &valid)],
                    FitOptions {
                        initial_model: initial.as_ref(),
                        evaluate_training: true,
                        callbacks: vec![Box::new(Capture {
                            case: &case,
                            calls: &mut calls,
                        })],
                        custom_metrics: vec![Box::new(metric)],
                        ..FitOptions::default()
                    },
                )
                .unwrap();
            assert_eq!(calls, rounds);
            assert_eq!(metric_buffers.len(), 2 * rounds);
            for (buffer, rows) in metric_buffers[2 * rounds - 2..]
                .iter()
                .zip([[0., 1., 2., 3.], [3., 2., 1., 0.]])
            {
                for class in 0..result.model.num_classes() {
                    for (row, x) in rows.iter().enumerate() {
                        close(
                            buffer[class * 4 + row],
                            result.model.predict_row(&[*x]).unwrap()[class],
                        );
                    }
                }
            }
            assert_eq!(result.evaluation_history.len(), rounds);
            if rounds == 2 {
                same_model(&result.model, &expected_at(&case, 1));
            }
            if let Some(initial) = initial {
                for (a, b) in result.model.trees().iter().zip(initial.trees()) {
                    assert_eq!(a.leaf_values(), b.leaf_values());
                    assert_eq!(a.shrinkage(), b.shrinkage());
                }
                same_model(
                    &result.model.truncate(initial.num_iterations()).unwrap(),
                    &initial,
                );
            }
            for (a, b) in result
                .training_scores
                .iter()
                .zip(margins(&result.model, false))
            {
                close(*a, b);
            }
            assert!(
                result
                    .model
                    .trees()
                    .iter()
                    .all(|t| t.shrinkage().is_finite() && t.shrinkage() >= 0.0)
            );
            Model::from_text(&result.model.to_text().unwrap()).unwrap();
            if case["name"].as_str().unwrap().contains("empty_") && rounds > 2 {
                assert!(
                    result.model.num_iterations() > 1,
                    "later nonempty bags must still train"
                );
            }
        }
    }
}

#[test]
fn first_continuation_attempt_without_a_split_keeps_only_the_saved_prefix() {
    let case = cases()
        .into_iter()
        .find(|c| c["name"] == "review_dart_continue_weighted")
        .unwrap();
    let initial = load(&case["initial_model"]);
    let mut config = config(&case, 4);
    config.min_gain_to_split = 1e20;
    let (data, valid) = datasets(&case, &config, Some(&initial));
    let result = Trainer::new(config)
        .unwrap()
        .fit_with_options(
            &data,
            &[("valid", &valid)],
            FitOptions {
                initial_model: Some(&initial),
                evaluate_training: true,
                ..FitOptions::default()
            },
        )
        .unwrap();
    same_model(&result.model, &initial);
    assert_eq!(result.evaluation_history.len(), 4);
    for entry in &result.evaluation_history {
        assert_eq!(entry.results, result.evaluation_history[0].results);
    }
    for (a, b) in result.training_scores.iter().zip(margins(&initial, false)) {
        close(*a, b);
    }
}

#[test]
fn safe_native_no_split_evidence_is_not_failed_round_parity() {
    for case in cases()
        .iter()
        .filter(|c| c["native_attempts"] == 2 && c["params"]["skip_drop"] == 0.0)
    {
        let step = &case["native_steps"][1];
        assert_eq!(step["finished"], true);
        assert!(
            step["shrinkage"]
                .as_array()
                .unwrap()
                .iter()
                .any(|s| s.as_f64().unwrap() < 0.0)
        );
        assert!(case["native_final_model"].is_null());
        assert!(Model::from_text(case["native_failed_model"]["text"].as_str().unwrap()).is_err());
        let completed = &case["native_steps"][0];
        assert_eq!(step["num_trees"], completed["num_trees"]);
        assert_eq!(step["evaluation"][1], completed["evaluation"][1]);
        assert_ne!(
            step["training_predictions"],
            completed["training_predictions"]
        );
    }
}
