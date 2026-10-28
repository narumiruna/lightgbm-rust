use lightgbm_rust::workflow::{
    CallbackAction, CallbackEnv, FitOptions, LearningRateSchedule, TrainingCallback,
};

struct AfterSchedule(LearningRateSchedule<'static>);
impl TrainingCallback for AfterSchedule {
    fn order(&self) -> i32 {
        35
    }
    fn call(&mut self, env: &CallbackEnv<'_>) -> lightgbm_rust::Result<CallbackAction> {
        self.0.call(env)
    }
}
use lightgbm_rust::{Config, Dataset, Model, Trainer};
use serde_json::Value;

struct Capture<'a> {
    steps: &'a [Value],
}
impl TrainingCallback for Capture<'_> {
    fn order(&self) -> i32 {
        25
    }
    fn call(&mut self, env: &CallbackEnv<'_>) -> lightgbm_rust::Result<CallbackAction> {
        let index = env.iteration - env.begin_iteration;
        let step = &self.steps[index];
        assert_eq!(env.iteration, step["iteration"].as_u64().unwrap() as usize);
        assert_eq!(
            env.begin_iteration,
            step["begin_iteration"].as_u64().unwrap() as usize
        );
        assert_eq!(
            env.end_iteration,
            step["end_iteration"].as_u64().unwrap() as usize
        );
        assert_eq!(env.evaluation_history.len(), index + 1);
        close(
            env.config.learning_rate,
            step["learning_rate"].as_f64().unwrap(),
        );
        for (a, b) in env
            .training_scores
            .iter()
            .zip(step["training_scores"].as_array().unwrap())
        {
            close(*a, b.as_f64().unwrap());
        }
        Ok(CallbackAction::Continue)
    }
}

fn config(c: &Value) -> Config {
    let mut config = Config::from_pairs(c["params"].as_object().unwrap().iter().map(|(k, v)| {
        let value = if let Some(values) = v.as_array() {
            values
                .iter()
                .map(|v| v.as_str().unwrap().to_owned())
                .collect::<Vec<_>>()
                .join(",")
        } else {
            v.as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| v.to_string())
        };
        (k.clone(), value)
    }))
    .unwrap();
    config.num_iterations = c["learning_rates"].as_array().unwrap().len();
    config
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-12 + 1e-10 * b.abs(), "{a} != {b}");
}
#[test]
fn sampled_tree_rate_reset_uses_the_existing_learner() {
    let config = Config {
        feature_fraction: 0.5,
        min_data_in_leaf: 1,
        min_data_in_bin: 1,
        feature_pre_filter: false,
        num_iterations: 2,
        ..Config::default()
    };
    let data = Dataset::from_dense(
        &(0..24).map(f64::from).collect::<Vec<_>>(),
        12,
        2,
        (0..12).map(|i| i as f32).collect(),
        &config,
    )
    .unwrap();
    let result = Trainer::new(config).unwrap().fit_with_options(
        &data,
        &[],
        FitOptions {
            callbacks: vec![Box::new(LearningRateSchedule::values(vec![0.2, 0.3]))],
            ..FitOptions::default()
        },
    );
    let result = result.unwrap();
    assert_eq!(result.model.num_iterations(), 2);
    assert!(result.training_scores.iter().all(|score| score.is_finite()));
}

#[test]
fn native_dart_resets_saved_prefix_and_goss_reset_state() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/workflow_extended.json")).unwrap();
    for c in fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["kind"] == "train")
    {
        let config = config(c);
        let rows: Vec<Vec<f64>> = serde_json::from_value(c["rows"].clone()).unwrap();
        let labels: Vec<f32> = serde_json::from_value(c["labels"].clone()).unwrap();
        let flat = rows.iter().flatten().copied().collect::<Vec<_>>();
        let mut data = Dataset::from_dense(&flat, rows.len(), 2, labels.clone(), &config).unwrap();
        let initial = c["initial_model"]["model_file"].as_str().map(|path| {
            Model::load(format!(
                "{}/tests/fixtures/{path}",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap()
        });
        if let Some(model) = &initial {
            data = data
                .with_initial_scores(
                    rows.iter()
                        .flat_map(|r| model.predict_raw_row(r).unwrap())
                        .collect(),
                    1,
                )
                .unwrap();
        }
        let reverse_rows = rows.iter().rev().flatten().copied().collect::<Vec<_>>();
        let mut valid = Dataset::validation_from_dense(
            &data,
            &reverse_rows,
            rows.len(),
            2,
            labels.into_iter().rev().collect(),
        )
        .unwrap();
        if let Some(model) = &initial {
            valid = valid
                .with_initial_scores(
                    rows.iter()
                        .rev()
                        .flat_map(|r| model.predict_raw_row(r).unwrap())
                        .collect(),
                    1,
                )
                .unwrap();
        }
        let schedule = LearningRateSchedule::values(
            serde_json::from_value(c["learning_rates"].clone()).unwrap(),
        );
        let schedule: Box<dyn TrainingCallback> = if c["schedule_after"].as_bool().unwrap() {
            Box::new(AfterSchedule(schedule))
        } else {
            Box::new(schedule)
        };
        let result = Trainer::new(config)
            .unwrap()
            .fit_with_options(
                &data,
                &[("valid", &valid)],
                FitOptions {
                    initial_model: initial.as_ref(),
                    evaluate_training: true,
                    training_name: "train".into(),
                    callbacks: vec![
                        schedule,
                        Box::new(Capture {
                            steps: c["steps"].as_array().unwrap(),
                        }),
                    ],
                    ..FitOptions::default()
                },
            )
            .unwrap();
        for (actual, expected) in result
            .evaluation_history
            .iter()
            .zip(c["steps"].as_array().unwrap())
        {
            for (a, b) in actual
                .results
                .iter()
                .zip(expected["evaluation"].as_array().unwrap())
            {
                assert_eq!(a.metric, b["metric_name"].as_str().unwrap());
                close(a.value, b["metric_value"].as_f64().unwrap());
            }
        }
        let reference = Model::load(format!(
            "{}/tests/fixtures/{}",
            env!("CARGO_MANIFEST_DIR"),
            c["final_model"]["model_file"].as_str().unwrap()
        ))
        .unwrap();
        assert_eq!(result.model.trees().len(), reference.trees().len());
        for (a, b) in result.model.trees().iter().zip(reference.trees()) {
            assert_eq!(a.split_features(), b.split_features());
            assert_eq!(a.thresholds(), b.thresholds());
            for (a, b) in a.leaf_values().iter().zip(b.leaf_values()) {
                close(*a, *b);
            }
        }
        if let Some(initial) = initial {
            for (a, b) in initial.trees().iter().zip(result.model.trees()) {
                assert_eq!(
                    a.leaf_values(),
                    b.leaf_values(),
                    "saved prefix must never be dropped"
                );
            }
        }
        for (row, expected) in rows.iter().zip(c["final_model"]["raw"].as_array().unwrap()) {
            close(
                result.model.predict_raw_row(row).unwrap()[0],
                expected[0].as_f64().unwrap(),
            );
        }
    }
}
