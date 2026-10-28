use lightgbm_rust::workflow::{CallbackAction, CallbackEnv, FitOptions, TrainingCallback};
use lightgbm_rust::{Config, Dataset, Model, Trainer};
use serde_json::Value;

struct Capture<'a> {
    steps: &'a [Value],
    index: usize,
    compare_buffer: bool,
}
impl TrainingCallback for Capture<'_> {
    fn before_iteration(&self) -> bool {
        true
    }
    fn call(&mut self, env: &CallbackEnv<'_>) -> lightgbm_rust::Result<CallbackAction> {
        let step = &self.steps[self.index];
        self.index += 1;
        assert_eq!(env.iteration, step["iteration"].as_u64().unwrap() as usize);
        assert_eq!(
            env.begin_iteration,
            step["begin_iteration"].as_u64().unwrap() as usize
        );
        assert_eq!(
            env.end_iteration,
            step["end_iteration"].as_u64().unwrap() as usize
        );
        assert_eq!(
            env.model_current_iteration,
            step["model"]["current_iteration"].as_u64().unwrap() as usize
        );
        // Native DART _inner_predict is a mutating dropout probe. The Rust
        // environment is read-only; its non-observational buffer is not an oracle.
        if self.compare_buffer {
            for (a, b) in env
                .training_scores
                .iter()
                .zip(step["training_score_buffer"].as_array().unwrap())
            {
                close(*a, b.as_f64().unwrap());
            }
        }
        Ok(CallbackAction::Continue)
    }
}
#[test]
fn stored_model_continuation_restarts_rng_and_requires_raw_margins() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/prediction_operations.json")).unwrap();
    for c in fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["operation"] == "continuation")
    {
        let name = c["name"].as_str().unwrap();
        let mut config =
            Config::from_pairs(c["params"].as_object().unwrap().iter().map(|(k, v)| {
                (
                    k.clone(),
                    v.as_str()
                        .map(str::to_owned)
                        .unwrap_or_else(|| v.to_string()),
                )
            }))
            .unwrap();
        config.num_iterations = c["num_boost_round"].as_u64().unwrap() as usize;
        let rows: Vec<Vec<f64>> = serde_json::from_value(c["rows"].clone()).unwrap();
        let flat = rows.iter().flatten().copied().collect::<Vec<_>>();
        let data = Dataset::from_dense(
            &flat,
            rows.len(),
            2,
            serde_json::from_value(c["labels"].clone()).unwrap(),
            &config,
        )
        .unwrap();
        let path = format!(
            "{}/tests/fixtures/{}",
            env!("CARGO_MANIFEST_DIR"),
            c["initial_model"]["model_file"].as_str().unwrap()
        );
        let model = Model::load(path).unwrap();
        assert!(
            Trainer::new(config.clone())
                .unwrap()
                .fit_with_options(
                    &data,
                    &[],
                    FitOptions {
                        initial_model: Some(&model),
                        ..FitOptions::default()
                    }
                )
                .is_err(),
            "missing raw margins: {name}"
        );
        let margins = rows
            .iter()
            .map(|r| model.predict_raw_row(r).unwrap()[0])
            .collect();
        let data = data.with_initial_scores(margins, 1).unwrap();
        let result = Trainer::new(config).unwrap().fit_with_options(
            &data,
            &[],
            FitOptions {
                initial_model: Some(&model),
                callbacks: vec![Box::new(Capture {
                    steps: c["steps"].as_array().unwrap(),
                    index: 0,
                    compare_buffer: name != "continue_dart",
                })],
                ..FitOptions::default()
            },
        );
        if c["error_type"].is_string() {
            assert!(result.is_err(), "{name}");
            continue;
        }
        let result = result.unwrap();
        // The separate non-mutating DART continuation fixture checks scores and
        // structure. This older capture changed native RNG state before training.
        if name == "continue_dart" {
            continue;
        }
        for (i, row) in rows.iter().enumerate() {
            close(
                result.model.predict_raw_row(row).unwrap()[0],
                c["final_model"]["raw"][i][0].as_f64().unwrap(),
            );
        }
    }
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-12 + 1e-10 * b.abs(), "{a} != {b}");
}
