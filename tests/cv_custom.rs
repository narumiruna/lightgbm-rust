use lightgbm_rust::cv::{CvCallback, CvCallbackEnv, CvOptions, Fold, FoldStrategy, cross_validate};
use lightgbm_rust::metric::MetricValue;
use lightgbm_rust::workflow::CallbackAction;
use lightgbm_rust::workflow::{
    CustomMetric, CustomObjective, EarlyStopping, LearningRateSchedule, PredictionKind,
};
use std::cell::Cell;
use std::rc::Rc;

struct AfterCvSchedule(LearningRateSchedule<'static>);
impl CvCallback for AfterCvSchedule {
    fn order(&self) -> i32 {
        35
    }
    fn call(&mut self, env: &CvCallbackEnv<'_>) -> lightgbm_rust::Result<CallbackAction> {
        CvCallback::call(&mut self.0, env)
    }
}
use lightgbm_rust::{Config, Dataset, Model};
use serde_json::Value;

struct L2;
impl CustomObjective for L2 {
    fn gradients(
        &mut self,
        scores: &[f64],
        data: &Dataset,
    ) -> lightgbm_rust::Result<(Vec<f32>, Vec<f32>)> {
        Ok((
            scores
                .iter()
                .zip(data.labels())
                .map(|(p, y)| (p - f64::from(*y)) as f32)
                .collect(),
            vec![1.0; scores.len()],
        ))
    }
}
impl CustomMetric for L2 {
    fn evaluate(
        &mut self,
        scores: &[f64],
        data: &Dataset,
        kind: PredictionKind,
    ) -> lightgbm_rust::Result<Vec<MetricValue>> {
        assert_eq!(kind, PredictionKind::Raw);
        Ok(vec![MetricValue {
            name: "custom_l2".into(),
            value: scores
                .iter()
                .zip(data.labels())
                .map(|(p, y)| (p - f64::from(*y)).powi(2))
                .sum::<f64>()
                / scores.len() as f64,
            higher_is_better: false,
        }])
    }
}
struct RecordedL2<'a> {
    fold: usize,
    round: usize,
    expected: &'a [Value],
    calls: Rc<Cell<usize>>,
}
impl CustomObjective for RecordedL2<'_> {
    fn gradients(
        &mut self,
        scores: &[f64],
        data: &Dataset,
    ) -> lightgbm_rust::Result<(Vec<f32>, Vec<f32>)> {
        let expected = &self.expected[self.round * 3 + self.fold];
        assert_eq!(scores.len(), expected["rows"].as_u64().unwrap() as usize);
        for (a, b) in scores
            .iter()
            .zip(expected["predictions"].as_array().unwrap())
        {
            close(*a, b.as_f64().unwrap());
        }
        self.round += 1;
        self.calls.set(self.calls.get() + 1);
        L2.gradients(scores, data)
    }
}
struct Observe<'a> {
    phase: &'static str,
    trace: &'a [Value],
    steps: &'a [Value],
    index: Rc<Cell<usize>>,
}
impl CvCallback for Observe<'_> {
    fn order(&self) -> i32 {
        if self.phase == "after_record" { 25 } else { 15 }
    }
    fn before_iteration(&self) -> bool {
        self.phase == "before"
    }
    fn call(&mut self, env: &CvCallbackEnv<'_>) -> lightgbm_rust::Result<CallbackAction> {
        let event = &self.trace[self.index.get()];
        self.index.set(self.index.get() + 1);
        assert_eq!(self.phase, event["phase"].as_str().unwrap());
        assert_eq!(env.iteration, event["iteration"].as_u64().unwrap() as usize);
        assert_eq!(
            env.evaluation_history.len(),
            event["history_length"].as_u64().unwrap() as usize
        );
        close(
            env.config.learning_rate,
            self.steps[env.iteration]["learning_rate"].as_f64().unwrap(),
        );
        let count = env.iteration + usize::from(self.phase != "before");
        assert_eq!(env.fold_current_iterations, &[count; 3]);
        assert_eq!(env.fold_num_trees, &[count; 3]);
        assert_eq!(env.evaluation.is_none(), self.phase == "before");
        if let Some(results) = env.evaluation {
            for (a, b) in results
                .iter()
                .zip(self.steps[env.iteration]["evaluation"].as_array().unwrap())
            {
                close(a.evaluation.value, b["metric_value"].as_f64().unwrap());
                close(a.std_dev, b["metric_std_dev"].as_f64().unwrap());
            }
        }
        Ok(CallbackAction::Continue)
    }
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-12 + 1e-10 * b.abs(), "{a} != {b}");
}
#[test]
fn native_custom_cv_schedules_and_common_stopping() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/workflow_extended.json")).unwrap();
    for c in fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["kind"] == "cv")
    {
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
        config.num_iterations = c["learning_rates"].as_array().unwrap().len();
        let rows: Vec<Vec<f64>> = serde_json::from_value(c["rows"].clone()).unwrap();
        let data = Dataset::from_dense(
            &rows.iter().flatten().copied().collect::<Vec<_>>(),
            rows.len(),
            2,
            serde_json::from_value(c["labels"].clone()).unwrap(),
            &config,
        )
        .unwrap();
        let folds = c["folds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| Fold {
                train_indices: serde_json::from_value(f["train_indices"].clone()).unwrap(),
                valid_indices: serde_json::from_value(f["valid_indices"].clone()).unwrap(),
            })
            .collect();
        let mut options = CvOptions {
            folds: FoldStrategy::Explicit(folds),
            evaluate_training: true,
            learning_rates: Some(serde_json::from_value(c["learning_rates"].clone()).unwrap()),
            early_stopping: c["stopping"]
                .as_bool()
                .unwrap()
                .then(|| EarlyStopping::new(2).min_delta(vec![1e6])),
            ..CvOptions::default()
        };
        if c["schedule_after"].as_bool().unwrap() {
            let rates = options.learning_rates.take().unwrap();
            options
                .callbacks
                .push(Box::new(AfterCvSchedule(LearningRateSchedule::values(
                    rates,
                ))));
        }
        let events = Rc::new(Cell::new(0));
        let calls = Rc::new(Cell::new(0));
        for phase in ["before", "after_early", "after_record"] {
            options.callbacks.push(Box::new(Observe {
                phase,
                trace: c["trace"].as_array().unwrap(),
                steps: c["steps"].as_array().unwrap(),
                index: events.clone(),
            }));
        }
        if c["custom"].as_bool().unwrap() {
            let expected = c["objective_calls"].as_array().unwrap();
            let calls = calls.clone();
            options.custom_objective_factory = Some(Box::new(move |fold| {
                Ok(Box::new(RecordedL2 {
                    fold,
                    round: 0,
                    expected,
                    calls: calls.clone(),
                }))
            }));
            options
                .custom_metric_factories
                .push(Box::new(|_| Ok(Box::new(L2))));
        }
        let result = cross_validate(config, &data, options).unwrap();
        assert_eq!(events.get(), c["trace"].as_array().unwrap().len());
        assert_eq!(calls.get(), c["objective_calls"].as_array().unwrap().len());
        assert_eq!(
            result.completed_rounds,
            c["steps"].as_array().unwrap().len()
        );
        assert_eq!(
            result.best_iteration.map(|i| i as i64).unwrap_or(-1),
            c["best_iteration"].as_i64().unwrap()
        );
        for round in &result.history {
            for (actual, expected) in round.results.iter().zip(
                c["steps"][round.iteration - 1]["evaluation"]
                    .as_array()
                    .unwrap(),
            ) {
                close(
                    actual.evaluation.value,
                    expected["metric_value"].as_f64().unwrap(),
                );
                close(actual.std_dev, expected["metric_std_dev"].as_f64().unwrap());
            }
        }
        for (fold, model) in result.models.iter().enumerate() {
            let reference = Model::load(format!(
                "{}/tests/fixtures/{}",
                env!("CARGO_MANIFEST_DIR"),
                c["models"][fold]["model_file"].as_str().unwrap()
            ))
            .unwrap();
            assert_eq!(model.trees().len(), reference.trees().len());
            for (a, b) in model.trees().iter().zip(reference.trees()) {
                assert_eq!(a.thresholds(), b.thresholds());
                for (a, b) in a.leaf_values().iter().zip(b.leaf_values()) {
                    close(*a, *b);
                }
            }
            for (row, raw) in rows
                .iter()
                .zip(c["models"][fold]["raw"].as_array().unwrap())
            {
                close(
                    model.predict_raw_row(row).unwrap()[0],
                    raw[0].as_f64().unwrap(),
                );
            }
        }
    }
}
