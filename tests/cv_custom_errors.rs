use lightgbm_rust::cv::{CvCallback, CvCallbackEnv, CvOptions, FoldStrategy, cross_validate};
use lightgbm_rust::metric::MetricValue;
use lightgbm_rust::workflow::{CallbackAction, CustomMetric, PredictionKind};
use lightgbm_rust::{Config, Dataset, Error, Objective};
use std::cell::Cell;
use std::rc::Rc;

fn data(config: &Config) -> Dataset {
    Dataset::from_dense(
        &(0..24).map(f64::from).collect::<Vec<_>>(),
        24,
        1,
        (0..24).map(|i| i as f32).collect(),
        config,
    )
    .unwrap()
}
fn config(custom: bool) -> Config {
    Config {
        objective: if custom {
            Objective::Custom
        } else {
            Objective::Regression
        },
        num_iterations: 6,
        min_data_in_leaf: 1,
        min_data_in_bin: 1,
        feature_pre_filter: false,
        ..Config::default()
    }
}
fn options() -> CvOptions<'static> {
    CvOptions {
        folds: FoldStrategy::Standard {
            nfold: 3,
            shuffle: false,
            seed: 0,
        },
        ..CvOptions::default()
    }
}
struct StopOrFail {
    calls: Rc<Cell<usize>>,
    fail: bool,
    before: bool,
}
impl CvCallback for StopOrFail {
    fn before_iteration(&self) -> bool {
        self.before
    }
    fn call(&mut self, env: &CvCallbackEnv<'_>) -> lightgbm_rust::Result<CallbackAction> {
        self.calls.set(self.calls.get() + 1);
        if self.fail {
            return Err(Error::InvalidInput("callback failed".into()));
        }
        Ok(CallbackAction::Stop {
            best_iteration: env.iteration + 1,
            best_scores: vec![],
        })
    }
}
#[test]
fn stop_and_callback_errors_never_train_future_fold_rounds() {
    for fail in [false, true] {
        let config = config(true);
        let data = data(&config);
        let objective_calls = Rc::new(Cell::new(0));
        let callbacks = Rc::new(Cell::new(0));
        let mut options = options();
        let calls = objective_calls.clone();
        options.custom_objective_factory = Some(Box::new(move |_| {
            let calls = calls.clone();
            Ok(Box::new(move |scores: &[f64], data: &Dataset| {
                calls.set(calls.get() + 1);
                Ok((
                    scores
                        .iter()
                        .zip(data.labels())
                        .map(|(p, y)| (p - f64::from(*y)) as f32)
                        .collect(),
                    vec![1.0; scores.len()],
                ))
            }))
        }));
        options.callbacks.push(Box::new(StopOrFail {
            calls: callbacks.clone(),
            fail,
            before: false,
        }));
        let result = cross_validate(config.clone(), &data, options);
        assert_eq!(objective_calls.get(), 3);
        assert_eq!(callbacks.get(), 1);
        assert_eq!(result.is_err(), fail);
        if let Ok(result) = result {
            assert_eq!(result.completed_rounds, 1);
            assert_eq!(result.best_iteration, Some(1));
        }
        assert_eq!(
            data.labels(),
            &(0..24).map(|i| i as f32).collect::<Vec<_>>()
        );
    }
}
#[test]
fn invalid_derivatives_and_factory_errors_publish_no_cv_result() {
    for mode in 0..4 {
        let config = config(true);
        let data = data(&config);
        let calls = Rc::new(Cell::new(0));
        let observed = calls.clone();
        let mut options = options();
        options.custom_objective_factory = Some(Box::new(move |fold| {
            if mode == 3 && fold == 1 {
                return Err(Error::InvalidInput("factory failed".into()));
            }
            let observed = observed.clone();
            Ok(Box::new(move |scores: &[f64], _: &Dataset| {
                observed.set(observed.get() + 1);
                let n = scores.len();
                Ok(match mode {
                    0 => (vec![0.0; n - 1], vec![1.0; n]),
                    1 => (vec![f32::NAN; n], vec![1.0; n]),
                    2 => (vec![0.0; n], vec![-1.0; n]),
                    _ => (vec![0.0; n], vec![1.0; n]),
                })
            }))
        }));
        assert!(cross_validate(config, &data, options).is_err());
        assert_eq!(calls.get(), usize::from(mode != 3));
    }
}
struct InvalidMetric {
    fold: usize,
    mode: usize,
}
impl CustomMetric for InvalidMetric {
    fn evaluate(
        &mut self,
        _: &[f64],
        _: &Dataset,
        _: PredictionKind,
    ) -> lightgbm_rust::Result<Vec<MetricValue>> {
        if self.mode == 4 && self.fold == 1 {
            return Ok(vec![]);
        }
        Ok(vec![MetricValue {
            name: if self.mode == 0 {
                ""
            } else if self.mode == 2 && self.fold == 1 {
                "different"
            } else {
                "metric"
            }
            .into(),
            value: if self.mode == 1 { f64::INFINITY } else { 1.0 },
            higher_is_better: self.mode == 3 && self.fold == 1,
        }])
    }
}
#[test]
fn invalid_metrics_and_inconsistent_fold_layouts_are_checked() {
    for mode in 0..5 {
        let config = config(false);
        let data = data(&config);
        let mut options = options();
        options.custom_metric_factories.push(Box::new(move |fold| {
            Ok(Box::new(InvalidMetric { fold, mode }))
        }));
        assert!(cross_validate(config, &data, options).is_err());
    }
}
struct CountingMetric(usize);
impl CustomMetric for CountingMetric {
    fn evaluate(
        &mut self,
        _: &[f64],
        _: &Dataset,
        kind: PredictionKind,
    ) -> lightgbm_rust::Result<Vec<MetricValue>> {
        assert_eq!(kind, PredictionKind::Transformed);
        self.0 += 1;
        Ok(vec![MetricValue {
            name: "calls".into(),
            value: self.0 as f64,
            higher_is_better: true,
        }])
    }
}
#[test]
fn custom_metric_state_is_independent_for_each_fold() {
    let config = config(false);
    let data = data(&config);
    let mut options = options();
    options.evaluate_training = true;
    options
        .custom_metric_factories
        .push(Box::new(|_| Ok(Box::new(CountingMetric(0)))));
    let result = cross_validate(config, &data, options).unwrap();
    for round in result.history {
        for metric in round
            .results
            .iter()
            .filter(|m| m.evaluation.metric == "calls")
        {
            let expected = round.iteration * 2 - usize::from(metric.evaluation.dataset == "train");
            assert_eq!(metric.evaluation.value, expected as f64);
            assert_eq!(metric.std_dev, 0.0);
        }
    }
}
#[test]
fn before_iteration_stop_is_rejected_before_objective_access() {
    let config = config(false);
    let data = data(&config);
    let mut options = options();
    options.callbacks.push(Box::new(StopOrFail {
        calls: Rc::new(Cell::new(0)),
        fail: false,
        before: true,
    }));
    assert!(cross_validate(config, &data, options).is_err());
}
