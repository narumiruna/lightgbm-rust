use super::*;
use crate::workflow::TrainingCallback;

fn metric(dataset: &str, value: f64) -> CvMetric {
    CvMetric {
        evaluation: Evaluation {
            dataset: dataset.into(),
            metric: "l2".into(),
            value,
            higher_is_better: false,
        },
        std_dev: 0.125,
    }
}

#[test]
fn builtin_bridge_does_not_copy_accumulated_history() {
    struct Probe;
    impl TrainingCallback for Probe {
        fn call(&mut self, env: &CallbackEnv<'_>) -> Result<CallbackAction> {
            assert!(env.evaluation_history.is_empty());
            assert_eq!(env.iteration, 9_999);
            assert_eq!(env.model_current_iteration, 7);
            assert_eq!(env.model_num_trees, 42);
            assert_eq!(env.best_iteration, 3);
            assert_eq!(env.training_name, "train");
            assert!(env.training_scores.is_empty());
            let values = env.evaluation.unwrap();
            assert_eq!(values.len(), 1);
            assert_eq!(values[0].dataset, "valid");
            assert_eq!(values[0].metric, "l2");
            assert_eq!(values[0].value, 0.5);
            Ok(CallbackAction::Continue)
        }
    }
    let config = Config::default();
    let values = [metric("valid", 0.5)];
    let history: Vec<_> = (0..10_000)
        .map(|iteration| CvIteration {
            iteration: iteration + 1,
            results: values.to_vec(),
        })
        .collect();
    let env = CvCallbackEnv {
        config: &config,
        iteration: 9_999,
        begin_iteration: 0,
        end_iteration: 10_000,
        fold_current_iterations: &[7, 7],
        fold_num_trees: &[21, 21],
        best_iteration: 3,
        evaluation: Some(&values),
        evaluation_history: &history,
    };
    assert_eq!(
        call_training_callback(&mut Probe, &env).unwrap(),
        CallbackAction::Continue
    );
    assert_eq!(history.len(), 10_000);
}

#[test]
fn builtin_cv_actions_match_training_callbacks_with_full_history() {
    let config = Config::default();
    let mut cv_stopping = EarlyStopping::new(2);
    let mut training_stopping = EarlyStopping::new(2);
    let rates = vec![0.1, 0.2, 0.3, 0.4];
    let mut cv_schedule = LearningRateSchedule::values(rates.clone());
    let mut training_schedule = LearningRateSchedule::values(rates);
    let mut cv_history = Vec::new();
    let mut training_history = Vec::new();
    for (iteration, value) in [0.5, 0.4, 0.45, 0.5].into_iter().enumerate() {
        let values = [metric("train", 0.1), metric("valid", value)];
        let evaluations: Vec<_> = values.iter().map(|v| v.evaluation.clone()).collect();
        cv_history.push(CvIteration {
            iteration: iteration + 1,
            results: values.to_vec(),
        });
        training_history.push(crate::trainer::IterationEvaluation {
            iteration: iteration + 1,
            results: evaluations.clone(),
        });
        let cv_env = CvCallbackEnv {
            config: &config,
            iteration,
            begin_iteration: 0,
            end_iteration: 4,
            fold_current_iterations: &[iteration + 1; 2],
            fold_num_trees: &[iteration + 1; 2],
            best_iteration: 0,
            evaluation: Some(&values),
            evaluation_history: &cv_history,
        };
        let training_env = CallbackEnv {
            config: &config,
            iteration,
            begin_iteration: 0,
            end_iteration: 4,
            model_current_iteration: iteration + 1,
            model_num_trees: (iteration + 1) * 2,
            best_iteration: 0,
            evaluation: Some(&evaluations),
            evaluation_history: &training_history,
            training_scores: &[],
            training_name: "train",
        };
        assert_eq!(
            CvCallback::call(&mut cv_schedule, &cv_env).unwrap(),
            TrainingCallback::call(&mut training_schedule, &training_env).unwrap()
        );
        assert_eq!(
            CvCallback::call(&mut cv_stopping, &cv_env).unwrap(),
            TrainingCallback::call(&mut training_stopping, &training_env).unwrap()
        );
    }
}

#[test]
fn custom_cv_callbacks_keep_complete_recorded_history_with_a_schedule() {
    use std::cell::RefCell;
    struct History<'a>(&'a RefCell<Vec<usize>>);
    impl CvCallback for History<'_> {
        fn order(&self) -> i32 {
            25
        }
        fn call(&mut self, env: &CvCallbackEnv<'_>) -> Result<CallbackAction> {
            assert_eq!(env.evaluation_history.len(), env.iteration + 1);
            assert_eq!(
                env.evaluation_history.last().unwrap().iteration,
                env.iteration + 1
            );
            assert!(env.evaluation.is_some());
            self.0.borrow_mut().push(env.evaluation_history.len());
            Ok(CallbackAction::Continue)
        }
    }
    let config = Config {
        num_iterations: 4,
        min_data_in_leaf: 1,
        min_data_in_bin: 1,
        feature_pre_filter: false,
        num_leaves: 2,
        metrics: vec!["l2".into()],
        ..Config::default()
    };
    let data = Dataset::from_dense(
        &[0., 1., 2., 3., 4., 5., 6., 7.],
        8,
        1,
        vec![0., 1., 2., 3., 4., 5., 6., 7.],
        &config,
    )
    .unwrap();
    let seen = RefCell::new(Vec::new());
    let result = cross_validate(
        config,
        &data,
        CvOptions {
            folds: FoldStrategy::Standard {
                nfold: 2,
                shuffle: false,
                seed: 0,
            },
            learning_rates: Some(vec![0.1, 0.2, 0.3, 0.4]),
            callbacks: vec![Box::new(History(&seen))],
            ..CvOptions::default()
        },
    )
    .unwrap();
    assert_eq!(result.history.len(), 4);
    assert_eq!(*seen.borrow(), [1, 2, 3, 4]);
}
