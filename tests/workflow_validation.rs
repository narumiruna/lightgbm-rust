use lightgbm_rust::workflow::{EarlyStopping, FitOptions, LearningRateSchedule};
use lightgbm_rust::{Config, Dataset, Trainer};
fn data() -> (Config, Dataset) {
    let c = Config {
        num_iterations: 4,
        min_data_in_bin: 1,
        min_data_in_leaf: 1,
        metrics: vec!["l2".into(), "l1".into()],
        ..Config::default()
    };
    let d = Dataset::from_dense(&[0., 1., 2., 3.], 4, 1, vec![0., 1., 2., 3.], &c).unwrap();
    (c, d)
}
#[test]
fn source_training_identity_is_excluded_even_under_an_alias() {
    let (c, d) = data();
    let result = Trainer::new(c)
        .unwrap()
        .fit_with_options(
            &d,
            &[("alias", &d)],
            FitOptions {
                callbacks: vec![Box::new(EarlyStopping::new(1).min_delta(vec![1e20]))],
                ..FitOptions::default()
            },
        )
        .unwrap();
    assert_eq!(result.best_iteration, 0);
    assert_eq!(result.evaluation_history.len(), 4);
    assert!(result.best_scores.iter().all(|v| v.dataset == "alias"));
}
#[test]
fn invalid_schedules_and_metric_deltas_fail() {
    let (c, d) = data();
    for rates in [
        vec![0.1],
        vec![-1.; 4],
        vec![f64::NAN; 4],
        vec![f64::INFINITY; 4],
        vec![0.; 4],
    ] {
        assert!(
            Trainer::new(c.clone())
                .unwrap()
                .fit_with_options(
                    &d,
                    &[],
                    FitOptions {
                        callbacks: vec![Box::new(LearningRateSchedule::values(rates))],
                        ..FitOptions::default()
                    }
                )
                .is_err()
        );
    }
    let v =
        Dataset::validation_from_dense(&d, &[0., 1., 2., 3.], 4, 1, vec![0., 1., 2., 3.]).unwrap();
    for delta in [vec![-1.], vec![f64::NAN], vec![1., 2., 3.]] {
        assert!(
            Trainer::new(c.clone())
                .unwrap()
                .fit_with_options(
                    &d,
                    &[("valid", &v)],
                    FitOptions {
                        callbacks: vec![Box::new(EarlyStopping::new(1).min_delta(delta))],
                        ..FitOptions::default()
                    }
                )
                .is_err()
        );
    }
}
#[test]
fn function_schedule_matches_value_schedule() {
    let (c, d) = data();
    let values = Trainer::new(c.clone())
        .unwrap()
        .fit_with_options(
            &d,
            &[],
            FitOptions {
                callbacks: vec![Box::new(LearningRateSchedule::values(vec![
                    0.4, 0.2, 0.1, 0.05,
                ]))],
                ..FitOptions::default()
            },
        )
        .unwrap();
    let function = Trainer::new(c)
        .unwrap()
        .fit_with_options(
            &d,
            &[],
            FitOptions {
                callbacks: vec![Box::new(LearningRateSchedule::function(|i| {
                    0.4 / 2_f64.powi(i as i32)
                }))],
                ..FitOptions::default()
            },
        )
        .unwrap();
    assert_eq!(values.training_scores, function.training_scores);
}
