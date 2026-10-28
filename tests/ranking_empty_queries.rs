use lightgbm_rust::{
    Config, Dataset, Error, Objective, Trainer, metric::evaluate_dataset, workflow::FitOptions,
};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/empty_queries.json")).unwrap()
}

fn config() -> Config {
    Config {
        objective: Objective::RankXendcg,
        objective_seed: 7,
        min_data_in_leaf: 1,
        min_data_in_bin: 1,
        feature_pre_filter: false,
        enable_bundle: false,
        num_leaves: 3,
        learning_rate: 0.1,
        metrics: vec!["ndcg".into(), "map".into()],
        ..Config::default()
    }
}

fn dataset(case: &Value, c: &Config) -> Dataset {
    let labels: Vec<f32> = serde_json::from_value(case["labels"].clone()).unwrap();
    let x: Vec<_> = (0..labels.len()).map(|i| i as f64).collect();
    let groups = serde_json::from_value(case["groups"].clone()).unwrap();
    let mut d = Dataset::from_dense(&x, labels.len(), 1, labels, c)
        .unwrap()
        .with_groups(groups)
        .unwrap();
    if let Some(p) = case.get("positions") {
        d = d
            .with_positions(serde_json::from_value(p.clone()).unwrap())
            .unwrap();
    }
    d
}

#[test]
fn native_empty_query_metrics_preserve_every_default_cutoff_and_denominator() {
    let f = fixture();
    assert_eq!(
        f["upstream_commit"],
        "5d84f3f222b41409a6983d2ac90513ad9702fa9e"
    );
    assert_eq!(
        f["loaded_library_sha256"],
        "9ed1d46d2d60bac2966fd03bd81e7d192025804f6a3e3ef24c3aa483d0ec7c1e"
    );
    let cases = f["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 14);
    for required in [
        "nonempty", "leading", "middle", "trailing", "repeated", "short", "one_row",
    ] {
        assert!(cases.iter().any(|c| c["name"] == required));
        assert!(
            cases
                .iter()
                .any(|c| c["name"] == format!("{required}_positions"))
        );
    }
    let c = config();
    for case in cases {
        let d = dataset(case, &c);
        let mut boundaries = vec![0];
        for &size in d.groups().unwrap() {
            boundaries.push(boundaries.last().unwrap() + size);
        }
        assert_eq!(
            serde_json::to_value(boundaries).unwrap(),
            case["boundaries"]
        );
        for step in case["steps"].as_array().unwrap() {
            let scores: Vec<f64> = serde_json::from_value(step["scores"].clone()).unwrap();
            let actual = evaluate_dataset(&c, &d, &scores).unwrap();
            let expected = step["metrics"].as_array().unwrap();
            assert_eq!(actual.len(), 10);
            for (a, e) in actual.iter().zip(expected) {
                assert_eq!(a.name, e["name"].as_str().unwrap());
                assert!(a.higher_is_better && a.value.is_finite());
                assert!(
                    (a.value - e["value"].as_f64().unwrap()).abs() <= 1e-12,
                    "{} {}",
                    case["name"],
                    a.name
                );
            }
        }
    }
}

#[test]
fn native_xendcg_training_preserves_query_seeds_and_round_state() {
    for case in fixture()["cases"].as_array().unwrap() {
        assert_native_training(case, "training");
    }
}

#[test]
fn native_query_bagging_retains_empty_query_indexes() {
    let f = fixture();
    let cases: Vec<_> = f["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|case| case.get("query_bagging").is_some())
        .collect();
    assert_eq!(cases.len(), 8);
    for case in cases {
        assert_native_training(case, "query_bagging");
    }
}

fn assert_native_training(case: &Value, field: &str) {
    let training = &case[field];
    let mut c = config();
    if field == "query_bagging" {
        c.bagging_by_query = true;
        c.bagging_freq = 1;
        c.bagging_fraction = 0.75;
        c.bagging_seed = 3;
    }
    let d = dataset(case, &c);
    let x: Vec<_> = (0..d.nrows()).map(|i| i as f64).collect();
    for rounds in 1..=3 {
        c.num_iterations = rounds;
        let result = Trainer::new(c.clone()).unwrap().fit(&d).unwrap();
        let actual = result.model.predict_raw_dense(&x, d.nrows(), 1).unwrap();
        let expected: Vec<f64> =
            serde_json::from_value(training["predictions"][rounds - 1].clone()).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (row, &e) in actual.iter().zip(&expected) {
            assert_eq!(row.len(), 1);
            let a = row[0];
            assert!(
                a.is_finite() && (a - e).abs() <= 1e-12 + 1e-10 * e.abs(),
                "{} {field} round {rounds}: {a} != {e}",
                case["name"]
            );
        }
        for record in &result.evaluation_history {
            for metric in &record.results {
                let e = training["metrics"][&metric.metric][record.iteration - 1]
                    .as_f64()
                    .unwrap();
                assert!(
                    (metric.value - e).abs() <= 1e-12,
                    "{} {field} round {} {}",
                    case["name"],
                    record.iteration,
                    metric.metric
                );
            }
        }
    }
}

#[test]
fn query_buffer_limits_are_checked_without_dropping_empty_groups() {
    let mut c = config();
    let mut groups = vec![0; 1025];
    groups.push(4);
    let d = Dataset::from_dense(&[0., 1., 2., 3.], 4, 1, vec![0., 1., 1., 0.], &c)
        .unwrap()
        .with_groups(groups.clone())
        .unwrap();
    c.num_iterations = 2;
    assert!(Trainer::new(c.clone()).unwrap().fit(&d).is_ok());
    assert_eq!(d.groups(), Some(groups.as_slice()));
    c.bagging_by_query = true;
    c.bagging_fraction = 0.75;
    c.bagging_freq = 1;
    assert!(
        matches!(Trainer::new(c).unwrap().fit(&d), Err(Error::Unsupported(ref message)) if message.contains("query buffers"))
    );
}

#[test]
fn query_rng_review_example_is_rejected_before_core_and_workflow_sampling() {
    let mut c = config();
    c.num_iterations = 1;
    c.bagging_by_query = true;
    c.bagging_freq = 1;
    c.bagging_fraction = 0.5;
    let mut groups = vec![0; 1025];
    groups[0] = 4;
    let d = Dataset::from_dense(&[0., 1., 2., 3.], 4, 1, vec![0., 1., 1., 0.], &c)
        .unwrap()
        .with_groups(groups.clone())
        .unwrap();
    let trainer = Trainer::new(c).unwrap();
    for result in [
        trainer.fit(&d),
        trainer.fit_with_options(&d, &[], FitOptions::default()),
    ] {
        assert!(
            matches!(result, Err(Error::Unsupported(ref message)) if message.contains("query buffers"))
        );
    }
    assert_eq!(d.groups(), Some(groups.as_slice()));
}

#[test]
fn undefined_native_combinations_return_typed_errors_without_execution() {
    let c = config();
    let f = fixture();
    let case = f["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "leading")
        .unwrap();
    let d = dataset(case, &c);
    assert!(matches!(
        d.clone().with_weights(vec![]),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        d.clone().with_weights(vec![1.; 4]),
        Err(Error::Unsupported(_))
    ));
    let weighted = Dataset::from_dense(&[0., 1., 2., 3.], 4, 1, vec![0., 1., 1., 0.], &c)
        .unwrap()
        .with_weights(vec![1.; 4])
        .unwrap();
    assert!(matches!(
        weighted.with_groups(vec![0, 2, 2]),
        Err(Error::Unsupported(_))
    ));
    let lambda = Config {
        objective: Objective::LambdaRank,
        ..c
    };
    let error = Trainer::new(lambda.clone()).unwrap().fit(&d).unwrap_err();
    assert!(matches!(error, Error::Unsupported(ref message) if message.contains("sorted_idx[0]")));
    // Metrics do not call the unsafe native LambdaRank derivative kernel.
    assert!(
        evaluate_dataset(&lambda, &d, &[0.; 4])
            .unwrap()
            .iter()
            .all(|m| m.value.is_finite())
    );
}
