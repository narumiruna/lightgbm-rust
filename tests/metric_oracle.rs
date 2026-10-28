use lightgbm_rust::metric::{evaluate, evaluate_dataset};
use lightgbm_rust::{Config, Dataset, Objective};
use serde_json::Value;

#[test]
fn every_cpu_metric_factory_matches_native_driver() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/metrics.json")).unwrap();
    assert_eq!(
        fixture["upstream_commit"],
        "5d84f3f222b41409a6983d2ac90513ad9702fa9e"
    );
    let mut families = std::collections::BTreeSet::new();
    for case in fixture["cases"].as_array().unwrap() {
        families.insert(case["metric"].as_str().unwrap());
        let pairs = case["params"].as_object().unwrap().iter().map(|(k, v)| {
            let value = match v {
                Value::String(s) => s.clone(),
                Value::Array(values) => values
                    .iter()
                    .map(Value::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
                _ => v.to_string(),
            };
            (k.clone(), value)
        });
        let mut config = Config::from_pairs(pairs).unwrap();
        config.metrics = vec![case["metric"].as_str().unwrap().into()];
        let labels: Vec<f32> = serde_json::from_value(case["labels"].clone()).unwrap();
        let weights: Option<Vec<f32>> = case
            .get("weights")
            .map(|v| serde_json::from_value(v.clone()).unwrap());
        let scores: Vec<f64> = serde_json::from_value(case["scores"].clone()).unwrap();
        let mut dataset = Dataset::from_dense(
            &vec![0.0; labels.len()],
            labels.len(),
            1,
            labels.clone(),
            &config,
        )
        .unwrap();
        if let Some(w) = &weights {
            dataset = dataset.with_weights(w.clone()).unwrap();
        }
        if let Some(g) = case.get("groups") {
            dataset = dataset
                .with_groups(serde_json::from_value(g.clone()).unwrap())
                .unwrap();
        }
        let actual = evaluate_dataset(&config, &dataset, &scores);
        let expected = case["results"].as_array().unwrap();
        if expected.iter().any(|v| v["value"].is_string()) {
            assert!(
                actual.is_err(),
                "{} nonfinite metric requires an explicit error",
                case["name"]
            );
            continue;
        }
        let actual = actual.unwrap_or_else(|e| panic!("{}: {e}", case["name"]));
        assert_eq!(actual.len(), expected.len(), "{} shape", case["name"]);
        for (a, e) in actual.iter().zip(expected) {
            assert_eq!(a.name, e["name"].as_str().unwrap(), "{} name", case["name"]);
            assert_eq!(a.higher_is_better, e["higher_is_better"].as_bool().unwrap());
            let expected = e["value"].as_f64().unwrap();
            assert!(
                (a.value - expected).abs() <= 1e-12 + 1e-12 * expected.abs(),
                "{} {}: {} != {expected}",
                case["name"],
                a.name,
                a.value
            );
        }
        if case.get("groups").is_none() {
            assert_eq!(
                actual,
                evaluate(&config, &labels, weights.as_deref(), &scores).unwrap()
            );
        }
    }
    assert_eq!(
        families.len(),
        24,
        "all pinned CPU metric factories must be covered"
    );
}

#[test]
fn source_default_metric_registry_and_disabled_metrics() {
    for (objective, classes, expected) in [
        (Objective::Regression, 1, "l2"),
        (Objective::RegressionL1, 1, "l1"),
        (Objective::Quantile, 1, "quantile"),
        (Objective::Huber, 1, "huber"),
        (Objective::Fair, 1, "fair"),
        (Objective::Poisson, 1, "poisson"),
        (Objective::Mape, 1, "mape"),
        (Objective::Gamma, 1, "gamma"),
        (Objective::Tweedie, 1, "tweedie"),
        (Objective::Binary, 1, "binary_logloss"),
        (Objective::Multiclass, 3, "multi_logloss"),
        (Objective::MulticlassOva, 3, "multi_logloss"),
        (Objective::CrossEntropy, 1, "cross_entropy"),
        (Objective::CrossEntropyLambda, 1, "cross_entropy_lambda"),
        (Objective::LambdaRank, 1, "ndcg@1"),
        (Objective::RankXendcg, 1, "ndcg@1"),
    ] {
        let c = Config {
            objective,
            num_class: classes,
            ..Config::default()
        };
        let data = Dataset::from_dense(&[0.0, 0.0], 2, 1, vec![1.0, 1.0], &c)
            .unwrap()
            .with_groups(vec![2])
            .unwrap();
        let values = evaluate_dataset(&c, &data, &vec![0.0; 2 * classes]).unwrap();
        assert_eq!(values[0].name, expected);
        assert_eq!(
            values.len(),
            if matches!(objective, Objective::LambdaRank | Objective::RankXendcg) {
                5
            } else {
                1
            }
        );
        let disabled = Config {
            metrics: vec!["custom".into()],
            ..c
        };
        assert!(
            evaluate_dataset(&disabled, &data, &vec![0.0; 2 * classes])
                .unwrap()
                .is_empty()
        );
    }
    let c = Config {
        objective: Objective::Custom,
        ..Config::default()
    };
    assert!(evaluate(&c, &[1.0], None, &[0.0]).unwrap().is_empty());
}

#[test]
fn incompatible_metadata_and_metrics_return_errors() {
    for metric in ["ndcg", "map"] {
        let config = Config {
            metrics: vec![metric.into()],
            ..Config::default()
        };
        assert!(evaluate(&config, &[0.0, 1.0], None, &[0.0; 2]).is_err());
    }
    for metric in ["gamma", "gamma_deviance"] {
        let config = Config {
            metrics: vec![metric.into()],
            ..Config::default()
        };
        assert!(evaluate(&config, &[0.0, 1.0], None, &[1.0; 2]).is_err());
    }
    for metric in ["cross_entropy", "cross_entropy_lambda", "kullback_leibler"] {
        let config = Config {
            metrics: vec![metric.into()],
            ..Config::default()
        };
        assert!(evaluate(&config, &[-0.1, 1.1], None, &[0.0; 2]).is_err());
    }
    let multi = Config {
        objective: Objective::Multiclass,
        num_class: 65_536,
        metrics: vec!["auc_mu".into()],
        ..Config::default()
    };
    assert!(evaluate(&multi, &[0.0], None, &vec![0.0; 65_536]).is_err());
    let multi = Config {
        objective: Objective::Multiclass,
        num_class: 3,
        metrics: vec!["auc_mu".into()],
        ..Config::default()
    };
    assert!(evaluate(&multi, &[0.0, 1.0], None, &[0.0; 6]).is_err());
    let multi = Config {
        metrics: vec!["multi_error".into()],
        ..multi
    };
    assert!(evaluate(&multi, &[0.0, 3.0], None, &[0.0; 6]).is_err());
}
