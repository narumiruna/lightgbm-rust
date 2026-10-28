use lightgbm_rust::metric::MetricValue;
use lightgbm_rust::workflow::{FitOptions, PredictionKind};
use lightgbm_rust::{Config, Dataset, Objective, Trainer};
use serde_json::Value;

#[test]
fn native_custom_objectives_and_raw_metrics() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/workflow.json")).unwrap();
    for name in ["custom_l2", "custom_binary", "invalid_custom_length"] {
        let case = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == name)
            .unwrap();
        let mut config = Config::from_pairs(
            case["params"]
                .as_object()
                .unwrap()
                .iter()
                .filter(|(k, _)| k.as_str() != "objective")
                .map(|(k, v)| {
                    (
                        k.clone(),
                        v.as_str()
                            .map(str::to_owned)
                            .unwrap_or_else(|| v.to_string()),
                    )
                }),
        )
        .unwrap();
        config.objective = Objective::Custom;
        config.num_iterations = 7;
        let rows: Vec<Vec<f64>> = serde_json::from_value(case["rows"].clone()).unwrap();
        let labels: Vec<f32> = serde_json::from_value(case["labels"].clone()).unwrap();
        let data = Dataset::from_dense(
            &rows.iter().flatten().copied().collect::<Vec<_>>(),
            rows.len(),
            2,
            labels,
            &config,
        )
        .unwrap();
        let valid = Dataset::validation_from_dense(
            &data,
            &rows.iter().flatten().copied().collect::<Vec<_>>(),
            rows.len(),
            2,
            data.labels().to_vec(),
        )
        .unwrap();
        let binary = name == "custom_binary";
        let mut calls = 0;
        let objective = |scores: &[f64], d: &Dataset| {
            for (a, b) in scores
                .iter()
                .zip(case["derivatives"][calls]["raw_input"].as_array().unwrap())
            {
                close(*a, b.as_f64().unwrap());
            }
            calls += 1;
            let mut g = Vec::new();
            let mut h = Vec::new();
            for (&s, &y) in scores.iter().zip(d.labels()) {
                let p = if binary { 1.0 / (1.0 + (-s).exp()) } else { s };
                g.push((p - f64::from(y)) as f32);
                h.push(if binary { (p * (1.0 - p)) as f32 } else { 1.0 });
            }
            if name == "invalid_custom_length" {
                g.pop();
            }
            Ok((g, h))
        };
        let metric = move |scores: &[f64], d: &Dataset, kind: PredictionKind| {
            assert_eq!(kind, PredictionKind::Raw);
            let value = scores
                .iter()
                .zip(d.labels())
                .map(|(&s, &y)| {
                    if binary {
                        let p = 1.0 / (1.0 + (-s).exp());
                        -f64::from(y) * p.max(1e-15).ln()
                            - (1.0 - f64::from(y)) * (1.0 - p).max(1e-15).ln()
                    } else {
                        (s - f64::from(y)).powi(2)
                    }
                })
                .sum::<f64>()
                / scores.len() as f64;
            Ok(vec![MetricValue {
                name: if binary {
                    "custom_binary_logloss"
                } else {
                    "custom_l2"
                }
                .into(),
                value,
                higher_is_better: false,
            }])
        };
        let result = Trainer::new(config).unwrap().fit_with_options(
            &data,
            &[("valid_a", &valid)],
            FitOptions {
                custom_objective: Some(Box::new(objective)),
                custom_metrics: vec![Box::new(metric)],
                evaluate_training: true,
                training_name: "train_alias".into(),
                ..FitOptions::default()
            },
        );
        if name == "invalid_custom_length" {
            assert!(result.is_err());
            assert_eq!(calls, 1);
            continue;
        }
        let result = result.unwrap();
        assert_eq!(calls, 7);
        for (i, row) in rows.iter().enumerate() {
            close(
                result.model.predict_raw_row(row).unwrap()[0],
                case["live_model"]["raw"][i][0].as_f64().unwrap(),
            );
        }
        for e in result.evaluation_history {
            for m in e.results {
                close(
                    m.value,
                    case["history"][&m.dataset][&m.metric][e.iteration - 1]
                        .as_f64()
                        .unwrap(),
                );
            }
        }
    }
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-12 + 1e-10 * b.abs(), "{a} != {b}");
}

#[test]
fn custom_errors_and_invalid_metrics_do_not_publish_a_model() {
    let config = Config {
        min_data_in_bin: 1,
        min_data_in_leaf: 1,
        ..Config::default()
    };
    let data = Dataset::from_dense(&[0., 1., 2., 3.], 4, 1, vec![0., 1., 2., 3.], &config).unwrap();
    let metric = |_: &[f64], _: &Dataset, _: PredictionKind| {
        Ok(vec![MetricValue {
            name: "invalid".into(),
            value: f64::NAN,
            higher_is_better: false,
        }])
    };
    assert!(
        Trainer::new(config.clone())
            .unwrap()
            .fit_with_options(
                &data,
                &[("alias", &data)],
                FitOptions {
                    custom_metrics: vec![Box::new(metric)],
                    ..FitOptions::default()
                }
            )
            .is_err()
    );
    let custom_config = Config {
        objective: Objective::Custom,
        ..config.clone()
    };
    let objective =
        |_: &[f64], _: &Dataset| Err(lightgbm_rust::Error::InvalidInput("custom failure".into()));
    assert!(
        Trainer::new(custom_config)
            .unwrap()
            .fit_with_options(
                &data,
                &[],
                FitOptions {
                    custom_objective: Some(Box::new(objective)),
                    ..FitOptions::default()
                }
            )
            .is_err()
    );
    assert!(Trainer::new(config).unwrap().fit(&data).is_ok());
}

#[test]
fn invalid_custom_outputs_do_not_publish_a_model() {
    let c = Config {
        objective: Objective::Custom,
        min_data_in_bin: 1,
        min_data_in_leaf: 1,
        ..Config::default()
    };
    let d = Dataset::from_dense(&[0., 1., 2., 3.], 4, 1, vec![0., 1., 2., 3.], &c).unwrap();
    for (g, h) in [
        (vec![0.; 3], vec![1.; 4]),
        (vec![f32::NAN; 4], vec![1.; 4]),
        (vec![0.; 4], vec![-1.; 4]),
        (vec![0.; 4], vec![f32::INFINITY; 4]),
    ] {
        let objective = move |_: &[f64], _: &Dataset| Ok((g.clone(), h.clone()));
        assert!(
            Trainer::new(c.clone())
                .unwrap()
                .fit_with_options(
                    &d,
                    &[],
                    FitOptions {
                        custom_objective: Some(Box::new(objective)),
                        ..FitOptions::default()
                    }
                )
                .is_err()
        );
    }
}
