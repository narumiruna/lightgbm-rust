use lightgbm_rust::metric::MetricValue;
use lightgbm_rust::workflow::{FitOptions, PredictionKind};
use lightgbm_rust::{Config, Dataset, Objective, Trainer};

#[test]
fn multiclass_custom_derivatives_keep_class_major_layout() {
    let config = Config {
        objective: Objective::Multiclass,
        num_class: 3,
        boost_from_average: false,
        num_iterations: 4,
        min_data_in_bin: 1,
        min_data_in_leaf: 1,
        num_leaves: 3,
        ..Config::default()
    };
    let rows = (0..18).map(|i| i as f64).collect::<Vec<_>>();
    let labels = (0..18).map(|i| (i % 3) as f32).collect::<Vec<_>>();
    let data = Dataset::from_dense(&rows, 18, 1, labels, &config).unwrap();
    let built = Trainer::new(config.clone()).unwrap().fit(&data).unwrap();
    let mut custom_config = config.clone();
    custom_config.objective = Objective::Custom;
    custom_config.metrics = vec!["custom".into()];
    let objective = |scores: &[f64], data: &Dataset| {
        let n = data.nrows();
        assert_eq!(scores.len(), n * 3);
        let mut g = vec![0.; n * 3];
        let mut h = vec![0.; n * 3];
        for row in 0..n {
            let max = (0..3)
                .map(|c| scores[c * n + row])
                .fold(f64::NEG_INFINITY, f64::max);
            let p = (0..3)
                .map(|c| (scores[c * n + row] - max).exp())
                .collect::<Vec<_>>();
            let sum = p.iter().sum::<f64>();
            for c in 0..3 {
                let p = p[c] / sum;
                g[c * n + row] = (p - f64::from(data.labels()[row] as usize == c)) as f32;
                h[c * n + row] = (1.5 * p * (1. - p)) as f32;
            }
        }
        Ok((g, h))
    };
    let custom = Trainer::new(custom_config)
        .unwrap()
        .fit_with_options(
            &data,
            &[],
            FitOptions {
                custom_objective: Some(Box::new(objective)),
                ..FitOptions::default()
            },
        )
        .unwrap();
    for row in rows {
        let a = built.model.predict_raw_row(&[row]).unwrap();
        let b = custom.model.predict_raw_row(&[row]).unwrap();
        for (a, b) in a.iter().zip(b) {
            assert!((a - b).abs() <= 1e-12 + 1e-10 * b.abs());
        }
    }
}

#[test]
fn builtin_custom_metric_receives_transformed_class_major_predictions() {
    for objective in [
        Objective::Binary,
        Objective::Multiclass,
        Objective::MulticlassOva,
    ] {
        let config = Config {
            objective,
            num_class: if objective == Objective::Binary { 1 } else { 3 },
            num_iterations: 3,
            min_data_in_bin: 1,
            min_data_in_leaf: 1,
            ..Config::default()
        };
        let rows = (0..18).map(|i| i as f64).collect::<Vec<_>>();
        let labels = (0..18)
            .map(|i| {
                if objective == Objective::Binary {
                    (i % 2) as f32
                } else {
                    (i % 3) as f32
                }
            })
            .collect();
        let data = Dataset::from_dense(&rows, 18, 1, labels, &config).unwrap();
        let mut snapshots = Vec::new();
        let metric = |scores: &[f64], d: &Dataset, kind: PredictionKind| {
            assert_eq!(kind, PredictionKind::Transformed);
            assert_eq!(scores.len(), d.nrows() * config.num_class);
            assert!(scores.iter().all(|p| *p >= 0. && *p <= 1.));
            snapshots.push(scores.to_vec());
            Ok(vec![MetricValue {
                name: "custom_zero".into(),
                value: 0.,
                higher_is_better: false,
            }])
        };
        let result = Trainer::new(config.clone())
            .unwrap()
            .fit_with_options(
                &data,
                &[("valid", &data)],
                FitOptions {
                    custom_metrics: vec![Box::new(metric)],
                    ..FitOptions::default()
                },
            )
            .unwrap();
        let last = snapshots.last().unwrap();
        for (r, x) in rows.iter().enumerate() {
            for (c, p) in result.model.predict_row(&[*x]).unwrap().iter().enumerate() {
                assert!((last[c * 18 + r] - p).abs() < 1e-14);
            }
        }
    }
}
