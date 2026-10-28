use lightgbm_rust::cv::{CvOptions, Fold, FoldStrategy, cross_validate, make_folds};
use lightgbm_rust::workflow::EarlyStopping;
use lightgbm_rust::{Config, Dataset, Model};
use serde_json::Value;

#[test]
fn native_folds_histories_population_std_and_stopping() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/cv.json")).unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let mut config =
            Config::from_pairs(c["params"].as_object().unwrap().iter().map(|(k, v)| {
                (
                    k.clone(),
                    if let Some(a) = v.as_array() {
                        a.iter()
                            .map(|v| {
                                v.as_str()
                                    .map(str::to_owned)
                                    .unwrap_or_else(|| v.to_string())
                            })
                            .collect::<Vec<_>>()
                            .join(",")
                    } else {
                        v.as_str()
                            .map(str::to_owned)
                            .unwrap_or_else(|| v.to_string())
                    },
                )
            }))
            .unwrap();
        config.num_iterations = c["num_boost_round"].as_u64().unwrap() as usize;
        let rows: Vec<Vec<f64>> = serde_json::from_value(c["rows"].clone()).unwrap();
        let mut data = Dataset::from_dense(
            &rows.iter().flatten().copied().collect::<Vec<_>>(),
            rows.len(),
            2,
            serde_json::from_value(c["labels"].clone()).unwrap(),
            &config,
        )
        .unwrap();
        if c["groups"].is_array() {
            data = data
                .with_groups(serde_json::from_value(c["groups"].clone()).unwrap())
                .unwrap();
        }
        let expected: Vec<Fold> = c["folds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| Fold {
                train_indices: serde_json::from_value(f["train_indices"].clone()).unwrap(),
                valid_indices: serde_json::from_value(f["valid_indices"].clone()).unwrap(),
            })
            .collect();
        let strategy = match name {
            "explicit" => FoldStrategy::Explicit(expected.clone()),
            "group_safe" => FoldStrategy::Group { nfold: 3 },
            "stratified_seeded" => FoldStrategy::Stratified {
                nfold: 3,
                shuffle: true,
                seed: 19,
            },
            _ => FoldStrategy::Standard {
                nfold: 3,
                shuffle: c["shuffle"].as_bool().unwrap(),
                seed: 19,
            },
        };
        assert_eq!(make_folds(&data, &strategy).unwrap(), expected, "{name}");
        let options = CvOptions {
            folds: strategy,
            evaluate_training: true,
            early_stopping: if c["stopping"].is_object() {
                Some(EarlyStopping::new(2).min_delta(vec![1e6]))
            } else {
                None
            },
            ..CvOptions::default()
        };
        let result = cross_validate(config, &data, options).unwrap();
        assert_eq!(
            result.best_iteration.map(|i| i as i64).unwrap_or(-1),
            c["best_iteration"].as_i64().unwrap(),
            "{name}"
        );
        let expected_len = c["history"]
            .as_object()
            .unwrap()
            .values()
            .next()
            .unwrap()
            .as_array()
            .unwrap()
            .len();
        assert_eq!(result.history.len(), expected_len, "{name}");
        for round in &result.history {
            for m in &round.results {
                let prefix = format!("{} {}", m.evaluation.dataset, m.evaluation.metric);
                close(
                    m.evaluation.value,
                    c["history"][format!("{prefix}-mean")][round.iteration - 1]
                        .as_f64()
                        .unwrap(),
                );
                close(
                    m.std_dev,
                    c["history"][format!("{prefix}-stdv")][round.iteration - 1]
                        .as_f64()
                        .unwrap(),
                );
            }
        }
        for (fold, model) in result.models.iter().enumerate() {
            let reference = Model::load(format!(
                "{}/tests/fixtures/{}",
                env!("CARGO_MANIFEST_DIR"),
                c["models"][fold]["full_model"]["model_file"]
                    .as_str()
                    .unwrap()
            ))
            .unwrap();
            assert_eq!(model.trees().len(), reference.trees().len());
            for (a, b) in model.trees().iter().zip(reference.trees()) {
                assert_eq!(a.split_features(), b.split_features());
                assert_eq!(a.thresholds(), b.thresholds());
                assert_eq!(a.left_children(), b.left_children());
                assert_eq!(a.right_children(), b.right_children());
                assert_eq!(a.leaf_counts(), b.leaf_counts());
                for (a, b) in a.leaf_values().iter().zip(b.leaf_values()) {
                    close(*a, *b);
                }
            }
            for (i, row) in rows.iter().enumerate() {
                for (a, b) in model.predict_raw_row(row).unwrap().iter().zip(
                    c["models"][fold]["full_model"]["raw"][i]
                        .as_array()
                        .unwrap(),
                ) {
                    close(*a, b.as_f64().unwrap());
                }
            }
        }
    }
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-12 + 1e-10 * b.abs(), "{a} != {b}");
}
