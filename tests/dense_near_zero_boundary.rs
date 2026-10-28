//! Preserve the pinned native difference between Dataset and dense prediction.
use lightgbm_rust::workflow::{FitOptions, PredictionKind};
use lightgbm_rust::{Config, Dataset, Model, Trainer};
use serde_json::Value;
use std::cell::RefCell;

fn dense(rows: &Value) -> Vec<f64> {
    rows.as_array()
        .unwrap()
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
        .collect()
}

fn close(got: f64, want: f64, maximum: &mut f64) {
    *maximum = maximum.max((got - want).abs());
    assert!(
        (got - want).abs() <= 1e-12 + 1e-10 * want.abs(),
        "{got} != native {want}"
    );
}

fn scores(got: &[f64], expected: &Value, maximum: &mut f64) {
    let expected: Vec<f64> = serde_json::from_value(expected.clone()).unwrap();
    assert_eq!(got.len(), expected.len());
    for (&a, &b) in got.iter().zip(&expected) {
        close(a, b, maximum);
    }
}

#[test]
fn dataset_and_dense_prediction_preserve_the_native_negative_inclusive_boundary() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/review_near_zero.json")).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 1);
    let case = &cases[0];
    assert_eq!(case["name"], "negative_inclusive");
    let pairs = case["params"].as_object().unwrap().iter().map(|(k, v)| {
        (
            k.clone(),
            v.as_str().map_or_else(|| v.to_string(), str::to_owned),
        )
    });
    let mut config = Config::from_pairs(pairs).unwrap();
    config.num_iterations = case["training"]["num_boost_round"].as_u64().unwrap() as usize;
    let training = &case["training"];
    let validation = &case["validation"];
    let rows = dense(&training["rows"]);
    let probes = dense(&validation["rows"]);
    let data = Dataset::from_dense(
        &rows,
        rows.len(),
        1,
        serde_json::from_value(training["labels"].clone()).unwrap(),
        &config,
    )
    .unwrap();
    let valid = data
        .validation_from_dense(
            &probes,
            probes.len(),
            1,
            serde_json::from_value(validation["labels"].clone()).unwrap(),
        )
        .unwrap();
    let z = 1e-35_f32 as f64;
    assert_eq!(
        case["zero_threshold"].as_f64().unwrap().to_bits(),
        z.to_bits()
    );
    assert_eq!(rows[1], -z);
    assert_eq!(
        probes,
        [
            -1.,
            (-z).next_down(),
            -z,
            (-z).next_up(),
            0.,
            z.next_down(),
            z,
            z.next_up(),
            1.
        ]
    );

    let captured = RefCell::new(Vec::new());
    let capture = |values: &[f64], dataset: &Dataset, kind: PredictionKind| {
        assert_eq!(kind, PredictionKind::Transformed);
        captured
            .borrow_mut()
            .push((std::ptr::eq(dataset, &data), values.to_vec()));
        Ok(vec![])
    };
    let result = Trainer::new(config.clone())
        .unwrap()
        .fit_with_options(
            &data,
            &[("validation", &valid)],
            FitOptions {
                evaluate_training: true,
                custom_metrics: vec![Box::new(capture)],
                ..FitOptions::default()
            },
        )
        .unwrap();
    let mut buffer_max = 0.0;
    scores(
        &result.training_scores,
        &training["scores"],
        &mut buffer_max,
    );
    let captured = captured.borrow();
    assert_eq!(captured.len(), 2);
    assert!(captured[0].0);
    assert!(!captured[1].0);
    scores(&captured[0].1, &training["scores"], &mut buffer_max);
    scores(&captured[1].1, &validation["scores"], &mut buffer_max);
    let mut metric_max = 0.0;
    assert_eq!(result.evaluation_history.len(), 1);
    assert_eq!(result.evaluation_history[0].results.len(), 2);
    for metric in &result.evaluation_history[0].results {
        assert!(!metric.higher_is_better);
        close(
            metric.value,
            case[&metric.dataset]["metrics"][&metric.metric][0]
                .as_f64()
                .unwrap(),
            &mut metric_max,
        );
    }

    assert_eq!(
        case["model_file"],
        "models/review_near_zero/negative_inclusive.txt"
    );
    let native = Model::from_text(include_str!(
        "fixtures/models/review_near_zero/negative_inclusive.txt"
    ))
    .unwrap();
    assert_eq!(result.model.trees().len(), 1);
    assert_eq!(native.trees().len(), 1);
    let got = &result.model.trees()[0];
    let want = &native.trees()[0];
    assert_eq!(got.num_leaves(), want.num_leaves());
    assert_eq!(got.split_features(), want.split_features());
    assert_eq!(got.thresholds(), want.thresholds());
    assert_eq!(got.left_children(), want.left_children());
    assert_eq!(got.right_children(), want.right_children());
    assert_eq!(got.decision_types(), want.decision_types());
    assert_eq!(got.leaf_counts(), want.leaf_counts());
    assert_eq!(got.internal_counts(), want.internal_counts());
    let mut structure_max = 0.0;
    for (a, b) in [
        (got.leaf_values(), want.leaf_values()),
        (got.leaf_weights(), want.leaf_weights()),
        (got.split_gains(), want.split_gains()),
        (got.internal_values(), want.internal_values()),
        (got.internal_weights(), want.internal_weights()),
    ] {
        assert_eq!(a.len(), b.len());
        for (&a, &b) in a.iter().zip(b) {
            close(a, b, &mut structure_max);
        }
    }
    let root = &case["native_structure"][0];
    assert_eq!(got.thresholds()[0], root["threshold"].as_f64().unwrap());
    let mut live_structure_max = 0.0;
    for (value, expected) in [
        (got.split_gains()[0], &root["split_gain"]),
        (got.internal_values()[0], &root["internal_value"]),
        (got.internal_weights()[0], &root["internal_weight"]),
        (got.leaf_values()[0], &root["left_child"]["leaf_value"]),
        (got.leaf_values()[1], &root["right_child"]["leaf_value"]),
        (got.leaf_weights()[0], &root["left_child"]["leaf_weight"]),
        (got.leaf_weights()[1], &root["right_child"]["leaf_weight"]),
    ] {
        close(value, expected.as_f64().unwrap(), &mut live_structure_max);
    }
    let mut dense_max = 0.0;
    for model in [&result.model, &native] {
        for (name, input) in [("training", &rows), ("validation", &probes)] {
            let expected = &case["dense"][name];
            let raw = model.predict_raw_dense(input, input.len(), 1).unwrap();
            let transformed = model.predict_dense(input, input.len(), 1).unwrap();
            let expected_raw: Vec<Vec<f64>> =
                serde_json::from_value(expected["raw"].clone()).unwrap();
            let expected_leaves: Vec<Vec<usize>> =
                serde_json::from_value(expected["leaves"].clone()).unwrap();
            assert_eq!(raw.len(), expected_raw.len());
            assert_eq!(transformed.len(), expected_raw.len());
            for ((a, t), b) in raw.iter().zip(&transformed).zip(&expected_raw) {
                assert_eq!(a.len(), b.len());
                assert_eq!(t.len(), b.len());
                for ((&a, &t), &b) in a.iter().zip(t).zip(b) {
                    close(a, b, &mut dense_max);
                    close(t, b, &mut dense_max);
                }
            }
            assert_eq!(
                model.predict_leaf_dense(input, input.len(), 1).unwrap(),
                expected_leaves
            );
        }
    }
    // Native Dataset maps raw -z to the left bin; native dense C API omits it
    // and routes zero right. Normalizing Dataset input would break parity.
    assert_ne!(data.get_bin(1, 0).unwrap(), data.get_bin(2, 0).unwrap());
    assert_ne!(valid.get_bin(2, 0).unwrap(), valid.get_bin(4, 0).unwrap());
    let dense_training = result
        .model
        .predict_raw_dense(&rows, rows.len(), 1)
        .unwrap();
    assert_ne!(result.training_scores[1], dense_training[1][0]);
    let dense_validation = result
        .model
        .predict_raw_dense(&probes, probes.len(), 1)
        .unwrap();
    assert_ne!(captured[1].1[2], dense_validation[2][0]);
    // Apply the proposed input rule without changing production code. Sampling
    // already omits near-zero inputs, so only stored training bins change.
    let normalized: Vec<f64> = rows
        .iter()
        .map(|&v| if (-z..=z).contains(&v) { 0.0 } else { v })
        .collect();
    let changed_data =
        Dataset::from_dense(&normalized, rows.len(), 1, data.labels().to_vec(), &config).unwrap();
    let changed = Trainer::new(config).unwrap().fit(&changed_data).unwrap();
    assert_ne!(changed.model.trees()[0].leaf_counts(), want.leaf_counts());
    let native_scores: Vec<f64> = serde_json::from_value(training["scores"].clone()).unwrap();
    assert_ne!(changed.training_scores, native_scores);
    eprintln!(
        "Rust/native max_abs: buffers={buffer_max:?}, metrics={metric_max:?}, live_structure={live_structure_max:?}, saved_structure={structure_max:?}, dense={dense_max:?}"
    );
}
