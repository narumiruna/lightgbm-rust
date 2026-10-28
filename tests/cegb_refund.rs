use lightgbm_rust::{Config, Dataset, Model, Trainer};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/review_cegb.json")).unwrap()
}

fn config(case: &Value) -> Config {
    let pairs = case["params"].as_object().unwrap().iter().map(|(k, v)| {
        let value = match v {
            Value::String(s) => s.clone(),
            Value::Array(a) => a.iter().map(Value::to_string).collect::<Vec<_>>().join(","),
            _ => v.to_string(),
        };
        (k.clone(), value)
    });
    let mut config = Config::from_pairs(pairs).unwrap();
    config.num_iterations = case["training"]["num_boost_round"].as_u64().unwrap() as usize;
    config
}

fn dense(rows: &Value) -> Vec<f64> {
    rows.as_array()
        .unwrap()
        .iter()
        .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
        .collect()
}

fn check(case: &Value) {
    let name = case["name"].as_str().unwrap();
    let config = config(case);
    let rows = case["training"]["rows"].as_array().unwrap().len();
    let data = Dataset::from_dense(
        &dense(&case["training"]["rows"]),
        rows,
        3,
        serde_json::from_value(case["training"]["labels"].clone()).unwrap(),
        &config,
    )
    .unwrap();
    let path = case["model_file"].as_str().unwrap();
    assert_eq!(path, format!("models/review_cegb/{name}.txt"));
    let native = Model::from_text(
        &std::fs::read_to_string(format!(
            "{}/tests/fixtures/{path}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap(),
    )
    .unwrap();
    let probes = dense(&case["rows"]);
    let probe_rows = case["rows"].as_array().unwrap().len();
    let expected_raw: Vec<Vec<f64>> = serde_json::from_value(case["raw"].clone()).unwrap();
    let expected_leaves: Vec<Vec<usize>> = serde_json::from_value(case["leaves"].clone()).unwrap();
    assert_eq!(expected_raw.len(), probe_rows);
    assert_eq!(expected_leaves.len(), probe_rows);
    for (cache, threads) in [(-1., 1), (0.000001, 1), (0.000001, 4)] {
        let mut config = config.clone();
        config.histogram_pool_size = cache;
        config.num_threads = threads;
        let result = Trainer::new(config).unwrap().fit(&data).unwrap();
        assert_eq!(result.model.trees().len(), native.trees().len(), "{name}");
        for (i, (got, want)) in result.model.trees().iter().zip(native.trees()).enumerate() {
            assert_eq!(got.num_leaves(), want.num_leaves(), "{name} tree{i}");
            assert_eq!(
                got.split_features(),
                want.split_features(),
                "{name} tree{i}"
            );
            assert_eq!(got.thresholds(), want.thresholds(), "{name} tree{i}");
            assert_eq!(got.left_children(), want.left_children(), "{name} tree{i}");
            assert_eq!(
                got.right_children(),
                want.right_children(),
                "{name} tree{i}"
            );
            assert_eq!(
                got.decision_types(),
                want.decision_types(),
                "{name} tree{i}"
            );
            assert_eq!(got.leaf_counts(), want.leaf_counts(), "{name} tree{i}");
            assert_eq!(
                got.internal_counts(),
                want.internal_counts(),
                "{name} tree{i}"
            );
            let gains: Vec<f64> =
                serde_json::from_value(case["native_split_gains"][i].clone()).unwrap();
            assert_eq!(got.split_gains().len(), gains.len());
            // Compare live native gains. The native text roundtrip rounds
            // this diagnostic field, even when scores and derivatives match.
            for (&a, &b) in got.split_gains().iter().zip(&gains) {
                assert_eq!(a.to_bits(), b.to_bits(), "{name} tree{i} gain");
            }
            for (&a, &b) in got.leaf_values().iter().zip(want.leaf_values()) {
                assert!(
                    (a - b).abs() <= 1e-12 + 1e-10 * b.abs(),
                    "{name} tree{i} output"
                );
            }
        }
        assert_eq!(
            result
                .model
                .predict_leaf_dense(&probes, probe_rows, 3)
                .unwrap(),
            expected_leaves,
            "{name} leaf predictions"
        );
        let raw = result
            .model
            .predict_raw_dense(&probes, probe_rows, 3)
            .unwrap();
        assert_eq!(raw.len(), expected_raw.len());
        for (got, want) in raw.iter().zip(&expected_raw) {
            assert_eq!(got.len(), want.len());
            for (&a, &b) in got.iter().zip(want) {
                assert!(
                    (a - b).abs() <= 1e-12 + 1e-10 * b.abs(),
                    "{name} raw {a} != {b}"
                );
            }
        }
    }
}

#[test]
fn native_pending_leaf_refund_preserves_pinned_behavior() {
    let fixture = fixture();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 12);
    for case in cases {
        check(case);
    }
}

#[test]
fn native_first_coupled_use_is_below_root_with_a_pending_competitor() {
    let fixture = fixture();
    let mixed = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "mixed_pending")
        .unwrap();
    let root = &mixed["training"]["expected_structure"][0];
    assert_eq!(root["split_feature"], 0);
    assert_eq!(root["left_child"]["split_feature"], 1);
    assert_eq!(root["left_child"]["split_index"], 1);
    assert_eq!(root["right_child"]["split_feature"], 1);
    assert_eq!(root["right_child"]["split_index"], 2);
    assert_eq!(root["right_child"]["split_gain"], 13.);
    assert_eq!(mixed["params"]["num_leaves"], 4);
    // The native cache contains raw gain 8. It adds coupled cost 5, producing
    // 13, above the pending feature-2 gain 12.5 - 8 * .1 = 11.7. Retaining
    // split/lazy costs would instead give 8 - 8 * (.1 + .2) = 5.6 and change
    // the last feature to 2. This test deliberately records pinned behavior.
    check(mixed);
}
