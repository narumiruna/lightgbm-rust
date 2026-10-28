//! Transfer the existing serial native-fixture gates to CPU execution modes.
//! Native structure/value checks remain in the original test suites.
use lightgbm_rust::{Config, Dataset, Trainer};
use serde_json::{Value, json};

#[test]
fn all_training_fixtures_repeat_exactly_across_threads_modes_and_cache_eviction() {
    let suites = [
        include_str!("fixtures/prediction.json"),
        include_str!("fixtures/prediction_regressions.json"),
        include_str!("fixtures/prediction_extended.json"),
        include_str!("fixtures/prediction_boosting.json"),
        include_str!("fixtures/prediction_constraints.json"),
        include_str!("fixtures/prediction_constraints_extra.json"),
        include_str!("fixtures/prediction_linear.json"),
        include_str!("fixtures/prediction_linear_training.json"),
    ];
    let mut count = 0;
    for suite in suites {
        let suite: Value = serde_json::from_str(suite).unwrap();
        for case in suite["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let pairs = case["params"]
                .as_object()
                .unwrap()
                .iter()
                .filter(|(key, _)| key.as_str() != "forcedsplits_filename")
                .map(|(key, value)| {
                    let value = match value {
                        Value::String(s) => s.clone(),
                        Value::Array(a) if !a.iter().any(Value::is_array) => {
                            a.iter().map(Value::to_string).collect::<Vec<_>>().join(",")
                        }
                        _ => value.to_string(),
                    };
                    (key.clone(), value)
                });
            let mut config = Config::from_pairs(pairs).unwrap();
            config.num_iterations = case["training"]["num_boost_round"].as_u64().unwrap() as usize;
            if case["params"].get("forcedsplits_filename").is_some() {
                config.forced_splits = Some(if config.linear_tree {
                    json!({"feature":0,"threshold":0.0,
                        "left":{"feature":1,"threshold":-12.0},
                        "right":{"feature":1,"threshold":12.0}})
                } else {
                    json!({"feature":1,"threshold":0.0,"left":{"feature":0,"threshold":0.0}})
                });
            }
            let rows = case["training"]["rows"].as_array().unwrap();
            let cols = rows[0].as_array().unwrap().len();
            let x: Vec<f64> = rows
                .iter()
                .flat_map(|row| {
                    row.as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap_or(f64::NAN))
                })
                .collect();
            let mut data = Dataset::from_dense(
                &x,
                rows.len(),
                cols,
                serde_json::from_value(case["training"]["labels"].clone()).unwrap(),
                &config,
            )
            .unwrap();
            if let Some(weights) = case["training"].get("weights") {
                data = data
                    .with_weights(serde_json::from_value(weights.clone()).unwrap())
                    .unwrap();
            }
            if let Some(groups) = case["training"].get("groups") {
                data = data
                    .with_groups(serde_json::from_value(groups.clone()).unwrap())
                    .unwrap();
            }
            if let Some(positions) = case["training"].get("positions") {
                data = data
                    .with_positions(serde_json::from_value(positions.clone()).unwrap())
                    .unwrap();
            }
            let expected = Trainer::new(config.clone())
                .unwrap()
                .fit(&data)
                .unwrap()
                .model
                .to_text()
                .unwrap();
            for threads in [2, 4] {
                for row_wise in [false, true] {
                    let mut c = config.clone();
                    c.num_threads = threads;
                    c.force_row_wise = row_wise;
                    c.force_col_wise = !row_wise;
                    c.histogram_pool_size = 0.000001;
                    for _ in 0..2 {
                        let actual = Trainer::new(c.clone())
                            .unwrap()
                            .fit(&data)
                            .unwrap()
                            .model
                            .to_text()
                            .unwrap();
                        assert_eq!(
                            actual, expected,
                            "{name}: threads={threads}, row_wise={row_wise}"
                        );
                    }
                }
            }
            count += 1;
        }
    }
    assert_eq!(count, 90, "every retained training fixture must be tested");
}
