use lightgbm_rust::{Config, Dataset, Trainer};

#[test]
fn replay_and_retained_histograms_match_with_forced_and_inactive_features() {
    let rows = 192;
    let cols = 5;
    let values: Vec<f64> = (0..rows * cols)
        .map(|i| {
            let row = i / cols;
            let feature = i % cols;
            if feature == 4 || (row + feature) % 13 < 9 {
                0.0
            } else {
                ((row * 17 + feature * 7) % 31) as f64
            }
        })
        .collect();
    for forced in [false, true] {
        let config = Config {
            num_iterations: 5,
            num_leaves: 9,
            min_data_in_leaf: 3,
            min_data_in_bin: 1,
            feature_pre_filter: false,
            max_bin: 17,
            forced_splits: forced.then(|| serde_json::json!({"feature": 0, "threshold": 12.0})),
            ..Config::default()
        };
        let data = Dataset::from_dense(
            &values,
            rows,
            cols,
            (0..rows).map(|row| (row % 23) as f32).collect(),
            &config,
        )
        .unwrap();
        let expected = Trainer::new(config.clone()).unwrap().fit(&data).unwrap();
        let expected_text = expected.model.to_text().unwrap();
        for cache in [0.000001, 0.01] {
            for threads in [1, 2, 4] {
                let mut replay = config.clone();
                replay.histogram_pool_size = cache;
                replay.num_threads = threads;
                let actual = Trainer::new(replay).unwrap().fit(&data).unwrap();
                assert_eq!(actual.model.to_text().unwrap(), expected_text);
            }
        }
    }
}
