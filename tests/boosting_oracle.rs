use lightgbm_rust::{Config, Dataset, Model, Trainer};
use serde_json::Value;

fn config(case: &Value, rounds: usize) -> Config {
    let mut pairs: Vec<(String, String)> = case["params"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                v.as_str().map_or_else(|| v.to_string(), str::to_owned),
            )
        })
        .collect();
    pairs.push(("num_iterations".into(), rounds.to_string()));
    Config::from_pairs(pairs).unwrap_or_else(|e| panic!("{}: {e}", case["name"]))
}

fn rows(value: &Value) -> (Vec<f64>, usize, usize) {
    let rows = value.as_array().unwrap();
    (
        rows.iter()
            .flat_map(|r| {
                r.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(f64::NAN))
            })
            .collect(),
        rows.len(),
        rows[0].as_array().unwrap().len(),
    )
}

fn close(got: f64, expected: f64, context: &str) {
    assert!(
        (got - expected).abs() <= 1e-12 + 1e-10 * expected.abs(),
        "{context}: {got} != {expected}"
    );
}

#[test]
fn seeded_boosting_matches_native_structures_and_every_round() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/prediction_boosting.json")).unwrap();
    assert_eq!(
        fixture["upstream_commit"],
        "5d84f3f222b41409a6983d2ac90513ad9702fa9e"
    );
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 13);
    let mut seen = std::collections::BTreeSet::new();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        assert!(seen.insert(name));
        let count = case["training"]["num_boost_round"].as_u64().unwrap() as usize;
        let c = config(case, count);
        let (x, n, p) = rows(&case["training"]["rows"]);
        let mut data = Dataset::from_dense(
            &x,
            n,
            p,
            serde_json::from_value(case["training"]["labels"].clone()).unwrap(),
            &c,
        )
        .unwrap();
        data = data
            .with_weights(serde_json::from_value(case["training"]["weights"].clone()).unwrap())
            .unwrap();
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
        let (probe, m, _) = rows(&case["rows"]);
        for step in case["training"]["steps"].as_array().unwrap() {
            let round = step["iteration"].as_u64().unwrap() as usize + 1;
            let result = Trainer::new(config(case, round))
                .unwrap()
                .fit(&data)
                .unwrap_or_else(|e| panic!("{name} round {round}: {e}"));
            let actual = result.model.predict_raw_dense(&probe, m, p).unwrap();
            for (got, expected) in actual.iter().flatten().zip(
                step["raw"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|r| r.as_array().unwrap()),
            ) {
                close(
                    *got,
                    expected.as_f64().unwrap(),
                    &format!("{name} round {round} raw"),
                );
            }
            let text = result.model.to_text().unwrap();
            let shrinkages: Vec<f64> = text
                .lines()
                .filter_map(|line| line.strip_prefix("shrinkage="))
                .map(|v| v.parse().unwrap())
                .collect();
            assert_eq!(
                shrinkages.len(),
                step["tree_shrinkage"].as_array().unwrap().len()
            );
            for (got, expected) in shrinkages
                .iter()
                .zip(step["tree_shrinkage"].as_array().unwrap())
            {
                close(
                    *got,
                    expected.as_f64().unwrap(),
                    &format!("{name} round {round} shrinkage"),
                );
            }
        }
        let result = Trainer::new(c).unwrap().fit(&data).unwrap();
        let native = Model::from_text(
            &std::fs::read_to_string(format!(
                "{}/tests/fixtures/{}",
                env!("CARGO_MANIFEST_DIR"),
                case["model_file"].as_str().unwrap()
            ))
            .unwrap(),
        )
        .unwrap();
        assert_eq!(result.model.trees().len(), native.trees().len(), "{name}");
        for (index, (got, expected)) in result.model.trees().iter().zip(native.trees()).enumerate()
        {
            assert_eq!(
                got.split_features(),
                expected.split_features(),
                "{name} tree {index} features"
            );
            assert_eq!(
                got.thresholds(),
                expected.thresholds(),
                "{name} tree {index} thresholds"
            );
            assert_eq!(
                got.left_children(),
                expected.left_children(),
                "{name} tree {index} left"
            );
            assert_eq!(
                got.right_children(),
                expected.right_children(),
                "{name} tree {index} right"
            );
            assert_eq!(
                got.decision_types(),
                expected.decision_types(),
                "{name} tree {index} missing"
            );
            assert_eq!(
                got.leaf_counts(),
                expected.leaf_counts(),
                "{name} tree {index} counts"
            );
            for (&a, &b) in got.leaf_values().iter().zip(expected.leaf_values()) {
                close(a, b, name);
            }
        }
        let expected_leaves: Vec<Vec<usize>> =
            serde_json::from_value(case["leaves"].clone()).unwrap();
        assert_eq!(
            result.model.predict_leaf_dense(&probe, m, p).unwrap(),
            expected_leaves,
            "{name}"
        );
        let transformed = result.model.predict_dense(&probe, m, p).unwrap();
        for (got, expected) in transformed.iter().flatten().zip(
            case["transformed"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|r| r.as_array().unwrap()),
        ) {
            close(*got, expected.as_f64().unwrap(), name);
        }
        let mut valid = data
            .validation_from_dense(&x, n, p, data.labels().to_vec())
            .unwrap();
        if let Some(groups) = data.groups() {
            valid = valid.with_groups(groups.to_vec()).unwrap();
        }
        let repeated = Trainer::new(config(case, count))
            .unwrap()
            .fit_with_validation(&data, &[("valid", &valid)])
            .unwrap();
        assert_eq!(
            repeated.model.to_text().unwrap(),
            result.model.to_text().unwrap(),
            "{name} repeatability"
        );
        if name.starts_with("rf_") {
            let raw = result.model.predict_raw_dense(&x, n, p).unwrap();
            for (row, scores) in raw.iter().enumerate() {
                for (class, &score) in scores.iter().enumerate() {
                    close(
                        result.training_scores[class * n + row],
                        score / count as f64,
                        "RF training average versus model raw sum",
                    );
                }
            }
        }
    }
}
