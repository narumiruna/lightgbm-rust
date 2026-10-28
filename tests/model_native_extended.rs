use lightgbm_rust::Model;
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}
fn suite(name: &str) -> Value {
    let value: Value =
        serde_json::from_str(&std::fs::read_to_string(fixtures().join(name)).unwrap()).unwrap();
    assert_eq!(value["schema_version"], 1);
    assert_eq!(
        value["upstream_commit"],
        "5d84f3f222b41409a6983d2ac90513ad9702fa9e"
    );
    value
}
fn compare(actual: &[f64], expected: &Value) {
    let expected = expected.as_array().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (&actual, expected) in actual.iter().zip(expected) {
        let expected = expected.as_f64().unwrap();
        assert!(actual.is_finite());
        assert!(
            (actual - expected).abs() <= 1e-12 + 1e-10 * expected.abs(),
            "{actual} != {expected}"
        );
    }
}

#[test]
fn imported_linear_leaves_match_native_predictions_and_roundtrip() {
    let suite = suite("prediction_linear.json");
    let cases = suite["cases"].as_array().unwrap();
    let names: BTreeSet<_> = cases.iter().map(|c| c["name"].as_str().unwrap()).collect();
    for required in [
        "linear_independent_lambda0",
        "linear_independent_lambda05",
        "linear_rank_deficient_lambda0",
        "linear_rank_deficient_lambda05",
    ] {
        assert!(names.contains(required));
    }
    for case in cases {
        let model = Model::load(fixtures().join(case["model_file"].as_str().unwrap())).unwrap();
        assert!(
            model
                .trees()
                .iter()
                .all(|tree| tree.linear_leaves().is_some())
        );
        let roundtrip = Model::from_text(&model.to_text().unwrap()).unwrap();
        for model in [&model, &roundtrip] {
            for (i, row) in case["rows"].as_array().unwrap().iter().enumerate() {
                let row: Vec<_> = row
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(f64::NAN))
                    .collect();
                compare(&model.predict_raw_row(&row).unwrap(), &case["raw"][i]);
                compare(&model.predict_row(&row).unwrap(), &case["transformed"][i]);
                let leaves: Vec<usize> = serde_json::from_value(case["leaves"][i].clone()).unwrap();
                assert_eq!(model.predict_leaf_row(&row).unwrap(), leaves);
            }
        }
    }
}

#[test]
fn native_csr_preserves_explicit_tiny_values_without_dense_row_allocation() {
    let suite = suite("prediction_sparse.json");
    let cases = suite["cases"].as_array().unwrap();
    assert!(cases.len() >= 18);
    for case in cases {
        let model = Model::load(fixtures().join(case["model_file"].as_str().unwrap())).unwrap();
        for (i, row) in case["rows"].as_array().unwrap().iter().enumerate() {
            let mut indices = Vec::new();
            let mut values = Vec::new();
            for (feature, value) in row.as_array().unwrap().iter().enumerate() {
                let value = value.as_f64().unwrap_or(f64::NAN);
                if value != 0.0 {
                    indices.push(feature);
                    values.push(value);
                }
            }
            compare(
                &model.predict_raw_csr_row(&indices, &values).unwrap(),
                &case["raw"][i],
            );
            compare(
                &model.predict_csr_row(&indices, &values).unwrap(),
                &case["transformed"][i],
            );
            let leaves: Vec<usize> = serde_json::from_value(case["leaves"][i].clone()).unwrap();
            assert_eq!(
                model.predict_leaf_csr_row(&indices, &values).unwrap(),
                leaves
            );
        }
        assert!(
            model
                .predict_csr_row(&[model.num_features()], &[1.0])
                .is_err()
        );
        assert!(model.predict_csr_row(&[0, 0], &[1.0, 2.0]).is_err());
        assert!(model.predict_csr_row(&[0], &[]).is_err());
    }
}

#[test]
fn malformed_linear_arrays_are_rejected_before_prediction() {
    let suite = suite("prediction_linear.json");
    let case = &suite["cases"][0];
    let text =
        std::fs::read_to_string(fixtures().join(case["model_file"].as_str().unwrap())).unwrap();
    // Remove byte lengths so each mutation reaches linear-array validation.
    let text = text
        .lines()
        .filter(|line| !line.starts_with("tree_sizes="))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    for prefix in [
        "leaf_const=",
        "num_features=",
        "leaf_features=",
        "leaf_coeff=",
    ] {
        let line = text.lines().find(|line| line.starts_with(prefix)).unwrap();
        let missing = text.replacen(&format!("{line}\n"), "", 1);
        assert!(Model::from_text(&missing).is_err(), "missing {prefix}");
    }
    let line = text
        .lines()
        .find(|line| line.starts_with("num_features="))
        .unwrap();
    let mut counts: Vec<String> = line
        .trim_start_matches("num_features=")
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    counts[0] = usize::MAX.to_string();
    let huge = text.replacen(line, &format!("num_features={}", counts.join(" ")), 1);
    assert!(Model::from_text(&huge).is_err());
    let line = text
        .lines()
        .find(|line| line.starts_with("leaf_const="))
        .unwrap();
    let mut constants: Vec<String> = line
        .trim_start_matches("leaf_const=")
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    constants[0] = "NaN".into();
    assert!(
        Model::from_text(&text.replacen(line, &format!("leaf_const={}", constants.join(" ")), 1))
            .is_err()
    );
}
