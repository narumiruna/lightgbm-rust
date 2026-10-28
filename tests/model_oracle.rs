//! Saved-reference inference checks. No native library is loaded by these tests.
use lightgbm_rust::model::Model;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};

const UPSTREAM: &str = "5d84f3f222b41409a6983d2ac90513ad9702fa9e";
const REQUIRED: &[&str] = &[
    "numeric_l2",
    "binary",
    "numeric_weighted",
    "missing_nan",
    "missing_zero",
    "categorical",
    "multiclass",
    "single_leaf",
    "regression_sqrt",
    "binary_sigmoid",
    "multiclassova",
    "poisson",
    "gamma",
    "tweedie",
    "cross_entropy",
    "cross_entropy_lambda",
    "rf_average",
    "missing_none",
];

#[derive(Clone, Deserialize)]
struct Suite {
    schema_version: usize,
    upstream_commit: String,
    cases: Vec<Case>,
}

#[derive(Clone, Deserialize)]
struct Case {
    name: String,
    model_file: String,
    rows: Vec<Vec<Option<f64>>>,
    raw: Vec<Vec<f64>>,
    transformed: Vec<Vec<f64>>,
    leaves: Vec<Vec<usize>>,
    #[serde(default)]
    params: serde_json::Value,
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn suite() -> Suite {
    let text = std::fs::read_to_string(fixtures().join("prediction.json"))
        .expect("required oracle fixtures are missing");
    serde_json::from_str(&text).expect("invalid oracle fixture schema")
}

fn compare(actual: &[f64], expected: &[f64]) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err("prediction width mismatch".into());
    }
    for (&actual, &expected) in actual.iter().zip(expected) {
        if !actual.is_finite()
            || !expected.is_finite()
            || (actual - expected).abs() > 1e-12 + 1e-10 * expected.abs()
        {
            return Err(format!(
                "numerical mismatch: actual={actual}, expected={expected}"
            ));
        }
    }
    Ok(())
}

fn check(suite: &Suite) -> Result<(), String> {
    if suite.schema_version != 1 || suite.upstream_commit != UPSTREAM {
        return Err("oracle schema/provenance mismatch".into());
    }
    let mut names = BTreeSet::new();
    for case in &suite.cases {
        if !names.insert(case.name.as_str()) {
            return Err("duplicate oracle case".into());
        }
    }
    for name in REQUIRED {
        if !names.contains(name) {
            return Err(format!("missing required oracle case {name}"));
        }
    }
    for case in &suite.cases {
        let path = Path::new(&case.model_file);
        if path.is_absolute()
            || path
                .components()
                .any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err("oracle model path must be relative to fixtures".into());
        }
        let model =
            Model::load(fixtures().join(path)).map_err(|e| format!("{}: {e}", case.name))?;
        if case.rows.is_empty()
            || case.raw.len() != case.rows.len()
            || case.transformed.len() != case.rows.len()
            || case.leaves.len() != case.rows.len()
        {
            return Err(format!("{}: prediction row shape mismatch", case.name));
        }
        let roundtrip = Model::from_text(&model.to_text().map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        for (i, row) in case.rows.iter().enumerate() {
            if row.len() != model.num_features()
                || case.raw[i].len() != model.num_classes()
                || case.transformed[i].len() != model.num_classes()
                || case.leaves[i].len() != model.trees().len()
            {
                return Err(format!("{}: invalid oracle column shape", case.name));
            }
            let row: Vec<f64> = row.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
            for model in [&model, &roundtrip] {
                compare(
                    &model.predict_raw_row(&row).map_err(|e| e.to_string())?,
                    &case.raw[i],
                )
                .map_err(|e| format!("{} raw row {i}: {e}", case.name))?;
                compare(
                    &model.predict_row(&row).map_err(|e| e.to_string())?,
                    &case.transformed[i],
                )
                .map_err(|e| format!("{} transformed row {i}: {e}", case.name))?;
                if model.predict_leaf_row(&row).map_err(|e| e.to_string())? != case.leaves[i] {
                    return Err(format!("{}: leaf mismatch at row {i}", case.name));
                }
                let indices: Vec<usize> = row
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| (v != 0.0).then_some(i))
                    .collect();
                let values: Vec<f64> = indices.iter().map(|&i| row[i]).collect();
                compare(
                    &model
                        .predict_sparse_row(&indices, &values)
                        .map_err(|e| e.to_string())?,
                    &case.transformed[i],
                )?;
            }
        }
        let dense: Vec<f64> = case
            .rows
            .iter()
            .flatten()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect();
        let raw = model
            .predict_raw_dense(&dense, case.rows.len(), model.num_features())
            .map_err(|e| e.to_string())?;
        let transformed = model
            .predict_dense(&dense, case.rows.len(), model.num_features())
            .map_err(|e| e.to_string())?;
        let leaves = model
            .predict_leaf_dense(&dense, case.rows.len(), model.num_features())
            .map_err(|e| e.to_string())?;
        if leaves != case.leaves {
            return Err(format!("{}: batch leaf mismatch", case.name));
        }
        for i in 0..case.rows.len() {
            compare(&raw[i], &case.raw[i])?;
            compare(&transformed[i], &case.transformed[i])?;
        }
    }
    Ok(())
}

#[test]
fn native_iteration_ranges_and_empty_class_match() {
    let suite: Suite = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("prediction_ranges.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(suite.schema_version, 1);
    assert_eq!(suite.upstream_commit, UPSTREAM);
    let required: BTreeSet<_> = [
        "multiclass_empty_class",
        "multiclass_start1_count1",
        "multiclass_start1_count2",
        "rf_average_start1_count1",
        "rf_average_start1_count2",
    ]
    .into_iter()
    .collect();
    let found: BTreeSet<_> = suite.cases.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(found, required);
    for case in suite.cases {
        let model = Model::load(fixtures().join(case.model_file)).unwrap();
        let start = case.params["start_iteration"].as_u64().unwrap_or(0) as usize;
        let count = case.params["num_iteration"]
            .as_u64()
            .map(|v| v as usize)
            .unwrap_or(model.num_iterations());
        let range = model.iteration_range(start..start + count).unwrap();
        assert_eq!(case.rows.len(), case.raw.len());
        assert_eq!(case.rows.len(), case.transformed.len());
        assert_eq!(case.rows.len(), case.leaves.len());
        for (i, row) in case.rows.iter().enumerate() {
            let row: Vec<_> = row.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
            compare(
                &model.predict_raw_row_with_iterations(&row, range).unwrap(),
                &case.raw[i],
            )
            .unwrap();
            compare(
                &model.predict_row_with_iterations(&row, range).unwrap(),
                &case.transformed[i],
            )
            .unwrap();
            assert_eq!(
                model.predict_leaf_row_with_iterations(&row, range).unwrap(),
                case.leaves[i]
            );
        }
    }
}

#[test]
fn json_preserves_exact_f64_traversal_boundaries() {
    // serde_json needs float_roundtrip: its default fast parser can shift these
    // inputs by one ULP and change leaf routing, even at tolerance 1e-10.
    for text in [
        "1.0000000180025095e-35",
        "-1.0000000180025095e-35",
        "1.5000000000000002",
    ] {
        let expected: f64 = text.parse().unwrap();
        let actual: f64 = serde_json::from_str(text).unwrap();
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "inexact JSON f64 parser for {text}"
        );
    }
}

#[test]
fn pinned_predictions_match_and_survive_text_roundtrip() {
    let suite = suite();
    check(&suite).unwrap();
    eprintln!(
        "prediction coverage: {} pinned cases, {} required",
        suite.cases.len(),
        REQUIRED.len()
    );
}

#[test]
fn oracle_comparator_rejects_value_shape_and_coverage_mutants() {
    let original = suite();
    check(&original).unwrap();
    let mut mutant = original.clone();
    mutant.cases[0].raw[0][0] += 0.01;
    assert!(check(&mutant).is_err(), "value mutant escaped");
    let mut mutant = original.clone();
    mutant.cases[0].raw[0].push(0.0);
    assert!(check(&mutant).is_err(), "column shape mutant escaped");
    let mut mutant = original.clone();
    mutant.cases[0].leaves.pop();
    assert!(check(&mutant).is_err(), "row shape mutant escaped");
    let mut mutant = original.clone();
    mutant.cases.retain(|c| c.name != REQUIRED[0]);
    assert!(check(&mutant).is_err(), "missing required case escaped");
    let mut mutant = original.clone();
    mutant.upstream_commit = "wrong".into();
    assert!(check(&mutant).is_err(), "provenance mutant escaped");
}
