//! TreeSHAP structural checks derived from the pinned LightGBM source.
//!
//! Copyright (c) 2016-2026 Microsoft Corporation and The LightGBM developers.
//! Licensed under the MIT License.

use lightgbm_rust::model::ImportanceType;
use lightgbm_rust::{Error, Model, Tree};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn model(features: usize, classes: usize, trees: &[String], average: bool) -> Model {
    let names = (0..features)
        .map(|i| format!("f{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    let infos = vec!["none"; features].join(" ");
    let mut text = format!(
        "tree\nversion=v4\nnum_class={classes}\nnum_tree_per_iteration={classes}\nlabel_index=0\nmax_feature_idx={}\nobjective=custom\nfeature_names={names}\nfeature_infos={infos}\n",
        features - 1
    );
    if average {
        text.push_str("average_output\n");
    }
    for (i, tree) in trees.iter().enumerate() {
        text.push_str(&format!("\nTree={i}\n{tree}\n\n"));
    }
    text.push_str("end of trees\n");
    Model::from_text(&text).unwrap()
}

fn constant(value: f64) -> String {
    format!("num_leaves=1\nnum_cat=0\nleaf_value={value}\nis_linear=0\nshrinkage=1")
}

fn stump(feature: usize, threshold: f64, decision: u8) -> String {
    format!(
        "num_leaves=2\nnum_cat=0\nsplit_feature={feature}\nsplit_gain=0.1\nthreshold={threshold}\ndecision_type={decision}\nleft_child=-1\nright_child=-2\nleaf_value=2 8\nleaf_count=1 3\ninternal_count=4\nis_linear=0\nshrinkage=0.01"
    )
}

fn repeated_tree() -> String {
    "num_leaves=5\nnum_cat=0\nsplit_feature=0 2 0 2\nsplit_gain=1 2 -1 0\nthreshold=0 1 2 1\ndecision_type=0 0 0 0\nleft_child=1 -1 -3 -4\nright_child=2 -2 3 -5\nleaf_value=1 5 -2 4 8\nleaf_count=1 3 2 1 1\ninternal_count=8 4 4 2\nis_linear=0\nshrinkage=1".into()
}

fn close(actual: f64, expected: f64) {
    assert!(actual.is_finite(), "nonfinite contribution {actual}");
    assert!(
        (actual - expected).abs() <= 1e-12 + 1e-10 * expected.abs(),
        "{actual} != {expected}"
    );
}

fn additive(model: &Model, row: &[f64]) {
    let contributions = model.predict_contrib_row(row).unwrap();
    assert_eq!(contributions.len(), model.num_classes());
    for (class, raw) in model.predict_raw_row(row).unwrap().iter().enumerate() {
        assert_eq!(contributions[class].len(), model.num_features() + 1);
        close(contributions[class].iter().sum(), *raw);
    }
}

#[test]
fn one_feature_contribution_is_prediction_minus_count_weighted_bias() {
    let model = model(3, 1, &[stump(2, 0.0, 0)], false);
    for (value, prediction) in [(-1.0, 2.0), (1.0, 8.0), (f64::NAN, 2.0)] {
        let output = model.predict_contrib_row(&[99.0, -99.0, value]).unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0][..2], [0.0, 0.0]);
        close(output[0][2], prediction - 6.5);
        close(output[0][3], 6.5);
        additive(&model, &[99.0, -99.0, value]);
    }
}

// Independent coalition evaluator for the numeric structural tree. An unknown
// feature follows both children with the stored sample-count proportions.
fn coalition(tree: &Tree, node: i32, known: usize, row: &[f64]) -> f64 {
    if node < 0 {
        return tree.leaf_values()[(!node) as usize];
    }
    let i = node as usize;
    let feature = tree.split_features()[i];
    let left = tree.left_children()[i];
    let right = tree.right_children()[i];
    if known & (1 << feature) != 0 {
        return coalition(
            tree,
            if row[feature] <= tree.thresholds()[i] {
                left
            } else {
                right
            },
            known,
            row,
        );
    }
    let count = |child: i32| {
        (if child < 0 {
            tree.leaf_counts()[(!child) as usize]
        } else {
            tree.internal_counts()[child as usize]
        }) as f64
    };
    (count(left) * coalition(tree, left, known, row)
        + count(right) * coalition(tree, right, known, row))
        / tree.internal_counts()[i] as f64
}

#[test]
fn repeated_features_match_exhaustive_two_feature_coalitions() {
    let model = model(3, 1, &[repeated_tree()], false);
    for x in [-1.0, 1.0, 3.0] {
        for z in [-1.0, 2.0] {
            let row = [x, 123.0, z];
            let tree = &model.trees()[0];
            let v = |known| coalition(tree, 0, known, &row);
            let output = &model.predict_contrib_row(&row).unwrap()[0];
            close(output[0], 0.5 * (v(1) - v(0) + v(5) - v(4)));
            close(output[1], 0.0);
            close(output[2], 0.5 * (v(4) - v(0) + v(5) - v(1)));
            close(output[3], v(0));
            additive(&model, &row);
        }
    }
}

#[test]
fn three_feature_repeated_paths_match_all_eight_coalitions() {
    let nodes = 15;
    let leaves = 16;
    let child = |index: usize| {
        if index < nodes {
            index as i32
        } else {
            !((index - nodes) as i32)
        }
    };
    let left: Vec<_> = (0..nodes).map(|i| child(2 * i + 1)).collect();
    let right: Vec<_> = (0..nodes).map(|i| child(2 * i + 2)).collect();
    let leaf_counts: Vec<_> = (0..leaves).map(|i| i % 3 + 1).collect();
    let mut internal_counts = vec![0; nodes];
    for i in (0..nodes).rev() {
        let count = |child: i32| {
            if child < 0 {
                leaf_counts[(!child) as usize]
            } else {
                internal_counts[child as usize]
            }
        };
        internal_counts[i] = count(left[i]) + count(right[i]);
    }
    let join = |values: Vec<String>| values.join(" ");
    let tree = format!(
        "num_leaves={leaves}\nnum_cat=0\nsplit_feature={}\nthreshold={}\ndecision_type={}\nleft_child={}\nright_child={}\nleaf_value={}\nleaf_count={}\ninternal_count={}\nis_linear=0\nshrinkage=1",
        join(
            (0..nodes)
                .map(|i| ((i * 7 + i / 2) % 3).to_string())
                .collect()
        ),
        join(
            (0..nodes)
                .map(|i| (i as f64 % 3.0 - 0.5).to_string())
                .collect()
        ),
        vec!["0"; nodes].join(" "),
        join(left.iter().map(ToString::to_string).collect()),
        join(right.iter().map(ToString::to_string).collect()),
        join(
            (0..leaves)
                .map(|i| ((i * 7 % 13) as f64 - 6.0).to_string())
                .collect()
        ),
        join(leaf_counts.iter().map(ToString::to_string).collect()),
        join(internal_counts.iter().map(ToString::to_string).collect())
    );
    let model = model(3, 1, &[tree], false);
    for x in [-1.0, 0.0, 2.0] {
        for y in [-1.0, 0.0, 2.0] {
            for z in [-1.0, 0.0, 2.0] {
                let row = [x, y, z];
                let v: Vec<_> = (0..8)
                    .map(|known| coalition(&model.trees()[0], 0, known, &row))
                    .collect();
                let output = model.predict_contrib_row(&row).unwrap();
                for (feature, &contribution) in output[0][..3].iter().enumerate() {
                    let mut expected = 0.0;
                    for known in 0usize..8 {
                        if known & (1 << feature) == 0 {
                            let weight = if known.count_ones() == 1 {
                                1.0 / 6.0
                            } else {
                                1.0 / 3.0
                            };
                            expected += weight * (v[known | (1 << feature)] - v[known]);
                        }
                    }
                    close(contribution, expected);
                }
                close(output[0][3], v[0]);
                additive(&model, &row);
            }
        }
    }
}

#[test]
fn multiclass_ranges_and_rf_contributions_are_raw_sums() {
    let trees = [
        constant(1.0),
        constant(-2.0),
        constant(3.0),
        stump(0, 0.0, 0),
        stump(1, 1.0, 0),
        constant(7.0),
    ];
    let model = model(2, 3, &trees, true);
    let row = [2.0, 0.0];
    let all = model.predict_contrib_row(&row).unwrap();
    let range = model.iteration_range(1..2).unwrap();
    let selected = model
        .predict_contrib_row_with_iterations(&row, range)
        .unwrap();
    for class in 0..3 {
        close(
            all[class].iter().sum(),
            model.predict_raw_row(&row).unwrap()[class],
        );
        close(
            selected[class].iter().sum(),
            model.predict_raw_row_with_iterations(&row, range).unwrap()[class],
        );
        close(all[class][2] - selected[class][2], [1.0, -2.0, 3.0][class]);
        close(
            all[class].iter().sum::<f64>() / 2.0,
            model.predict_row(&row).unwrap()[class],
        );
    }
    assert_ne!(all, selected);
}

#[test]
fn sparse_getter_keeps_explicit_tiny_values_and_validates_rows() {
    let model = model(2, 1, &[stump(1, -1e-36, 0)], false);
    let dense = model.predict_contrib_row(&[0.0, -1e-36]).unwrap();
    let sparse = model.predict_contrib_csr_row(&[1], &[-1e-36]).unwrap();
    close(dense[0].iter().sum(), 8.0);
    close(sparse[0].iter().sum(), 2.0);
    assert_ne!(dense, sparse);
    assert_eq!(
        model.predict_contrib_csr_row(&[], &[]).unwrap(),
        model.predict_contrib_row(&[0.0, 0.0]).unwrap()
    );
    assert!(model.predict_contrib_row(&[0.0]).is_err());
    assert!(model.predict_contrib_csr_row(&[0], &[]).is_err());
    assert!(model.predict_contrib_csr_row(&[1, 0], &[1.0, 2.0]).is_err());
    assert!(model.predict_contrib_csr_row(&[0, 0], &[1.0, 2.0]).is_err());
    assert!(model.predict_contrib_csr_row(&[2], &[1.0]).is_err());
}

#[test]
fn categorical_and_missing_routes_use_the_prediction_decision() {
    let category = stump(0, 0.0, 3).replace("num_cat=0", "num_cat=1")
        + "\ncat_boundaries=0 1\ncat_threshold=5";
    for (tree, routes) in [
        (
            category,
            vec![
                (0.0, 2.0),
                (2.0, 2.0),
                (-0.5, 2.0),
                (-1.0, 8.0),
                (1.0, 8.0),
                (99.0, 8.0),
                (f64::NAN, 8.0),
            ],
        ),
        (
            stump(0, -1.0, 10),
            vec![(f64::NAN, 2.0), (0.0, 8.0), (-2.0, 2.0)],
        ),
        (
            stump(0, -1.0, 6),
            vec![(f64::NAN, 2.0), (0.0, 2.0), (1e-36, 2.0), (1.0, 8.0)],
        ),
    ] {
        let model = model(1, 1, &[tree], false);
        for (value, expected) in routes {
            let output = &model.predict_contrib_row(&[value]).unwrap()[0];
            close(output[0], expected - 6.5);
            close(output[1], 6.5);
            additive(&model, &[value]);
        }
    }
}

#[test]
fn importance_uses_positive_native_float_gains_and_iteration_selection() {
    let model = model(3, 1, &[repeated_tree(), stump(1, 0.0, 0)], true);
    assert_eq!(
        model.feature_importance(ImportanceType::Split).unwrap(),
        vec![1.0, 1.0, 1.0]
    );
    let gain = model.feature_importance(ImportanceType::Gain).unwrap();
    assert_eq!(gain, vec![1.0, 0.1_f32 as f64, 2.0]);
    let range = model.iteration_range(1..2).unwrap();
    assert_eq!(
        model
            .feature_importance_with_iterations(ImportanceType::Split, range)
            .unwrap(),
        vec![0.0, 1.0, 0.0]
    );
    assert_eq!(
        model
            .feature_importance_with_iterations(ImportanceType::Gain, range)
            .unwrap(),
        vec![0.0, 0.1_f32 as f64, 0.0]
    );
    // A positive decimal below f32 range is zero in native model storage.
    let tiny = self::model(
        1,
        1,
        &[stump(0, 0.0, 0).replace("split_gain=0.1", "split_gain=1e-100")],
        false,
    );
    assert_eq!(
        tiny.feature_importance(ImportanceType::Split).unwrap(),
        vec![0.0]
    );
}

#[test]
fn linear_trees_are_unsupported_even_with_empty_coefficients_or_outside_range() {
    let linear = stump(0, 0.0, 0).replace("is_linear=0", "is_linear=1")
        + "\nleaf_const=2 8\nnum_features=0 0\nleaf_features=\nleaf_coeff=";
    let model = model(1, 1, &[constant(1.0), linear], false);
    let range = model.iteration_range(0..1).unwrap();
    for result in [
        model.predict_contrib_row(&[1.0]),
        model.predict_contrib_row_with_iterations(&[1.0], range),
        model.predict_contrib_csr_row(&[], &[]),
    ] {
        assert!(matches!(result, Err(Error::Unsupported(message)) if message.contains("linear")));
    }
    assert!(model.feature_importance(ImportanceType::Gain).is_ok());
}

#[test]
fn missing_zero_or_inconsistent_counts_return_clear_errors_not_nan() {
    let original = stump(0, 0.0, 0);
    for tree in [
        original.replace("leaf_count=1 3\n", ""),
        original.replace("internal_count=4\n", ""),
        original.replace("leaf_count=1 3", "leaf_count=0 4"),
        original.replace("internal_count=4", "internal_count=5"),
    ] {
        let model = model(1, 1, &[tree], false);
        assert!(
            matches!(model.predict_contrib_row(&[1.0]), Err(Error::InvalidModel(message)) if message.contains("count"))
        );
        assert!(model.feature_importance(ImportanceType::Split).is_ok());
    }
    let constant = model(1, 1, &[constant(7.0)], false);
    assert_eq!(
        constant.predict_contrib_row(&[f64::NAN]).unwrap(),
        vec![vec![0.0, 7.0]]
    );
}

fn chain(nodes: usize) -> String {
    let join = |values: Vec<String>| values.join(" ");
    format!(
        "num_leaves={}\nnum_cat=0\nsplit_feature={}\nthreshold={}\ndecision_type={}\nleft_child={}\nright_child={}\nleaf_value={}\nleaf_count={}\ninternal_count={}\nis_linear=0\nshrinkage=1",
        nodes + 1,
        vec!["0"; nodes].join(" "),
        vec!["0"; nodes].join(" "),
        vec!["0"; nodes].join(" "),
        join((0..nodes).map(|i| (!(i as i32)).to_string()).collect()),
        join(
            (0..nodes)
                .map(|i| if i + 1 == nodes {
                    (!(nodes as i32)).to_string()
                } else {
                    (i + 1).to_string()
                })
                .collect()
        ),
        vec!["1"; nodes + 1].join(" "),
        vec!["1"; nodes + 1].join(" "),
        join((0..nodes).map(|i| (nodes + 1 - i).to_string()).collect())
    )
}

#[test]
fn depth_boundary_and_ensemble_path_work_limit_are_checked() {
    let boundary = model(1, 1, &[chain(256)], false);
    additive(&boundary, &[1.0]);
    additive(&boundary, &[-1.0]);
    let excessive_work = model(1, 1, &vec![chain(256); 7], false);
    assert!(
        matches!(excessive_work.predict_contrib_row(&[1.0]), Err(Error::Unsupported(message)) if message.contains("path-work"))
    );
}

#[test]
fn numerical_overflow_is_an_error_not_a_nonfinite_contribution() {
    let tree = stump(0, 0.0, 0).replace("leaf_value=2 8", "leaf_value=-1.7e308 1.7e308");
    let model = model(1, 1, &[tree], false);
    assert!(
        matches!(model.predict_contrib_row(&[-1.0]), Err(Error::Unsupported(message)) if message.contains("finite numerical range"))
    );
}

#[test]
fn deep_valid_tree_is_rejected_without_recursive_walk_or_quadratic_allocation() {
    let model = model(1, 1, &[chain(2048)], false);
    close(model.predict_raw_row(&[1.0]).unwrap()[0], 1.0);
    assert!(
        matches!(model.predict_contrib_row(&[1.0]), Err(Error::Unsupported(message)) if message.contains("depth"))
    );
}

#[test]
fn contribution_shape_limit_is_checked_before_output_allocation() {
    let model = model(4096, 4096, &vec![constant(1.0); 4096], false);
    assert!(
        matches!(model.predict_contrib_row(&vec![0.0; 4096]), Err(Error::Unsupported(message)) if message.contains("output"))
    );
}

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn explain_fixture() -> Value {
    let suite: Value = serde_json::from_str(
        &std::fs::read_to_string(fixture_path().join("explain.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(suite["schema_version"], 1);
    assert_eq!(
        suite["upstream_commit"],
        "5d84f3f222b41409a6983d2ac90513ad9702fa9e"
    );
    assert_eq!(suite["gain_precision"]["storage_type"], "f32");
    assert_eq!(suite["gain_precision"]["accumulation_type"], "f64");
    suite
}

fn compare_json(actual: &[f64], expected: &Value) {
    let expected = expected.as_array().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (&actual, expected) in actual.iter().zip(expected) {
        close(actual, expected.as_f64().unwrap());
    }
}

#[test]
fn saved_native_treeshap_and_importance_match_dense_csr_ranges_and_roundtrip() {
    let suite = explain_fixture();
    let cases = suite["cases"].as_array().unwrap();
    let names: BTreeSet<_> = cases
        .iter()
        .map(|case| case["name"].as_str().unwrap())
        .collect();
    assert_eq!(names.len(), cases.len());
    for base in [
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
        "multiclass_empty_class",
        "multiclass_start1_count1",
        "multiclass_start1_count2",
        "rf_average_start1_count1",
        "rf_average_start1_count2",
    ] {
        assert!(names.contains(base), "missing {base}");
        assert!(
            names.contains(format!("{base}_csr").as_str()),
            "missing {base}_csr"
        );
    }
    for case in cases {
        let imported =
            Model::load(fixture_path().join(case["model_file"].as_str().unwrap())).unwrap();
        let roundtrip = Model::from_text(&imported.to_text().unwrap()).unwrap();
        for model in [&imported, &roundtrip] {
            let start = case["params"]["start_iteration"].as_u64().unwrap_or(0) as usize;
            let count = case["params"]["num_iteration"]
                .as_u64()
                .map(|v| v as usize)
                .unwrap_or(model.num_iterations() - start);
            let range = model.iteration_range(start..start + count).unwrap();
            compare_json(
                &model
                    .feature_importance_with_iterations(ImportanceType::Split, range)
                    .unwrap(),
                &case["importance_split"],
            );
            compare_json(
                &model
                    .feature_importance_with_iterations(ImportanceType::Gain, range)
                    .unwrap(),
                &case["importance_gain"],
            );
            for (row_index, row) in case["rows"].as_array().unwrap().iter().enumerate() {
                let row: Vec<f64> = row
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(f64::NAN))
                    .collect();
                let contributions = if case.get("sparse").is_some() {
                    let sparse = &case["sparse"];
                    let start = sparse["indptr"][row_index].as_u64().unwrap() as usize;
                    let end = sparse["indptr"][row_index + 1].as_u64().unwrap() as usize;
                    let indices: Vec<_> = sparse["indices"].as_array().unwrap()[start..end]
                        .iter()
                        .map(|v| v.as_u64().unwrap() as usize)
                        .collect();
                    let values: Vec<_> = sparse["data"].as_array().unwrap()[start..end]
                        .iter()
                        .map(|v| v.as_f64().unwrap_or(f64::NAN))
                        .collect();
                    model
                        .predict_contrib_csr_row_with_iterations(&indices, &values, range)
                        .unwrap()
                } else {
                    model
                        .predict_contrib_row_with_iterations(&row, range)
                        .unwrap()
                };
                assert_eq!(contributions.len(), model.num_classes(), "{}", case["name"]);
                let flat: Vec<_> = contributions.iter().flatten().copied().collect();
                compare_json(&flat, &case["contributions"][row_index]);
                for (class, output) in contributions.iter().enumerate() {
                    assert_eq!(output.len(), model.num_features() + 1);
                    close(
                        output.iter().sum(),
                        case["raw"][row_index][class].as_f64().unwrap(),
                    );
                }
            }
        }
    }
}

#[test]
fn saved_native_linear_contribution_errors_are_explicit() {
    let suite = explain_fixture();
    let errors = suite["expected_errors"].as_array().unwrap();
    assert_eq!(errors.len(), 4);
    for case in errors {
        assert!(case["expected_error"].as_str().unwrap().contains("linear"));
        let model = Model::load(fixture_path().join(case["model_file"].as_str().unwrap())).unwrap();
        assert!(
            matches!(model.predict_contrib_row(&vec![0.0; model.num_features()]), Err(Error::Unsupported(message)) if message.contains("linear"))
        );
    }
}

#[test]
fn iteration_ranges_from_another_model_are_checked() {
    let short = model(1, 1, &[constant(1.0)], false);
    let long = model(1, 1, &[constant(1.0), constant(2.0)], false);
    let range = long.all_iterations();
    assert!(
        short
            .predict_contrib_row_with_iterations(&[1.0], range)
            .is_err()
    );
    assert!(
        short
            .feature_importance_with_iterations(ImportanceType::Split, range)
            .is_err()
    );
}
