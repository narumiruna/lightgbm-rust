use super::*;
use serde_json::Value;

// Single-leaf checked entry for fixture assertions and batch equivalence tests.
impl ObjectiveState<'_> {
    fn renew_leaf_output(
        &self,
        class: usize,
        rows: &[usize],
        scores: &[f64],
        newton: f64,
    ) -> Result<f64> {
        let mut outputs = [newton];
        self.renew_leaf_outputs(class, &[rows], scores, &mut outputs)?;
        Ok(outputs[0])
    }
}

#[test]
fn all_cpu_objectives_and_state_progression_match_native_driver() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/extended_objectives.json"
    ))
    .unwrap();
    let mut families = std::collections::BTreeSet::new();
    for case in fixture["cases"].as_array().unwrap() {
        families.insert(case["objective"].as_str().unwrap());
        let pairs = case["params"].as_object().unwrap().iter().map(|(k, v)| {
            let value = match v {
                Value::String(s) => s.clone(),
                Value::Array(values) => values
                    .iter()
                    .map(Value::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
                _ => v.to_string(),
            };
            (k.clone(), value)
        });
        let c = Config::from_pairs(pairs).unwrap();
        let labels: Vec<f32> = serde_json::from_value(case["labels"].clone()).unwrap();
        let weights: Option<Vec<f32>> = case
            .get("weights")
            .map(|v| serde_json::from_value(v.clone()).unwrap());
        let groups: Option<Vec<usize>> = case
            .get("groups")
            .map(|v| serde_json::from_value(v.clone()).unwrap());
        let positions: Option<Vec<i32>> = case
            .get("positions")
            .map(|v| serde_json::from_value(v.clone()).unwrap());
        let scores: Vec<f64> = serde_json::from_value(case["scores"].clone()).unwrap();
        let mut state = ObjectiveState::new(
            &c,
            ObjectiveInput {
                labels: &labels,
                weights: weights.as_deref(),
                groups: groups.as_deref(),
                positions: positions.as_deref(),
            },
        )
        .unwrap();
        assert_eq!(state.num_classes(), c.num_class);
        for (class, initial) in case["initial_scores"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            if let Some(expected) = initial.as_f64() {
                assert!(
                    (state.initial_score(class).unwrap() - expected).abs() <= 1e-12,
                    "{} class {class} prior",
                    case["name"]
                );
            } else {
                assert!(
                    state.initial_score(class).is_err(),
                    "nonfinite native initialization must return an error"
                );
            }
        }
        let (g, h) = state.gradients(&scores).unwrap();
        let expected_g: Vec<f32> = serde_json::from_value(case["gradients"].clone()).unwrap();
        let expected_h: Vec<f32> = serde_json::from_value(case["hessians"].clone()).unwrap();
        assert_eq!(g, expected_g, "{} gradients", case["name"]);
        assert_eq!(h, expected_h, "{} Hessians", case["name"]);
        if let Some(steps) = case["steps"].as_array() {
            for step in steps {
                let step_scores: Vec<f64> = serde_json::from_value(step["scores"].clone()).unwrap();
                let (g, h) = state.gradients(&step_scores).unwrap();
                let eg: Vec<f32> = serde_json::from_value(step["gradients"].clone()).unwrap();
                let eh: Vec<f32> = serde_json::from_value(step["hessians"].clone()).unwrap();
                assert_eq!(g, eg, "{} sequential gradients", case["name"]);
                assert_eq!(h, eh, "{} sequential Hessians", case["name"]);
            }
        }
        if let Some(renewals) = case["renewals"].as_array() {
            for renewal in renewals {
                let rows: Vec<usize> = serde_json::from_value(renewal["rows"].clone()).unwrap();
                let output = state
                    .renew_leaf_output(
                        renewal["class"].as_u64().unwrap() as usize,
                        &rows,
                        &scores,
                        renewal["newton_output"].as_f64().unwrap(),
                    )
                    .unwrap();
                assert!(
                    (output - renewal["expected_output"].as_f64().unwrap()).abs() <= 1e-12,
                    "{} leaf output: {output}",
                    case["name"]
                );
            }
        }
    }
    assert_eq!(
        families.len(),
        16,
        "every CPU built-in objective factory must be covered"
    );
}

#[test]
fn source_metadata_class_skips_and_checked_leaf_access() {
    for (objective, classes, sqrt, sigmoid, text) in [
        (Objective::Regression, 1, false, 1.0, "regression"),
        (Objective::Regression, 1, true, 1.0, "regression sqrt"),
        (Objective::RegressionL1, 1, true, 1.0, "regression_l1 sqrt"),
        (
            Objective::Binary,
            1,
            false,
            1.23456789,
            "binary sigmoid:1.23457",
        ),
        (
            Objective::Multiclass,
            3,
            false,
            1.0,
            "multiclass num_class:3",
        ),
        (
            Objective::MulticlassOva,
            3,
            false,
            2.0,
            "multiclassova num_class:3 sigmoid:2",
        ),
        (
            Objective::CrossEntropyLambda,
            1,
            false,
            1.0,
            "cross_entropy_lambda",
        ),
        (Objective::LambdaRank, 1, false, 1.0, "lambdarank"),
        (Objective::RankXendcg, 1, false, 1.0, "rank_xendcg"),
    ] {
        let c = Config {
            objective,
            num_class: classes,
            reg_sqrt: sqrt,
            sigmoid,
            ..Config::default()
        };
        let state = ObjectiveState::new(
            &c,
            ObjectiveInput {
                labels: &[0.0, 1.0],
                weights: None,
                groups: Some(&[2]),
                positions: None,
            },
        )
        .unwrap();
        assert_eq!(state.model_objective(), text);
        assert!(!state.needs_training(classes));
        assert!(state.initial_score(classes).is_err());
        assert!(
            state
                .renew_leaf_output(classes, &[0], &vec![0.0; 2 * classes], 0.5)
                .is_err()
        );
        assert!(
            state
                .renew_leaf_output(0, &[2], &vec![0.0; 2 * classes], 0.5)
                .is_err()
        );
        if !matches!(
            objective,
            Objective::RegressionL1 | Objective::Quantile | Objective::Mape
        ) {
            assert_eq!(
                state
                    .renew_leaf_output(0, &[0], &vec![0.0; 2 * classes], 0.375)
                    .unwrap(),
                0.375
            );
        }
    }
    for objective in [Objective::Multiclass, Objective::MulticlassOva] {
        let c = Config {
            objective,
            num_class: 3,
            ..Config::default()
        };
        let state = ObjectiveState::new(
            &c,
            ObjectiveInput {
                labels: &[0.0, 1.0],
                weights: None,
                groups: None,
                positions: None,
            },
        )
        .unwrap();
        assert!(state.needs_training(0));
        assert!(state.needs_training(1));
        assert!(!state.needs_training(2));
    }
    assert_eq!(cpp_number(0.00001), "1e-05");
    assert_eq!(cpp_number(1_000_000.0), "1e+06");
}

#[test]
fn renewal_batches_match_single_leaves_with_weights_transforms_and_duplicates() {
    let labels = [-9.0, -1.0, 0.0, 1.0, 4.0, 16.0, 25.0];
    let weights = [0.0, 0.25, 3.0, 1.5, 2.0, 0.5, 4.0];
    let leaves = [vec![5, 1, 5, 2], vec![0, 4, 3], vec![6], vec![2, 1, 2]];
    for objective in [
        Objective::RegressionL1,
        Objective::Quantile,
        Objective::Mape,
    ] {
        let alphas: &[f64] = if objective == Objective::Quantile {
            &[0.1, 0.50000001, 0.9]
        } else {
            &[0.9]
        };
        for &alpha in alphas {
            for reg_sqrt in [false, true] {
                for weighted in [false, true] {
                    let config = Config {
                        objective,
                        alpha,
                        reg_sqrt,
                        ..Config::default()
                    };
                    let state = ObjectiveState::new(
                        &config,
                        ObjectiveInput {
                            labels: &labels,
                            weights: weighted.then_some(&weights),
                            groups: None,
                            positions: None,
                        },
                    )
                    .unwrap();
                    // RF uses fixed bias scores; other boosters use current margins.
                    let rf_scores = vec![state.initial_score(0).unwrap(); labels.len()];
                    let current_scores = [-3.0, 0.25, 0.0, 2.0, 1.0, -4.0, 8.0];
                    for scores in [&rf_scores[..], &current_scores[..]] {
                        let mut outputs = [0.25, -0.75, 0.0, 1.0];
                        let single: Vec<_> = leaves
                            .iter()
                            .zip(outputs)
                            .map(|(rows, newton)| {
                                state.renew_leaf_output(0, rows, scores, newton).unwrap()
                            })
                            .collect();
                        state
                            .renew_leaf_outputs(0, &leaves, scores, &mut outputs)
                            .unwrap();
                        assert_eq!(outputs.as_slice(), single);
                    }
                }
            }
        }
    }
}

#[test]
fn renewal_batches_check_global_scores_leaf_rows_newton_and_counts() {
    for objective in [
        Objective::RegressionL1,
        Objective::Quantile,
        Objective::Mape,
        Objective::Regression,
        Objective::Multiclass,
    ] {
        let config = Config {
            objective,
            num_class: if objective.is_multiclass() { 3 } else { 1 },
            ..Config::default()
        };
        let state = ObjectiveState::new(
            &config,
            ObjectiveInput {
                labels: &[0.0, 1.0, 0.0],
                weights: None,
                groups: None,
                positions: None,
            },
        )
        .unwrap();
        let scores = vec![0.0; 3 * config.num_class];
        let leaves: [&[usize]; 2] = [&[0], &[1, 1]];
        for class in [config.num_class, usize::MAX] {
            assert!(
                state
                    .renew_leaf_outputs(class, &leaves, &scores, &mut [0.5; 2])
                    .is_err()
            );
        }
        for len in [0, scores.len() - 1, scores.len() + 1] {
            assert!(
                state
                    .renew_leaf_outputs(0, &leaves, &vec![0.0; len], &mut [0.5; 2])
                    .is_err()
            );
        }
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            // Row 2 and, for multiclass, the final class are not selected.
            for index in [0, 2, scores.len() - 1] {
                let mut invalid_scores = scores.clone();
                invalid_scores[index] = bad;
                assert!(
                    state
                        .renew_leaf_outputs(0, &leaves, &invalid_scores, &mut [0.5; 2])
                        .is_err()
                );
            }
            for leaf in 0..leaves.len() {
                let mut outputs = [0.5; 2];
                outputs[leaf] = bad;
                assert!(
                    state
                        .renew_leaf_outputs(0, &leaves, &scores, &mut outputs)
                        .is_err()
                );
            }
        }
        for bad_rows in [&[][..], &[3][..], &[usize::MAX][..]] {
            for invalid_leaves in [[bad_rows, leaves[1]], [leaves[0], bad_rows]] {
                assert!(
                    state
                        .renew_leaf_outputs(0, &invalid_leaves, &scores, &mut [0.5; 2])
                        .is_err()
                );
            }
        }
        for count in [0, 1, 3] {
            assert!(
                state
                    .renew_leaf_outputs(0, &leaves, &scores, &mut vec![0.5; count])
                    .is_err()
            );
        }
        let no_leaves: [&[usize]; 0] = [];
        assert!(
            state
                .renew_leaf_outputs(0, &no_leaves, &scores, &mut [0.5])
                .is_err()
        );
        assert!(
            state
                .renew_leaf_outputs(config.num_class, &no_leaves, &scores, &mut [])
                .is_err()
        );
        assert!(
            state
                .renew_leaf_outputs(0, &no_leaves, &[], &mut [])
                .is_err()
        );
        state
            .renew_leaf_outputs(0, &no_leaves, &scores, &mut [])
            .unwrap();
        if matches!(objective, Objective::Regression | Objective::Multiclass) {
            let mut outputs = [0.375, -0.25];
            state
                .renew_leaf_outputs(config.num_class - 1, &leaves, &scores, &mut outputs)
                .unwrap();
            assert_eq!(outputs, [0.375, -0.25]);
        }
    }
    // Finite input scores can still overflow percentile interpolation.
    for objective in [
        Objective::RegressionL1,
        Objective::Quantile,
        Objective::Mape,
    ] {
        let state = ObjectiveState::new(
            &Config {
                objective,
                alpha: 0.5,
                ..Config::default()
            },
            ObjectiveInput {
                labels: &[0.0; 3],
                weights: None,
                groups: None,
                positions: None,
            },
        )
        .unwrap();
        assert!(
            state
                .renew_leaf_outputs(
                    0,
                    &[&[0, 1, 2][..]],
                    &[f64::MAX, -f64::MAX, -f64::MAX],
                    &mut [0.0]
                )
                .is_err()
        );
    }
}

#[test]
fn checked_metadata_layout_and_allocation_limits() {
    let c = Config::default();
    for groups in [&[0, 1][..], &[1][..], &[usize::MAX, 1][..]] {
        assert!(
            ObjectiveState::new(
                &c,
                ObjectiveInput {
                    labels: &[0.0, 1.0],
                    weights: None,
                    groups: Some(groups),
                    positions: None
                }
            )
            .is_err()
        );
    }
    let rank = Config {
        objective: Objective::LambdaRank,
        ..c.clone()
    };
    for labels in [&[-1.0, 1.0][..], &[0.5, 1.0][..], &[31.0, 1.0][..]] {
        assert!(
            ObjectiveState::new(
                &rank,
                ObjectiveInput {
                    labels,
                    weights: None,
                    groups: Some(&[2]),
                    positions: None
                }
            )
            .is_err()
        );
    }
    assert!(
        ObjectiveState::new(
            &rank,
            ObjectiveInput {
                labels: &[0.0, 1.0],
                weights: None,
                groups: Some(&[2]),
                positions: Some(&[7])
            }
        )
        .is_err()
    );
    let multi = Config {
        objective: Objective::Multiclass,
        num_class: 65_536,
        ..c
    };
    let labels = vec![0.0; 257];
    assert!(
        ObjectiveState::new(
            &multi,
            ObjectiveInput {
                labels: &labels,
                weights: None,
                groups: None,
                positions: None
            }
        )
        .is_err()
    );
}
