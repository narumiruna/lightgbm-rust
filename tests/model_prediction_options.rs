use lightgbm_rust::{ImportanceType, Model, PredictionEarlyStop};
use serde_json::Value;
use std::path::Path;

fn operations() -> Value {
    serde_json::from_str(include_str!("fixtures/prediction_operations.json")).unwrap()
}
fn load(record: &Value) -> Model {
    Model::load(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(record["model_file"].as_str().unwrap()),
    )
    .unwrap()
}
fn close(got: &[f64], want: &Value) {
    let want = want.as_array().unwrap();
    assert_eq!(got.len(), want.len());
    for (&got, want) in got.iter().zip(want) {
        let want = want.as_f64().unwrap();
        assert!(
            got.is_finite() && (got - want).abs() <= 1e-12 + 1e-10 * want.abs(),
            "{got} != {want}"
        );
    }
}

#[test]
fn actual_native_prediction_stopping_matches_all_class_rounds() {
    let fixtures = operations();
    let mut names = std::collections::BTreeSet::new();
    for case in fixtures["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|case| case["operation"] == "prediction_early_stop")
    {
        names.insert(case["name"].as_str().unwrap());
        let imported = load(&case["initial_model"]);
        let roundtrip = Model::from_text(&imported.to_text().unwrap()).unwrap();
        for model in [&imported, &roundtrip] {
            for probe in case["probes"].as_array().unwrap() {
                if probe.get("raw").is_none() {
                    continue;
                }
                let parameters = &probe["params"];
                let stopping = PredictionEarlyStop::new(
                    parameters["pred_early_stop_freq"].as_u64().unwrap() as usize,
                    parameters["pred_early_stop_margin"].as_f64().unwrap(),
                )
                .unwrap();
                for (i, row) in case["rows"].as_array().unwrap().iter().enumerate() {
                    let row: Vec<_> = row
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap_or(f64::NAN))
                        .collect();
                    let range = model.all_iterations();
                    close(
                        &model
                            .predict_raw_row_early_stop(&row, range, stopping)
                            .unwrap(),
                        &probe["raw"][i],
                    );
                    close(
                        &model.predict_row_early_stop(&row, range, stopping).unwrap(),
                        &probe["transformed"][i],
                    );
                    let (indices, values): (Vec<_>, Vec<_>) = row
                        .iter()
                        .copied()
                        .enumerate()
                        .filter(|(_, v)| *v != 0.0)
                        .unzip();
                    close(
                        &model
                            .predict_raw_csr_row_early_stop(&indices, &values, range, stopping)
                            .unwrap(),
                        &probe["raw"][i],
                    );
                    close(
                        &model
                            .predict_csr_row_early_stop(&indices, &values, range, stopping)
                            .unwrap(),
                        &probe["transformed"][i],
                    );
                    let leaves: Vec<usize> =
                        serde_json::from_value(probe["leaves"][i].clone()).unwrap();
                    assert_eq!(model.predict_leaf_row(&row).unwrap(), leaves);
                }
            }
        }
    }
    assert_eq!(
        names,
        [
            "predict_early_binary",
            "predict_early_linear",
            "predict_early_multiclass"
        ]
        .into_iter()
        .collect()
    );
}

fn constants(classes: usize, values: &[f64], objective: &str, rf: bool) -> Model {
    let mut text = format!(
        "tree\nversion=v4\nnum_class={classes}\nnum_tree_per_iteration={classes}\nlabel_index=0\nmax_feature_idx=0\nobjective={objective}\nfeature_names=x\nfeature_infos=none\n"
    );
    if rf {
        text.push_str("average_output\n");
    }
    for (i, value) in values.iter().enumerate() {
        text.push_str(&format!(
            "\nTree={i}\nnum_leaves=1\nnum_cat=0\nleaf_value={value}\nis_linear=0\nshrinkage=1\n"
        ));
    }
    text.push_str("\nend of trees\n");
    Model::from_text(&text).unwrap()
}

#[test]
fn strict_margin_period_range_and_rf_denominator_follow_source() {
    for (period, margin) in [
        (0, 0.),
        (usize::MAX, 0.),
        (1, -1.),
        (1, f64::NAN),
        (1, f64::INFINITY),
    ] {
        assert!(PredictionEarlyStop::new(period, margin).is_err());
    }
    let stop = PredictionEarlyStop::new(1, 0.5).unwrap();
    let model = constants(1, &[0.25, 1., 10., 100.], "binary sigmoid:1", false);
    assert_eq!(
        model
            .predict_raw_row_early_stop(&[0.], model.all_iterations(), stop)
            .unwrap(),
        [1.25]
    );
    let range = model.iteration_range(1..4).unwrap();
    assert_eq!(
        model
            .predict_raw_row_early_stop(&[0.], range, PredictionEarlyStop::new(2, 0.).unwrap())
            .unwrap(),
        [11.]
    );
    let multi = constants(
        3,
        &[1., 1., 0., 2., -1., 0.],
        "multiclass num_class:3",
        false,
    );
    assert_eq!(
        multi
            .predict_raw_row_early_stop(
                &[0.],
                multi.all_iterations(),
                PredictionEarlyStop::new(1, 0.).unwrap()
            )
            .unwrap(),
        [3., 0., 0.]
    );
    let accurate = constants(1, &[1., 100.], "regression", false);
    assert_eq!(
        accurate
            .predict_raw_row_early_stop(
                &[0.],
                accurate.all_iterations(),
                PredictionEarlyStop::new(1, 0.).unwrap()
            )
            .unwrap(),
        [101.]
    );
    let rf = constants(1, &[1., 2.], "binary sigmoid:1", true);
    let got = rf
        .predict_row_early_stop(
            &[0.],
            rf.all_iterations(),
            PredictionEarlyStop::new(1, 0.).unwrap(),
        )
        .unwrap();
    assert_eq!(got, [1.0 / (1.0 + (-0.5_f64).exp())]);
    assert!(
        model
            .predict_raw_row_early_stop(&[], model.all_iterations(), stop)
            .is_err()
    );
    assert!(
        model
            .predict_csr_row_early_stop(&[0, 0], &[1., 2.], model.all_iterations(), stop)
            .is_err()
    );
}

#[test]
fn native_infinite_float_gain_preserves_predictions_and_diagnostic_importance() {
    let fixtures = operations();
    let case = fixtures["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "extreme_f32_gain")
        .unwrap();
    assert_eq!(case["importance_gain"][0], "Infinity");
    let model = load(&case["final_model"]);
    assert!(
        model
            .trees()
            .iter()
            .all(|tree| tree.split_gains()[0] == f64::INFINITY)
    );
    assert_eq!(
        model.feature_importance(ImportanceType::Gain).unwrap(),
        [f64::INFINITY]
    );
    let expected = case["model_dump"]["tree_info"][0]["tree_structure"]["split_gain"]
        .as_f64()
        .unwrap();
    let json = serde_json::to_value(&model.trees()[0]).unwrap();
    assert_eq!(json["split_gains"][0].as_f64().unwrap(), expected);
    let reloaded = Model::from_text(&model.to_text().unwrap()).unwrap();
    for model in [&model, &reloaded] {
        for (i, row) in case["rows"].as_array().unwrap().iter().enumerate() {
            let row: Vec<_> = row
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            close(
                &model.predict_raw_row(&row).unwrap(),
                &case["final_model"]["raw"][i],
            );
            let phi = model.predict_contrib_row(&row).unwrap();
            close(&phi[0], &case["contributions"][i]);
        }
    }
}
