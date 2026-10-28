//! R5: live objectives retain full sigmoid precision until text reload.
use lightgbm_rust::{Config, Dataset, Model, ParallelPredictor, PredictionEarlyStop, Trainer};
use serde_json::Value;

fn suite() -> Value {
    let suite: Value = serde_json::from_str(include_str!("fixtures/review_refit.json")).unwrap();
    assert_eq!(
        suite["upstream_commit"],
        "5d84f3f222b41409a6983d2ac90513ad9702fa9e"
    );
    assert_eq!(
        suite["library"]["loaded_sha256"],
        "9ed1d46d2d60bac2966fd03bd81e7d192025804f6a3e3ef24c3aa483d0ec7c1e"
    );
    assert_eq!(suite["cases"].as_array().unwrap().len(), 4);
    suite
}

fn trained(case: &Value) -> (Model, Config) {
    let mut config =
        Config::from_pairs(case["params"].as_object().unwrap().iter().map(|(k, v)| {
            (
                k.clone(),
                v.as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| v.to_string()),
            )
        }))
        .unwrap();
    config.num_iterations = case["training"]["num_boost_round"].as_u64().unwrap() as usize;
    let rows: Vec<Vec<f64>> = serde_json::from_value(case["training"]["rows"].clone()).unwrap();
    let data = Dataset::from_dense(
        &rows.concat(),
        rows.len(),
        rows[0].len(),
        serde_json::from_value(case["training"]["labels"].clone()).unwrap(),
        &config,
    )
    .unwrap()
    .with_weights(serde_json::from_value(case["training"]["weights"].clone()).unwrap())
    .unwrap();
    let model = Trainer::new(config.clone())
        .unwrap()
        .fit(&data)
        .unwrap()
        .model;
    (model, config)
}

fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &e) in actual.iter().zip(expected) {
        assert!(
            a.is_finite() && (a - e).abs() <= 1e-12 + 1e-10 * e.abs(),
            "{a} != {e}"
        );
    }
}

fn report(label: &str, actual: &[Vec<f64>], expected: &[Vec<f64>]) {
    assert_eq!(actual.len(), expected.len());
    let mut exact = 0;
    let mut count = 0;
    let mut delta = 0.0_f64;
    for (a, e) in actual.iter().zip(expected) {
        close(a, e);
        for (&a, &e) in a.iter().zip(e) {
            exact += usize::from(a.to_bits() == e.to_bits());
            count += 1;
            delta = delta.max((a - e).abs());
        }
    }
    println!("{label}: bit-exact={exact}/{count}, max_abs_delta={delta:.17e}");
}

fn transformed(raw: &[f64], sigmoid: f64) -> Vec<f64> {
    raw.iter()
        .map(|v| 1.0 / (1.0 + (-sigmoid * v).exp()))
        .collect()
}

fn check_native(name: &str) {
    let suite = suite();
    let case = suite["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == name)
        .unwrap();
    let (model, _) = trained(case);
    let rows: Vec<Vec<f64>> = serde_json::from_value(case["rows"].clone()).unwrap();
    let live: Vec<Vec<f64>> =
        serde_json::from_value(case["live_initial"]["transformed"].clone()).unwrap();
    let saved: Vec<Vec<f64>> =
        serde_json::from_value(case["initial_model"]["transformed"].clone()).unwrap();
    let raw: Vec<Vec<f64>> = serde_json::from_value(case["live_initial"]["raw"].clone()).unwrap();
    assert_eq!(case["live_initial"]["raw"], case["initial_model"]["raw"]);
    assert_ne!(live, saved);
    assert_eq!(model.objective(), case["objective"].as_str().unwrap());
    let text = model.to_text().unwrap();
    let reloaded = Model::from_text(&text).unwrap();
    assert_eq!(reloaded.objective(), model.objective());
    let inspection = serde_json::to_value(&model).unwrap();
    assert!(inspection.get("live_sigmoid").is_none());
    assert_eq!(inspection.as_object().unwrap().len(), 9);
    assert_eq!(
        inspection["trees"],
        serde_json::to_value(&reloaded).unwrap()["trees"]
    );
    assert_eq!(
        model.num_iterations(),
        case["live_initial"]["current_iteration"].as_u64().unwrap() as usize
    );
    assert_eq!(
        model.trees().len(),
        case["live_initial"]["num_trees"].as_u64().unwrap() as usize
    );
    let dense = rows.concat();
    report(
        &format!("{name}/initial/raw"),
        &model
            .predict_raw_dense(&dense, rows.len(), rows[0].len())
            .unwrap(),
        &raw,
    );
    report(
        &format!("{name}/initial/live"),
        &model
            .predict_dense(&dense, rows.len(), rows[0].len())
            .unwrap(),
        &live,
    );
    report(
        &format!("{name}/initial/reloaded"),
        &reloaded
            .predict_dense(&dense, rows.len(), rows[0].len())
            .unwrap(),
        &saved,
    );
    let mut pointers = vec![0];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        close(&model.predict_raw_row(row).unwrap(), &raw[i]);
        close(&model.predict_row(row).unwrap(), &live[i]);
        close(&reloaded.predict_raw_row(row).unwrap(), &raw[i]);
        close(&reloaded.predict_row(row).unwrap(), &saved[i]);
        let leaves: Vec<usize> =
            serde_json::from_value(case["live_initial"]["leaves"][i].clone()).unwrap();
        assert_eq!(model.predict_leaf_row(row).unwrap(), leaves);
        let (ix, val): (Vec<_>, Vec<_>) = row
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, v)| *v != 0.0)
            .unzip();
        for (m, expected) in [(&model, &live[i]), (&reloaded, &saved[i])] {
            close(&m.predict_sparse_row(&ix, &val).unwrap(), expected);
            close(&m.predict_csr_row(&ix, &val).unwrap(), expected);
            let stop = PredictionEarlyStop::new(1, f64::MAX).unwrap();
            close(
                &m.predict_row_early_stop(row, m.all_iterations(), stop)
                    .unwrap(),
                expected,
            );
            close(
                &m.predict_csr_row_early_stop(&ix, &val, m.all_iterations(), stop)
                    .unwrap(),
                expected,
            );
        }
        indices.extend(ix);
        values.extend(val);
        pointers.push(indices.len());
    }
    for (m, expected) in [(&model, &live), (&reloaded, &saved)] {
        for (a, e) in m
            .predict_dense(&rows.concat(), rows.len(), rows[0].len())
            .unwrap()
            .iter()
            .zip(expected)
        {
            close(a, e);
        }
        for threads in [1, 2] {
            let predictor = ParallelPredictor::new(threads).unwrap();
            for batch in [
                predictor
                    .predict_dense(m, &rows.concat(), rows.len(), rows[0].len())
                    .unwrap(),
                predictor
                    .predict_csr(m, &pointers, &indices, &values)
                    .unwrap(),
            ] {
                for (a, e) in batch.iter().zip(expected) {
                    close(a, e);
                }
            }
        }
    }
}

#[test]
fn native_live_binary_decimal() {
    check_native("binary_decimal");
}
#[test]
fn native_live_binary_scientific() {
    check_native("binary_scientific");
}
#[test]
fn native_live_ova_decimal() {
    check_native("multiclassova_decimal");
}
#[test]
fn native_live_ova_scientific() {
    check_native("multiclassova_scientific");
}

#[test]
fn live_precision_survives_clone_ranges_truncate_leaf_edit_and_stopping() {
    for case in suite()["cases"].as_array().unwrap() {
        let (original, config) = trained(case);
        let rows: Vec<Vec<f64>> = serde_json::from_value(case["rows"].clone()).unwrap();
        let clone = original.clone();
        let prefix = original.truncate(2).unwrap();
        let rollback = original.rollback_one_iteration().unwrap();
        let leaf = original.with_leaf_output(0, 0, 0.123456789).unwrap();
        for model in [&clone, &prefix, &rollback, &leaf] {
            for row in &rows {
                let (ix, val): (Vec<_>, Vec<_>) = row
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(_, v)| *v != 0.0)
                    .unzip();
                let range = model.iteration_range(1..model.num_iterations()).unwrap();
                let expected = transformed(
                    &model.predict_raw_row_with_iterations(row, range).unwrap(),
                    config.sigmoid,
                );
                close(
                    &model.predict_row_with_iterations(row, range).unwrap(),
                    &expected,
                );
                close(
                    &model
                        .predict_csr_row_with_iterations(&ix, &val, range)
                        .unwrap(),
                    &expected,
                );
                let stop = PredictionEarlyStop::new(1, 0.0).unwrap();
                let raw = model.predict_raw_row_early_stop(row, range, stop).unwrap();
                let expected = transformed(&raw, config.sigmoid);
                close(
                    &model.predict_row_early_stop(row, range, stop).unwrap(),
                    &expected,
                );
                close(
                    &model
                        .predict_csr_row_early_stop(&ix, &val, range, stop)
                        .unwrap(),
                    &expected,
                );
            }
        }
    }
}

#[test]
fn native_fresh_and_reloaded_refit_use_full_or_rounded_caller_sigmoid() {
    for case in suite()["cases"].as_array().unwrap() {
        let (fresh, config) = trained(case);
        let reloaded = Model::from_text(&fresh.to_text().unwrap()).unwrap();
        let rows: Vec<Vec<f64>> = serde_json::from_value(case["rows"].clone()).unwrap();
        let leaves: Vec<Vec<usize>> = serde_json::from_value(case["leaf_pred"].clone()).unwrap();
        for (mode, original) in [("fresh", &fresh), ("reloaded", &reloaded)] {
            for rounded in [false, true] {
                let mut config = config.clone();
                if rounded {
                    config.sigmoid = case["serialized_sigmoid"].as_f64().unwrap();
                }
                let data = Dataset::from_dense(
                    &rows.concat(),
                    rows.len(),
                    rows[0].len(),
                    serde_json::from_value(case["labels"].clone()).unwrap(),
                    &config,
                )
                .unwrap()
                .with_weights(serde_json::from_value(case["weights"].clone()).unwrap())
                .unwrap();
                let result = original
                    .refit(
                        &data,
                        &leaves,
                        &config,
                        case["decay_rate"].as_f64().unwrap(),
                    )
                    .unwrap();
                let expected = if rounded {
                    &case["rounded_refit"]
                } else {
                    &case["native_refit"][mode]
                };
                let saved = if rounded {
                    expected
                } else {
                    &case["final_model"]
                };
                let raw: Vec<Vec<f64>> = serde_json::from_value(expected["raw"].clone()).unwrap();
                let live: Vec<Vec<f64>> =
                    serde_json::from_value(expected["transformed"].clone()).unwrap();
                let saved: Vec<Vec<f64>> =
                    serde_json::from_value(saved["transformed"].clone()).unwrap();
                let label = format!(
                    "{}/{mode}/rounded={rounded}",
                    case["name"].as_str().unwrap()
                );
                report(
                    &format!("{label}/raw"),
                    &result
                        .predict_raw_dense(&rows.concat(), rows.len(), rows[0].len())
                        .unwrap(),
                    &raw,
                );
                report(
                    &format!("{label}/live"),
                    &result
                        .predict_dense(&rows.concat(), rows.len(), rows[0].len())
                        .unwrap(),
                    &live,
                );
                let text = result.to_text().unwrap();
                assert_eq!(result.objective(), case["objective"].as_str().unwrap());
                let restored = Model::from_text(&text).unwrap();
                report(
                    &format!("{label}/reloaded"),
                    &restored
                        .predict_dense(&rows.concat(), rows.len(), rows[0].len())
                        .unwrap(),
                    &saved,
                );
                assert_eq!(
                    result
                        .predict_raw_dense(&rows.concat(), rows.len(), rows[0].len())
                        .unwrap(),
                    restored
                        .predict_raw_dense(&rows.concat(), rows.len(), rows[0].len())
                        .unwrap()
                );
            }
        }
    }
}

#[test]
fn imported_high_precision_sigmoid_is_not_rounded() {
    for case in suite()["cases"].as_array().unwrap() {
        let (live, config) = trained(case);
        let token = live
            .objective()
            .split_whitespace()
            .find(|t| t.starts_with("sigmoid:"))
            .unwrap();
        let text =
            live.to_text()
                .unwrap()
                .replacen(token, &format!("sigmoid:{:.17e}", config.sigmoid), 1);
        let imported = Model::from_text(&text).unwrap();
        let reloaded = Model::from_text(&imported.to_text().unwrap()).unwrap();
        let rows: Vec<Vec<f64>> = serde_json::from_value(case["rows"].clone()).unwrap();
        for row in &rows {
            close(
                &imported.predict_row(row).unwrap(),
                &live.predict_row(row).unwrap(),
            );
            close(
                &reloaded.predict_row(row).unwrap(),
                &live.predict_row(row).unwrap(),
            );
        }
    }
}
