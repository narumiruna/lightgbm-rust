use lightgbm_rust::{Config, Dataset, Error, Model};
use serde_json::Value;
use std::path::Path;

#[test]
fn actual_native_refit_decay_errors_are_checked_without_mutation() {
    let suite: Value =
        serde_json::from_str(include_str!("fixtures/prediction_operations.json")).unwrap();
    let mut covered = 0;
    for case in suite["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["operation"] == "refit" && c.get("error").is_some())
    {
        assert_eq!(case["error_type"], "LightGBMError");
        let config =
            Config::from_pairs(case["params"].as_object().unwrap().iter().map(|(k, v)| {
                (
                    k.clone(),
                    v.as_str()
                        .map(str::to_owned)
                        .unwrap_or_else(|| v.to_string()),
                )
            }))
            .unwrap();
        let rows = case["rows"].as_array().unwrap();
        let ncols = rows[0].as_array().unwrap().len();
        let values: Vec<f64> = rows
            .iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect();
        let mut data = Dataset::from_dense(
            &values,
            rows.len(),
            ncols,
            serde_json::from_value(case["labels"].clone()).unwrap(),
            &config,
        )
        .unwrap();
        if let Some(weights) = case["weights"].as_array() {
            data = data
                .with_weights(weights.iter().map(|v| v.as_f64().unwrap() as f32).collect())
                .unwrap();
        }
        let model = Model::load(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures")
                .join(case["initial_model"]["model_file"].as_str().unwrap()),
        )
        .unwrap();
        let before = model.to_text().unwrap();
        let leaves: Vec<Vec<usize>> = serde_json::from_value(case["leaf_pred"].clone()).unwrap();
        assert!(matches!(
            model.refit(
                &data,
                &leaves,
                &config,
                case["decay_rate"].as_f64().unwrap()
            ),
            Err(Error::InvalidInput(_))
        ));
        assert_eq!(model.to_text().unwrap(), before);
        covered += 1;
    }
    assert_eq!(covered, 2);
}
