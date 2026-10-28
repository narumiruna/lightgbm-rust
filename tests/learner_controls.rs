use lightgbm_rust::{Config, Dataset, Model, Trainer};
use serde_json::Value;

fn check(name: &str) {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/prediction_constraints.json")).unwrap();
    let boosting: Value =
        serde_json::from_str(include_str!("fixtures/prediction_boosting.json")).unwrap();
    let extended: Value =
        serde_json::from_str(include_str!("fixtures/prediction_constraints_extra.json")).unwrap();
    let case = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .chain(boosting["cases"].as_array().unwrap())
        .chain(extended["cases"].as_array().unwrap())
        .find(|c| c["name"] == name)
        .unwrap();
    let mut pairs = Vec::new();
    for (k, v) in case["params"].as_object().unwrap() {
        if k == "forcedsplits_filename" {
            continue;
        }
        let value = match v {
            Value::String(s) => s.clone(),
            Value::Array(a) if !a.iter().any(Value::is_array) => {
                a.iter().map(Value::to_string).collect::<Vec<_>>().join(",")
            }
            _ => v.to_string(),
        };
        pairs.push((k.clone(), value));
    }
    pairs.push((
        "num_iterations".into(),
        case["training"]["num_boost_round"].to_string(),
    ));
    let mut config = Config::from_pairs(pairs).unwrap();
    if name == "forced_splits" {
        config.forced_splits = Some(
            serde_json::json!({"feature":1,"threshold":0.0,"left":{"feature":0,"threshold":0.0}}),
        );
    }
    let rows = case["training"]["rows"].as_array().unwrap();
    let cols = rows[0].as_array().unwrap().len();
    let x: Vec<f64> = rows
        .iter()
        .flat_map(|r| {
            r.as_array()
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
    if let Some(w) = case["training"].get("weights") {
        data = data
            .with_weights(serde_json::from_value(w.clone()).unwrap())
            .unwrap();
    }
    let result = Trainer::new(config).unwrap().fit(&data).unwrap();
    let native = Model::from_text(
        &std::fs::read_to_string(format!(
            "{}/tests/fixtures/{}",
            env!("CARGO_MANIFEST_DIR"),
            case["model_file"].as_str().unwrap()
        ))
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        result.model.trees().len(),
        native.trees().len(),
        "{name} tree count"
    );
    for (i, (got, want)) in result.model.trees().iter().zip(native.trees()).enumerate() {
        assert_eq!(
            got.split_features(),
            want.split_features(),
            "{name} tree{i} features\ngot={got:?}\nwant={want:?}"
        );
        assert_eq!(
            got.thresholds(),
            want.thresholds(),
            "{name} tree{i} thresholds"
        );
        assert_eq!(
            got.left_children(),
            want.left_children(),
            "{name} tree{i} left"
        );
        assert_eq!(
            got.right_children(),
            want.right_children(),
            "{name} tree{i} right"
        );
        assert_eq!(
            got.decision_types(),
            want.decision_types(),
            "{name} tree{i} directions"
        );
        assert_eq!(
            got.leaf_counts(),
            want.leaf_counts(),
            "{name} tree{i} counts"
        );
        assert_eq!(
            got.cat_boundaries(),
            want.cat_boundaries(),
            "{name} tree{i} category boundaries"
        );
        assert_eq!(
            got.cat_threshold(),
            want.cat_threshold(),
            "{name} tree{i} category codes"
        );
        for (&a, &b) in got.leaf_values().iter().zip(want.leaf_values()) {
            assert!(
                (a - b).abs() <= 1e-12 + 1e-10 * b.abs(),
                "{name} tree{i} leaf {a} != {b}"
            );
        }
    }
    let probes = case["rows"].as_array().unwrap();
    let x: Vec<f64> = probes
        .iter()
        .flat_map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap_or(f64::NAN))
        })
        .collect();
    assert_eq!(
        result
            .model
            .predict_leaf_dense(&x, probes.len(), cols)
            .unwrap(),
        serde_json::from_value::<Vec<Vec<usize>>>(case["leaves"].clone()).unwrap(),
        "{name} probe leaves"
    );
    for (row, want) in result
        .model
        .predict_raw_dense(&x, probes.len(), cols)
        .unwrap()
        .iter()
        .zip(case["raw"].as_array().unwrap())
    {
        for (&a, b) in row.iter().zip(want.as_array().unwrap()) {
            let b = b.as_f64().unwrap();
            assert!(
                (a - b).abs() <= 1e-12 + 1e-10 * b.abs(),
                "{name} prediction {a} != {b}"
            );
        }
    }
}
macro_rules! cases { ($($name:ident),* $(,)?) => {$ (#[test] fn $name(){check(stringify!($name));})*}; }
cases!(
    controls_baseline,
    monotone_basic,
    monotone_intermediate,
    monotone_advanced,
    monotone_penalty,
    interaction,
    extra_trees,
    forced_splits,
    path_smooth,
    feature_penalties,
    cegb_split,
    cegb_coupled,
    cegb_lazy,
    controls_combined,
    quantized_deterministic,
    quantized_stochastic,
    quantized_renew,
    quantized_binary,
    feature_tree,
    feature_node,
    feature_tree_node,
    deep_monotone_basic,
    deep_monotone_intermediate,
    deep_monotone_advanced,
    deep_monotone_smooth,
    cat_smooth,
    cat_extra_onehot,
    cat_extra_sorted,
    cat_quantized,
    cat_quantized_renew,
    interaction_sampled
);
