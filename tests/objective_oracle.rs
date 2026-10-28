use lightgbm_rust::objective::{gradients, initial_score};
use lightgbm_rust::{Config, Objective};
use serde_json::Value;

#[test]
fn default_f32_objectives_match_compiled_cpp_driver() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/objectives.json")).unwrap();
    assert_eq!(
        fixture["upstream_commit"],
        "5d84f3f222b41409a6983d2ac90513ad9702fa9e"
    );
    assert_eq!(fixture["schema_version"], 1);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 4);
    for case in cases {
        let labels: Vec<f32> = serde_json::from_value(case["labels"].clone()).unwrap();
        let weights: Option<Vec<f32>> = case
            .get("weights")
            .map(|v| serde_json::from_value(v.clone()).unwrap());
        let scores: Vec<f64> = serde_json::from_value(case["scores"].clone()).unwrap();
        let expected_g: Vec<f32> = serde_json::from_value(case["gradients"].clone()).unwrap();
        let expected_h: Vec<f32> = serde_json::from_value(case["hessians"].clone()).unwrap();
        let c = Config {
            objective: if case["objective"] == "binary" {
                Objective::Binary
            } else {
                Objective::Regression
            },
            sigmoid: case["params"]["sigmoid"].as_f64().unwrap(),
            is_unbalance: case["params"]["is_unbalance"].as_bool().unwrap(),
            scale_pos_weight: case["params"]["scale_pos_weight"].as_f64().unwrap(),
            ..Config::default()
        };
        let (g, h) = gradients(&c, &labels, weights.as_deref(), &scores).unwrap();
        assert_eq!(g, expected_g, "{} gradients", case["name"]);
        assert_eq!(h, expected_h, "{} Hessians", case["name"]);
        let initial = initial_score(&c, &labels, weights.as_deref()).unwrap();
        assert!((initial - case["init_score"].as_f64().unwrap()).abs() <= 1e-12);
    }
}
