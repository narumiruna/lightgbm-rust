use super::*;
use serde_json::Value;

#[test]
fn native_dart_drop_indices_and_weight_state_every_round() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../tests/fixtures/sampling.json")).unwrap();
    let mut seen = Vec::new();
    for case in fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["kind"] == "dart")
    {
        seen.push(case["name"].as_str().unwrap());
        let config =
            Config::from_pairs(case["params"].as_object().unwrap().iter().map(|(k, v)| {
                (
                    k.clone(),
                    v.as_str().map_or_else(|| v.to_string(), str::to_owned),
                )
            }))
            .unwrap();
        let mut state = DartState::new(&config);
        for step in case["steps"].as_array().unwrap() {
            let iteration = step["iteration"].as_u64().unwrap() as usize;
            let (dropped, rate) = state.select(iteration, &config);
            let expected: Vec<usize> =
                serde_json::from_value(step["drop_indices"].clone()).unwrap();
            assert_eq!(dropped, expected, "{} round {iteration}", case["name"]);
            assert_eq!(step["stopped"], false);
            state.finish(&dropped, rate, &config);
            let expected: Vec<f64> = serde_json::from_value(step["tree_weights"].clone()).unwrap();
            assert_eq!(state.weights.len(), expected.len());
            for (&got, &expected) in state.weights.iter().zip(&expected) {
                assert!((got - expected).abs() < 1e-14);
            }
            assert!((state.sum - step["sum_weight"].as_f64().unwrap()).abs() < 1e-14);
        }
    }
    assert_eq!(seen, ["dart_normalized", "dart_uniform", "dart_xgboost"]);
}
