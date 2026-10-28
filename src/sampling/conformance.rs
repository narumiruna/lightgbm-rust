use super::*;
use serde_json::Value;

#[test]
fn native_row_selection_and_rescaling_every_seeded_round() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../tests/fixtures/sampling.json")).unwrap();
    assert_eq!(
        fixture["upstream_commit"],
        "5d84f3f222b41409a6983d2ac90513ad9702fa9e"
    );
    let mut seen = Vec::new();
    for case in fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["kind"] == "rows")
    {
        let name = case["name"].as_str().unwrap();
        seen.push(name);
        let config =
            Config::from_pairs(case["params"].as_object().unwrap().iter().map(|(k, v)| {
                (
                    k.clone(),
                    v.as_str().map_or_else(|| v.to_string(), str::to_owned),
                )
            }))
            .unwrap();
        let x: Vec<f64> = case["rows"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect();
        let labels: Vec<f32> = serde_json::from_value(case["labels"].clone()).unwrap();
        let n = labels.len();
        let mut data = Dataset::from_dense(&x, n, x.len() / n, labels, &config).unwrap();
        data = data
            .with_weights(serde_json::from_value(case["weights"].clone()).unwrap())
            .unwrap();
        if let Some(groups) = case.get("groups") {
            data = data
                .with_groups(serde_json::from_value(groups.clone()).unwrap())
                .unwrap();
        }
        let mut sampler = RowSampler::new(&config, &data).unwrap();
        for step in case["steps"].as_array().unwrap() {
            let iteration = step["iteration"].as_u64().unwrap() as usize;
            let mut g: Vec<f32> = serde_json::from_value(step["input_gradients"].clone()).unwrap();
            let mut h: Vec<f32> = serde_json::from_value(step["input_hessians"].clone()).unwrap();
            sampler
                .sample(iteration, &config, &data, &mut g, &mut h)
                .unwrap();
            if step["all_rows"] == true {
                assert_eq!(sampler.rows(), (0..n).collect::<Vec<_>>());
            } else {
                let expected: Vec<usize> =
                    serde_json::from_value(step["selected_rows"].clone()).unwrap();
                assert_eq!(sampler.rows(), expected, "{name} round {iteration}");
            }
            if let Some(queries) = step.get("selected_queries") {
                let expected: Vec<usize> = serde_json::from_value(queries.clone()).unwrap();
                assert_eq!(
                    sampler.queries(),
                    expected,
                    "{name} round {iteration} queries"
                );
            }
            let expected_g: Vec<f32> = serde_json::from_value(step["gradients"].clone()).unwrap();
            let expected_h: Vec<f32> = serde_json::from_value(step["hessians"].clone()).unwrap();
            assert_eq!(g, expected_g, "{name} round {iteration} gradient rescaling");
            assert_eq!(h, expected_h, "{name} round {iteration} Hessian rescaling");
        }
    }
    assert_eq!(
        seen,
        [
            "bagging_rows",
            "bagging_balanced",
            "bagging_queries",
            "goss_warmup",
            "goss_multiclass"
        ]
    );
}
