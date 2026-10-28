use lightgbm_rust::{Config, Objective, metric};
use serde_json::Value;

fn config(classes: usize) -> Config {
    Config {
        objective: Objective::Custom,
        num_class: classes,
        metrics: vec!["auc_mu".into()],
        ..Config::default()
    }
}

#[test]
fn native_auc_mu_near_epsilon_fixtures() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/metric_edges.json")).unwrap();
    assert_eq!(
        fixture["upstream_commit"],
        "5d84f3f222b41409a6983d2ac90513ad9702fa9e"
    );
    assert_eq!(
        fixture["loaded_library_sha256"],
        "9ed1d46d2d60bac2966fd03bd81e7d192025804f6a3e3ef24c3aa483d0ec7c1e"
    );
    assert_eq!(fixture["absolute_bound"].as_f64().unwrap(), 1e-12);
    assert_eq!(fixture["schema_version"], 1);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 68, "retain all native EPS edge cases");
    assert_eq!(cases[0]["labels"].as_array().unwrap().len(), 128);
    assert_eq!(
        cases[0]["results"][0]["value"].as_f64().unwrap(),
        0.5346219721066797
    );
    for (index, case) in cases.iter().enumerate() {
        let labels: Vec<f32> = serde_json::from_value(case["labels"].clone()).unwrap();
        let weights: Option<Vec<f32>> = case
            .get("weights")
            .map(|w| serde_json::from_value(w.clone()).unwrap());
        let scores: Vec<f64> = case["score_bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| f64::from_bits(u64::from_str_radix(v.as_str().unwrap(), 16).unwrap()))
            .collect();
        let mut c = config(case["params"]["num_class"].as_u64().unwrap() as usize);
        if let Some(matrix) = case["params"].get("auc_mu_weights") {
            c.auc_mu_weights = serde_json::from_value(matrix.clone()).unwrap();
        }
        let expected = case["results"][0]["value"].as_f64().unwrap();
        if index == 0 {
            assert_eq!(scores.len(), 256);
            let mut state = 1_u64;
            for row in 0..128 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                assert_eq!(labels[row], ((state >> 32) % 2) as f32);
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let score = ((state >> 32) % 40) as f64 * 1e-15 / 8.0;
                assert_eq!(scores[row].to_bits(), score.to_bits());
                assert_eq!(scores[128 + row].to_bits(), 0_f64.to_bits());
            }
        }
        let actual = metric::evaluate(&c, &labels, weights.as_deref(), &scores).unwrap()[0].value;
        if index == 0 {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        assert!(
            (actual - expected).abs() <= 1e-12,
            "{}: {actual} != {expected}",
            case["name"]
        );
        assert_eq!(
            actual.to_bits(),
            metric::evaluate(&c, &labels, weights.as_deref(), &scores).unwrap()[0]
                .value
                .to_bits()
        );
    }
}

#[test]
fn auc_mu_large_finite_epsilon_clusters_are_deterministic() {
    // The public row limit is 512 Ki rows. Exercise it without a native runtime.
    let rows = 512 * 1024;
    let mut labels = Vec::with_capacity(rows);
    let mut scores = vec![0.0; 2 * rows];
    let mut weights = Vec::with_capacity(rows);
    let mut state = 2026_u64;
    for score in &mut scores[..rows] {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        labels.push(((state >> 32) % 2) as f32);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *score = ((state >> 32) % 40) as f64 * 1e-15 / 8.0;
        weights.push((1 + (state >> 40) % 17) as f32 / 8.0);
    }
    let c = config(2);
    let first = metric::evaluate(&c, &labels, Some(&weights), &scores).unwrap()[0].value;
    assert!(first.is_finite() && (0.0..=1.0).contains(&first));
    let second = metric::evaluate(&c, &labels, Some(&weights), &scores).unwrap()[0].value;
    assert_eq!(first.to_bits(), second.to_bits());
}

#[test]
fn auc_mu_nonfinite_input_is_an_error() {
    let c = config(2);
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(metric::evaluate(&c, &[0., 1.], None, &[invalid, 0., 0., 0.]).is_err());
    }
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(metric::evaluate(&c, &[invalid, 1.], None, &[0.; 4]).is_err());
        assert!(metric::evaluate(&c, &[0., 1.], Some(&[invalid, 1.]), &[0.; 4]).is_err());
    }
    // Finite scores can still overflow a separating distance. This is an error,
    // not an unchecked sort comparison.
    assert!(metric::evaluate(&c, &[0., 1.], None, &[f64::MAX, 0., -f64::MAX, 0.]).is_err());
}
