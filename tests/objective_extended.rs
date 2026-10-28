use lightgbm_rust::objective::{gradients, initial_score};
use lightgbm_rust::{Config, Error, Objective};
use serde_json::Value;

fn config(case: &Value) -> Config {
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
    Config::from_pairs(pairs).unwrap()
}

#[test]
fn extended_nonranking_wrappers_match_native_f32() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/extended_objectives.json")).unwrap();
    assert_eq!(
        fixture["upstream_commit"],
        "5d84f3f222b41409a6983d2ac90513ad9702fa9e"
    );
    let mut families = std::collections::BTreeSet::new();
    for case in fixture["cases"].as_array().unwrap() {
        let c = config(case);
        if matches!(c.objective, Objective::LambdaRank | Objective::RankXendcg) {
            continue;
        }
        families.insert(case["objective"].as_str().unwrap());
        let labels: Vec<f32> = serde_json::from_value(case["labels"].clone()).unwrap();
        let weights: Option<Vec<f32>> = case
            .get("weights")
            .map(|v| serde_json::from_value(v.clone()).unwrap());
        let scores: Vec<f64> = serde_json::from_value(case["scores"].clone()).unwrap();
        let expected_g: Vec<f32> = serde_json::from_value(case["gradients"].clone()).unwrap();
        let expected_h: Vec<f32> = serde_json::from_value(case["hessians"].clone()).unwrap();
        let (g, h) = gradients(&c, &labels, weights.as_deref(), &scores).unwrap();
        assert_eq!(g, expected_g, "{} gradients", case["name"]);
        assert_eq!(h, expected_h, "{} Hessians", case["name"]);
        let initial = initial_score(&c, &labels, weights.as_deref());
        if let Some(expected) = case["initial_scores"][0].as_f64() {
            assert!(
                (initial.unwrap() - expected).abs() <= 1e-12,
                "{} initialization",
                case["name"]
            );
        } else {
            assert!(
                initial.is_err(),
                "nonfinite native initialization must return an error"
            );
        }
    }
    assert_eq!(
        families.len(),
        14,
        "all nonranking CPU objective factories must be covered"
    );
}

#[test]
fn objective_domains_and_missing_capabilities_are_explicit() {
    for objective in [Objective::Poisson, Objective::Gamma, Objective::Tweedie] {
        let c = Config {
            objective,
            ..Config::default()
        };
        assert!(gradients(&c, &[-1.0, 2.0], None, &[0.0; 2]).is_err());
        assert!(initial_score(&c, &[0.0; 2], None).is_err());
    }
    for objective in [Objective::CrossEntropy, Objective::CrossEntropyLambda] {
        let c = Config {
            objective,
            ..Config::default()
        };
        assert!(gradients(&c, &[-0.1, 1.0], None, &[0.0; 2]).is_err());
        assert!(gradients(&c, &[0.0, 1.1], None, &[0.0; 2]).is_err());
    }
    let lambda = Config {
        objective: Objective::CrossEntropyLambda,
        ..Config::default()
    };
    assert!(gradients(&lambda, &[0.0, 1.0], Some(&[0.0, 1.0]), &[0.0; 2]).is_err());
    assert!(initial_score(&lambda, &[0.0; 2], None).is_err());
    for objective in [Objective::Multiclass, Objective::MulticlassOva] {
        let c = Config {
            objective,
            num_class: 3,
            ..Config::default()
        };
        for labels in [[-1.0, 1.0], [0.0, 3.0], [0.0, 0.5]] {
            assert!(gradients(&c, &labels, None, &[0.0; 6]).is_err());
        }
        assert!(gradients(&c, &[0.0, 1.0], None, &[0.0; 2]).is_err());
    }
    for objective in [Objective::LambdaRank, Objective::RankXendcg] {
        let c = Config {
            objective,
            ..Config::default()
        };
        assert!(gradients(&c, &[0.0, 1.0], None, &[0.0; 2]).is_err());
    }
    let custom = Config {
        objective: Objective::Custom,
        ..Config::default()
    };
    assert!(matches!(
        gradients(&custom, &[0.0], None, &[0.0]),
        Err(Error::Unsupported(_))
    ));
}
