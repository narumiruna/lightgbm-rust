use lightgbm_rust::{Config, Dataset, random::Random};
use serde_json::Value;

fn number(value: &Value) -> f64 {
    match value {
        Value::Null => f64::NAN,
        Value::Number(n) => n.as_f64().unwrap(),
        Value::String(s) if s == "Infinity" => f64::INFINITY,
        Value::String(s) if s == "-Infinity" => f64::NEG_INFINITY,
        Value::String(s) if s == "NaN" => f64::NAN,
        _ => panic!("invalid fixture number: {value}"),
    }
}

#[test]
fn pinned_rng_sequences_and_sample_match_exactly() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/rng.json")).unwrap();
    assert_eq!(
        fixture["upstream_commit"],
        "5d84f3f222b41409a6983d2ac90513ad9702fa9e"
    );
    let cases = fixture["cases"].as_array().unwrap();
    assert!(!cases.is_empty());
    for case in cases {
        let seed = case["seed"].as_u64().unwrap() as u32;
        let lower = case["lower"].as_i64().unwrap() as i32;
        let upper = case["upper"].as_i64().unwrap() as i32;
        let mut random = Random::new(seed);
        for expected in case["next_short"].as_array().unwrap() {
            assert_eq!(
                random.next_short(lower, upper).unwrap(),
                expected.as_i64().unwrap() as i32,
                "{}",
                case["name"]
            );
        }
        let mut random = Random::new(seed);
        for expected in case["next_int"].as_array().unwrap() {
            assert_eq!(
                random.next_int(lower, upper).unwrap(),
                expected.as_i64().unwrap() as i32,
                "{}",
                case["name"]
            );
        }
        let mut random = Random::new(seed);
        for expected in case["next_float"].as_array().unwrap() {
            assert_eq!(
                f64::from(random.next_float()),
                number(expected),
                "{}",
                case["name"]
            );
        }
        let sample = Random::new(seed)
            .sample(
                case["sample_n"].as_u64().unwrap() as usize,
                case["sample_k"].as_u64().unwrap() as usize,
            )
            .unwrap();
        let expected: Vec<_> = case["sample"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(sample, expected, "{}", case["name"]);
        if let Some(dense) = case.get("sample_dense") {
            let actual = Random::new(seed)
                .sample(
                    dense["n"].as_u64().unwrap() as usize,
                    dense["k"].as_u64().unwrap() as usize,
                )
                .unwrap();
            let expected: Vec<_> = dense["indices"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(actual, expected, "{} dense sample", case["name"]);
        }
        if let Some(mixed) = case["mixed"].as_array() {
            let mut random = Random::new(seed);
            for draw in mixed {
                let actual = match draw["method"].as_str().unwrap() {
                    "NextShort" => f64::from(random.next_short(lower, upper).unwrap()),
                    "NextInt" => f64::from(random.next_int(lower, upper).unwrap()),
                    "NextFloat" => f64::from(random.next_float()),
                    method => panic!("unknown RNG method {method}"),
                };
                assert_eq!(
                    actual,
                    number(&draw["value"]),
                    "{} mixed draw",
                    case["name"]
                );
            }
        }
    }
}

#[test]
fn pinned_bin_maps_and_probe_assignments_match_exactly() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/bins.json")).unwrap();
    assert_eq!(
        fixture["upstream_commit"],
        "5d84f3f222b41409a6983d2ac90513ad9702fa9e"
    );
    let cases = fixture["cases"].as_array().unwrap();
    for required in [
        "numeric",
        "nan_missing",
        "zero_missing",
        "constant",
        "max_bin4",
        "zero_separated",
        "missing_disabled",
        "max_bin2",
        "max_bin3",
        "min_data_in_bin3",
        "adjacent_doubles",
        "categorical_ties",
        "categorical_missing",
        "categorical_rare",
    ] {
        assert!(
            cases.iter().any(|c| c["name"] == required),
            "missing fixture {required}"
        );
    }
    for case in cases {
        let mut values: Vec<_> = case["sample_values"]
            .as_array()
            .unwrap()
            .iter()
            .map(number)
            .collect();
        let total = case["total_sample_count"].as_u64().unwrap() as usize;
        assert!(values.len() <= total);
        values.resize(total, 0.0);
        let p = &case["params"];
        let config = Config {
            max_bin: p["max_bin"].as_u64().unwrap() as usize,
            min_data_in_bin: p["min_data_in_bin"].as_u64().unwrap() as usize,
            min_data_in_leaf: p["min_split_data"].as_u64().unwrap() as usize,
            feature_pre_filter: p["pre_filter"].as_bool().unwrap(),
            use_missing: p["use_missing"].as_bool().unwrap(),
            zero_as_missing: p["zero_as_missing"].as_bool().unwrap(),
            categorical_features: if case["is_categorical"].as_bool().unwrap_or(false) {
                vec![0]
            } else {
                vec![]
            },
            ..Config::default()
        };
        let dataset = Dataset::from_dense(&values, total, 1, vec![0.0; total], &config).unwrap();
        let mapper = dataset.get_mapper(0).unwrap();
        assert_eq!(
            mapper.num_bins(),
            case["num_bin"].as_u64().unwrap() as u32,
            "{}",
            case["name"]
        );
        assert_eq!(
            format!("{:?}", mapper.missing_type()),
            case["missing_type"].as_str().unwrap(),
            "{}",
            case["name"]
        );
        assert_eq!(
            mapper.default_bin(),
            case["default_bin"].as_u64().unwrap() as u32,
            "{}",
            case["name"]
        );
        if !mapper.is_trivial() {
            assert_eq!(
                mapper.most_freq_bin(),
                case["most_freq_bin"].as_u64().unwrap() as u32,
                "{}",
                case["name"]
            );
        }
        let expected = case["upper_bounds"].as_array().unwrap();
        let actual_bounds: Vec<_> = if mapper.is_categorical() {
            mapper.categories().iter().map(|&c| f64::from(c)).collect()
        } else {
            mapper.thresholds().to_vec()
        };
        assert_eq!(actual_bounds.len(), expected.len());
        for (&actual, expected) in actual_bounds.iter().zip(expected) {
            let expected = number(expected);
            assert!(
                actual.to_bits() == expected.to_bits() || actual.is_nan() && expected.is_nan(),
                "{}: actual {actual:?}, expected {expected:?}",
                case["name"]
            );
        }
        let probes = case["probe_values"].as_array().unwrap();
        let bins = case["value_bins"].as_array().unwrap();
        assert_eq!(probes.len(), bins.len());
        for (value, bin) in probes.iter().zip(bins) {
            assert_eq!(
                mapper.bin(number(value)),
                bin.as_u64().unwrap() as u32,
                "{} probe {value}",
                case["name"]
            );
        }
    }
}
