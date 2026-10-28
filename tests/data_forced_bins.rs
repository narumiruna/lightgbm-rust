use lightgbm_rust::{Config, Dataset};
use serde_json::Value;

fn number(value: &Value) -> f64 {
    match value {
        Value::Null => f64::NAN,
        Value::String(s) if s == "Infinity" => f64::INFINITY,
        _ => value.as_f64().unwrap(),
    }
}

#[test]
fn native_forced_cuts_missing_modes_and_per_feature_caps_match_exactly() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/forced_bins.json")).unwrap();
    assert_eq!(
        fixture["upstream_commit"],
        "5d84f3f222b41409a6983d2ac90513ad9702fa9e"
    );
    let cases = fixture["cases"].as_array().unwrap();
    for required in [
        "forced_two_cuts",
        "forced_outside_sample",
        "forced_max_bin3",
        "forced_nan",
        "forced_zero_missing",
        "forced_missing_disabled",
        "feature0_maxbin4",
        "feature1_maxbin7",
        "forced_zero_cut",
    ] {
        assert!(
            cases.iter().any(|c| c["name"] == required),
            "missing fixture {required}"
        );
    }
    for case in cases {
        let p = &case["params"];
        let mut values: Vec<_> = case["sample_values"]
            .as_array()
            .unwrap()
            .iter()
            .map(number)
            .collect();
        let total = case["total_sample_count"].as_u64().unwrap() as usize;
        values.resize(total, 0.0);
        let c = Config {
            // Deliberately different global cap proves that the feature cap wins.
            max_bin: 63,
            max_bin_by_feature: vec![p["max_bin"].as_u64().unwrap() as usize],
            forced_bins: vec![
                case["forced_upper_bounds"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(number)
                    .collect(),
            ],
            min_data_in_bin: p["min_data_in_bin"].as_u64().unwrap() as usize,
            min_data_in_leaf: p["min_split_data"].as_u64().unwrap() as usize,
            feature_pre_filter: p["pre_filter"].as_bool().unwrap(),
            use_missing: p["use_missing"].as_bool().unwrap(),
            zero_as_missing: p["zero_as_missing"].as_bool().unwrap(),
            ..Config::default()
        };
        let pointers: Vec<_> = (0..=total).collect();
        let indexes: Vec<_> = (0..total).collect();
        let datasets = [
            Dataset::from_dense(&values, total, 1, vec![0.0; total], &c).unwrap(),
            Dataset::from_csr(
                &pointers,
                &vec![0; total],
                &values,
                total,
                1,
                vec![0.0; total],
                &c,
            )
            .unwrap(),
            Dataset::from_csc(
                &[0, total],
                &indexes,
                &values,
                total,
                1,
                vec![0.0; total],
                &c,
            )
            .unwrap(),
        ];
        for d in datasets {
            let mapper = d.get_mapper(0).unwrap();
            assert_eq!(
                mapper.num_bins(),
                case["num_bin"].as_u64().unwrap() as u32,
                "{}",
                case["name"]
            );
            assert_eq!(
                mapper.default_bin(),
                case["default_bin"].as_u64().unwrap() as u32,
                "{}",
                case["name"]
            );
            assert_eq!(
                mapper.most_freq_bin(),
                case["most_freq_bin"].as_u64().unwrap() as u32,
                "{}",
                case["name"]
            );
            assert_eq!(
                format!("{:?}", mapper.missing_type()),
                case["missing_type"].as_str().unwrap(),
                "{}",
                case["name"]
            );
            let expected: Vec<_> = case["upper_bounds"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| number(v).to_bits())
                .collect();
            assert_eq!(
                mapper
                    .thresholds()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected,
                "{}",
                case["name"]
            );
            for (value, expected) in case["probe_values"]
                .as_array()
                .unwrap()
                .iter()
                .zip(case["value_bins"].as_array().unwrap())
            {
                assert_eq!(
                    mapper.bin(number(value)),
                    expected.as_u64().unwrap() as u32,
                    "{} probe {value}",
                    case["name"]
                );
            }
            let subset = d.subset(&[total - 1, 0]).unwrap();
            let validation = d.validation_from_dense(&[-999.0], 1, 1, vec![0.0]).unwrap();
            assert!(std::ptr::eq(mapper, subset.get_mapper(0).unwrap()));
            assert!(std::ptr::eq(mapper, validation.get_mapper(0).unwrap()));
        }
    }
}

#[test]
fn feature_caps_are_selected_by_original_column_index() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/forced_bins.json")).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    let a = cases
        .iter()
        .find(|c| c["name"] == "feature0_maxbin4")
        .unwrap();
    let b = cases
        .iter()
        .find(|c| c["name"] == "feature1_maxbin7")
        .unwrap();
    let mut values: Vec<_> = a["sample_values"]
        .as_array()
        .unwrap()
        .iter()
        .map(number)
        .collect();
    let total = a["total_sample_count"].as_u64().unwrap() as usize;
    values.resize(total, 0.0);
    let matrix: Vec<_> = values.iter().flat_map(|&v| [v, v]).collect();
    let c = Config {
        max_bin: 63,
        max_bin_by_feature: vec![4, 7],
        min_data_in_bin: 1,
        min_data_in_leaf: 1,
        feature_pre_filter: false,
        ..Config::default()
    };
    let d = Dataset::from_dense(&matrix, total, 2, vec![0.0; total], &c).unwrap();
    for (feature, expected) in [a, b].iter().enumerate() {
        let bounds: Vec<_> = expected["upper_bounds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| number(v).to_bits())
            .collect();
        assert_eq!(
            d.get_mapper(feature)
                .unwrap()
                .thresholds()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            bounds
        );
    }
}
