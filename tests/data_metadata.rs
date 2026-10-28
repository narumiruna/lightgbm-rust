use lightgbm_rust::{Config, Dataset};
use serde_json::Value;

#[test]
fn native_dataset_metadata_matches_with_lossless_position_identifiers() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/dataset_metadata.json")).unwrap();
    assert_eq!(
        fixture["upstream_commit"],
        "5d84f3f222b41409a6983d2ac90513ad9702fa9e"
    );
    let cases = fixture["cases"].as_array().unwrap();
    for required in [
        "labels_only",
        "weighted_scalar_init",
        "weighted_multiclass_init",
        "ranking_groups_positions",
        "ranking_unweighted",
        "negative_position_codes",
    ] {
        assert!(
            cases.iter().any(|c| c["name"] == required),
            "missing fixture {required}"
        );
    }
    for case in cases {
        let labels: Vec<_> = case["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let nrows = labels.len();
        let mut d = Dataset::from_dense(&vec![0.0; nrows], nrows, 1, labels, &config()).unwrap();
        if let Some(values) = case["weights"].as_array() {
            d = d
                .with_weights(values.iter().map(|v| v.as_f64().unwrap() as f32).collect())
                .unwrap();
        }
        if let Some(values) = case["initial_scores"].as_array() {
            d = d
                .with_initial_scores(
                    values.iter().map(|v| v.as_f64().unwrap()).collect(),
                    case["num_class"].as_u64().unwrap() as usize,
                )
                .unwrap();
        }
        if let Some(values) = case["groups"].as_array() {
            d = d
                .with_groups(
                    values
                        .iter()
                        .map(|v| v.as_u64().unwrap() as usize)
                        .collect(),
                )
                .unwrap();
        }
        if let Some(values) = case["positions"].as_array() {
            d = d
                .with_positions(values.iter().map(|v| v.as_i64().unwrap() as i32).collect())
                .unwrap();
        }
        let e = &case["expected"];
        assert_eq!(
            d.labels(),
            e["labels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect::<Vec<_>>(),
            "{}",
            case["name"]
        );
        if let Some(w) = d.weights() {
            assert_eq!(
                w,
                e["weights"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect::<Vec<_>>()
            );
        }
        if let Some(s) = d.initial_scores() {
            assert_eq!(
                s,
                e["initial_scores"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect::<Vec<_>>()
            );
        }
        if let Some(groups) = d.groups() {
            let mut boundaries = vec![0];
            for &len in groups {
                boundaries.push(boundaries.last().unwrap() + len);
            }
            assert_eq!(
                serde_json::to_value(boundaries).unwrap(),
                e["query_boundaries"]
            );
        }
        if let Some(positions) = d.positions() {
            // The public API keeps the original codes. First-seen encoding must
            // reproduce native codes without assigning an ordinal meaning.
            let mut ids = Vec::new();
            let encoded: Vec<_> = positions
                .iter()
                .map(|&p| {
                    if let Some(i) = ids.iter().position(|&id| id == p) {
                        i
                    } else {
                        ids.push(p);
                        ids.len() - 1
                    }
                })
                .collect();
            assert_eq!(serde_json::to_value(encoded).unwrap(), e["positions"]);
            assert_eq!(
                serde_json::to_value(ids.iter().map(i32::to_string).collect::<Vec<_>>()).unwrap(),
                e["position_ids"]
            );
        }
    }
}

fn config() -> Config {
    Config {
        min_data_in_leaf: 1,
        min_data_in_bin: 1,
        ..Config::default()
    }
}

fn dataset() -> Dataset {
    Dataset::from_dense(
        &[0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0, 4.0, 14.0],
        5,
        2,
        vec![0.0, 1.0, 2.0, 3.0, 4.0],
        &config(),
    )
    .unwrap()
}

#[test]
fn metadata_is_checked_and_scores_remain_class_major() {
    let d = dataset();
    assert!(d.initial_scores().is_none());
    assert!(d.groups().is_none());
    assert!(d.positions().is_none());
    for classes in [0, 1, 2, usize::MAX] {
        assert!(
            d.clone()
                .with_initial_scores(vec![1.0; 3], classes)
                .is_err()
        );
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(d.clone().with_initial_scores(vec![value; 5], 1).is_err());
    }
    for groups in [vec![], vec![0, 4], vec![2, 2], vec![usize::MAX, 6], vec![6]] {
        assert!(d.clone().with_groups(groups).is_err());
    }
    assert!(d.clone().with_positions(vec![1; 4]).is_err());
    let scores: Vec<_> = (0..15).map(f64::from).collect();
    let d = d
        .with_initial_scores(scores.clone(), 3)
        .unwrap()
        .with_positions(vec![i32::MIN, 9, 9, i32::MAX, -7])
        .unwrap()
        .with_weights(vec![1.0, 2.0, 3.0, 4.0, 5.0])
        .unwrap();
    assert_eq!(d.clone().initial_scores(), Some(scores.as_slice()));
    let s = d.subset(&[4, 1, 4]).unwrap();
    assert_eq!(
        s.initial_scores(),
        Some(&[4.0, 1.0, 4.0, 9.0, 6.0, 9.0, 14.0, 11.0, 14.0][..])
    );
    assert_eq!(s.positions(), Some(&[-7, 9, -7][..]));
    assert_eq!(s.weights(), Some(&[5.0, 2.0, 5.0][..]));
    assert_eq!(s.labels(), &[4.0, 1.0, 4.0]);
}

#[test]
fn ranking_subsets_preserve_complete_noninterleaved_groups() {
    let d = dataset().with_groups(vec![2, 1, 2]).unwrap();
    for rows in [
        vec![0],
        vec![1, 0],
        vec![0, 2],
        vec![0, 2, 1],
        vec![0, 1, 0, 1],
        vec![3, 4, 1],
        vec![3, 0, 4, 1],
    ] {
        assert!(d.subset(&rows).is_err(), "{rows:?}");
    }
    let s = d.subset(&[3, 4, 0, 1]).unwrap();
    assert_eq!(s.groups(), Some(&[2, 2][..]));
    assert_eq!(s.labels(), &[3.0, 4.0, 0.0, 1.0]);
    assert_eq!(d.subset(&[2]).unwrap().groups(), Some(&[1][..]));
    assert_eq!(s.subset(&[2, 3]).unwrap().groups(), Some(&[2][..]));
    assert_eq!(d.clone().groups(), Some(&[2, 1, 2][..]));
}

#[test]
fn validation_owns_row_metadata_and_shares_feature_metadata() {
    let d = dataset()
        .with_feature_names(vec!["first value".into(), "second".into()])
        .unwrap()
        .with_initial_scores(vec![1.0; 15], 3)
        .unwrap()
        .with_positions(vec![-9; 5])
        .unwrap()
        .with_groups(vec![2, 3])
        .unwrap()
        .with_weights(vec![1.0; 5])
        .unwrap();
    let v = d
        .validation_from_dense(&[100.0, -100.0, 200.0, -200.0], 2, 2, vec![9.0, 8.0])
        .unwrap();
    assert!(v.initial_scores().is_none());
    assert!(v.weights().is_none());
    assert!(v.positions().is_none());
    assert!(v.groups().is_none());
    assert_eq!(v.feature_names(), ["first_value", "second"]);
    assert!(std::ptr::eq(
        v.get_mapper(0).unwrap(),
        d.get_mapper(0).unwrap()
    ));
    assert_eq!(v.get_mapper(0).unwrap().numerical_range(), Some((0.0, 4.0)));
    let v = v
        .with_groups(vec![2])
        .unwrap()
        .with_positions(vec![i32::MIN, i32::MAX])
        .unwrap()
        .with_initial_scores(vec![1.0, 2.0], 1)
        .unwrap();
    assert_eq!(v.groups(), Some(&[2][..]));
    assert_eq!(d.groups(), Some(&[2, 3][..]));
    assert_eq!(
        d.subset(&[2, 3, 4]).unwrap().feature_names(),
        d.feature_names()
    );
}

#[test]
fn feature_name_defaults_normalization_and_rejections() {
    let d = dataset();
    assert_eq!(d.feature_names(), ["Column_0", "Column_1"]);
    for names in [
        vec![],
        vec!["a".into()],
        vec!["a".into(), "a".into()],
        vec!["a b".into(), "a_b".into()],
    ] {
        assert!(d.clone().with_feature_names(names).is_err());
    }
    for name in [
        "",
        "bad\nname",
        "bad\tname",
        "bad\0name",
        "[x",
        "x]",
        "x{",
        "x}",
        "x:",
        "x,",
        "x\"",
    ] {
        assert!(
            d.clone()
                .with_feature_names(vec![name.into(), "second".into()])
                .is_err(),
            "{name:?}"
        );
    }
    assert_eq!(
        d.with_feature_names(vec!["number one".into(), "category".into()])
            .unwrap()
            .feature_names(),
        ["number_one", "category"]
    );
}

#[test]
fn raw_values_are_opt_in_checked_and_preserved_for_dense_and_sparse() {
    assert!(!dataset().has_raw_values());
    assert!(dataset().raw_value(0, 0).is_err());
    let c = Config {
        linear_tree: true,
        categorical_features: vec![1],
        ..config()
    };
    let dense = [0.0, 1.0, 1.25, 2.0, f64::NAN, 1.0];
    let d = Dataset::from_dense(&dense, 3, 2, vec![0.0; 3], &c).unwrap();
    let csr = Dataset::from_csr(
        &[0, 1, 3, 5],
        &[1, 0, 1, 0, 1],
        &[1.0, 1.25, 2.0, f64::NAN, 1.0],
        3,
        2,
        vec![0.0; 3],
        &c,
    )
    .unwrap();
    let csc = Dataset::from_csc(
        &[0, 2, 5],
        &[1, 2, 0, 1, 2],
        &[1.25, f64::NAN, 1.0, 2.0, 1.0],
        3,
        2,
        vec![0.0; 3],
        &c,
    )
    .unwrap();
    for d in [&d, &csr, &csc] {
        assert!(d.has_raw_values());
        assert_eq!(d.raw_value(0, 0).unwrap(), 0.0);
        assert_eq!(d.raw_value(1, 0).unwrap(), 1.25);
        assert!(d.raw_value(2, 0).unwrap().is_nan());
        assert!(d.raw_value(0, 1).is_err());
        assert!(d.raw_value(3, 0).is_err());
        assert!(d.raw_value(0, 2).is_err());
        let subset = d.subset(&[2, 1, 1]).unwrap();
        assert!(subset.raw_value(0, 0).unwrap().is_nan());
        assert_eq!(subset.raw_value(1, 0).unwrap(), 1.25);
        assert_eq!(subset.raw_value(2, 0).unwrap(), 1.25);
        assert!(
            d.validation_from_dense(&[5.0, 9.0], 1, 2, vec![0.0])
                .unwrap()
                .has_raw_values()
        );
    }
}

#[test]
fn retained_raw_values_use_native_float32_precision() {
    let value = 1.23456789_f64;
    let c = Config {
        linear_tree: true,
        ..config()
    };
    let d = Dataset::from_dense(&[value], 1, 1, vec![0.0], &c).unwrap();
    assert_ne!(value, f64::from(value as f32));
    assert_eq!(d.raw_value(0, 0).unwrap(), f64::from(value as f32));
}

#[test]
fn raw_retention_rejects_sparse_dimension_amplification_before_materialization() {
    let c = Config {
        linear_tree: true,
        ..config()
    };
    // Very small pointer/label arrays declare more than 16 Mi dense elements.
    let rows = 257;
    let cols = 65_536;
    assert!(
        Dataset::from_csr(
            &vec![0; rows + 1],
            &[],
            &[],
            rows,
            cols,
            vec![0.0; rows],
            &c
        )
        .is_err()
    );
    assert!(
        Dataset::from_csc(
            &vec![0; cols + 1],
            &[],
            &[],
            rows,
            cols,
            vec![0.0; rows],
            &c
        )
        .is_err()
    );
}

#[test]
fn source_category_values_have_checked_domains_for_all_layouts_and_validation() {
    let c = Config {
        categorical_features: vec![0],
        ..config()
    };
    for v in [f64::INFINITY, f64::NEG_INFINITY, 2147483648.0] {
        assert!(Dataset::from_dense(&[v], 1, 1, vec![0.0], &c).is_err());
        assert!(Dataset::from_csr(&[0, 1], &[0], &[v], 1, 1, vec![0.0], &c).is_err());
        assert!(Dataset::from_csc(&[0, 1], &[0], &[v], 1, 1, vec![0.0], &c).is_err());
        let d = Dataset::from_dense(&[0.0], 1, 1, vec![0.0], &c).unwrap();
        assert!(d.validation_from_dense(&[v], 1, 1, vec![0.0]).is_err());
    }
}

#[test]
fn per_feature_caps_and_forced_bins_require_matching_feature_counts() {
    for c in [
        Config {
            max_bin_by_feature: vec![3, 4],
            ..config()
        },
        Config {
            forced_bins: vec![vec![], vec![1.0]],
            ..config()
        },
    ] {
        assert!(Dataset::from_dense(&[0.0], 1, 1, vec![0.0], &c).is_err());
        assert!(Dataset::from_csr(&[0, 0], &[], &[], 1, 1, vec![0.0], &c).is_err());
        assert!(Dataset::from_csc(&[0, 0], &[], &[], 1, 1, vec![0.0], &c).is_err());
    }
}
