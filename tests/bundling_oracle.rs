use lightgbm_rust::{Config, Dataset, Model, Trainer};
use serde_json::Value;

fn config(case: &Value) -> Config {
    let mut pairs: Vec<(String, String)> = case["params"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                v.as_str().map_or_else(|| v.to_string(), str::to_owned),
            )
        })
        .collect();
    pairs.push(("num_iterations".into(), "2".into()));
    Config::from_pairs(pairs).unwrap()
}

fn data(case: &Value) -> (Vec<f64>, usize, usize, Vec<f32>) {
    let rows = case["rows"].as_array().unwrap();
    let p = rows[0].as_array().unwrap().len();
    let x = rows
        .iter()
        .flat_map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap_or(f64::NAN))
        })
        .collect();
    (
        x,
        rows.len(),
        p,
        serde_json::from_value(case["labels"].clone()).unwrap(),
    )
}

fn sparse(x: &[f64], n: usize, p: usize, csr: bool) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut pointers = vec![0];
    let mut indexes = Vec::new();
    let mut values = Vec::new();
    for a in 0..if csr { n } else { p } {
        for b in 0..if csr { p } else { n } {
            let v = if csr { x[a * p + b] } else { x[b * p + a] };
            if v != 0.0 {
                indexes.push(b);
                values.push(v);
            }
        }
        pointers.push(indexes.len());
    }
    (pointers, indexes, values)
}

#[test]
fn native_bundle_bins_trees_and_predictions_dense_csr_csc() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/bundling.json")).unwrap();
    assert_eq!(
        fixture["upstream_commit"],
        "5d84f3f222b41409a6983d2ac90513ad9702fa9e"
    );
    assert_eq!(
        fixture["loaded_library"]["loaded_sha256"],
        "9ed1d46d2d60bac2966fd03bd81e7d192025804f6a3e3ef24c3aa483d0ec7c1e"
    );
    let cases = fixture["cases"].as_array().unwrap();
    for name in [
        "exclusive",
        "sparse",
        "conflict",
        "missing",
        "zero_missing",
        "frequent_nonzero",
        "filtered",
        "categorical",
        "categorical_nan",
        "reordered",
        "extra_trees",
    ] {
        for flag in ["true", "false"] {
            assert!(cases.iter().any(|c| c["name"] == format!("{name}_{flag}")));
        }
    }
    for case in cases {
        let cfg = config(case);
        let (x, n, p, y) = data(case);
        let (rp, ci, rv) = sparse(&x, n, p, true);
        let (cp, ri, cv) = sparse(&x, n, p, false);
        let datasets = [
            Dataset::from_dense(&x, n, p, y.clone(), &cfg).unwrap(),
            Dataset::from_csr(&rp, &ci, &rv, n, p, y.clone(), &cfg).unwrap(),
            Dataset::from_csc(&cp, &ri, &cv, n, p, y, &cfg).unwrap(),
        ];
        let reference = Model::from_text(
            &std::fs::read_to_string(format!(
                "{}/tests/fixtures/{}",
                env!("CARGO_MANIFEST_DIR"),
                case["model"].as_str().unwrap()
            ))
            .unwrap(),
        )
        .unwrap();
        for dataset in &datasets {
            for f in 0..p {
                if case["maps"][f].is_null() {
                    assert!(dataset.get_mapper(f).unwrap().is_trivial());
                    continue;
                }
                let m = dataset.get_mapper(f).unwrap();
                assert_eq!(
                    u64::from(m.num_bins()),
                    case["maps"][f]["num_bins"].as_u64().unwrap(),
                    "{} f{f}",
                    case["name"]
                );
                let missing = match m.missing_type() {
                    lightgbm_rust::MissingType::None => 0,
                    lightgbm_rust::MissingType::Zero => 1,
                    lightgbm_rust::MissingType::NaN => 2,
                };
                assert_eq!(
                    missing,
                    case["maps"][f]["missing_type"].as_u64().unwrap(),
                    "{} f{f}",
                    case["name"]
                );
                assert_eq!(
                    u64::from(m.default_bin()),
                    case["maps"][f]["default_bin"].as_u64().unwrap(),
                    "{} f{f}",
                    case["name"]
                );
                assert_eq!(
                    u64::from(m.most_freq_bin()),
                    case["maps"][f]["most_freq_bin"].as_u64().unwrap(),
                    "{} f{f}",
                    case["name"]
                );
                for row in 0..n {
                    assert_eq!(
                        u64::from(dataset.get_bin(row, f).unwrap()),
                        case["bins"][row][f].as_u64().unwrap(),
                        "{} row{row} f{f}",
                        case["name"]
                    );
                }
            }
            let model = Trainer::new(cfg.clone())
                .unwrap()
                .fit(dataset)
                .unwrap()
                .model;
            assert_eq!(
                model.trees().len(),
                reference.trees().len(),
                "{}",
                case["name"]
            );
            for (actual, expected) in model.trees().iter().zip(reference.trees()) {
                assert_eq!(
                    actual.split_features(),
                    expected.split_features(),
                    "{}",
                    case["name"]
                );
                assert_eq!(
                    actual.thresholds(),
                    expected.thresholds(),
                    "{}",
                    case["name"]
                );
                assert_eq!(
                    actual.left_children(),
                    expected.left_children(),
                    "{}",
                    case["name"]
                );
                assert_eq!(
                    actual.right_children(),
                    expected.right_children(),
                    "{}",
                    case["name"]
                );
                assert_eq!(
                    actual.decision_types(),
                    expected.decision_types(),
                    "{}",
                    case["name"]
                );
                assert_eq!(
                    actual.cat_threshold(),
                    expected.cat_threshold(),
                    "{}",
                    case["name"]
                );
                assert_eq!(
                    actual.leaf_counts(),
                    expected.leaf_counts(),
                    "{}",
                    case["name"]
                );
            }
            for (actual, expected) in model
                .predict_raw_dense(&x, n, p)
                .unwrap()
                .iter()
                .zip(case["raw_predictions"].as_array().unwrap())
            {
                let expected = expected.as_f64().unwrap();
                assert!(
                    (actual[0] - expected).abs() <= 1e-12 + 1e-10 * expected.abs(),
                    "{} {} != {expected}",
                    case["name"],
                    actual[0]
                );
            }
        }
    }
}

#[test]
fn packed_and_unpacked_training_matches() {
    let n = 240;
    let p = 12;
    let x: Vec<_> = (0..n)
        .flat_map(|i| (0..p).map(move |j| f64::from(i % p == j)))
        .collect();
    let cfg = Config {
        min_data_in_bin: 1,
        min_data_in_leaf: 1,
        ..Config::default()
    };
    let labels: Vec<_> = (0..n).map(|r| (r % p) as f32).collect();
    let on = Dataset::from_dense(&x, n, p, labels.clone(), &cfg).unwrap();
    let off = Dataset::from_dense(
        &x,
        n,
        p,
        labels,
        &Config {
            enable_bundle: false,
            ..cfg.clone()
        },
    )
    .unwrap();
    let a = Trainer::new(cfg.clone()).unwrap().fit(&on).unwrap();
    let b = Trainer::new(Config {
        enable_bundle: false,
        ..cfg
    })
    .unwrap()
    .fit(&off)
    .unwrap();
    assert_eq!(
        a.model.predict_raw_dense(&x, n, p).unwrap(),
        b.model.predict_raw_dense(&x, n, p).unwrap()
    );
}
