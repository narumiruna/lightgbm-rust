// Private learner tests; compiled as a module of src/learner.rs.
use super::*;
#[test]
fn categorical_bitset_safety() {
    assert_eq!(
        category_bitset(&[0, 31, 32, 40], 0).unwrap(),
        [0x80000001, 257]
    );
    assert!(category_bitset(&[], 0).is_err());
    for (codes, existing) in [
        (vec![u32::MAX], 0),
        (vec![32], 999999),
        (vec![0], usize::MAX),
    ] {
        assert!(matches!(
            category_bitset(&codes, existing),
            Err(Error::Unsupported(_))
        ));
    }
}
#[test]
fn unsorted_selected_rows_keep_partition_order_and_missing_indices() {
    let config = Config {
        min_data_in_leaf: 1,
        min_data_in_bin: 1,
        feature_pre_filter: false,
        num_leaves: 3,
        ..Config::default()
    };
    let data = Dataset::from_dense(&[0., 1., 2., 3., 4., 5.], 6, 1, vec![0.; 6], &config).unwrap();
    let mut learner = TreeLearner::new(&config, &data).unwrap();
    let selected = [5, 1, 4, 0];
    let fitted = learner
        .fit(
            &data,
            &[-2., -1., 0., 0., 1., 2.],
            &[1.; 6],
            &config,
            &selected,
            false,
        )
        .unwrap();
    assert_eq!(fitted.leaf_indices[2], usize::MAX);
    assert_eq!(fitted.leaf_indices[3], usize::MAX);
    for (leaf, rows) in fitted.leaf_rows.iter().enumerate() {
        let want: Vec<usize> = selected
            .iter()
            .copied()
            .filter(|&r| binned_leaf(&fitted.tree, &fitted.threshold_bins, &data, r) == leaf)
            .collect();
        assert_eq!(rows, &want);
    }
    assert!(
        learner
            .fit(&data, &[0.; 6], &[1.; 6], &config, &[0, 0], false)
            .is_err()
    );
    assert!(
        learner
            .fit(&data, &[0.; 6], &[1.; 6], &config, &[6], false)
            .is_err()
    );
}
#[test]
fn native_feature_samples_match_including_filtered_columns() {
    let f: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/sampling.json")).unwrap();
    let mut count = 0;
    for case in f["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["kind"] == "features")
    {
        count += 1;
        let pairs = case["params"].as_object().unwrap().iter().map(|(k, v)| {
            (
                k.clone(),
                v.as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| v.to_string()),
            )
        });
        let config = Config::from_pairs(pairs).unwrap();
        let rows = case["rows"].as_array().unwrap();
        let cols = rows[0].as_array().unwrap().len();
        let x: Vec<f64> = rows
            .iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect();
        let data = Dataset::from_dense(
            &x,
            rows.len(),
            cols,
            serde_json::from_value(case["labels"].clone()).unwrap(),
            &config,
        )
        .unwrap();
        let mut learner = TreeLearner::new(&config, &data).unwrap();
        let inner: Vec<i64> =
            serde_json::from_value(case["inner_feature_indices"].clone()).unwrap();
        let n = learner.eligible.len();
        let mut leaf = Leaf {
            rows: (0..rows.len()).collect(),
            depth: 0,
            gradient: 0.,
            hessian: 1.,
            histograms: Vec::new(),
            split: None,
            candidates: Vec::new(),
            active: Vec::new(),
            branch: Vec::new(),
            bounds: (f64::NEG_INFINITY, f64::INFINITY),
            parent_output: 0.,
            advanced_bounds: Vec::new(),
            quant_scales: None,
        };
        for (iteration, step) in case["steps"].as_array().unwrap().iter().enumerate() {
            if iteration > 0 {
                learner.sample_tree(&config).unwrap();
            }
            let mut tree = vec![0; n];
            for &f in &learner.tree_features {
                tree[inner[f] as usize] = 1;
            }
            assert_eq!(
                tree,
                serde_json::from_value::<Vec<i32>>(step["by_tree"].clone()).unwrap(),
                "{} tree{iteration}",
                case["name"]
            );
            for expected in step["by_node"].as_array().unwrap() {
                let mask = learner.sample_node(&leaf, &config).unwrap();
                let mut node = vec![0; n];
                for &f in &learner.eligible {
                    node[inner[f] as usize] = i32::from(mask[f]);
                }
                assert_eq!(
                    node,
                    serde_json::from_value::<Vec<i32>>(expected.clone()).unwrap(),
                    "{} node iteration{iteration}",
                    case["name"]
                );
            }
            leaf.branch.clear();
        }
    }
    assert_eq!(count, 4);
}
