//! Focused coverage for the global numerical-feature NaN scan in linear refit.
use lightgbm_rust::{Config, Dataset, Error, Model};

fn config() -> Config {
    Config {
        linear_tree: true,
        min_data_in_bin: 1,
        min_data_in_leaf: 1,
        feature_pre_filter: false,
        ..Config::default()
    }
}

// A right-growing chain with four leaves. Leaf selection always uses the
// original model; only the global scan depends on all three split entries.
fn model(splits: [usize; 3], intercept_only: bool, categorical: bool) -> Model {
    let (counts, features, coefficients) = if intercept_only {
        ("0 0 0 0", "", "")
    } else {
        ("1 1 1 1", "0 0 0 0", "7 7 7 7")
    };
    let (cats, decisions, thresholds) = if categorical {
        (
            "num_cat=1\ncat_boundaries=0 1\ncat_threshold=8",
            "8 8 1",
            "0.5 1.5 0",
        )
    } else {
        ("num_cat=0", "8 8 8", "0.5 1.5 2.5")
    };
    Model::from_text(&format!(
        "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nlabel_index=0\n\
         max_feature_idx=2\nobjective=regression\nfeature_names=x y z\n\
         feature_infos=[0:4] [0:4] [0:4]\n\nTree=0\nnum_leaves=4\n{cats}\n\
         split_feature={} {} {}\nthreshold={thresholds}\ndecision_type={decisions}\n\
         left_child=-1 -2 -3\nright_child=1 2 -4\nleaf_value=4 4 4 4\n\
         leaf_weight=2 2 2 2\nleaf_count=2 2 2 2\ninternal_value=4 4 4\n\
         internal_weight=8 6 4\ninternal_count=8 6 4\nsplit_gain=1 1 1\n\
         is_linear=1\nleaf_const=20 20 20 20\nnum_features={counts}\n\
         leaf_features={features}\nleaf_coeff={coefficients}\nshrinkage=0.5\n\nend of trees\n",
        splits[0], splits[1], splits[2]
    ))
    .unwrap()
}

fn refit(model: &Model, data: &Dataset, raw: &[[f64; 3]], config: &Config) -> Model {
    let leaves: Vec<_> = raw
        .iter()
        .map(|row| model.predict_leaf_row(row).unwrap())
        .collect();
    model.refit(data, &leaves, config, 0.5).unwrap()
}

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
}

#[test]
fn repeated_numerical_features_match_unique_equivalent_splits() {
    let config = config();
    let raw: Vec<_> = [0., 0.25, 1., 1.25, 2., 2.25, 3., 3.25]
        .map(|x| [x; 3])
        .into();
    let labels = raw.iter().map(|row| (2. + 3. * row[0]) as f32).collect();
    let data = Dataset::from_dense(raw.as_flattened(), raw.len(), 3, labels, &config).unwrap();
    let repeated = refit(&model([0, 0, 0], false, false), &data, &raw, &config);
    let unique = refit(&model([0, 1, 2], false, false), &data, &raw, &config);
    assert_eq!(
        repeated.trees()[0].leaf_values(),
        unique.trees()[0].leaf_values()
    );
    let linear = repeated.trees()[0].linear_leaves().unwrap();
    assert_eq!(
        serde_json::to_value(linear).unwrap(),
        serde_json::to_value(unique.trees()[0].linear_leaves().unwrap()).unwrap()
    );
    for leaf in 0..4 {
        assert_eq!(linear.features()[leaf], [0]);
        close(linear.coefficients()[leaf][0], 4.25); // 0.5 * 7 + 0.5 * 3 * 0.5
        close(linear.constants()[leaf], 10.5); // 0.5 * 20 + 0.5 * 2 * 0.5
    }
}

#[test]
fn every_unique_split_feature_can_activate_nan_branch_outside_selected_leaf() {
    let config = config();
    for splits in [[0, 0, 0], [0, 1, 0], [0, 1, 2], [2, 0, 1]] {
        for feature in splits
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>()
        {
            let original = model(splits, true, false);
            let mut raw = [[0.; 3], [0.25; 3], [3.; 3]];
            let data = Dataset::from_dense(raw.as_flattened(), 3, 3, vec![6.; 3], &config).unwrap();
            let finite = refit(&original, &data, &raw, &config);
            close(
                finite.trees()[0].linear_leaves().unwrap().constants()[0],
                11.5,
            );
            raw[2][feature] = f64::NAN;
            assert_eq!(original.predict_leaf_row(&raw[2]).unwrap(), [3]);
            let data = Dataset::from_dense(raw.as_flattened(), 3, 3, vec![6.; 3], &config).unwrap();
            let missing = refit(&original, &data, &raw, &config);
            // Leaf 0 has no NaN and no selected features. Global HAS_NAN makes
            // its count zero, so native refit uses the fallback intercept.
            close(
                missing.trees()[0].linear_leaves().unwrap().constants()[0],
                10.875,
            );
        }
    }
}

#[test]
fn categorical_nan_is_excluded_from_global_scan_and_raw_access() {
    let mut config = config();
    config.categorical_features = vec![2];
    let original = model([0, 1, 2], true, true);
    let raw = [[0.; 3], [0.25, 0.25, 0.], [3., 3., f64::NAN]];
    let data = Dataset::from_dense(raw.as_flattened(), 3, 3, vec![6.; 3], &config).unwrap();
    assert!(data.raw_value(2, 2).is_err());
    let actual = refit(&original, &data, &raw, &config);
    close(
        actual.trees()[0].linear_leaves().unwrap().constants()[0],
        11.5,
    );
}

#[test]
fn nan_does_not_hide_nonfinite_raw_features() {
    let config = config();
    let original = model([0, 1, 2], true, false);
    for value in [f64::INFINITY, f64::NEG_INFINITY, f64::MAX] {
        let raw = [[f64::NAN, 0., 0.], [3., 3., value]];
        let data = Dataset::from_dense(raw.as_flattened(), 2, 3, vec![6.; 2], &config).unwrap();
        assert!(data.raw_value(1, 2).unwrap().is_infinite());
        let leaves: Vec<_> = raw
            .iter()
            .map(|row| original.predict_leaf_row(row).unwrap())
            .collect();
        let before = original.to_text().unwrap();
        assert!(matches!(
            original.refit(&data, &leaves, &config, 0.5),
            Err(Error::InvalidInput(message)) if message.contains("linear raw features must be finite or NaN")
        ));
        assert_eq!(original.to_text().unwrap(), before);
    }
}

#[test]
fn repeated_feature_scan_keeps_raw_f32_precision() {
    let config = config();
    let original = model([0, 0, 0], false, false);
    let raw = [0.1, 0.3, 1.1, 1.3, 2.1, 2.3, 3.1, 3.3].map(|x| [x; 3]);
    let rounded = raw.map(|row| row.map(|x| f64::from(x as f32)));
    assert_ne!(raw[0][0], rounded[0][0]);
    let mut results = Vec::new();
    for rows in [&raw, &rounded] {
        let data = Dataset::from_dense(
            rows.as_flattened(),
            8,
            3,
            vec![1., 2., 3., 4., 5., 6., 7., 8.],
            &config,
        )
        .unwrap();
        results.push(refit(&original, &data, rows, &config).to_text().unwrap());
    }
    assert_eq!(results[0], results[1]);
}
