// Private learner tests; compiled as a module of src/learner.rs.
use super::*;
#[test]
fn quantized_histograms_reuse_casts_for_subsets_masks_and_cache_replay() {
    const N: usize = 2048;
    const P: usize = 8;
    let root: Vec<_> = (0..N).collect();
    let small: Vec<_> = (0..512).collect();
    let child: Vec<_> = (512..768).collect();
    let large: Vec<_> = (512..N).collect();
    let unsorted = vec![2047, 0, 1025, 3, 800];
    let mask: Vec<_> = (0..P).map(|f| f % 3 != 1).collect();
    let g: Vec<_> = (0..N).map(|r| (r % 37) as f32 - 18.0).collect();
    let variable_h: Vec<_> = (0..N).map(|r| (r % 17) as f32 / 16.0).collect();
    let serial = crate::parallel::Executor::new(1).unwrap();
    let parallel = crate::parallel::Executor::new(2).unwrap();
    for enable_bundle in [false, true] {
        let base = Config {
            enable_bundle,
            min_data_in_leaf: 1,
            min_data_in_bin: 1,
            feature_pre_filter: false,
            use_quantized_grad: true,
            ..Config::default()
        };
        let x: Vec<_> = (0..N)
            .flat_map(|r| (0..P).map(move |f| f64::from(r % P == f)))
            .collect();
        let dense = Dataset::from_dense(&x, N, P, vec![0.; N], &base).unwrap();
        let sparse = Dataset::from_csr(
            &(0..=N).collect::<Vec<_>>(),
            &(0..N).map(|r| r % P).collect::<Vec<_>>(),
            &vec![1.; N],
            N,
            P,
            vec![0.; N],
            &base,
        )
        .unwrap();
        if enable_bundle {
            assert!(dense.bin_storage_bytes() < N * P / 4);
            assert!(sparse.bin_storage_bytes() < N * 2);
        }
        for data in [&dense, &sparse] {
            for stochastic_rounding in [false, true] {
                let mut config = Config {
                    stochastic_rounding,
                    ..base.clone()
                };
                for (gradients, hessians, constant) in [
                    (g.clone(), variable_h.clone(), false),
                    (g.clone(), vec![1.; N], true),
                    (vec![0.; N], vec![0.; N], false),
                    (vec![0.; N], vec![0.; N], true),
                ] {
                    let q = Quantizer::new(42, N)
                        .discretize(&gradients, &hessians, constant, &config)
                        .unwrap();
                    let storage = (q.histogram_gradient.as_ptr(), q.histogram_hessian.as_ptr());
                    for (row_wise, executor) in
                        [(false, &serial), (false, &parallel), (true, &parallel)]
                    {
                        config.force_row_wise = row_wise;
                        config.force_col_wise = !row_wise;
                        // Keep the previous implementation as the exact numerical oracle.
                        let old = |rows: &[usize], active: Option<&[bool]>| {
                            let cast_g: Vec<f32> = q.gradient.iter().map(|&v| v as f32).collect();
                            let cast_h: Vec<f32> = q.hessian.iter().map(|&v| v as f32).collect();
                            let mut bins = histogram::build_training(
                                data, rows, &cast_g, &cast_h, false, &config, executor, active,
                            )
                            .unwrap();
                            histogram::fix(
                                data,
                                &mut bins,
                                rows.iter().map(|&r| q.gradient[r]).sum(),
                                rows.iter().map(|&r| q.hessian[r]).sum(),
                            );
                            for feature in &mut bins {
                                for bin in feature {
                                    bin.gradient *= q.gradient_scale;
                                    bin.hessian *= q.hessian_scale;
                                }
                            }
                            bins
                        };
                        let bits = |bins: &[Vec<BinStats>]| {
                            bins.iter()
                                .flat_map(|f| {
                                    f.iter()
                                        .map(|b| (b.gradient.to_bits(), b.hessian.to_bits()))
                                })
                                .collect::<Vec<_>>()
                        };
                        for rows in [&root, &small, &large, &unsorted] {
                            for active in [None, Some(mask.as_slice())] {
                                let actual = build_hist(
                                    data,
                                    rows,
                                    &gradients,
                                    &hessians,
                                    constant,
                                    Some(&q),
                                    &config,
                                    executor,
                                    active,
                                )
                                .unwrap();
                                assert_eq!(bits(&actual), bits(&old(rows, active)));
                                assert_eq!(
                                    storage,
                                    (q.histogram_gradient.as_ptr(), q.histogram_hessian.as_ptr())
                                );
                            }
                        }
                        // Two evicted large-leaf histograms replay parent-minus-child arithmetic.
                        let direct = |rows: &Vec<usize>| HistogramRecipe::Direct {
                            rows: rows.clone(),
                            gradient: 0.,
                            hessian: 0.,
                        };
                        let replay = vec![
                            direct(&root),
                            direct(&small),
                            HistogramRecipe::Difference {
                                parent: 0,
                                small: 1,
                            },
                            direct(&child),
                            HistogramRecipe::Difference {
                                parent: 2,
                                small: 3,
                            },
                        ];
                        let mut leaf = Leaf {
                            rows: (768..N).collect(),
                            depth: 2,
                            gradient: 0.,
                            hessian: 0.,
                            histograms: Vec::new(),
                            split: None,
                            candidates: Vec::new(),
                            active: Vec::new(),
                            branch: Vec::new(),
                            bounds: (f64::NEG_INFINITY, f64::INFINITY),
                            parent_output: 0.,
                            advanced_bounds: Vec::new(),
                            quant_scales: Some((q.gradient_scale, q.hessian_scale)),
                        };
                        let mut expected = old(&root, None);
                        subtract_in_place(&mut expected, &old(&small, None)).unwrap();
                        subtract_in_place(&mut expected, &old(&child, None)).unwrap();
                        for _ in 0..2 {
                            leaf.histograms.clear();
                            restore_hist(
                                &mut leaf,
                                4,
                                &replay,
                                data,
                                &gradients,
                                &hessians,
                                constant,
                                Some(&q),
                                &config,
                                executor,
                            )
                            .unwrap();
                            assert_eq!(bits(&leaf.histograms), bits(&expected));
                            assert_eq!(
                                storage,
                                (q.histogram_gradient.as_ptr(), q.histogram_hessian.as_ptr())
                            );
                        }
                    }
                }
            }
        }
    }
}

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
