use lightgbm_rust::{Config, Dataset, Model, Objective, ParallelPredictor, Trainer};

fn values(rows: usize, cols: usize) -> Vec<f64> {
    (0..rows * cols)
        .map(|i| {
            let row = i / cols;
            let col = i % cols;
            if (row + col).is_multiple_of(37) {
                f64::NAN
            } else if (row * 7 + col) % 11 < 6 {
                0.0
            } else {
                ((row * (col + 3) + col * 11) % 97) as f64 / 17.0
            }
        })
        .collect()
}

fn config() -> Config {
    Config {
        num_iterations: 7,
        num_leaves: 9,
        min_data_in_leaf: 3,
        min_data_in_bin: 1,
        max_bin: 23,
        feature_pre_filter: false,
        deterministic: true,
        force_col_wise: true,
        ..Config::default()
    }
}

fn assert_execution_modes(data: &Dataset, config: &Config) -> Model {
    let expected = Trainer::new(config.clone())
        .unwrap()
        .fit(data)
        .unwrap()
        .model;
    let text = expected.to_text().unwrap();
    for threads in [1, 2, 4] {
        for row_wise in [false, true] {
            for cache_mib in [-1.0, 0.000001, 0.01] {
                let mut c = config.clone();
                c.num_threads = threads;
                c.force_col_wise = !row_wise;
                c.force_row_wise = row_wise;
                c.histogram_pool_size = cache_mib;
                for _ in 0..2 {
                    let actual = Trainer::new(c.clone()).unwrap().fit(data).unwrap().model;
                    assert_eq!(
                        actual.to_text().unwrap(),
                        text,
                        "threads={threads}, row_wise={row_wise}, cache_mib={cache_mib}"
                    );
                }
            }
        }
    }
    expected
}

#[test]
fn coarse_parallel_histogram_and_split_jobs_keep_exact_model() {
    let mut c = config();
    c.num_iterations = 3;
    c.num_leaves = 5;
    c.extra_trees = true;
    let x = values(4096, 8);
    let y = (0..4096).map(|r| ((r * 17) % 23) as f32).collect();
    let data = Dataset::from_dense(&x, 4096, 8, y, &c).unwrap();
    assert_execution_modes(&data, &c);
}

#[test]
fn dense_missing_constant_and_variable_hessians_preserve_exact_model() {
    let x = values(160, 8);
    let y: Vec<f32> = (0..160).map(|r| 1.0 + (r % 19) as f32 / 7.0).collect();
    for objective in [Objective::Regression, Objective::Poisson] {
        let mut c = config();
        c.objective = objective;
        let data = Dataset::from_dense(&x, 160, 8, y.clone(), &c).unwrap();
        assert_execution_modes(&data, &c);
    }
}

#[test]
fn categorical_extra_sampling_and_quantized_execution_are_repeatable() {
    let mut x = values(160, 8);
    for r in 0..160 {
        x[r * 8] = (r % 7) as f64;
    }
    let y: Vec<f32> = (0..160).map(|r| ((r * 17) % 23) as f32).collect();
    for quantized in [false, true] {
        let mut c = config();
        c.categorical_features = vec![0];
        c.cat_smooth = 1.0;
        c.min_data_per_group = 2;
        c.max_cat_to_onehot = 3;
        c.extra_trees = true;
        c.feature_fraction = 0.8;
        c.feature_fraction_bynode = 0.8;
        c.use_quantized_grad = quantized;
        let data = Dataset::from_dense(&x, 160, 8, y.clone(), &c).unwrap();
        assert_execution_modes(&data, &c);
    }
}

#[test]
fn sparse_histograms_and_batched_predictions_preserve_order() {
    let x = values(160, 8);
    let mut pointers = vec![0];
    let mut indices = Vec::new();
    let mut nonzeros = Vec::new();
    for row in x.chunks(8) {
        for (col, &value) in row.iter().enumerate() {
            if value != 0.0 {
                indices.push(col);
                nonzeros.push(value);
            }
        }
        pointers.push(indices.len());
    }
    let c = config();
    let data = Dataset::from_csr(
        &pointers,
        &indices,
        &nonzeros,
        160,
        8,
        (0..160).map(|r| (r % 13) as f32).collect(),
        &c,
    )
    .unwrap();
    let model = assert_execution_modes(&data, &c);
    let dense = model.predict_dense(&x, 160, 8).unwrap();
    for threads in [0, 1, 2, 4] {
        let predictor = ParallelPredictor::new(threads).unwrap();
        assert_eq!(predictor.num_threads(), threads.max(1));
        assert_eq!(predictor.predict_dense(&model, &x, 160, 8).unwrap(), dense);
        assert_eq!(
            predictor
                .predict_csr(&model, &pointers, &indices, &nonzeros)
                .unwrap(),
            dense
        );
        assert!(predictor.predict_dense(&model, &x, usize::MAX, 8).is_err());
        assert!(
            predictor
                .predict_csr(&model, &[0, usize::MAX], &[], &[])
                .is_err()
        );
        assert!(
            predictor
                .predict_csr(&model, &[0, 1], &[8], &[1.0])
                .is_err()
        );
    }
    assert!(ParallelPredictor::new(257).is_err());
}

#[test]
fn cache_replay_preserves_advanced_constraints_and_cegb_state() {
    let x = values(160, 8);
    let mut c = config();
    c.monotone_constraints = vec![1, 0, -1, 0, 0, 0, 0, 0];
    c.monotone_constraints_method = lightgbm_rust::MonotoneMethod::Advanced;
    c.path_smooth = 2.0;
    c.cegb_tradeoff = 0.1;
    c.cegb_penalty_split = 0.01;
    c.cegb_penalty_feature_coupled = vec![0.1; 8];
    c.cegb_penalty_feature_lazy = vec![0.01; 8];
    let y = (0..160).map(|r| ((r * 11) % 31) as f32).collect();
    let data = Dataset::from_dense(&x, 160, 8, y, &c).unwrap();
    assert_execution_modes(&data, &c);
}

#[test]
fn sparse_large_dimensions_keep_inactive_lazy_cegb_unallocated() {
    let rows = 8193;
    let cols = 2048;
    let mut c = config();
    c.num_iterations = 1;
    c.num_threads = 4;
    c.histogram_pool_size = 0.000001;
    let data =
        Dataset::from_csr(&vec![0; rows + 1], &[], &[], rows, cols, vec![0.; rows], &c).unwrap();
    Trainer::new(c.clone()).unwrap().fit(&data).unwrap();
    c.cegb_penalty_feature_lazy = vec![1.0; cols];
    assert!(matches!(
        Trainer::new(c).unwrap().fit(&data),
        Err(lightgbm_rust::Error::Unsupported(_))
    ));
}

#[test]
fn execution_controls_defaults_aliases_conflicts_and_domains() {
    let c = Config::default();
    assert!(!c.deterministic && !c.force_col_wise && !c.force_row_wise);
    assert!(c.enable_bundle);
    assert_eq!(c.histogram_pool_size, -1.0);
    for alias in ["enable_bundle", "is_enable_bundle", "bundle"] {
        assert!(
            !Config::from_pairs([(alias, "false")])
                .unwrap()
                .enable_bundle
        );
    }
    assert_eq!(
        Config::from_pairs([("hist_pool_size", "0.25")])
            .unwrap()
            .histogram_pool_size,
        0.25
    );
    Config::from_pairs([
        ("num_threads", "4"),
        ("deterministic", "false"),
        ("force_row_wise", "true"),
    ])
    .unwrap();
    assert!(Config::from_pairs([("force_row_wise", "true"), ("force_col_wise", "true")]).is_err());
    for value in ["NaN", "inf", "-inf"] {
        assert!(Config::from_pairs([("histogram_pool_size", value)]).is_err());
    }
    assert!(Config::from_pairs([("num_threads", "257")]).is_err());
}
