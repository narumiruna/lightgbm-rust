use lightgbm_rust::cv::{CvOptions, Fold, FoldStrategy, cross_validate, make_folds};
use lightgbm_rust::{Config, Dataset, Error, Objective};
use std::cell::Cell;

fn config() -> Config {
    Config {
        num_iterations: 1,
        min_data_in_leaf: 1,
        min_data_in_bin: 1,
        feature_pre_filter: false,
        ..Config::default()
    }
}
fn data(nrows: usize, config: &Config) -> Dataset {
    Dataset::from_dense(&vec![0.0; nrows], nrows, 1, vec![0.0; nrows], config).unwrap()
}
fn standard(nfold: usize) -> FoldStrategy {
    FoldStrategy::Standard {
        nfold,
        shuffle: false,
        seed: 0,
    }
}
fn error<T: std::fmt::Debug>(result: lightgbm_rust::Result<T>, expected: &str) {
    match result.unwrap_err() {
        Error::InvalidInput(message) => assert!(message.contains(expected), "{message}"),
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn generated_index_amplification_is_rejected_before_fold_construction() {
    // The source is only 100000 x 1. The old fold-index output would contain
    // ten billion usize entries, about 74.5 GiB on a 64-bit target.
    let data = data(100_000, &config())
        .with_groups(vec![1; 100_000])
        .unwrap();
    for strategy in [
        standard(100_000),
        FoldStrategy::Stratified {
            nfold: 100_000,
            shuffle: true,
            seed: 7,
        },
        FoldStrategy::Group { nfold: 100_000 },
    ] {
        error(make_folds(&data, &strategy), "aggregate fold indexes");
    }
    // A small source also crosses the 16 Mi entry limit without allocating
    // the invalid output. Counts outside the row domain remain typed errors.
    let small = self::data(4097, &config());
    error(
        make_folds(&small, &standard(4097)),
        "aggregate fold indexes",
    );
    error(make_folds(&small, &standard(usize::MAX)), "fold count");
}

#[test]
fn valid_512_by_512_index_output_is_not_rejected() {
    let data = data(512, &config());
    let folds = make_folds(&data, &standard(512)).unwrap();
    assert_eq!(folds.len(), 512);
    assert_eq!(
        folds
            .iter()
            .map(|fold| fold.train_indices.len() + fold.valid_indices.len())
            .sum::<usize>(),
        262_144
    );
    for (index, fold) in folds.iter().enumerate() {
        assert_eq!(fold.valid_indices, [index]);
        assert_eq!(fold.train_indices.len(), 511);
    }
}

#[test]
fn normal_profile_shape_keeps_five_folds() {
    // The retained performance profiles use 8192 training rows and 16 features.
    let data =
        Dataset::from_dense(&vec![0.0; 8192 * 16], 8192, 16, vec![0.0; 8192], &config()).unwrap();
    let result = cross_validate(config(), &data, CvOptions::default()).unwrap();
    assert_eq!(result.folds.len(), 5);
    assert_eq!(result.models.len(), 5);
    assert_eq!(result.completed_rounds, 1);
}

#[test]
fn explicit_budget_precedes_clone_sort_and_index_validation() {
    let data = data(2, &config());
    // Caller-owned memory is small. Retained per-fold state alone exceeds the
    // documented byte estimate. The overlap error must not run first.
    let strategy = FoldStrategy::Explicit(vec![
        Fold {
            train_indices: vec![0],
            valid_indices: vec![0],
        };
        8193
    ]);
    error(make_folds(&data, &strategy), "storage estimate");
    let FoldStrategy::Explicit(folds) = &strategy else {
        unreachable!()
    };
    assert_eq!(folds[0].valid_indices, [0]);
}

#[test]
fn sparse_dimensions_and_empty_column_headers_are_charged_before_copy() {
    let config = config();
    let sparse = Dataset::from_csc(
        &vec![0; 1025],
        &[],
        &[],
        1024,
        1024,
        vec![0.0; 1024],
        &config,
    )
    .unwrap();
    assert_eq!(sparse.bin_storage_bytes(), 0);
    error(make_folds(&sparse, &standard(5)), "storage estimate");
    // The dense upper-bound entry cap is independent of stored sparse nnz.
    error(make_folds(&sparse, &standard(17)), "row-feature pairs");

    let wide = Dataset::from_csc(&vec![0; 257], &[], &[], 2, 256, vec![0.0; 2], &config).unwrap();
    let folds = FoldStrategy::Explicit(vec![
        Fold {
            train_indices: vec![0],
            valid_indices: vec![1],
        };
        1024
    ]);
    // Row-feature storage and fixed state alone fit; column/fold headers push
    // the combined estimate over 256 MiB even with zero bin-value bytes.
    error(make_folds(&wide, &folds), "storage estimate");
}

#[test]
fn aggregate_scores_fail_before_factories_or_session_state() {
    let config = Config {
        objective: Objective::Multiclass,
        num_class: 65_536,
        ..config()
    };
    let data = data(129, &config);
    let factory_called = Cell::new(false);
    let options = CvOptions {
        folds: standard(2),
        custom_metric_factories: vec![Box::new(|_| {
            factory_called.set(true);
            panic!("resource rejection must precede factories")
        })],
        ..CvOptions::default()
    };
    error(cross_validate(config, &data, options), "aggregate scores");
    assert!(!factory_called.get());
}

#[test]
fn scores_below_entry_limit_still_obey_combined_byte_budget() {
    let config = Config {
        objective: Objective::Multiclass,
        num_class: 10_000,
        ..config()
    };
    let data = data(100, &config);
    // Five million aggregate scores fit the entry limit, but the combined
    // score/derivative state estimate alone exceeds 256 MiB.
    error(
        cross_validate(config, &data, CvOptions::default()),
        "storage estimate",
    );
}

#[test]
fn initial_margin_metadata_is_bounded_before_subsets() {
    let data = data(2, &config())
        .with_initial_scores(vec![0.0; 4096], 2048)
        .unwrap();
    let folds = FoldStrategy::Explicit(vec![
        Fold {
            train_indices: vec![0],
            valid_indices: vec![1],
        };
        4097
    ]);
    error(make_folds(&data, &folds), "aggregate initial margins");
}

#[test]
fn retained_pool_product_fails_before_factories_and_allocation() {
    let config = Config {
        num_threads: 65,
        ..config()
    };
    let data = data(8, &config);
    let factory_called = Cell::new(false);
    let options = CvOptions {
        folds: standard(4),
        custom_metric_factories: vec![Box::new(|_| {
            factory_called.set(true);
            panic!("pool rejection must precede factories")
        })],
        ..CvOptions::default()
    };
    error(cross_validate(config, &data, options), "256 workers");
    assert!(!factory_called.get());
}

#[test]
fn five_normal_folds_keep_serial_parallel_outputs_and_source_unchanged() {
    let mut config = config();
    let data = Dataset::from_dense(
        &(0..100).map(f64::from).collect::<Vec<_>>(),
        100,
        1,
        (0..100).map(|row| (row % 7) as f32).collect(),
        &config,
    )
    .unwrap();
    let labels = data.labels().to_vec();
    let bins = (0..100)
        .map(|row| data.get_bin(row, 0).unwrap())
        .collect::<Vec<_>>();
    let mut previous = None;
    for threads in [0, 1, 2] {
        config.num_threads = threads;
        let result = cross_validate(config.clone(), &data, CvOptions::default()).unwrap();
        assert_eq!(result.folds.len(), 5);
        assert_eq!(result.history.len(), 1);
        let value = result.history[0].results[0].evaluation.value;
        if let Some(previous) = previous {
            assert_eq!(value, previous);
        }
        previous = Some(value);
        assert_eq!(data.labels(), labels);
        assert_eq!(
            (0..100)
                .map(|row| data.get_bin(row, 0).unwrap())
                .collect::<Vec<_>>(),
            bins
        );
    }
}

#[test]
fn explicit_validation_and_complete_group_rules_remain_intact() {
    let data = data(8, &config()).with_groups(vec![2, 3, 1, 2]).unwrap();
    let fold = |train_indices, valid_indices| {
        FoldStrategy::Explicit(vec![Fold {
            train_indices,
            valid_indices,
        }])
    };
    let valid = make_folds(&data, &fold(vec![5, 1, 0], vec![7, 6])).unwrap();
    assert_eq!(valid[0].train_indices, [0, 1, 5]);
    assert_eq!(valid[0].valid_indices, [6, 7]);
    for (train, valid) in [
        (vec![], vec![5]),
        (vec![0, 0], vec![5]),
        (vec![usize::MAX], vec![5]),
        (vec![0, 1], vec![0, 1]),
        (vec![0], vec![5]),
        (vec![3, 4], vec![5]),
        (vec![0, 1, 2, 3], vec![5]),
    ] {
        assert!(matches!(
            make_folds(&data, &fold(train, valid)),
            Err(Error::InvalidInput(_))
        ));
    }
    error(
        make_folds(&data, &FoldStrategy::Explicit(vec![])),
        "at least one fold",
    );
    assert!(make_folds(&data, &standard(3)).is_err());
    assert!(make_folds(&data, &FoldStrategy::Group { nfold: 3 }).is_ok());
}
