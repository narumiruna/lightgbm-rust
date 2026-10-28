use lightgbm_rust::{
    Boosting, Config, DataSampleStrategy, Dataset, MonotoneMethod, Objective, Trainer,
};

#[test]
fn pinned_sampling_and_learner_defaults() {
    let c = Config::default();
    assert_eq!(c.boosting, Boosting::Gbdt);
    assert_eq!(c.data_sample_strategy, DataSampleStrategy::Bagging);
    assert_eq!(
        (
            c.bagging_fraction,
            c.pos_bagging_fraction,
            c.neg_bagging_fraction
        ),
        (1.0, 1.0, 1.0)
    );
    assert_eq!(
        (c.bagging_freq, c.bagging_seed, c.bagging_by_query),
        (0, 3, false)
    );
    assert_eq!(
        (
            c.feature_fraction,
            c.feature_fraction_bynode,
            c.feature_fraction_seed
        ),
        (1.0, 1.0, 2)
    );
    assert_eq!((c.extra_trees, c.extra_seed), (false, 6));
    assert_eq!((c.top_rate, c.other_rate), (0.2, 0.1));
    assert_eq!(
        (c.drop_rate, c.max_drop, c.skip_drop, c.drop_seed),
        (0.1, 50, 0.5, 4)
    );
    assert!(!c.uniform_drop && !c.xgboost_dart_mode);
    assert_eq!(c.monotone_constraints_method, MonotoneMethod::Basic);
    assert!(
        c.monotone_constraints.is_empty()
            && c.interaction_constraints.is_empty()
            && c.feature_contri.is_empty()
    );
    assert!(c.forced_splits.is_none());
    assert_eq!(
        (
            c.monotone_penalty,
            c.path_smooth,
            c.cegb_tradeoff,
            c.cegb_penalty_split
        ),
        (0.0, 0.0, 1.0, 0.0)
    );
    assert!(c.cegb_penalty_feature_coupled.is_empty() && c.cegb_penalty_feature_lazy.is_empty());
    assert!(!c.use_quantized_grad && !c.quant_train_renew_leaf && c.stochastic_rounding);
    assert_eq!(c.num_grad_quant_bins, 4);
}

#[test]
fn aliases_precedence_seed_derivation_and_source_conflicts() {
    let c = Config::from_pairs([
        ("random_state", "42"),
        ("subsample", "0.3"),
        ("bagging_fraction", "0.7"),
        ("subsample_freq", "2"),
        ("colsample_bytree", "0.5"),
        ("colsample_bynode", "0.8"),
        ("extra_tree", "true"),
        ("rate_drop", "0.4"),
        ("mc", "1,0,-1"),
        ("mc_method", "advanced"),
    ])
    .unwrap();
    assert_eq!(
        (
            c.bagging_fraction,
            c.bagging_freq,
            c.feature_fraction,
            c.feature_fraction_bynode
        ),
        (0.7, 2, 0.5, 0.8)
    );
    assert_eq!(c.monotone_constraints_method, MonotoneMethod::Basic);
    assert_eq!(c.monotone_constraints, [1, 0, -1]);
    assert!(c.extra_trees);
    assert_eq!(c.drop_rate, 0.4);
    let mut rng = lightgbm_rust::random::Random::new(42);
    let actual = [
        c.data_random_seed,
        c.bagging_seed,
        c.drop_seed,
        c.feature_fraction_seed,
        c.objective_seed,
        c.extra_seed,
    ];
    let expected: Vec<u32> = (0..6)
        .map(|_| rng.next_short(0, 32767).unwrap() as u32)
        .collect();
    assert_eq!(actual.as_slice(), expected);
    let explicit =
        Config::from_pairs([("seed", "42"), ("bagging_seed", "9"), ("drop_seed", "12")]).unwrap();
    assert_eq!(
        (
            explicit.bagging_seed,
            explicit.drop_seed,
            explicit.extra_seed
        ),
        (9, 12, c.extra_seed)
    );
    let goss =
        Config::from_pairs([("boosting", "goss"), ("data_sample_strategy", "bagging")]).unwrap();
    assert_eq!(
        (goss.boosting, goss.data_sample_strategy),
        (Boosting::Gbdt, DataSampleStrategy::Goss)
    );
    assert_eq!(
        Config::from_pairs([("boosting", "random_forest"), ("feature_fraction", "0.8")])
            .unwrap()
            .boosting,
        Boosting::Rf
    );
    assert_eq!(
        Config::from_pairs([("path_smooth", "1"), ("min_data_in_leaf", "1")])
            .unwrap()
            .min_data_in_leaf,
        2
    );
}

#[test]
fn invalid_sampling_domains_and_combinations_fail() {
    for key in [
        "bagging_fraction",
        "pos_bagging_fraction",
        "neg_bagging_fraction",
        "feature_fraction",
        "feature_fraction_bynode",
    ] {
        for value in ["0", "-0.1", "1.1", "NaN", "inf"] {
            assert!(Config::from_pairs([(key, value)]).is_err(), "{key}={value}");
        }
    }
    for key in ["drop_rate", "skip_drop", "top_rate", "other_rate"] {
        for value in ["-0.1", "1.1", "NaN", "inf"] {
            assert!(Config::from_pairs([(key, value)]).is_err());
        }
    }
    for pairs in [
        vec![("data_sample_strategy", "goss"), ("top_rate", "0")],
        vec![("data_sample_strategy", "goss"), ("other_rate", "0")],
        vec![
            ("data_sample_strategy", "goss"),
            ("top_rate", "0.9"),
            ("other_rate", "0.2"),
        ],
        vec![
            ("data_sample_strategy", "goss"),
            ("bagging_fraction", "0.7"),
            ("bagging_freq", "1"),
        ],
        vec![("boosting", "rf")],
        vec![
            ("boosting", "rf"),
            ("objective", "custom"),
            ("feature_fraction", "0.5"),
        ],
        vec![("boosting", "unknown")],
        vec![("data_sample_strategy", "unknown")],
        vec![("monotone_constraints", "2")],
        vec![
            ("objective", "regression_l1"),
            ("monotone_constraints", "1"),
        ],
        vec![("objective", "quantile"), ("monotone_constraints", "0")],
        vec![("objective", "mape"), ("monotone_constraints", "-1")],
        vec![("monotone_constraints_method", "unknown")],
        vec![("interaction_constraints", "[[0],[-1]]")],
        vec![("use_quantized_grad", "true"), ("num_grad_quant_bins", "1")],
        vec![("linear_tree", "true"), ("zero_as_missing", "true")],
    ] {
        assert!(Config::from_pairs(pairs.clone()).is_err(), "{pairs:?}");
    }
}

fn dataset(c: &Config) -> Dataset {
    let x: Vec<f64> = (0..32).map(f64::from).collect();
    Dataset::from_dense(&x, 32, 1, (0..32).map(|i| (i % 7) as f32).collect(), c).unwrap()
}

#[test]
fn sampling_state_is_fit_owned_and_does_not_modify_dataset() {
    for strategy in [DataSampleStrategy::Bagging, DataSampleStrategy::Goss] {
        let c = Config {
            min_data_in_leaf: 1,
            min_data_in_bin: 1,
            num_iterations: 9,
            learning_rate: 0.25,
            feature_pre_filter: false,
            data_sample_strategy: strategy,
            bagging_fraction: if strategy == DataSampleStrategy::Bagging {
                0.7
            } else {
                1.0
            },
            bagging_freq: 2,
            ..Config::default()
        };
        let data = dataset(&c);
        let before: Vec<u32> = (0..data.nrows())
            .map(|r| data.get_bin(r, 0).unwrap())
            .collect();
        let labels = data.labels().to_vec();
        let trainer = Trainer::new(c).unwrap();
        let a = trainer.fit(&data).unwrap();
        let b = trainer.fit(&data).unwrap();
        assert_eq!(a.model.to_text().unwrap(), b.model.to_text().unwrap());
        assert_eq!(a.training_scores, b.training_scores);
        assert_eq!(data.labels(), labels);
        assert_eq!(
            (0..data.nrows())
                .map(|r| data.get_bin(r, 0).unwrap())
                .collect::<Vec<_>>(),
            before
        );
    }
}

#[test]
fn query_metadata_and_rf_initial_margin_errors_are_explicit() {
    let c = Config {
        min_data_in_leaf: 1,
        min_data_in_bin: 1,
        feature_pre_filter: false,
        bagging_by_query: true,
        bagging_fraction: 0.7,
        bagging_freq: 1,
        ..Config::default()
    };
    let data = dataset(&c);
    assert!(Trainer::new(c.clone()).unwrap().fit(&data).is_err());
    let c = Config {
        boosting: Boosting::Rf,
        bagging_by_query: false,
        ..c
    };
    let data = data.with_initial_scores(vec![1.0; 32], 1).unwrap();
    assert!(Trainer::new(c).unwrap().fit(&data).is_err());
    let custom = Config {
        objective: Objective::Custom,
        ..Config::default()
    };
    assert!(Trainer::new(custom).unwrap().fit(&data).is_err());
}
