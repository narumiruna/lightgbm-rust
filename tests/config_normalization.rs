use lightgbm_rust::{Config, DataSampleStrategy};

#[test]
fn parsed_goss_disables_query_bagging_like_the_pinned_config() {
    for key in ["data_sample_strategy", "boosting"] {
        let config = Config::from_pairs([(key, "goss"), ("bagging_by_query", "true")]).unwrap();
        assert_eq!(config.data_sample_strategy, DataSampleStrategy::Goss);
        assert!(!config.bagging_by_query);
    }
    // Struct fields describe a canonical configuration, without parser rewrites.
    assert!(
        Config {
            data_sample_strategy: DataSampleStrategy::Goss,
            bagging_by_query: true,
            ..Config::default()
        }
        .validate()
        .is_err()
    );
}
