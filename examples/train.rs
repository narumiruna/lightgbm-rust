//! Run: cargo run --example train -- target/example-models
use lightgbm_rust::{Config, Dataset, Model, Objective, Result, Trainer};

fn main() -> Result<()> {
    let directory = std::env::args().nth(1).ok_or_else(|| {
        lightgbm_rust::Error::InvalidInput("usage: train NEW_OUTPUT_DIRECTORY".into())
    })?;
    std::fs::create_dir_all(&directory)?;
    let features = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    for (name, objective, labels) in [
        (
            "regression",
            Objective::Regression,
            vec![-2.0, -1.0, -1.0, 0.0, 1.0, 2.0, 2.0, 3.0],
        ),
        (
            "binary",
            Objective::Binary,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        ),
    ] {
        let config = Config {
            objective,
            num_iterations: 10,
            num_leaves: 3,
            min_data_in_leaf: 1,
            min_data_in_bin: 1,
            ..Config::default()
        };
        let dataset = Dataset::from_dense(&features, 8, 1, labels, &config)?;
        let trained = Trainer::new(config)?.fit(&dataset)?;
        let path = std::path::Path::new(&directory).join(format!("{name}.txt"));
        trained.model.save(&path)?;
        let loaded = Model::load(&path)?;
        assert_eq!(
            trained.model.predict_dense(&features, 8, 1)?,
            loaded.predict_dense(&features, 8, 1)?
        );
        println!(
            "{name}: {} iterations, prediction at 2 = {:?}",
            loaded.num_iterations(),
            loaded.predict_row(&[2.0])?
        );
    }
    Ok(())
}
