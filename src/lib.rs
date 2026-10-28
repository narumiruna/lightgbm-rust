//! Pure Rust LightGBM algorithms and model prediction.
//!
//! See `docs/compatibility.md` for supported capabilities and open conformance gates.
//!
//! ```
//! use lightgbm_rust::{Config, Dataset, Trainer};
//! let config = Config {
//!     min_data_in_leaf: 1,
//!     min_data_in_bin: 1,
//!     num_iterations: 3,
//!     ..Config::default()
//! };
//! let data = Dataset::from_dense(&[0.0, 1.0, 2.0, 3.0], 4, 1,
//!                                vec![0.0, 0.0, 1.0, 1.0], &config)?;
//! let trained = Trainer::new(config)?.fit(&data)?;
//! assert!(trained.model.predict_row(&[2.5])?[0].is_finite());
//! # Ok::<(), lightgbm_rust::Error>(())
//! ```

mod boosting;
pub mod config;
pub mod cv;
pub mod data;
pub mod error;
mod histogram;
pub mod io;
mod learner;
mod linear;
pub mod metric;
pub mod model;
pub mod objective;
pub mod parallel;
pub mod random;
mod sampling;
pub mod trainer;
pub mod workflow;

pub use config::{Boosting, Config, DataSampleStrategy, MonotoneMethod, Objective};
pub use data::{BinMapper, Dataset, MissingType};
pub use error::{Error, Result};
pub use model::{ImportanceType, IterationRange, LinearLeaves, Model, PredictionEarlyStop, Tree};
pub use parallel::ParallelPredictor;
pub use trainer::{Evaluation, IterationEvaluation, TrainResult, Trainer};
