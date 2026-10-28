# lightgbm-rust

A pure-Rust library and CLI for LightGBM CPU training and prediction.

**The implementation plan is not complete.** Release checks and matched performance evidence remain open. This is an experimental package, not a claim of complete upstream compatibility.

## CPU features

- GBDT, DART, RF, GOSS, and row, class, query, and feature sampling.
- All 16 pinned CPU objective factories and 24 metric factories, plus Rust custom objectives and metrics.
- Numerical and categorical splits, monotonicity and interaction constraints, forced splits, smoothing, feature costs, linear leaves, and quantized gradients.
- Dense, CSR, and CSC datasets with labels, weights, initial margins, query groups, positions, subsets, and shared validation bins.
- LightGBM `v4` text models, raw and transformed predictions, leaf indexes, iteration ranges, TreeSHAP, importance, and prediction stopping.
- Ordered callbacks, schedules, early stopping, cross-validation, model warm starts, refit, and immutable model operations.
- CSV, TSV, and LibSVM train, predict, and refit commands.
- Bounded CPU pools, row/column histograms, lossless feature bundles, and a bounded histogram cache.

See the [compatibility contract](docs/compatibility.md) for checked limits, caller responsibilities, unsupported combinations, and evidence. The [active plan](docs/plans/2026-10-28_pure-rust-rewrite-plan.md) tracks remaining gates.

## Rust API

Rust 1.92.0 is pinned. Product builds and ordinary tests do not require Python, C++, CMake, OpenMP, Eigen, or `lib_lightgbm`.

```rust
use lightgbm_rust::{Config, Dataset, Trainer};

let config = Config {
    min_data_in_leaf: 1,
    min_data_in_bin: 1,
    num_iterations: 3,
    ..Config::default()
};
let dataset = Dataset::from_dense(
    &[0.0, 1.0, 2.0, 3.0],
    4,
    1,
    vec![0.0, 0.0, 1.0, 1.0],
    &config,
)?;
let trained = Trainer::new(config)?.fit(&dataset)?;
let predictions = trained.model.predict_row(&[2.5])?;
# Ok::<(), lightgbm_rust::Error>(())
```

`Trainer::fit_with_options` provides Python-style round attempts and callbacks. `workflow::FitOptions` accepts custom objectives and metrics. `cv::cross_validate` supplies fold generation and population standard deviations. These are Rust interfaces, not Python-compatible signatures.

Run the regression and binary save/load example with a new output directory:

```sh
cargo run --locked --example train -- target/example-models
```

## CLI

CSV/TSV inputs have no header. Labeled input contains a label followed by feature values. Prediction input contains features only. LibSVM always contains a label; prediction validates and discards it. LibSVM feature indexes start at one.

```sh
cargo run --locked -- train --input train.csv --format csv --rows 100 --features 4 --model model.txt objective=binary min_data_in_leaf=5
cargo run --locked -- predict --input predict.csv --format csv --rows 20 --features 4 --model model.txt --output predictions.tsv
cargo run --locked -- refit --input new-labels.csv --format csv --rows 100 --features 4 --model model.txt --output refitted.txt --decay-rate 0.9 objective=binary
```

Refit parameters must match the model objective, class layout, feature types, and linear-tree mode. Supply the required training parameters explicitly. Use the Rust API for weights and query metadata. Labels and category codebooks remain the caller's responsibility.

Use `--help` for limits and input rules. File output is atomic and never replaces an existing path. Unsupported options return errors.

## Verification

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --all-targets
cargo test --locked --doc
cargo test --locked --offline --all-targets
```

Saved text fixtures pin LightGBM commit `5d84f3f222b41409a6983d2ac90513ad9702fa9e`. Tests compare predictions, tree structure, bins, random sequences, derivatives, histograms, split outputs, and workflow events. Malformed-input and mutation tests check failure behavior.

See the [conformance specification](docs/conformance.md) and [isolated oracle workflow](tools/oracle/README.md). Oracle tools are development-only and excluded from the Cargo package. Performance budgets remain required even when correctness tests pass. Package publication and deployment replacement need separate authorization.

## License

Original project code and translated LightGBM code use MIT. The checked LLVM-derived sorting routine uses Apache-2.0 WITH LLVM-exception. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [licenses/LLVM.txt](licenses/LLVM.txt).
