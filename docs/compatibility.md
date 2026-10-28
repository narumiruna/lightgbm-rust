# Compatibility contract

## Reference and status

The reference is LightGBM commit `5d84f3f222b41409a6983d2ac90513ad9702fa9e` (`4.7.0.99`, text model format `v4`). This is a development snapshot, not a stable upstream release.

The CPU implementation has saved native conformance evidence. The [active plan](plans/2026-10-28_pure-rust-rewrite-plan.md) is still open. Passing the cases below does not establish complete upstream parity. Final performance, package, and platform checks remain release gates.

## Feature matrix

| Family | Implemented behavior and evidence |
| --- | --- |
| Dense, CSR, CSC, missing values | Checked builders, compact storage, sampled bins, shared validation maps; `data_oracle`, `data_bins`, `data_validation`, `data_random` tests. |
| Categorical data and splits | Integer codes, one-hot and sorted partitions, smoothing and penalties; `categorical_training`, `training_oracle`, `learner_controls`. |
| Metadata | Names, weights, class-major initial margins, groups, positions, and group-safe subsets; `data_metadata`, `data_feature_info`. |
| Bins and tree growth | Forced/per-feature bins, histograms, subtraction, regularization, missing directions, ties, best-first growth; `data_forced_bins`, histogram unit tests, `training_oracle`. |
| Objectives | All 16 pinned CPU factory entries, weighted derivatives and initialization, nonsmooth leaf renewal; objective unit tests and `objective_extended`/`objective_oracle`. |
| Multiclass and ranking | Softmax, OVA, absent classes, LambdaRank, XE-NDCG, query weights, positions; native objective state and training cases. |
| Metrics | All 24 pinned CPU metric factories, directions, defaults, disabled metrics, ties and degenerate targets; `metric_oracle`. |
| Sampling and ensembles | Row/class/query bagging, feature sampling, GOSS, DART, RF; internal sampling fixtures, `boosting_oracle`, `boosting_query`, `boosting_regressions`. |
| Learner controls | All monotonicity methods, interactions, forced splits, extra trees, smoothing, feature penalties and CEGB; `learner_controls`, `constraints_invariants`. |
| Linear leaves and quantization | Pure-Rust complete-pivot solver, missing-row rules, singular systems, stochastic rounding and renewal; `linear_training`, quantizer and learner tests. |
| Prediction and persistence | Checked text import/export, dense and sparse traversal, transforms, class layout and ranges; model tests and native round trips. |
| Explanation and approximate prediction | Count-weighted TreeSHAP, split/gain importance, margin-based stopping; `model_shap`, `model_prediction_options`. |
| Model operations | Immutable leaf changes, complete-iteration prefixes, rollback to zero, nonlinear/linear refit; `model_operations`, `refit_domain`. |
| Rust workflows | Typed callbacks, schedules, custom losses/metrics, best scores, Python stopping rules, warm starts and CV; workflow, custom, continuation and CV tests. |
| CLI | Train, predict and refit over checked CSV/TSV/LibSVM adapters; `cli`, `io`. |
| CPU execution | Bounded Rayon pools, row/column histograms, cache eviction/replay, lossless bundles and native feature order; parallel and bundling tests. |
| GPU/CUDA/ROCm and distributed learners | Outside this plan; unsupported. |
| Python/R/C ABI, sklearn/Dask/pandas/Arrow/Polars adapters and plotting | Outside this plan; unsupported. |
| Pickle, upstream binary datasets and C++ code generation | Outside this plan; unsupported. |

Test names identify Cargo integration targets or module tests. The reference manifest and [conformance specification](conformance.md) identify the generated inputs, native identity and comparison rules. No unsupported option may silently select another algorithm.

## Parameter dispositions

| Upstream family | Rust contract |
| --- | --- |
| Task, objective, boosting, iterations, learning rate, leaf/depth limits | Typed `Config`; CLI commands select workflows. |
| Defaults, aliases, seed derivation and conflicts | `Config::from_pairs`; canonical names win, then the shortest alias, then lexical order. Duplicate names fail. Typed fields represent canonical values. |
| Binning, sparse data, missing values, categorical indexes, filtering and bundling | Dataset parameters and checked metadata. Named-column and automatic Python adapters are not implemented. |
| Regularization, sampling, constraints and costs | Typed learner fields with source-backed validation. Forced-split JSON is a Rust value, not an implicit file read. |
| Objective parameters and metrics | Typed fields and the CPU registry; unknown names fail, with no regression fallback. |
| Hooks, schedules, evaluation, stopping and CV | `workflow` and `cv` interfaces, not Python-compatible signatures. |
| Prediction options and persistence | Checked `Model` methods, `PredictionEarlyStop`, `ImportanceType` and immutable operations. Shape checks cannot be disabled. |
| Threads, histogram modes and cache size | `num_threads`, row/column mode controls and `histogram_pool_size`; bounded CPU execution. |
| Network, distributed, GPU and adapter-specific options | Unsupported unless a documented Rust equivalent exists. |

The parser applies source conflict adjustments. In particular, GOSS disables `bagging_by_query`, as in `config.cpp`. A directly constructed `Config` must already satisfy canonical constraints. Neutral values do not make an otherwise unimplemented parameter supported.

## Data and precision

- Dense input is row-major. Declared dimensions must agree with checked length multiplication.
- CSR/CSC offsets and indexes are checked before access. Duplicate coordinates fail; they are not summed. Dataset builders accept unsorted sparse indexes and normalize them.
- Labels, weights, gradients and Hessians use reference-default `f32` boundaries. Scores and histogram sums use `f64` where the reference does.
- Initial margins are class-major: `class * rows + row`. They remain external to newly fitted model trees. Supply separate margins for validation data.
- Validation data share training bin maps. They do not learn thresholds. Subsets preserve metadata and cannot select partial or interleaved ranking groups.
- Public bin, mapper and bin-count access uses checked `get_bin`, `get_mapper` and `get_bin_counts` methods. Invalid indexes return errors.
- Unweighted XE-NDCG preserves empty logical queries, their RNG indexes, and NDCG/MAP denominators. Empty LambdaRank, weighted empty queries, and ambiguous nonidentity row subsets return errors instead of reproducing native undefined behavior. Identity subsets preserve every query. Query bagging rejects query counts above row counts because the native buffer layout is not safe for that shape. See `ranking_empty_queries`.
- Raw numerical values are retained as `f32` only for linear trees. Refit and continuation must not infer original raw routing from new, lossy bin maps.
- Caller-owned class labels and categorical codebooks must remain consistent across training and prediction. There is no automatic Python category alignment.
- Categorical values truncate toward zero. For example, `-0.5` becomes code zero; integer negative codes and NaN route right in model traversal. Dataset categorical casts outside the signed 32-bit domain fail. Prediction treats such values as unseen instead of using undefined native casts.

### Dense and sparse prediction

Dense native prediction omits values in the inclusive near-zero interval defined by `1e-35_f32` promoted to `f64`. `predict_row`, `predict_sparse_row` and `predict_leaf_sparse_row` apply this normalization. The latter two are dense-equivalent sparse adapters.

`predict_csr_row` and its raw, leaf, contribution and stopping variants preserve explicit tiny values. Their sparse indexes must be strictly increasing. Absent values are zero. This distinction produces real differences at some native split boundaries.

Prediction requires exactly the model feature count. Regression, binary and ranking return one scalar. Multiclass returns one value per class. Leaf output contains one index per selected tree. A multiclass iteration contains one complete class-ordered set of trees.

### AUC-mu ordering

The native epsilon comparator is not a total order. Rust's standard sort can panic on finite inputs with this comparator. A checked LLVM-derived sort preserves the pinned single-thread Apple clang/libc++ ordering, with 68 native edge cases in `auc_mu_safety`. This is not a claim about every C++ standard library or toolchain. Nonfinite or overflowing distances return errors.

### Model numbers and limits

- Text input is limited to 64 MiB and 1,000,000 lines, features, trees and total leaves. Class count is limited to 65,536. Serialized tree sizes refer to exact bytes; retain the producer's line endings. Fixture checkout disables newline conversion.
- Batch and class-major score outputs are limited to 16,777,216 elements before allocation. Sparse dimensions do not permit unchecked dense allocation.
- Numerical thresholds can be infinite. Split gains use native `f32` storage; finite labels can produce an infinite diagnostic gain with finite predictions. Text and gain importance preserve it. Structured inspection uses native clamps: `1e300` for double thresholds and `1e38_f32` for gains.
- TreeSHAP requires consistent positive sample counts on split trees. It rejects linear models, as the native implementation does. It checks a depth limit of 256, 16 Mi output entries and 64 Mi path-work units before traversal. It does not substitute Hessians for missing counts.
- Linear solvers check coefficient, matrix and work limits before allocation. Invalid or overflowing numerical systems return errors, not nonfinite models.

## Workflows and model operations

`Trainer::fit` and `fit_with_validation` use the core stop-on-no-split path. `fit_with_options` uses Python-style round attempts and ordered before/after callbacks, including no-split rounds. Equal callback orders retain insertion order. Evaluation recording occurs at order 20; built-in stopping occurs at order 30.

Early stopping excludes training metrics, follows metric direction and strict `min_delta` comparisons, supports `first_metric_only`, and is disabled for DART. `TrainResult.best_iteration` is one-based; zero means no selection. The returned live model retains patience rounds. Select `model.iteration_range(0..best_iteration)` explicitly when using the best prefix.

Custom derivatives and custom multiclass predictions are class-major. Derivative lengths, finite values and nonnegative Hessians are checked before learner access. Built-in objectives pass transformed predictions to custom metrics; custom objectives pass raw margins. Callback and custom-function errors return no partially published model.

Learning-rate resets preserve learner feature-sampling RNG state and perform the native DART/GOSS resets. Native fixtures cover schedules with tree feature sampling and saved DART prefixes; `workflow_extended` checks scores and structure.

CV provides explicit, standard, stratified and group-safe folds, population standard deviations and a common stopping iteration. NumPy legacy permutation and pinned stratification cases have native evidence. General equal-size group ties use documented descending group index and first least-loaded fold ordering; NumPy's unstable sort is not a portable tie contract.

Stored-model warm starts restart trainer random state. They are not exact training checkpoints. Supply the original model's raw margins for training and every validation set. RF rejects this initial-margin continuation, as the native implementation does.

Refit takes the original model's raw leaf indexes, not new binned routing. It preserves structure and diagnostic metadata and uses each old tree's shrinkage. RF, DART, GBDT and GOSS have saved native refit cases. Imported-model `path_smooth` refit is unsupported: the pinned native loader does not restore the parent state that this path reads. No undefined native memory access is reproduced.

Model leaf changes and rollback return validated copies. Prefix truncation keeps complete class rounds and can produce an empty ensemble. It does not restore historical DART normalization, trainer buffers or RNG state. Empty raw predictions are zero; transformed RF prediction on an empty range returns an error instead of native division by zero.

## Resources, failures and release restrictions

CV checks aggregate storage before fold construction or explicit-fold cloning. It limits indexes, row-feature pairs, class-major scores and initial margins to 16 Mi entries each. Its conservative combined storage estimate is at most 256 MiB; retained fold pool workers total at most 256. This estimate is not a peak-RSS guarantee. See `cv_resources`.

CPU pools contain at most 256 workers. Zero and one select serial execution. Nested calls from a Rayon worker use the serial path. Per-feature accumulation and ordered tie reduction remain deterministic for a documented build and thread configuration. Cross-thread tests do not imply bit-identical results across hardware and compilers.

Histogram cache retention is bounded independently from active working buffers. Eviction replays the same row order without new sampling draws. Bundling preserves original feature identity and uses lossless packing or conflict fallback. `bin_storage_bytes()` counts allocated bin data and sparse indexes, not allocator overhead, metadata, raw values or process RSS.

Model readers reject invalid indexes, cycles, shared children, truncated arrays, incompatible objective metadata, unknown capabilities and unsupported versions. Conversion does not overwrite input data. File outputs use temporary files and atomic, no-clobber publication.

Product builds and ordinary tests do not execute Python, C++, CMake, OpenMP, Eigen or native ML code. Oracle tooling is isolated and excluded from the Cargo package. Preserve upstream MIT notices and the LLVM sorting license in `licenses/LLVM.txt`.

All applicable plan gates remain required. Do not infer release readiness from test counts or similar quality alone. Native platform support needs native execution evidence. Publication and deployment replacement require separate authorization.
