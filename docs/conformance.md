# CPU conformance contract

## Baseline and limits

The reference is LightGBM `5d84f3f222b41409a6983d2ac90513ad9702fa9e`, version `4.7.0.99` (development snapshot). `reference-manifest.json` records the source, compiler, build flags, actual loaded image, and artifact SHA-256 values. The schema requires these identities.

The current reference is a CPU-only Release build with `USE_OPENMP=OFF`, `USE_GPU=OFF`, `USE_CUDA=OFF`, `USE_ROCM=OFF`, and `USE_MPI=OFF`. It uses Apple clang, not a wheel. The effective configuration is serial, `num_threads=1`, `deterministic=true`, `force_col_wise=true`. There is no evidence of OpenMP or parallel speed equivalence. An OpenMP or cross-thread baseline needs separate recorded evidence; it must not replace this baseline without review.

This contract fixes the cases and bounds before Rust optimization. A failed bound stays open. Do not increase a bound to conceal an implementation defect.

## Numerical and structural gates

- Imported model scores: `abs(actual - expected) <= 1e-12 + 1e-10 * abs(expected)`.
  - Apply this bound separately to raw and transformed outputs.
  - Match row count, class count, tree count, leaf indexes, category membership, and iteration selection exactly.
- Default `f32` gradients and Hessians: `1e-6 + 1e-5 * abs(expected)`.
- Objective initial scores, leaf renewal, and metric results: `1e-12 + 1e-10 * abs(expected)`.
  - Compare native exceptional values by their explicit `Infinity`, `-Infinity`, or `NaN` string, not a finite approximation.
- Double-precision histogram sums, gains, and leaf outputs: `1e-12 + 1e-10 * abs(expected)`.
- Bin count, missing type, default bin, most-frequent bin, category codes, selected threshold index, row counts, and RNG results: exact equality.
  - Numeric bin thresholds: exact IEEE-754 equality for this serial baseline. In particular, retain the reference's adjacent-double and zero-separation behavior.
- Tiny training cases: exact split feature, decision type, split topology, missing direction, threshold index, and leaf index.
  - Leaf values and numeric split thresholds use the double-precision bound above.
  - Compare tree structure, not only final model quality.
- Each deterministic case must give the same result on two independent runs with this build and thread setting.
  - Test a different compiler, CPU, or thread count separately. Do not infer equivalence.

Labels, weights, gradients, and Hessians are `f32`. Raw input features and prediction scores are `f64`. L2 weighted gradients round the residual to `f32` before multiplying by the `f32` weight. Binary derivatives compute in `f64` and then round the weighted result to `f32`. Histogram accumulation converts `f32` derivatives to `f64`. The compiled primitive driver checks the reference type sizes.

## Saved cases

`tests/fixtures/prediction.json` and `prediction.schema.json` define row-major raw scores, transformed scores, and per-tree leaf indexes. `model_file` is relative to `tests/fixtures/`. `null` in an input row means IEEE NaN. JSON numbers are finite; model files remain unmodified upstream `v4` text.

The first eight required cases are `numeric_l2`, `binary`, `numeric_weighted`, `missing_nan`, `missing_zero`, `categorical`, `multiclass`, and `single_leaf`. Additional imported-model cases cover square-root regression, non-unit sigmoid, one-vs-all multiclass, Poisson, Gamma, Tweedie, cross entropy, cross-entropy lambda, random-forest averaging, and disabled missing-value handling. Numerical probes include each serialized threshold and its two adjacent `f64` values, and the `f32` zero threshold and its adjacent `f64` values.

`tests/fixtures/prediction_ranges.json` uses the same schema for multiclass/RF ranges with `params.start_iteration=1` and `params.num_iteration=1` or `2`. It also retains a three-class model trained without class 2. Range leaves contain only the selected trees. Base prediction coverage remains the original 18 cases.

Training records contain the original rows, `f32` labels, optional `f32` weights, round count, and the serialized reference tree structure. Extended ranking records also contain positive query lengths in `training.groups` and integer codes in `training.positions`. Extra objective fixtures do not imply Rust training support.

`prediction_regressions.json` retains the native missing-only and NaN-direction tie training cases. `rust-trained-model-roundtrip.json` records the main task's native reload of actual Rust-trained L2 and binary models. This differs from importing and exporting a reference-trained model; neither check waives training structure or quality gates.

The new families are separate from all base evidence:

| File | Native evidence |
| --- | --- |
| `extended_objectives.json` | All 16 CPU objective factories; weighted/unweighted initialization and derivatives; parameter variants; L1/quantile/MAPE renewal; multiclass full and absent classes; ranking groups, weights, ties, positions, sequential RNG/state updates; binary single-class cases |
| `metrics.json` | All 24 CPU metric factories; weighted/unweighted results and direction; ranking cutoffs and zero relevance; AUC/AP ties and single class; AUC-mu ties and custom costs; multiclass softmax/OVA layouts; R2 constant targets; parameter variants |
| `dataset_metadata.json` | Native label/weight/initial-score storage; group boundaries and query mean weights; position compaction; actual shared training/validation Dataset bin storage with forced per-feature cutpoints and bin counts |
| `forced_bins.json` | Native forced upper-bound selection, adjacent probes, out-of-sample cuts, per-feature max bins, NaN/zero/disabled missing behavior |
| `prediction_extended.json` | Tiny trained models for all CPU objectives; categorical one-hot and sorted partitions with/without weights and missing codes; multiclass full and absent classes |
| `prediction_sparse.json` | Actual native SciPy CSR raw/transformed/leaf predictions for all 18 base cases, including near-zero probes |
| `prediction_linear.json` | Four native linear-leaf models: independent/rank-deficient columns, linear regularization 0/0.5, first tree, and NaN fallback |
| `data_feature_info.json` | 14 exact native feature-info strings: numeric formatting, adjacent values, implicit zeros, infinities, NaN, and categories; original helper retained |
| `explain.json` | 46 actual dense/CSR TreeSHAP and split/gain importance cases from 18 base models plus five ranges/absent-class cases; four native linear-contribution errors |
| `prediction_boosting.json` | 13 tiny weighted training cases for row/class/query/feature/node sampling, GOSS scalar/multiclass warm-up, three DART modes, and RF regression/binary averaging |
| `prediction_constraints.json` | 18 tiny weighted training cases for monotone methods/penalty, interaction, extra trees, forced splits, path smoothing, feature penalties, CEGB, combinations, and gradient quantization/renewal |
| `prediction_constraints_extra.json` | Ten additional native deep-monotone, categorical smoothing/extra-tree/quantization/renewal, and sampled-interaction cases; original case values and model bytes retained; safe paths under `models/constraints_extra/` |
| `sampling.json` | 16 actual internal row/class/query/feature/GOSS/DART/quantization cases; selected indexes/masks, rescaled derivatives, dropout state, quantized bytes/scales |
| `prediction_linear_training.json` | Three additional linear-leaf cases with NaN in training rows, weights including zero, and both controls together |
| `bundling.json` | 22 actual native enabled/disabled bundle cases; exclusive/sparse/conflicting features, NaN/zero handling, frequent nonzero bins, filtering, categorical values, reordered features, and extra trees; native groups/maps/stored bins and tiny trained models |

Objective and metric scores are flat class-major `f64`: index `class * rows + row`. Derivatives are actual native `f32`. Objective `initial_scores` contains one `BoostFromScore` result per class; metadata `initial_scores` instead contains one score per row per class. Scalar objectives use one class. `steps` contains the second call on the same objective instance, after the top-level derivative output. Do not initialize a new objective for each step. Renewal rows are original zero-based row indexes and the residual is `label - score`; `newton_output` is the input supplied to native renewal. Query groups are positive lengths, not prefix sums. Positions are arbitrary signed `i32` codes; native metadata compacts them in first-seen order and exposes their string IDs.

The six added saturated metric cases use wrong-class raw scores of `+1000` and `-1000`, zero-score ties, and weighted/unweighted binary, multiclass softmax, and multiclass OVA. Weights include zero but have a positive total. These records add actual `source_class_need_train` and serialized `model_objective` strings. The binary and multiclass metric clamp calls `std::log(kEpsilon)` with `kEpsilon` of type `float`: the logarithm rounds to `f32` before promotion to `f64`. Do not replace this with a double-precision logarithm. Earlier 68 metric cases and all earlier primitive/model outputs are unchanged.

TreeSHAP contributions are row-major flat class blocks of `features + 1` values. The final value in each block is the bias. Each block must sum to its native raw class score within the imported-score bound. Split importance is exact; gain importance uses the double bound. For ranges, the native model text slice is reloaded before native importance is called. `Tree` parses serialized `split_gain` with `StringToArrayFast<float>` (`src/io/tree.cpp`), then importance accumulates these loaded `f32` values in `f64`. Do not sum the decimal text tokens directly as doubles. `gain_precision` retains an actual loaded example. Linear TreeSHAP rejection is separate from valid case outputs.

New control training records use the existing prediction schema. Optional `training.steps` contains zero-based `iteration`, native row-major `raw`, and native `tree_shrinkage` after each round. No sampler output comes from NumPy. In `sampling.json`, input gradient/Hessian numbers are converted to native `score_t` (`f32`) before kernel calls. Row steps retain actual input/output derivatives and selected original row indexes; `all_rows=true` denotes GOSS warm-up, for which native code does not populate selected-index storage. Query selections preserve whole groups. Feature steps contain the actual tree mask and two sequential root-node masks per tree. Masks use native inner-feature order. `eligible_features` contains original feature indexes; `inner_feature_indices` maps each original index to its native inner index, or `-1` when filtered. The constant-column case tests a non-identity mapping. They test this declared primitive call order, not all nodes of a trained tree. DART steps expose actual drop indexes, tree weights, and `sum_weight`; in pinned `xgboost_dart_mode`, `sum_weight` need not equal the sum of tree weights. Preserve the native state rather than enforcing that untrue invariant. Quantization stores signed bytes in `[hessian, gradient]` order per row, plus native scales, across four calls to one instance. Internal selection evidence is separate from full tiny-model training evidence.

The forced-bin cases retain the original primitive schema plus `forced_upper_bounds`. Shared Dataset cases put input rows, validation rows, per-feature forced bounds, per-feature maximum bins, and zero-missing policy in `binning`; `expected.binning` contains actual row-major stored bins and feature upper bounds. Validation uses `Dataset::CreateValid`, not a separately fitted mapper. Far outside validation values test that training cutpoints are retained.

CSR records add `sparse` with `indptr`, `indices`, `data`, and `shape`. A null data entry is NaN. Zero entries omitted by SciPy remain implicit zero, while tiny nonzero entries remain explicit. Native CSR inference can differ from native dense inference near the zero threshold. Test each against its own saved outputs; do not claim that dense traversal of a reconstructed matrix is native CSR parity.

The compiled driver produces these independent primitive families:

| File | Assertions |
| --- | --- |
| `rng.json` | Independent short/int/float streams; one interleaved stream; dense and Floyd sampling branches; seeds 0, 1, 42 |
| `objectives.json` | L2/binary initialization and derivatives, with and without weights |
| `bins.json` | Numeric zero separation; missing NaN/zero/disabled; constant input; max bins 2/3/4; minimum bin count; adjacent doubles; categorical ties, negative/NaN codes, rare-category retention |
| `histograms.json` | Actual dense-bin histogram sums, numerical best split, gain and leaf outputs, with L2 regularization 0/1 |
| `parameters.json` | All 141 pinned canonical names, 167 aliases, types, compiled declared defaults, and serialized default configuration; inventory is not a support claim |

For bin fixtures, `sample_values` excludes explicit zeros and `total_sample_count` includes implicit zeros. `probe_values` aligns with `value_bins`. `upper_bounds` records `BinToValue`; for categories it contains category codes, including the dummy `-1`. The final numerical missing-bin sentinel can be the number `2`: upstream passes the unscoped `MissingType::NaN` enum to `push_back`, not an IEEE NaN. Preserve this evidence. A string `Infinity` represents an infinite upper bound without invalid JSON.

Category inputs are raw numeric codes, not Python category names. Codes are truncated toward zero; negative integer codes and NaN use missing behavior. Callers own stable codebook alignment. These fixtures do not prove pandas codebook behavior.

## Python workflows and model operations

Three additional families have separate schemas. They use tiny fixed integer inputs, original `BASE` fields, all training seeds zero, CPU serial execution, one thread, and the same numerical bounds as earlier evidence. Each generator enforces the 512 MiB process and 300-second case limits. These are saved reference observations, not a claim that the Rust workflow implementation passes them.

| File | Actual reference evidence |
| --- | --- |
| `workflow.json` | 15 original Python engine cases; ordered before/after callbacks; learning-rate reset schedule; native metric history and directions; multiple validation sets; first-metric-only and scalar/vector minimum delta; training exclusion; training-only/no-validation behavior; DART-disabled stopping; final-round stopping; no-split callbacks; custom L2/binary derivatives and metrics; safe invalid-length rejection |
| `cv.json` | Six original Python CV cases; explicit, standard shuffled/unshuffled, stratified, and group-safe folds; train/valid metrics per fold; aggregate means/population standard deviations; common best iteration and truncated history; saved CVBooster and individual fold models |
| `prediction_operations.json` | 66 native cases; original 36 unchanged; GBDT/DART/GOSS continuation and RF continuation rejection; initial-score buffers versus model margins; 52 weighted refits at decay 0/0.5/0.9/1 for regression/binary/multiclass/categorical/ranking/linear/RF/DART/L1/quantile/MAPE/query-bagging paths; two finite out-of-range decay errors; actual refit derivatives; rollback through zero trees; leaf setter/getter; prediction early stopping; linear restrictions; infinite-gain diagnostic |

Workflow events contain `enter`, `exit`, or `raise`, the callback name, zero-based iteration, begin/end iteration, model iteration, one-based model best iteration, learning rate, evaluation results, and history lengths. Events retain the original ordered evaluation fields: `dataset_name`, `metric_name`, `metric_value`, `maximize`, and nullable `metric_std_dev`. The early-stop exception records its zero-based best index separately. `train()` assigns the final one-based best iteration only after the exception. Callbacks after the throwing callback do not run in that iteration. The last history entry can therefore be later than the best saved tree. `final_model` uses default best-iteration serialization; `full_model` retains all actual trees. Training without splits still invokes every scheduled Python round, although the native model remains one tree.

The binary first-metric-only pair uses native log loss followed by AUC and patience 1. All metrics select best iteration 3 with four history entries; first-metric-only reaches iteration 7. Training-only L2 disables early stopping. No validation produces the actual Python `ValueError` after the first native update. DART produces the original disabling warning. Custom L2 and binary start at zero with `boost_from_average=false`; saved native raw predictions equal the equivalent built-in runs exactly. Custom metric values match actual built-in native metric results within the existing double bound. Invalid gradient length records the original Python validation error before learner access. The C API has no derivative-length argument; this fixture does not make an unsafe out-of-bounds call or invent a C++ length error.

CV uses seed 19. Standard shuffled folds use the original `numpy.random.RandomState(19).permutation` and `array_split`, not a LightGBM RNG. The record includes the exact permutation, source fold arrays, and sorted indexes actually supplied to native Dataset subsets. Stratified and ranking cases delegate to the original engine-selected scikit-learn 1.6.1 `StratifiedKFold` and `GroupKFold`; observer wrappers only record yielded arrays. Ranking groups stay intact. The upstream default GroupKFold does not shuffle or use the CV seed. Every step retains per-fold native train/valid metrics and the original engine aggregate. Standard deviation uses population normalization (`ddof=0`). Early stopping truncates all returned histories to one common best iteration; full saved fold models retain the extra patience rounds. CVBooster JSON and per-fold text models are both retained. Dependency versions and Python source hashes are part of each family's provenance; distribution hashes are in the oracle lockfile.

Continuation starts from the exact saved source model. It is a model warm start, not an exact training-state checkpoint or RNG restoration promise. The original engine overwrites supplied initial scores with source-model predictions when both are provided. Separate initial-score-only evidence shows the training score buffer includes supplied scores, while a saved model contains only the learned margins. RF rejects continuation with initial scores using the recorded native error. Refit records original and new rows/labels/weights/groups, effective parameters, native `leaf_pred`, and new model probes. Initial ranking training has positions; the original Python refit API does not pass positions into its new Dataset, so new positions are explicitly null. A native helper observes each original refit derivative call, in flat class-major order, and verifies its final raw predictions against Python refit exactly. No derivative or leaf-renewal kernel is reimplemented by the helper.

The appended refit cases establish actual RF and DART success; the RF continuation error does not imply refit rejection. The helper preserves the native algorithm and does not call virtual `GetTrainingScore()` merely to observe scores, because DART uses that method to change dropout state. RF calls its original `RF::Boosting()` on each refit iteration; after merge, `BoostFromAverage` returns zero, so RF derivatives are fixed from zero scores even as refit score storage accumulates. DART refit does not execute the `TrainOneIter` dropout-normalization loop. L1, quantile, and MAPE refit call generic `FitByExistingTree` Newton output calculation without `RenewTreeOutput`; their original training leaf renewal still applies during initial training.

The appended successful cases retain `backend_params` from the actual saved result. GBDT query bagging remains enabled. For GOSS, `Config::CheckParamConflict` normalizes requested `bagging_by_query=true` to false (`src/io/config.cpp:477-479`); this is not evidence of GOSS query sampling. Decay -0.1 and 1.1 produce actual native `LightGBMError` checks; decay 0 and 1 succeed. No nonfinite decay or unsafe leaf-index call is used.

Imported-model refit with positive `path_smooth` is not executed. `Tree` text loading does not populate `leaf_parent_`, while `SerialTreeLearner::FitByExistingTree` can read it when smoothing non-root leaves. This can cause undefined behavior. The helper rejects the probe before refit. This is a source-based safety exclusion, not a recorded native error or a claim that smoothing refit is supported.

Rollback reaches zero trees and two subsequent native calls are no-ops; the saved empty forest returns zero margins and empty leaf arrays. Leaf mutation records the original value, assigned value, getter result, unchanged leaf indexes, and changed native scores. Prediction early-stop cases retain frequency/margin parameters, actual raw/transformed/leaf results, and independently requested native prefix scores. Binary stopping tests `2 * abs(raw) > margin`; multiclass tests the top-two score gap. The regression linear case does not stop early, and native TreeSHAP rejects linear models.

### Finite labels with infinite stored gain

`operations_extreme_gain.txt` is an unmodified native model trained on four rows with finite `f32` labels near ±1e20. All four serialized trees contain `split_gain=inf`. Loaded native gain importance returns positive infinity, encoded as the explicit `Infinity` string permitted by the operation schema. Raw predictions and TreeSHAP contributions are finite; contributions sum to raw scores within the unchanged bound. Native `dump_model()` instead reports `9.999999680285692e37` for each gain because `Tree::NodeToJSON` calls `Common::AvoidInf(float)`. Do not infer stored gain from that clamped JSON inspection. Positive infinite split gain in text is therefore not, by itself, evidence of an invalid model or nonfinite predictions. No input-domain rule or numerical bound is relaxed by this diagnostic.

## Fixed larger quality and performance cases

`tools/oracle/benchmark.py` is the executable data definition. `tests/fixtures/benchmark-contract.json` retains its dimensions, quality limits, and budgets. Each case uses NumPy PCG64 with seed 0 and an independent, fixed validation partition. Generator dependencies are locked. Inputs are generated, not downloaded.

Each case has 8,192 training rows, 2,048 validation rows, 16 columns, 50 boosting rounds, 15 leaves, maximum bin count 63, learning rate 0.1, all seeds 0, one thread, and forced column-wise histograms. Ranking queries have 16 rows. Sparse storage is canonical CSR with sorted indexes and no duplicate entries; density is 10%. Resource limits are 512 MiB per process and 300 seconds per case. No GPU is permitted.

| Case | Objective / representation | Validation quality gate |
| --- | --- | --- |
| `dense` | L2 / dense `f64` | RMSE <= 0.45 |
| `sparse` | L2 / CSR `f64` | RMSE <= 0.30 |
| `categorical` | Binary / first column has raw integer codes 0..7 | Binary log loss <= 0.15 |
| `multiclass` | Softmax / three classes | Multiclass log loss <= 0.45 |
| `ranking` | LambdaRank / relevance 0..3, groups of 16 | Mean NDCG@5 >= 0.90 |

Every case must also satisfy the same-host reference quality delta: loss/RMSE at most reference + 0.02, or NDCG@5 at least reference - 0.01. Both the absolute gate and delta gate apply. Quality alone does not establish algorithm conformance.

Performance budgets remain those in the plan, per case, not only in aggregate:

- Median training time: Rust/reference <= **1.5x**.
- Median inference time: Rust/reference <= **1.25x**.
- Peak resident memory: Rust/reference <= **1.5x**.

Measure dataset construction separately. Use release builds on the same host, inputs, rounds, thread counts, and preprocessing boundary. Use one warm-up and at least five measured runs. Record every sample, hardware, compiler, flags, shape, sparsity, and validation metric. Measure input construction, training, and prediction separately. Include the model import boundary only when both sides include it.

`docs/performance.md` and `docs/performance.json` record a source-identified snapshot. All five quality, saved-baseline timing, matched native-only peak-memory and native inference gates pass. Dense and multiclass native training meet the unchanged 1.5x budget; sparse, categorical and ranking training remain open. These results apply only to the recorded source hash. A later source change requires a new measurement.

The optional reference-only timing report is not a Rust comparison. Python process peak RSS includes the Python/NumPy/SciPy harness and must not be used as a product-memory ratio. The matched report uses native-only child processes and a common allocation boundary. Matched native parallel performance and parity remain unverified because this macOS ARM64 baseline has OpenMP off. Separate Rust tests cover deterministic 1/2/4-thread execution.

## Regeneration and checking

See `tools/oracle/README.md`. Normal Cargo tests must use only the saved text evidence. They must not invoke Python, compile C++, or load the reference library. The oracle is an isolated developer tool, not a product build dependency.

An oracle check must reject a different source commit, changed tracked reference source, unexpected Python source, unrelated installed wheel, changed compiler, or loaded-library path/hash mismatch. Hash checks detect changed evidence; numerical tests and required-case coverage detect defects. The saved inventory gate requires 141 canonical parameters, 167 distinct aliases, compiled default types, and the three text-backed vector fields. Reference mode compares the alias/type maps and all 16 CPU objective and 24 CPU metric factory names directly with the pinned source. This inventory audit is not a claim that every parameter is supported in Rust. A deliberate output-value mutation, shape mutation, and missing-case mutation must fail the Rust fixture runner before its coverage gate is considered complete.

The final native families have separate strict schemas and exact required-case sets:

| File | Required cases and native source handles |
| --- | --- |
| `workflow_extended.json` | 15 cases from `generate_workflow_extended.py::NAMES`; DART schedules, continuation, uniform/weighted/XGBoost dropout, GOSS schedules, custom/built-in CV schedules and stopping, and GBDT/DART/GOSS learning-rate plus feature-fraction combinations |
| `metric_edges.json` | 68 AUC-mu cases from `generate_metric_edges.py::generate`; original 57 plus seven size-128 cases and four exact-EPS/asymmetric-cost cases; native `extended_driver.cpp` calls the CPU metric factory |
| `empty_queries.json` | 14 unweighted XE-NDCG cases from `generate_empty_queries.py::generate`; nonempty, leading/middle/trailing/repeated zero query groups, short/one-row groups, each with and without positions; four sequential factory calls and three real native training rounds |

Workflow checks validate exact rows/labels/parameters, schedule phase and iteration offsets, continuation model counts, custom derivative input state, callback trace, fold membership, population deviations, histories, saved-model shapes, and final scores. The GOSS native training score buffer is retained separately from all-row model predictions. Reading DART's training score before an update can consume dropout RNG; these fixtures preserve the actual producer's call order. All `models/workflow_extra*.txt` files regenerate byte-for-byte. Model paths must be safe relative paths without `..`.

AUC-mu inputs retain one IEEE-754 `score_bits` entry per score. Checks require exact input values, class-major shapes, weight shapes, case coverage, metric direction, and the native result range. This evidence fixes the single-thread Apple clang 21.0.0 / libc++ 2200.27.0 sort behavior. LightGBM's EPS comparator is not a strict weak ordering; this is not a portable claim about other standard libraries. The manifest records all six sort/heap header hashes. The translated Rust sorting portion has the Apache-2.0 WITH LLVM-exception license; `licenses/LLVM.txt` retains the complete license and notices. Other original code remains MIT unless stated otherwise.

Empty-query checks require exact zero-group locations and cumulative boundaries, signed position codes, four-step full/reversed/sampled/full call order, unchanged derivatives outside sampled queries, and position-pair RNG equivalence. NDCG/MAP names, cutoffs, result ranges, training shapes, and safe query-bagging coverage are checked. Empty-query LambdaRank, weighted empty groups, query bagging when query count exceeds row count, and unsafe row subsets are not executed; their source-based restrictions remain in the fixture. No formula replaces a native derivative, metric, or training result.

Reference mode generates all three families and their schemas in temporary fixtures. Workflow JSON, schemas, and all model bytes must match exactly. For AUC-mu and empty-query JSON, only the explicitly recorded `native_driver_sha256` can differ after Mach-O relinking. Every deterministic field must match; numeric evidence is never excluded. The manifest records each producer's exact production compile/link commands, compiler identity, compile-command-file hash, driver source/header/object hashes, all linked production object/archive hashes, and the verified loaded-library identity. Reference mode checks these build inputs before and after regeneration. It also checks the actual libc++ sort headers. Saved artifact hashes still cover the full original JSON, including each retained executable hash.

Bundling has its own strict schema and required-case set. Its checker verifies shapes, feature/group membership, decoded conflict counts, model hashes, and helper source/object/library identity. Reference regeneration runs the unchanged producer with its output root redirected to a temporary directory. All deterministic JSON fields and all 22 model files must match. The retained record's `resources` samples and `native_helper.executable_sha256` are validated but not required to match between runs: runtime measurements and Mach-O relink hashes can change. No prediction, model byte, bin, mapping, group, parameter, source hash, object hash, or library identity is excluded from comparison.

## PR #1 review evidence

Three separate native families cover the review outcomes. `review_refit` has four binary/OVA decimal/scientific-sigmoid cases. They compare fresh and reloaded refit, full-precision derivatives, live transformed output and the rounded saved-model boundary. `review_cegb` has 12 coupled/split/lazy-cost cases, including pending-leaf competition, masks, constraints and cache replay. The pinned reference caches a raw candidate and adds its coupled refund; the proposed effective-cost change is not adopted. `review_dart` has 13 cases. It retains native successful prefixes and safe nonempty-bag failure observations. Rust restores a failed dropout attempt to the last valid prefix rather than returning native negative shrinkage.

The producers use the verified native library and original Python API. No new helper or numeric substitute is used. The schemas require complete shapes and reject extra fields. Checks cover exact canonical inputs, source/compiler/library/producer identity, safe model paths, native model hashes and the declared checked DART departure. Reference mode regenerates all JSON fields and model bytes without exclusions. Mutation tests cover missing cases, changed values, malformed rows, wrong identities, unsafe paths, the CEGB pending-leaf choice, sigmoid precision and the excluded empty native learner path. The review ledger is `docs/reviews/pr-1.md`.

## Executed oracle wave

The Phase 7 native wave passed `check.py --reference --self-test` with 145 artifact hashes. All deterministic fixtures and model text regenerated byte-for-byte. Missing coverage, malformed shapes, changed values, missing/changed helper identities, changed library hashes, unrelated wheel paths, and an unrelated actual loaded C image were rejected. Earlier model/primitive/prediction fixture bytes were checked against the pre-wave manifest and remained unchanged; metrics only appended six cases and schemas expanded. This oracle check is not a Rust implementation or performance pass.

The Phase 8 native wave passed `check.py --reference --self-test` with 281 artifact hashes. It regenerated all 15 workflow cases, six CV cases, 36 operation cases, their schemas, and all earlier deterministic evidence byte-for-byte. The check also recompiled the refit observer against the pinned CPU objects. New missing-case, changed-value, malformed-shape, missing-provenance, and refit-helper identity mutations were rejected. The RF continuation and linear TreeSHAP native error lines in the check log are expected rejection cases, not failed checks. All earlier fixture/model hashes were audited before the manifest update and were unchanged. Python/C++ generator resource gates passed; no benchmark case, budget, or baseline was changed. Dense/sparse Rust training performance failures and unmatched memory remain open.

The post-Phase-8 extension passed `check.py --reference --self-test` with 357 artifact hashes. It regenerated 66 operation cases, ten migrated extra constraint cases, and all earlier deterministic non-bundling evidence byte-for-byte. The original first 36 operation records remained unchanged. All ten migrated constraint model files and prior case values were compared unchanged, except for safe model paths and top-level provenance. The focused offline `learner_controls` Cargo suite passed all 31 tests. Package-list inspection included the migrated fixtures and excluded Python/C++ oracle code. Bundling's 22 actual deterministic payloads and all model bytes regenerated unchanged under the explicit runtime/build-field rule above. The saved parameter inventory and pinned CPU factory source audit passed. These checks do not waive Rust correctness, quality, or performance gates.

The final native integration passed `check.py --reference --self-test` with **399 artifact hashes** and **126 deliberate mutation rejections**. It regenerated 15 extended workflow, 68 AUC-mu edge, and 14 empty-query cases. All 357 prior artifacts were audited before the manifest update: prior fixture and model bytes were unchanged; only intended oracle/documentation records and the parent-owned notices changed. Self-tests reject missing cases, changed native values, wrong IEEE score bits, malformed shapes, extra fields, changed native/Python/compiler identities, changed sort headers, zero-group relocation, position mutation, sampled-query order changes, stateful RNG grouping changes, schedule phase changes, custom derivative input-state changes, and CV deviation changes. The check also preserved 18 base prediction cases, 74 metric cases, ten extra constraint cases, 22 bundling cases, 15 original workflow cases, six CV cases, 66 operation cases, 141 canonical parameters, 167 aliases, and all 16 objective/24 metric CPU factory names. This oracle gate is separate from product and performance gates.

## Remaining evidence

The saved set is not exhaustive CPU conformance. Factory coverage does not prove all objective domains or metric parameter combinations. Invalid-domain native error fixtures, dataset subset/continuation metadata, categorical smoothing/regularization combinations, extreme multiclass stability cases, and broader ranking combinations remain incomplete. The four original linear models and three added NaN/weighted cases establish native examples, not Rust linear training. Saved constraints, sampling, quantization, and contribution cases are compact evidence, not exhaustive combination coverage. Internal full-training node call order, broader overflow/quantization boundaries, invalid constraint combinations, broader callback/CV/model-operation combinations, and platform/thread combinations need further evidence. The new isolated workflow and operation examples do not establish full CPU conformance. All five quality, saved-baseline timing, matched memory and native inference gates pass for the source snapshot in `docs/performance.md`. Dense and multiclass native training also pass. Sparse, categorical and ranking native training remain above 1.5x, so full performance and release readiness remain open. Later source changes require a new checked measurement. Product-only Cargo, package and native-platform evidence is recorded in `docs/verification.md`.
