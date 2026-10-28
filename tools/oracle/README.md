# Isolated CPU reference oracle

This directory is a developer tool. Product builds and normal Cargo tests must not invoke it. It does not install a LightGBM wheel. Native sources and binaries remain under ignored `references/` and `target/oracle/`.

## Prerequisites

Use `uv`, Git, and the existing Apple clang/Command Line Tools. The recorded run used macOS arm64, Python 3.12.12, Apple clang 21.0.0, CMake 3.31.6, and uv-provided Ninja. `pyproject.toml` and `uv.lock` isolate Python, CMake, and Ninja dependencies. Do not use Homebrew or a system package install.

The source must be a clean checkout of commit `5d84f3f222b41409a6983d2ac90513ad9702fa9e`, with `VERSION.txt` equal to `4.7.0.99`. If the ignored checkout is absent, prepare it from the repository root:

```sh
git clone --no-checkout https://github.com/lightgbm-org/LightGBM references/LightGBM
git -C references/LightGBM checkout --detach 5d84f3f222b41409a6983d2ac90513ad9702fa9e
```

Do not overwrite an existing checkout or local changes. `build.py` rejects a different commit or modified tracked source. It fetches only the pinned Eigen, fmt, fast_double_parser, and nanoarrow submodules required by the CPU sources. `.gitmodules` and `CMakeLists.txt` are the dependency definitions. Boost.Compute is not required or fetched. Submodule initialization uses the parent commit's Git links, not branch heads.

## Build and regenerate

Run these noninteractive commands from the repository root. Use a 300-second command timeout. The environment path is absolute so uv cannot create `.venv` beside the scripts.

```sh
export UV_PROJECT_ENVIRONMENT="$PWD/target/oracle/venv"
uv sync --project tools/oracle --locked
uv run --project tools/oracle --locked python tools/oracle/build.py
uv run --project tools/oracle --locked python tools/oracle/generate_predictions.py
uv run --project tools/oracle --locked python tools/oracle/generate_primitives.py
uv run --project tools/oracle --locked python tools/oracle/generate_regressions.py
uv run --project tools/oracle --locked python tools/oracle/generate_extended.py
uv run --project tools/oracle --locked python tools/oracle/generate_prediction_extended.py
uv run --project tools/oracle --locked python tools/oracle/generate_linear.py
uv run --project tools/oracle --locked python tools/oracle/generate_feature_info.py
uv run --project tools/oracle --locked python tools/oracle/generate_explain.py
uv run --project tools/oracle --locked python tools/oracle/generate_training_controls.py
uv run --project tools/oracle --locked python tools/oracle/generate_constraints_extra.py
uv run --project tools/oracle --locked python tools/oracle/generate_bundling.py
uv run --project tools/oracle --locked python tools/oracle/check_bundling.py
uv run --project tools/oracle --locked python tools/oracle/generate_sampling.py
uv run --project tools/oracle --locked python tools/oracle/generate_linear_training.py
uv run --project tools/oracle --locked python tools/oracle/generate_workflow.py
uv run --project tools/oracle --locked python tools/oracle/generate_cv.py
uv run --project tools/oracle --locked python tools/oracle/generate_operations.py
uv run --project tools/oracle --locked python tools/oracle/generate_workflow_schemas.py
uv run --project tools/oracle --locked python tools/oracle/benchmark.py
uv run --project tools/oracle --locked python tools/oracle/manifest.py
uv run --project tools/oracle --locked python tools/oracle/check.py --reference --self-test
```

`build.py` records the exact CMake flags and compiler command. It requests Release, `USE_OPENMP=OFF`, `USE_GPU=OFF`, `USE_CUDA=OFF`, `USE_ROCM=OFF`, `USE_MPI=OFF`, `USE_HOMEBREW_FALLBACK=OFF`, and no CLI or C++ test suite. CMake compiles at most two files at once; this build-job limit does not make reference training parallel. The library output is explicitly redirected out of the source checkout to `target/oracle/lib/`.

The staged Python package is copied from the pinned checkout, not installed from PyPI. Each load checks source commit/tree/version/cleanliness, Python source hashes, compiler hash, compile-command hash, and library hash. `dladdr(LGBM_GetLastError)` verifies the image actually supplying the C symbol, not only a requested filename. Package-path checks reject an installed wheel. `otool -L` records native dependencies; there is no OpenMP runtime in this build. Python optimization mode is rejected because it disables assertions.

`generate_predictions.py` trains tiny models, saves unmodified upstream text, reloads that text, and serializes reference results with finite JSON and round-trip float precision. `prediction.json` has the 18 base cases. `prediction_ranges.json` has four selected-iteration cases and one absent-class multiclass case, using the same schema. `generate_primitives.py` compiles `primitive_driver.cpp` against the actual CMake-built CPU object files. It runs real `Random`, `BinMapper`, objective, dense-bin histogram, and split-search kernels. It also generates a private include from the public `Config` data members and compiles it to extract every canonical declared default, including no-save parameters. Parameter types and ordered aliases come from the reference registry. Empty default vectors can be expanded by `Config::Set`; the inventory distinguishes declared fields from serialized defaults.

`generate_extended.py` compiles `extended_driver.cpp`. Both drivers reuse `driver_common.hpp`; the original primitive fixture bytes are unchanged. The primitive helper uses `-O0` to keep compiler memory below 512 MiB; the linked native CPU objects retain the recorded Release flags. The driver record stores the effective helper command and shared-header hash. It calls the actual CPU objective and metric factories, `Metadata`, `BinMapper`, and `Dataset` storage. Inputs are in `specifications()`. Ranking `steps` records the second derivative call on the same state. Shared validation cases use `Dataset::CreateValid` and read the actual native bin iterators. Each feature receives its own forced cutpoints and maximum bin count.

`generate_prediction_extended.py` saves tiny trained models for all 16 CPU objectives and weighted/unweighted categorical one-hot and sorted-partition splits. It separately predicts the 18 base cases through SciPy CSR input; these are native sparse results, not dense predictions copied to a sparse adapter. `generate_linear.py` saves four actual linear-leaf models with regularization, rank-deficient columns, first-tree behavior, and NaN fallback probes. Singular cases use generated forced splits on both dependent columns to exercise a two-feature path; the native solver can remove a zero coefficient from the saved leaf. These generators do not change the base fixtures.

`generate_feature_info.py` compiles the retained original `data_feature_info.cpp` and records all 14 native `BinMapper::bin_info_string()` cases. `generate_explain.py` reloads the 18 base models and five range/absent-class cases, calls native dense and CSR TreeSHAP and importance, and checks class-wise contribution sums against native raw scores. It records the four native linear-contribution errors separately. Gain metadata is parsed as `float` by `Tree`; the explanatory record keeps serialized tokens and actual loaded gains.

`generate_training_controls.py` writes separate `prediction_boosting.json` and `prediction_constraints.json` families. It retains all active parameters, labels, weights, query groups, positions, native model text, and per-round raw scores and shrinkage. GOSS runs nine rounds at learning rate 0.25, beyond its four-round warm-up. DART uses active dropout with `skip_drop=0`; RF uses actual row and feature sampling. `generate_linear_training.py` adds three separate NaN-training/weighted linear cases without changing the original four.

`generate_sampling.py` compiles `sampling_driver.cpp` through `native.py`. It calls actual `SampleStrategy`, `ColSampler`, `DART`, and `GradientDiscretizer` methods. The DART probe exposes private state only in this development helper. It does not implement a replacement sampler. Feature cases call two sequential root-node samples per tree; these are controlled call-order primitives, not a claim to record every training node. GOSS records input and rescaled derivatives, selected indexes, and the explicit all-row warm-up state. Quantization records signed interleaved Hessian/gradient bytes and scales from four calls on the same native instance. Both new helpers retain compile/link commands, source/executable/library hashes, and every linked CPU/C API object hash in the manifest.

`generate_workflow.py` records 15 actual `engine.train()` runs and all invoked callback entry, exit, and exception events. It wraps the original callbacks, without replacing their state transitions. `generate_cv.py` records six actual `engine.cv()` runs, observed fold arrays, per-fold native metrics, population standard deviation, history truncation, and saved CVBooster/fold models. The isolated environment adds scikit-learn 1.6.1 for the upstream stratified and group-safe default splitters. Standard shuffled folds use upstream NumPy `RandomState(19)`, not a LightGBM RNG. The records retain its permutation and identify every fold generator. NumPy does not implement an alternative learner or metric kernel.

`generate_constraints_extra.py` retains ten additional native constraint cases in `prediction_constraints_extra.json` with models under `models/constraints_extra/`. It preserves the prior models and case values byte-for-byte except for model paths and top-level provenance. The separate schema checks safe fixture-relative paths. Python producer code stays outside the Cargo package.

`generate_operations.py` records 66 native cases for continuation, initial scores, weighted refit, rollback through a zero-tree model, leaf mutation, prediction early stopping, linear restrictions, and finite-label/infinite-gain behavior. The original 36 cases are unchanged and the new cases are appended. `refit_driver.cpp` subclasses the selected native GBDT, RF, or DART implementation only to observe its original virtual `Boosting()` call inside `RefitTree()`. It calls the selected original method before it reads native gradients and Hessians. The observer reads score storage directly; it must not invoke DART's state-changing `GetTrainingScore()`. Every helper-refitted model must predict exactly the same raw values as the Python refit result. The manifest records compiler/link/object identities for this helper. `generate_workflow_schemas.py` writes the three separate schemas. `check_workflows.py` checks ordering, shapes, serialized model counts, aggregation, and operation invariants. These files do not change the original prediction schema or earlier fixture outputs.

New refit cases cover RF, DART, L1, quantile, MAPE, and query bagging at decay 0/0.5/0.9/1. RF and DART succeed. RF computes fixed derivatives from zero scores after merge, not from the accumulating refit score buffer. Non-smooth objectives use native `FitByExistingTree` Newton outputs without percentile renewal. GOSS normalizes requested query bagging to false; each new successful case retains selected actual saved backend parameters. Finite decay -0.1 and 1.1 produce native parameter errors. Imported-model `path_smooth` refit is not executed: native text loading does not restore `leaf_parent_`, which smoothing can access. The helper rejects that unsafe probe before calling native refit.

`generate_bundling.py` and its retained native driver produce 22 actual enabled/disabled bundling cases, including conflicts, NaN/zero handling, categorical codes, filtering, frequent nonzero bins, reordered feature mappings, and extra trees. `check_bundling.py` adds the schema and validates group membership, decoded bins/conflicts, model hashes, and native build identity without changing that producer. Reference mode redirects only the producer's output root into a temporary directory. Every model byte and deterministic JSON value must match. The existing bundling record also contains runtime resource samples and a helper executable hash; these fields are validated but excluded from equality across runs because elapsed time/RSS and Mach-O relink bytes can change. This exception does not apply to predictions, bins, mappings, groups, parameters, source/object/library hashes, or model bytes.

`benchmark.py` fixes the five resource-bounded data generators and saves one warm-up plus five timing samples for each case. It checks predeclared absolute quality gates and repeatability. It does not time Rust, establish any time ratio, or measure memory at a matched native boundary. Timing samples naturally change when regenerated. Run `benchmark.py --contract-only` to write only the deterministic case contract. Run `benchmark.py --export-inputs` to write the exact dense/sparse inputs to ignored `target/oracle/benchmark-input-{dense,sparse}.json`. Both files contain row-major flat values and `f32` labels; the sparse file also contains canonical CSR arrays. Load and validate these files outside measured Rust timers. Sparse reference inference uses CSR input, so a dense Rust inference path is not a matched representation.

`manifest.py` records SHA-256 for saved models, JSON, schemas, notices, conformance rules, and oracle source/lock files. Review generated changes before accepting them. A source/baseline change requires explicit review; do not update expectations only to remove a test failure. A regenerated build on a different host has different identity fields and can have different binaries; byte identity is asserted only for the recorded build's generated evidence.

## Check saved evidence

```sh
export UV_PROJECT_ENVIRONMENT="$PWD/target/oracle/venv"
uv run --project tools/oracle --locked python tools/oracle/check.py --self-test
```

This checks schema requirements, required case coverage, output shapes, finite JSON, benchmark contract/quality, artifact hashes, and the frozen parameter/alias inventory. Reference mode also audits all 16 CPU objective and 24 CPU metric factory names against the pinned source registry. It does not need the ignored reference checkout or native library. It still uses the isolated developer Python environment; normal Cargo fixture tests do not.

`--reference` also validates the loaded image, regenerates all deterministic base, regression, extended objective/metric/metadata/bin, categorical, sparse, linear, feature-info, TreeSHAP, boosting, constraint, internal sampling, quantization, Python workflow/CV, model-operation, and workflow-schema evidence into a temporary directory under `target/oracle/`, recompiles and runs the driver, and requires byte-identical saved evidence. It does not overwrite retained fixtures. Bundling uses the explicit runtime/build-field comparison rule above. `--self-test` proves missing cases, malformed output shapes, changed numeric values, and missing manifest identity are rejected. With `--reference`, it also tests changed library hashes, unrelated wheel paths, and an unrelated actual loaded C image with a claimed reference filename.

## Check Rust-exported models in the native reference

The Rust integration test can export models without invoking native tools. Then run the native check separately, using the directory it reported:

```sh
export UV_PROJECT_ENVIRONMENT="$PWD/target/oracle/venv"
uv run --project tools/oracle --locked python tools/oracle/check_roundtrip.py target/model-roundtrip-80588
uv run --project tools/oracle --locked python tools/oracle/manifest.py
uv run --project tools/oracle --locked python tools/oracle/check.py --reference --self-test
```

The directory name is an example from the recorded run, not a permanent path. The check requires all 18 exported models, loads them with the verified native reference, and checks every retained input row. `tests/fixtures/rust-model-roundtrip.json` records hashes and results. Exported models remain ignored under `target/`. This is interoperability evidence, not identical model serialization or unsupported Rust training support.

## Evidence and gaps

See `docs/conformance.md`, `docs/reference-manifest.json`, and `THIRD_PARTY_NOTICES.md`. The primitive cases are not an exhaustive CPU test suite. Other operating systems, compilers, OpenMP, cross-thread determinism, and native-memory comparisons remain unverified. Main-task measurements passed dense/sparse quality but failed the unchanged training-time budget; no complete performance pass is claimed. No parallel speed equivalence is claimed.
