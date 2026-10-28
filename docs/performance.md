# CPU performance evidence

## Gate status

**The required training-time gate remains open for sparse, categorical and ranking cases.** Dense and multiclass training pass. All five quality, matched peak-memory and inference gates pass on this source snapshot. No budget or fixed input was changed.

[The machine-readable report](performance.json) retains raw samples, parameters, input hashes, source/build identities and per-case comparisons. Budgets remain training <=1.5x, inference <=1.25x and peak memory <=1.5x, for every case.

## Source and measurement

- Measurement: 2026-10-28T14:27:02Z.
- Host: macOS ARM64, Apple M4, 10 logical CPUs.
- Rust 1.92.0, release profile with thin LTO and one codegen unit. No unsafe product code.
- Reference: LightGBM `5d84f3f222b41409a6983d2ac90513ad9702fa9e`; Apple clang 21.0.0, production `-O3` flags.
- Loaded library SHA-256: `9ed1d46d2d60bac2966fd03bd81e7d192025804f6a3e3ef24c3aa483d0ec7c1e`.
- Rust build-input SHA-256: `130b9c380fd908d05136b2e947dc9df027a2eb8bf44d0bec48521a492a553608`.
- Source hashes matched before the isolated snapshot, after compilation and after measurement. This is a source identity, not a Git commit identity.

The source digest covers `src/**/*.rs`, `benches/**/*.rs`, Cargo manifest/lockfile, pinned toolchain and both benchmark examples. The report records each file hash and executable hash.

Each fixed case has 8,192 training rows, 2,048 validation rows, 16 features, 50 rounds, 15 leaves and `max_bin=63`. Seeds, labels, sparse arrays and categorical codes match the unchanged generator. Ranking groups contain 16 rows. None of these cases has weights.

Five fresh processes per implementation run in alternating native/Rust order. Each process performs one warm-up and five measured iterations; medians use all 25 samples. Both drivers read the same typed little-endian arrays. Python generates/checks evidence outside the measured processes.

- Construction includes metadata.
- Native training includes `LGBM_BoosterCreate` and 50 `LGBM_BoosterUpdateOneIter` calls; Rust uses product `Trainer::fit`.
- `metric=None` disables metric kernels in both drivers. Rust still validates scores and retains history records. No estimated overhead is subtracted.
- Serialization and batch prediction have separate timers.
- Entire-process peak RSS includes loading, raw arrays, warm-up, construction, training, serialization, prediction and output. No baseline subtraction or discarded first peak is used.
- The controller verifies `dladdr(LGBM_GetLastError)`, library/source/compiler hashes and compile commands before and after measurement.

## Timing

| Case | Rust/native training | Training gate | Rust/native inference | Rust/saved Python training |
| --- | ---: | --- | ---: | ---: |
| dense | 1.443 | Pass | 0.859 | 0.405 |
| sparse | 3.164 | **Open** | 0.854 | 0.910 |
| categorical | 1.568 | **Open** | 0.969 | 0.309 |
| multiclass | 1.427 | Pass | 0.734 | 0.276 |
| ranking | 1.815 | **Open** | 0.755 | 0.567 |

All saved Python-engine training/inference comparisons pass. These are not native-only throughput passes. The direct C API results remain separate; their failures are not waived.

Dataset construction and training medians, in seconds:

| Case | Rust construction | Native construction | Rust training | Native training |
| --- | ---: | ---: | ---: | ---: |
| dense | 0.003264 | 0.003805 | 0.023152 | 0.016049 |
| sparse | 0.001327 | 0.001110 | 0.021649 | 0.006842 |
| categorical | 0.003329 | 0.003744 | 0.014284 | 0.009108 |
| multiclass | 0.003329 | 0.003835 | 0.078126 | 0.054740 |
| ranking | 0.003266 | 0.003896 | 0.047508 | 0.026179 |

## Peak memory

| Case | Maximum Rust bytes | Maximum native bytes | Peak ratio | Max Rust / min native |
| --- | ---: | ---: | ---: | ---: |
| dense | 11517952 | 11091968 | 1.038 | 1.200 |
| sparse | 6832128 | 6766592 | 1.010 | 1.035 |
| categorical | 8404992 | 9830400 | 0.855 | 0.874 |
| multiclass | 10240000 | 10977280 | 0.933 | 0.963 |
| ranking | 8749056 | 18497536 | 0.473 | 0.481 |

Every conservative ratio is <=1.5. All measured peaks are below the fixed 512 MiB process limit. macOS `/usr/bin/time -l` reports bytes. The tool's Linux conversion path is not native Linux measurement evidence.

## Quality and parallel limits

All absolute quality bounds and reference deltas pass. Native C API predictions exactly equal the independently regenerated Python-workflow predictions. Rust predictions satisfy `1e-12 + 1e-10 * abs(reference)` for every row/class and fresh process. These checks do not prove all intermediate states or exhaustive algorithm parity.

Supplemental Rust runs at two and four threads retain timings, peaks and prediction checks. They are **not matched parallel budget evidence**: the pinned native library has OpenMP off and remains serial. No alternate baseline replaced it. Other platforms and actual matched native parallel performance remain unverified.

## Reproduction

```sh
uv run --project tools/oracle --locked python tools/oracle/measure_memory.py
```

The command needs the pinned isolated reference build and exported fixed inputs. Binary arrays, helper objects/executables, snapshots and raw logs remain under `target/phase9-final-memory*`; they are not committed. Python/C++ oracle tools are excluded from the Cargo package.

A source change requires a new measurement. Correctness, package and native-platform gates are separate. This report does not complete the active plan or authorize publication.
