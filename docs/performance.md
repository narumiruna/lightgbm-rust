# CPU performance evidence

## Gate status

**The required training-time gate remains open for sparse, categorical and ranking cases.** Dense and multiclass training pass. All five quality, matched peak-memory and inference gates pass on this source snapshot. No budget or fixed input was changed.

[The machine-readable report](performance.json) retains raw samples, parameters, input hashes, source/build identities and per-case comparisons. Budgets remain training <=1.5x, inference <=1.25x and peak memory <=1.5x, for every case.

## Source and measurement

- Measurement: 2026-10-28T17:00:47Z.
- Host: macOS ARM64, Apple M4, 10 logical CPUs.
- Rust 1.92.0, release profile with thin LTO and one codegen unit. No unsafe product code.
- Reference: LightGBM `5d84f3f222b41409a6983d2ac90513ad9702fa9e`; Apple clang 21.0.0, production `-O3` flags.
- Loaded library SHA-256: `9ed1d46d2d60bac2966fd03bd81e7d192025804f6a3e3ef24c3aa483d0ec7c1e`.
- Rust build-input SHA-256: `6b361a683565d0d59bc57d05c964fe16edcf81c2dc2043d13bcb57b194d06a61`.
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
| dense | 1.443 | Pass | 0.812 | 0.406 |
| sparse | 3.181 | **Open** | 0.829 | 0.917 |
| categorical | 1.572 | **Open** | 0.958 | 0.311 |
| multiclass | 1.427 | Pass | 0.726 | 0.276 |
| ranking | 1.815 | **Open** | 0.712 | 0.565 |

All saved Python-engine training/inference comparisons pass. These are not native-only throughput passes. The direct C API results remain separate; their failures are not waived.

Dataset construction and training medians, in seconds:

| Case | Rust construction | Native construction | Rust training | Native training |
| --- | ---: | ---: | ---: | ---: |
| dense | 0.003295 | 0.003886 | 0.023241 | 0.016110 |
| sparse | 0.001325 | 0.001129 | 0.021823 | 0.006860 |
| categorical | 0.003172 | 0.003753 | 0.014375 | 0.009142 |
| multiclass | 0.003269 | 0.003856 | 0.078196 | 0.054780 |
| ranking | 0.003321 | 0.003841 | 0.047334 | 0.026081 |

## Peak memory

| Case | Maximum Rust bytes | Maximum native bytes | Peak ratio | Max Rust / min native |
| --- | ---: | ---: | ---: | ---: |
| dense | 11550720 | 10813440 | 1.068 | 1.191 |
| sparse | 6848512 | 6782976 | 1.010 | 1.040 |
| categorical | 8404992 | 9846784 | 0.854 | 0.859 |
| multiclass | 10240000 | 12189696 | 0.840 | 0.967 |
| ranking | 8667136 | 18432000 | 0.470 | 0.478 |

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
