# CPU performance evidence

## Gate status

**The required training-time gate remains open for sparse, categorical and ranking cases.** Dense and multiclass training pass. All five quality, matched peak-memory and inference gates pass on this source snapshot. No budget or fixed input was changed.

[The machine-readable report](performance.json) retains raw samples, parameters, input hashes, source/build identities and per-case comparisons. Budgets remain training <=1.5x, inference <=1.25x and peak memory <=1.5x, for every case.

## Source and measurement

- Measurement: 2026-10-28T15:00:33Z.
- Host: macOS ARM64, Apple M4, 10 logical CPUs.
- Rust 1.92.0, release profile with thin LTO and one codegen unit. No unsafe product code.
- Reference: LightGBM `5d84f3f222b41409a6983d2ac90513ad9702fa9e`; Apple clang 21.0.0, production `-O3` flags.
- Loaded library SHA-256: `9ed1d46d2d60bac2966fd03bd81e7d192025804f6a3e3ef24c3aa483d0ec7c1e`.
- Rust build-input SHA-256: `6bd3f302f676ad2c641ee0348173469f01769f96e62bd311be8e63c843a323e0`.
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
| dense | 1.446 | Pass | 0.837 | 0.404 |
| sparse | 3.171 | **Open** | 0.859 | 0.908 |
| categorical | 1.556 | **Open** | 0.960 | 0.307 |
| multiclass | 1.421 | Pass | 0.730 | 0.275 |
| ranking | 1.822 | **Open** | 0.736 | 0.569 |

All saved Python-engine training/inference comparisons pass. These are not native-only throughput passes. The direct C API results remain separate; their failures are not waived.

Dataset construction and training medians, in seconds:

| Case | Rust construction | Native construction | Rust training | Native training |
| --- | ---: | ---: | ---: | ---: |
| dense | 0.003455 | 0.003825 | 0.023139 | 0.015997 |
| sparse | 0.001326 | 0.001126 | 0.021607 | 0.006813 |
| categorical | 0.003188 | 0.003708 | 0.014229 | 0.009144 |
| multiclass | 0.003244 | 0.003807 | 0.077701 | 0.054682 |
| ranking | 0.003256 | 0.003826 | 0.047644 | 0.026153 |

## Peak memory

| Case | Maximum Rust bytes | Maximum native bytes | Peak ratio | Max Rust / min native |
| --- | ---: | ---: | ---: | ---: |
| dense | 11026432 | 10993664 | 1.003 | 1.152 |
| sparse | 6701056 | 6782976 | 0.988 | 1.015 |
| categorical | 7995392 | 9895936 | 0.808 | 0.837 |
| multiclass | 11829248 | 10944512 | 1.081 | 1.111 |
| ranking | 8323072 | 18677760 | 0.446 | 0.454 |

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
