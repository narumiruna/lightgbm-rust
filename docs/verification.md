# CPU verification status

Release status: **open**. The required native-core training-time gate has not passed. Do not publish or replace a deployed implementation from this branch.

## Executed evidence

Rust toolchain: 1.92.0. Local measurements use native macOS ARM64. [Native CI](https://github.com/narumiruna/lightgbm-rust/actions/runs/40678701542) passed format, Clippy, all targets, doctests, offline fixtures and package verification on Ubuntu, macOS and Windows for commit `0e0f949`. These are Rust product checks against saved reference evidence, not new platform-specific LightGBM reference builds.

- Normal Cargo tests use saved fixtures. They do not load the ignored reference checkout or execute Python or C++.
- The native oracle check validates 399 artifact hashes and rejects 126 deliberate mutations. Extended families include 15 workflow, 68 AUC-mu edge and 14 empty-query cases. See `docs/conformance.md`.
- Clean package verification passed with `cargo package --locked` locally and in all three native CI jobs. The package contains 462 text/source files; it excludes the reference checkout and oracle tools.
- All package targets passed offline tests from the extracted package with a restricted toolchain/system PATH. No reference checkout or oracle tools exist inside that package. This also tests Cargo's normalized manifest.
- Normal dependencies contain no native ML or numerical library. The inspected release executable links only `libSystem.B.dylib` on macOS. See `Cargo.toml` and the retained license notices.
- Matched quality, memory and inference have passing source-identified reports. Native-core training remains above the unchanged per-case 1.5x limit. Reports are snapshots, not proof for later source changes. See `docs/performance.md` and `docs/performance.json`.

Final-source format, Clippy, 295 all-target tests, doctests, offline tests, Criterion and extracted-package tests passed locally. A fixture line-ending regression increases the native CI total to 296 passing tests. Native regeneration passed after the producer/checker changes. Checks must be repeated after later source changes. A successful benchmark command does not mean that its budgets passed.

## Reproduction

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --all-targets
cargo test --locked --doc
cargo test --locked --offline --all-targets
cargo bench --locked
cargo package --locked
```

The isolated native regeneration and performance commands are in `docs/conformance.md` and `docs/performance.md`. They are not normal product requirements.

## External handoff

Native Linux, macOS and Windows CI results are retained at the run linked above. The initial Windows checkout changed model line endings and failed strict serialized byte-length validation. `.gitattributes` now preserves native fixture bytes; a simulated CRLF checkout and a dedicated regression pass. Only the tested Rust toolchain/platform scope is verified. Other architectures, platform-native reference builds and matched native parallel performance remain unverified.

The package maintainer can authorize registry publication only after the required gates pass or an explicit exception is recorded. No package publication, version tag, release workflow or deployment replacement is authorized by this implementation task. Installation from a published registry package remains unverified.

The active implementation plan remains at `docs/plans/2026-10-28_pure-rust-rewrite-plan.md` while required gates are open.
