"""Record real source, compiler, loaded image, driver, and saved-artifact identity."""
from datetime import datetime, timezone
import json
import os
import platform
import subprocess
import sys

from common import OUT, ROOT, SOURCE, load_reference, save_json, sha
from check_final import helper_build_record, sort_header_record

REFERENCE_FILES = (
    ".gitmodules", "CMakeLists.txt", "VERSION.txt", "LICENSE", "include/LightGBM/meta.h",
    "include/LightGBM/config.h", "include/LightGBM/utils/random.h", "include/LightGBM/bin.h",
    "src/io/config.cpp", "src/io/config_auto.cpp", "src/io/bin.cpp", "src/io/dense_bin.hpp",
    "src/c_api.cpp", "src/application/predictor.hpp",
    "src/objective/regression_objective.hpp", "src/objective/binary_objective.hpp",
    "src/objective/multiclass_objective.hpp", "src/objective/xentropy_objective.hpp",
    "src/objective/objective_function.cpp", "src/objective/rank_objective.hpp",
    "include/LightGBM/objective_function.h", "include/LightGBM/metric.h",
    "src/metric/metric.cpp", "src/metric/regression_metric.hpp", "src/metric/binary_metric.hpp",
    "src/metric/multiclass_metric.hpp", "src/metric/rank_metric.hpp", "src/metric/map_metric.hpp",
    "src/metric/xentropy_metric.hpp", "src/io/metadata.cpp", "include/LightGBM/dataset.h",
    "src/io/dataset_loader.cpp", "include/LightGBM/dataset_loader.h", "include/LightGBM/feature_group.h",
    "src/treelearner/linear_tree_learner.cpp", "src/treelearner/linear_tree_learner.h",
    "src/treelearner/feature_histogram.hpp", "src/treelearner/feature_histogram.cpp",
    "src/treelearner/split_info.hpp", "src/boosting/gbdt_model_text.cpp",
    "src/boosting/gbdt_prediction.cpp", "src/boosting/gbdt.cpp", "src/boosting/gbdt.h",
    "src/boosting/prediction_early_stop.cpp", "src/treelearner/serial_tree_learner.cpp",
    "src/io/tree.cpp", "include/LightGBM/tree.h",
    "include/LightGBM/sample_strategy.h", "include/LightGBM/tree_learner.h",
    "src/boosting/bagging.hpp", "src/boosting/goss.hpp", "src/boosting/dart.hpp", "src/boosting/rf.hpp",
    "src/treelearner/col_sampler.hpp", "src/treelearner/gradient_discretizer.hpp",
    "src/treelearner/gradient_discretizer.cpp", "src/treelearner/monotone_constraints.hpp",
    "src/treelearner/cost_effective_gradient_boosting.hpp",
    "python-package/lightgbm/libpath.py", "python-package/lightgbm/basic.py",
    "python-package/lightgbm/engine.py", "python-package/lightgbm/callback.py", "python-package/lightgbm/compat.py",
)


def artifact_paths():
    paths = list((ROOT / "tests/fixtures").rglob("*.json"))
    paths += list((ROOT / "tests/fixtures").rglob("*.txt"))
    paths += [p for p in (ROOT / "tools/oracle").iterdir() if p.is_file()]
    paths += [ROOT / name for name in ("docs/reference-manifest.schema.json", "docs/conformance.md", "THIRD_PARTY_NOTICES.md", "licenses/LLVM.txt")]
    return sorted(paths)


def main():
    _, record = load_reference()
    commands = json.loads((OUT / "build/compile_commands.json").read_text())
    assert not any("SCORE_T_USE_DOUBLE" in c["command"] or "LABEL_T_USE_DOUBLE" in c["command"] for c in commands)
    if sys.platform == "darwin":
        linked = subprocess.check_output(["otool", "-L", record["library"]["path"]], text=True).strip().splitlines()
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        memory = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    else:
        linked = subprocess.check_output(["ldd", record["library"]["path"]], text=True).strip().splitlines()
        cpu = platform.processor() or platform.machine()
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    assert not any("libomp" in line or "libgomp" in line for line in linked), "unexpected OpenMP dependency"
    library = dict(record["library"], linked_libraries=linked)
    seeds = {name: 0 for name in ("seed", "data_random_seed", "feature_fraction_seed", "bagging_seed",
                                "drop_seed", "extra_seed", "objective_seed")}
    result = dict(
        schema_version=1, generated_at_utc=datetime.now(timezone.utc).isoformat(), source=record["source"],
        license_provenance=dict(notice_file="THIRD_PARTY_NOTICES.md", upstream_license_sha256=record["source"]["license_sha256"]),
        precision=dict(label_t="f32", weight_t="f32", score_t="f32", hist_t="f64", prediction_t="f64",
                       macros=dict(SCORE_T_USE_DOUBLE=False, LABEL_T_USE_DOUBLE=False)),
        execution=dict(device="cpu", histogram_mode="force_col_wise", deterministic=True, threads=1,
                       seeds=seeds, rng_fixture_seeds=[0, 1, 42]),
        build=dict(compiler=record["compiler"], cmake_version=record["cmake_version"], ninja_version=record["ninja_version"],
                   configure_flags=record["configure_flags"], compile_commands_sha256=record["compile_commands_sha256"],
                   representative_compile_command=commands[0]["command"], openmp=False, gpu=False, cuda=False, rocm=False, mpi=False),
        loaded_library=library, driver=json.loads((OUT / "driver-record.json").read_text()),
        extended_driver=json.loads((OUT / "extended-driver-record.json").read_text()),
        native_helpers={name: json.loads((OUT / f'{name}-record.json').read_text())
                        for name in ('data_feature_info', 'sampling_driver', 'refit_driver', 'bundling_driver')},
        family_builds={name: helper_build_record(name, library['loaded_sha256']) for name in ('metric_edges', 'empty_queries')},
        sort_headers=sort_header_record(),
        host=dict(platform=platform.platform(), machine=platform.machine(), cpu=cpu,
                  logical_cpus=os.cpu_count(), memory_bytes=memory),
        dependencies=dict(lockfile="tools/oracle/uv.lock", lockfile_sha256=sha(ROOT / "tools/oracle/uv.lock"),
                          python=record["python"], submodules=record["submodules"]),
        reference_sources=[dict(path=str(p), sha256=sha(SOURCE / p)) for p in REFERENCE_FILES],
        artifacts=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in artifact_paths()],
        limitations=["OpenMP is disabled; no parallel speed equivalence is claimed.",
                     "Only this macOS arm64 build was executed; other hosts are unverified.",
                     "All five fixed quality and saved-baseline timing gates and matched native-only memory passed for the historical source snapshot in docs/performance.md; changed sources require a new checked measurement.",
                     "Fresh native training passes for dense/multiclass but exceeds 1.5x for sparse/categorical/ranking in docs/performance.md; the full performance gate remains open.",
                     "Initial compact evidence is not complete CPU compatibility."])
    save_json(ROOT / "docs/reference-manifest.json", result)
    print(f"Recorded loaded library {library['loaded_path']} SHA-256 {library['loaded_sha256']}")


if __name__ == "__main__":
    main()
