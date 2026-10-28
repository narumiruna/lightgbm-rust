"""Matched native-only RSS evidence; run with uv run --project tools/oracle python.

All generated arrays, binaries, snapshots, and raw logs stay under target.
Python generates/checks evidence but is never inside either measured process.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import statistics
import struct
import subprocess
import time

from build import COMMIT, OUT, ROOT, SOURCE, run, sha, source_identity

PREFIX = ROOT / "target/phase9-final-memory"
PINNED_LIBRARY_SHA = "9ed1d46d2d60bac2966fd03bd81e7d192025804f6a3e3ef24c3aa483d0ec7c1e"
STAGES = ("dataset_construction_seconds", "training_seconds", "serialization_seconds", "inference_seconds")


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_record(root=ROOT):
    paths = [*sorted((root / "src").rglob("*.rs")), *sorted((root / "benches").rglob("*.rs")), root / "Cargo.toml", root / "Cargo.lock",
             root / "rust-toolchain.toml", root / "examples/memory_case.rs", root / "examples/benchmark_case.rs"]
    files = {str(p.relative_to(root)): sha(p) for p in paths}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return dict(sha256=digest, files=files)


def verify_native():
    record = json.loads((OUT / "build-record.json").read_text())
    assert source_identity() == record["source"]
    assert record["source"]["commit"] == COMMIT
    assert sha(OUT / "build/compile_commands.json") == record["compile_commands_sha256"]
    assert sha(record["compiler"]["path"]) == record["compiler"]["sha256"]
    assert sha(record["compiler"]["selected_toolchain_path"]) == record["compiler"]["selected_toolchain_sha256"]
    assert run(record["compiler"]["path"], "--version") == record["compiler"]["version"]
    assert "-DUSE_OPENMP=OFF" in record["configure_flags"]
    assert sha(record["library"]["path"]) == record["library"]["sha256"] == PINNED_LIBRARY_SHA
    return record


def build_native(record):
    command = shlex.split(json.loads((OUT / "build/compile_commands.json").read_text())[0]["command"])
    obj = Path(str(PREFIX) + "-native.o")
    command[command.index("-o") + 1] = str(obj)
    command[command.index("-c") + 1] = str(ROOT / "tools/oracle/benchmark_memory.cpp")
    subprocess.run(command, check=True, timeout=300)
    binary = Path(str(PREFIX) + "-native")
    link = [*command[:command.index("-o")], str(obj), record["library"]["path"],
            f"-Wl,-rpath,{Path(record['library']['path']).parent}", "-o", str(binary)]
    if platform.system() != "Darwin":
        link += ["-ldl"]
    subprocess.run(link, check=True, timeout=300)
    return binary, dict(compile_command=command, link_command=link, object_sha256=sha(obj),
                        executable_sha256=sha(binary), source_sha256=sha(ROOT / "tools/oracle/benchmark_memory.cpp"))


def build_rust():
    before = source_record()
    snapshot = Path(str(PREFIX) + "-source")
    snapshot.mkdir(exist_ok=True)
    shutil.copytree(ROOT / "src", snapshot / "src", dirs_exist_ok=True)
    if (ROOT / "benches").is_dir():
        shutil.copytree(ROOT / "benches", snapshot / "benches", dirs_exist_ok=True)
    (snapshot / "examples").mkdir(exist_ok=True)
    for name in ("memory_case.rs", "benchmark_case.rs"):
        shutil.copyfile(ROOT / "examples" / name, snapshot / "examples" / name)
    for name in ("Cargo.toml", "Cargo.lock", "rust-toolchain.toml"):
        shutil.copyfile(ROOT / name, snapshot / name)
    assert source_record(snapshot) == before, "source changed during snapshot"
    # Isolate this snapshot from any parent workspace; do not edit the real manifest.
    with (snapshot / "Cargo.toml").open("a") as output:
        output.write("\n[workspace]\n")
    target = Path(str(PREFIX) + "-build")
    command = ["cargo", "build", "--offline", "--locked", "--release", "--manifest-path", str(snapshot / "Cargo.toml"),
               "--target-dir", str(target), "--example", "memory_case", "--example", "benchmark_case"]
    env = os.environ.copy()
    env.pop("RUSTFLAGS", None)
    env.pop("CARGO_ENCODED_RUSTFLAGS", None)
    subprocess.run(command, check=True, timeout=300, env=env)
    after = source_record()
    return target / "release/examples/memory_case", target / "release/examples/benchmark_case", dict(
        command=command, rustc=run("rustc", "-vV"), cargo=run("cargo", "-V"),
        rustc_path=str(Path(shutil.which("rustc")).resolve()),
        rustc_launcher_sha256=sha(shutil.which("rustc")),
        rustc_toolchain_path=run("rustup", "which", "rustc"),
        rustc_toolchain_sha256=sha(run("rustup", "which", "rustc")),
        source_before=before, source_after_build=after, source_unchanged_during_build=before == after,
        snapshot=str(snapshot), snapshot_manifest_sha256=sha(snapshot / "Cargo.toml"),
        flags="Cargo.toml release profile; RUSTFLAGS/CARGO_ENCODED_RUSTFLAGS unset; no native ML dependency")


def export(name, reference):
    source = OUT / f"benchmark-input-{name}.json"
    case = json.loads(source.read_text())
    assert case["upstream_commit"] == COMMIT and case["schema_version"] == 1
    assert (case["rows"], case["features"], case["validation_rows"], case["num_boost_round"]) == (8192, 16, 2048, 50)
    assert case["params"] == reference["params"]
    assert case["reference_expected_quality"] == reference["quality"][0]
    sparse = name == "sparse"
    binary = Path(str(PREFIX) + f"-input-{name}.bin")
    def array(output, values, code):
        output.write(struct.pack("<Q", len(values)))
        output.write(struct.pack(f"<{len(values)}{code}", *values))
    groups = [16] * 512 if name == "ranking" else []
    with binary.open("wb") as output:
        output.write(b"LGBMEM01")
        output.write(struct.pack("<5Q", 8192, 16, 2048, 50, int(sparse)))
        array(output, [] if sparse else case["train_values"], "d")
        array(output, [] if sparse else case["validation_values"], "d")
        array(output, case["train_labels"], "f")
        array(output, case["validation_labels"], "f")
        array(output, [], "f")  # Fixed cases have no weights; neither driver adds any.
        array(output, groups, "i")
        for key in ("train_csr", "validation_csr"):
            csr = case[key] if sparse else dict(indptr=[], indices=[], values=[])
            for field, code in (("indptr", "q"), ("indices", "i"), ("values", "d")):
                array(output, csr[field], code)
    params = dict(case["params"])
    if name == "categorical":
        params["categorical_feature"] = "0"  # Python Dataset categorical_feature=[0].
    param_path = Path(str(PREFIX) + f"-input-{name}.params")
    param_path.write_text(" ".join(f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in sorted(params.items())) + "\n")
    return binary, param_path, case, dict(json_path=str(source.relative_to(ROOT)), json_sha256=sha(source),
        binary_path=str(binary.relative_to(ROOT)), binary_sha256=sha(binary), parameter_text=param_path.read_text().strip(),
        parameter_sha256=sha(param_path), params=case["params"], weights="absent", query_groups=groups,
        train_rows=8192, validation_rows=2048, features=16, rounds=50,
        nonzero_training_values=reference["nonzero_training_values"], csr=sparse)


def measured(command, stem):
    report, rss, pred = (Path(str(PREFIX) + f"-{stem}{suffix}") for suffix in (".json", ".rss", ".pred.bin"))
    timer = ["/usr/bin/time", "-l"] if platform.system() == "Darwin" else ["/usr/bin/time", "-v"]
    with report.open("w") as stdout, rss.open("w") as stderr:
        subprocess.run([*timer, *map(str, command), str(pred)], stdout=stdout, stderr=stderr, check=True, timeout=300)
    data = json.loads(report.read_text())
    text = rss.read_text()
    match = re.search(r"(\d+)\s+maximum resident set size", text) if platform.system() == "Darwin" else re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    assert match, text
    data["peak_rss_bytes"] = int(match[1]) * (1 if platform.system() == "Darwin" else 1024)
    assert data["peak_rss_bytes"] <= 512 * 1024 * 1024
    data["command"] = [*timer, *map(str, command), str(pred)]
    data["raw_stdout"] = str(report.relative_to(ROOT))
    data["raw_rss"] = str(rss.relative_to(ROOT))
    data["predictions_sha256"] = sha(pred)
    payload = pred.read_bytes()
    predictions = struct.unpack(f"<{len(payload)//8}d", payload)
    assert len(predictions) == data["prediction_count"]
    assert data["warmup_runs"] == 1 and data["measured_runs"] == 5
    for stage in STAGES:
        assert len(data[stage]) == 5
    return data, predictions


def quality(name, labels, values):
    import math
    if name in ("dense", "sparse"):
        return math.sqrt(sum((y-p)**2 for y, p in zip(labels, values)) / len(labels))
    if name == "categorical":
        return -sum(y*math.log(max(1e-15, min(1-1e-15, p))) + (1-y)*math.log(max(1e-15, min(1-1e-15, 1-p))) for y, p in zip(labels, values)) / len(labels)
    if name == "multiclass":
        return -sum(math.log(max(1e-15, values[i*3+int(y)])) for i, y in enumerate(labels)) / len(labels)
    scores = []
    for start in range(0, len(labels), 16):
        y, p = labels[start:start+16], values[start:start+16]
        dcg = lambda a: sum((2**v-1)/math.log2(i+2) for i, v in enumerate(a[:5]))
        ideal = dcg(sorted(y, reverse=True))
        scores.append(dcg([y[i] for i in sorted(range(16), key=lambda i: -p[i])]) / ideal if ideal else 1.)
    return sum(scores) / len(scores)


def summary(samples):
    return {**{f"median_{stage}": statistics.median([v for s in samples for v in s[stage]]) for stage in STAGES},
            "median_peak_rss_bytes": statistics.median(s["peak_rss_bytes"] for s in samples),
            "maximum_peak_rss_bytes": max(s["peak_rss_bytes"] for s in samples)}


def main():
    started = time.time()
    contract = json.loads((ROOT / "tests/fixtures/benchmark-contract.json").read_text())
    baseline = json.loads((ROOT / "tests/fixtures/reference-baseline.json").read_text())
    record = verify_native()
    native, native_build = build_native(record)
    rust, benchmark, rust_build = build_rust()
    rust_build["executable_sha256"] = sha(rust)
    rust_build["benchmark_executable_sha256"] = sha(benchmark)
    # The Python oracle is used only outside /usr/bin/time's measured children.
    from common import load_reference
    from benchmark import data as fixed_data
    import numpy as np
    from scipy import sparse as scipy_sparse
    lgb, loaded_record = load_reference()
    cases = []
    for gate in contract["cases"]:
        name = gate["name"]
        ref = next(c for c in baseline["cases"] if c["name"] == name)
        binary, params, inputs, input_record = export(name, ref)
        # Independently prove native C API predictions match the original Python workflow.
        train = np.array(inputs["train_values"]).reshape(8192,16)
        valid = np.array(inputs["validation_values"]).reshape(2048,16)
        if name == "sparse":
            def csr(key, rows):
                c = inputs[key]
                return scipy_sparse.csr_matrix((np.array(c["values"], dtype=np.float64), np.array(c["indices"], dtype=np.int32),
                                               np.array(c["indptr"], dtype=np.int32)), shape=(rows,16))
            train, valid = csr("train_csr",8192), csr("validation_csr",2048)
        expected_train, expected_labels, expected_valid, expected_valid_labels = fixed_data(name)
        if name == "sparse":
            for actual, expected in ((train,expected_train), (valid,expected_valid)):
                assert np.array_equal(actual.indptr,expected.indptr)
                assert np.array_equal(actual.indices,expected.indices)
                assert np.array_equal(actual.data,expected.data)
        else:
            assert np.array_equal(train,expected_train) and np.array_equal(valid,expected_valid)
        assert np.array_equal(np.array(inputs["train_labels"],dtype=np.float32),expected_labels)
        assert np.array_equal(np.array(inputs["validation_labels"],dtype=np.float32),expected_valid_labels)
        ds = lgb.Dataset(train, label=np.array(inputs["train_labels"], dtype=np.float32),
                         group=[16]*512 if name == "ranking" else None,
                         categorical_feature=[0] if name == "categorical" else "auto", params=ref["params"])
        oracle = lgb.train(ref["params"], ds, num_boost_round=50).predict(valid, num_threads=1).ravel()
        oracle_path = Path(str(PREFIX) + f"-{name}-python-check.pred.bin")
        oracle.astype("<f8").tofile(oracle_path)
        all_samples = {"native": [], "rust": []}
        for process in range(5):
            for kind, executable in (("native",native), ("rust",rust)):
                data, predictions = measured([executable,binary,params,"1"], f"{name}-{kind}-p{process+1}")
                if kind == "native":
                    assert Path(data["loaded_library"]).resolve() == Path(record["library"]["path"]).resolve()
                    assert sha(data["loaded_library"]) == PINNED_LIBRARY_SHA
                delta = np.abs(np.asarray(predictions) - oracle)
                assert np.all(delta <= 1e-12 + 1e-10*np.abs(oracle)), (name,kind,float(delta.max()))
                data["max_prediction_absolute_delta_from_python"] = float(delta.max())
                data["quality"] = quality(name, inputs["validation_labels"], predictions)
                all_samples[kind].append(data)
        stats = {kind: summary(samples) for kind, samples in all_samples.items()}
        # Retain the original fixed timing baseline. New C API times are diagnostic, not a replacement baseline.
        timing_ratios = dict(training=stats["rust"]["median_training_seconds"] / ref["median_training_seconds"],
                             inference=stats["rust"]["median_inference_seconds"] / ref["median_inference_seconds"])
        native_timing_ratios = dict(training=stats["rust"]["median_training_seconds"] / stats["native"]["median_training_seconds"],
                                    inference=stats["rust"]["median_inference_seconds"] / stats["native"]["median_inference_seconds"])
        peak_ratio = stats["rust"]["maximum_peak_rss_bytes"] / stats["native"]["maximum_peak_rss_bytes"]
        # Also require every Rust peak <= 1.5 * the smallest native peak (conservative noise check).
        conservative_peak_ratio = max(s["peak_rss_bytes"] for s in all_samples["rust"]) / min(s["peak_rss_bytes"] for s in all_samples["native"])
        bound = gate["quality_bound"]
        delta_bound = gate["reference_delta"]
        qualities = [s["quality"] for samples in all_samples.values() for s in samples]
        quality_pass = all(q >= max(bound,ref["quality"][0]-delta_bound) if name == "ranking" else q <= min(bound,ref["quality"][0]+delta_bound) for q in qualities)
        supplemental = []
        for threads in (2,4):
            data, predictions = measured([rust,binary,params,str(threads)], f"{name}-rust-t{threads}")
            assert np.all(np.abs(np.asarray(predictions)-oracle) <= 1e-12+1e-10*np.abs(oracle))
            data["quality"] = quality(name,inputs["validation_labels"],predictions)
            data["matched_parallel_budget"] = None
            supplemental.append(data)
        cases.append(dict(name=name, inputs=input_record, quality_gate=gate, reference_quality=ref["quality"][0],
            python_check_predictions_sha256=sha(oracle_path), samples=all_samples, summary=stats,
            timing_ratios_to_fixed_baseline=timing_ratios, timing_ratios_to_fresh_native=native_timing_ratios,
            fresh_native_training_pass=native_timing_ratios["training"]<=1.5,
            fresh_native_inference_pass=native_timing_ratios["inference"]<=1.25,
            fixed_reference_median_training_seconds=ref["median_training_seconds"],
            fixed_reference_median_inference_seconds=ref["median_inference_seconds"],
            peak_memory_ratio=peak_ratio, conservative_peak_memory_ratio=conservative_peak_ratio,
            median_peak_memory_ratio=stats["rust"]["median_peak_rss_bytes"]/stats["native"]["median_peak_rss_bytes"],
            training_pass=timing_ratios["training"]<=1.5, inference_pass=timing_ratios["inference"]<=1.25,
            memory_pass=peak_ratio<=1.5 and conservative_peak_ratio<=1.5, quality_pass=quality_pass,
            supplemental_rust_parallel=supplemental))
        print(name, timing_ratios, "peak",peak_ratio,"conservative",conservative_peak_ratio,flush=True)
    assert verify_native() == record, "native identity changed during measurement"
    after = source_record()
    rust_build["source_after_measurement"] = after
    rust_build["source_unchanged_through_measurement"] = after == rust_build["source_before"]
    rust_build["git_status"] = run("git","status","--porcelain")
    report = dict(schema_version=1, measured_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime(started)),
        host=dict(platform=platform.platform(),machine=platform.machine(),cpu=run("sysctl","-n","machdep.cpu.brand_string") if platform.system()=="Darwin" else run("uname","-m"), logical_cpus=os.cpu_count()),
        contract_sha256=sha(ROOT/"tests/fixtures/benchmark-contract.json"), baseline_sha256=sha(ROOT/"tests/fixtures/reference-baseline.json"),
        budgets=contract["performance_budgets"], native_identity=loaded_record, native_build=native_build, rust_build=rust_build,
        measurement_script_sha256=sha(Path(__file__)),
        boundary="Absolute entire fresh-process peak, including binary loader, OS libraries, raw input loading, six sequential construction/train/save/predict iterations, and final prediction output. No subtraction. One warm-up is included in each peak; five measured iterations per process, five fresh processes per implementation/case. Python is outside measured children.",
        input_format="LGBMEM01; five u64 dimensions/rounds/CSR flag; u64-length-prefixed little-endian arrays in order: f64 dense train/valid (empty for CSR), f32 train/valid labels, f32 weights, i32 group lengths, i64 CSR offsets/i32 indexes/f64 data for train then valid. Sparse Rust indexes convert to usize; C API uses i32 indexes/i64 offsets. No whole-input JSON in either process.",
        timing_boundary="Construction includes metadata; training includes BoosterCreate+50 UpdateOneIter in native, Trainer::fit in Rust. Serialization is separate, then batch prediction. Rust also validates scores and retains history records; metric=None disables metric kernels in both drivers. Fixed saved Python-engine timing baseline remains unchanged. Fresh native-only timing comparisons are separately reported, including failed budgets; they do not replace the fixed baseline.",
        parallel_restriction="Original pinned native baseline is USE_OPENMP=OFF and actually serial even with num_threads=2/4. Supplemental Rust 2/4 runs are not matched parallel budget evidence. No alternate OpenMP library was built.",
        historical_phase9_summary_sha256=sha(ROOT/"target/phase9-summary.json"), cases=cases,
        fixed_timing_and_matched_memory_pass=all(c["training_pass"] and c["inference_pass"] and c["memory_pass"] and c["quality_pass"] for c in cases),
        fresh_native_all_budgets_pass=all(c["fresh_native_training_pass"] and c["fresh_native_inference_pass"] and c["memory_pass"] and c["quality_pass"] for c in cases),
        current_source_matches_snapshot=rust_build["source_unchanged_through_measurement"],
        remaining_gates=[
            "Fresh matched native-only training exceeds 1.5x for: " + ", ".join(c["name"] for c in cases if not c["fresh_native_training_pass"]) + ". Fixed saved Python-engine timing passes are not native-only throughput passes.",
            "Matched actual 2/4-thread native timing and memory are unverified; pinned baseline has OpenMP off.",
            "Other correctness/package/platform gates belong to the main task; these measurements do not complete the full plan."])
    save(str(PREFIX)+"-report.json",report)
    save(ROOT/"docs/performance.json",report)


if __name__ == "__main__":
    main()
